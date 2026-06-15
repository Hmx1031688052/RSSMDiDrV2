"""RL-finetune the RSSM-conditioned DiffusionDrive planner on CARLA.

This is the CARLA/RSSM counterpart of DiffusionDriveV2's RL stage:

* no NAVSIM PDM score or metric cache is used;
* a frozen DreamerV3/JAX RSSM provides imagined returns for sampled plans;
* the Torch planner is updated through diffusion-chain log-probabilities;
* an imitation L1 term against replay expert waypoints keeps training anchored.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    torch = None
    F = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dreamerv3") not in sys.path:
    sys.path.insert(0, str(ROOT / "dreamerv3"))

from RSSMDiDrOnCarla.scripts.train_torch_selector_ranking_with_jax_rssm import (
    JAXRSSMModeScorer,
    ReplaySequenceDataset,
    load_torch_planner,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline_replay_dir", required=True)
    parser.add_argument("--rssm_checkpoint", required=True)
    parser.add_argument("--planner_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--anchor_path", default=None)
    parser.add_argument("--task", default="carla_roundabout")

    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imag_horizon", type=int, default=8)
    parser.add_argument("--discount", type=float, default=0.997)

    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--rl_steps", type=int, default=10)
    parser.add_argument("--rollout_span", type=int, default=20)
    parser.add_argument("--trunc_timestep", type=int, default=8)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--adv_discount", type=float, default=0.8)
    parser.add_argument("--adv_eps", type=float, default=1e-4)
    parser.add_argument("--positive_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt_threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_logprob_std", type=float, default=0.1)

    parser.add_argument("--rl_weight", type=float, default=1.0)
    parser.add_argument("--il_weight_positive", type=float, default=0.1)
    parser.add_argument("--il_weight_no_positive", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--jax_platform", default=None, choices=("cpu", "gpu", "tpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--no_tqdm", action="store_true")

    parser.add_argument("--plan_x_sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--plan_y_sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--waypoint_dt", type=float, default=0.5)
    parser.add_argument("--wheelbase", type=float, default=2.875)
    parser.add_argument("--max_steer_rad", type=float, default=0.65)
    parser.add_argument("--steer_sign", type=float, default=-1.0)
    parser.add_argument("--steer_gain", type=float, default=0.7)
    parser.add_argument("--lookahead_min", type=float, default=4.5)
    parser.add_argument("--lookahead_max", type=float, default=14.0)
    parser.add_argument("--lookahead_gain", type=float, default=1.0)
    parser.add_argument("--speed_kp", type=float, default=1.0)
    parser.add_argument("--ctrl_acc_min", type=float, default=-3.0)
    parser.add_argument("--ctrl_acc_max", type=float, default=3.0)
    parser.add_argument("--ctrl_target_speed_max", type=float, default=8.0)
    parser.add_argument("--ctrl_soft_lookup_temp", type=float, default=0.75)
    return parser.parse_known_args()


def freeze_for_rl(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("decoder.") and not name.startswith("decoder.selector_head.")
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable decoder parameters found for RL finetuning.")
    return trainable


def scalarize(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            out[key] = float(value.detach().cpu())
        else:
            out[key] = float(value)
    return out


def save_checkpoint(path: Path, model, optimizer, config, iteration: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(iteration),
            "rl_finetune_iteration": int(iteration),
        },
        path,
    )


def extract_gt_xy(batch_np: Dict[str, np.ndarray], device) -> Optional[torch.Tensor]:
    """Return [B, 8, 2] expert xy if replay contains planner targets."""
    if "trajectory" in batch_np:
        traj = np.asarray(batch_np["trajectory"], dtype=np.float32)
        if traj.ndim == 4 and traj.shape[-2:] == (8, 3):
            return torch.from_numpy(traj[:, -1, :, :2]).to(device=device, dtype=torch.float32)
        if traj.ndim == 3 and traj.shape[-2:] == (8, 3):
            return torch.from_numpy(traj[:, :, :2]).to(device=device, dtype=torch.float32)

    if "expert_waypoints8" in batch_np:
        wp = np.asarray(batch_np["expert_waypoints8"], dtype=np.float32)
        if wp.ndim == 3 and wp.shape[-1] == 16:
            xy = wp[:, -1].reshape(wp.shape[0], 8, 2)
            return torch.from_numpy(xy).to(device=device, dtype=torch.float32)
        if wp.ndim == 2 and wp.shape[-1] == 16:
            xy = wp.reshape(wp.shape[0], 8, 2)
            return torch.from_numpy(xy).to(device=device, dtype=torch.float32)
    return None


def score_gt_if_available(scorer, state, speed, gt_xy: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if gt_xy is None:
        return None
    returns = scorer.score(state, speed, gt_xy[:, None].detach().cpu().numpy().astype(np.float32))
    return torch.from_numpy(returns[:, 0]).to(device=gt_xy.device, dtype=torch.float32)


def rollout_timesteps(args, device) -> torch.Tensor:
    step_ratio = float(args.rollout_span) / max(int(args.rl_steps), 1)
    steps = (np.arange(0, int(args.rl_steps)) * step_ratio).round()[::-1].copy().astype(np.int64)
    steps = np.clip(steps, 0, None)
    return torch.from_numpy(steps).to(device=device, dtype=torch.long)


def ddim_step_with_logprob(
    model,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    eta: float,
    min_logprob_std: float,
    prev_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DDIM-style sample-prediction step with DiffusionDriveV2-like log prob."""
    alpha_cumprod = model.scheduler.alpha_cumprod.to(sample.device)
    t = int(timestep.detach().cpu().item())
    prev_t = t - 1
    alpha_t = alpha_cumprod[t]
    alpha_prev = alpha_cumprod[prev_t] if prev_t >= 0 else torch.ones_like(alpha_t)
    beta_t = 1.0 - alpha_t

    pred_original = model_output
    pred_epsilon = (sample - alpha_t.sqrt() * pred_original) / beta_t.sqrt().clamp_min(1e-12)
    pred_original = pred_original.clamp(-1.0, 1.0)

    variance = ((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev)).clamp_min(1e-20)
    std = (float(eta) * variance.sqrt()).clamp_min(1e-10)
    direction = (1.0 - alpha_prev - std**2).clamp_min(0.0).sqrt() * pred_epsilon
    prev_mean = alpha_prev.sqrt() * pred_original + direction

    if prev_sample is None:
        noise_h = torch.randn((*model_output.shape[:2], 1, 1), device=model_output.device, dtype=model_output.dtype)
        noise_v = torch.randn((*model_output.shape[:2], 1, 1), device=model_output.device, dtype=model_output.dtype)
        mul_noise = torch.cat([noise_h, noise_v], dim=-1).repeat(1, 1, model_output.shape[2], 1)
        prev_sample = prev_mean * (1.0 + mul_noise * std.clamp_min(0.04))

    log_std = std.clamp_min(float(min_logprob_std))
    log_prob = (
        -((prev_sample.detach() - prev_mean) ** 2) / (2.0 * log_std**2)
        - torch.log(log_std)
        - math.log(math.sqrt(2.0 * math.pi))
    ).sum(dim=(-2, -1))
    return prev_sample.to(sample.dtype), log_prob, prev_mean.to(sample.dtype)


def sample_old_chain(model, condition: torch.Tensor, args) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = condition.shape[0]
    device = condition.device
    modes = int(model.config.num_modes)
    groups = int(args.num_groups)
    anchor = model.plan_anchor.to(device).unsqueeze(0).unsqueeze(1).repeat(bs, groups, 1, 1, 1)
    anchor = anchor.reshape(bs, groups * modes, model.config.num_poses, 2)

    x = model._normalize_xy(anchor)
    if int(args.trunc_timestep) > 0:
        t = torch.full((bs,), int(args.trunc_timestep), device=device, dtype=torch.long)
        x = model.scheduler.add_noise(x, torch.randn_like(x), t).clamp(-1.0, 1.0)

    all_x = [x]
    all_log_probs = []
    for t in rollout_timesteps(args, device):
        noisy_xy = model._denormalize_xy(x.clamp(-1.0, 1.0))
        timesteps = t.expand(bs)
        poses_reg, _ = model.decoder(condition, noisy_xy, timesteps)
        x_start = model._normalize_xy(poses_reg[..., :2])
        x, log_prob, _ = ddim_step_with_logprob(
            model,
            x_start,
            t,
            x,
            eta=float(args.eta),
            min_logprob_std=float(args.min_logprob_std),
        )
        all_log_probs.append(log_prob)
        all_x.append(x)
    return torch.stack(all_x, dim=-1), torch.stack(all_log_probs, dim=-1)


def compute_advantages(
    rewards: torch.Tensor,
    gt_rewards: Optional[torch.Tensor],
    args,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    bs, total_modes = rewards.shape
    groups = int(args.num_groups)
    modes = total_modes // groups
    grouped = rewards.reshape(bs, groups, modes)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True).clamp_min(float(args.adv_eps))
    advantages = (grouped - mean) / std

    if bool(args.positive_only):
        advantages = advantages.clamp_min(0.0)
    if bool(args.gt_threshold) and gt_rewards is not None:
        advantages = advantages * (grouped > gt_rewards[:, None, None]).float()

    flat = advantages.reshape(bs, total_modes)
    discount = torch.tensor(
        [float(args.adv_discount) ** (int(args.rl_steps) - i - 1) for i in range(int(args.rl_steps))],
        device=rewards.device,
        dtype=rewards.dtype,
    )
    step_adv = flat.detach().unsqueeze(-1) * discount
    metrics = {
        "reward_mean": rewards.mean(),
        "reward_max": rewards.max(dim=-1).values.mean(),
        "adv_mean": step_adv.mean(),
        "positive_rate": (flat > 0).float().mean(),
    }
    if gt_rewards is not None:
        metrics["gt_reward"] = gt_rewards.mean()
        metrics["beats_gt_rate"] = (grouped > gt_rewards[:, None, None]).float().mean()
    return step_adv, metrics


def recompute_rl_loss(
    model,
    condition: torch.Tensor,
    chains: torch.Tensor,
    advantages: torch.Tensor,
    gt_xy: Optional[torch.Tensor],
    args,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    bs, total_modes = chains.shape[:2]
    device = condition.device
    chain_x = chains[..., :-1]
    chain_prev = chains[..., 1:]

    log_probs = []
    il_losses = []
    for idx, t in enumerate(rollout_timesteps(args, device)):
        x = chain_x[..., idx]
        noisy_xy = model._denormalize_xy(x.clamp(-1.0, 1.0))
        timesteps = t.expand(bs)
        poses_reg, _ = model.decoder(condition, noisy_xy, timesteps)
        x_start = model._normalize_xy(poses_reg[..., :2])
        _, log_prob, _ = ddim_step_with_logprob(
            model,
            x_start,
            t,
            x,
            eta=float(args.eta),
            min_logprob_std=float(args.min_logprob_std),
            prev_sample=chain_prev[..., idx],
        )
        log_probs.append(log_prob)
        if gt_xy is not None:
            target = gt_xy[:, None].repeat(1, total_modes, 1, 1)
            il_losses.append(F.l1_loss(poses_reg[..., :2], target, reduction="none").mean(dim=(1, 2, 3)))

    log_probs = torch.stack(log_probs, dim=-1)
    per_token_loss = -torch.exp(log_probs - log_probs.detach()) * advantages
    mask = advantages != 0
    rl_loss_b = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    rl_loss_b = rl_loss_b.mean(dim=-1)

    if il_losses:
        il_loss_b = torch.stack(il_losses, dim=0).mean(dim=0)
    else:
        il_loss_b = torch.zeros_like(rl_loss_b)

    has_positive = (advantages > 0).any(dim=2).any(dim=1)
    il_weight = torch.where(
        has_positive,
        torch.full_like(rl_loss_b, float(args.il_weight_positive)),
        torch.full_like(rl_loss_b, float(args.il_weight_no_positive)),
    )
    loss_b = float(args.rl_weight) * rl_loss_b + il_weight * il_loss_b
    loss = loss_b.mean()
    metrics = {
        "loss": loss.detach(),
        "rl_loss": rl_loss_b.mean().detach(),
        "il_loss": il_loss_b.mean().detach(),
        "il_weight": il_weight.mean().detach(),
        "logprob": log_probs.mean().detach(),
    }
    return loss, metrics


def main() -> None:
    args, extra = parse_args()
    if torch is None:
        raise ModuleNotFoundError("This script requires PyTorch.") from TORCH_IMPORT_ERROR
    if args.jax_platform:
        import jax

        jax.config.update("jax_platform_name", args.jax_platform)

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, planner_config, checkpoint_epoch = load_torch_planner(args.planner_checkpoint, args.anchor_path, device)
    trainable = freeze_for_rl(model)
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))

    dataset = ReplaySequenceDataset(args.offline_replay_dir, args.batch_length)
    scorer = JAXRSSMModeScorer(args, extra, planner_config)

    run_config = {
        "args": vars(args),
        "planner_config": planner_config.to_dict(),
        "planner_checkpoint_epoch": checkpoint_epoch,
        "trainable_parameters": [name for name, param in model.named_parameters() if param.requires_grad],
        "replay_sequences": len(dataset),
        "reward_source": "frozen_jax_rssm_imagined_return",
        "pdms_used": False,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    history = []
    best_reward = None
    iterator = range(1, int(args.iterations) + 1)
    if not args.no_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc="rssm-didr-rl", dynamic_ncols=True)

    for iteration in iterator:
        batch_np = dataset.sample(args.batch_size, np_rng)
        state, speed, condition_np = scorer.start(batch_np)
        condition = torch.from_numpy(condition_np).to(device=device, dtype=torch.float32)
        gt_xy = extract_gt_xy(batch_np, device)

        with torch.no_grad():
            chains, _ = sample_old_chain(model, condition, args)
            final_xy = model._denormalize_xy(chains[..., -1]).detach()
            rewards_np = scorer.score(state, speed, final_xy.cpu().numpy().astype(np.float32))
            rewards = torch.from_numpy(rewards_np).to(device=device, dtype=torch.float32)
            gt_rewards = score_gt_if_available(scorer, state, speed, gt_xy)
            advantages, reward_metrics = compute_advantages(rewards, gt_rewards, args)

        loss, loss_metrics = recompute_rl_loss(model, condition, chains.detach(), advantages, gt_xy, args)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        optimizer.step()

        metrics_t = {**loss_metrics, **reward_metrics}
        metrics = {"iteration": int(iteration), **scalarize(metrics_t)}
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "rew": f"{metrics['reward_mean']:.3f}",
                    "max": f"{metrics['reward_max']:.3f}",
                    "pos": f"{metrics['positive_rate']:.3f}",
                }
            )

        if iteration == 1 or iteration % int(args.log_every) == 0:
            history.append(metrics)
            message = "[rssm_didr_rl] " + " ".join(
                f"{key}={value:.5f}" for key, value in metrics.items() if key != "iteration"
            )
            if tqdm is not None and hasattr(iterator, "write"):
                iterator.write(message)
            else:
                print(message, flush=True)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        monitor = metrics["reward_mean"]
        if best_reward is None or monitor > best_reward:
            best_reward = monitor
            save_checkpoint(output_dir / "best.pt", model, optimizer, planner_config, iteration)
        if args.save_every > 0 and iteration % int(args.save_every) == 0:
            save_checkpoint(output_dir / f"iteration_{iteration:06d}.pt", model, optimizer, planner_config, iteration)

    save_checkpoint(output_dir / "last.pt", model, optimizer, planner_config, int(args.iterations))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[rssm_didr_rl] Wrote checkpoints under {output_dir}")


if __name__ == "__main__":
    main()
