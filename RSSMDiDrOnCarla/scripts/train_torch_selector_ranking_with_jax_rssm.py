"""Finetune the Torch planner selector using fixed JAX RSSM imagined returns.

This script keeps the DreamerV3/JAX RSSM frozen and updates only the Torch
DiffusionDrive planner selector head. The planner proposes all modes, the JAX
RSSM scores each mode with imagined returns, and the selector is trained with a
ranking/RL/KL/entropy objective.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
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
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--policy_temperature", type=float, default=1.0)
    parser.add_argument("--return_temperature", type=float, default=1.0)
    parser.add_argument("--ranking_weight", type=float, default=1.0)
    parser.add_argument("--rl_weight", type=float, default=0.1)
    parser.add_argument("--kl_weight", type=float, default=0.05)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--adv_eps", type=float, default=1e-4)
    parser.add_argument("--eval_timestep", type=int, default=8)
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


def iter_replay_paths(directory: str | Path):
    yield from sorted(Path(directory).glob("*.npz"))


def load_replay_chunk(path: str | Path):
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


class ReplaySequenceDataset:
    def __init__(self, replay_dir: str | Path, batch_length: int):
        self.replay_dir = Path(replay_dir)
        self.batch_length = int(batch_length)
        self.paths = []
        self.lengths = []
        for path in iter_replay_paths(self.replay_dir):
            chunk = load_replay_chunk(path)
            if "action" not in chunk or "reward" not in chunk:
                continue
            length = int(len(chunk["action"]))
            if length >= self.batch_length:
                self.paths.append(path)
                self.lengths.append(length - self.batch_length + 1)
        if not self.paths:
            raise FileNotFoundError(f"No usable replay chunks in {self.replay_dir}")
        self.cumulative = np.cumsum(self.lengths)

    def __len__(self):
        return int(self.cumulative[-1])

    def sample(self, batch_size: int, rng: np.random.Generator):
        rows = [self._get(int(rng.integers(0, len(self)))) for _ in range(int(batch_size))]
        keys = sorted(set.intersection(*(set(row.keys()) for row in rows)))
        return {key: np.stack([row[key] for row in rows], axis=0) for key in keys}

    def _get(self, index: int):
        chunk_idx = int(np.searchsorted(self.cumulative, index, side="right"))
        prev = 0 if chunk_idx == 0 else int(self.cumulative[chunk_idx - 1])
        start = index - prev
        end = start + self.batch_length
        chunk = load_replay_chunk(self.paths[chunk_idx])
        keys = [key for key, value in chunk.items() if np.asarray(value).ndim > 0 and len(value) >= end]
        row = {key: np.asarray(chunk[key][start:end]) for key in keys}
        if "executed_control" in row:
            row["action"] = row["executed_control"]
        row.setdefault("is_first", np.zeros((self.batch_length,), dtype=bool))
        row.setdefault("is_terminal", np.zeros((self.batch_length,), dtype=bool))
        row.setdefault("reward", np.zeros((self.batch_length,), dtype=np.float32))
        return row


def flatten_state_feat_jax(jnp, state):
    deter = state["deter"].reshape((state["deter"].shape[0], -1))
    stoch = state["stoch"].reshape((state["stoch"].shape[0], -1))
    return jnp.concatenate([deter, stoch], axis=-1)


class JAXRSSMModeScorer:
    def __init__(self, args: argparse.Namespace, extra: list[str], planner_config):
        import jax
        import jax.numpy as jnp
        from JAXRSSMJAXDiDr.models import JAXDiDrConfig, apply_plan_sign, differentiable_pidpp
        from RSSMDiDrOnCarla.scripts import train_offline_rssm as offline

        self.jax = jax
        self.jnp = jnp
        self.apply_plan_sign = apply_plan_sign
        self.differentiable_pidpp = differentiable_pidpp
        offline.import_runtime()
        rssm_args = argparse.Namespace(
            task=args.task,
            replay_dir=args.offline_replay_dir,
            logdir=str(Path(args.output_dir) / "dreamer_runtime"),
            batch_length=args.batch_length,
            batch_size=args.batch_size,
            replay_size=int(1e6),
            jax_platform=args.jax_platform,
            from_checkpoint="",
        )
        raw_env, config = offline.build_config(rssm_args, extra)
        obs_space, act_space = offline.get_spaces(raw_env, config.dreamerv3)
        step = offline.embodied.Counter()
        self.agent = offline.dreamerv3.Agent(obs_space, act_space, step, config.dreamerv3)
        if len(self.agent.train_devices) != 1:
            raise ValueError("Set dreamerv3.jax.train_devices=[0].")
        checkpoint = offline.embodied.Checkpoint(log=False, parallel=False)
        checkpoint.agent = self.agent
        checkpoint.load(args.rssm_checkpoint, keys=["agent"])
        self.device = self.agent.train_devices[0]
        self.discount = float(args.discount)
        self.imag_horizon = int(args.imag_horizon)
        self.plan_x_sign = float(args.plan_x_sign)
        self.plan_y_sign = float(args.plan_y_sign)
        self.ctrl_config = JAXDiDrConfig(
            latent_dim=int(planner_config.latent_dim),
            plan_anchor_path=str(planner_config.plan_anchor_path),
            waypoint_dt=float(args.waypoint_dt),
            wheelbase=float(args.wheelbase),
            max_steer_rad=float(args.max_steer_rad),
            steer_sign=float(args.steer_sign),
            steer_gain=float(args.steer_gain),
            lookahead_min=float(args.lookahead_min),
            lookahead_max=float(args.lookahead_max),
            lookahead_gain=float(args.lookahead_gain),
            speed_kp=float(args.speed_kp),
            ctrl_acc_min=float(args.ctrl_acc_min),
            ctrl_acc_max=float(args.ctrl_acc_max),
            ctrl_target_speed_max=float(args.ctrl_target_speed_max),
            ctrl_soft_lookup_temp=float(args.ctrl_soft_lookup_temp),
        )
        self.start_fn = offline.nj.jit(offline.nj.pure(self._start), device=self.device)
        self.return_fn = offline.nj.jit(offline.nj.pure(self._mode_returns), device=self.device)

    def _start(self, data):
        data = self.agent.agent.preprocess(data)
        embed = self.agent.agent.wm.encoder(data)
        prev_latent, prev_action = self.agent.agent.wm.initial(data["action"].shape[0])
        prev_actions = self.jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], axis=1)
        post, _ = self.agent.agent.wm.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        state = {key: value[:, -1] for key, value in post.items()}
        if "ego_speed" in data:
            speed = data["ego_speed"][:, -1].reshape((data["action"].shape[0], -1))[:, :1]
        else:
            speed = self.jnp.zeros((data["action"].shape[0], 1), self.jnp.float32)
        return state, speed, flatten_state_feat_jax(self.jnp, state)

    def _reward(self, state):
        return self.agent.agent.wm.heads["reward"](state).mode().reshape((-1,))

    def _cont(self, state):
        return self.agent.agent.wm.heads["cont"](state).mean().reshape((-1,))

    def _repeat_state_modes(self, state, modes: int):
        return self.jax.tree_util.tree_map(
            lambda value: self.jnp.repeat(value[:, None], modes, axis=1).reshape((value.shape[0] * modes,) + value.shape[1:]),
            state,
        )

    def _mode_returns(self, state, speed, modes_xy):
        batch, modes = modes_xy.shape[:2]
        modes_xy = self.apply_plan_sign(modes_xy, self.plan_x_sign, self.plan_y_sign)
        actions = self.differentiable_pidpp(modes_xy, speed, self.ctrl_config).reshape((batch * modes, -1))
        cur_state = self._repeat_state_modes(state, modes)
        returns = self.jnp.zeros((batch * modes,), self.jnp.float32)
        discount = self.jnp.ones((batch * modes,), self.jnp.float32)
        for _ in range(self.imag_horizon):
            cur_state = self.agent.agent.wm.rssm.img_step(cur_state, actions)
            reward = self._reward(cur_state)
            cont = self._cont(cur_state)
            returns = returns + discount * reward
            discount = discount * float(self.discount) * cont
        return returns.reshape((batch, modes))

    def start(self, batch: Dict[str, np.ndarray]):
        data = self.jax.tree_util.tree_map(lambda x: self.jax.device_put(x, self.device), batch)
        rng = self.agent._next_rngs(self.agent.train_devices)
        (state, speed, feat), self.agent.varibs = self.start_fn(self.agent.varibs, rng, data)
        return state, speed, self.jax.device_get(feat)

    def score(self, state, speed, modes_xy: np.ndarray) -> np.ndarray:
        modes_xy = self.jax.device_put(modes_xy.astype(np.float32), self.device)
        rng = self.agent._next_rngs(self.agent.train_devices)
        returns, self.agent.varibs = self.return_fn(self.agent.varibs, rng, state, speed, modes_xy)
        return np.asarray(self.jax.device_get(returns), dtype=np.float32)


def load_torch_planner(checkpoint_path: str | Path, anchor_path: str | None, device):
    from RSSMDiDrOnCarla.models.rssm_didr_planner import RSSMDiDrConfig, RSSMDiffusionDrivePlanner

    payload = torch.load(checkpoint_path, map_location=device)
    if "config" not in payload or "model" not in payload:
        raise KeyError(f"{checkpoint_path} is not a planner checkpoint with `config` and `model` keys")
    config_dict = dict(payload["config"])
    if anchor_path:
        config_dict["plan_anchor_path"] = str(anchor_path)
    config = RSSMDiDrConfig(**config_dict)
    model = RSSMDiffusionDrivePlanner(config).to(device)
    model.load_state_dict(payload["model"])
    return model, config, int(payload.get("epoch", 0))


def freeze_except_selector(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("decoder.selector_head.")
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No decoder.selector_head parameters found.")
    return trainable


def make_noisy_xy(model, batch_size: int, device, eval_timestep: int):
    anchor = model.plan_anchor.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    timesteps = torch.full((batch_size,), int(eval_timestep), device=device, dtype=torch.long)
    if int(eval_timestep) <= 0:
        return anchor, timesteps
    normalized = model._normalize_xy(anchor)
    noise = torch.zeros_like(normalized)
    noisy = model.scheduler.add_noise(normalized, noise, timesteps)
    return model._denormalize_xy(noisy), timesteps


def selector_losses(new_logits, old_logits, jax_returns, args, device):
    policy_temperature = max(float(args.policy_temperature), 1e-6)
    return_temperature = max(float(args.return_temperature), 1e-6)
    adv_eps = max(float(args.adv_eps), 1e-8)

    new_logp = torch.log_softmax(new_logits / policy_temperature, dim=-1)
    new_prob = torch.softmax(new_logits / policy_temperature, dim=-1)
    old_logp = torch.log_softmax(old_logits / policy_temperature, dim=-1).detach()
    old_prob = torch.softmax(old_logits / policy_temperature, dim=-1).detach()

    returns = torch.from_numpy(jax_returns).to(device=device, dtype=torch.float32).detach()
    rank_return = returns - returns.mean(dim=-1, keepdim=True)
    rank_return = rank_return / torch.clamp(returns.std(dim=-1, keepdim=True), min=adv_eps)
    target = torch.softmax(rank_return / return_temperature, dim=-1).detach()
    ranking_loss = -(target * new_logp).sum(dim=-1).mean()

    adv = returns - returns.mean(dim=-1, keepdim=True)
    adv = adv / torch.clamp(returns.std(dim=-1, keepdim=True), min=adv_eps)
    adv = adv.detach()
    rl_loss = -(new_prob * adv).sum(dim=-1).mean()
    kl_loss = (old_prob * (old_logp - new_logp)).sum(dim=-1).mean()
    entropy = -(new_prob * new_logp).sum(dim=-1).mean()
    loss = (
        float(args.ranking_weight) * ranking_loss
        + float(args.rl_weight) * rl_loss
        + float(args.kl_weight) * kl_loss
        - float(args.entropy_weight) * entropy
    )
    with torch.no_grad():
        selected_idx = torch.argmax(new_logits, dim=-1)
        old_idx = torch.argmax(old_logits, dim=-1)
        oracle_idx = torch.argmax(returns, dim=-1)
        batch_idx = torch.arange(returns.shape[0], device=device)
        metrics = {
            "loss": loss.detach(),
            "ranking_loss": ranking_loss.detach(),
            "rl_loss": rl_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "entropy": entropy.detach(),
            "return_mean": returns.mean(),
            "return_std": returns.std(),
            "selected_return": returns[batch_idx, selected_idx].mean(),
            "old_selected_return": returns[batch_idx, old_idx].mean(),
            "oracle_return": returns.max(dim=-1).values.mean(),
            "rank_acc": (selected_idx == oracle_idx).float().mean(),
        }
    return loss, metrics


def scalarize(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def save_checkpoint(path: Path, model, optimizer, config, iteration: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(iteration),
            "selector_ranking_iteration": int(iteration),
        },
        path,
    )


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
    old_model = copy.deepcopy(model).to(device)
    old_model.eval()
    for param in old_model.parameters():
        param.requires_grad_(False)
    trainable = freeze_except_selector(model)
    model.eval()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    dataset = ReplaySequenceDataset(args.offline_replay_dir, args.batch_length)
    scorer = JAXRSSMModeScorer(args, extra, planner_config)

    run_config = {
        "args": vars(args),
        "planner_config": planner_config.to_dict(),
        "planner_checkpoint_epoch": checkpoint_epoch,
        "trainable_parameters": [name for name, param in model.named_parameters() if param.requires_grad],
        "replay_sequences": len(dataset),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    history = []
    iterator = range(1, int(args.iterations) + 1)
    if not args.no_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc="selector-ranking", dynamic_ncols=True)
    best_return = None

    for iteration in iterator:
        batch_np = dataset.sample(args.batch_size, np_rng)
        state, speed, condition_np = scorer.start(batch_np)
        condition = torch.from_numpy(condition_np).to(device=device, dtype=torch.float32)
        noisy_xy, timesteps = make_noisy_xy(model, condition.shape[0], device, args.eval_timestep)

        poses_reg, new_logits = model.decoder(condition, noisy_xy, timesteps)
        with torch.no_grad():
            _, old_logits = old_model.decoder(condition, noisy_xy, timesteps)
            modes_xy = poses_reg[..., :2].detach().cpu().numpy().astype(np.float32)

        jax_returns = scorer.score(state, speed, modes_xy)
        loss, metrics_t = selector_losses(new_logits, old_logits, jax_returns, args, device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        metrics = {"iteration": int(iteration), **scalarize(metrics_t)}
        history.append(metrics)
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "sel_ret": f"{metrics['selected_return']:.3f}",
                    "oracle": f"{metrics['oracle_return']:.3f}",
                    "acc": f"{metrics['rank_acc']:.3f}",
                }
            )
        if iteration == 1 or iteration % int(args.log_every) == 0:
            print(
                "[selector_ranking] "
                + " ".join(
                    f"{key}={value:.5f}" for key, value in metrics.items() if key != "iteration"
                ),
                flush=True,
            )
            (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        monitor = metrics["selected_return"]
        if best_return is None or monitor > best_return:
            best_return = monitor
            save_checkpoint(output_dir / "best.pt", model, optimizer, planner_config, iteration)
        if args.save_every > 0 and iteration % int(args.save_every) == 0:
            save_checkpoint(output_dir / f"iteration_{iteration:06d}.pt", model, optimizer, planner_config, iteration)

    save_checkpoint(output_dir / "last.pt", model, optimizer, planner_config, int(args.iterations))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[selector_ranking] Wrote checkpoints under {output_dir}")


if __name__ == "__main__":
    main()
