"""Open-loop evaluate RSSM posterior latents plus the DiffusionDrive planner.

The script reads replay chunks, builds RSSM posterior latents either from a
DreamerV3 checkpoint or from a pre-exported latent directory, runs the planner,
and measures the predicted ego-future trajectory against replay ego futures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dreamerv3") not in sys.path:
    sys.path.insert(0, str(ROOT / "dreamerv3"))

from JAXRSSMJAXDiDr.data.polyplanner_targets import (
    load_replay_chunk,
    require_ego_future_waypoints,
    waypoints8_to_trajectory,
)
from JAXRSSMJAXDiDr.data.gt_history_features import build_gt_history_features
from JAXRSSMJAXDiDr.scripts.export_planner_dataset import extract_rssm_latent


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_dir", required=True, help="Prepared replay directory with ego-future expert_waypoints8.")
    parser.add_argument("--planner_checkpoint", required=True, help="Planner checkpoint, for example best.pt or last.pt.")
    parser.add_argument("--output_dir", required=True, help="Directory for metrics and optional predictions.")
    parser.add_argument("--rssm_checkpoint", default=None, help="DreamerV3 RSSM checkpoint.ckpt. Used when --latent_dir is absent.")
    parser.add_argument("--latent_dir", default=None, help="Optional pre-exported RSSM latent chunks.")
    parser.add_argument("--anchor_path", default=None, help="Override anchor path stored in planner checkpoint.")
    parser.add_argument("--task", default="carla_roundabout")
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--rssm_batch_size", type=int, default=16)
    parser.add_argument("--jax_platform", default=None, choices=("cpu", "gpu", "tpu"))
    parser.add_argument(
        "--structured_world_model",
        action="store_true",
        help="Use the structured-only Dreamer RSSM config when exporting RSSM latents online.",
    )
    parser.add_argument("--latent_key", default=None, help="Optional latent key in latent chunks.")
    parser.add_argument("--history_length", type=int, default=None, help="Override GT-history length for gt_history planner checkpoints.")
    parser.add_argument("--no_neighbors", action="store_true", help="Use ego history only for gt_history evaluation.")
    parser.add_argument("--no_align_neighbor_ids", action="store_true", help="Disable actor-id neighbor history alignment for gt_history evaluation.")
    parser.add_argument("--pair_by_order", action="store_true", help="Pair replay and latent chunks by sorted order.")
    parser.add_argument("--allow_length_mismatch", action="store_true", help="Trim replay/latent to the shorter length.")
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--eval_noise", choices=("clean", "fixed", "random"), default="clean")
    parser.add_argument("--eval_timestep", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--max_samples_per_chunk", type=int, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--save_plots", action="store_true", help="Save trajectory visualization PNGs.")
    parser.add_argument("--plot_samples", type=int, default=16, help="Maximum plotted samples per chunk.")
    parser.add_argument("--plot_stride", type=int, default=1, help="Stride between candidate samples for plotting.")
    parser.add_argument("--plot_modes", type=int, default=6, help="Number of top-scored planner modes to draw per sample.")
    parser.add_argument("--no_plot_neighbors", action="store_true", help="Do not draw neighbor_vehicles_local in saved plots/animations.")
    parser.add_argument("--plot_neighbor_limit", type=int, default=8, help="Maximum nearby vehicles to draw per frame.")
    parser.add_argument("--save_animation", action="store_true", help="Save animated trajectory visualization per chunk.")
    parser.add_argument("--animation_format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--animation_fps", type=int, default=6)
    parser.add_argument("--animation_samples", type=int, default=120, help="Maximum animation frames per chunk.")
    parser.add_argument("--animation_stride", type=int, default=1, help="Stride between animation frames.")
    return parser.parse_known_args()


def iter_replay_paths(replay_dir: str | Path, max_chunks: Optional[int]) -> list[Path]:
    paths = sorted(Path(replay_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No replay chunks found in: {replay_dir}")
    if max_chunks is not None:
        paths = paths[: max(0, max_chunks)]
    return paths


def path_arg(value: str | None, name: str) -> Path | None:
    if value is None:
        return None
    if value.startswith(":"):
        raise ValueError(
            f"{name}={value!r} looks like a PowerShell environment variable was used in a POSIX shell. "
            "Use `$RUN_ROOT/...` on bash/zsh, or `$env:RUN_ROOT/...` only in PowerShell."
        )
    return Path(value).expanduser()


def build_latent_pairs(replay_paths: list[Path], latent_dir: str | Path, pair_by_order: bool) -> Dict[Path, Path]:
    latent_dir = Path(latent_dir)
    if pair_by_order:
        latent_paths = sorted(latent_dir.glob("*.npz"))
        if len(replay_paths) != len(latent_paths):
            raise ValueError(
                f"Cannot pair by order: replay chunks={len(replay_paths)} latent chunks={len(latent_paths)}"
            )
        return dict(zip(replay_paths, latent_paths))

    pairs = {}
    for replay_path in replay_paths:
        latent_path = latent_dir / replay_path.name
        if not latent_path.is_file():
            raise FileNotFoundError(f"Missing latent chunk for {replay_path.name}: {latent_path}")
        pairs[replay_path] = latent_path
    return pairs


def load_latent_chunk(path: Path, latent_key: Optional[str]) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        chunk = {key: np.asarray(data[key]) for key in data.files}
    return extract_rssm_latent(chunk, latent_key=latent_key)


def flatten_rssm_latent(post: Dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for key in ("deter", "stoch"):
        if key in post:
            value = np.asarray(post[key], dtype=np.float32)
            parts.append(value.reshape(value.shape[0], -1))
    if not parts:
        raise KeyError(f"Could not build rssm_latent from posterior keys: {sorted(post.keys())}")
    return np.concatenate(parts, axis=-1).astype(np.float32)


def temporal_batch(chunk: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if "action" not in chunk:
        raise KeyError("Replay chunk is missing required `action` field")
    length = len(chunk["action"])
    batch = {}
    for key, value in chunk.items():
        value = np.asarray(value)
        if value.ndim == 0 or value.shape[0] != length:
            continue
        batch[key] = value[None]
    return batch


def make_rssm_export_fn(agent, nj):
    def export_post(data):
        data = agent.agent.preprocess(data)
        embed = agent.agent.wm.encoder(data)
        prev_latent, prev_action = agent.agent.wm.initial(data["action"].shape[0])
        prev_actions = jax_jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)
        post, _ = agent.agent.wm.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        return post

    return nj.jit(nj.pure(export_post), device=agent.train_devices[0])


jax = None
jax_jnp = None


class RSSMLatentExporter:
    def __init__(self, args: argparse.Namespace, extra: list[str]):
        global jax, jax_jnp
        import jax as jax_module
        import jax.numpy as jnp_module
        from JAXRSSMJAXDiDr.scripts import train_offline_rssm as offline

        jax = jax_module
        jax_jnp = jnp_module
        offline.import_runtime()

        rssm_args = argparse.Namespace(
            task=args.task,
            replay_dir=args.replay_dir,
            logdir=str(Path(args.output_dir) / "rssm_eval_runtime"),
            batch_length=args.batch_length,
            batch_size=args.rssm_batch_size,
            replay_size=int(1e6),
            jax_platform=args.jax_platform,
            from_checkpoint="",
            structured_world_model=bool(args.structured_world_model),
        )
        _, config = offline.build_config(rssm_args, extra)
        obs_space, act_space = offline.infer_spaces_from_replay(args.replay_dir)
        step = offline.embodied.Counter()
        self.agent = offline.dreamerv3.Agent(obs_space, act_space, step, config.dreamerv3)
        if len(self.agent.train_devices) != 1:
            raise ValueError("Open-loop RSSM export expects one train device. Set dreamerv3.jax.train_devices=[0].")

        checkpoint = offline.embodied.Checkpoint(log=False, parallel=False)
        checkpoint.agent = self.agent
        checkpoint.load(args.rssm_checkpoint, keys=["agent"])
        self.export_fn = make_rssm_export_fn(self.agent, offline.nj)

    def export(self, chunk: Dict[str, np.ndarray]) -> np.ndarray:
        batch = temporal_batch(chunk)
        data = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.agent.train_devices[0]), batch)
        rng = self.agent._next_rngs(self.agent.train_devices)
        post, _ = self.export_fn(self.agent.varibs, rng, data)
        post = jax.device_get(post)
        post = {key: np.asarray(value[0], dtype=np.float32) for key, value in post.items()}
        return flatten_rssm_latent(post)


def load_planner(checkpoint_path: str | Path, anchor_path: Optional[str], device: torch.device):
    from JAXRSSMJAXDiDr.models.rssm_didr_planner import RSSMDiDrConfig, RSSMDiffusionDrivePlanner

    payload = torch.load(checkpoint_path, map_location=device)
    if "config" not in payload or "model" not in payload:
        raise KeyError(f"{checkpoint_path} is not a planner checkpoint with `config` and `model` keys")
    config_dict = dict(payload["config"])
    if anchor_path:
        config_dict["plan_anchor_path"] = str(anchor_path)
    config = RSSMDiDrConfig(**config_dict)
    model = RSSMDiffusionDrivePlanner(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


def make_noisy_xy(model, batch_size: int, device: torch.device, args: argparse.Namespace, generator):
    anchor = model.plan_anchor.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    if args.eval_noise == "clean":
        timesteps = torch.zeros((batch_size,), device=device, dtype=torch.long)
        return anchor, timesteps

    timesteps = torch.full((batch_size,), int(args.eval_timestep), device=device, dtype=torch.long)
    normalized = model._normalize_xy(anchor)
    if args.eval_noise == "fixed":
        noise = torch.randn(normalized.shape, generator=generator, device=device, dtype=normalized.dtype)
    else:
        noise = torch.randn_like(normalized)
    noisy = model.scheduler.add_noise(normalized, noise, timesteps).clamp(-1.0, 1.0)
    return model._denormalize_xy(noisy), timesteps


def metric_sums(model, latent: np.ndarray, trajectory: np.ndarray, device: torch.device, args: argparse.Namespace):
    totals = {
        "samples": 0,
        "loss_sum": 0.0,
        "reg_loss_sum": 0.0,
        "cls_loss_sum": 0.0,
        "oracle_ade_sum": 0.0,
        "oracle_fde_sum": 0.0,
        "selected_ade_sum": 0.0,
        "selected_fde_sum": 0.0,
        "anchor_oracle_ade_sum": 0.0,
        "anchor_oracle_fde_sum": 0.0,
        "cls_correct": 0.0,
    }
    predictions = {
        "selected": [],
        "oracle": [],
        "anchor_oracle": [],
        "mode_idx": [],
        "selected_idx": [],
        "top_modes": [],
        "top_mode_idx": [],
        "top_mode_scores": [],
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    with torch.no_grad():
        for start in range(0, len(latent), args.eval_batch_size):
            end = min(start + args.eval_batch_size, len(latent))
            latent_t = torch.from_numpy(latent[start:end]).to(device=device, dtype=torch.float32)
            target_t = torch.from_numpy(trajectory[start:end]).to(device=device, dtype=torch.float32)
            batch_size = end - start

            noisy_xy, timesteps = make_noisy_xy(model, batch_size, device, args, generator)
            poses_reg, poses_cls = model.decoder(latent_t, noisy_xy, timesteps)
            selected = model.select_best(poses_reg, poses_cls)

            plan_anchor = model.plan_anchor.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            dist = torch.linalg.norm(target_t.unsqueeze(1)[..., :2] - plan_anchor, dim=-1).mean(dim=-1)
            mode_idx = torch.argmin(dist, dim=-1)
            selected_idx = poses_cls.argmax(dim=-1)
            gather_idx = mode_idx[:, None, None, None].repeat(1, 1, model.config.num_poses, 3)
            oracle = torch.gather(poses_reg, 1, gather_idx).squeeze(1)
            anchor_oracle = torch.gather(
                plan_anchor,
                1,
                mode_idx[:, None, None, None].repeat(1, 1, model.config.num_poses, 2),
            ).squeeze(1)

            reg_loss = F.l1_loss(oracle, target_t)
            cls_loss = F.cross_entropy(poses_cls, mode_idx)
            loss = model.config.reg_loss_weight * reg_loss + model.config.cls_loss_weight * cls_loss
            oracle_ade = torch.linalg.norm(oracle[..., :2] - target_t[..., :2], dim=-1).mean()
            oracle_fde = torch.linalg.norm(oracle[..., -1, :2] - target_t[..., -1, :2], dim=-1).mean()
            selected_ade = torch.linalg.norm(selected[..., :2] - target_t[..., :2], dim=-1).mean()
            selected_fde = torch.linalg.norm(selected[..., -1, :2] - target_t[..., -1, :2], dim=-1).mean()
            anchor_ade = torch.linalg.norm(anchor_oracle - target_t[..., :2], dim=-1).mean()
            anchor_fde = torch.linalg.norm(anchor_oracle[..., -1, :] - target_t[..., -1, :2], dim=-1).mean()

            totals["samples"] += batch_size
            totals["loss_sum"] += float(loss.cpu()) * batch_size
            totals["reg_loss_sum"] += float(reg_loss.cpu()) * batch_size
            totals["cls_loss_sum"] += float(cls_loss.cpu()) * batch_size
            totals["oracle_ade_sum"] += float(oracle_ade.cpu()) * batch_size
            totals["oracle_fde_sum"] += float(oracle_fde.cpu()) * batch_size
            totals["selected_ade_sum"] += float(selected_ade.cpu()) * batch_size
            totals["selected_fde_sum"] += float(selected_fde.cpu()) * batch_size
            totals["anchor_oracle_ade_sum"] += float(anchor_ade.cpu()) * batch_size
            totals["anchor_oracle_fde_sum"] += float(anchor_fde.cpu()) * batch_size
            totals["cls_correct"] += float((selected_idx == mode_idx).float().sum().cpu())

            if args.save_predictions:
                topk = min(max(int(getattr(args, "plot_modes", 6)), 1), poses_reg.shape[1])
                top_scores, top_indices = torch.topk(poses_cls, k=topk, dim=-1)
                top_gather = top_indices[:, :, None, None].repeat(1, 1, model.config.num_poses, 3)
                top_modes = torch.gather(poses_reg, 1, top_gather)
                predictions["selected"].append(selected.cpu().numpy())
                predictions["oracle"].append(oracle.cpu().numpy())
                predictions["anchor_oracle"].append(anchor_oracle.cpu().numpy())
                predictions["mode_idx"].append(mode_idx.cpu().numpy())
                predictions["selected_idx"].append(selected_idx.cpu().numpy())
                predictions["top_modes"].append(top_modes.cpu().numpy())
                predictions["top_mode_idx"].append(top_indices.cpu().numpy())
                predictions["top_mode_scores"].append(top_scores.cpu().numpy())

    return totals, predictions


def finalize_metrics(totals: Dict[str, float]) -> Dict[str, float]:
    samples = max(int(totals["samples"]), 1)
    return {
        "samples": int(totals["samples"]),
        "loss": totals["loss_sum"] / samples,
        "reg_loss": totals["reg_loss_sum"] / samples,
        "cls_loss": totals["cls_loss_sum"] / samples,
        "oracle_ade": totals["oracle_ade_sum"] / samples,
        "oracle_fde": totals["oracle_fde_sum"] / samples,
        "selected_ade": totals["selected_ade_sum"] / samples,
        "selected_fde": totals["selected_fde_sum"] / samples,
        "anchor_oracle_ade": totals["anchor_oracle_ade_sum"] / samples,
        "anchor_oracle_fde": totals["anchor_oracle_fde_sum"] / samples,
        "cls_acc": totals["cls_correct"] / samples,
    }


def add_totals(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + float(value)


def extract_neighbor_vehicles(arrays: Dict[str, np.ndarray], idx: int, args: argparse.Namespace) -> np.ndarray:
    if args.no_plot_neighbors or "neighbor_vehicles_local" not in arrays:
        return np.zeros((0, 11), dtype=np.float32)
    neighbor = np.asarray(arrays["neighbor_vehicles_local"][idx], dtype=np.float32)
    if neighbor.ndim != 1 or neighbor.size % 11 != 0:
        return np.zeros((0, 11), dtype=np.float32)
    neighbor = neighbor.reshape(-1, 11)
    valid = neighbor[:, 0] > 0.5
    neighbor = neighbor[valid]
    if len(neighbor) == 0:
        return np.zeros((0, 11), dtype=np.float32)
    neighbor = neighbor[np.argsort(np.linalg.norm(neighbor[:, 1:3], axis=-1))]
    return neighbor[: max(int(args.plot_neighbor_limit), 0)]


def neighbor_vehicle_polygons(neighbor: np.ndarray) -> list[np.ndarray]:
    polygons = []
    for veh in np.asarray(neighbor, dtype=np.float32):
        x, y = float(veh[1]), float(veh[2])
        yaw = math.atan2(float(veh[5]), float(veh[6]))
        length = float(np.clip(veh[7], 2.0, 8.0))
        width = float(np.clip(veh[8], 1.2, 3.5))
        forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        right = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        center = np.asarray([x, y], dtype=np.float32)
        polygons.append(
            np.asarray(
                [
                    center + 0.5 * length * forward + 0.5 * width * right,
                    center + 0.5 * length * forward - 0.5 * width * right,
                    center - 0.5 * length * forward - 0.5 * width * right,
                    center - 0.5 * length * forward + 0.5 * width * right,
                ],
                dtype=np.float32,
            )
        )
    return polygons


def draw_neighbor_vehicles(ax, neighbor: np.ndarray, label: bool = True) -> None:
    from matplotlib.patches import Polygon

    for poly_idx, poly in enumerate(neighbor_vehicle_polygons(neighbor)):
        patch = Polygon(
            poly,
            closed=True,
            facecolor="#9edae5",
            edgecolor="#006d77",
            linewidth=1.1,
            alpha=0.65,
            zorder=4,
            label="Neighbor vehicles" if label and poly_idx == 0 else None,
        )
        ax.add_patch(patch)
    if len(neighbor):
        ax.scatter(
            neighbor[:, 1],
            neighbor[:, 2],
            marker=".",
            color="#004f57",
            s=18,
            zorder=5,
        )


def plot_chunk_trajectories(output_dir: Path, replay_path: Path, arrays: Dict[str, np.ndarray], args: argparse.Namespace) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    count = min(int(args.plot_samples), len(arrays["trajectory"]))
    stride = max(int(args.plot_stride), 1)
    indices = list(range(0, len(arrays["trajectory"]), stride))[:count]
    written = []

    for idx in indices:
        gt = arrays["trajectory"][idx, :, :2]
        selected = arrays["selected"][idx, :, :2]
        oracle = arrays["oracle"][idx, :, :2]
        anchor = arrays["anchor_oracle"][idx]
        top_modes = arrays["top_modes"][idx, :, :, :2]
        top_mode_idx = arrays["top_mode_idx"][idx]
        top_mode_scores = arrays["top_mode_scores"][idx]
        mode_idx = int(arrays["mode_idx"][idx])
        selected_idx = int(arrays["selected_idx"][idx])
        neighbor = extract_neighbor_vehicles(arrays, idx, args)

        fig, ax = plt.subplots(figsize=(5.8, 5.8), dpi=140)
        mode_colors = ["#d62728", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
        for rank, mode_xy in enumerate(top_modes):
            color = mode_colors[rank % len(mode_colors)]
            alpha = 0.9 if rank == 0 else 0.46
            linewidth = 2.1 if rank == 0 else 1.25
            label = f"Top {rank + 1} mode {int(top_mode_idx[rank])} score={float(top_mode_scores[rank]):.2f}"
            ax.plot(
                mode_xy[:, 0],
                mode_xy[:, 1],
                "o-",
                color=color,
                linewidth=linewidth,
                markersize=2.8,
                alpha=alpha,
                label=label,
            )
        ax.plot(gt[:, 0], gt[:, 1], "o-", color="#111111", linewidth=2.4, markersize=4, label="GT ego future")
        ax.plot(selected[:, 0], selected[:, 1], "-", color="#d62728", linewidth=3.0, alpha=0.95, label="Selected planner")
        ax.plot(oracle[:, 0], oracle[:, 1], "o--", color="#1f77b4", linewidth=1.8, markersize=3, label="Oracle mode reg")
        ax.plot(anchor[:, 0], anchor[:, 1], "x--", color="#2ca02c", linewidth=1.4, markersize=4, label="Closest anchor")
        draw_neighbor_vehicles(ax, neighbor)
        ax.scatter([0.0], [0.0], marker="+", color="#555555", s=80, label="Ego")
        ax.set_title(f"{replay_path.name} sample {idx} | selected={selected_idx} target={mode_idx}")
        ax.set_xlabel("x meters")
        ax.set_ylabel("y meters")
        ax.axis("equal")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()

        output_path = plot_dir / f"{replay_path.stem}_sample_{idx:06d}.png"
        fig.savefig(output_path)
        plt.close(fig)
        written.append(str(output_path))

    return written


def save_chunk_animation(output_dir: Path, replay_path: Path, arrays: Dict[str, np.ndarray], args: argparse.Namespace) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    anim_dir = output_dir / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    stride = max(int(args.animation_stride), 1)
    count = min(int(args.animation_samples), len(arrays["trajectory"]))
    indices = list(range(0, len(arrays["trajectory"]), stride))[:count]
    if not indices:
        raise ValueError(f"No animation frames for {replay_path.name}")

    top_modes_all = arrays["top_modes"][indices, :, :, :2]
    xy_parts = [
        arrays["trajectory"][indices, :, :2].reshape(-1, 2),
        arrays["selected"][indices, :, :2].reshape(-1, 2),
        arrays["oracle"][indices, :, :2].reshape(-1, 2),
        arrays["anchor_oracle"][indices].reshape(-1, 2),
        top_modes_all.reshape(-1, 2),
        np.zeros((1, 2), dtype=np.float32),
    ]
    if not args.no_plot_neighbors and "neighbor_vehicles_local" in arrays:
        neighbor_parts = [extract_neighbor_vehicles(arrays, idx, args)[:, 1:3] for idx in indices]
        neighbor_xy = np.concatenate([part for part in neighbor_parts if len(part)], axis=0) if any(len(part) for part in neighbor_parts) else None
        if neighbor_xy is not None:
            xy_parts.append(neighbor_xy)
    finite = np.concatenate([part[np.isfinite(part).all(axis=1)] for part in xy_parts if len(part)], axis=0)
    if len(finite) == 0:
        finite = np.zeros((1, 2), dtype=np.float32)
    pad = 3.0
    x_min = min(-5.0, float(np.min(finite[:, 0])) - pad)
    x_max = max(15.0, float(np.max(finite[:, 0])) + pad)
    y_abs = max(8.0, float(np.max(np.abs(finite[:, 1]))) + pad)

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=120)
    mode_colors = ["#d62728", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
    mode_lines = []
    for rank in range(top_modes_all.shape[1]):
        (line,) = ax.plot(
            [],
            [],
            "o-",
            color=mode_colors[rank % len(mode_colors)],
            linewidth=2.0 if rank == 0 else 1.2,
            markersize=2.7,
            alpha=0.9 if rank == 0 else 0.46,
        )
        mode_lines.append(line)
    (gt_line,) = ax.plot([], [], "o-", color="#111111", linewidth=2.4, markersize=4, label="GT ego future")
    (selected_line,) = ax.plot([], [], "-", color="#d62728", linewidth=3.0, alpha=0.95, label="Selected planner")
    (oracle_line,) = ax.plot([], [], "o--", color="#1f77b4", linewidth=1.8, markersize=3, label="Oracle mode reg")
    (anchor_line,) = ax.plot([], [], "x--", color="#2ca02c", linewidth=1.4, markersize=4, label="Closest anchor")
    max_neighbor_patches = 0
    if not args.no_plot_neighbors and "neighbor_vehicles_local" in arrays:
        max_neighbor_patches = min(
            max(int(args.plot_neighbor_limit), 0),
            int(np.asarray(arrays["neighbor_vehicles_local"]).shape[-1] // 11),
        )
    neighbor_patches = []
    for patch_idx in range(max_neighbor_patches):
        patch = Polygon(
            np.zeros((4, 2), dtype=np.float32),
            closed=True,
            facecolor="#9edae5",
            edgecolor="#006d77",
            linewidth=1.1,
            alpha=0.65,
            visible=False,
            zorder=4,
            label="Neighbor vehicles" if patch_idx == 0 else None,
        )
        ax.add_patch(patch)
        neighbor_patches.append(patch)
    neighbor_points = ax.scatter([], [], marker=".", color="#004f57", s=18, zorder=5)
    ax.scatter([0.0], [0.0], marker="+", color="#555555", s=80, label="Ego")
    title = ax.set_title("")
    ax.set_xlabel("x meters")
    ax.set_ylabel("y meters")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_abs, y_abs)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()

    def update(frame_idx: int):
        idx = indices[frame_idx]
        gt = arrays["trajectory"][idx, :, :2]
        selected = arrays["selected"][idx, :, :2]
        oracle = arrays["oracle"][idx, :, :2]
        anchor = arrays["anchor_oracle"][idx]
        top_modes = arrays["top_modes"][idx, :, :, :2]
        top_mode_idx = arrays["top_mode_idx"][idx]
        top_mode_scores = arrays["top_mode_scores"][idx]
        mode_idx = int(arrays["mode_idx"][idx])
        selected_idx = int(arrays["selected_idx"][idx])
        neighbor = extract_neighbor_vehicles(arrays, idx, args)

        for rank, line in enumerate(mode_lines):
            if rank < len(top_modes):
                mode_xy = top_modes[rank]
                line.set_data(mode_xy[:, 0], mode_xy[:, 1])
                line.set_label(f"Top {rank + 1} mode {int(top_mode_idx[rank])} score={float(top_mode_scores[rank]):.2f}")
            else:
                line.set_data([], [])
        gt_line.set_data(gt[:, 0], gt[:, 1])
        selected_line.set_data(selected[:, 0], selected[:, 1])
        oracle_line.set_data(oracle[:, 0], oracle[:, 1])
        anchor_line.set_data(anchor[:, 0], anchor[:, 1])
        polygons = neighbor_vehicle_polygons(neighbor)
        for patch_idx, patch in enumerate(neighbor_patches):
            if patch_idx < len(polygons):
                patch.set_xy(polygons[patch_idx])
                patch.set_visible(True)
            else:
                patch.set_visible(False)
        if len(neighbor):
            neighbor_points.set_offsets(neighbor[:, 1:3])
        else:
            neighbor_points.set_offsets(np.zeros((0, 2), dtype=np.float32))
        title.set_text(f"{replay_path.name} sample {idx} | selected={selected_idx} target={mode_idx}")
        return [*mode_lines, gt_line, selected_line, oracle_line, anchor_line, *neighbor_patches, neighbor_points, title]

    anim = animation.FuncAnimation(fig, update, frames=len(indices), interval=1000 / max(int(args.animation_fps), 1), blit=False)
    output_path = anim_dir / f"{replay_path.stem}.{args.animation_format}"
    if args.animation_format == "gif":
        writer = animation.PillowWriter(fps=max(int(args.animation_fps), 1))
    else:
        if not animation.writers.is_available("ffmpeg"):
            plt.close(fig)
            raise RuntimeError("Matplotlib ffmpeg writer is not available. Install ffmpeg or use --animation_format gif.")
        writer = animation.FFMpegWriter(fps=max(int(args.animation_fps), 1), bitrate=1800)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    return str(output_path)


def main() -> None:
    args, extra = parse_args()
    if torch is None:
        raise ModuleNotFoundError("Open-loop planner evaluation requires PyTorch.") from TORCH_IMPORT_ERROR

    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    replay_dir = path_arg(args.replay_dir, "--replay_dir")
    output_dir = path_arg(args.output_dir, "--output_dir")
    latent_dir = path_arg(args.latent_dir, "--latent_dir")
    rssm_checkpoint = path_arg(args.rssm_checkpoint, "--rssm_checkpoint")
    planner_checkpoint = path_arg(args.planner_checkpoint, "--planner_checkpoint")
    anchor_path = path_arg(args.anchor_path, "--anchor_path")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, config = load_planner(planner_checkpoint, str(anchor_path) if anchor_path else None, device)
    condition_key = getattr(config, "condition_key", "rssm_latent")
    condition_type = getattr(config, "condition_type", "rssm_latent")
    uses_gt_history = condition_key == "gt_history" or condition_type == "gt_history"
    history_length = int(args.history_length or getattr(config, "gt_history_length", 10))
    include_neighbors = bool(getattr(config, "gt_history_include_neighbors", True)) and not args.no_neighbors
    align_neighbor_ids = bool(getattr(config, "gt_history_align_neighbor_ids", True)) and not args.no_align_neighbor_ids
    if not uses_gt_history and not latent_dir and not rssm_checkpoint:
        raise ValueError("Provide either --rssm_checkpoint for joint RSSM+planner eval or --latent_dir for cached latents.")

    replay_paths = iter_replay_paths(replay_dir, args.max_chunks)
    latent_pairs = build_latent_pairs(replay_paths, latent_dir, args.pair_by_order) if latent_dir and not uses_gt_history else {}
    args.replay_dir = str(replay_dir)
    args.latent_dir = str(latent_dir) if latent_dir else None
    args.rssm_checkpoint = str(rssm_checkpoint) if rssm_checkpoint else None
    rssm_exporter = None if (args.latent_dir or uses_gt_history) else RSSMLatentExporter(args, extra)

    total_sums: Dict[str, float] = {}
    chunk_metrics = []
    for replay_path in replay_paths:
        chunk = load_replay_chunk(replay_path, trim_to_length=True)
        waypoints = require_ego_future_waypoints(chunk, replay_path)
        trajectory = waypoints8_to_trajectory(waypoints, waypoint_scale=args.waypoint_scale)
        if uses_gt_history:
            latent = build_gt_history_features(
                chunk,
                history_length=history_length,
                waypoint_scale=args.waypoint_scale,
                include_neighbors=include_neighbors,
                align_neighbor_ids=align_neighbor_ids,
            )
        elif args.latent_dir:
            latent = load_latent_chunk(latent_pairs[replay_path], args.latent_key)
        else:
            latent = rssm_exporter.export(chunk)

        length = min(len(latent), len(trajectory))
        if len(latent) != len(trajectory) and not args.allow_length_mismatch:
            raise ValueError(
                f"Length mismatch for {replay_path.name}: latent={len(latent)} trajectory={len(trajectory)}. "
                "Use --allow_length_mismatch to trim to the shorter length."
            )
        if args.max_samples_per_chunk is not None:
            length = min(length, args.max_samples_per_chunk)
        latent = latent[:length].astype(np.float32)
        trajectory = trajectory[:length].astype(np.float32)
        neighbor_vehicles_local = None
        if "neighbor_vehicles_local" in chunk:
            neighbor_arr = np.asarray(chunk["neighbor_vehicles_local"], dtype=np.float32)
            neighbor_len = int(neighbor_arr.shape[0]) if neighbor_arr.ndim >= 1 else 0
            if neighbor_len >= length:
                neighbor_vehicles_local = neighbor_arr[:length].astype(np.float32)
            elif args.save_plots or args.save_animation:
                print(
                    f"[open_loop] {replay_path.name}: skip neighbor visualization because "
                    f"neighbor_vehicles_local length={neighbor_len} < samples={length}"
                )

        need_arrays = args.save_predictions or args.save_plots or args.save_animation
        eval_args = argparse.Namespace(**vars(args))
        eval_args.save_predictions = need_arrays
        sums, predictions = metric_sums(model, latent, trajectory, device, eval_args)
        metrics = finalize_metrics(sums)
        metrics["replay_chunk"] = replay_path.name
        chunk_metrics.append(metrics)
        add_totals(total_sums, sums)
        print(
            f"[open_loop] {replay_path.name} samples={metrics['samples']} "
            f"selected_ADE={metrics['selected_ade']:.4f} oracle_ADE={metrics['oracle_ade']:.4f} "
            f"cls_acc={metrics['cls_acc']:.4f}"
        )

        if need_arrays:
            out = {
                "trajectory": trajectory,
                "selected": np.concatenate(predictions["selected"], axis=0),
                "oracle": np.concatenate(predictions["oracle"], axis=0),
                "anchor_oracle": np.concatenate(predictions["anchor_oracle"], axis=0),
                "mode_idx": np.concatenate(predictions["mode_idx"], axis=0),
                "selected_idx": np.concatenate(predictions["selected_idx"], axis=0),
                "top_modes": np.concatenate(predictions["top_modes"], axis=0),
                "top_mode_idx": np.concatenate(predictions["top_mode_idx"], axis=0),
                "top_mode_scores": np.concatenate(predictions["top_mode_scores"], axis=0),
            }
            if neighbor_vehicles_local is not None:
                out["neighbor_vehicles_local"] = neighbor_vehicles_local
            if args.save_predictions:
                np.savez_compressed(output_dir / f"{replay_path.stem}_predictions.npz", **out)
            if args.save_plots:
                plot_paths = plot_chunk_trajectories(output_dir, replay_path, out, args)
                metrics["plot_files"] = plot_paths
            if args.save_animation:
                animation_path = save_chunk_animation(output_dir, replay_path, out, args)
                metrics["animation_file"] = animation_path

    summary = {
        "replay_dir": str(args.replay_dir),
        "latent_dir": str(args.latent_dir) if args.latent_dir else None,
        "rssm_checkpoint": str(args.rssm_checkpoint) if args.rssm_checkpoint else None,
        "planner_checkpoint": str(planner_checkpoint),
        "device": str(device),
        "eval_noise": args.eval_noise,
        "eval_timestep": int(args.eval_timestep),
        "planner_config": config.to_dict(),
        "metrics": finalize_metrics(total_sums),
        "chunks": chunk_metrics,
    }
    summary_path = output_dir / "open_loop_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[open_loop] Wrote metrics: {summary_path}")


if __name__ == "__main__":
    main()
