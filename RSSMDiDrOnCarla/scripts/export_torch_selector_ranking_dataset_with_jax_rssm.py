"""Export selector-ranking data from a Torch planner and frozen JAX RSSM.

The exported dataset is meant for selector-only finetuning. It caches the
frozen planner decoder features, old selector logits, and JAX RSSM imagined
returns for every candidate mode, so the later training stage only needs
PyTorch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import ruamel.yaml as yaml

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
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_stride", type=int, default=1)
    parser.add_argument("--output_chunk_size", type=int, default=4096)
    parser.add_argument("--max_replay_chunks", type=int, default=None)
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--imag_horizon", type=int, default=8)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--eval_timestep", type=int, default=8)
    parser.add_argument("--save_trajectories", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jax_platform", default=None, choices=("cpu", "gpu", "tpu"))
    parser.add_argument("--seed", type=int, default=0)
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


def iter_replay_paths(directory: str | Path, max_chunks: int | None = None):
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No replay chunks found in {directory}")
    if max_chunks is not None:
        paths = paths[: max(0, int(max_chunks))]
    return paths


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def flatten_state_feat_jax(jnp, state):
    deter = state["deter"].reshape((state["deter"].shape[0], -1))
    stoch = state["stoch"].reshape((state["stoch"].shape[0], -1))
    return jnp.concatenate([deter, stoch], axis=-1)


def normalize_embodied_flags(extra: list[str]) -> list[str]:
    normalized = []
    for item in extra:
        if "=" in item and item.startswith("--"):
            key, value = item.split("=", 1)
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            normalized.append(f"{key}={value}")
        elif item.startswith("[") and item.endswith("]"):
            normalized.append(item[1:-1])
        else:
            normalized.append(item)
    return normalized


def build_offline_dreamer_config(embodied, args: argparse.Namespace, extra: list[str]):
    yaml_path = ROOT / "dreamerv3" / "dreamerv3.yaml"
    model_configs = yaml.YAML(typ="safe").load(embodied.Path(str(yaml_path)).read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})
    updates = {
        "dreamerv3.logdir": str(Path(args.output_dir) / "dreamer_runtime"),
        "dreamerv3.replay_dir": str(args.offline_replay_dir),
        "dreamerv3.batch_length": int(args.batch_length),
        "dreamerv3.batch_size": int(args.batch_size),
        "dreamerv3.replay_size": int(1e6),
        "dreamerv3.run.from_checkpoint": "",
    }
    if args.jax_platform:
        updates["dreamerv3.jax.platform"] = args.jax_platform
    config = config.update(updates)
    return embodied.Flags(config).parse(normalize_embodied_flags(extra))


def make_space(embodied, array: np.ndarray, *, low=None, high=None):
    array = np.asarray(array)
    shape = tuple(array.shape[1:]) if array.ndim > 0 else ()
    dtype = np.dtype(array.dtype).type
    try:
        return embodied.Space(dtype, shape, low=low, high=high)
    except TypeError:
        try:
            return embodied.Space(dtype, shape)
        except TypeError:
            return embodied.Space(dtype=dtype, shape=shape, low=low, high=high)


def infer_spaces_from_replay(embodied, replay_dir: str | Path):
    paths = sorted(Path(replay_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No replay chunks found in {replay_dir}")
    for path in paths:
        chunk = load_npz(path)
        if "action" not in chunk:
            continue
        action = np.asarray(chunk["action"])
        if action.ndim == 0:
            continue
        length = int(action.shape[0])
        obs_space = {}
        for key, value in chunk.items():
            value = np.asarray(value)
            if key == "action" or value.ndim == 0 or value.shape[0] != length:
                continue
            obs_space[key] = make_space(embodied, value)
        obs_space.setdefault("reward", make_space(embodied, np.zeros((length,), dtype=np.float32)))
        obs_space.setdefault("is_first", make_space(embodied, np.zeros((length,), dtype=bool)))
        obs_space.setdefault("is_last", make_space(embodied, np.zeros((length,), dtype=bool)))
        obs_space.setdefault("is_terminal", make_space(embodied, np.zeros((length,), dtype=bool)))
        act_space = {
            "action": make_space(embodied, action, low=-1.0, high=1.0),
            "reset": make_space(embodied, np.zeros((length,), dtype=bool)),
        }
        return obs_space, act_space
    raise KeyError(f"No replay chunk with temporal `action` found in {replay_dir}")


class ReplayWindowIterator:
    def __init__(self, replay_dir: str | Path, batch_length: int, stride: int, max_chunks: int | None):
        self.paths = iter_replay_paths(replay_dir, max_chunks=max_chunks)
        self.batch_length = int(batch_length)
        self.stride = max(int(stride), 1)

    def total_windows(self, max_sequences: int | None = None) -> int:
        total = 0
        for path in self.paths:
            chunk = load_npz(path)
            if "action" not in chunk:
                continue
            length = int(len(chunk["action"]))
            if length >= self.batch_length:
                total += len(range(0, length - self.batch_length + 1, self.stride))
        return min(total, int(max_sequences)) if max_sequences is not None else total

    def batches(self, batch_size: int, max_sequences: int | None = None):
        rows = []
        meta = []
        emitted = 0
        for chunk_idx, path in enumerate(self.paths):
            chunk = load_npz(path)
            if "action" not in chunk:
                continue
            length = int(len(chunk["action"]))
            if length < self.batch_length:
                continue
            keys = [key for key, value in chunk.items() if np.asarray(value).ndim > 0 and len(value) >= length]
            for start in range(0, length - self.batch_length + 1, self.stride):
                if max_sequences is not None and emitted >= int(max_sequences):
                    break
                end = start + self.batch_length
                row = {key: np.asarray(chunk[key][start:end]) for key in keys}
                if "executed_control" in row:
                    row["action"] = row["executed_control"]
                row.setdefault("is_first", np.zeros((self.batch_length,), dtype=bool))
                row.setdefault("is_terminal", np.zeros((self.batch_length,), dtype=bool))
                row.setdefault("reward", np.zeros((self.batch_length,), dtype=np.float32))
                rows.append(row)
                meta.append((chunk_idx, start))
                emitted += 1
                if len(rows) >= int(batch_size):
                    yield self._collate(rows), np.asarray(meta, dtype=np.int32)
                    rows, meta = [], []
            if max_sequences is not None and emitted >= int(max_sequences):
                break
        if rows:
            yield self._collate(rows), np.asarray(meta, dtype=np.int32)

    @staticmethod
    def _collate(rows):
        keys = sorted(set.intersection(*(set(row.keys()) for row in rows)))
        return {key: np.stack([row[key] for row in rows], axis=0) for key in keys}


class JAXRSSMModeScorer:
    def __init__(self, args: argparse.Namespace, extra: list[str], planner_config):
        import jax
        import jax.numpy as jnp
        import embodied
        import dreamerv3
        from dreamerv3 import ninjax as nj
        from JAXRSSMJAXDiDr.models import JAXDiDrConfig, apply_plan_sign, differentiable_pidpp

        self.jax = jax
        self.jnp = jnp
        self.apply_plan_sign = apply_plan_sign
        self.differentiable_pidpp = differentiable_pidpp
        config = build_offline_dreamer_config(embodied, args, extra)
        obs_space, act_space = infer_spaces_from_replay(embodied, args.offline_replay_dir)
        step = embodied.Counter()
        self.agent = dreamerv3.Agent(obs_space, act_space, step, config.dreamerv3)
        if len(self.agent.train_devices) != 1:
            raise ValueError("Set dreamerv3.jax.train_devices=[0].")
        checkpoint = embodied.Checkpoint(args.rssm_checkpoint, log=False, parallel=False)
        checkpoint.agent = self.agent
        checkpoint.load(keys=["agent"])
        self.device = self.agent.train_devices[0]
        print(
            f"[export_selector_ranking] loaded JAX RSSM checkpoint={args.rssm_checkpoint} "
            f"train_device={self.device}",
            flush=True,
        )
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
        self.start_fn = nj.jit(nj.pure(self._start), device=self.device)
        self.return_fn = nj.jit(nj.pure(self._mode_returns), device=self.device)

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
    model.eval()
    print(
        f"[export_selector_ranking] loaded Torch planner checkpoint={checkpoint_path} "
        f"epoch={int(payload.get('epoch', 0))} device={device} modes={config.num_modes} "
        f"poses={config.num_poses} latent_dim={config.latent_dim}",
        flush=True,
    )
    return model, config, int(payload.get("epoch", 0))


def make_noisy_xy(model, batch_size: int, device, eval_timestep: int):
    anchor = model.plan_anchor.to(device).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    timesteps = torch.full((batch_size,), int(eval_timestep), device=device, dtype=torch.long)
    if int(eval_timestep) <= 0:
        return anchor, timesteps
    normalized = model._normalize_xy(anchor)
    noise = torch.zeros_like(normalized)
    noisy = model.scheduler.add_noise(normalized, noise, timesteps)
    return model._denormalize_xy(noisy), timesteps


def planner_modes_and_features(model, condition_np: np.ndarray, device, eval_timestep: int):
    condition = torch.from_numpy(np.array(condition_np, dtype=np.float32, copy=True)).to(device=device)
    noisy_xy, timesteps = make_noisy_xy(model, condition.shape[0], device, eval_timestep)
    with torch.no_grad():
        features = model.decoder.decode_features(condition, noisy_xy, timesteps)
        delta = model.decoder.delta_head(features).reshape(
            condition.shape[0],
            model.config.num_modes,
            model.config.num_poses,
            3,
        )
        poses_reg = delta.clone()
        poses_reg[..., :2] = poses_reg[..., :2] + noisy_xy
        poses_reg[..., 2] = torch.tanh(poses_reg[..., 2]) * np.pi
        logits = model.decoder.selector_head(features).squeeze(-1)
    return (
        features.detach().cpu().numpy().astype(np.float32),
        logits.detach().cpu().numpy().astype(np.float32),
        poses_reg.detach().cpu().numpy().astype(np.float32),
    )


class ChunkWriter:
    def __init__(self, output_dir: Path, chunk_size: int, save_trajectories: bool):
        self.output_dir = output_dir
        self.chunk_size = int(chunk_size)
        self.save_trajectories = bool(save_trajectories)
        self.buffers = []
        self.index = 0
        self.samples = 0
        self.files = []

    def add(self, row: Dict[str, np.ndarray]):
        self.buffers.append(row)
        self.samples += int(row["returns"].shape[0])
        while self._buffer_len() >= self.chunk_size:
            self.flush(limit=self.chunk_size)

    def _buffer_len(self):
        return sum(int(row["returns"].shape[0]) for row in self.buffers)

    def flush(self, limit: int | None = None):
        if not self.buffers:
            return
        rows = []
        remaining = self._buffer_len() if limit is None else int(limit)
        next_buffers = []
        for row in self.buffers:
            n = int(row["returns"].shape[0])
            if remaining <= 0:
                next_buffers.append(row)
            elif n <= remaining:
                rows.append(row)
                remaining -= n
            else:
                take = remaining
                rows.append({key: value[:take] for key, value in row.items()})
                next_buffers.append({key: value[take:] for key, value in row.items()})
                remaining = 0
        self.buffers = next_buffers
        if not rows:
            return
        arrays = {key: np.concatenate([row[key] for row in rows], axis=0) for key in rows[0].keys()}
        if not self.save_trajectories and "poses_reg" in arrays:
            arrays.pop("poses_reg")
        path = self.output_dir / f"selector_ranking_{self.index:06d}.npz"
        np.savez_compressed(path, **arrays)
        self.files.append(path.name)
        self.index += 1


def main() -> None:
    args, extra = parse_args()
    if torch is None:
        raise ModuleNotFoundError("This script requires PyTorch.") from TORCH_IMPORT_ERROR

    if args.jax_platform:
        import jax

        jax.config.update("jax_platform_name", args.jax_platform)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, planner_config, checkpoint_epoch = load_torch_planner(args.planner_checkpoint, args.anchor_path, device)
    window_iter = ReplayWindowIterator(
        args.offline_replay_dir,
        batch_length=args.batch_length,
        stride=args.sequence_stride,
        max_chunks=args.max_replay_chunks,
    )
    total = window_iter.total_windows(args.max_sequences)
    scorer = JAXRSSMModeScorer(args, extra, planner_config)
    writer = ChunkWriter(output_dir, args.output_chunk_size, args.save_trajectories)

    iterator = window_iter.batches(args.batch_size, args.max_sequences)
    if not args.no_tqdm and tqdm is not None:
        iterator = tqdm(iterator, total=int(np.ceil(total / max(args.batch_size, 1))), desc="export-selector-ranking", dynamic_ncols=True)

    exported = 0
    for batch_np, meta in iterator:
        state, speed, condition_np = scorer.start(batch_np)
        selector_features, old_logits, poses_reg = planner_modes_and_features(
            model,
            condition_np,
            device,
            args.eval_timestep,
        )
        returns = scorer.score(state, speed, poses_reg[..., :2])
        row = {
            "selector_features": selector_features,
            "old_logits": old_logits,
            "returns": returns,
            "rssm_latent": condition_np.astype(np.float32),
            "source_chunk_index": meta[:, 0].astype(np.int32),
            "source_start": meta[:, 1].astype(np.int32),
        }
        if args.save_trajectories:
            row["poses_reg"] = poses_reg
        writer.add(row)
        exported += int(returns.shape[0])
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"samples": exported, "ret": f"{float(np.mean(returns)):.3f}"})

    writer.flush()
    metadata = {
        "args": vars(args),
        "planner_config": planner_config.to_dict(),
        "planner_checkpoint_epoch": checkpoint_epoch,
        "samples": int(exported),
        "files": writer.files,
        "replay_chunks": [path.name for path in window_iter.paths],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[export_selector_ranking] samples={exported} files={len(writer.files)} output_dir={output_dir}")


if __name__ == "__main__":
    main()
