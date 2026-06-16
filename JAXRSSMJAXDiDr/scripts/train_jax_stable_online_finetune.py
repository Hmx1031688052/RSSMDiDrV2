"""Stable collect/RSSM/selector fine-tuning loop for JAX RSSM + JAX DiDr.

The loop is intentionally conservative:

  collect CARLA replay with 0.5s receding-horizon plans and 0.1s controls
  -> update the online DreamerV3/JAX RSSM from offline + online replay
  -> snapshot/freeze the RSSM variables for planner scoring
  -> score every planner candidate mode with frozen RSSM imagination
  -> update only the JAX planner selector_head
  -> collect again

The diffusion body, trajectory decoder, encoders, and plan anchors are kept
fixed during selector updates. Candidate scoring never uses soft trajectory
averaging; each mode is controlled and imagined separately.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime as _datetime
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dreamerv3") not in sys.path:
    sys.path.insert(0, str(ROOT / "dreamerv3"))

import jax
import jax.numpy as jnp
import optax

from JAXRSSMJAXDiDr.models import load_checkpoint, save_checkpoint
from JAXRSSMJAXDiDr.models.controller import apply_plan_sign
from JAXRSSMJAXDiDr.models.jax_didr_planner import _decode, deterministic_anchors
from RSSMDiDrOnCarla.scripts import eval_close_loop_rssm_didr as closed_loop
from RSSMDiDrOnCarla.scripts import train_offline_rssm as offline_rssm


warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline_replay_dir", required=True)
    parser.add_argument("--rssm_checkpoint", required=True)
    parser.add_argument("--planner_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--online_replay_dir", default=None)
    parser.add_argument("--anchor_path", default=None)
    parser.add_argument("--task", default="carla_roundabout")

    parser.add_argument("--outer_iterations", type=int, default=10)
    parser.add_argument("--collect_episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--plan_interval_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--wm_updates", type=int, default=200)
    parser.add_argument("--selector_updates", type=int, default=200)
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--replay_size", type=float, default=1e6)
    parser.add_argument("--offline_ratio", type=float, default=0.7)

    parser.add_argument("--selector_lr", type=float, default=1e-5)
    parser.add_argument("--selector_weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_timestep", type=int, default=8)
    parser.add_argument("--score_horizon_steps", type=int, default=15)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--policy_temperature", type=float, default=1.0)
    parser.add_argument("--adv_clip", type=float, default=3.0)
    parser.add_argument("--adv_eps", type=float, default=1e-4)
    parser.add_argument("--ranking_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=0.05)
    parser.add_argument("--entropy_weight", type=float, default=0.005)

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--waypoint_dt", type=float, default=0.5)
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument(
        "--planner_output_unit",
        choices=("meters", "normalized"),
        default="meters",
        help="Unit of poses_reg xy. Current JAX planner checkpoints output meters; use normalized for custom normalized checkpoints.",
    )
    parser.add_argument("--plan_x_sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--plan_y_sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--target_speed_min", type=float, default=0.0)
    parser.add_argument("--target_speed_max", type=float, default=8.0)
    parser.add_argument("--fallback_speed", type=float, default=1.5)
    parser.add_argument("--stop_speed", type=float, default=0.25)
    parser.add_argument("--speed_kp", type=float, default=1.0)
    parser.add_argument("--speed_ki", type=float, default=0.0)
    parser.add_argument("--speed_kd", type=float, default=0.0)
    parser.add_argument("--acc_min", type=float, default=-3.0)
    parser.add_argument("--acc_max", type=float, default=2.0)
    parser.add_argument("--env_acc_min", type=float, default=-3.0)
    parser.add_argument("--env_acc_max", type=float, default=3.0)
    parser.add_argument("--wheelbase", type=float, default=2.875)
    parser.add_argument("--max_steer_rad", type=float, default=0.65)
    parser.add_argument("--steer_gain", type=float, default=0.5)
    parser.add_argument("--steer_sign", type=float, default=-1.0)
    parser.add_argument("--steer_rate_limit", type=float, default=0.08)
    parser.add_argument("--lookahead_min", type=float, default=4.5)
    parser.add_argument("--lookahead_max", type=float, default=14.0)
    parser.add_argument("--lookahead_gain", type=float, default=1.0)
    parser.add_argument("--min_valid_points", type=int, default=3)
    parser.add_argument("--min_forward_x", type=float, default=0.2)
    parser.add_argument("--max_abs_y", type=float, default=20.0)
    parser.add_argument("--max_step_distance", type=float, default=12.0)

    parser.add_argument("--jax_platform", default=None, choices=("cpu", "gpu", "tpu"))
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every_outer", type=int, default=1)
    parser.add_argument("--no_collect", action="store_true")
    return parser.parse_known_args()


def ensure_control_action_extra(extra: list[str]) -> list[str]:
    extra = closed_loop.normalize_embodied_flags(extra)
    if not closed_loop.has_flag(extra, "env.planner_target.use_waypoint_action"):
        extra = [*extra, "--env.planner_target.use_waypoint_action", "False"]
    return extra


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def replay_paths(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(directory.glob("*.npz")) if directory.exists() else []


class ReplaySequenceDataset:
    """Numpy sequence sampler for Dreamer row-format replay chunks."""

    def __init__(
        self,
        replay_dirs: Iterable[str | Path],
        batch_length: int,
        allowed_keys: Optional[set[str]] = None,
    ):
        self.replay_dirs = [Path(path) for path in replay_dirs if path is not None]
        self.batch_length = int(batch_length)
        self.allowed_keys = set(allowed_keys) if allowed_keys is not None else None
        self.paths: list[Path] = []
        self.lengths: list[int] = []
        for directory in self.replay_dirs:
            for path in replay_paths(directory):
                chunk = load_npz(path)
                if "action" not in chunk:
                    continue
                length = int(len(chunk["action"]))
                if length >= self.batch_length:
                    self.paths.append(path)
                    self.lengths.append(length - self.batch_length + 1)
        if not self.paths:
            raise FileNotFoundError(f"No usable replay chunks found in {self.replay_dirs}")
        self.cumulative = np.cumsum(self.lengths)

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        rows = [self._get(int(rng.integers(0, len(self)))) for _ in range(int(batch_size))]
        keys = sorted(set.intersection(*(set(row.keys()) for row in rows)))
        return {key: np.stack([row[key] for row in rows], axis=0) for key in keys}

    def _get(self, index: int) -> Dict[str, np.ndarray]:
        chunk_idx = int(np.searchsorted(self.cumulative, index, side="right"))
        prev = 0 if chunk_idx == 0 else int(self.cumulative[chunk_idx - 1])
        start = index - prev
        end = start + self.batch_length
        chunk = load_npz(self.paths[chunk_idx])
        keys = [key for key, value in chunk.items() if np.asarray(value).ndim > 0 and len(value) >= end]
        if self.allowed_keys is not None:
            keys = [key for key in keys if key in self.allowed_keys or key == "executed_control"]
        row = {key: np.asarray(chunk[key][start:end]) for key in keys}
        if "executed_control" in row:
            row["action"] = row["executed_control"]
            if self.allowed_keys is not None and "executed_control" not in self.allowed_keys:
                row.pop("executed_control", None)
        row.setdefault("is_first", np.zeros((self.batch_length,), dtype=bool))
        row.setdefault("is_last", np.zeros((self.batch_length,), dtype=bool))
        row.setdefault("is_terminal", np.zeros((self.batch_length,), dtype=bool))
        row.setdefault("reward", np.zeros((self.batch_length,), dtype=np.float32))
        return row


class MixedReplaySampler:
    def __init__(
        self,
        offline_dir: str | Path,
        online_dir: str | Path,
        batch_length: int,
        allowed_keys: set[str],
        offline_ratio: float,
    ):
        self.offline = ReplaySequenceDataset([offline_dir], batch_length, allowed_keys)
        self.online = None
        if replay_paths(online_dir):
            self.online = ReplaySequenceDataset([online_dir], batch_length, allowed_keys)
        self.offline_ratio = float(np.clip(offline_ratio, 0.0, 1.0))

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if self.online is None:
            return self.offline.sample(batch_size, rng)
        offline_count = int(round(int(batch_size) * self.offline_ratio))
        online_count = int(batch_size) - offline_count
        if online_count <= 0:
            return self.offline.sample(batch_size, rng)
        if offline_count <= 0:
            return self.online.sample(batch_size, rng)
        a = self.offline.sample(offline_count, rng)
        b = self.online.sample(online_count, rng)
        keys = sorted(set(a.keys()) & set(b.keys()))
        return {key: np.concatenate([a[key], b[key]], axis=0) for key in keys}


def scalarize(metrics: Dict[str, object]) -> Dict[str, float]:
    out = {}
    for key, value in metrics.items():
        arr = np.asarray(jax.device_get(value))
        if arr.shape == ():
            out[key] = float(arr)
    return out


def deep_copy_to_device(tree, device):
    with transfer_guard_allow():
        host = jax.device_get(tree)
        return jax.tree_util.tree_map(lambda x: jax.device_put(np.array(x), device), host)


def allow_jax_transfers() -> None:
    """Dreamer sets transfer_guard=disallow; this script explicitly feeds numpy replay/checkpoints."""
    for option in (
        "jax_transfer_guard",
        "jax_transfer_guard_host_to_device",
        "jax_transfer_guard_device_to_host",
        "jax_transfer_guard_device_to_device",
    ):
        try:
            jax.config.update(option, "allow")
        except Exception:
            pass


def transfer_guard_allow():
    guard = getattr(jax, "transfer_guard", None)
    return guard("allow") if guard is not None else nullcontext()


def to_device_tree(tree, device):
    with transfer_guard_allow():
        host = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, device), host)


def replace_selector_params(params, selector_params):
    params = dict(params)
    params["selector_head"] = selector_params
    return params


def flatten_state_feat(state):
    deter = state["deter"].reshape((state["deter"].shape[0], -1))
    stoch = state["stoch"].reshape((state["stoch"].shape[0], -1))
    return jnp.concatenate([deter, stoch], axis=-1)


def planner_modes_logits(params, config, latent, timestep: int):
    noisy_xy, timesteps = deterministic_anchors(params, config, latent.shape[0], timestep=timestep)
    poses_reg, poses_cls = _decode(params, config, latent, noisy_xy, timesteps, training=False)
    return poses_reg, poses_cls


def poses_xy_meters(poses_reg, args: argparse.Namespace):
    xy = poses_reg[..., :2]
    if args.planner_output_unit == "normalized":
        xy = xy * float(args.waypoint_scale)
    return xy


def override_anchor(config, params, anchor_path: Optional[str]):
    if not anchor_path:
        return config, params
    anchors = np.load(anchor_path).astype(np.float32)
    expected = (int(config.num_modes), int(config.num_poses), 2)
    if tuple(anchors.shape) != expected:
        raise ValueError(f"Expected anchor override {expected}, got {anchors.shape}")
    params = dict(params)
    params["plan_anchor"] = jnp.asarray(anchors, dtype=jnp.float32)
    config.plan_anchor_path = str(anchor_path)
    return config, params


def obs_scalar(obs: Dict[str, np.ndarray], key: str, default: float = 0.0) -> float:
    return closed_loop.obs_scalar(obs, key, default)


def ego_xy_to_world(xy: np.ndarray, obs: Dict[str, np.ndarray]) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    ex = obs_scalar(obs, "ego_x", 0.0)
    ey = obs_scalar(obs, "ego_y", 0.0)
    yaw = obs_scalar(obs, "ego_yaw", 0.0)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    wx = ex + cos_y * xy[..., 0] - sin_y * xy[..., 1]
    wy = ey + sin_y * xy[..., 0] + cos_y * xy[..., 1]
    return np.stack([wx, wy], axis=-1).astype(np.float32)


def world_xy_to_ego(xy: np.ndarray, obs: Dict[str, np.ndarray]) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    ex = obs_scalar(obs, "ego_x", 0.0)
    ey = obs_scalar(obs, "ego_y", 0.0)
    yaw = obs_scalar(obs, "ego_yaw", 0.0)
    dx = xy[..., 0] - ex
    dy = xy[..., 1] - ey
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    lx = cos_y * dx + sin_y * dy
    ly = -sin_y * dx + cos_y * dy
    return np.stack([lx, ly], axis=-1).astype(np.float32)


def xy_to_traj(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    if len(xy) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if len(xy) == 1:
        heading = np.zeros((1,), dtype=np.float32)
    else:
        d = np.diff(xy, axis=0)
        heading = np.arctan2(d[:, 1], d[:, 0]).astype(np.float32)
        heading = np.concatenate([heading, heading[-1:]], axis=0)
    return np.concatenate([xy, heading[:, None]], axis=-1).astype(np.float32)


def physical_to_normalized_action(acc: float, steer: float, args: argparse.Namespace) -> np.ndarray:
    return closed_loop.physical_to_normalized_action(acc, steer, args)


class EpisodeReplayWriter:
    def __init__(self, replay_dir: str | Path):
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[Dict[str, np.ndarray]] = []

    def append(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        planner_xy: Optional[np.ndarray],
        selected_mode: int,
        selector_logits: Optional[np.ndarray],
        plan_age: int,
    ) -> None:
        row = {key: np.asarray(value).copy() for key, value in obs.items()}
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        row["action"] = action.copy()
        row["executed_control"] = action.copy()
        if planner_xy is None:
            planner_xy = np.zeros((8, 2), dtype=np.float32)
        planner_xy = np.asarray(planner_xy, dtype=np.float32).reshape(8, 2)
        row["planner_waypoints8"] = planner_xy.reshape(-1).astype(np.float32)
        row["selected_mode"] = np.asarray(selected_mode, dtype=np.int32)
        if selector_logits is None:
            selector_logits = np.zeros((1,), dtype=np.float32)
        row["selector_logits"] = np.asarray(selector_logits, dtype=np.float32)
        row["plan_age"] = np.asarray(plan_age, dtype=np.int32)
        self.rows.append(row)

    def save(self, outer_idx: int, episode_idx: int) -> Optional[Path]:
        if not self.rows:
            return None
        keys = sorted(set.intersection(*(set(row.keys()) for row in self.rows)))
        arrays = {key: np.stack([row[key] for row in self.rows], axis=0) for key in keys}
        stamp = _datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self.replay_dir / f"online_o{outer_idx:04d}_e{episode_idx:03d}_{stamp}.npz"
        np.savez_compressed(path, **arrays)
        return path


class StableOnlineTrainer:
    def __init__(self, args: argparse.Namespace, extra: list[str]):
        self.args = args
        self.extra = ensure_control_action_extra(extra)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.online_replay_dir = Path(args.online_replay_dir or (self.output_dir / "online_replay"))
        self.online_replay_dir.mkdir(parents=True, exist_ok=True)
        self.rng_np = np.random.default_rng(int(args.seed))

        offline_rssm.import_runtime()
        rssm_args = argparse.Namespace(
            task=args.task,
            replay_dir=args.offline_replay_dir,
            logdir=str(self.output_dir / "dreamer_runtime"),
            batch_length=args.batch_length,
            batch_size=args.batch_size,
            replay_size=int(args.replay_size),
            jax_platform=args.jax_platform,
            from_checkpoint="",
        )
        self.raw_env, self.config = offline_rssm.build_config(rssm_args, self.extra)
        env = offline_rssm.from_gym.FromGym(self.raw_env)
        self.env = offline_rssm.wrap_env(env, self.config.dreamerv3)

        self.obs_space = self.env.obs_space
        self.act_space = self.env.act_space
        self.action_shape = tuple(self.act_space["action"].shape)
        if self.action_shape != (2,):
            raise ValueError(f"Expected normalized [acc, steer] action shape (2,), got {self.action_shape}")

        self.step = offline_rssm.embodied.Counter()
        self.agent = offline_rssm.dreamerv3.Agent(self.obs_space, self.act_space, self.step, self.config.dreamerv3)
        allow_jax_transfers()
        if len(self.agent.train_devices) != 1:
            raise ValueError("Set dreamerv3.jax.train_devices=[0].")
        self.device = self.agent.train_devices[0]
        checkpoint = offline_rssm.embodied.Checkpoint(log=False, parallel=False)
        checkpoint.agent = self.agent
        checkpoint.load(args.rssm_checkpoint, keys=["agent"])

        rssm_online_path = self.output_dir / "rssm_online" / "checkpoint.ckpt"
        rssm_online_path.parent.mkdir(parents=True, exist_ok=True)
        self.rssm_online_ckpt = offline_rssm.embodied.Checkpoint(
            rssm_online_path,
            log=False,
            parallel=False,
        )
        self.rssm_online_ckpt.agent = self.agent

        self.planner_config, planner_params, _, planner_epoch = load_checkpoint(args.planner_checkpoint)
        self.planner_config, planner_params = override_anchor(self.planner_config, planner_params, args.anchor_path)
        if not np.isclose(float(self.planner_config.waypoint_scale), float(args.waypoint_scale)):
            print(
                "[stable_online] WARNING: checkpoint waypoint_scale="
                f"{float(self.planner_config.waypoint_scale):.6g} but --waypoint_scale={float(args.waypoint_scale):.6g}. "
                "Using the checkpoint scale inside the planner and --waypoint_scale only for optional normalized output conversion.",
                flush=True,
            )
        self.planner_config.waypoint_dt = float(args.waypoint_dt)
        self.planner_config.wheelbase = float(args.wheelbase)
        self.planner_config.max_steer_rad = float(args.max_steer_rad)
        self.planner_config.steer_sign = float(args.steer_sign)
        self.planner_config.steer_gain = float(args.steer_gain)
        self.planner_config.lookahead_min = float(args.lookahead_min)
        self.planner_config.lookahead_max = float(args.lookahead_max)
        self.planner_config.lookahead_gain = float(args.lookahead_gain)
        self.planner_config.speed_kp = float(args.speed_kp)
        self.planner_config.ctrl_acc_min = float(args.env_acc_min)
        self.planner_config.ctrl_acc_max = float(args.env_acc_max)
        self.planner_config.ctrl_target_speed_max = float(args.target_speed_max)
        self.planner_params = to_device_tree(planner_params, self.device)
        if "selector_head" not in self.planner_params:
            raise KeyError("Planner checkpoint does not contain params['selector_head'].")
        self.selector_params = self.planner_params["selector_head"]
        self.ref_planner_params = jax.tree_util.tree_map(jax.lax.stop_gradient, self.planner_params)
        self.planner_epoch = int(planner_epoch)

        self.selector_opt = optax.chain(
            optax.clip_by_global_norm(float(args.grad_clip)),
            optax.adamw(float(args.selector_lr), weight_decay=float(args.selector_weight_decay)),
        )
        with transfer_guard_allow():
            self.selector_state = self.selector_opt.init(self.selector_params)
            self.target_varibs = deep_copy_to_device(self.agent.varibs, self.device)
        self.wm_train_state = None

        self._wm_train_fn = self._make_wm_train_fn()
        self._rssm_init_fn = self._make_rssm_init_fn()
        self._rssm_obs_step_fn = self._make_rssm_obs_step_fn()
        self._start_fn = self._make_start_fn()
        self._returns_fn = self._make_returns_fn()
        self._selector_step = jax.jit(self._selector_step_impl)
        self._plan_fn = jax.jit(self._plan_impl)

        self.allowed_data_keys = set(self.obs_space.keys()) | {"action"}
        self.history: list[dict] = []

    def _make_wm_train_fn(self):
        def train_world_model(data, state):
            data = self.agent.agent.preprocess(data)
            state, _, metrics = self.agent.agent.wm.train(data, state)
            metrics["stable_online_wm_loss"] = metrics["model_loss_mean"]
            return state, metrics

        return offline_rssm.nj.jit(offline_rssm.nj.pure(train_world_model), device=self.device)

    def _make_rssm_init_fn(self):
        def init_state():
            return self.agent.agent.wm.initial(1)

        return offline_rssm.nj.jit(offline_rssm.nj.pure(init_state), device=self.device)

    def _make_rssm_obs_step_fn(self):
        def obs_step(data, prev_latent, prev_action):
            data = self.agent.agent.preprocess(data)
            embed = self.agent.agent.wm.encoder(data)
            latent, _ = self.agent.agent.wm.rssm.obs_step(prev_latent, prev_action, embed, data["is_first"])
            return latent

        return offline_rssm.nj.jit(offline_rssm.nj.pure(obs_step), device=self.device)

    def _make_start_fn(self):
        def start(data):
            data = self.agent.agent.preprocess(data)
            embed = self.agent.agent.wm.encoder(data)
            prev_latent, prev_action = self.agent.agent.wm.initial(data["action"].shape[0])
            prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], axis=1)
            post, _ = self.agent.agent.wm.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
            state = {key: value[:, -1] for key, value in post.items()}
            if "ego_speed" in data:
                speed = data["ego_speed"][:, -1].reshape((data["action"].shape[0], -1))[:, :1]
            else:
                speed = jnp.zeros((data["action"].shape[0], 1), jnp.float32)
            return state, speed, flatten_state_feat(state)

        return offline_rssm.nj.jit(offline_rssm.nj.pure(start), device=self.device)

    def _make_returns_fn(self):
        def mode_returns(state, speed, modes_xy):
            batch, modes = modes_xy.shape[:2]
            modes_xy = apply_plan_sign(modes_xy, self.args.plan_x_sign, self.args.plan_y_sign)
            flat_modes = modes_xy.reshape((batch * modes, modes_xy.shape[-2], 2))
            cur_state = jax.tree_util.tree_map(
                lambda value: jnp.repeat(value[:, None], modes, axis=1).reshape(
                    (batch * modes,) + value.shape[1:]
                ),
                state,
            )
            cur_speed = jnp.repeat(speed[:, None], modes, axis=1).reshape((batch * modes, 1))
            pose_x = jnp.zeros((batch * modes,), jnp.float32)
            pose_y = jnp.zeros((batch * modes,), jnp.float32)
            pose_yaw = jnp.zeros((batch * modes,), jnp.float32)
            prev_steer = jnp.zeros((batch * modes,), jnp.float32)
            pid_integral = jnp.zeros((batch * modes,), jnp.float32)
            pid_prev_error = jnp.zeros((batch * modes,), jnp.float32)
            pid_ready = jnp.zeros((batch * modes,), bool)
            returns = jnp.zeros((batch * modes,), jnp.float32)
            discounts = jnp.ones((batch * modes,), jnp.float32)
            point_idx = jnp.arange(flat_modes.shape[1], dtype=jnp.int32)
            seg_idx = jnp.arange(max(flat_modes.shape[1] - 1, 1), dtype=jnp.int32)

            def to_current_ego(path_xy, x, y, yaw):
                dx = path_xy[..., 0] - x[:, None]
                dy = path_xy[..., 1] - y[:, None]
                cos_y = jnp.cos(yaw)[:, None]
                sin_y = jnp.sin(yaw)[:, None]
                lx = cos_y * dx + sin_y * dy
                ly = -sin_y * dx + cos_y * dy
                return jnp.stack([lx, ly], axis=-1)

            def eval_like_controller(rel_xy, ego_speed, last_steer, integral, prev_error, ready):
                finite = jnp.all(jnp.isfinite(rel_xy), axis=(1, 2))
                lateral_ok = jnp.max(jnp.abs(rel_xy[..., 1]), axis=1) <= float(self.args.max_abs_y)
                step_dist = jnp.linalg.norm(jnp.diff(rel_xy, axis=1), axis=-1)
                step_ok = jnp.max(step_dist, axis=1) <= float(self.args.max_step_distance)
                forward_mask = rel_xy[..., 0] >= float(self.args.min_forward_x)
                forward_count = jnp.sum(forward_mask, axis=1)
                valid = (
                    finite
                    & lateral_ok
                    & step_ok
                    & (forward_count >= int(self.args.min_valid_points))
                )

                # Match eval_close_loop's forward-point filtering while keeping
                # static shapes for JIT: move valid forward points to the front,
                # preserving their original order.
                sort_key = jnp.where(forward_mask, point_idx[None], point_idx[None] + flat_modes.shape[1])
                order = jnp.argsort(sort_key, axis=1)
                forward_xy = jnp.take_along_axis(rel_xy, order[..., None], axis=1)

                first4 = forward_xy[:, :4]
                first4_seg = jnp.linalg.norm(jnp.diff(first4, axis=1), axis=-1)
                nseg = jnp.minimum(jnp.maximum(forward_count - 1, 0), 3)
                s0 = first4_seg[:, 0]
                s1 = first4_seg[:, 1]
                s2 = first4_seg[:, 2]
                median1 = s0
                median2 = 0.5 * (s0 + s1)
                median3 = s0 + s1 + s2 - jnp.minimum(jnp.minimum(s0, s1), s2) - jnp.maximum(jnp.maximum(s0, s1), s2)
                median_seg = jnp.where(nseg <= 1, median1, jnp.where(nseg == 2, median2, median3))
                speed_est = median_seg / max(float(self.args.waypoint_dt), 1e-6)
                speed_est = jnp.where(
                    jnp.isfinite(speed_est) & (speed_est >= float(self.args.stop_speed)),
                    speed_est,
                    float(self.args.fallback_speed),
                )
                target_speed = jnp.clip(
                    speed_est,
                    float(self.args.target_speed_min),
                    float(self.args.target_speed_max),
                )
                target_speed = jnp.where(valid, target_speed, 0.0)

                forward_seg = jnp.linalg.norm(jnp.diff(forward_xy, axis=1), axis=-1)
                seg_mask = seg_idx[None] < jnp.maximum(forward_count[:, None] - 1, 0)
                forward_seg = jnp.where(seg_mask, forward_seg, 0.0)
                cumulative = jnp.concatenate(
                    [jnp.zeros((forward_xy.shape[0], 1), jnp.float32), jnp.cumsum(forward_seg, axis=1)],
                    axis=1,
                )
                valid_points = point_idx[None] < forward_count[:, None]
                ld = jnp.clip(
                    float(self.args.lookahead_min) + float(self.args.lookahead_gain) * jnp.maximum(ego_speed, 0.0),
                    float(self.args.lookahead_min),
                    float(self.args.lookahead_max),
                )
                lookaheads = jnp.stack([0.75 * ld, ld, 1.35 * ld], axis=1)
                errors = jnp.abs(cumulative[:, None, :] - lookaheads[:, :, None])
                errors = jnp.where(valid_points[:, None, :], errors, 1e6)
                target_idx = jnp.argmin(errors, axis=-1)
                gather_source = jnp.repeat(forward_xy[:, None, :, :], 3, axis=1)
                targets = jnp.take_along_axis(gather_source, target_idx[..., None, None], axis=2).squeeze(2)
                mix = jnp.asarray([0.25, 0.50, 0.25], dtype=jnp.float32)
                target = (targets * mix[None, :, None]).sum(axis=1)

                distance = jnp.linalg.norm(target, axis=-1)
                dist2 = jnp.maximum(jnp.square(target[..., 0]) + jnp.square(target[..., 1]), 1e-6)
                curvature = 2.0 * target[..., 1] / dist2
                steer_angle = jnp.arctan(float(self.args.wheelbase) * curvature)
                steer_raw = (
                    float(self.args.steer_sign)
                    * float(self.args.steer_gain)
                    * steer_angle
                    / max(float(self.args.max_steer_rad), 1e-6)
                )
                steer_limited = jnp.clip(
                    steer_raw,
                    last_steer - float(self.args.steer_rate_limit),
                    last_steer + float(self.args.steer_rate_limit),
                )
                steer_limited = jnp.clip(steer_limited, -1.0, 1.0)
                steer_valid = jnp.where(distance < 1e-3, 0.0, steer_limited)
                steer = jnp.where(valid, steer_valid, jnp.clip(last_steer, -0.2, 0.2))

                error = target_speed - ego_speed
                next_integral_valid = jnp.clip(integral + error * float(self.args.dt), -10.0, 10.0)
                derivative = jnp.where(
                    ready,
                    (error - prev_error) / max(float(self.args.dt), 1e-6),
                    0.0,
                )
                acc_valid = jnp.clip(
                    float(self.args.speed_kp) * error
                    + float(self.args.speed_ki) * next_integral_valid
                    + float(self.args.speed_kd) * derivative,
                    float(self.args.acc_min),
                    float(self.args.acc_max),
                )
                acc_invalid = jnp.where(
                    ego_speed > float(self.args.stop_speed),
                    float(self.args.acc_min),
                    -1.0,
                )
                acc = jnp.where(valid, acc_valid, acc_invalid)
                next_integral = jnp.where(valid, next_integral_valid, integral)
                next_prev_error = jnp.where(valid, error, prev_error)
                next_ready = ready | valid
                acc_norm = 2.0 * (acc - float(self.args.env_acc_min)) / max(
                    float(self.args.env_acc_max) - float(self.args.env_acc_min),
                    1e-6,
                ) - 1.0
                action = jnp.stack([jnp.clip(acc_norm, -1.0, 1.0), steer], axis=-1)
                return action, acc, steer, next_integral, next_prev_error, next_ready

            for _ in range(int(self.args.score_horizon_steps)):
                rel_xy = to_current_ego(flat_modes, pose_x, pose_y, pose_yaw)
                speed_vec = cur_speed.reshape((-1,))
                action, acc, prev_steer, pid_integral, pid_prev_error, pid_ready = eval_like_controller(
                    rel_xy,
                    speed_vec,
                    prev_steer,
                    pid_integral,
                    pid_prev_error,
                    pid_ready,
                )
                cur_state = self.agent.agent.wm.rssm.img_step(cur_state, action)
                reward = self.agent.agent.wm.heads["reward"](cur_state).mode().reshape((-1,))
                cont = self.agent.agent.wm.heads["cont"](cur_state).mean().reshape((-1,))
                returns = returns + discounts * reward
                discounts = discounts * float(self.args.discount) * cont

                next_speed = jnp.maximum(speed_vec + acc * float(self.args.dt), 0.0)
                steer_angle = action[:, 1] * float(self.args.max_steer_rad)
                yaw_rate = speed_vec / max(float(self.args.wheelbase), 1e-6) * jnp.tan(steer_angle)
                pose_x = pose_x + speed_vec * jnp.cos(pose_yaw) * float(self.args.dt)
                pose_y = pose_y + speed_vec * jnp.sin(pose_yaw) * float(self.args.dt)
                pose_yaw = pose_yaw + yaw_rate * float(self.args.dt)
                cur_speed = next_speed[:, None]

            return returns.reshape((batch, modes))

        return offline_rssm.nj.jit(offline_rssm.nj.pure(mode_returns), device=self.device)

    def _plan_impl(self, params, condition):
        poses_reg, poses_cls = planner_modes_logits(
            params,
            self.planner_config,
            condition.reshape((1, -1)),
            int(self.args.eval_timestep),
        )
        return poses_reg[0], poses_cls[0]

    def _selector_step_impl(self, selector_params, opt_state, params, ref_params, feat, returns):
        args = self.args

        def objective(current_selector_params):
            current_params = replace_selector_params(params, current_selector_params)
            _, logits = planner_modes_logits(current_params, self.planner_config, feat, int(args.eval_timestep))
            _, ref_logits = planner_modes_logits(ref_params, self.planner_config, feat, int(args.eval_timestep))
            temperature = max(float(args.policy_temperature), 1e-6)
            logp = jax.nn.log_softmax(logits / temperature, axis=-1)
            prob = jax.nn.softmax(logits / temperature, axis=-1)
            ref_logp = jax.nn.log_softmax(jax.lax.stop_gradient(ref_logits) / temperature, axis=-1)
            ref_prob = jax.nn.softmax(jax.lax.stop_gradient(ref_logits) / temperature, axis=-1)

            adv = returns - returns.mean(axis=-1, keepdims=True)
            adv = adv / jnp.maximum(returns.std(axis=-1, keepdims=True), float(args.adv_eps))
            adv = jnp.clip(adv, -float(args.adv_clip), float(args.adv_clip))
            adv = jax.lax.stop_gradient(adv)
            ranking_loss = -(adv * logp).sum(axis=-1).mean()
            kl_loss = (ref_prob * (ref_logp - logp)).sum(axis=-1).mean()
            entropy = -(prob * logp).sum(axis=-1).mean()
            loss = (
                float(args.ranking_weight) * ranking_loss
                + float(args.kl_weight) * kl_loss
                - float(args.entropy_weight) * entropy
            )
            selected = jnp.argmax(logits, axis=-1)
            oracle = jnp.argmax(returns, axis=-1)
            batch_idx = jnp.arange(returns.shape[0])
            metrics = {
                "selector_loss": loss,
                "ranking_loss": ranking_loss,
                "kl_loss": kl_loss,
                "entropy": entropy,
                "return_mean": returns.mean(),
                "return_std": returns.std(),
                "selected_return": returns[batch_idx, selected].mean(),
                "oracle_return": returns.max(axis=-1).mean(),
                "rank_acc": (selected == oracle).astype(jnp.float32).mean(),
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(selector_params)
        updates, opt_state = self.selector_opt.update(grads, opt_state, selector_params)
        selector_params = optax.apply_updates(selector_params, updates)
        metrics = dict(metrics)
        metrics["selector_loss"] = loss
        return selector_params, opt_state, metrics

    def init_rssm_episode(self):
        rng = self.agent._next_rngs(self.agent.train_devices)
        (prev_latent, prev_action), self.agent.varibs = self._rssm_init_fn(self.agent.varibs, rng)
        return prev_latent, prev_action

    def encode_obs(self, obs, prev_action, prev_latent):
        data = {key: np.asarray(value)[None] for key, value in obs.items()}
        data = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.device), data)
        prev_action = jax.device_put(np.asarray(prev_action, dtype=np.float32).reshape(1, -1), self.device)
        rng = self.agent._next_rngs(self.agent.train_devices)
        latent, self.agent.varibs = self._rssm_obs_step_fn(
            self.agent.varibs,
            rng,
            data,
            prev_latent,
            prev_action,
        )
        feat = jax.device_get(flatten_state_feat(latent))[0]
        return latent, feat

    def plan_np(self, condition: np.ndarray):
        poses_reg, logits = self._plan_fn(self.planner_params, jnp.asarray(condition, dtype=jnp.float32))
        poses_reg = np.asarray(jax.device_get(poses_reg), dtype=np.float32)
        logits = np.asarray(jax.device_get(logits), dtype=np.float32)
        selected_idx = int(np.argmax(logits))
        selected = poses_reg[selected_idx].copy()
        selected[:, :2] = poses_xy_meters(selected, self.args)
        return selected, selected_idx, logits, poses_reg

    def collect(self, outer_idx: int) -> dict:
        if self.args.no_collect:
            return {"episodes": 0, "steps": 0, "mean_return": 0.0}
        summaries = []
        total_steps = 0
        plan_interval = max(int(self.args.plan_interval_steps), 1)

        for episode in range(1, int(self.args.collect_episodes) + 1):
            writer = EpisodeReplayWriter(self.online_replay_dir)
            obs, _ = self.env.step({"reset": True, "action": np.zeros(self.action_shape, dtype=np.float32)})
            prev_latent, _ = self.init_rssm_episode()
            previous_action = np.zeros(self.action_shape, dtype=np.float32)
            previous_steer = 0.0
            pid = closed_loop.PID(
                self.args.speed_kp,
                self.args.speed_ki,
                self.args.speed_kd,
                self.args.dt,
                (self.args.acc_min, self.args.acc_max),
            )
            cached_world_xy = None
            cached_logits = None
            cached_selected = -1
            cached_plan_age = 0
            episode_return = 0.0
            episode_steps = 0

            for step_idx in range(int(self.args.max_steps)):
                prev_latent, condition = self.encode_obs(obs, previous_action, prev_latent)
                need_plan = cached_world_xy is None or (step_idx % plan_interval == 0)
                if need_plan:
                    selected, cached_selected, cached_logits, _ = self.plan_np(condition)
                    selected = closed_loop.apply_plan_signs(
                        selected,
                        self.args.plan_x_sign,
                        self.args.plan_y_sign,
                    )
                    cached_world_xy = ego_xy_to_world(selected[:, :2], obs)
                    cached_plan_age = 0

                current_xy = world_xy_to_ego(cached_world_xy, obs)
                traj = xy_to_traj(current_xy)
                valid, reason, forward_xy = closed_loop.validate_trajectory(traj, self.args)
                current_speed = obs_scalar(obs, "ego_speed", 0.0)

                if valid:
                    target_speed = closed_loop.estimate_speed(forward_xy, self.args)
                    steer, _ = closed_loop.multi_point_pure_pursuit(
                        forward_xy,
                        current_speed,
                        previous_steer,
                        self.args,
                    )
                    acc = pid.step(target_speed, current_speed)
                else:
                    target_speed = 0.0
                    steer = float(np.clip(previous_steer, -0.2, 0.2))
                    acc = self.args.acc_min if current_speed > self.args.stop_speed else -1.0
                    print(
                        f"[collect] outer={outer_idx} episode={episode} step={step_idx} invalid_plan={reason}",
                        flush=True,
                    )

                action = physical_to_normalized_action(acc, steer, self.args)
                writer.append(obs, action, current_xy, cached_selected, cached_logits, cached_plan_age)
                next_obs, info = self.env.step({"reset": False, "action": action})
                reward = float(np.asarray(next_obs.get("reward", 0.0)))
                episode_return += reward
                episode_steps += 1
                total_steps += 1

                is_last = bool(np.asarray(next_obs.get("is_last", False)))
                is_terminal = bool(np.asarray(next_obs.get("is_terminal", False)))
                previous_action = action
                previous_steer = steer
                obs = next_obs
                cached_plan_age += 1

                if is_last or is_terminal:
                    writer.append(
                        obs,
                        np.zeros(self.action_shape, dtype=np.float32),
                        world_xy_to_ego(cached_world_xy, obs) if cached_world_xy is not None else None,
                        cached_selected,
                        cached_logits,
                        cached_plan_age,
                    )
                    break

            path = writer.save(outer_idx, episode)
            summary = {
                "episode": episode,
                "steps": episode_steps,
                "return": float(episode_return),
                "path": str(path) if path else None,
            }
            for key in ("destination_reached", "is_success", "out_of_lane", "time_exceeded", "is_collision"):
                if key in obs:
                    summary[key] = bool(np.asarray(obs[key]).reshape(-1)[0])
            summaries.append(summary)
            print(
                f"[collect] outer={outer_idx} episode={episode:03d} "
                f"steps={episode_steps} return={episode_return:.3f} replay={path}",
                flush=True,
            )

        mean_return = float(np.mean([item["return"] for item in summaries])) if summaries else 0.0
        return {"episodes": summaries, "steps": int(total_steps), "mean_return": mean_return}

    def make_sampler(self) -> MixedReplaySampler:
        return MixedReplaySampler(
            self.args.offline_replay_dir,
            self.online_replay_dir,
            self.args.batch_length,
            self.allowed_data_keys,
            self.args.offline_ratio,
        )

    def update_rssm(self, outer_idx: int) -> dict:
        sampler = self.make_sampler()
        metrics_last = {}
        for update in range(1, int(self.args.wm_updates) + 1):
            batch = sampler.sample(int(self.args.batch_size), self.rng_np)
            batch = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.device), batch)
            if self.wm_train_state is None:
                rng = self.agent._next_rngs(self.agent.train_devices)
                self.wm_train_state, self.agent.varibs = self.agent._init_train(
                    self.agent.varibs,
                    rng,
                    batch["is_first"],
                )
            rng = self.agent._next_rngs(self.agent.train_devices)
            (self.wm_train_state, metrics), self.agent.varibs = self._wm_train_fn(
                self.agent.varibs,
                rng,
                batch,
                self.wm_train_state,
            )
            self.step.increment(int(self.args.batch_size * self.args.batch_length))
            metrics_last = scalarize(metrics)
            if update == 1 or update % int(self.args.log_every) == 0:
                loss = metrics_last.get("stable_online_wm_loss", float("nan"))
                print(f"[wm] outer={outer_idx} update={update} loss={loss:.6f}", flush=True)

        self.rssm_online_ckpt.save()
        self.target_varibs = deep_copy_to_device(self.agent.varibs, self.device)
        return metrics_last

    def start_from_target(self, batch):
        data = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.device), batch)
        rng = self.agent._next_rngs(self.agent.train_devices)
        (state, speed, feat), _ = self._start_fn(self.target_varibs, rng, data)
        return state, speed, feat

    def returns_from_target(self, state, speed, modes_xy):
        rng = self.agent._next_rngs(self.agent.train_devices)
        returns, _ = self._returns_fn(self.target_varibs, rng, state, speed, modes_xy)
        return returns

    def update_selector(self, outer_idx: int) -> dict:
        sampler = self.make_sampler()
        metrics_last = {}
        for update in range(1, int(self.args.selector_updates) + 1):
            batch = sampler.sample(int(self.args.batch_size), self.rng_np)
            batch = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.device), batch)
            state, speed, feat = self.start_from_target(batch)
            poses_reg, _ = planner_modes_logits(
                self.planner_params,
                self.planner_config,
                feat,
                int(self.args.eval_timestep),
            )
            modes_xy = jax.lax.stop_gradient(poses_xy_meters(poses_reg, self.args))
            returns = jax.lax.stop_gradient(self.returns_from_target(state, speed, modes_xy))
            self.selector_params, self.selector_state, metrics = self._selector_step(
                self.selector_params,
                self.selector_state,
                self.planner_params,
                self.ref_planner_params,
                feat,
                returns,
            )
            self.planner_params = replace_selector_params(self.planner_params, self.selector_params)
            metrics_last = scalarize(metrics)
            if update == 1 or update % int(self.args.log_every) == 0:
                print(
                    "[selector] "
                    f"outer={outer_idx} update={update} "
                    + " ".join(f"{k}={v:.5f}" for k, v in metrics_last.items()),
                    flush=True,
                )
        return metrics_last

    def save_planner(self, outer_idx: int) -> None:
        save_checkpoint(
            self.output_dir / "planner_selector_online.pkl.gz",
            self.planner_config,
            self.planner_params,
            self.selector_state,
            outer_idx,
        )
        if int(self.args.save_every_outer) > 0 and outer_idx % int(self.args.save_every_outer) == 0:
            save_checkpoint(
                self.output_dir / f"planner_selector_online_outer_{outer_idx:04d}.pkl.gz",
                self.planner_config,
                self.planner_params,
                self.selector_state,
                outer_idx,
            )

    def run(self) -> None:
        run_config = {
            "args": vars(self.args),
            "extra": self.extra,
            "planner_checkpoint_epoch": self.planner_epoch,
            "trainable": "selector_head_only",
            "rssm_update": "online_world_model_mixed_offline_online_replay",
            "planner_update_rssm": "frozen_target_varibs_snapshot",
        }
        (self.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        for outer in range(1, int(self.args.outer_iterations) + 1):
            print(f"[stable_online] outer={outer} collect", flush=True)
            collect_metrics = self.collect(outer)
            print(f"[stable_online] outer={outer} update_rssm", flush=True)
            wm_metrics = self.update_rssm(outer)
            print(f"[stable_online] outer={outer} update_selector", flush=True)
            selector_metrics = self.update_selector(outer)
            self.save_planner(outer)
            row = {
                "outer": outer,
                "collect": collect_metrics,
                "wm": wm_metrics,
                "selector": selector_metrics,
            }
            self.history.append(row)
            (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            print(
                f"[stable_online] outer={outer} "
                f"mean_return={collect_metrics.get('mean_return', 0.0):.3f} "
                f"wm_loss={wm_metrics.get('stable_online_wm_loss', float('nan')):.6f} "
                f"selected_return={selector_metrics.get('selected_return', float('nan')):.5f}",
                flush=True,
            )

        self.rssm_online_ckpt.save()
        self.save_planner(int(self.args.outer_iterations))
        try:
            self.env.close()
        except Exception:
            pass
        print(f"[stable_online] Wrote outputs under {self.output_dir}", flush=True)


def main() -> None:
    args, extra = parse_args()
    if args.jax_platform:
        jax.config.update("jax_platform_name", args.jax_platform)
    trainer = StableOnlineTrainer(args, extra)
    trainer.run()


if __name__ == "__main__":
    main()
