"""Closed-loop CARLA eval for RSSM-conditioned DiffusionDrive planner.

The loop is:
  env observation -> online RSSM posterior latent -> planner trajectory
  -> safety checks -> PID speed control + multi-point pure pursuit -> env action.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import ruamel.yaml as yaml

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dreamerv3") not in sys.path:
    sys.path.insert(0, str(ROOT / "dreamerv3"))

import embodied
from collect_utils import wrap_env
from JAXRSSMJAXDiDr.data.gt_history_features import build_gt_history_features


warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

jax = None
jnp = None
nj = None


def import_jax_runtime() -> None:
    global jax, jnp, nj
    import jax as jax_module
    import jax.numpy as jnp_module
    from dreamerv3 import ninjax as ninjax_module

    jax = jax_module
    jnp = jnp_module
    nj = ninjax_module


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="carla_roundabout")
    parser.add_argument("--rssm_checkpoint", default=None, help="DreamerV3 checkpoint for RSSM route.")
    parser.add_argument("--planner_checkpoint", required=True)
    parser.add_argument("--anchor_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--jax_platform", default=None, choices=("cpu", "gpu", "tpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--waypoint_dt", type=float, default=0.5)
    parser.add_argument("--plan_x_sign", type=float, default=1.0, choices=(-1.0, 1.0), help="Flip planner output x before control/visualization.")
    parser.add_argument("--plan_y_sign", type=float, default=1.0, choices=(-1.0, 1.0), help="Flip planner output y before control/visualization.")
    parser.add_argument("--history_length", type=int, default=None)
    parser.add_argument("--no_neighbors", action="store_true")
    parser.add_argument("--no_align_neighbor_ids", action="store_true")

    parser.add_argument("--target_speed_min", type=float, default=0.0)
    parser.add_argument("--target_speed_max", type=float, default=8.0)
    parser.add_argument("--fallback_speed", type=float, default=1.5)
    parser.add_argument("--stop_speed", type=float, default=0.25)
    parser.add_argument("--speed_kp", type=float, default=1.0)
    parser.add_argument("--speed_ki", type=float, default=0.00)
    parser.add_argument("--speed_kd", type=float, default=0.00)
    parser.add_argument("--acc_min", type=float, default=-3.0)
    parser.add_argument("--acc_max", type=float, default=2.0)
    parser.add_argument("--env_acc_min", type=float, default=-3.0)
    parser.add_argument("--env_acc_max", type=float, default=3.0)

    parser.add_argument("--wheelbase", type=float, default=2.875)
    parser.add_argument("--max_steer_rad", type=float, default=0.65)
    parser.add_argument("--steer_gain", type=float, default=0.5, help="Scale pure-pursuit steer command before rate limiting.")
    parser.add_argument(
        "--steer_sign",
        type=float,
        default=-1.0,
        help="Pure-pursuit steer sign. CARLA ego-local y is right-positive, so -1 maps positive y to a right turn through env action.",
    )
    parser.add_argument("--steer_rate_limit", type=float, default=0.08)
    parser.add_argument("--lookahead_min", type=float, default=4.5)
    parser.add_argument("--lookahead_max", type=float, default=14.0)
    parser.add_argument("--lookahead_gain", type=float, default=1.0)
    parser.add_argument("--lateral_error_soft_stop", type=float, default=5.0)

    parser.add_argument("--min_valid_points", type=int, default=3)
    parser.add_argument("--min_forward_x", type=float, default=0.2)
    parser.add_argument("--max_abs_y", type=float, default=20.0)
    parser.add_argument("--max_step_distance", type=float, default=12.0)
    parser.add_argument("--safety_ttc", type=float, default=1.5)
    parser.add_argument("--safety_distance_min", type=float, default=4.0)
    parser.add_argument("--safety_lateral_width", type=float, default=2.2)
    parser.add_argument("--emergency_distance", type=float, default=2.0)
    parser.add_argument("--safety_filter", dest="safety_filter", action="store_true", default=True)
    parser.add_argument("--no_safety_filter", dest="safety_filter", action="store_false")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--live_plot", action="store_true", help="Show a realtime ego-frame window with top planner modes.")
    parser.add_argument("--plot_modes", type=int, default=6, help="Number of top-scored planner modes to draw.")
    parser.add_argument("--live_plot_pause", type=float, default=0.001, help="Matplotlib pause in seconds after each live update.")
    parser.add_argument(
        "--live_plot_yaw_offset_deg",
        type=float,
        default=90.0,
        help="Display-only yaw offset for live plot, in degrees. +90 matches CARLA top-down orientation in this setup.",
    )
    parser.add_argument("--carla_live_draw", action="store_true", help="Draw top planner modes directly in CARLA world debug view.")
    parser.add_argument("--carla_draw_lifetime", type=float, default=0.2, help="CARLA debug draw lifetime in seconds.")
    parser.add_argument("--carla_draw_z", type=float, default=0.35, help="Height offset for CARLA trajectory debug lines.")
    parser.add_argument("--carla_spectator_topdown", action="store_true", help="Move CARLA spectator to a top-down ego-follow view.")
    parser.add_argument("--carla_spectator_height", type=float, default=45.0, help="Top-down spectator height above ego.")
    return parser.parse_known_args()


def normalize_embodied_flags(extra: list[str]) -> list[str]:
    """Accept shell-friendly `[0]` tuple syntax for Embodied flags.

    Embodied's flag parser expects tuple values as `0` or `0,1`, while many
    project notes use YAML-style `[0]`.  Normalize both `--x=[0]` and
    `--x [0]` before handing the remaining flags to Embodied.
    """

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


def has_flag(extra: list[str], name: str) -> bool:
    prefix = f"--{name}"
    return any(item == prefix or item.startswith(prefix + "=") for item in extra)


def build_config(args: argparse.Namespace, extra: list[str]):
    import car_dreamer

    extra = normalize_embodied_flags(extra)
    if not has_flag(extra, "env.planner_target.use_waypoint_action"):
        extra = [*extra, "--env.planner_target.use_waypoint_action", "False"]
    yaml_path = ROOT / "dreamerv3" / "dreamerv3.yaml"
    model_configs = yaml.YAML(typ="safe").load(embodied.Path(str(yaml_path)).read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})
    raw_env, env_config = car_dreamer.create_task(args.task, extra)
    config = config.update(env_config)
    updates = {"dreamerv3.logdir": str(Path(args.output_dir) / "dreamerv3_runtime")}
    if args.jax_platform:
        updates["dreamerv3.jax.platform"] = args.jax_platform
    config = config.update(updates)
    config = embodied.Flags(config).parse(extra)
    return raw_env, config


def load_planner(checkpoint_path: str | Path, anchor_path: Optional[str], device: torch.device):
    from JAXRSSMJAXDiDr.models.rssm_didr_planner import RSSMDiDrConfig, RSSMDiffusionDrivePlanner

    payload = torch.load(checkpoint_path, map_location=device)
    config_dict = dict(payload["config"])
    if anchor_path:
        config_dict["plan_anchor_path"] = str(anchor_path)
    config = RSSMDiDrConfig(**config_dict)
    model = RSSMDiffusionDrivePlanner(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


def flatten_rssm_latent(post: Dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for key in ("deter", "stoch"):
        if key in post:
            value = np.asarray(post[key], dtype=np.float32)
            parts.append(value.reshape(value.shape[0], -1))
    if not parts:
        raise KeyError(f"Could not flatten RSSM posterior keys: {sorted(post.keys())}")
    return np.concatenate(parts, axis=-1).astype(np.float32)


class OnlineRSSMEncoder:
    def __init__(self, obs_space, act_space, checkpoint_path: str, config):
        import_jax_runtime()
        import dreamerv3

        step = embodied.Counter()
        self.agent = dreamerv3.Agent(obs_space, act_space, step, config.dreamerv3)
        if len(self.agent.train_devices) != 1:
            raise ValueError("Closed-loop RSSM eval expects one train device. Set dreamerv3.jax.train_devices=[0].")
        checkpoint = embodied.Checkpoint(log=False, parallel=False)
        checkpoint.agent = self.agent
        checkpoint.load(checkpoint_path, keys=["agent"])
        self.device = self.agent.train_devices[0]
        self.init_fn = self._make_init_fn()
        self.step_fn = self._make_step_fn()
        self.prev_latent = None
        self.prev_action = None

    def _make_init_fn(self):
        def init_state():
            return self.agent.agent.wm.initial(1)

        return nj.jit(nj.pure(init_state), device=self.device)

    def _make_step_fn(self):
        def rssm_step(data, prev_latent, prev_action):
            data = self.agent.agent.preprocess(data)
            embed = self.agent.agent.wm.encoder(data)
            latent, _ = self.agent.agent.wm.rssm.obs_step(prev_latent, prev_action, embed, data["is_first"])
            return latent

        return nj.jit(nj.pure(rssm_step), device=self.device)

    def reset(self, action_shape):
        rng = self.agent._next_rngs(self.agent.train_devices)
        (self.prev_latent, self.prev_action), self.agent.varibs = self.init_fn(self.agent.varibs, rng)

    def encode(self, obs: Dict[str, np.ndarray], prev_action: np.ndarray) -> np.ndarray:
        data = {}
        for key, value in obs.items():
            arr = np.asarray(value)
            data[key] = arr[None]
        prev_action = np.asarray(prev_action, dtype=np.float32).reshape(1, -1)
        data = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.device), data)
        prev_action_jax = jax.device_put(prev_action, self.device)
        rng = self.agent._next_rngs(self.agent.train_devices)
        latent, self.agent.varibs = self.step_fn(self.agent.varibs, rng, data, self.prev_latent, prev_action_jax)
        self.prev_latent = latent
        self.prev_action = prev_action_jax
        post = jax.device_get(latent)
        return flatten_rssm_latent({key: np.asarray(value) for key, value in post.items()})[0]


class OnlineGTHistory:
    def __init__(self, history_length: int, waypoint_scale: float, include_neighbors: bool, align_neighbor_ids: bool):
        self.history_length = int(history_length)
        self.waypoint_scale = float(waypoint_scale)
        self.include_neighbors = bool(include_neighbors)
        self.align_neighbor_ids = bool(align_neighbor_ids)
        self.buffers = defaultdict(list)

    def reset(self):
        self.buffers.clear()

    def encode(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> np.ndarray:
        keys = (
            "ego_x",
            "ego_y",
            "ego_yaw",
            "ego_speed",
            "ego_yawrate",
            "neighbor_vehicles_local",
            "neighbor_vehicles_world",
        )
        for key in keys:
            if key in obs:
                self.buffers[key].append(np.asarray(obs[key], dtype=np.float32).copy())
        self.buffers["action"].append(np.asarray(action, dtype=np.float32).reshape(-1)[:2].copy())
        length = len(self.buffers["action"])
        chunk = {
            "expert_waypoints8": np.zeros((length, 16), dtype=np.float32),
            "action": np.asarray(self.buffers["action"], dtype=np.float32),
        }
        for key, values in self.buffers.items():
            if key == "action":
                continue
            chunk[key] = np.asarray(values, dtype=np.float32)
        feats = build_gt_history_features(
            chunk,
            history_length=self.history_length,
            waypoint_scale=self.waypoint_scale,
            include_neighbors=self.include_neighbors,
            align_neighbor_ids=self.align_neighbor_ids,
        )
        return feats[-1]


class PID:
    def __init__(self, kp: float, ki: float, kd: float, dt: float, output_limits: tuple[float, float]):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.low, self.high = map(float, output_limits)
        self.integral = 0.0
        self.prev_error = 0.0
        self.ready = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.ready = False

    def step(self, target: float, current: float) -> float:
        error = float(target - current)
        self.integral = float(np.clip(self.integral + error * self.dt, -10.0, 10.0))
        derivative = 0.0 if not self.ready else (error - self.prev_error) / max(self.dt, 1e-6)
        self.prev_error = error
        self.ready = True
        return float(np.clip(self.kp * error + self.ki * self.integral + self.kd * derivative, self.low, self.high))


def obs_scalar(obs: Dict[str, np.ndarray], key: str, default: float = 0.0) -> float:
    if key not in obs:
        return default
    arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size else default


def plan_with_model(
    model,
    condition: np.ndarray,
    device: torch.device,
    eval_timestep: int = 0,
    top_modes: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        cond = torch.from_numpy(condition.astype(np.float32)).to(device).unsqueeze(0)
        anchor = model.plan_anchor.to(device).unsqueeze(0)
        if eval_timestep > 0:
            timesteps = torch.full((1,), int(eval_timestep), device=device, dtype=torch.long)
            noise = torch.zeros_like(model._normalize_xy(anchor))
            noisy = model.scheduler.add_noise(model._normalize_xy(anchor), noise, timesteps)
            noisy_xy = model._denormalize_xy(noisy)
        else:
            timesteps = torch.zeros((1,), device=device, dtype=torch.long)
            noisy_xy = anchor
        poses_reg, poses_cls = model.decoder(cond, noisy_xy, timesteps)
        selected = model.select_best(poses_reg, poses_cls)[0].detach().cpu().numpy()
        scores = poses_cls[0].detach().cpu().numpy()
        topk = min(max(int(top_modes), 1), poses_reg.shape[1])
        top_scores_t, top_indices_t = torch.topk(poses_cls, k=topk, dim=-1)
        top_gather = top_indices_t[:, :, None, None].repeat(1, 1, model.config.num_poses, 3)
        top_modes_t = torch.gather(poses_reg, 1, top_gather)
        top_modes_np = top_modes_t[0].detach().cpu().numpy()
        top_indices = top_indices_t[0].detach().cpu().numpy()
        top_scores = top_scores_t[0].detach().cpu().numpy()
    return selected, scores, top_modes_np, top_indices, top_scores


def apply_plan_signs(traj: np.ndarray, x_sign: float = 1.0, y_sign: float = 1.0) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32).copy()
    sx = float(x_sign)
    sy = float(y_sign)
    traj[..., 0] *= sx
    traj[..., 1] *= sy
    if traj.shape[-1] >= 3:
        heading = traj[..., 2]
        traj[..., 2] = np.arctan2(sy * np.sin(heading), sx * np.cos(heading)).astype(np.float32)
    return traj


def validate_trajectory(traj: np.ndarray, args: argparse.Namespace) -> tuple[bool, str, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.shape != (8, 3) or not np.isfinite(traj).all():
        return False, "nonfinite_or_bad_shape", traj
    xy = traj[:, :2].copy()
    if np.max(np.abs(xy[:, 1])) > args.max_abs_y:
        return False, "lateral_outlier", xy
    if len(xy) > 1 and np.max(np.linalg.norm(np.diff(xy, axis=0), axis=-1)) > args.max_step_distance:
        return False, "jump_outlier", xy
    forward = xy[xy[:, 0] >= args.min_forward_x]
    if len(forward) < args.min_valid_points:
        return False, "too_few_forward_points", xy
    return True, "ok", forward


def estimate_speed(forward_xy: np.ndarray, args: argparse.Namespace) -> float:
    if len(forward_xy) < 2:
        return args.fallback_speed
    seg = np.linalg.norm(np.diff(forward_xy[:4], axis=0), axis=-1)
    speed = float(np.median(seg) / max(args.waypoint_dt, 1e-6))
    if not np.isfinite(speed) or speed < args.stop_speed:
        speed = args.fallback_speed
    return float(np.clip(speed, args.target_speed_min, args.target_speed_max))


def safety_adjust_speed(obs: Dict[str, np.ndarray], current_speed: float, target_speed: float, args: argparse.Namespace) -> tuple[float, bool, bool]:
    if "neighbor_vehicles_local" not in obs:
        return target_speed, False, False
    neigh = np.asarray(obs["neighbor_vehicles_local"], dtype=np.float32).reshape(-1, 11)
    valid = neigh[:, 0] > 0.5
    if not valid.any():
        return target_speed, False, False
    vehicles = neigh[valid]
    ahead = vehicles[(vehicles[:, 1] > 0.0) & (np.abs(vehicles[:, 2]) < args.safety_lateral_width)]
    if len(ahead) == 0:
        return target_speed, False, False
    closest = ahead[np.argmin(ahead[:, 1])]
    gap = float(closest[1])
    emergency = gap < args.emergency_distance
    safety_gap = args.safety_distance_min + max(current_speed, 0.0) * args.safety_ttc
    if gap < safety_gap:
        ratio = max(0.0, gap - args.emergency_distance) / max(safety_gap - args.emergency_distance, 1e-6)
        return min(target_speed, target_speed * ratio), True, emergency
    return target_speed, False, False


def multi_point_pure_pursuit(forward_xy: np.ndarray, current_speed: float, prev_steer: float, args: argparse.Namespace) -> tuple[float, float]:
    if len(forward_xy) == 0:
        return 0.0, 0.0
    ld = float(np.clip(args.lookahead_min + args.lookahead_gain * max(current_speed, 0.0), args.lookahead_min, args.lookahead_max))
    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(forward_xy, axis=0), axis=-1))])
    lookaheads = np.asarray([0.75 * ld, ld, 1.35 * ld], dtype=np.float32)
    weights = np.asarray([0.25, 0.50, 0.25], dtype=np.float32)
    targets = []
    for dist in lookaheads:
        idx = int(np.argmin(np.abs(cumulative - dist)))
        targets.append(forward_xy[min(idx, len(forward_xy) - 1)])
    target = np.sum(np.asarray(targets) * weights[:, None], axis=0)
    distance = float(np.linalg.norm(target))
    if distance < 1e-3:
        return 0.0, ld
    curvature = 2.0 * float(target[1]) / max(distance * distance, 1e-6)
    steer_angle = math.atan(args.wheelbase * curvature)
    steer = args.steer_sign * args.steer_gain * steer_angle / max(args.max_steer_rad, 1e-6)
    steer = float(np.clip(steer, prev_steer - args.steer_rate_limit, prev_steer + args.steer_rate_limit))
    return float(np.clip(steer, -1.0, 1.0)), ld


def physical_to_normalized_action(acc: float, steer: float, args: argparse.Namespace) -> np.ndarray:
    acc_norm = 2.0 * (float(acc) - args.env_acc_min) / max(args.env_acc_max - args.env_acc_min, 1e-6) - 1.0
    return np.asarray([np.clip(acc_norm, -1.0, 1.0), np.clip(steer, -1.0, 1.0)], dtype=np.float32)


class CarlaWorldTrajectoryDrawer:
    def __init__(
        self,
        raw_env,
        draw_lifetime: float = 0.2,
        draw_z: float = 0.35,
        draw_enabled: bool = True,
        spectator_topdown: bool = False,
        spectator_height: float = 45.0,
    ):
        import carla

        self.carla = carla
        self.raw_env = raw_env.unwrapped if hasattr(raw_env, "unwrapped") else raw_env
        self.draw_lifetime = float(draw_lifetime)
        self.draw_z = float(draw_z)
        self.draw_enabled = bool(draw_enabled)
        self.spectator_topdown = bool(spectator_topdown)
        self.spectator_height = float(spectator_height)
        self.mode_colors = [
            carla.Color(255, 40, 40),
            carla.Color(160, 90, 255),
            carla.Color(255, 145, 25),
            carla.Color(30, 210, 230),
            carla.Color(150, 90, 70),
            carla.Color(235, 95, 180),
        ]
        self.selected_color = carla.Color(255, 255, 255)

    def update(self, top_modes: np.ndarray, top_indices: np.ndarray, top_scores: np.ndarray, selected: np.ndarray) -> None:
        ego = self._ego()
        world = self._world()
        if ego is None or world is None:
            return
        if self.spectator_topdown:
            self._update_spectator(world, ego)
        if not self.draw_enabled:
            return

        top_modes = np.asarray(top_modes, dtype=np.float32)
        if top_modes.ndim == 4:
            top_modes = top_modes[0]
        if top_modes.ndim != 3:
            return

        for rank, mode in enumerate(top_modes):
            color = self.mode_colors[rank % len(self.mode_colors)]
            thickness = 0.12 if rank == 0 else 0.055
            self._draw_traj(world, ego, mode[:, :2], color=color, thickness=thickness)
            if len(mode):
                end_loc = self._ego_xy_to_world_loc(ego, mode[-1, :2])
                world.debug.draw_string(
                    end_loc,
                    f"T{rank + 1}:{int(top_indices[rank])}/{float(top_scores[rank]):.2f}",
                    draw_shadow=False,
                    color=color,
                    life_time=self.draw_lifetime,
                    persistent_lines=False,
                )

        self._draw_traj(world, ego, np.asarray(selected, dtype=np.float32)[:, :2], color=self.selected_color, thickness=0.16)

    def update_spectator_only(self) -> None:
        if not self.spectator_topdown:
            return
        ego = self._ego()
        world = self._world()
        if ego is not None and world is not None:
            self._update_spectator(world, ego)

    def _ego(self):
        if hasattr(self.raw_env, "get_ego_vehicle"):
            return self.raw_env.get_ego_vehicle()
        return getattr(self.raw_env, "ego", None)

    def _world(self):
        manager = getattr(self.raw_env, "_world", None)
        if manager is not None:
            if hasattr(manager, "carla_world"):
                return manager.carla_world
            if hasattr(manager, "_world"):
                return manager._world
        ego = self._ego()
        return ego.get_world() if ego is not None else None

    def _ego_xy_to_world_loc(self, ego, xy: np.ndarray):
        tf = ego.get_transform()
        loc = tf.location
        yaw = math.radians(float(tf.rotation.yaw))
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x = float(xy[0])
        y = float(xy[1])
        return self.carla.Location(
            x=float(loc.x) + cos_y * x - sin_y * y,
            y=float(loc.y) + sin_y * x + cos_y * y,
            z=float(loc.z) + self.draw_z,
        )

    def _draw_traj(self, world, ego, xy: np.ndarray, color, thickness: float) -> None:
        xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) == 0 or not np.isfinite(xy).all():
            return
        points = [self._ego_xy_to_world_loc(ego, point[:2]) for point in xy]
        for point in points:
            world.debug.draw_point(
                point,
                size=max(float(thickness) * 0.9, 0.04),
                color=color,
                life_time=self.draw_lifetime,
                persistent_lines=False,
            )
        for start, end in zip(points[:-1], points[1:]):
            world.debug.draw_line(
                start,
                end,
                thickness=float(thickness),
                color=color,
                life_time=self.draw_lifetime,
                persistent_lines=False,
            )

    def _update_spectator(self, world, ego) -> None:
        tf = ego.get_transform()
        loc = tf.location
        spectator = world.get_spectator()
        spectator.set_transform(
            self.carla.Transform(
                self.carla.Location(x=loc.x, y=loc.y, z=loc.z + self.spectator_height),
                self.carla.Rotation(pitch=-90.0, yaw=tf.rotation.yaw, roll=0.0),
            )
        )


class LiveTrajectoryPlotter:
    def __init__(self, pause_sec: float = 0.001, yaw_offset_deg: float = 90.0):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        self.plt = plt
        self.Polygon = Polygon
        self.pause_sec = float(max(pause_sec, 0.0))
        self.yaw_offset_rad = math.radians(float(yaw_offset_deg))
        self.mode_colors = ["#d62728", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots(figsize=(6.2, 6.2), dpi=110)
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Closed-loop RSSM-DiDr top trajectories")

    def update(
        self,
        episode: int,
        step_idx: int,
        top_modes: np.ndarray,
        top_indices: np.ndarray,
        top_scores: np.ndarray,
        selected: np.ndarray,
        current_speed: float,
        target_speed: float,
        valid: bool,
        reason: str,
        ego_yaw: float,
    ) -> None:
        self.ax.clear()
        top_modes = np.asarray(top_modes, dtype=np.float32)
        ego_yaw = float(ego_yaw)
        visual_yaw = ego_yaw + self.yaw_offset_rad
        selected_xy = self._ego_to_display_relative(np.asarray(selected, dtype=np.float32)[:, :2], visual_yaw)

        all_xy = [selected_xy]
        if top_modes.ndim == 4:
            top_modes = top_modes[0]
        if top_modes.ndim == 3:
            all_xy.extend([self._ego_to_display_relative(mode[:, :2], visual_yaw) for mode in top_modes])

        for rank, mode in enumerate(top_modes):
            mode_xy = self._ego_to_display_relative(np.asarray(mode, dtype=np.float32)[:, :2], visual_yaw)
            color = self.mode_colors[rank % len(self.mode_colors)]
            alpha = 0.95 if rank == 0 else 0.48
            linewidth = 2.2 if rank == 0 else 1.35
            label = f"Top {rank + 1} mode {int(top_indices[rank])} score={float(top_scores[rank]):.2f}"
            self.ax.plot(
                mode_xy[:, 0],
                mode_xy[:, 1],
                "o-",
                color=color,
                linewidth=linewidth,
                markersize=3.0,
                alpha=alpha,
                label=label,
            )

        self.ax.plot(selected_xy[:, 0], selected_xy[:, 1], "-", color="#111111", linewidth=2.8, label="Selected")
        self._draw_ego(visual_yaw)

        finite_parts = [xy for xy in all_xy if xy.size and np.isfinite(xy).all()]
        finite_xy = np.concatenate(finite_parts, axis=0) if finite_parts else np.zeros((1, 2), dtype=np.float32)
        x_min = min(-12.0, float(np.min(finite_xy[:, 0])) - 4.0)
        x_max = max(12.0, float(np.max(finite_xy[:, 0])) + 4.0)
        y_abs = max(10.0, float(np.max(np.abs(finite_xy[:, 1]))) + 4.0)
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(-y_abs, y_abs)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linewidth=0.4, alpha=0.35)
        self.ax.set_xlabel("display x meters")
        self.ax.set_ylabel("display y meters")
        status = "valid" if valid else f"invalid: {reason}"
        self.ax.set_title(
            f"Episode {episode:03d} Step {step_idx:04d} | {status} | "
            f"speed {current_speed:.2f} -> {target_speed:.2f} m/s | "
            f"yaw {math.degrees(ego_yaw):.1f} deg, vis {math.degrees(visual_yaw):.1f} deg"
        )
        self.ax.legend(fontsize=7, loc="best")
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.plt.pause(self.pause_sec)

    def _ego_to_display_relative(self, xy: np.ndarray, ego_yaw: float) -> np.ndarray:
        """Map CARLA ego-local xy to display coordinates.

        In these replay/env features, local x is forward and local y is right.
        Matplotlib's usual rotation formula assumes y-left, so the lateral
        basis is flipped here to keep positive local y on the vehicle's right.
        """

        xy = np.asarray(xy, dtype=np.float32)
        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)
        out = np.empty_like(xy)
        out[:, 0] = cos_y * xy[:, 0] + sin_y * xy[:, 1]
        out[:, 1] = sin_y * xy[:, 0] - cos_y * xy[:, 1]
        return out

    def _draw_ego(self, ego_yaw: float) -> None:
        body_xy = np.asarray(
            [
                [-1.2, -0.95],
                [3.0, -0.95],
                [3.0, 0.95],
                [-1.2, 0.95],
            ],
            dtype=np.float32,
        )
        nose_xy = np.asarray(
            [
                [3.0, -0.95],
                [4.0, 0.0],
                [3.0, 0.95],
            ],
            dtype=np.float32,
        )
        body = self.Polygon(
            self._ego_to_display_relative(body_xy, ego_yaw),
            closed=True,
            facecolor="#f2f2f2",
            edgecolor="#111111",
            linewidth=1.7,
            zorder=10,
        )
        nose = self.Polygon(
            self._ego_to_display_relative(nose_xy, ego_yaw),
            closed=True,
            facecolor="#111111",
            edgecolor="#111111",
            linewidth=1.2,
            zorder=11,
        )
        self.ax.add_patch(body)
        self.ax.add_patch(nose)
        self.ax.scatter([0.0], [0.0], marker="+", color="#d62728", s=90, linewidths=2.0, zorder=12, label="Ego")

    def close(self) -> None:
        self.plt.ioff()
        self.plt.close(self.fig)


def plot_episode(output_dir: Path, episode: int, traces: list[dict]) -> None:
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
    for item in traces[:: max(len(traces) // 30, 1)]:
        traj = np.asarray(item.get("trajectory", []), dtype=np.float32)
        if traj.shape == (8, 3):
            ax.plot(traj[:, 0], traj[:, 1], color="#1f77b4", alpha=0.25, linewidth=1.0)
    ax.scatter([0], [0], color="#111111", marker="+", s=90)
    ax.set_title(f"Episode {episode} planned ego-frame trajectories")
    ax.set_xlabel("x meters")
    ax.set_ylabel("y meters")
    ax.axis("equal")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(plot_dir / f"episode_{episode:03d}_plans.png")
    plt.close(fig)


def main() -> None:
    args, extra = parse_args()
    if torch is None:
        raise ModuleNotFoundError("Closed-loop planner evaluation requires PyTorch.") from TORCH_IMPORT_ERROR
    from embodied.envs import from_gym

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw_env, config = build_config(args, extra)
    env = from_gym.FromGym(raw_env)
    env = wrap_env(env, config.dreamerv3)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    planner, planner_config = load_planner(args.planner_checkpoint, args.anchor_path, device)
    condition_key = getattr(planner_config, "condition_key", "rssm_latent")
    uses_gt_history = condition_key == "gt_history" or getattr(planner_config, "condition_type", "rssm_latent") == "gt_history"

    if not uses_gt_history and not args.rssm_checkpoint:
        raise ValueError("RSSM planner checkpoint requires --rssm_checkpoint for closed-loop RSSM encoding.")

    action_shape = tuple(env.act_space["action"].shape)
    if action_shape != (2,):
        raise ValueError(f"Closed-loop PID+pure-pursuit controller expects 2D action space, got {action_shape}.")

    rssm = None
    gt_history = None
    if uses_gt_history:
        gt_history = OnlineGTHistory(
            history_length=int(args.history_length or getattr(planner_config, "gt_history_length", 10)),
            waypoint_scale=args.waypoint_scale,
            include_neighbors=bool(getattr(planner_config, "gt_history_include_neighbors", True)) and not args.no_neighbors,
            align_neighbor_ids=bool(getattr(planner_config, "gt_history_align_neighbor_ids", True)) and not args.no_align_neighbor_ids,
        )
    else:
        rssm = OnlineRSSMEncoder(env.obs_space, env.act_space, args.rssm_checkpoint, config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "closed_loop_steps.jsonl"
    summary_path = output_dir / "closed_loop_summary.json"
    step_log = jsonl_path.open("w", encoding="utf-8", buffering=1)

    episode_summaries = []
    pid = PID(args.speed_kp, args.speed_ki, args.speed_kd, args.dt, (args.acc_min, args.acc_max))
    previous_action = np.zeros(action_shape, dtype=np.float32)
    previous_steer = 0.0
    live_plotter = (
        LiveTrajectoryPlotter(args.live_plot_pause, args.live_plot_yaw_offset_deg)
        if args.live_plot
        else None
    )
    carla_drawer = (
        CarlaWorldTrajectoryDrawer(
            raw_env,
            draw_lifetime=args.carla_draw_lifetime,
            draw_z=args.carla_draw_z,
            draw_enabled=args.carla_live_draw,
            spectator_topdown=args.carla_spectator_topdown,
            spectator_height=args.carla_spectator_height,
        )
        if args.carla_live_draw or args.carla_spectator_topdown
        else None
    )
    reverse_hint_printed = False

    try:
        for episode in range(1, args.episodes + 1):
            obs, _ = env.step({"reset": True, "action": np.zeros(action_shape, dtype=np.float32)})
            pid.reset()
            previous_action[:] = 0.0
            previous_steer = 0.0
            if rssm is not None:
                rssm.reset(action_shape)
            if gt_history is not None:
                gt_history.reset()

            total_reward = 0.0
            invalid_count = 0
            safety_count = 0
            emergency_count = 0
            traces = []
            start_time = time.time()

            for step_idx in range(args.max_steps):
                if rssm is not None:
                    condition = rssm.encode(obs, previous_action)
                else:
                    condition = gt_history.encode(obs, previous_action)

                traj, scores, top_modes, top_mode_idx, top_mode_scores = plan_with_model(
                    planner,
                    condition,
                    device,
                    top_modes=args.plot_modes,
                )
                traj = apply_plan_signs(traj, args.plan_x_sign, args.plan_y_sign)
                top_modes = apply_plan_signs(top_modes, args.plan_x_sign, args.plan_y_sign)
                if (
                    not reverse_hint_printed
                    and args.plan_x_sign > 0.0
                    and np.isfinite(top_modes).all()
                    and float(np.median(top_modes[:, -1, 0])) < -args.min_forward_x
                ):
                    print(
                        "[closed_loop] Planner top modes mostly end behind ego. "
                        "If the live trajectories look reversed, retry with --plan_x_sign -1.",
                        flush=True,
                    )
                    reverse_hint_printed = True
                valid, reason, forward_xy = validate_trajectory(traj, args)
                current_speed = obs_scalar(obs, "ego_speed", 0.0)
                ego_yaw = obs_scalar(obs, "ego_yaw", 0.0)

                if valid:
                    target_speed = estimate_speed(forward_xy, args)
                    safety_active = False
                    emergency = False
                    if bool(args.safety_filter):
                        target_speed, safety_active, emergency = safety_adjust_speed(obs, current_speed, target_speed, args)
                    steer, lookahead = multi_point_pure_pursuit(forward_xy, current_speed, previous_steer, args)
                    # if np.max(np.abs(forward_xy[:, 1])) > args.lateral_error_soft_stop:
                    #     target_speed = min(target_speed, args.fallback_speed)
                    #     safety_active = True
                    if emergency:
                        acc = args.acc_min
                        target_speed = 0.0
                    else:
                        acc = pid.step(target_speed, current_speed)
                    safety_count += int(safety_active)
                    emergency_count += int(emergency)
                else:
                    invalid_count += 1
                    target_speed = 0.0
                    lookahead = 0.0
                    steer = float(np.clip(previous_steer, -0.2, 0.2))
                    acc = args.acc_min if current_speed > args.stop_speed else -1.0

                if carla_drawer is not None:
                    carla_drawer.update(
                        top_modes=top_modes,
                        top_indices=top_mode_idx,
                        top_scores=top_mode_scores,
                        selected=traj,
                    )

                if live_plotter is not None:
                    live_plotter.update(
                        episode=episode,
                        step_idx=step_idx,
                        top_modes=top_modes,
                        top_indices=top_mode_idx,
                        top_scores=top_mode_scores,
                        selected=traj,
                        current_speed=current_speed,
                        target_speed=target_speed,
                        valid=bool(valid),
                        reason=reason,
                        ego_yaw=ego_yaw,
                    )

                action = physical_to_normalized_action(acc, steer, args)
                next_obs, info = env.step({"reset": False, "action": action})
                if carla_drawer is not None:
                    carla_drawer.update_spectator_only()
                reward = float(np.asarray(next_obs.get("reward", 0.0)))
                total_reward += reward
                is_last = bool(np.asarray(next_obs.get("is_last", False)))
                is_terminal = bool(np.asarray(next_obs.get("is_terminal", False)))

                row = {
                    "episode": episode,
                    "step": step_idx,
                    "reward": reward,
                    "total_reward": total_reward,
                    "is_last": is_last,
                    "is_terminal": is_terminal,
                    "valid_plan": bool(valid),
                    "plan_status": reason,
                    "target_speed": float(target_speed),
                    "current_speed": float(current_speed),
                    "ego_yaw": float(ego_yaw),
                    "acc": float(acc),
                    "steer": float(steer),
                    "lookahead": float(lookahead),
                    "selected_mode": int(np.argmax(scores)),
                    "selected_score": float(np.max(scores)),
                    "top_mode_indices": top_mode_idx.astype(int).tolist(),
                    "top_mode_scores": top_mode_scores.astype(float).tolist(),
                    "steer_sign": float(args.steer_sign),
                    "steer_gain": float(args.steer_gain),
                    "steer_rate_limit": float(args.steer_rate_limit),
                    "plan_x_sign": float(args.plan_x_sign),
                    "plan_y_sign": float(args.plan_y_sign),
                    "action": action.tolist(),
                    "trajectory": traj.tolist(),
                }
                for key in ("destination_reached", "is_success", "out_of_lane", "time_exceeded", "is_collision"):
                    if key in info:
                        row[key] = bool(np.asarray(info[key]).reshape(-1)[0])
                    elif key in next_obs:
                        row[key] = bool(np.asarray(next_obs[key]).reshape(-1)[0])
                step_log.write(json.dumps(row) + "\n")
                traces.append(row)

                obs = next_obs
                previous_action = action
                previous_steer = steer
                if is_last or is_terminal:
                    break

            summary = {
                "episode": episode,
                "steps": len(traces),
                "return": float(total_reward),
                "invalid_plans": int(invalid_count),
                "safety_events": int(safety_count),
                "emergency_events": int(emergency_count),
                "elapsed_sec": float(time.time() - start_time),
            }
            if traces:
                for key in ("destination_reached", "is_success", "out_of_lane", "time_exceeded", "is_collision"):
                    summary[key] = bool(traces[-1].get(key, False))
            episode_summaries.append(summary)
            print(
                f"[closed_loop] episode={episode:03d} steps={summary['steps']} "
                f"return={summary['return']:.3f} invalid={invalid_count} safety={safety_count} emergency={emergency_count}",
                flush=True,
            )
            if args.save_plots:
                plot_episode(output_dir, episode, traces)
    finally:
        step_log.close()
        if live_plotter is not None:
            live_plotter.close()
        try:
            env.close()
        except Exception:
            pass

    final = {
        "task": args.task,
        "rssm_checkpoint": args.rssm_checkpoint,
        "planner_checkpoint": args.planner_checkpoint,
        "condition_key": condition_key,
        "steer_sign": float(args.steer_sign),
        "steer_gain": float(args.steer_gain),
        "steer_rate_limit": float(args.steer_rate_limit),
        "lookahead_min": float(args.lookahead_min),
        "lookahead_max": float(args.lookahead_max),
        "lookahead_gain": float(args.lookahead_gain),
        "plan_x_sign": float(args.plan_x_sign),
        "plan_y_sign": float(args.plan_y_sign),
        "carla_live_draw": bool(args.carla_live_draw),
        "carla_spectator_topdown": bool(args.carla_spectator_topdown),
        "carla_spectator_height": float(args.carla_spectator_height),
        "episodes": episode_summaries,
        "mean_return": float(np.mean([item["return"] for item in episode_summaries])) if episode_summaries else 0.0,
        "success_rate": float(np.mean([item.get("destination_reached", item.get("is_success", False)) for item in episode_summaries]))
        if episode_summaries
        else 0.0,
        "collision_rate": float(np.mean([item.get("is_collision", False) for item in episode_summaries])) if episode_summaries else 0.0,
    }
    summary_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"[closed_loop] Wrote summary: {summary_path}")
    print(f"[closed_loop] Wrote step log: {jsonl_path}")


if __name__ == "__main__":
    main()
