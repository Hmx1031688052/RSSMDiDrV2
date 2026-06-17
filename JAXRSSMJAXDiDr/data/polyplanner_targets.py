"""Targets derived from ego-future `expert_waypoints8` replay fields.

The project convention after running `dreamerv3/replace_expert_with_ego_traj.py`
is that `expert_waypoints8` stores the ego vehicle's actual future path:
8 waypoints at 0.5s spacing, normalized by `waypoint_scale`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np


NUM_WAYPOINTS = 8
WAYPOINT_DIM = 2
DEFAULT_WAYPOINT_SCALE = 30.0


def _first_sequence_length(chunk: Dict[str, np.ndarray]) -> Optional[int]:
    for value in chunk.values():
        if value.ndim > 0:
            return int(value.shape[0])
    return None


def replay_chunk_length(path: str | Path, fallback: Optional[int] = None) -> int:
    """Return the valid replay chunk length encoded in Dreamer chunk filenames."""

    stem = Path(path).stem
    parts = stem.split("-")
    if len(parts) >= 4:
        try:
            return int(parts[3])
        except ValueError:
            pass
    if fallback is not None:
        return int(fallback)
    raise ValueError(f"Could not infer replay chunk length from filename: {path}")


def load_replay_chunk(path: str | Path, trim_to_length: bool = True) -> Dict[str, np.ndarray]:
    """Load one replay chunk and optionally trim arrays to the filename length."""

    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        chunk = {key: np.asarray(data[key]) for key in data.files}

    if trim_to_length:
        fallback = _first_sequence_length(chunk)
        length = replay_chunk_length(path, fallback=fallback)
        chunk = {
            key: value[:length] if value.ndim > 0 and value.shape[0] >= length else value
            for key, value in chunk.items()
        }
    return chunk


def iter_replay_chunks(replay_dir: str | Path, trim_to_length: bool = True) -> Iterator[Tuple[Path, Dict[str, np.ndarray]]]:
    """Yield `(path, chunk)` pairs for all `.npz` replay chunks in order."""

    replay_dir = Path(replay_dir)
    for path in sorted(replay_dir.glob("*.npz")):
        yield path, load_replay_chunk(path, trim_to_length=trim_to_length)


def require_ego_future_waypoints(chunk: Dict[str, np.ndarray], path: str | Path = "<chunk>") -> np.ndarray:
    """Return validated replacement `expert_waypoints8` as `[T, 16]` float32."""

    if "expert_waypoints8" not in chunk:
        raise KeyError(f"{path}: missing required field `expert_waypoints8`")

    waypoints = np.asarray(chunk["expert_waypoints8"], dtype=np.float32)
    if waypoints.ndim != 2 or waypoints.shape[-1] != NUM_WAYPOINTS * WAYPOINT_DIM:
        raise ValueError(
            f"{path}: expected expert_waypoints8 shape [T, {NUM_WAYPOINTS * WAYPOINT_DIM}], "
            f"got {waypoints.shape}"
        )
    if not np.isfinite(waypoints).all():
        raise ValueError(f"{path}: expert_waypoints8 contains NaN or Inf")
    return waypoints


def waypoints8_to_xy(expert_waypoints8: np.ndarray, waypoint_scale: float = DEFAULT_WAYPOINT_SCALE) -> np.ndarray:
    """Convert normalized `[T, 16]` or `[16]` waypoints to unnormalized xy."""

    waypoints = np.asarray(expert_waypoints8, dtype=np.float32)
    if waypoints.shape[-1] != NUM_WAYPOINTS * WAYPOINT_DIM:
        raise ValueError(f"Expected final dimension 16 for expert_waypoints8, got {waypoints.shape}")
    xy = waypoints.reshape(*waypoints.shape[:-1], NUM_WAYPOINTS, WAYPOINT_DIM)
    return xy * np.float32(waypoint_scale)


def xy_to_heading(xy: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Estimate local heading for each waypoint from consecutive xy deltas.

    Repeated or nearly static points reuse the most recent valid heading.
    """

    xy = np.asarray(xy, dtype=np.float32)
    if xy.shape[-2:] != (NUM_WAYPOINTS, WAYPOINT_DIM):
        raise ValueError(f"Expected xy shape [..., {NUM_WAYPOINTS}, {WAYPOINT_DIM}], got {xy.shape}")

    flat = xy.reshape(-1, NUM_WAYPOINTS, WAYPOINT_DIM)
    headings = np.zeros((flat.shape[0], NUM_WAYPOINTS), dtype=np.float32)

    for sample_idx, points in enumerate(flat):
        previous = np.float32(0.0)
        for waypoint_idx in range(NUM_WAYPOINTS):
            if waypoint_idx == 0:
                delta = points[0]
            else:
                delta = points[waypoint_idx] - points[waypoint_idx - 1]

            norm = float(np.linalg.norm(delta))
            if norm > eps:
                previous = np.float32(np.arctan2(float(delta[1]), float(delta[0])))
            headings[sample_idx, waypoint_idx] = previous

    return headings.reshape(xy.shape[:-1])


def waypoints8_to_trajectory(
    expert_waypoints8: np.ndarray,
    waypoint_scale: float = DEFAULT_WAYPOINT_SCALE,
) -> np.ndarray:
    """Convert replacement `expert_waypoints8` to DiffusionDrive `[T, 8, 3]`.

    The output channels are `(x, y, heading)` in the same ego-local frame used
    by the replacement script. Heading is estimated from waypoint differences.
    """

    xy = waypoints8_to_xy(expert_waypoints8, waypoint_scale=waypoint_scale)
    heading = xy_to_heading(xy)
    return np.concatenate([xy, heading[..., None]], axis=-1).astype(np.float32)


def iter_replay_trajectories(
    replay_dir: str | Path,
    waypoint_scale: float = DEFAULT_WAYPOINT_SCALE,
) -> Iterator[Tuple[Path, np.ndarray]]:
    """Yield trajectory arrays `[T, 8, 3]` from every processed replay chunk."""

    for path, chunk in iter_replay_chunks(replay_dir):
        waypoints = require_ego_future_waypoints(chunk, path)
        yield path, waypoints8_to_trajectory(waypoints, waypoint_scale=waypoint_scale)


def collect_anchor_xy(
    replay_dir: str | Path,
    waypoint_scale: float = DEFAULT_WAYPOINT_SCALE,
    min_motion_m: float = 0.25,
) -> np.ndarray:
    """Collect `[N, 8, 2]` anchor trajectories from processed replay chunks."""

    samples = []
    for path, chunk in iter_replay_chunks(replay_dir):
        waypoints = require_ego_future_waypoints(chunk, path)
        xy = waypoints8_to_xy(waypoints, waypoint_scale=waypoint_scale)
        displacement = np.linalg.norm(xy[..., -1, :] - xy[..., 0, :], axis=-1)
        samples.append(xy[displacement >= min_motion_m])

    if not samples:
        raise ValueError(f"No anchor samples found in replay directory: {replay_dir}")
    out = np.concatenate(samples, axis=0).astype(np.float32)
    if out.size == 0:
        raise ValueError(
            f"All anchor samples were filtered out by min_motion_m={min_motion_m}; "
            "lower the threshold or inspect expert_waypoints8."
        )
    return out


def validate_processed_chunk(
    path: str | Path,
    waypoint_scale: float = DEFAULT_WAYPOINT_SCALE,
    dt: float = 0.1,
    waypoint_interval: int = 5,
) -> Dict[str, object]:
    """Validate processed chunk fields and return a concise summary."""

    chunk = load_replay_chunk(path)
    waypoints = require_ego_future_waypoints(chunk, path)
    trajectory = waypoints8_to_trajectory(waypoints, waypoint_scale=waypoint_scale)

    if trajectory.shape != (len(waypoints), NUM_WAYPOINTS, 3):
        raise ValueError(f"{path}: expected trajectory [T, 8, 3], got {trajectory.shape}")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{path}: converted trajectory contains NaN or Inf")

    missing_pose = [key for key in ("ego_x", "ego_y") if key not in chunk]
    if missing_pose:
        raise KeyError(f"{path}: missing required ego pose keys after collection: {missing_pose}")

    return {
        "path": str(path),
        "length": int(len(waypoints)),
        "expert_waypoints8_shape": tuple(waypoints.shape),
        "trajectory_shape": tuple(trajectory.shape),
        "source": "ego_future",
        "waypoint_scale": float(waypoint_scale),
        "dt": float(dt),
        "waypoint_interval": int(waypoint_interval),
    }
