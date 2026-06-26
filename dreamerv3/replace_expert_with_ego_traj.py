"""Replace expert_waypoints8 with ego's true future 4s trajectory.

The replay saver may split one episode across multiple .npz chunks. Directory
processing therefore computes future waypoints on the concatenated valid replay
timeline, then writes the matching slice back to each chunk.
"""

import argparse
import glob
import math
import os
import sys

import numpy as np


def world_to_ego_xy(wx, wy, ego_x, ego_y, ego_yaw):
    """Convert world coordinate to ego frame. Carla convention: x forward, y right."""
    dx = wx - ego_x
    dy = wy - ego_y
    lx = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
    ly = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy
    return lx, ly


def replay_chunk_length(npz_path, fallback):
    """Read valid chunk length from Dreamer chunk filenames."""
    parts = os.path.splitext(os.path.basename(npz_path))[0].split("-")
    if len(parts) >= 4:
        try:
            return int(parts[3])
        except ValueError:
            pass
    return int(fallback)


def compute_future_waypoints(
    ego_x,
    ego_y,
    ego_yaw,
    is_first,
    waypoint_scale=30.0,
    dt=0.1,
    waypoint_interval=5,
):
    """Build [T, 16] ego-future waypoints across complete replay sequences."""
    is_first = np.asarray(is_first).reshape(-1).astype(bool)
    ego_x = np.asarray(ego_x, dtype=np.float64).reshape(-1)
    ego_y = np.asarray(ego_y, dtype=np.float64).reshape(-1)
    ego_yaw = np.asarray(ego_yaw, dtype=np.float64).reshape(-1)

    total = len(is_first)
    num_wp = 8
    stride = int(waypoint_interval)
    scale = float(waypoint_scale)
    dt = float(dt)
    new_expert = np.zeros((total, num_wp * 2), dtype=np.float32)

    ep_starts = np.where(is_first)[0].tolist()
    if not ep_starts:
        ep_starts = [0]
    ep_ends = ep_starts[1:] + [total]

    for start, end in zip(ep_starts, ep_ends):
        ep_len = end - start
        if ep_len <= 0:
            continue
        for t in range(start, end):
            wps_ego = []
            for k in range(num_wp):
                future_idx = t + (k + 1) * stride
                if future_idx < end:
                    fx = ego_x[future_idx]
                    fy = ego_y[future_idx]
                else:
                    remaining = end - 1
                    lookback = min(5, max(1, ep_len - 1))
                    if lookback >= 2:
                        base = max(start, remaining - lookback)
                        denom = max(1, remaining - base)
                        vx = (ego_x[remaining] - ego_x[base]) / denom * (1.0 / dt)
                        vy = (ego_y[remaining] - ego_y[base]) / denom * (1.0 / dt)
                    else:
                        vx, vy = 0.0, 0.0

                    extra_steps = future_idx - remaining
                    fx = ego_x[remaining] + vx * extra_steps * dt
                    fy = ego_y[remaining] + vy * extra_steps * dt

                    dx = fx - ego_x[remaining]
                    dy = fy - ego_y[remaining]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > 30.0:
                        fx = ego_x[remaining] + dx / dist * 30.0
                        fy = ego_y[remaining] + dy / dist * 30.0

                lx, ly = world_to_ego_xy(fx, fy, ego_x[t], ego_y[t], ego_yaw[t])
                wps_ego.extend([lx / scale, ly / scale])
            new_expert[t] = np.clip(np.asarray(wps_ego, dtype=np.float32), -1.0, 1.0)

    return new_expert


def _replace_chunk(npz_path, new_expert_valid, length):
    data = dict(np.load(npz_path, allow_pickle=True))
    full_len = len(data["is_first"])
    new_expert = np.zeros((full_len, 16), dtype=np.float32)
    new_expert[:length] = new_expert_valid
    data["expert_waypoints8"] = new_expert
    os.remove(npz_path)
    np.savez_compressed(npz_path, **data)


def process_chunk(npz_path, waypoint_scale=30.0, dt=0.1, waypoint_interval=5):
    """Process one chunk in isolation.

    This is kept for compatibility. Prefer process_replay_dir() for small
    chunks, because future waypoints can cross chunk boundaries.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        if "ego_x" not in data or "ego_y" not in data or "is_first" not in data:
            print("  SKIP: no ego_x/ego_y/is_first in chunk")
            return False
        length = replay_chunk_length(npz_path, len(data["is_first"]))
        is_first = np.asarray(data["is_first"])[:length]
        if not is_first.any():
            return False
        ego_x = np.asarray(data["ego_x"], dtype=np.float64).reshape(-1)[:length]
        ego_y = np.asarray(data["ego_y"], dtype=np.float64).reshape(-1)[:length]
        ego_yaw = np.asarray(
            data["ego_yaw"] if "ego_yaw" in data else np.zeros(length, dtype=np.float64),
            dtype=np.float64,
        ).reshape(-1)[:length]

    new_expert = compute_future_waypoints(
        ego_x,
        ego_y,
        ego_yaw,
        is_first,
        waypoint_scale=waypoint_scale,
        dt=dt,
        waypoint_interval=waypoint_interval,
    )
    _replace_chunk(npz_path, new_expert, length)
    print(f"  OK: replaced expert_waypoints8 length={length}")
    return True


def process_replay_dir(replay_dir, waypoint_scale=30.0, dt=0.1, waypoint_interval=5):
    """Process all chunks together so small chunks preserve future horizons."""
    npz_files = sorted(glob.glob(os.path.join(replay_dir, "*.npz")))
    if not npz_files:
        return 0, 0, 0

    lengths = []
    is_first_parts = []
    ego_x_parts = []
    ego_y_parts = []
    ego_yaw_parts = []

    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=True) as data:
            if "ego_x" not in data or "ego_y" not in data or "is_first" not in data:
                print(f"[{os.path.basename(npz_path)}] SKIP: missing ego_x/ego_y/is_first")
                return 0, len(npz_files), len(npz_files)
            length = replay_chunk_length(npz_path, len(data["is_first"]))
            lengths.append(length)
            is_first_parts.append(np.asarray(data["is_first"])[:length].reshape(-1))
            ego_x_parts.append(np.asarray(data["ego_x"], dtype=np.float64)[:length].reshape(-1))
            ego_y_parts.append(np.asarray(data["ego_y"], dtype=np.float64)[:length].reshape(-1))
            ego_yaw_parts.append(
                np.asarray(
                    data["ego_yaw"] if "ego_yaw" in data else np.zeros(length, dtype=np.float64),
                    dtype=np.float64,
                )[:length].reshape(-1)
            )

    is_first = np.concatenate(is_first_parts, axis=0)
    ego_x = np.concatenate(ego_x_parts, axis=0)
    ego_y = np.concatenate(ego_y_parts, axis=0)
    ego_yaw = np.concatenate(ego_yaw_parts, axis=0)
    new_expert = compute_future_waypoints(
        ego_x,
        ego_y,
        ego_yaw,
        is_first,
        waypoint_scale=waypoint_scale,
        dt=dt,
        waypoint_interval=waypoint_interval,
    )

    offset = 0
    ok = 0
    for npz_path, length in zip(npz_files, lengths):
        _replace_chunk(npz_path, new_expert[offset : offset + length], length)
        offset += length
        print(f"[{os.path.basename(npz_path)}] OK: replaced expert_waypoints8 length={length}")
        ok += 1

    return ok, 0, len(npz_files)


def main():
    parser = argparse.ArgumentParser(description="Replace expert_waypoints8 with ego true future trajectory")
    parser.add_argument("--replay_dir", type=str, required=True, help="Path to replay directory containing *.npz chunks")
    parser.add_argument(
        "--waypoint_scale",
        type=float,
        default=30.0,
        help="Normalization scale for waypoints (default: 30.0, must match planner config)",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Env step time (default: 0.1s)")
    parser.add_argument(
        "--waypoint_interval",
        type=int,
        default=5,
        help="Steps per waypoint (default: 5, i.e. 0.5s)",
    )
    args = parser.parse_args()

    replay_dir = args.replay_dir
    npz_files = sorted(glob.glob(os.path.join(replay_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {replay_dir}")
        sys.exit(1)

    print(f"Processing {len(npz_files)} chunks in {replay_dir}")
    print(f"  waypoint_scale = {args.waypoint_scale}")
    print(f"  dt = {args.dt}s, waypoint_interval = {args.waypoint_interval} steps")
    print(f"  {args.waypoint_interval * args.dt}s per waypoint, {8 * args.waypoint_interval * args.dt}s total horizon")
    print()

    ok, skip, _ = process_replay_dir(
        replay_dir,
        waypoint_scale=args.waypoint_scale,
        dt=args.dt,
        waypoint_interval=args.waypoint_interval,
    )
    print(f"\nDone. OK={ok}, skip={skip}, total={len(npz_files)}")


if __name__ == "__main__":
    main()
