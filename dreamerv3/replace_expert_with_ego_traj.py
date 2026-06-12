"""Replace expert_waypoints8 with ego's true future 4s trajectory (8 waypoints @ 0.5s).

Reads a replay directory, processes each chunk, and writes back the modified
expert_waypoints8 computed from the ego vehicle's actual driven path.

Usage:
    python -u dreamerv3/replace_expert_with_ego_traj.py \
        --replay_dir ./expert_replay/replay \
        --waypoint_scale 15.0 \
        --dt 0.1
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


def process_chunk(npz_path, waypoint_scale=30.0, dt=0.1, waypoint_interval=5):
    """Process a single replay chunk, replacing expert_waypoints8.

    waypoint_interval: number of env steps between waypoints.
                       dt=0.1, interval=5 → 0.5s per waypoint.
    """
    data = dict(np.load(npz_path, allow_pickle=True))
    keys_before = set(data.keys())

    if "ego_x" not in data or "ego_y" not in data:
        print(f"  SKIP: no ego_x/ego_y in chunk (recollect with updated car_dreamer)")
        return False

    is_first = np.asarray(data["is_first"])
    if not is_first.any():
        return False

    ep_starts = np.where(is_first)[0].tolist()
    ep_ends = ep_starts[1:] + [len(is_first)]

    T_total = len(is_first)
    num_wp = 8
    stride = waypoint_interval
    wp_scale = waypoint_scale

    # Build new expert waypoints
    new_expert = np.zeros((T_total, num_wp * 2), dtype=np.float32)

    ego_x = np.asarray(data["ego_x"], dtype=np.float64).reshape(-1)
    ego_y = np.asarray(data["ego_y"], dtype=np.float64).reshape(-1)
    ego_yaw = np.asarray(data.get("ego_yaw",
                        np.zeros(T_total, dtype=np.float64))).reshape(-1)

    for ep_i, (start, end) in enumerate(zip(ep_starts, ep_ends)):
        ep_len = end - start

        for t in range(start, end):
            rel = t - start  # relative index in episode

            wps_ego = []
            valid_count = 0
            for k in range(num_wp):
                future_idx = t + (k + 1) * stride

                if future_idx < end:
                    # Within episode: use real ego position
                    fx = ego_x[future_idx]
                    fy = ego_y[future_idx]
                    valid_count += 1
                else:
                    # Past end of episode: extrapolate using last known velocity
                    remaining = end - 1
                    lookback = min(5, max(1, ep_len - 1))
                    if lookback >= 2:
                        vx = (ego_x[remaining] - ego_x[max(start, remaining - lookback)]) / \
                             max(1, remaining - max(start, remaining - lookback)) * (1.0 / dt)
                        vy = (ego_y[remaining] - ego_y[max(start, remaining - lookback)]) / \
                             max(1, remaining - max(start, remaining - lookback)) * (1.0 / dt)
                    else:
                        vx, vy = 0.0, 0.0

                    extra_steps = future_idx - (end - 1)
                    fx = ego_x[end - 1] + vx * extra_steps * dt
                    fy = ego_y[end - 1] + vy * extra_steps * dt

                    # Clamp extrapolation to a reasonable distance (30m max)
                    dx = fx - ego_x[end - 1]
                    dy = fy - ego_y[end - 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > 30.0:
                        fx = ego_x[end - 1] + dx / dist * 30.0
                        fy = ego_y[end - 1] + dy / dist * 30.0

                # Convert world → ego frame at time t
                lx, ly = world_to_ego_xy(fx, fy, ego_x[t], ego_y[t], ego_yaw[t])
                wps_ego.extend([lx / wp_scale, ly / wp_scale])

            new_expert[t] = np.clip(np.asarray(wps_ego, dtype=np.float32), -1.0, 1.0)

    # Replace in data
    data["expert_waypoints8"] = new_expert

    # Keep ego_x, ego_y, ego_yaw — they are useful observations for the
    # world model encoder (spatial localization).  Do NOT remove them.
    keys_after = set(data.keys())

    # Save back
    os.remove(npz_path)
    np.savez_compressed(npz_path, **data)

    print(f"  OK: replaced expert_waypoints8 ({keys_before - keys_after} keys removed)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Replace expert_waypoints8 with ego true future trajectory")
    parser.add_argument("--replay_dir", type=str, required=True,
                        help="Path to replay directory containing *.npz chunks")
    parser.add_argument("--waypoint_scale", type=float, default=30.0,
                        help="Normalization scale for waypoints (default: 30.0, must match planner config)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Env step time (default: 0.1s)")
    parser.add_argument("--waypoint_interval", type=int, default=5,
                        help="Steps per waypoint (default: 5, i.e. 0.5s)")
    args = parser.parse_args()

    replay_dir = args.replay_dir
    npz_files = sorted(glob.glob(os.path.join(replay_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {replay_dir}")
        sys.exit(1)

    print(f"Processing {len(npz_files)} chunks in {replay_dir}")
    print(f"  waypoint_scale = {args.waypoint_scale}")
    print(f"  dt = {args.dt}s, waypoint_interval = {args.waypoint_interval} steps")
    print(f"  → {args.waypoint_interval * args.dt}s per waypoint, "
          f"{8 * args.waypoint_interval * args.dt}s total horizon")
    print()

    ok = 0
    skip = 0
    for npz_file in npz_files:
        fname = os.path.basename(npz_file)
        print(f"[{fname}]")
        try:
            if process_chunk(npz_file, args.waypoint_scale, args.dt, args.waypoint_interval):
                ok += 1
            else:
                skip += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            skip += 1

    print(f"\nDone. OK={ok}, skip={skip}, total={len(npz_files)}")


if __name__ == "__main__":
    main()
