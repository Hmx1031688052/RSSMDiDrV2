"""Prepare PolyPlanner expert replay for RSSM-DiDr planner training.

This script runs the existing ego-future replacement step and validates that
`expert_waypoints8` now means true ego future trajectory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RSSMDiDrOnCarla.data.polyplanner_targets import validate_processed_chunk


def _load_replacement_process_chunk():
    script_path = ROOT / "dreamerv3" / "replace_expert_with_ego_traj.py"
    spec = importlib.util.spec_from_file_location("replace_expert_with_ego_traj", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load replacement script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_chunk


process_chunk = _load_replacement_process_chunk()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_dir", required=True, help="Directory containing Dreamer replay `.npz` chunks.")
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--waypoint_interval", type=int, default=5)
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate already-processed replay chunks; do not modify files.",
    )
    parser.add_argument(
        "--summary_path",
        default=None,
        help="Optional JSON summary path. Defaults to `<replay_dir>/ego_future_summary.json`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_dir = Path(args.replay_dir)
    if not replay_dir.is_dir():
        raise FileNotFoundError(f"Replay directory does not exist: {replay_dir}")

    chunk_paths = sorted(replay_dir.glob("*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No `.npz` replay chunks found in: {replay_dir}")

    summaries = []
    replaced = 0
    skipped = 0

    for path in chunk_paths:
        print(f"[prepare] {path.name}")
        if not args.validate_only:
            ok = process_chunk(
                path,
                waypoint_scale=args.waypoint_scale,
                dt=args.dt,
                waypoint_interval=args.waypoint_interval,
            )
            if ok:
                replaced += 1
            else:
                skipped += 1

        summary = validate_processed_chunk(
            path,
            waypoint_scale=args.waypoint_scale,
            dt=args.dt,
            waypoint_interval=args.waypoint_interval,
        )
        summaries.append(summary)
        print(
            "  OK: "
            f"length={summary['length']} "
            f"expert_waypoints8={summary['expert_waypoints8_shape']} "
            f"trajectory={summary['trajectory_shape']}"
        )

    out = {
        "replay_dir": str(replay_dir),
        "chunks": len(chunk_paths),
        "replaced": replaced,
        "skipped": skipped,
        "validate_only": bool(args.validate_only),
        "source": "ego_future",
        "waypoint_scale": float(args.waypoint_scale),
        "dt": float(args.dt),
        "waypoint_interval": int(args.waypoint_interval),
        "summaries": summaries,
    }
    summary_path = Path(args.summary_path) if args.summary_path else replay_dir / "ego_future_summary.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[prepare] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
