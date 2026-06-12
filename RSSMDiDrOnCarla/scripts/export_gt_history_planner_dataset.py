"""Export planner dataset conditioned on GT ego/neighbor state history."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RSSMDiDrOnCarla.data.gt_history_features import build_gt_history_features
from RSSMDiDrOnCarla.data.polyplanner_targets import (
    iter_replay_chunks,
    require_ego_future_waypoints,
    waypoints8_to_trajectory,
)
from RSSMDiDrOnCarla.data.rssm_planner_dataset import validate_planner_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_dir", required=True, help="Prepared replay dir after prepare_polyplanner_replay.py.")
    parser.add_argument("--output_dir", required=True, help="Output planner dataset dir.")
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument("--history_length", type=int, default=10)
    parser.add_argument("--no_neighbors", action="store_true", help="Use ego history only.")
    parser.add_argument("--no_align_neighbor_ids", action="store_true", help="Do not align neighbor history by actor id.")
    parser.add_argument("--summary_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for replay_path, replay in iter_replay_chunks(replay_dir):
        waypoints = require_ego_future_waypoints(replay, replay_path)
        trajectory = waypoints8_to_trajectory(waypoints, waypoint_scale=args.waypoint_scale)
        gt_history = build_gt_history_features(
            replay,
            history_length=args.history_length,
            waypoint_scale=args.waypoint_scale,
            include_neighbors=not args.no_neighbors,
            align_neighbor_ids=not args.no_align_neighbor_ids,
        )
        length = min(len(gt_history), len(trajectory), len(waypoints))
        output_path = output_dir / replay_path.name
        np.savez_compressed(
            output_path,
            gt_history=gt_history[:length].astype(np.float32),
            expert_waypoints8=waypoints[:length].astype(np.float32),
            future_ego_waypoints8=waypoints[:length].astype(np.float32),
            trajectory=trajectory[:length].astype(np.float32),
            condition_type=np.asarray("gt_history"),
            condition_key=np.asarray("gt_history"),
            history_length=np.int32(args.history_length),
            include_neighbors=np.asarray(not args.no_neighbors),
            align_neighbor_ids=np.asarray(not args.no_align_neighbor_ids),
            waypoint_scale=np.float32(args.waypoint_scale),
            target_source=np.asarray("ego_future"),
            replay_chunk=np.asarray(replay_path.name),
        )
        summary = validate_planner_chunk(output_path, condition_key="gt_history")
        summary.update(
            {
                "replay_chunk": str(replay_path),
                "output_chunk": str(output_path),
                "history_length": int(args.history_length),
                "include_neighbors": bool(not args.no_neighbors),
                "align_neighbor_ids": bool(not args.no_align_neighbor_ids),
                "waypoint_scale": float(args.waypoint_scale),
            }
        )
        summaries.append(summary)
        print(f"[export_gt_history] {replay_path.name}: gt_history={gt_history[:length].shape}")

    out = {
        "source": "ego_future",
        "condition_type": "gt_history",
        "condition_key": "gt_history",
        "replay_dir": str(replay_dir),
        "output_dir": str(output_dir),
        "chunks": len(summaries),
        "samples": int(sum(item["length"] for item in summaries)),
        "history_length": int(args.history_length),
        "include_neighbors": bool(not args.no_neighbors),
        "align_neighbor_ids": bool(not args.no_align_neighbor_ids),
        "waypoint_scale": float(args.waypoint_scale),
        "summaries": summaries,
    }
    summary_path = Path(args.summary_path) if args.summary_path else output_dir / "gt_history_planner_dataset_summary.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[export_gt_history] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
