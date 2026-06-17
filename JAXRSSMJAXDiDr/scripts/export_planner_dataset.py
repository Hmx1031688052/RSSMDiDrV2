"""Export aligned RSSM-latent planner chunks from processed Dreamer replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from JAXRSSMJAXDiDr.data.polyplanner_targets import load_replay_chunk, require_ego_future_waypoints, waypoints8_to_trajectory
from JAXRSSMJAXDiDr.data.rssm_planner_dataset import validate_planner_chunk


RSSM_COMPONENT_KEYS = ("deter", "stoch", "logit", "mean", "std")


def _load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def extract_rssm_latent(chunk: Dict[str, np.ndarray], latent_key: Optional[str] = None) -> np.ndarray:
    """Return a flattened `[T, D]` RSSM latent from an exported Dreamer chunk."""

    if latent_key:
        if latent_key not in chunk:
            raise KeyError(f"Latent key `{latent_key}` not found. Available keys: {sorted(chunk.keys())}")
        latent = np.asarray(chunk[latent_key], dtype=np.float32)
    elif "rssm_latent" in chunk:
        latent = np.asarray(chunk["rssm_latent"], dtype=np.float32)
    else:
        parts = []
        for key in RSSM_COMPONENT_KEYS:
            if key in chunk:
                value = np.asarray(chunk[key], dtype=np.float32)
                if value.ndim == 0:
                    continue
                parts.append(value.reshape(value.shape[0], -1))
        if not parts:
            raise KeyError(
                "Could not infer RSSM latent. Expected `rssm_latent` or one of "
                f"{RSSM_COMPONENT_KEYS}; got keys {sorted(chunk.keys())}"
            )
        latent = np.concatenate(parts, axis=-1)

    if latent.ndim < 2:
        raise ValueError(f"Expected latent with leading time dimension and feature dims, got {latent.shape}")
    return latent.reshape(latent.shape[0], -1).astype(np.float32)


def _pair_by_name(replay_dir: Path, latent_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for replay_path in sorted(replay_dir.glob("*.npz")):
        latent_path = latent_dir / replay_path.name
        if not latent_path.is_file():
            raise FileNotFoundError(
                f"Missing latent chunk with matching name for {replay_path.name}: {latent_path}"
            )
        pairs.append((replay_path, latent_path))
    return pairs


def _pair_by_order(replay_dir: Path, latent_dir: Path) -> list[tuple[Path, Path]]:
    replay_paths = sorted(replay_dir.glob("*.npz"))
    latent_paths = sorted(latent_dir.glob("*.npz"))
    if len(replay_paths) != len(latent_paths):
        raise ValueError(
            f"Cannot pair by order: replay chunks={len(replay_paths)} latent chunks={len(latent_paths)}"
        )
    return list(zip(replay_paths, latent_paths))


def export_pair(
    replay_path: Path,
    latent_path: Path,
    output_path: Path,
    waypoint_scale: float,
    dt: float,
    waypoint_interval: int,
    latent_key: Optional[str],
    allow_length_mismatch: bool,
) -> Dict[str, object]:
    replay = load_replay_chunk(replay_path)
    latent_chunk = _load_npz(latent_path)

    waypoints = require_ego_future_waypoints(replay, replay_path)
    trajectory = waypoints8_to_trajectory(waypoints, waypoint_scale=waypoint_scale)
    latent = extract_rssm_latent(latent_chunk, latent_key=latent_key)

    length = min(len(latent), len(trajectory))
    if len(latent) != len(trajectory) and not allow_length_mismatch:
        raise ValueError(
            f"Length mismatch for {replay_path.name}: latent={len(latent)} trajectory={len(trajectory)}. "
            "Use --allow_length_mismatch to trim to the shorter length."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        rssm_latent=latent[:length].astype(np.float32),
        expert_waypoints8=waypoints[:length].astype(np.float32),
        future_ego_waypoints8=waypoints[:length].astype(np.float32),
        trajectory=trajectory[:length].astype(np.float32),
        waypoint_scale=np.float32(waypoint_scale),
        dt=np.float32(dt),
        waypoint_interval=np.int32(waypoint_interval),
        target_source=np.asarray("ego_future"),
        replay_chunk=np.asarray(replay_path.name),
        latent_chunk=np.asarray(latent_path.name),
    )

    summary = validate_planner_chunk(output_path)
    summary.update(
        {
            "replay_chunk": str(replay_path),
            "latent_chunk": str(latent_path),
            "output_chunk": str(output_path),
            "waypoint_scale": float(waypoint_scale),
            "dt": float(dt),
            "waypoint_interval": int(waypoint_interval),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_dir", required=True, help="Processed replay dir after prepare_polyplanner_replay.py.")
    parser.add_argument("--latent_dir", required=True, help="Dreamer RSSM latent `.npz` chunks.")
    parser.add_argument("--output_dir", required=True, help="Output planner dataset dir.")
    parser.add_argument("--latent_key", default=None, help="Optional explicit latent key inside latent chunks.")
    parser.add_argument("--pair_by_order", action="store_true", help="Pair sorted replay/latent chunks by order instead of name.")
    parser.add_argument("--allow_length_mismatch", action="store_true", help="Trim replay/latent chunks to the shorter length.")
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--waypoint_interval", type=int, default=5)
    parser.add_argument("--summary_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_dir = Path(args.replay_dir)
    latent_dir = Path(args.latent_dir)
    output_dir = Path(args.output_dir)

    if not replay_dir.is_dir():
        raise FileNotFoundError(f"Replay directory does not exist: {replay_dir}")
    if not latent_dir.is_dir():
        raise FileNotFoundError(f"Latent directory does not exist: {latent_dir}")

    pairs = _pair_by_order(replay_dir, latent_dir) if args.pair_by_order else _pair_by_name(replay_dir, latent_dir)
    if not pairs:
        raise FileNotFoundError(f"No replay chunks found in: {replay_dir}")

    summaries = []
    for replay_path, latent_path in pairs:
        output_path = output_dir / replay_path.name
        print(f"[export] {replay_path.name} + {latent_path.name} -> {output_path.name}")
        summaries.append(
            export_pair(
                replay_path,
                latent_path,
                output_path,
                waypoint_scale=args.waypoint_scale,
                dt=args.dt,
                waypoint_interval=args.waypoint_interval,
                latent_key=args.latent_key,
                allow_length_mismatch=args.allow_length_mismatch,
            )
        )

    out = {
        "source": "ego_future",
        "replay_dir": str(replay_dir),
        "latent_dir": str(latent_dir),
        "output_dir": str(output_dir),
        "chunks": len(summaries),
        "samples": int(sum(item["length"] for item in summaries)),
        "waypoint_scale": float(args.waypoint_scale),
        "dt": float(args.dt),
        "waypoint_interval": int(args.waypoint_interval),
        "summaries": summaries,
    }
    summary_path = Path(args.summary_path) if args.summary_path else output_dir / "planner_dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[export] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
