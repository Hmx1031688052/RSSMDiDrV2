"""Run the CARLA replay to RSSM-DiffusionDrive planner training pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print("[pipeline]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_dir", required=True, help="Dreamer replay dir collected by collect_polyplanner.py.")
    parser.add_argument("--latent_dir", required=True, help="RSSM latent chunks exported from DreamerV3.")
    parser.add_argument("--work_dir", required=True, help="Pipeline output workspace.")
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--waypoint_interval", type=int, default=5)
    parser.add_argument("--num_modes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--latent_key", default=None)
    parser.add_argument("--allow_length_mismatch", action="store_true")
    parser.add_argument("--skip_prepare", action="store_true", help="Replay is already ego-future processed.")
    parser.add_argument("--skip_export", action="store_true", help="Planner dataset already exists in work_dir/planner_dataset.")
    parser.add_argument("--skip_anchor", action="store_true", help="Anchor file already exists in work_dir/anchors.npy.")
    parser.add_argument("--skip_train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_dir = Path(args.replay_dir)
    latent_dir = Path(args.latent_dir)
    work_dir = Path(args.work_dir)
    planner_dataset_dir = work_dir / "planner_dataset"
    anchor_path = work_dir / "anchors.npy"
    checkpoint_dir = work_dir / "checkpoints"
    work_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        _run(
            [
                sys.executable,
                "-m",
                "RSSMDiDrOnCarla.scripts.prepare_polyplanner_replay",
                "--replay_dir",
                str(replay_dir),
                "--waypoint_scale",
                str(args.waypoint_scale),
                "--dt",
                str(args.dt),
                "--waypoint_interval",
                str(args.waypoint_interval),
            ]
        )

    if not args.skip_export:
        cmd = [
            sys.executable,
            "-m",
            "RSSMDiDrOnCarla.scripts.export_planner_dataset",
            "--replay_dir",
            str(replay_dir),
            "--latent_dir",
            str(latent_dir),
            "--output_dir",
            str(planner_dataset_dir),
            "--waypoint_scale",
            str(args.waypoint_scale),
            "--dt",
            str(args.dt),
            "--waypoint_interval",
            str(args.waypoint_interval),
        ]
        if args.latent_key:
            cmd.extend(["--latent_key", args.latent_key])
        if args.allow_length_mismatch:
            cmd.append("--allow_length_mismatch")
        _run(cmd)

    if not args.skip_anchor:
        _run(
            [
                sys.executable,
                "-m",
                "RSSMDiDrOnCarla.tools.kmeans_polyplanner_anchors",
                "--replay_dir",
                str(replay_dir),
                "--output",
                str(anchor_path),
                "--num_modes",
                str(args.num_modes),
                "--waypoint_scale",
                str(args.waypoint_scale),
            ]
        )

    if not args.skip_train:
        _run(
            [
                sys.executable,
                "-m",
                "RSSMDiDrOnCarla.scripts.train_rssm_didr_planner",
                "--dataset_dir",
                str(planner_dataset_dir),
                "--anchor_path",
                str(anchor_path),
                "--output_dir",
                str(checkpoint_dir),
                "--batch_size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--device",
                args.device,
                "--waypoint_scale",
                str(args.waypoint_scale),
            ]
        )

    print(f"[pipeline] Done: {work_dir}")


if __name__ == "__main__":
    main()
