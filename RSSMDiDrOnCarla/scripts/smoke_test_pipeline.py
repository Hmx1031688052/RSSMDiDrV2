"""Synthetic smoke test for the RSSM-DiffusionDrive-on-CARLA training pipeline."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print("[smoke]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _make_fake_replay(replay_dir: Path, chunks: int, length: int) -> None:
    replay_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(chunks):
        t = np.arange(length, dtype=np.float32) * 0.1
        x = t * (3.0 + idx)
        y = np.sin(t + idx) * 0.5
        yaw = np.arctan2(np.gradient(y), np.gradient(x)).astype(np.float32)
        is_first = np.zeros(length, dtype=bool)
        is_first[0] = True
        path = replay_dir / f"20260608T000000-{idx:08d}-00000000-{length}.npz"
        np.savez_compressed(
            path,
            expert_waypoints8=np.zeros((length, 16), dtype=np.float32),
            ego_x=x.astype(np.float32),
            ego_y=y.astype(np.float32),
            ego_yaw=yaw.astype(np.float32),
            is_first=is_first,
        )


def _make_fake_latents(latent_dir: Path, replay_dir: Path, latent_dim: int) -> None:
    latent_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for replay_path in sorted(replay_dir.glob("*.npz")):
        with np.load(replay_path) as data:
            length = len(data["ego_x"])
        latent = rng.normal(size=(length, latent_dim)).astype(np.float32)
        np.savez_compressed(latent_dir / replay_path.name, rssm_latent=latent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", default="RSSMDiDrOnCarla/_smoke")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--require_torch", action="store_true", help="Fail if PyTorch is not installed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = ROOT / args.work_dir
    if work_dir.exists():
        shutil.rmtree(work_dir)
    replay_dir = work_dir / "replay"
    latent_dir = work_dir / "latents"
    planner_dir = work_dir / "planner_dataset"
    anchors = work_dir / "anchors.npy"
    ckpts = work_dir / "ckpts"

    _make_fake_replay(replay_dir, chunks=3, length=80)
    _run([sys.executable, "-m", "RSSMDiDrOnCarla.scripts.prepare_polyplanner_replay", "--replay_dir", str(replay_dir)])
    _make_fake_latents(latent_dir, replay_dir, latent_dim=64)
    _run(
        [
            sys.executable,
            "-m",
            "RSSMDiDrOnCarla.scripts.export_planner_dataset",
            "--replay_dir",
            str(replay_dir),
            "--latent_dir",
            str(latent_dir),
            "--output_dir",
            str(planner_dir),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "RSSMDiDrOnCarla.tools.kmeans_polyplanner_anchors",
            "--replay_dir",
            str(replay_dir),
            "--output",
            str(anchors),
            "--num_modes",
            "20",
            "--min_motion_m",
            "0.0",
        ]
    )
    if importlib.util.find_spec("torch") is None:
        message = "[smoke] PyTorch is not installed; skipped planner train step."
        if args.require_torch:
            raise ModuleNotFoundError(message)
        print(message)
    else:
        _run(
            [
                sys.executable,
                "-m",
                "RSSMDiDrOnCarla.scripts.train_rssm_didr_planner",
                "--dataset_dir",
                str(planner_dir),
                "--anchor_path",
                str(anchors),
                "--output_dir",
                str(ckpts),
                "--epochs",
                "1",
                "--batch_size",
                "32",
                "--hidden_dim",
                "64",
                "--decoder_heads",
                "4",
                "--max_train_batches",
                "2",
                "--max_val_batches",
                "1",
                "--device",
                "cpu",
            ]
        )

    print(f"[smoke] OK: {work_dir}")
    if not args.keep:
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
