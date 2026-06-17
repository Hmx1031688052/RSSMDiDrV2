"""Cluster CARLA ego-future trajectories into DiffusionDrive plan anchors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from JAXRSSMJAXDiDr.data.polyplanner_targets import collect_anchor_xy


def _kmeans_numpy(data: np.ndarray, clusters: int, iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic NumPy K-Means fallback for `[N, D]` data."""

    if len(data) < clusters:
        raise ValueError(f"Need at least {clusters} samples for K-Means, got {len(data)}")

    rng = np.random.default_rng(seed)
    centers = data[rng.choice(len(data), size=clusters, replace=False)].copy()
    labels = np.zeros(len(data), dtype=np.int64)

    for _ in range(iterations):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for idx in range(clusters):
            mask = new_labels == idx
            if mask.any():
                new_centers[idx] = data[mask].mean(axis=0)
            else:
                new_centers[idx] = data[rng.integers(0, len(data))]
        labels = new_labels
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    return centers, labels


def _kmeans(data: np.ndarray, clusters: int, iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Run sklearn K-Means if available, otherwise use NumPy fallback."""

    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=clusters, random_state=seed, n_init=10, max_iter=iterations)
        labels = km.fit_predict(data)
        return km.cluster_centers_.astype(np.float32), labels.astype(np.int64), "sklearn"
    except Exception:
        centers, labels = _kmeans_numpy(data, clusters=clusters, iterations=iterations, seed=seed)
        return centers.astype(np.float32), labels.astype(np.int64), "numpy"


def cluster_anchors(
    xy: np.ndarray,
    num_modes: int = 20,
    iterations: int = 300,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Cluster `[N, 8, 2]` ego-future xy samples into `[K, 8, 2]` anchors."""

    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[1:] != (8, 2):
        raise ValueError(f"Expected xy samples with shape [N, 8, 2], got {xy.shape}")

    flat = xy.reshape(len(xy), -1)
    centers, labels, backend = _kmeans(flat, num_modes, iterations, seed)
    anchors = centers.reshape(num_modes, 8, 2).astype(np.float32)
    return anchors, labels, backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_dir", required=True, help="Prepared replay directory containing `.npz` chunks.")
    parser.add_argument("--output", required=True, help="Output `.npy` path for `[K, 8, 2]` anchors.")
    parser.add_argument("--num_modes", type=int, default=20)
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument("--min_motion_m", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xy = collect_anchor_xy(
        args.replay_dir,
        waypoint_scale=args.waypoint_scale,
        min_motion_m=args.min_motion_m,
    )
    anchors, labels, backend = cluster_anchors(
        xy,
        num_modes=args.num_modes,
        iterations=args.iterations,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, anchors)

    counts = np.bincount(labels, minlength=args.num_modes)
    summary = {
        "source": "ego_future",
        "replay_dir": str(args.replay_dir),
        "output": str(output),
        "backend": backend,
        "num_samples": int(len(xy)),
        "num_modes": int(args.num_modes),
        "anchor_shape": list(anchors.shape),
        "waypoint_scale": float(args.waypoint_scale),
        "min_motion_m": float(args.min_motion_m),
        "cluster_counts": counts.tolist(),
    }
    summary_path = Path(args.summary_path) if args.summary_path else output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[anchors] Wrote anchors: {output} shape={anchors.shape}")
    print(f"[anchors] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
