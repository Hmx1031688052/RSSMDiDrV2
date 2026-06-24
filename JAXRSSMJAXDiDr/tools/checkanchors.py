'''
python -m JAXRSSMJAXDiDr.tools.checkanchors --anchor_path "E:\carla_code\ForDebug/anchors.npy" --output polyplanner_anchors_15.png --show_counts
  '''
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_summary(anchor_path: Path):
    json_path = anchor_path.with_suffix(".json")
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_path", required=True, help="Path to anchors .npy file")
    parser.add_argument("--output", default=None, help="Output png path")
    parser.add_argument("--show_counts", action="store_true")
    args = parser.parse_args()

    anchor_path = Path(args.anchor_path)
    anchors = np.load(anchor_path)

    print("[anchors] path:", anchor_path)
    print("[anchors] shape:", anchors.shape)
    print("[anchors] min/max:", anchors.min(), anchors.max())

    if anchors.ndim != 3 or anchors.shape[1:] != (8, 2):
        raise ValueError(f"Expected anchors shape [K, 8, 2], got {anchors.shape}")

    summary = load_summary(anchor_path)
    counts = None
    if summary is not None:
        counts = summary.get("cluster_counts", None)
        print("[summary] num_samples:", summary.get("num_samples"))
        print("[summary] num_modes:", summary.get("num_modes"))
        print("[summary] waypoint_scale:", summary.get("waypoint_scale"))
        print("[summary] cluster_counts:", counts)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter([0.0], [0.0], marker="x", s=100, label="ego origin")


    for k in range(anchors.shape[0]):
        traj = anchors[k]   # [8, 2]
        x = traj[:, 0]
        y = traj[:, 1]

        label = f"{k}"
        if args.show_counts and counts is not None:
            label = f"{k}, n={counts[k]}"

        ax.plot(x, y, marker="o", linewidth=2, markersize=4, alpha=0.85)
        ax.text(x[-1], y[-1], label, fontsize=9)

        ax.annotate(
            "",
            xy=(x[-1], y[-1]),
            xytext=(x[-2], y[-2]),
            arrowprops=dict(arrowstyle="->", linewidth=1.0, alpha=0.7),
        )

    ax.set_title(f"PolyPlanner KMeans anchors, K={anchors.shape[0]}")
    ax.set_xlabel("local x / m")
    ax.set_ylabel("local y / m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)
    ax.legend()

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print("[visualize] saved to:", output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
