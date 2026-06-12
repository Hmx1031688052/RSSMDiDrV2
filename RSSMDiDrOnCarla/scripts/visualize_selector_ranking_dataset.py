"""Visualize returns in an exported selector-ranking dataset.
python -m RSSMDiDrOnCarla.scripts.visualize_selector_ranking_dataset \
  --dataset_dir "$RUN_ROOT/selector_ranking_dataset" \
  --output_dir "$RUN_ROOT/selector_ranking_dataset_vis" \
  --max_samples 200000
  """

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=200000)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dataset_paths(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        files = metadata.get("files", [])
        paths = [dataset_dir / name for name in files]
    else:
        paths = sorted(dataset_dir.glob("*.npz"))
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No ranking dataset .npz files found in {dataset_dir}")
    return paths


def load_returns_and_logits(paths: list[Path], max_samples: int, seed: int):
    returns_rows = []
    logits_rows = []
    total = 0
    rng = np.random.default_rng(seed)
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            if "returns" not in data.files or "old_logits" not in data.files:
                raise KeyError(f"{path}: expected keys `returns` and `old_logits`")
            returns = np.asarray(data["returns"], dtype=np.float32)
            old_logits = np.asarray(data["old_logits"], dtype=np.float32)
        if returns.ndim != 2 or old_logits.shape != returns.shape:
            raise ValueError(f"{path}: bad shapes returns={returns.shape} old_logits={old_logits.shape}")
        total += int(returns.shape[0])
        remaining = max(int(max_samples) - sum(row.shape[0] for row in returns_rows), 0)
        if remaining <= 0:
            continue
        if returns.shape[0] > remaining:
            idx = rng.choice(returns.shape[0], size=remaining, replace=False)
            returns = returns[idx]
            old_logits = old_logits[idx]
        returns_rows.append(returns)
        logits_rows.append(old_logits)
    if not returns_rows:
        raise ValueError("No samples loaded from ranking dataset")
    return np.concatenate(returns_rows, axis=0), np.concatenate(logits_rows, axis=0), total


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(np.sum(exp, axis=axis, keepdims=True), 1e-12)


def summarize(returns: np.ndarray, old_logits: np.ndarray, total_samples: int, top_k: int) -> Dict[str, object]:
    old_idx = np.argmax(old_logits, axis=-1)
    oracle_idx = np.argmax(returns, axis=-1)
    row = np.arange(returns.shape[0])
    old_selected = returns[row, old_idx]
    oracle = returns[row, oracle_idx]
    gap = oracle - old_selected
    rank_order = np.argsort(-returns, axis=-1)
    old_rank = np.argmax(rank_order == old_idx[:, None], axis=-1) + 1
    target_prob = softmax((returns - returns.mean(axis=-1, keepdims=True)) / np.maximum(returns.std(axis=-1, keepdims=True), 1e-4))
    entropy = -np.sum(target_prob * np.log(np.maximum(target_prob, 1e-12)), axis=-1)

    mode_mean = returns.mean(axis=0)
    best_modes = np.argsort(-mode_mean)[: max(int(top_k), 1)]
    return {
        "total_dataset_samples": int(total_samples),
        "loaded_samples": int(returns.shape[0]),
        "num_modes": int(returns.shape[1]),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "old_selected_return_mean": float(np.mean(old_selected)),
        "oracle_return_mean": float(np.mean(oracle)),
        "oracle_gap_mean": float(np.mean(gap)),
        "oracle_gap_p50": float(np.percentile(gap, 50)),
        "oracle_gap_p90": float(np.percentile(gap, 90)),
        "old_top1_acc": float(np.mean(old_idx == oracle_idx)),
        "old_rank_mean": float(np.mean(old_rank)),
        "old_rank_p50": float(np.percentile(old_rank, 50)),
        "old_rank_p90": float(np.percentile(old_rank, 90)),
        "target_entropy_mean": float(np.mean(entropy)),
        "mode_return_mean": mode_mean.astype(float).tolist(),
        "best_mean_return_modes": best_modes.astype(int).tolist(),
    }


def plot_all(returns: np.ndarray, old_logits: np.ndarray, summary: Dict[str, object], output_dir: Path, bins: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    old_idx = np.argmax(old_logits, axis=-1)
    oracle_idx = np.argmax(returns, axis=-1)
    row = np.arange(returns.shape[0])
    old_selected = returns[row, old_idx]
    oracle = returns[row, oracle_idx]
    gap = oracle - old_selected
    mode_mean = returns.mean(axis=0)
    mode_std = returns.std(axis=0)
    rank_order = np.argsort(-returns, axis=-1)
    old_rank = np.argmax(rank_order == old_idx[:, None], axis=-1) + 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    ax = axes[0, 0]
    ax.hist(returns.reshape(-1), bins=bins, alpha=0.75, color="#4c78a8", label="All mode returns")
    ax.hist(old_selected, bins=bins, alpha=0.55, color="#f58518", label="Old selected")
    ax.hist(oracle, bins=bins, alpha=0.45, color="#54a24b", label="Oracle best")
    ax.set_title("Return Distribution")
    ax.set_xlabel("RSSM imagined return")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    ax = axes[0, 1]
    ax.hist(gap, bins=bins, color="#e45756", alpha=0.75)
    ax.axvline(float(np.mean(gap)), color="#111111", linewidth=1.5, label=f"mean={np.mean(gap):.3f}")
    ax.set_title("Oracle Gap: best_return - old_selected_return")
    ax.set_xlabel("Return gap")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    ax = axes[1, 0]
    x = np.arange(returns.shape[1])
    ax.bar(x, mode_mean, yerr=mode_std, color="#72b7b2", alpha=0.85, capsize=2)
    ax.set_title("Mean Return By Mode")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Return mean +/- std")
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)

    ax = axes[1, 1]
    bins_rank = np.arange(1, returns.shape[1] + 2) - 0.5
    ax.hist(old_rank, bins=bins_rank, color="#b279a2", alpha=0.8, rwidth=0.85)
    ax.set_title("Old Selector Rank Under RSSM Returns")
    ax.set_xlabel("Rank of old selected mode; 1 is best")
    ax.set_ylabel("Count")
    ax.set_xticks(np.arange(1, returns.shape[1] + 1))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)

    fig.suptitle(
        f"samples={summary['loaded_samples']} old_top1_acc={summary['old_top1_acc']:.3f} "
        f"gap_mean={summary['oracle_gap_mean']:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    path = output_dir / "ranking_returns_summary.png"
    fig.savefig(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    confusion = np.zeros((returns.shape[1], returns.shape[1]), dtype=np.float32)
    np.add.at(confusion, (old_idx, oracle_idx), 1.0)
    confusion = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)
    im = ax.imshow(confusion, cmap="viridis", aspect="auto")
    ax.set_title("Old Selected Mode vs Oracle Best Mode")
    ax.set_xlabel("Oracle best mode")
    ax.set_ylabel("Old selected mode")
    ax.set_xticks(np.arange(returns.shape[1]))
    ax.set_yticks(np.arange(returns.shape[1]))
    fig.colorbar(im, ax=ax, label="row-normalized frequency")
    fig.tight_layout()
    heatmap_path = output_dir / "old_vs_oracle_mode_heatmap.png"
    fig.savefig(heatmap_path)
    plt.close(fig)

    return [str(path), str(heatmap_path)]


def main() -> None:
    args = parse_args()
    paths = dataset_paths(args.dataset_dir)
    returns, old_logits, total = load_returns_and_logits(paths, args.max_samples, args.seed)
    summary = summarize(returns, old_logits, total, args.top_k)
    output_dir = Path(args.output_dir)
    plot_files = plot_all(returns, old_logits, summary, output_dir, args.bins)
    summary["plot_files"] = plot_files
    summary_path = output_dir / "ranking_returns_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "[visualize_selector_ranking] "
        f"samples={summary['loaded_samples']} return_mean={summary['return_mean']:.5f} "
        f"old_top1_acc={summary['old_top1_acc']:.5f} gap_mean={summary['oracle_gap_mean']:.5f}"
    )
    print(f"[visualize_selector_ranking] wrote {summary_path}")
    for path in plot_files:
        print(f"[visualize_selector_ranking] wrote {path}")


if __name__ == "__main__":
    main()
