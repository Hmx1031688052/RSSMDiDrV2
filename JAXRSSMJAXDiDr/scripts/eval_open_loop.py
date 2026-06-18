"""Open-loop evaluation for the pure JAX DiffusionDrive planner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from JAXRSSMJAXDiDr.data import PlannerDataset
from JAXRSSMJAXDiDr.models import load_checkpoint, predict

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--planner_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_key", default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_timestep", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def num_batches(dataset: PlannerDataset, batch_size: int, max_batches: int | None = None) -> int:
    count = int(np.ceil(len(dataset) / max(int(batch_size), 1)))
    return min(count, int(max_batches)) if max_batches is not None else count


def progress_iter(iterator, total: int, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(iterator, total=total, desc="jax-didr open-loop", dynamic_ncols=True)
    return iterator


def main():
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[jax_didr_eval] Loading checkpoint: {args.planner_checkpoint}", flush=True)
    config, params, _, epoch = load_checkpoint(args.planner_checkpoint)
    if args.condition_key:
        config.condition_key = args.condition_key
    print(
        f"[jax_didr_eval] Loaded checkpoint epoch={epoch} condition_key={config.condition_key}",
        flush=True,
    )
    print(f"[jax_didr_eval] Scanning dataset: {args.dataset_dir}", flush=True)
    dataset = PlannerDataset(args.dataset_dir, condition_key=config.condition_key)
    total_batches = num_batches(dataset, args.batch_size, args.max_batches)
    print(
        f"[jax_didr_eval] Dataset samples={len(dataset)} chunks={len(dataset.chunk_paths)} "
        f"feature_dim={dataset.feature_dim} batch_size={args.batch_size} batches={total_batches}",
        flush=True,
    )

    def eval_batch(batch):
        latent = batch[config.condition_key] if config.condition_key in batch else batch["condition"]
        out = predict(params, config, latent, timestep=args.eval_timestep)
        pred = out["trajectory"]
        target = batch["trajectory"]
        dist = jnp.linalg.norm(pred[..., :2] - target[..., :2], axis=-1)
        return {
            "ade": dist.mean(),
            "fde": dist[..., -1].mean(),
            "pred": pred,
            "target": target,
            "poses_cls": out["poses_cls"],
        }

    eval_batch = jax.jit(eval_batch)
    rows = []
    preds = []
    seen = 0
    iterator = dataset.batches(args.batch_size, shuffle=False)
    iterator = progress_iter(iterator, total_batches, not args.no_tqdm)
    for batch_idx, batch_np in enumerate(iterator):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        if batch_idx == 0:
            print("[jax_didr_eval] Running first batch; this includes initial JAX compile.", flush=True)
        batch = jax.tree_util.tree_map(jnp.asarray, batch_np)
        out = jax.device_get(eval_batch(batch))
        batch_samples = int(batch_np["trajectory"].shape[0])
        seen += batch_samples
        rows.append({"ade": float(out["ade"]), "fde": float(out["fde"]), "samples": batch_samples})
        if batch_idx == 0:
            print("[jax_didr_eval] First batch finished; continuing evaluation.", flush=True)
        if args.save_predictions:
            preds.append({k: np.asarray(out[k]) for k in ("pred", "target", "poses_cls")})
        if tqdm is None or args.no_tqdm:
            print(
                f"[jax_didr_eval] batch={batch_idx + 1}/{total_batches} samples={seen}/{len(dataset)} "
                f"ade={rows[-1]['ade']:.4f} fde={rows[-1]['fde']:.4f}",
                flush=True,
            )
    if tqdm is not None and hasattr(iterator, "close"):
        iterator.close()
    total_weight = float(sum(r["samples"] for r in rows))
    metrics = {
        "checkpoint_epoch": int(epoch),
        "samples": int(seen),
        "ade": float(sum(r["ade"] * r["samples"] for r in rows) / total_weight) if rows else float("nan"),
        "fde": float(sum(r["fde"] * r["samples"] for r in rows) / total_weight) if rows else float("nan"),
        "elapsed_sec": float(time.time() - started),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if preds:
        np.savez_compressed(
            output_dir / "predictions.npz",
            pred=np.concatenate([p["pred"] for p in preds], axis=0),
            target=np.concatenate([p["target"] for p in preds], axis=0),
            poses_cls=np.concatenate([p["poses_cls"] for p in preds], axis=0),
        )
    print("[jax_didr_eval] " + " ".join(f"{k}={v}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()

