"""Open-loop evaluation for the pure JAX DiffusionDrive planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from JAXRSSMJAXDiDr.data import PlannerDataset
from JAXRSSMJAXDiDr.models import load_checkpoint, predict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--planner_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_key", default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_timestep", type=int, default=0)
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, params, _, epoch = load_checkpoint(args.planner_checkpoint)
    if args.condition_key:
        config.condition_key = args.condition_key
    dataset = PlannerDataset(args.dataset_dir, condition_key=config.condition_key)

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
    for batch_np in dataset.batches(args.batch_size, shuffle=False):
        batch = jax.tree_util.tree_map(jnp.asarray, batch_np)
        out = jax.device_get(eval_batch(batch))
        rows.append({"ade": float(out["ade"]), "fde": float(out["fde"])})
        if args.save_predictions:
            preds.append({k: np.asarray(out[k]) for k in ("pred", "target", "poses_cls")})
    metrics = {
        "checkpoint_epoch": int(epoch),
        "samples": len(dataset),
        "ade": float(np.mean([r["ade"] for r in rows])) if rows else float("nan"),
        "fde": float(np.mean([r["fde"] for r in rows])) if rows else float("nan"),
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

