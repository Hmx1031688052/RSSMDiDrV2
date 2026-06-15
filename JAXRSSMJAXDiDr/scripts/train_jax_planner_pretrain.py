"""Pretrain the pure JAX RSSM-conditioned DiffusionDrive planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from JAXRSSMJAXDiDr.data import PlannerDataset, split_chunk_paths
from JAXRSSMJAXDiDr.models import (
    JAXDiDrConfig,
    init_planner,
    loss_and_metrics,
    save_checkpoint,
)
from JAXRSSMJAXDiDr.models.jax_didr_planner import freeze_plan_anchor_updates

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--anchor_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_key", default="rssm_latent")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--decoder_heads", type=int, default=4)
    parser.add_argument("--decoder_ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument("--latent_noise_std", type=float, default=0.0)
    parser.add_argument("--latent_dropout", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def scalarize(metrics):
    return {k: float(np.asarray(v)) for k, v in metrics.items()}


def average(rows):
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def num_batches(dataset: PlannerDataset, batch_size: int, max_batches: int | None = None) -> int:
    count = int(np.ceil(len(dataset) / max(int(batch_size), 1)))
    return min(count, int(max_batches)) if max_batches is not None else count


def progress_iter(iterator, total: int, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(iterator, total=total, desc=desc, dynamic_ncols=True, leave=False)
    return iterator


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(int(args.seed))

    train_paths, val_paths = split_chunk_paths(args.dataset_dir, args.val_fraction, args.seed)
    train_dataset = PlannerDataset(args.dataset_dir, train_paths, condition_key=args.condition_key)
    val_dataset = PlannerDataset(args.dataset_dir, val_paths, condition_key=args.condition_key) if val_paths else None

    config = JAXDiDrConfig(
        latent_dim=train_dataset.feature_dim,
        plan_anchor_path=str(args.anchor_path),
        condition_key=args.condition_key,
        hidden_dim=args.hidden_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ffn_dim=args.decoder_ffn_dim,
        dropout=args.dropout,
        waypoint_scale=args.waypoint_scale,
        latent_noise_std=args.latent_noise_std,
        latent_dropout=args.latent_dropout,
    )
    rng, init_rng = jax.random.split(rng)
    params = init_planner(init_rng, config)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr, weight_decay=args.weight_decay))
    opt_state = optimizer.init(params)

    def train_step(params, opt_state, rng, batch):
        def objective(p):
            return loss_and_metrics(p, config, rng, batch, training=True)

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        updates = freeze_plan_anchor_updates(updates)
        params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["loss"] = loss
        return params, opt_state, metrics

    def eval_step(params, rng, batch):
        _, metrics = loss_and_metrics(params, config, rng, batch, training=False)
        return metrics

    train_step = jax.jit(train_step)
    eval_step = jax.jit(eval_step)

    run_config = {
        "args": vars(args),
        "model_config": config.to_dict(),
        "train_chunks": [str(p) for p in train_paths],
        "val_chunks": [str(p) for p in val_paths],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    history = []
    best_val = None
    for epoch in range(1, int(args.epochs) + 1):
        train_rows = []
        train_iter = train_dataset.batches(args.batch_size, shuffle=True, seed=args.seed + epoch, drop_last=False)
        train_iter = progress_iter(
            train_iter,
            num_batches(train_dataset, args.batch_size, args.max_train_batches),
            f"jax-didr epoch {epoch:03d} train",
            not args.no_tqdm,
        )
        for batch_idx, batch_np in enumerate(train_iter):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            rng, step_rng = jax.random.split(rng)
            batch = jax.tree_util.tree_map(jnp.asarray, batch_np)
            params, opt_state, metrics = train_step(params, opt_state, step_rng, batch)
            row_metrics = scalarize(metrics)
            train_rows.append(row_metrics)
            if tqdm is not None and hasattr(train_iter, "set_postfix"):
                train_iter.set_postfix(
                    {
                        "loss": f"{row_metrics.get('loss', float('nan')):.4f}",
                        "ade": f"{row_metrics.get('ade', float('nan')):.3f}",
                        "cls": f"{row_metrics.get('cls_loss', float('nan')):.3f}",
                    }
                )
        if tqdm is not None and hasattr(train_iter, "close"):
            train_iter.close()
        train_metrics = average(train_rows)

        val_metrics = {}
        if val_dataset is not None:
            val_rows = []
            val_iter = val_dataset.batches(args.batch_size, shuffle=False, seed=args.seed)
            val_iter = progress_iter(
                val_iter,
                num_batches(val_dataset, args.batch_size, args.max_val_batches),
                f"jax-didr epoch {epoch:03d} val",
                not args.no_tqdm,
            )
            for batch_idx, batch_np in enumerate(val_iter):
                if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                    break
                rng, step_rng = jax.random.split(rng)
                row_metrics = scalarize(eval_step(params, step_rng, jax.tree_util.tree_map(jnp.asarray, batch_np)))
                val_rows.append(row_metrics)
                if tqdm is not None and hasattr(val_iter, "set_postfix"):
                    val_iter.set_postfix(
                        {
                            "loss": f"{row_metrics.get('loss', float('nan')):.4f}",
                            "ade": f"{row_metrics.get('ade', float('nan')):.3f}",
                        }
                    )
            if tqdm is not None and hasattr(val_iter, "close"):
                val_iter.close()
            val_metrics = average(val_rows)

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"[jax_didr_pretrain] epoch={epoch:03d} "
            f"train_loss={train_metrics.get('loss', float('nan')):.6f} "
            f"train_ade={train_metrics.get('ade', float('nan')):.4f} "
            f"val_loss={val_metrics.get('loss', float('nan')):.6f}",
            flush=True,
        )

        if args.save_every > 0 and epoch % int(args.save_every) == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:04d}.pkl.gz", config, params, opt_state, epoch)
        monitor = val_metrics.get("loss", train_metrics.get("loss"))
        if monitor is not None and (best_val is None or monitor < best_val):
            best_val = monitor
            save_checkpoint(output_dir / "best.pkl.gz", config, params, opt_state, epoch)

    save_checkpoint(output_dir / "last.pkl.gz", config, params, opt_state, args.epochs)
    print(f"[jax_didr_pretrain] Wrote final checkpoint: {output_dir / 'last.pkl.gz'}")


if __name__ == "__main__":
    main()
