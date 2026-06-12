"""Train the RSSM-conditioned DiffusionDrive planner on exported CARLA chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as exc:
    torch = None
    DataLoader = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RSSMDiDrOnCarla.data.rssm_planner_dataset import RSSMPlannerDataset, split_chunk_paths


def _move_batch(batch, device: torch.device):
    features, targets = batch
    features = {key: value.to(device) for key, value in features.items()}
    targets = {key: value.to(device) for key, value in targets.items()}
    return features, targets


def _scalarize(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if torch.is_tensor(value) and value.numel() == 1:
            out[key] = float(value.detach().cpu())
    return out


def run_epoch(
    model: RSSMDiffusionDrivePlanner,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    desc: str = "",
    use_tqdm: bool = True,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    batches = 0

    iterator = loader
    progress = None
    if use_tqdm and tqdm is not None:
        progress = tqdm(loader, total=len(loader), desc=desc, dynamic_ncols=True, leave=False)
        iterator = progress

    for batch in iterator:
        features, targets = _move_batch(batch, device)
        with torch.set_grad_enabled(training):
            output = model(features, targets)
            loss = output["loss"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        for key, value in _scalarize(output).items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
        if progress is not None:
            scalars = _scalarize(output)
            postfix = {}
            for key in ("loss", "ade", "cls_loss"):
                if key in scalars:
                    postfix[key] = f"{scalars[key]:.4f}"
            progress.set_postfix(postfix)

    return {key: value / max(batches, 1) for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True, help="Planner dataset exported by export_planner_dataset.py.")
    parser.add_argument("--anchor_path", required=True, help="CARLA anchor `.npy` with shape [20, 8, 2].")
    parser.add_argument("--output_dir", required=True, help="Checkpoint and log directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--decoder_heads", type=int, default=4)
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument("--condition_type", choices=("rssm_latent", "gt_history"), default="rssm_latent")
    parser.add_argument("--condition_key", default=None, help="Feature key to condition on. Defaults from condition_type.")
    parser.add_argument("--history_length", type=int, default=10, help="GT-history length, recorded for gt_history checkpoints.")
    parser.add_argument("--no_neighbors", action="store_true", help="Record gt_history checkpoint as ego-only.")
    parser.add_argument("--no_align_neighbor_ids", action="store_true", help="Record gt_history checkpoint as unaligned neighbor history.")
    parser.add_argument("--latent_noise_std", type=float, default=0.0)
    parser.add_argument("--latent_dropout", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--no_tqdm", action="store_true", help="Disable per-epoch tqdm progress bars.")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args()


def _limit_loader(loader: DataLoader, max_batches: Optional[int]) -> Iterable:
    if max_batches is None:
        yield from loader
        return
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        yield batch


class LimitedLoader:
    def __init__(self, loader: DataLoader, max_batches: Optional[int]):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        return _limit_loader(self.loader, self.max_batches)

    def __len__(self):
        if self.max_batches is None:
            return len(self.loader)
        return min(len(self.loader), self.max_batches)


def main() -> None:
    args = parse_args()
    if torch is None:
        raise ModuleNotFoundError("train_rssm_didr_planner.py requires PyTorch. Install torch before training.") from TORCH_IMPORT_ERROR

    from RSSMDiDrOnCarla.models.rssm_didr_planner import RSSMDiDrConfig, RSSMDiffusionDrivePlanner

    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    print(device_name)
    device = torch.device(device_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_key = args.condition_key or ("gt_history" if args.condition_type == "gt_history" else "rssm_latent")

    train_paths, val_paths = split_chunk_paths(args.dataset_dir, val_fraction=args.val_fraction, seed=args.seed)
    train_dataset = RSSMPlannerDataset(args.dataset_dir, chunk_paths=train_paths, condition_key=condition_key)
    val_dataset = RSSMPlannerDataset(args.dataset_dir, chunk_paths=val_paths, condition_key=condition_key) if val_paths else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )

    config = RSSMDiDrConfig(
        latent_dim=train_dataset.feature_dim,
        plan_anchor_path=str(args.anchor_path),
        condition_type=args.condition_type,
        condition_key=condition_key,
        gt_history_length=args.history_length,
        gt_history_include_neighbors=not args.no_neighbors,
        gt_history_align_neighbor_ids=not args.no_align_neighbor_ids,
        hidden_dim=args.hidden_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        waypoint_scale=args.waypoint_scale,
        latent_noise_std=args.latent_noise_std,
        latent_dropout=args.latent_dropout,
    )
    model = RSSMDiffusionDrivePlanner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_config = {
        "args": vars(args),
        "model_config": config.to_dict(),
        "train_chunks": [str(path) for path in train_paths],
        "val_chunks": [str(path) for path in val_paths],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset is not None else 0,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    history = []
    best_val = None
    for epoch in range(1, args.epochs + 1):
        print(f"[epoch {epoch:03d}/{args.epochs:03d}] start", flush=True)
        train_metrics = run_epoch(
            model,
            LimitedLoader(train_loader, args.max_train_batches),
            optimizer,
            device,
            desc=f"epoch {epoch:03d} train",
            use_tqdm=not args.no_tqdm,
        )
        val_metrics = (
            run_epoch(
                model,
                LimitedLoader(val_loader, args.max_val_batches),
                None,
                device,
                desc=f"epoch {epoch:03d} val",
                use_tqdm=not args.no_tqdm,
            )
            if val_loader is not None
            else {}
        )

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        train_loss = train_metrics.get("loss", float("nan"))
        val_loss = val_metrics.get("loss", float("nan"))
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)

        if args.save_every > 0 and epoch % args.save_every == 0:
            model.save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", optimizer=optimizer, epoch=epoch)

        if val_metrics:
            current = val_metrics["loss"]
            if best_val is None or current < best_val:
                best_val = current
                model.save_checkpoint(output_dir / "best.pt", optimizer=optimizer, epoch=epoch)

    model.save_checkpoint(output_dir / "last.pt", optimizer=optimizer, epoch=args.epochs)
    print(f"[train] Wrote final checkpoint: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
