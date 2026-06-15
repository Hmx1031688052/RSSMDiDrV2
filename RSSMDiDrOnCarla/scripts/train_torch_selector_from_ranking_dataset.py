"""Train a Torch planner selector from an exported ranking dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--planner_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--anchor_path", default=None)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--policy_temperature", type=float, default=1.0)
    parser.add_argument("--return_temperature", type=float, default=1.0)
    parser.add_argument("--ranking_weight", type=float, default=1.0)
    parser.add_argument("--rl_weight", type=float, default=0.1)
    parser.add_argument("--kl_weight", type=float, default=0.05)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--adv_eps", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--no_preload", action="store_true", help="Disable loading the ranking dataset into RAM.")
    parser.add_argument("--no_tqdm", action="store_true")
    return parser.parse_args()


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


class RankingDataset:
    def __init__(self, dataset_dir: str | Path, preload: bool = True):
        self.dataset_dir = Path(dataset_dir)
        metadata_path = self.dataset_dir / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            files = metadata.get("files", [])
            self.paths = [self.dataset_dir / name for name in files]
        else:
            metadata = {}
            self.paths = sorted(self.dataset_dir.glob("*.npz"))
        self.paths = [path for path in self.paths if path.is_file()]
        if not self.paths:
            raise FileNotFoundError(f"No selector ranking .npz files found in {self.dataset_dir}")
        self.metadata = metadata
        self.lengths = []
        for path in self.paths:
            with np.load(path, allow_pickle=True) as data:
                for key in ("selector_features", "old_logits", "returns"):
                    if key not in data.files:
                        raise KeyError(f"{path}: missing required key {key}")
                self.lengths.append(int(data["returns"].shape[0]))
        self.cumulative = np.cumsum(self.lengths)
        self.cache_path = None
        self.cache = None
        self.preloaded = None
        if preload:
            rows = {"selector_features": [], "old_logits": [], "returns": []}
            for path in self.paths:
                chunk = load_npz(path)
                for key in rows:
                    rows[key].append(np.asarray(chunk[key], dtype=np.float32))
            self.preloaded = {key: np.concatenate(value, axis=0) for key, value in rows.items()}
            size_mb = sum(value.nbytes for value in self.preloaded.values()) / (1024.0 * 1024.0)
            print(f"[train_selector_dataset] preloaded ranking arrays into RAM: {size_mb:.1f} MiB", flush=True)

    def __len__(self):
        return int(self.cumulative[-1])

    def _load_cached(self, path: Path):
        if self.cache_path != path:
            self.cache = load_npz(path)
            self.cache_path = path
        return self.cache

    def sample(self, batch_size: int, rng: np.random.Generator):
        indices = rng.integers(0, len(self), size=int(batch_size))
        if self.preloaded is not None:
            return {key: value[indices] for key, value in self.preloaded.items()}
        chunk_indices = np.searchsorted(self.cumulative, indices, side="right").astype(np.int64)
        parts = {"selector_features": [], "old_logits": [], "returns": []}
        for chunk_idx in np.unique(chunk_indices):
            mask = chunk_indices == chunk_idx
            global_chunk_indices = indices[mask]
            prev = 0 if chunk_idx == 0 else int(self.cumulative[int(chunk_idx) - 1])
            local_indices = (global_chunk_indices - prev).astype(np.int64)
            chunk = self._load_cached(self.paths[int(chunk_idx)])
            for key in parts:
                parts[key].append(np.asarray(chunk[key][local_indices], dtype=np.float32))
        return {key: np.concatenate(value, axis=0) for key, value in parts.items()}


def load_torch_planner(checkpoint_path: str | Path, anchor_path: str | None, device):
    from RSSMDiDrOnCarla.models.rssm_didr_planner import RSSMDiDrConfig, RSSMDiffusionDrivePlanner

    payload = torch.load(checkpoint_path, map_location=device)
    if "config" not in payload or "model" not in payload:
        raise KeyError(f"{checkpoint_path} is not a planner checkpoint with `config` and `model` keys")
    config_dict = dict(payload["config"])
    if anchor_path:
        config_dict["plan_anchor_path"] = str(anchor_path)
    config = RSSMDiDrConfig(**config_dict)
    model = RSSMDiffusionDrivePlanner(config).to(device)
    model.load_state_dict(payload["model"])
    print(
        f"[train_selector_dataset] loaded planner checkpoint={checkpoint_path} "
        f"epoch={int(payload.get('epoch', 0))} device={device} modes={config.num_modes} "
        f"poses={config.num_poses} latent_dim={config.latent_dim}",
        flush=True,
    )
    return model, config, int(payload.get("epoch", 0))


def freeze_except_selector(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("decoder.selector_head.")
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No decoder.selector_head parameters found.")
    return trainable


def selector_losses(new_logits, old_logits, returns, args):
    policy_temperature = max(float(args.policy_temperature), 1e-6)
    return_temperature = max(float(args.return_temperature), 1e-6)
    adv_eps = max(float(args.adv_eps), 1e-8)

    new_logp = torch.log_softmax(new_logits / policy_temperature, dim=-1)
    new_prob = torch.softmax(new_logits / policy_temperature, dim=-1)
    old_logp = torch.log_softmax(old_logits / policy_temperature, dim=-1).detach()
    old_prob = torch.softmax(old_logits / policy_temperature, dim=-1).detach()
    returns = returns.detach()

    rank_return = returns - returns.mean(dim=-1, keepdim=True)
    rank_return = rank_return / torch.clamp(returns.std(dim=-1, keepdim=True), min=adv_eps)
    target = torch.softmax(rank_return / return_temperature, dim=-1).detach()
    ranking_loss = -(target * new_logp).sum(dim=-1).mean()

    adv = returns - returns.mean(dim=-1, keepdim=True)
    adv = adv / torch.clamp(returns.std(dim=-1, keepdim=True), min=adv_eps)
    adv = adv.detach()

    rl_loss = -(new_prob * adv).sum(dim=-1).mean()
    kl_loss = (old_prob * (old_logp - new_logp)).sum(dim=-1).mean()
    entropy = -(new_prob * new_logp).sum(dim=-1).mean()
    loss = (
        float(args.ranking_weight) * ranking_loss
        + float(args.rl_weight) * rl_loss
        + float(args.kl_weight) * kl_loss
        - float(args.entropy_weight) * entropy
    )

    with torch.no_grad():
        selected_idx = torch.argmax(new_logits, dim=-1)
        old_idx = torch.argmax(old_logits, dim=-1)
        oracle_idx = torch.argmax(returns, dim=-1)
        batch_idx = torch.arange(returns.shape[0], device=returns.device)
        metrics = {
            "loss": loss.detach(),
            "ranking_loss": ranking_loss.detach(),
            "rl_loss": rl_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "entropy": entropy.detach(),
            "return_mean": returns.mean(),
            "return_std": returns.std(),
            "selected_return": returns[batch_idx, selected_idx].mean(),
            "old_selected_return": returns[batch_idx, old_idx].mean(),
            "oracle_return": returns.max(dim=-1).values.mean(),
            "rank_acc": (selected_idx == oracle_idx).float().mean(),
            "old_rank_acc": (old_idx == oracle_idx).float().mean(),
        }
    return loss, metrics


def batch_to_device(batch: Dict[str, np.ndarray], device):
    return {key: torch.from_numpy(value).to(device=device, dtype=torch.float32) for key, value in batch.items()}


def scalarize(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def save_checkpoint(path: Path, model, optimizer, config, iteration: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(iteration),
            "selector_ranking_iteration": int(iteration),
        },
        path,
    )


def log_message(iterator, message: str) -> None:
    if tqdm is not None and hasattr(iterator, "write"):
        iterator.write(message)
    else:
        print(message, flush=True)


def evaluate(model, dataset: RankingDataset, args, device, rng: np.random.Generator):
    model.eval()
    rows = []
    with torch.no_grad():
        for _ in range(int(args.eval_batches)):
            batch = batch_to_device(dataset.sample(args.batch_size, rng), device)
            new_logits = model.decoder.selector_head(batch["selector_features"]).squeeze(-1)
            _, metrics_t = selector_losses(new_logits, batch["old_logits"], batch["returns"], args)
            rows.append(scalarize(metrics_t))
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def main() -> None:
    args = parse_args()
    if torch is None:
        raise ModuleNotFoundError("This script requires PyTorch.") from TORCH_IMPORT_ERROR

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 1000003)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RankingDataset(args.dataset_dir, preload=not args.no_preload)
    print(
        f"[train_selector_dataset] loaded ranking dataset dir={args.dataset_dir} "
        f"files={len(dataset.paths)} samples={len(dataset)} preload={dataset.preloaded is not None}",
        flush=True,
    )
    model, planner_config, checkpoint_epoch = load_torch_planner(args.planner_checkpoint, args.anchor_path, device)
    trainable = freeze_except_selector(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    run_config = {
        "args": vars(args),
        "dataset_metadata": dataset.metadata,
        "dataset_samples": len(dataset),
        "planner_config": planner_config.to_dict(),
        "planner_checkpoint_epoch": checkpoint_epoch,
        "trainable_parameters": [name for name, param in model.named_parameters() if param.requires_grad],
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    history = []
    best_return = None
    iterator = range(1, int(args.iterations) + 1)
    if not args.no_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc="train-selector-ranking", dynamic_ncols=True)

    for iteration in iterator:
        model.train()
        batch = batch_to_device(dataset.sample(args.batch_size, rng), device)
        new_logits = model.decoder.selector_head(batch["selector_features"]).squeeze(-1)
        loss, metrics_t = selector_losses(new_logits, batch["old_logits"], batch["returns"], args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        metrics = {"iteration": int(iteration), **scalarize(metrics_t)}
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "loss": f"{metrics['loss']:.4f}",
                    "sel_ret": f"{metrics['selected_return']:.3f}",
                    "old_ret": f"{metrics['old_selected_return']:.3f}",
                    "acc": f"{metrics['rank_acc']:.3f}",
                }
            )

        if iteration == 1 or iteration % int(args.log_every) == 0:
            history.append({"train": metrics})
            log_message(
                iterator,
                "[train_selector_dataset] "
                + " ".join(f"{key}={value:.5f}" for key, value in metrics.items() if key != "iteration"),
            )
            (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if args.eval_every > 0 and iteration % int(args.eval_every) == 0:
            eval_metrics = evaluate(model, dataset, args, device, eval_rng)
            history.append({"iteration": int(iteration), "eval": eval_metrics})
            log_message(
                iterator,
                "[train_selector_dataset_eval] "
                + " ".join(f"{key}={value:.5f}" for key, value in eval_metrics.items()),
            )
            monitor = eval_metrics["selected_return"]
            if best_return is None or monitor > best_return:
                best_return = monitor
                save_checkpoint(output_dir / "best.pt", model, optimizer, planner_config, iteration)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        if args.save_every > 0 and iteration % int(args.save_every) == 0:
            save_checkpoint(output_dir / f"iteration_{iteration:06d}.pt", model, optimizer, planner_config, iteration)

    if best_return is None:
        save_checkpoint(output_dir / "best.pt", model, optimizer, planner_config, int(args.iterations))
    save_checkpoint(output_dir / "last.pt", model, optimizer, planner_config, int(args.iterations))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[train_selector_dataset] Wrote checkpoints under {output_dir}")


if __name__ == "__main__":
    main()
