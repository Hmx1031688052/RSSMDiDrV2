"""Dataset for RSSM latent conditioned DiffusionDrive planner training."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


REQUIRED_KEYS = ("expert_waypoints8", "trajectory")


def _load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def validate_planner_chunk(path: str | Path, condition_key: str = "rssm_latent") -> Dict[str, object]:
    """Validate one exported planner chunk."""

    chunk = _load_npz(path)
    missing = [key for key in (*REQUIRED_KEYS, condition_key) if key not in chunk]
    if missing:
        raise KeyError(f"{path}: missing required planner keys: {missing}")

    condition = np.asarray(chunk[condition_key], dtype=np.float32)
    waypoints = np.asarray(chunk["expert_waypoints8"], dtype=np.float32)
    trajectory = np.asarray(chunk["trajectory"], dtype=np.float32)

    if condition.ndim != 2:
        raise ValueError(f"{path}: expected {condition_key} [T, D], got {condition.shape}")
    if waypoints.ndim != 2 or waypoints.shape[-1] != 16:
        raise ValueError(f"{path}: expected expert_waypoints8 [T, 16], got {waypoints.shape}")
    if trajectory.ndim != 3 or trajectory.shape[1:] != (8, 3):
        raise ValueError(f"{path}: expected trajectory [T, 8, 3], got {trajectory.shape}")
    if not (len(condition) == len(waypoints) == len(trajectory)):
        raise ValueError(
            f"{path}: length mismatch {condition_key}={len(condition)} "
            f"waypoints={len(waypoints)} trajectory={len(trajectory)}"
        )
    if not np.isfinite(condition).all():
        raise ValueError(f"{path}: {condition_key} contains NaN or Inf")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{path}: trajectory contains NaN or Inf")

    return {
        "path": str(path),
        "length": int(len(condition)),
        "condition_key": str(condition_key),
        "condition_dim": int(condition.shape[-1]),
        "latent_dim": int(condition.shape[-1]),
        "trajectory_shape": tuple(trajectory.shape),
        "source": str(chunk.get("target_source", "ego_future")),
    }


def iter_planner_chunks(dataset_dir: str | Path) -> Iterable[Path]:
    dataset_dir = Path(dataset_dir)
    yield from sorted(dataset_dir.glob("*.npz"))


class RSSMPlannerDataset(Dataset):
    """Loads exported planner chunks with RSSM latent features and trajectory targets."""

    def __init__(
        self,
        dataset_dir: str | Path,
        chunk_paths: Optional[List[str | Path]] = None,
        condition_key: str = "rssm_latent",
    ):
        if torch is None:
            raise ModuleNotFoundError("RSSMPlannerDataset requires PyTorch. Install torch before training.")
        self.dataset_dir = Path(dataset_dir)
        self.condition_key = str(condition_key)
        if chunk_paths is None:
            self.chunk_paths = list(iter_planner_chunks(self.dataset_dir))
        else:
            self.chunk_paths = [Path(path) for path in chunk_paths]
        if not self.chunk_paths:
            raise FileNotFoundError(f"No planner `.npz` chunks found in: {self.dataset_dir}")

        self.summaries = [validate_planner_chunk(path, condition_key=self.condition_key) for path in self.chunk_paths]
        self.lengths = [int(summary["length"]) for summary in self.summaries]
        self.cumulative = np.cumsum(self.lengths).tolist()
        self.feature_dim = int(self.summaries[0]["condition_dim"])
        self.latent_dim = self.feature_dim

        for summary in self.summaries:
            if int(summary["condition_dim"]) != self.feature_dim:
                raise ValueError(
                    f"Mixed condition dims are not supported: expected {self.feature_dim}, "
                    f"got {summary['condition_dim']} in {summary['path']}"
                )

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def _locate(self, index: int) -> tuple[Path, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chunk_idx = bisect_right(self.cumulative, index)
        prev = 0 if chunk_idx == 0 else self.cumulative[chunk_idx - 1]
        return self.chunk_paths[chunk_idx], index - prev

    def __getitem__(self, index: int) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        path, local_idx = self._locate(index)
        chunk = _load_npz(path)

        condition = np.asarray(chunk[self.condition_key][local_idx], dtype=np.float32).reshape(-1)
        trajectory = np.asarray(chunk["trajectory"][local_idx], dtype=np.float32)
        waypoints = np.asarray(chunk["expert_waypoints8"][local_idx], dtype=np.float32)

        features = {
            self.condition_key: torch.from_numpy(condition),
            "condition": torch.from_numpy(condition),
            "expert_waypoints8": torch.from_numpy(waypoints),
        }
        targets = {
            "trajectory": torch.from_numpy(trajectory),
        }
        return features, targets


def split_chunk_paths(
    dataset_dir: str | Path,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[List[Path], List[Path]]:
    """Split exported chunks into train/val lists without mixing timesteps within a chunk."""

    paths = list(iter_planner_chunks(dataset_dir))
    if not paths:
        raise FileNotFoundError(f"No planner `.npz` chunks found in: {dataset_dir}")
    rng = np.random.default_rng(seed)
    order = np.arange(len(paths))
    rng.shuffle(order)
    val_count = int(round(len(paths) * val_fraction))
    if len(paths) > 1:
        val_count = max(1, min(val_count, len(paths) - 1))
    else:
        val_count = 0
    val_indices = set(order[:val_count].tolist())
    train = [path for idx, path in enumerate(paths) if idx not in val_indices]
    val = [path for idx, path in enumerate(paths) if idx in val_indices]
    return train, val
