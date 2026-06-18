"""Numpy dataset for JAX RSSM latent planner chunks."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


def _load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def iter_chunks(dataset_dir: str | Path) -> Iterable[Path]:
    yield from sorted(Path(dataset_dir).glob("*.npz"))


def validate_chunk(path: str | Path, condition_key: str = "rssm_latent") -> Dict[str, int]:
    chunk = _load_npz(path)
    missing = [key for key in (condition_key, "trajectory") if key not in chunk]
    if missing:
        raise KeyError(f"{path}: missing required keys {missing}")
    cond = np.asarray(chunk[condition_key], dtype=np.float32)
    traj = np.asarray(chunk["trajectory"], dtype=np.float32)
    if cond.ndim != 2:
        raise ValueError(f"{path}: expected {condition_key} [T,D], got {cond.shape}")
    if traj.ndim != 3 or traj.shape[1:] != (8, 3):
        raise ValueError(f"{path}: expected trajectory [T,8,3], got {traj.shape}")
    if len(cond) != len(traj):
        raise ValueError(f"{path}: length mismatch {len(cond)} vs {len(traj)}")
    return {"length": int(len(cond)), "feature_dim": int(cond.shape[-1])}


def split_chunk_paths(dataset_dir: str | Path, val_fraction: float = 0.05, seed: int = 0):
    paths = list(iter_chunks(dataset_dir))
    if not paths:
        raise FileNotFoundError(f"No planner .npz chunks found in {dataset_dir}")
    rng = np.random.default_rng(seed)
    order = np.arange(len(paths))
    rng.shuffle(order)
    val_count = int(round(len(paths) * float(val_fraction)))
    if len(paths) > 1:
        val_count = max(1, min(val_count, len(paths) - 1))
    else:
        val_count = 0
    val = set(order[:val_count].tolist())
    return [p for i, p in enumerate(paths) if i not in val], [p for i, p in enumerate(paths) if i in val]


class PlannerDataset:
    def __init__(self, dataset_dir: str | Path, chunk_paths=None, condition_key: str = "rssm_latent"):
        self.dataset_dir = Path(dataset_dir)
        self.condition_key = str(condition_key)
        self.chunk_paths = [Path(p) for p in chunk_paths] if chunk_paths is not None else list(iter_chunks(dataset_dir))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No planner chunks found in {dataset_dir}")
        summaries = [validate_chunk(p, self.condition_key) for p in self.chunk_paths]
        self.lengths = [s["length"] for s in summaries]
        self.cumulative = np.cumsum(self.lengths).tolist()
        self.feature_dim = int(summaries[0]["feature_dim"])
        for path, summary in zip(self.chunk_paths, summaries):
            if int(summary["feature_dim"]) != self.feature_dim:
                raise ValueError(f"{path}: mixed feature dims are not supported")

    def __len__(self):
        return int(self.cumulative[-1])

    def _locate(self, index: int):
        index = int(index) % len(self)
        chunk_idx = bisect_right(self.cumulative, index)
        prev = 0 if chunk_idx == 0 else self.cumulative[chunk_idx - 1]
        return self.chunk_paths[chunk_idx], index - prev

    def get(self, index: int):
        path, local = self._locate(index)
        chunk = _load_npz(path)
        cond = np.asarray(chunk[self.condition_key][local], dtype=np.float32).reshape(-1)
        traj = np.asarray(chunk["trajectory"][local], dtype=np.float32)
        return {self.condition_key: cond, "condition": cond, "trajectory": traj}

    def batches(self, batch_size: int, shuffle: bool = True, seed: int = 0, drop_last: bool = False):
        indices = np.arange(len(self))
        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            if drop_last and len(batch_idx) < int(batch_size):
                continue
            chunk_cache = {}
            rows = []
            for index in batch_idx:
                path, local = self._locate(int(index))
                if path not in chunk_cache:
                    chunk_cache[path] = _load_npz(path)
                chunk = chunk_cache[path]
                cond = np.asarray(chunk[self.condition_key][local], dtype=np.float32).reshape(-1)
                traj = np.asarray(chunk["trajectory"][local], dtype=np.float32)
                rows.append({self.condition_key: cond, "condition": cond, "trajectory": traj})
            keys = rows[0].keys()
            yield {key: np.stack([row[key] for row in rows], axis=0).astype(np.float32) for key in keys}

