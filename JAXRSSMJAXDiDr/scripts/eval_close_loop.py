"""Closed-loop CARLA eval for the pure JAX RSSM-conditioned DiDr planner.

This reuses the existing CARLA/RSSM/controller evaluation loop from
`RSSMDiDrOnCarla.scripts.eval_close_loop_rssm_didr` and swaps only the planner
backend from PyTorch to the JAX checkpoint format.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dreamerv3") not in sys.path:
    sys.path.insert(0, str(ROOT / "dreamerv3"))

import jax
import jax.numpy as jnp

from JAXRSSMJAXDiDr.models import load_checkpoint, predict
from RSSMDiDrOnCarla.scripts import eval_close_loop_rssm_didr as base


EVAL_TIMESTEP = 0
PLAN_INTERVAL_STEPS = 5


class _CudaShim:
    @staticmethod
    def is_available() -> bool:
        try:
            return bool(jax.devices("gpu"))
        except RuntimeError:
            return False


class _TorchShim:
    """Enough of torch's surface for the shared closed-loop main()."""

    cuda = _CudaShim()

    @staticmethod
    def manual_seed(seed: int) -> None:
        del seed

    @staticmethod
    def device(name: str) -> str:
        return str(name)


class JAXClosedLoopPlanner:
    def __init__(self, checkpoint_path: str | Path, anchor_path: Optional[str] = None):
        config, params, _, epoch = load_checkpoint(checkpoint_path)
        if anchor_path:
            anchors = np.load(anchor_path).astype(np.float32)
            expected = (int(config.num_modes), int(config.num_poses), 2)
            if tuple(anchors.shape) != expected:
                raise ValueError(f"Expected anchor override {expected}, got {anchors.shape}")
            params = dict(params)
            params["plan_anchor"] = jnp.asarray(anchors, dtype=jnp.float32)
            config.plan_anchor_path = str(anchor_path)
        if "selector_head" not in params:
            raise KeyError(f"Expected JAX planner params to contain selector_head, got keys {sorted(params.keys())}")
        self.config = config
        self.params = jax.tree_util.tree_map(jnp.asarray, params)
        self.epoch = int(epoch)
        self._plan_fns = {}

    def _plan_fn(self, timestep: int):
        timestep = int(timestep)
        if timestep not in self._plan_fns:
            config = self.config

            def _plan(params, condition):
                return predict(params, config, condition, timestep=timestep)

            self._plan_fns[timestep] = jax.jit(_plan)
        return self._plan_fns[timestep]

    def plan(self, condition: np.ndarray, timestep: int = 0):
        condition = np.asarray(condition, dtype=np.float32).reshape(1, -1)
        out = self._plan_fn(int(timestep))(self.params, jnp.asarray(condition))
        return jax.device_get(out)


def load_jax_planner(checkpoint_path: str | Path, anchor_path: Optional[str], device):
    del device
    planner = JAXClosedLoopPlanner(checkpoint_path, anchor_path=anchor_path)
    print(
        f"[jax_closed_loop] loaded planner epoch={planner.epoch} "
        f"condition_key={planner.config.condition_key} modes={planner.config.num_modes} "
        f"eval_timestep={EVAL_TIMESTEP} plan_interval_steps={PLAN_INTERVAL_STEPS}",
        flush=True,
    )
    return planner, planner.config


def plan_with_jax_model(
    model: JAXClosedLoopPlanner,
    condition: np.ndarray,
    device,
    eval_timestep: int = 0,
    top_modes: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del device
    timestep = EVAL_TIMESTEP if int(eval_timestep) == 0 else int(eval_timestep)
    out = model.plan(condition, timestep=timestep)
    poses_reg = np.asarray(out["poses_reg"][0], dtype=np.float32)
    poses_cls = np.asarray(out["poses_cls"][0], dtype=np.float32)
    selected_idx = int(np.argmax(poses_cls))
    selected = poses_reg[selected_idx]
    topk = min(max(int(top_modes), 1), poses_reg.shape[0])
    top_indices = np.argsort(-poses_cls)[:topk].astype(np.int32)
    top_scores = poses_cls[top_indices].astype(np.float32)
    top_modes_np = poses_reg[top_indices].astype(np.float32)
    return selected, poses_cls, top_modes_np, top_indices, top_scores


def consume_int_flag(argv: list[str], name: str, default: int) -> int:
    value = int(default)
    cleaned = [argv[0]]
    idx = 1
    flag = f"--{name}"
    prefix = f"{flag}="
    while idx < len(argv):
        item = argv[idx]
        if item == flag:
            if idx + 1 >= len(argv):
                raise ValueError(f"{flag} requires an integer value")
            value = int(argv[idx + 1])
            idx += 2
        elif item.startswith(prefix):
            value = int(item.split("=", 1)[1])
            idx += 1
        else:
            cleaned.append(item)
            idx += 1
    sys.argv[:] = cleaned
    return value


def patch_parse_args_for_jax_flags() -> None:
    original_parse_args = base.parse_args

    def parse_args():
        args, extra = original_parse_args()
        args.plan_interval_steps = int(PLAN_INTERVAL_STEPS)
        return args, extra

    base.parse_args = parse_args


def main() -> None:
    global EVAL_TIMESTEP, PLAN_INTERVAL_STEPS
    EVAL_TIMESTEP = consume_int_flag(sys.argv, "eval_timestep", 0)
    PLAN_INTERVAL_STEPS = consume_int_flag(sys.argv, "plan_interval_steps", 5)
    # The shared loop only uses torch for seeding/device selection. Provide a
    # tiny shim when PyTorch is absent so JAX eval does not depend on torch.
    if base.torch is None:
        base.torch = _TorchShim()
    base.TORCH_IMPORT_ERROR = None
    patch_parse_args_for_jax_flags()
    base.load_planner = load_jax_planner
    base.plan_with_model = plan_with_jax_model
    base.main()


if __name__ == "__main__":
    main()
