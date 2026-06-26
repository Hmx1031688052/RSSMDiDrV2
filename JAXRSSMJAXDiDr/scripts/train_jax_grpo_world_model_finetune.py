"""World-model GRPO fine-tuning for JAX RSSM + JAX DiDr.

This script alternates online replay collection with RSSM/planner training.
Planner training first applies a DiffusionDriveV2-style diffusion log-prob
GRPO update to the trajectory body, then runs a small selector-only update.
The reward source is the frozen online RSSM target model rather than NAVSIM PDM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from JAXRSSMJAXDiDr.models import save_checkpoint
from JAXRSSMJAXDiDr.models.jax_didr_planner import (
    recompute_diffusion_chain_logprob,
    sample_diffusion_chain_with_logprob,
)
from JAXRSSMJAXDiDr.scripts.train_jax_stable_online_finetune import (
    MixedReplaySampler,
    StableOnlineTrainer,
    concatenate_mixed_replay_batches,
    _skip_sidecar_field,
    load_npz,
    parse_args as stable_parse_args,
    planner_modes_logits,
    poses_xy_meters,
    replace_selector_params,
    replay_paths,
    scalarize,
)


GRPO_TRAJECTORY_BODY_KEYS = {
    "anchor_encoder",
    "time_encoder",
    "latent_encoder",
    "decoder_layers",
    "delta_head",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    grpo_parser = argparse.ArgumentParser(add_help=False)
    grpo_parser.add_argument(
        "--online_schedule",
        choices=("alternate_collect_train", "every_outer", "train_only"),
        default="alternate_collect_train",
    )
    grpo_parser.add_argument(
        "--grpo_trainable",
        choices=("trajectory_body", "all_except_plan_anchor"),
        default="trajectory_body",
    )
    grpo_parser.add_argument("--grpo_freeze_selector", action=argparse.BooleanOptionalAction, default=True)
    grpo_parser.add_argument("--grpo_updates", type=int, default=200)
    grpo_parser.add_argument("--grpo_lr", type=float, default=1e-5)
    grpo_parser.add_argument("--grpo_groups", type=int, default=4)
    grpo_parser.add_argument("--grpo_step_num", type=int, default=10)
    grpo_parser.add_argument("--grpo_trunc_timestep", type=int, default=8)
    grpo_parser.add_argument("--grpo_eta", type=float, default=1.0)
    grpo_parser.add_argument("--grpo_discount_base", type=float, default=0.8)
    grpo_parser.add_argument("--grpo_adv_eps", type=float, default=1e-4)
    grpo_parser.add_argument("--grpo_il_weight_positive", type=float, default=0.1)
    grpo_parser.add_argument("--grpo_il_weight_no_positive", type=float, default=1.0)
    grpo_parser.add_argument("--grpo_unsafe_adv", type=float, default=-1.0)
    grpo_parser.add_argument("--grpo_grad_clip", type=float, default=1.0)
    grpo_parser.add_argument("--grpo_min_logprob_std", type=float, default=0.1)
    grpo_parser.add_argument("--grpo_additive_noise", action="store_true")
    grpo_parser.add_argument("--selector_after_grpo_updates", type=int, default=20)
    grpo_parser.add_argument("--selector_after_grpo_lr", type=float, default=2e-6)
    grpo_parser.add_argument("--selector_after_grpo_ranking_weight", type=float, default=1.0)
    grpo_parser.add_argument("--selector_after_grpo_kl_weight", type=float, default=0.2)
    grpo_parser.add_argument("--selector_after_grpo_entropy_weight", type=float, default=0.0)
    grpo_args, stable_argv = grpo_parser.parse_known_args()

    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], *stable_argv]
    try:
        args, extra = stable_parse_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(grpo_args).items():
        setattr(args, key, value)
    return args, extra


def _target_xy_from_array(value: np.ndarray) -> Optional[np.ndarray]:
    value = np.asarray(value)
    if value.ndim >= 3 and value.shape[-2:] == (8, 3):
        return value[..., :2].astype(np.float32)
    if value.ndim >= 3 and value.shape[-2:] == (8, 2):
        return value.astype(np.float32)
    if value.ndim >= 2 and value.shape[-1] == 16:
        return value.reshape(value.shape[:-1] + (8, 2)).astype(np.float32)
    return None


def _zero_tree(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _filter_grpo_trainable_tree(tree, trainable: str, freeze_selector: bool):
    if not isinstance(tree, dict):
        return tree
    filtered = dict(tree)
    if trainable == "trajectory_body":
        for key, value in list(filtered.items()):
            if key not in GRPO_TRAJECTORY_BODY_KEYS:
                filtered[key] = _zero_tree(value)
        return filtered

    if "plan_anchor" in filtered:
        filtered["plan_anchor"] = _zero_tree(filtered["plan_anchor"])
    if bool(freeze_selector) and "selector_head" in filtered:
        filtered["selector_head"] = _zero_tree(filtered["selector_head"])
    return filtered


class GRPOReplaySequenceDataset:
    """Replay sequence sampler that normalizes available targets to grpo_target_xy."""

    def __init__(
        self,
        replay_dirs: Iterable[str | Path],
        batch_length: int,
        allowed_keys: Optional[set[str]] = None,
        allow_planner_waypoints_target: bool = True,
    ):
        self.replay_dirs = [Path(path) for path in replay_dirs if path is not None]
        self.batch_length = int(batch_length)
        self.allowed_keys = set(allowed_keys) if allowed_keys is not None else None
        self.allow_planner_waypoints_target = bool(allow_planner_waypoints_target)
        self.paths: list[Path] = []
        self.lengths: list[int] = []
        for directory in self.replay_dirs:
            for path in replay_paths(directory):
                chunk = load_npz(path)
                if "action" not in chunk:
                    continue
                length = int(len(chunk["action"]))
                if length >= self.batch_length:
                    self.paths.append(path)
                    self.lengths.append(length - self.batch_length + 1)
        if not self.paths:
            raise FileNotFoundError(f"No usable replay chunks found in {self.replay_dirs}")
        self.cumulative = np.cumsum(self.lengths)

    def __len__(self) -> int:
        return int(self.cumulative[-1])

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        rows = [self._get(int(rng.integers(0, len(self)))) for _ in range(int(batch_size))]
        keys = sorted(set.intersection(*(set(row.keys()) for row in rows)))
        return {key: np.stack([row[key] for row in rows], axis=0) for key in keys}

    def _get(self, index: int) -> Dict[str, np.ndarray]:
        chunk_idx = int(np.searchsorted(self.cumulative, index, side="right"))
        prev = 0 if chunk_idx == 0 else int(self.cumulative[chunk_idx - 1])
        start = index - prev
        end = start + self.batch_length
        chunk = load_npz(self.paths[chunk_idx])
        keys = [
            key
            for key, value in chunk.items()
            if not _skip_sidecar_field(key) and np.asarray(value).ndim > 0 and len(value) >= end
        ]
        if self.allowed_keys is not None:
            keys = [key for key in keys if key in self.allowed_keys or key == "executed_control"]
        row = {key: np.asarray(chunk[key][start:end]) for key in keys}
        if "executed_control" in row:
            row["action"] = row["executed_control"]
            if self.allowed_keys is not None and "executed_control" not in self.allowed_keys:
                row.pop("executed_control", None)

        target_keys = ["expert_waypoints8", "trajectory"]
        if self.allow_planner_waypoints_target:
            target_keys.append("planner_waypoints8")
        for target_key in target_keys:
            if target_key in chunk and len(chunk[target_key]) >= end:
                target_xy = _target_xy_from_array(np.asarray(chunk[target_key][start:end]))
                if target_xy is not None:
                    row["grpo_target_xy"] = target_xy
                    break
        row.setdefault(
            "grpo_target_xy",
            np.full((self.batch_length, 8, 2), np.nan, dtype=np.float32),
        )

        row.setdefault("is_first", np.zeros((self.batch_length,), dtype=bool))
        row.setdefault("is_last", np.zeros((self.batch_length,), dtype=bool))
        row.setdefault("is_terminal", np.zeros((self.batch_length,), dtype=bool))
        row.setdefault("reward", np.zeros((self.batch_length,), dtype=np.float32))
        for key in ("destination_reached", "is_success", "out_of_lane", "time_exceeded", "is_collision"):
            row.setdefault(key, np.zeros((self.batch_length,), dtype=bool))
        return row


class GRPOMixedReplaySampler:
    def __init__(
        self,
        offline_dir: str | Path,
        online_dir: str | Path,
        batch_length: int,
        allowed_keys: set[str],
        offline_ratio: float,
    ):
        self.offline = GRPOReplaySequenceDataset(
            [offline_dir],
            batch_length,
            allowed_keys,
            allow_planner_waypoints_target=True,
        )
        self.online = None
        if replay_paths(online_dir):
            self.online = GRPOReplaySequenceDataset(
                [online_dir],
                batch_length,
                allowed_keys,
                allow_planner_waypoints_target=False,
            )
        self.offline_ratio = float(np.clip(offline_ratio, 0.0, 1.0))

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if self.online is None:
            return self.offline.sample(batch_size, rng)
        offline_count = int(round(int(batch_size) * self.offline_ratio))
        online_count = int(batch_size) - offline_count
        if online_count <= 0:
            return self.offline.sample(batch_size, rng)
        if offline_count <= 0:
            return self.online.sample(batch_size, rng)
        offline_batch = self.offline.sample(offline_count, rng)
        online_batch = self.online.sample(online_count, rng)
        return concatenate_mixed_replay_batches(offline_batch, online_batch)


class GRPOWorldModelTrainer(StableOnlineTrainer):
    def __init__(self, args: argparse.Namespace, extra: list[str]):
        super().__init__(args, extra)
        self.rssm_data_keys = set(self.obs_space.keys()) | {
            "action",
            "destination_reached",
            "is_success",
            "out_of_lane",
            "time_exceeded",
            "is_collision",
        }
        self.allowed_data_keys = set(self.rssm_data_keys) | {
            "trajectory",
            "expert_waypoints8",
            "planner_waypoints8",
            "grpo_target_xy",
        }
        self.grpo_opt = optax.chain(
            optax.clip_by_global_norm(float(args.grpo_grad_clip)),
            optax.adamw(float(args.grpo_lr), weight_decay=float(args.selector_weight_decay)),
        )
        self.selector_light_opt = optax.chain(
            optax.clip_by_global_norm(float(args.grad_clip)),
            optax.adamw(float(args.selector_after_grpo_lr), weight_decay=float(args.selector_weight_decay)),
        )
        self.grpo_state = self.grpo_opt.init(self.planner_params)
        self.selector_light_state = self.selector_light_opt.init(self.selector_params)
        self.grpo_rng = jax.random.PRNGKey(int(args.seed) + 1701)
        self._sample_chain = jax.jit(self._sample_chain_impl)
        self._grpo_step = jax.jit(self._grpo_step_impl)
        self._selector_light_step = jax.jit(self._selector_light_step_impl)

    def make_sampler(self) -> MixedReplaySampler:
        return MixedReplaySampler(
            self.args.offline_replay_dir,
            self.online_replay_dir,
            self.args.batch_length,
            self.rssm_data_keys,
            self.args.offline_ratio,
        )

    def make_grpo_sampler(self) -> GRPOMixedReplaySampler:
        return GRPOMixedReplaySampler(
            self.args.offline_replay_dir,
            self.online_replay_dir,
            self.args.batch_length,
            self.allowed_data_keys,
            self.args.offline_ratio,
        )

    def _sample_chain_impl(self, params, feat, rng):
        return sample_diffusion_chain_with_logprob(
            params,
            self.planner_config,
            feat,
            rng,
            groups=int(self.args.grpo_groups),
            step_num=int(self.args.grpo_step_num),
            trunc_timestep=int(self.args.grpo_trunc_timestep),
            eta=float(self.args.grpo_eta),
            min_logprob_std=float(self.args.grpo_min_logprob_std),
            multiplicative_noise=not bool(self.args.grpo_additive_noise),
        )

    def _rssm_batch(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        return {key: value for key, value in batch.items() if key in self.rssm_data_keys}

    def _target_from_batch(self, batch: Dict[str, jnp.ndarray], batch_size: int):
        if "grpo_target_xy" not in batch:
            target_xy = jnp.zeros((batch_size, int(self.planner_config.num_poses), 2), jnp.float32)
            target_valid = jnp.zeros((batch_size,), bool)
            return target_xy, target_valid
        target_xy = batch["grpo_target_xy"]
        if target_xy.ndim == 4:
            target_xy = target_xy[:, -1]
        target_xy = target_xy[..., :2].astype(jnp.float32)
        target_valid = jnp.all(jnp.isfinite(target_xy), axis=(-2, -1))
        return target_xy, target_valid

    def _trajectory_safety_mask(self, xy):
        finite = jnp.all(jnp.isfinite(xy), axis=(-2, -1))
        lateral_ok = jnp.max(jnp.abs(xy[..., 1]), axis=-1) <= float(self.args.max_abs_y)
        step_dist = jnp.linalg.norm(jnp.diff(xy, axis=-2), axis=-1)
        step_ok = jnp.max(step_dist, axis=-1) <= float(self.args.max_step_distance)
        forward_count = jnp.sum(xy[..., 0] >= float(self.args.min_forward_x), axis=-1)
        return finite & lateral_ok & step_ok & (forward_count >= int(self.args.min_valid_points))

    def _grpo_step_impl(
        self,
        params,
        opt_state,
        feat,
        chain,
        rewards,
        target_rewards,
        target_xy,
        target_valid,
        safety_mask,
    ):
        args = self.args

        def objective(current_params):
            safe_target_xy = jnp.where(target_valid[:, None, None], target_xy, 0.0)
            out = recompute_diffusion_chain_logprob(
                current_params,
                self.planner_config,
                feat,
                jax.lax.stop_gradient(chain),
                groups=int(args.grpo_groups),
                step_num=int(args.grpo_step_num),
                trunc_timestep=int(args.grpo_trunc_timestep),
                eta=float(args.grpo_eta),
                min_logprob_std=float(args.grpo_min_logprob_std),
                multiplicative_noise=not bool(args.grpo_additive_noise),
            )
            log_probs = out["log_probs"]
            pred_xy_steps = out["pred_xy_steps"]
            reward_group = jax.lax.stop_gradient(rewards)
            safety = jax.lax.stop_gradient(safety_mask)

            mean_grouped = reward_group.mean(axis=1, keepdims=True)
            std_grouped = reward_group.std(axis=1, keepdims=True)
            advantages = (reward_group - mean_grouped) / (std_grouped + float(args.grpo_adv_eps))

            gt_mask = reward_group > (target_rewards[:, None, None] - 1e-6)
            adv_with_gt = jnp.maximum(advantages, 0.0) * gt_mask.astype(advantages.dtype)
            advantages = jnp.where(target_valid[:, None, None], adv_with_gt, 0.0)
            advantages = jnp.where(target_valid[:, None, None] & safety, advantages, 0.0)
            advantages = jnp.where(target_valid[:, None, None] & (~safety), float(args.grpo_unsafe_adv), advantages)
            advantages = jnp.clip(advantages, -float(args.adv_clip), float(args.adv_clip))

            discounts = jnp.asarray(
                [float(args.grpo_discount_base) ** (int(args.grpo_step_num) - idx - 1) for idx in range(int(args.grpo_step_num))],
                dtype=log_probs.dtype,
            )
            advantages_steps = jax.lax.stop_gradient(advantages[..., None] * discounts)
            per_token_loss = -jnp.exp(log_probs - jax.lax.stop_gradient(log_probs)) * advantages_steps
            token_mask = jnp.abs(advantages_steps) > 1e-8
            token_count = jnp.maximum(token_mask.sum(axis=(1, 2)), 1)
            rl_loss_per_step = (per_token_loss * token_mask).sum(axis=(1, 2)) / token_count
            rl_loss_b = rl_loss_per_step.mean(axis=-1)
            rl_loss = rl_loss_b.mean()

            pred_sign = jnp.asarray([float(args.plan_x_sign), float(args.plan_y_sign)], dtype=pred_xy_steps.dtype)
            signed_pred_xy = pred_xy_steps * pred_sign.reshape((1, 1, 1, 1, 2, 1))
            il_per_batch = jnp.abs(signed_pred_xy - safe_target_xy[:, None, None, :, :, None]).mean(axis=(1, 2, 3, 4, 5))
            has_positive = (advantages_steps > 0.0).any(axis=(1, 2, 3))
            il_weight = jnp.where(
                has_positive,
                float(args.grpo_il_weight_positive),
                float(args.grpo_il_weight_no_positive),
            )
            il_weight = il_weight * target_valid.astype(il_weight.dtype)
            valid_count = jnp.maximum(target_valid.astype(il_weight.dtype).sum(), 1.0)
            il_loss = (il_per_batch * il_weight).sum() / valid_count
            loss = rl_loss + il_loss

            positive = advantages > 0.0
            positive_count = jnp.maximum(positive.sum(), 1)
            reward_positive_mean = jnp.where(
                positive.any(),
                (reward_group * positive).sum() / positive_count,
                reward_group.mean(),
            )
            metrics = {
                "grpo_loss": loss,
                "rl_loss": rl_loss,
                "il_loss": il_loss,
                "reward_mean": reward_group.mean(),
                "reward_positive_mean": reward_positive_mean,
                "positive_ratio": positive.astype(jnp.float32).mean(),
                "unsafe_ratio": (~safety).astype(jnp.float32).mean(),
                "advantage_mean": advantages.mean(),
                "advantage_std": advantages.std(),
                "logprob_mean": log_probs.mean(),
                "target_used": target_valid.astype(jnp.float32).mean(),
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params)
        grads = _filter_grpo_trainable_tree(grads, args.grpo_trainable, bool(args.grpo_freeze_selector))
        grads_finite = jnp.stack(
            [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads)]
        ).all()
        finite_step = jnp.isfinite(loss) & grads_finite
        grads = jax.tree_util.tree_map(lambda grad: jnp.where(finite_step, grad, jnp.zeros_like(grad)), grads)
        updates, opt_state = self.grpo_opt.update(grads, opt_state, params)
        updates = _filter_grpo_trainable_tree(updates, args.grpo_trainable, bool(args.grpo_freeze_selector))
        updates = jax.tree_util.tree_map(lambda update: jnp.where(finite_step, update, jnp.zeros_like(update)), updates)
        params = optax.apply_updates(params, updates)
        metrics = dict(metrics)
        metrics["grpo_loss"] = loss
        metrics["finite_step"] = finite_step.astype(jnp.float32)
        return params, opt_state, metrics

    def update_grpo(self, outer_idx: int) -> dict:
        sampler = self.make_grpo_sampler()
        metrics_last = {}
        for update in range(1, int(self.args.grpo_updates) + 1):
            batch_np = sampler.sample(int(self.args.batch_size), self.rng_np)
            batch = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.device), batch_np)
            rssm_batch = self._rssm_batch(batch)
            state, speed, feat, neighbors = self.start_from_target(rssm_batch)
            target_xy, target_valid = self._target_from_batch(batch, int(feat.shape[0]))

            self.grpo_rng, sample_rng = jax.random.split(self.grpo_rng)
            sampled = self._sample_chain(self.planner_params, feat, sample_rng)
            final_xy = sampled["final_xy"]
            batch_size, groups, modes = final_xy.shape[:3]
            flat_xy = final_xy.reshape((batch_size, groups * modes, final_xy.shape[-2], 2))
            rewards = self.returns_from_target(state, speed, neighbors, flat_xy).reshape((batch_size, groups, modes))
            safe_target_xy = jnp.where(target_valid[:, None, None], target_xy, 0.0)
            target_rewards = self.returns_from_target(state, speed, neighbors, safe_target_xy[:, None])[:, 0]
            safety_mask = self._trajectory_safety_mask(final_xy)

            self.planner_params, self.grpo_state, metrics = self._grpo_step(
                self.planner_params,
                self.grpo_state,
                feat,
                sampled["chain"],
                jax.lax.stop_gradient(rewards),
                jax.lax.stop_gradient(target_rewards),
                target_xy,
                target_valid,
                jax.lax.stop_gradient(safety_mask),
            )
            metrics_last = scalarize(metrics)
            if update == 1 or update % int(self.args.log_every) == 0:
                print(
                    "[grpo] "
                    f"outer={outer_idx} update={update} "
                    + " ".join(f"{k}={v:.5f}" for k, v in metrics_last.items()),
                    flush=True,
                )
        return metrics_last

    def _selector_light_step_impl(self, selector_params, opt_state, params, ref_params, feat, returns):
        args = self.args

        def objective(current_selector_params):
            current_params = replace_selector_params(params, current_selector_params)
            _, logits = planner_modes_logits(
                current_params,
                self.planner_config,
                feat,
                int(args.eval_timestep),
            )
            _, ref_logits = planner_modes_logits(
                ref_params,
                self.planner_config,
                feat,
                int(args.eval_timestep),
            )
            temperature = max(float(args.policy_temperature), 1e-6)
            logp = jax.nn.log_softmax(logits / temperature, axis=-1)
            prob = jax.nn.softmax(logits / temperature, axis=-1)
            ref_logp = jax.nn.log_softmax(jax.lax.stop_gradient(ref_logits) / temperature, axis=-1)
            ref_prob = jax.nn.softmax(jax.lax.stop_gradient(ref_logits) / temperature, axis=-1)

            adv = returns - returns.mean(axis=-1, keepdims=True)
            adv = adv / jnp.maximum(returns.std(axis=-1, keepdims=True), float(args.adv_eps))
            adv = jnp.clip(adv, -float(args.adv_clip), float(args.adv_clip))
            adv = jax.lax.stop_gradient(adv)
            ranking_loss = -(adv * logp).sum(axis=-1).mean()
            kl_loss = (ref_prob * (ref_logp - logp)).sum(axis=-1).mean()
            entropy = -(prob * logp).sum(axis=-1).mean()
            loss = (
                float(args.selector_after_grpo_ranking_weight) * ranking_loss
                + float(args.selector_after_grpo_kl_weight) * kl_loss
                - float(args.selector_after_grpo_entropy_weight) * entropy
            )
            selected = jnp.argmax(logits, axis=-1)
            oracle = jnp.argmax(returns, axis=-1)
            batch_idx = jnp.arange(returns.shape[0])
            metrics = {
                "selector_loss": loss,
                "ranking_loss": ranking_loss,
                "kl_loss": kl_loss,
                "entropy": entropy,
                "return_mean": returns.mean(),
                "return_std": returns.std(),
                "selected_return": returns[batch_idx, selected].mean(),
                "oracle_return": returns.max(axis=-1).mean(),
                "rank_acc": (selected == oracle).astype(jnp.float32).mean(),
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(selector_params)
        grads_finite = jnp.stack(
            [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(grads)]
        ).all()
        finite_step = jnp.isfinite(loss) & grads_finite
        grads = jax.tree_util.tree_map(lambda grad: jnp.where(finite_step, grad, jnp.zeros_like(grad)), grads)
        updates, opt_state = self.selector_light_opt.update(grads, opt_state, selector_params)
        updates = jax.tree_util.tree_map(lambda update: jnp.where(finite_step, update, jnp.zeros_like(update)), updates)
        selector_params = optax.apply_updates(selector_params, updates)
        metrics = dict(metrics)
        metrics["selector_loss"] = loss
        metrics["finite_step"] = finite_step.astype(jnp.float32)
        return selector_params, opt_state, metrics

    def update_selector_light(self, outer_idx: int) -> dict:
        updates_total = int(self.args.selector_after_grpo_updates)
        if updates_total <= 0:
            return {"updates": 0}
        sampler = self.make_sampler()
        ref_params = jax.tree_util.tree_map(jax.lax.stop_gradient, self.planner_params)
        metrics_last = {}
        for update in range(1, updates_total + 1):
            batch = sampler.sample(int(self.args.batch_size), self.rng_np)
            batch = jax.tree_util.tree_map(lambda x: jax.device_put(x, self.device), batch)
            state, speed, feat, neighbors = self.start_from_target(batch)
            poses_reg, _ = planner_modes_logits(
                self.planner_params,
                self.planner_config,
                feat,
                int(self.args.eval_timestep),
            )
            modes_xy = jax.lax.stop_gradient(poses_xy_meters(poses_reg, self.args))
            returns = jax.lax.stop_gradient(self.returns_from_target(state, speed, neighbors, modes_xy))
            self.selector_params, self.selector_light_state, metrics = self._selector_light_step(
                self.selector_params,
                self.selector_light_state,
                self.planner_params,
                ref_params,
                feat,
                returns,
            )
            self.planner_params = replace_selector_params(self.planner_params, self.selector_params)
            metrics_last = scalarize(metrics)
            if update == 1 or update % int(self.args.log_every) == 0:
                print(
                    "[selector_light] "
                    f"outer={outer_idx} update={update} "
                    + " ".join(f"{k}={v:.5f}" for k, v in metrics_last.items()),
                    flush=True,
                )
        metrics_last["updates"] = float(updates_total)
        return metrics_last

    def save_planner(self, outer_idx: int) -> None:
        opt_state = {"grpo": self.grpo_state, "selector_light": self.selector_light_state}
        save_checkpoint(
            self.output_dir / "planner_grpo_online.pkl.gz",
            self.planner_config,
            self.planner_params,
            opt_state,
            outer_idx,
        )
        if int(self.args.save_every_outer) > 0 and outer_idx % int(self.args.save_every_outer) == 0:
            save_checkpoint(
                self.output_dir / f"planner_grpo_online_outer_{outer_idx:04d}.pkl.gz",
                self.planner_config,
                self.planner_params,
                opt_state,
                outer_idx,
            )

    def run(self) -> None:
        run_config = {
            "args": vars(self.args),
            "extra": self.extra,
            "planner_checkpoint_epoch": self.planner_epoch,
            "online_schedule": self.args.online_schedule,
            "trainable": self.args.grpo_trainable,
            "rssm_update": "online_mixed_replay",
            "planner_update": "diffusiondrivev2_style_grpo_then_light_selector",
        }
        (self.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

        for outer in range(1, int(self.args.outer_iterations) + 1):
            if self.args.online_schedule == "alternate_collect_train":
                phase = "collect" if outer % 2 == 1 else "train"
            elif self.args.online_schedule == "every_outer":
                phase = "collect_train"
            else:
                phase = "train"

            collect_metrics = {"episodes": 0, "steps": 0, "mean_return": 0.0}
            wm_metrics = {}
            grpo_metrics = {}
            selector_metrics = {}

            if phase in {"collect", "collect_train"}:
                print(f"[grpo_online] outer={outer} phase={phase} collect", flush=True)
                collect_metrics = self.collect(outer)

            if phase in {"train", "collect_train"}:
                print(f"[grpo_online] outer={outer} phase={phase} update_rssm", flush=True)
                wm_metrics = self.update_rssm(outer)
                print(f"[grpo_online] outer={outer} phase={phase} update_grpo", flush=True)
                grpo_metrics = self.update_grpo(outer)
                print(f"[grpo_online] outer={outer} phase={phase} update_selector_light", flush=True)
                selector_metrics = self.update_selector_light(outer)
                self.save_planner(outer)

            row = {
                "outer": outer,
                "phase": phase,
                "collect": collect_metrics,
                "wm": wm_metrics,
                "grpo": grpo_metrics,
                "selector_light": selector_metrics,
            }
            self.history.append(row)
            (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
            print(
                f"[grpo_online] outer={outer} phase={phase} "
                f"mean_return={collect_metrics.get('mean_return', 0.0):.3f} "
                f"wm_loss={wm_metrics.get('stable_online_wm_loss', float('nan')):.6f} "
                f"reward_mean={grpo_metrics.get('reward_mean', float('nan')):.5f} "
                f"selector_rank_acc={selector_metrics.get('rank_acc', float('nan')):.5f}",
                flush=True,
            )

        self.rssm_online_ckpt.save()
        self.save_planner(int(self.args.outer_iterations))
        try:
            self.env.close()
        except Exception:
            pass
        print(f"[grpo_online] Wrote outputs under {self.output_dir}", flush=True)


def main() -> None:
    args, extra = parse_args()
    if args.jax_platform:
        jax.config.update("jax_platform_name", args.jax_platform)
    trainer = GRPOWorldModelTrainer(args, extra)
    trainer.run()


if __name__ == "__main__":
    main()
