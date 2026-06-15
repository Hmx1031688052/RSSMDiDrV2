"""Dreamer-style online fine-tuning for the JAX DiffusionDrive planner.

This keeps the existing `dreamerv3/` RSSM/world model in JAX and ports only the
DiffusionDrive side to JAX:

  DreamerV3 RSSM posterior/start state
  -> JAX DiDr soft waypoint actor
  -> JAX differentiable PID pure-pursuit controller
  -> DreamerV3 RSSM imagination
  -> DreamerV3 reward/continue heads + JAX critic lambda return
  -> actor/critic updates

The world model is intentionally not reimplemented here. Use the existing
`RSSMDiDrOnCarla.scripts.train_offline_rssm` and Dreamer checkpoints for RSSM
training; this script fine-tunes the JAX planner and JAX critic against that
RSSM.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import sys
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from JAXRSSMJAXDiDr.models import (
    apply_plan_sign,
    critic_loss,
    critic_value,
    differentiable_pidpp,
    init_critic,
    lambda_return,
    load_checkpoint,
    normalized_acc_to_phys,
    predict,
    save_checkpoint,
)
from JAXRSSMJAXDiDr.models.jax_didr_planner import freeze_plan_anchor_updates, soft_select


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dreamerv3") not in sys.path:
    sys.path.insert(0, str(ROOT / "dreamerv3"))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline_replay_dir", required=True)
    parser.add_argument("--online_replay_dir", default=None)
    parser.add_argument("--rssm_checkpoint", required=True)
    parser.add_argument("--planner_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", default="carla_roundabout")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imag_horizon", type=int, default=15)
    parser.add_argument("--actor_updates", type=int, default=1)
    parser.add_argument("--critic_updates", type=int, default=1)
    parser.add_argument("--actor_lr", type=float, default=3e-5)
    parser.add_argument("--critic_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--bc_weight", type=float, default=0.1)
    parser.add_argument("--wp_smooth_weight", type=float, default=0.05)
    parser.add_argument("--ctrl_smooth_weight", type=float, default=0.05)
    parser.add_argument("--plan_x_sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--plan_y_sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--jax_platform", default=None, choices=("cpu", "gpu", "tpu"))
    parser.add_argument("--save_every", type=int, default=100)
    return parser.parse_known_args()


def iter_replay_paths(directory: str | Path):
    yield from sorted(Path(directory).glob("*.npz"))


def load_replay_chunk(path: str | Path):
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


class ReplaySequenceDataset:
    """Small numpy replay sequence loader with Dreamer row semantics."""

    def __init__(self, replay_dirs, batch_length: int):
        self.replay_dirs = [Path(replay_dirs)] if isinstance(replay_dirs, (str, Path)) else [Path(p) for p in replay_dirs]
        self.batch_length = int(batch_length)
        self.paths = []
        self.lengths = []
        for directory in self.replay_dirs:
            if directory is None or not directory.exists():
                continue
            for path in iter_replay_paths(directory):
                chunk = load_replay_chunk(path)
                if "action" not in chunk or "reward" not in chunk:
                    continue
                length = int(len(chunk["action"]))
                if length >= self.batch_length:
                    self.paths.append(path)
                    self.lengths.append(length - self.batch_length + 1)
        if not self.paths:
            raise FileNotFoundError(f"No usable replay chunks in {self.replay_dirs}")
        self.cumulative = np.cumsum(self.lengths)

    def __len__(self):
        return int(self.cumulative[-1])

    def sample(self, batch_size: int, rng: np.random.Generator):
        rows = [self._get(int(rng.integers(0, len(self)))) for _ in range(int(batch_size))]
        keys = sorted(set.intersection(*(set(row.keys()) for row in rows)))
        return {key: np.stack([row[key] for row in rows], axis=0) for key in keys}

    def _get(self, index: int):
        chunk_idx = int(np.searchsorted(self.cumulative, index, side="right"))
        prev = 0 if chunk_idx == 0 else int(self.cumulative[chunk_idx - 1])
        start = index - prev
        end = start + self.batch_length
        chunk = load_replay_chunk(self.paths[chunk_idx])
        keys = [k for k, v in chunk.items() if np.asarray(v).ndim > 0 and len(v) >= end]
        row = {key: np.asarray(chunk[key][start:end]) for key in keys}
        if "executed_control" in row:
            row["action"] = row["executed_control"]
        row.setdefault("is_first", np.zeros((self.batch_length,), dtype=bool))
        row.setdefault("is_terminal", np.zeros((self.batch_length,), dtype=bool))
        row.setdefault("reward", np.zeros((self.batch_length,), dtype=np.float32))
        return row


def scalarize(metrics):
    return {k: float(np.asarray(v)) for k, v in metrics.items() if np.asarray(v).shape == ()}


def save_tree_checkpoint(path: str | Path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = jax.device_get(payload)
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f)


def flatten_state_feat(state):
    deter = state["deter"].reshape((state["deter"].shape[0], -1))
    stoch = state["stoch"].reshape((state["stoch"].shape[0], -1))
    return jnp.concatenate([deter, stoch], axis=-1)


def stack_time(items):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *items)


def waypoint_smoothness(wp):
    return jnp.asarray(0.0, jnp.float32) if wp.shape[-2] < 3 else jnp.linalg.norm(jnp.diff(wp, n=2, axis=-2), axis=-1).mean()


def control_smoothness(actions):
    return jnp.asarray(0.0, jnp.float32) if actions.shape[0] < 2 else jnp.square(jnp.diff(actions, axis=0)).mean()


class DreamerAdapter:
    def __init__(self, args: argparse.Namespace, extra: list[str], planner_config):
        from RSSMDiDrOnCarla.scripts import train_offline_rssm as offline

        offline.import_runtime()
        rssm_args = argparse.Namespace(
            task=args.task,
            replay_dir=args.offline_replay_dir,
            logdir=str(Path(args.output_dir) / "dreamer_runtime"),
            batch_length=args.batch_length,
            batch_size=args.batch_size,
            replay_size=int(1e6),
            jax_platform=args.jax_platform,
            from_checkpoint="",
        )
        config = offline.build_config(rssm_args, extra)
        obs_space, act_space = offline.infer_spaces_from_replay(args.offline_replay_dir)
        step = offline.embodied.Counter()
        self.agent = offline.dreamerv3.Agent(obs_space, act_space, step, config.dreamerv3)
        if len(self.agent.train_devices) != 1:
            raise ValueError("Set dreamerv3.jax.train_devices=[0].")
        checkpoint = offline.embodied.Checkpoint(log=False, parallel=False)
        checkpoint.agent = self.agent
        checkpoint.load(args.rssm_checkpoint, keys=["agent"])
        self.device = self.agent.train_devices[0]
        self.nj = offline.nj
        self.planner_config = planner_config
        self.loss_args = args
        self.start_fn = self.nj.jit(self.nj.pure(self._start_state), device=self.device)
        self.actor_fn = self.nj.jit(self.nj.pure(self._actor_loss), device=self.device)
        self.critic_fn = self.nj.jit(self.nj.pure(self._critic_loss), device=self.device)

    def _start_state(self, data):
        data = self.agent.agent.preprocess(data)
        embed = self.agent.agent.wm.encoder(data)
        prev_latent, prev_action = self.agent.agent.wm.initial(data["action"].shape[0])
        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], axis=1)
        post, _ = self.agent.agent.wm.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        state = {key: value[:, -1] for key, value in post.items()}
        if "ego_speed" in data:
            speed = data["ego_speed"][:, -1].reshape((data["action"].shape[0], -1))[:, :1]
        else:
            speed = jnp.zeros((data["action"].shape[0], 1), jnp.float32)
        return state, speed

    def _reward(self, state):
        return self.agent.agent.wm.heads["reward"](state).mode()

    def _cont(self, state):
        return self.agent.agent.wm.heads["cont"](state).mean()

    def _imagine(self, planner_params, state, speed):
        planner_config = self.planner_config
        args = self.loss_args
        feats = [flatten_state_feat(state)]
        rewards = [self._reward(state).reshape((-1,))]
        conts = [self._cont(state).reshape((-1,))]
        actions = []
        waypoints = []
        cur_state = state
        cur_speed = speed
        for _ in range(int(args.imag_horizon)):
            feat = feats[-1]
            out = predict(planner_params, planner_config, feat, timestep=planner_config.truncated_eval_step)
            wp = apply_plan_sign(
                soft_select(out["poses_reg"], out["poses_cls"], temperature=planner_config.actor_softmax_temperature),
                args.plan_x_sign,
                args.plan_y_sign,
            )
            action = differentiable_pidpp(wp, cur_speed, planner_config)
            cur_state = self.agent.agent.wm.rssm.img_step(cur_state, action)
            next_feat = flatten_state_feat(cur_state)
            reward = self._reward(cur_state).reshape((-1,))
            cont = self._cont(cur_state).reshape((-1,))
            acc = normalized_acc_to_phys(action, planner_config.ctrl_acc_min, planner_config.ctrl_acc_max)
            cur_speed = jnp.maximum(cur_speed.reshape((-1,)) + acc * float(planner_config.waypoint_dt), 0.0)[:, None]
            feats.append(next_feat)
            rewards.append(reward)
            conts.append(cont)
            actions.append(action)
            waypoints.append(wp)
        return {
            "features": jnp.stack(feats, axis=0),
            "rewards": jnp.stack(rewards, axis=0),
            "conts": jnp.stack(conts, axis=0),
            "actions": jnp.stack(actions, axis=0),
            "waypoints": jnp.stack(waypoints, axis=0),
        }

    def _actor_loss(
        self,
        planner_params,
        critic_params,
        ref_planner_params,
        data,
    ):
        planner_config = self.planner_config
        args = self.loss_args
        state, speed = self._start_state(data)
        imagined = self._imagine(planner_params, state, speed)
        value = critic_value(critic_params, imagined["features"])
        disc = float(args.discount) * imagined["conts"]
        returns = lambda_return(imagined["rewards"][1:], value[:-1], disc[1:], bootstrap=value[-1], lambda_=args.lambda_)
        actor_return = returns.mean()
        start_feat = flatten_state_feat(state)
        ref_out = predict(ref_planner_params, planner_config, start_feat, timestep=planner_config.truncated_eval_step)
        bc_ref = apply_plan_sign(
            soft_select(ref_out["poses_reg"], ref_out["poses_cls"], temperature=planner_config.actor_softmax_temperature),
            args.plan_x_sign,
            args.plan_y_sign,
        )
        bc_loss = jnp.mean(jnp.abs(imagined["waypoints"][0] - jax.lax.stop_gradient(bc_ref)))
        loss = (
            -actor_return
            + float(args.bc_weight) * bc_loss
            + float(args.wp_smooth_weight) * waypoint_smoothness(imagined["waypoints"])
            + float(args.ctrl_smooth_weight) * control_smoothness(imagined["actions"])
        )
        metrics = {"actor_loss": loss, "imag_return": actor_return, "bc_loss": bc_loss}
        return loss, metrics

    def _critic_loss(self, planner_params, critic_params, data):
        planner_config = self.planner_config
        args = self.loss_args
        state, speed = self._start_state(data)
        imagined = self._imagine(planner_params, state, speed)
        value = critic_value(critic_params, imagined["features"])
        disc = float(args.discount) * imagined["conts"]
        target = lambda_return(imagined["rewards"][1:], value[:-1], disc[1:], bootstrap=value[-1], lambda_=args.lambda_)
        loss = critic_loss(critic_params, jax.lax.stop_gradient(imagined["features"][:-1]), jax.lax.stop_gradient(target))
        return loss, {"critic_loss": loss, "target_return": target.mean()}

    def actor_loss(self, *args):
        rng = self.agent._next_rngs(self.agent.train_devices)
        out, self.agent.varibs = self.actor_fn(self.agent.varibs, rng, *args)
        return out

    def critic_loss(self, *args):
        rng = self.agent._next_rngs(self.agent.train_devices)
        out, self.agent.varibs = self.critic_fn(self.agent.varibs, rng, *args)
        return out


def main():
    args, extra = parse_args()
    if args.jax_platform:
        jax.config.update("jax_platform_name", args.jax_platform)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    planner_config, planner_params, _, _ = load_checkpoint(args.planner_checkpoint)
    planner_config.imag_horizon = int(args.imag_horizon)
    ref_planner_params = copy.deepcopy(planner_params)
    rng = jax.random.PRNGKey(0)
    rng, critic_rng = jax.random.split(rng)
    critic_params = init_critic(critic_rng, planner_config.latent_dim)
    actor_opt = optax.chain(optax.clip_by_global_norm(100.0), optax.adamw(args.actor_lr, weight_decay=args.weight_decay))
    critic_opt = optax.chain(optax.clip_by_global_norm(100.0), optax.adamw(args.critic_lr, weight_decay=args.weight_decay))
    actor_state = actor_opt.init(planner_params)
    critic_state = critic_opt.init(critic_params)
    adapter = DreamerAdapter(args, extra, planner_config)

    replay_dirs = [args.offline_replay_dir]
    if args.online_replay_dir and Path(args.online_replay_dir).exists() and list(Path(args.online_replay_dir).glob("*.npz")):
        replay_dirs.append(args.online_replay_dir)
    dataset = ReplaySequenceDataset(replay_dirs, args.batch_length)
    np_rng = np.random.default_rng(0)

    def actor_step(planner_params, actor_state, critic_params, batch):
        def objective(p):
            return adapter.actor_loss(
                p,
                jax.lax.stop_gradient(critic_params),
                ref_planner_params,
                batch,
            )

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(planner_params)
        updates, actor_state = actor_opt.update(grads, actor_state, planner_params)
        updates = freeze_plan_anchor_updates(updates)
        planner_params = optax.apply_updates(planner_params, updates)
        metrics = dict(metrics)
        metrics["actor_loss"] = loss
        return planner_params, actor_state, metrics

    def critic_step(critic_params, critic_state, planner_params, batch):
        def objective(c):
            return adapter.critic_loss(
                jax.lax.stop_gradient(planner_params),
                c,
                batch,
            )

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(critic_params)
        updates, critic_state = critic_opt.update(grads, critic_state, critic_params)
        critic_params = optax.apply_updates(critic_params, updates)
        metrics = dict(metrics)
        metrics["critic_loss"] = loss
        return critic_params, critic_state, metrics

    history = []
    for iteration in range(1, int(args.iterations) + 1):
        logs = {"iteration": iteration}
        for _ in range(int(args.actor_updates)):
            batch = jax.tree_util.tree_map(jnp.asarray, dataset.sample(args.batch_size, np_rng))
            planner_params, actor_state, metrics = actor_step(planner_params, actor_state, critic_params, batch)
            logs.update(scalarize(metrics))
        for _ in range(int(args.critic_updates)):
            batch = jax.tree_util.tree_map(jnp.asarray, dataset.sample(args.batch_size, np_rng))
            critic_params, critic_state, metrics = critic_step(critic_params, critic_state, planner_params, batch)
            logs.update(scalarize(metrics))
        history.append(logs)
        if iteration == 1 or iteration % 10 == 0:
            print("[jax_online] " + " ".join(f"{k}={v:.5f}" for k, v in logs.items() if isinstance(v, float)), flush=True)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.save_every > 0 and iteration % int(args.save_every) == 0:
            save_checkpoint(output_dir / "planner_online.pkl.gz", planner_config, planner_params, actor_state, iteration)
            save_tree_checkpoint(output_dir / "critic_online.pkl.gz", critic_params=critic_params, opt_state=critic_state, iteration=iteration)

    save_checkpoint(output_dir / "planner_online.pkl.gz", planner_config, planner_params, actor_state, args.iterations)
    save_tree_checkpoint(output_dir / "critic_online.pkl.gz", critic_params=critic_params, opt_state=critic_state, iteration=args.iterations)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[jax_online] Wrote checkpoints under {output_dir}")


if __name__ == "__main__":
    main()
