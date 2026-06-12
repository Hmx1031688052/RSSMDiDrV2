"""Export DreamerV3 RSSM posterior latents for processed expert replay chunks."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import ruamel.yaml as yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dreamerv3") not in sys.path:
    sys.path.insert(0, str(ROOT / "dreamerv3"))

from RSSMDiDrOnCarla.data.polyplanner_targets import load_replay_chunk, replay_chunk_length


warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

jax = None
jnp = None
embodied = None
car_dreamer = None
dreamerv3 = None
nj = None
wrap_env = None
from_gym = None


def import_runtime() -> None:
    global jax, jnp, embodied, car_dreamer, dreamerv3, nj, wrap_env, from_gym
    import jax as jax_module
    import jax.numpy as jnp_module
    import embodied as embodied_module
    import car_dreamer as car_dreamer_module
    import dreamerv3 as dreamerv3_module
    from dreamerv3 import ninjax as ninjax_module
    from dreamerv3.collect_utils import wrap_env as wrap_env_fn
    from embodied.envs import from_gym as from_gym_module

    jax = jax_module
    jnp = jnp_module
    embodied = embodied_module
    car_dreamer = car_dreamer_module
    dreamerv3 = dreamerv3_module
    nj = ninjax_module
    wrap_env = wrap_env_fn
    from_gym = from_gym_module


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", default="carla_roundabout")
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--jax_platform", default=None, choices=("cpu", "gpu", "tpu"))
    parser.add_argument("--no_save_components", action="store_true")
    return parser.parse_known_args()


def build_config(args: argparse.Namespace, extra: list[str]):
    yaml_path = ROOT / "dreamerv3" / "dreamerv3.yaml"
    model_configs = yaml.YAML(typ="safe").load(embodied.Path(str(yaml_path)).read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    raw_env, env_config = car_dreamer.create_task(args.task, extra)
    config = config.update(env_config)
    updates = {
        "dreamerv3.batch_length": args.batch_length,
        "dreamerv3.batch_size": args.batch_size,
    }
    if args.jax_platform:
        updates["dreamerv3.jax.platform"] = args.jax_platform
    config = config.update(updates)
    config = embodied.Flags(config).parse(extra)
    return raw_env, config


def get_spaces(raw_env, dreamerv3_config):
    env = from_gym.FromGym(raw_env)
    env = wrap_env(env, dreamerv3_config)
    obs_space = env.obs_space
    act_space = env.act_space
    try:
        env.close()
    except Exception:
        pass
    return obs_space, act_space


def make_export_fn(agent):
    def export_post(data):
        data = agent.agent.preprocess(data)
        embed = agent.agent.wm.encoder(data)
        prev_latent, prev_action = agent.agent.wm.initial(data["action"].shape[0])
        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)
        post, _ = agent.agent.wm.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        return post

    return nj.jit(nj.pure(export_post), device=agent.train_devices[0])


def temporal_batch(chunk: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    length = len(chunk["action"])
    batch = {}
    for key, value in chunk.items():
        value = np.asarray(value)
        if value.ndim == 0:
            continue
        if value.shape[0] != length:
            continue
        batch[key] = value[None]
    return batch


def flatten_latent(post: Dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for key in ("deter", "stoch"):
        if key in post:
            value = np.asarray(post[key], dtype=np.float32)
            parts.append(value.reshape(value.shape[0], -1))
    if not parts:
        raise KeyError(f"Could not build rssm_latent from posterior keys: {sorted(post.keys())}")
    return np.concatenate(parts, axis=-1).astype(np.float32)


def main() -> None:
    args, extra = parse_args()
    import_runtime()
    raw_env, config = build_config(args, extra)
    cfg = config.dreamerv3
    obs_space, act_space = get_spaces(raw_env, cfg)
    step = embodied.Counter()

    print("Task:", args.task)
    print("Checkpoint:", args.checkpoint)
    print("Replay dir:", args.replay_dir)
    print("Output dir:", args.output_dir)

    agent = dreamerv3.Agent(obs_space, act_space, step, cfg)
    if len(agent.train_devices) != 1:
        raise ValueError("export_rssm_latents currently expects one train device. Set dreamerv3.jax.train_devices=[0].")

    checkpoint = embodied.Checkpoint(log=False, parallel=False)
    checkpoint.agent = agent
    checkpoint.load(args.checkpoint, keys=["agent"])

    export_fn = make_export_fn(agent)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_paths = sorted(Path(args.replay_dir).glob("*.npz"))
    if not replay_paths:
        raise FileNotFoundError(f"No replay chunks found in {args.replay_dir}")

    summaries = []
    for replay_path in replay_paths:
        chunk = load_replay_chunk(replay_path, trim_to_length=True)
        batch = temporal_batch(chunk)
        if "action" not in batch:
            raise KeyError(f"{replay_path}: missing temporal `action` field")

        data = jax.tree_util.tree_map(lambda x: jax.device_put(x, agent.train_devices[0]), batch)
        rng = agent._next_rngs(agent.train_devices)
        post, _ = export_fn(agent.varibs, rng, data)
        post = jax.device_get(post)
        post = {key: np.asarray(value[0], dtype=np.float32) for key, value in post.items()}
        rssm_latent = flatten_latent(post)

        out = {
            "rssm_latent": rssm_latent,
            "replay_chunk": np.asarray(replay_path.name),
            "valid_length": np.int32(replay_chunk_length(replay_path, fallback=len(rssm_latent))),
        }
        if not args.no_save_components:
            out.update(post)

        output_path = output_dir / replay_path.name
        np.savez_compressed(output_path, **out)
        summary = {
            "replay_chunk": str(replay_path),
            "output_chunk": str(output_path),
            "length": int(len(rssm_latent)),
            "latent_dim": int(rssm_latent.shape[-1]),
            "component_keys": sorted(post.keys()),
        }
        summaries.append(summary)
        print(f"[export_rssm_latents] {replay_path.name}: rssm_latent={rssm_latent.shape}")

    summary_path = output_dir / "rssm_latents_summary.json"
    summary_path.write_text(json.dumps({"chunks": len(summaries), "summaries": summaries}, indent=2), encoding="utf-8")
    print(f"[export_rssm_latents] Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
