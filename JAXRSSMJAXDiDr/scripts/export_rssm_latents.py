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

from JAXRSSMJAXDiDr.data.polyplanner_targets import load_replay_chunk, replay_chunk_length


warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

jax = None
jnp = None
embodied = None
dreamerv3 = None
nj = None
wrap_env = None
from_gym = None


def import_runtime() -> None:
    global jax, jnp, embodied, dreamerv3, nj, wrap_env, from_gym
    import jax as jax_module
    import jax.numpy as jnp_module
    import embodied as embodied_module
    import dreamerv3 as dreamerv3_module
    from dreamerv3 import ninjax as ninjax_module
    from dreamerv3.collect_utils import wrap_env as wrap_env_fn
    from embodied.envs import from_gym as from_gym_module

    jax = jax_module
    jnp = jnp_module
    embodied = embodied_module
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
    parser.add_argument(
        "--structured_world_model",
        action="store_true",
        help="Use structured ego/neighbor/route observations only, matching train_offline_rssm.",
    )
    return parser.parse_known_args()


def load_task_config(task: str, extra: list[str]):
    config_dir = ROOT / "car_dreamer" / "configs"
    yaml_loader = yaml.YAML(typ="safe")
    common = yaml_loader.load((config_dir / "common.yaml").read_text())
    tasks = yaml_loader.load((config_dir / "tasks.yaml").read_text())
    if task not in tasks:
        raise KeyError(f"Unknown task '{task}'. Available tasks: {sorted(tasks)}")
    task_config = embodied.Config(common).update(tasks[task])
    task_config, _ = embodied.Flags(task_config).parse_known(extra)
    return task_config


def build_config(args: argparse.Namespace, extra: list[str]):
    yaml_path = ROOT / "dreamerv3" / "dreamerv3.yaml"
    model_configs = yaml.YAML(typ="safe").load(embodied.Path(str(yaml_path)).read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    env_config = load_task_config(args.task, extra)
    config = config.update(env_config)
    updates = {
        "dreamerv3.batch_length": args.batch_length,
        "dreamerv3.batch_size": args.batch_size,
    }
    if getattr(args, "structured_world_model", False):
        structured_keys = (
            "ego_.*|neighbor_vehicles_local|route_waypoints8|global_path_ego|"
            "global_path_ego_mask|target_region|route_remaining"
        )
        updates.update(
            {
                "dreamerv3.encoder.cnn_keys": "none",
                "dreamerv3.decoder.cnn_keys": "none",
                "dreamerv3.encoder.mlp_keys": structured_keys,
                "dreamerv3.decoder.mlp_keys": structured_keys,
                "dreamerv3.run.log_keys_video": ["none"],
            }
        )
    if args.jax_platform:
        updates["dreamerv3.jax.platform"] = args.jax_platform
    config = config.update(updates)
    config = embodied.Flags(config).parse(extra)
    return config


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


def make_space(array: np.ndarray, *, low=None, high=None):
    array = np.asarray(array)
    shape = tuple(array.shape[1:]) if array.ndim > 0 else ()
    dtype = np.dtype(array.dtype).type
    try:
        return embodied.Space(dtype, shape, low=low, high=high)
    except TypeError:
        try:
            return embodied.Space(dtype, shape)
        except TypeError:
            return embodied.Space(dtype=dtype, shape=shape, low=low, high=high)


def _skip_replay_field(key: str, value: np.ndarray) -> bool:
    value = np.asarray(value)
    return key.startswith("__") or key.endswith("_path") or value.dtype.kind in ("O", "S", "U")


def infer_spaces_from_replay(replay_dir: str | Path):
    paths = sorted(Path(replay_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No replay chunks found in {replay_dir}")
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            chunk = {key: np.asarray(data[key]) for key in data.files}
        if "action" not in chunk:
            continue
        action = np.asarray(chunk["action"])
        if action.ndim == 0:
            continue
        length = int(action.shape[0])
        obs_space = {}
        for key, value in chunk.items():
            value = np.asarray(value)
            if key == "action" or _skip_replay_field(key, value) or value.ndim == 0 or value.shape[0] != length:
                continue
            obs_space[key] = make_space(value)
        obs_space.setdefault("reward", make_space(np.zeros((length,), dtype=np.float32)))
        obs_space.setdefault("is_first", make_space(np.zeros((length,), dtype=bool)))
        obs_space.setdefault("is_last", make_space(np.zeros((length,), dtype=bool)))
        obs_space.setdefault("is_terminal", make_space(np.zeros((length,), dtype=bool)))
        act_space = {
            "action": make_space(action, low=-1.0, high=1.0),
            "reset": make_space(np.zeros((length,), dtype=bool)),
        }
        return obs_space, act_space
    raise KeyError(f"No replay chunk with temporal `action` found in {replay_dir}")


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
        if _skip_replay_field(key, value):
            continue
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
    config = build_config(args, extra)
    cfg = config.dreamerv3
    obs_space, act_space = infer_spaces_from_replay(args.replay_dir)
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
