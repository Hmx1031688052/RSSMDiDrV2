"""Train DreamerV3 RSSM world model offline from CARLA expert replay."""

from __future__ import annotations

import argparse
import datetime
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


warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

embodied = None
car_dreamer = None
dreamerv3 = None
nj = None
wrap_env = None
from_gym = None


def import_runtime() -> None:
    global embodied, car_dreamer, dreamerv3, nj, wrap_env, from_gym
    import embodied as embodied_module
    import car_dreamer as car_dreamer_module
    import dreamerv3 as dreamerv3_module
    from dreamerv3 import ninjax as ninjax_module
    from dreamerv3.collect_utils import wrap_env as wrap_env_fn
    from embodied.envs import from_gym as from_gym_module

    embodied = embodied_module
    car_dreamer = car_dreamer_module
    dreamerv3 = dreamerv3_module
    nj = ninjax_module
    wrap_env = wrap_env_fn
    from_gym = from_gym_module


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="carla_roundabout")
    parser.add_argument("--replay_dir", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--updates", type=int, default=100000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--replay_size", type=float, default=1e6)
    parser.add_argument("--jax_platform", default=None, choices=("cpu", "gpu", "tpu"))
    parser.add_argument("--from_checkpoint", default="")
    parser.add_argument(
        "--structured_world_model",
        action="store_true",
        help="Use structured ego/neighbor/route observations only, disabling BEV image encoder/decoder keys.",
    )
    return parser.parse_known_args()


def build_config(args: argparse.Namespace, extra: list[str]):
    yaml_path = ROOT / "dreamerv3" / "dreamerv3.yaml"
    model_configs = yaml.YAML(typ="safe").load(embodied.Path(str(yaml_path)).read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    raw_env, env_config = car_dreamer.create_task(args.task, extra)
    config = config.update(env_config)
    updates = {
        "dreamerv3.logdir": args.logdir,
        "dreamerv3.replay_dir": args.replay_dir,
        "dreamerv3.batch_length": args.batch_length,
        "dreamerv3.batch_size": args.batch_size,
        "dreamerv3.replay_size": int(args.replay_size),
        "dreamerv3.run.from_checkpoint": args.from_checkpoint,
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
                "dreamerv3.run.log_keys_video": [],
            }
        )
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


def infer_spaces_from_replay(replay_dir: str | Path):
    """Infer Dreamer obs/action spaces from row-format replay chunks."""

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
            if key == "action" or value.ndim == 0 or value.shape[0] != length:
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


def make_world_model_train(agent):
    def train_world_model(data, state):
        data = agent.agent.preprocess(data)
        state, _, metrics = agent.agent.wm.train(data, state)
        metrics["offline_wm_loss"] = metrics["model_loss_mean"]
        return state, metrics

    return nj.jit(nj.pure(train_world_model), device=agent.train_devices[0])


def scalarize(metrics: Dict[str, object]) -> Dict[str, float]:
    out = {}
    for key, value in metrics.items():
        arr = np.asarray(value)
        if arr.shape == ():
            out[key] = float(arr)
    return out


def main() -> None:
    args, extra = parse_args()
    import_runtime()
    raw_env, config = build_config(args, extra)
    cfg = config.dreamerv3
    logdir = embodied.Path(cfg.logdir)
    logdir.mkdirs()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config.save(str(logdir / f"offline_rssm_config_{timestamp}.yaml"))

    obs_space, act_space = get_spaces(raw_env, cfg)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    print("Task:", args.task)
    print("Replay dir:", args.replay_dir)
    print("Logdir:", logdir)
    print("Observation space:", embodied.format(obs_space), sep="\n")
    print("Action space:", embodied.format(act_space), sep="\n")

    agent = dreamerv3.Agent(obs_space, act_space, step, cfg)
    if len(agent.train_devices) != 1:
        raise ValueError("train_offline_rssm currently expects one train device. Set dreamerv3.jax.train_devices=[0].")

    replay = embodied.replay.Uniform(cfg.batch_length, cfg.replay_size, embodied.Path(args.replay_dir))
    if len(replay) == 0:
        raise RuntimeError(f"No replay sequences loaded from {args.replay_dir}")
    dataset = agent.dataset(replay.dataset)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt", parallel=False)
    checkpoint.step = step
    checkpoint.agent = agent
    if cfg.run.from_checkpoint:
        checkpoint.load(cfg.run.from_checkpoint, keys=["agent"])
    elif checkpoint.exists():
        checkpoint.load()
    else:
        checkpoint.save()

    wm_train = make_world_model_train(agent)
    state = None
    metrics_accum = embodied.Metrics()
    print("Start offline RSSM world model training.")

    for update in range(1, int(args.updates) + 1):
        batch = next(dataset)
        if state is None:
            rng = agent._next_rngs(agent.train_devices)
            state, agent.varibs = agent._init_train(agent.varibs, rng, batch["is_first"])

        rng = agent._next_rngs(agent.train_devices)
        (state, metrics), agent.varibs = wm_train(agent.varibs, rng, batch, state)
        metrics = agent._convert_mets(metrics, agent.train_devices)
        metrics_accum.add(metrics, prefix="offline_rssm")
        step.increment(int(cfg.batch_size * cfg.batch_length))

        if update % int(args.log_every) == 0:
            scalars = scalarize(metrics_accum.result())
            logger.add(scalars)
            logger.add(replay.stats, prefix="replay")
            logger.write()
            loss = scalars.get("offline_rssm/offline_wm_loss", np.nan)
            print(f"[offline_rssm] update={update} step={int(step)} loss={loss:.6f}")

        if update % int(args.save_every) == 0:
            checkpoint.save()

    checkpoint.save()
    logger.write()
    print(f"[offline_rssm] Wrote checkpoint: {logdir / 'checkpoint.ckpt'}")


if __name__ == "__main__":
    main()
