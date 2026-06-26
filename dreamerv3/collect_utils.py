import numpy as np

import embodied


def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            # Skip NormalizeAction when action space is already [-1, 1]^N
            # (e.g., waypoint actions).  NormalizeAction would be identity.
            low = np.asarray(space.low, dtype=np.float64)
            high = np.asarray(space.high, dtype=np.float64)
            if np.allclose(low, -1.0) and np.allclose(high, 1.0):
                pass
            else:
                env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env

def make_replay(config, logdir):
    replay_dir_value = getattr(config, "replay_dir", None)
    replay_dir = embodied.Path(replay_dir_value or (logdir / "replay"))
    replay_dir.mkdirs()
    replay_chunks = int(getattr(config, "replay_chunks", 1024))
    replay_disk_buffer = bool(getattr(config, "replay_disk_buffer", False))
    return embodied.replay.Uniform(
        config.batch_length,
        config.replay_size,
        replay_dir,
        chunks=replay_chunks,
        disk_buffer=replay_disk_buffer,
    ), replay_dir
