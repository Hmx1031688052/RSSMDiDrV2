"""Closed-loop CARLA eval: B2D VAD latent -> JAX RSSM -> JAX DiffusionDrive.

The VAD perception model stays in PyTorch because the B2D checkpoint is a
Bench2DriveZoo/mmcv checkpoint. The world model and diffusion planner are the
JAX checkpoints used by `eval_close_loop.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dreamerv3") not in sys.path:
    sys.path.insert(0, str(ROOT / "dreamerv3"))

from JAXRSSMJAXDiDr.scripts import eval_close_loop as jax_loop
from JAXRSSMJAXDiDr.scripts import eval_close_loop_rssm_didr as base
from JAXRSSMJAXDiDr.scripts.export_vad_latents import (
    DEFAULT_VAD_LIDAR_X,
    DEFAULT_VAD_LIDAR_Y,
    DEFAULT_VAD_LIDAR_Z,
    build_vad,
    extract_vad_scene_latent,
    make_can_bus,
    make_frame_img_meta,
    make_img_meta,
    normalize_vad_runtime_args,
    vad_scene_latent_dim,
)
from JAXRSSMJAXDiDr.scripts.visualize_vad_carla_realtime import (
    ego_pose_from_env,
    stack_vad_images,
)
from JAXRSSMJAXDiDr.vad_carla import install_vad_carla_patches


VAD_ARGS = None
RAW_ENV = None
ORIGINAL_BUILD_CONFIG = base.build_config
ORIGINAL_PARSE_ARGS = base.parse_args
ORIGINAL_ONLINE_RSSM = base.OnlineRSSMEncoder


def consume_vad_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vad_runtime", choices=("official", "b2d"), default="b2d")
    parser.add_argument("--vad_root", default=str(ROOT / "VAD"))
    parser.add_argument("--vad_model", choices=("tiny", "base"), default="base")
    parser.add_argument("--vad_checkpoint", default="")
    parser.add_argument("--checkpoint_strict", action="store_true")
    parser.add_argument(
        "--b2d_root",
        default=str(ROOT / "Bench2DriveZoo-uniad-vad" / "Bench2DriveZoo-uniad-vad"),
    )
    parser.add_argument("--b2d_config", default="adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py")
    parser.add_argument("--vad_device", default="cuda:0")
    parser.add_argument("--camera_width", type=int, default=1600)
    parser.add_argument("--camera_height", type=int, default=900)
    parser.add_argument("--camera_fov", type=float, default=70.0)
    parser.add_argument("--camera_profile", choices=("nusc", "b2d"), default="b2d")
    parser.add_argument("--sensor_tick", type=float, default=0.1)
    parser.add_argument("--vad_lidar_x", type=float, default=DEFAULT_VAD_LIDAR_X)
    parser.add_argument("--vad_lidar_y", type=float, default=DEFAULT_VAD_LIDAR_Y)
    parser.add_argument("--vad_lidar_z", type=float, default=DEFAULT_VAD_LIDAR_Z)
    parser.add_argument("--vad_obs_key", default="vad_scene_latent")
    parser.add_argument("--latent_components", default="bev,agent,map,ego")
    parser.add_argument("--pool", choices=("mean", "mean_std"), default="mean_std")
    parser.add_argument("--pretrained_norm", action="store_true", default=False)
    parser.add_argument("--no_pretrained_norm", dest="pretrained_norm", action="store_false")
    parser.add_argument("--no_temporal_bev", action="store_true")
    parser.add_argument("--vad_debug", action="store_true")
    parser.add_argument("--vad_debug_every", type=int, default=100)
    args, remaining = parser.parse_known_args(argv[1:])
    if not args.vad_checkpoint and not any(item in ("-h", "--help") for item in remaining):
        raise ValueError("--vad_checkpoint is required for VAD->RSSM closed-loop evaluation.")
    args.device = args.vad_device
    argv[:] = [argv[0], *remaining]
    return normalize_vad_runtime_args(args)


def build_config_with_vad(args: argparse.Namespace, extra: list[str]):
    global RAW_ENV
    assert VAD_ARGS is not None
    install_vad_carla_patches(
        task=args.task,
        width=VAD_ARGS.camera_width,
        height=VAD_ARGS.camera_height,
        fov=VAD_ARGS.camera_fov,
        sensor_tick=VAD_ARGS.sensor_tick,
        include_birdeye=True,
        camera_profile=VAD_ARGS.camera_profile,
        preserve_existing_observations=True,
    )
    raw_env, config = ORIGINAL_BUILD_CONFIG(args, extra)
    RAW_ENV = raw_env
    vad_key = str(VAD_ARGS.vad_obs_key)
    config = config.update(
        {
            "dreamerv3.encoder.cnn_keys": "none",
            "dreamerv3.decoder.cnn_keys": "none",
            "dreamerv3.encoder.mlp_keys": vad_key,
            "dreamerv3.decoder.mlp_keys": vad_key,
            "dreamerv3.run.log_keys_video": ["none"],
        }
    )
    return raw_env, config


def make_vad_rssm_obs_space(latent_dim: int) -> Dict[str, object]:
    space = base.embodied.Space
    return {
        str(VAD_ARGS.vad_obs_key): space(np.float32, (int(latent_dim),)),
        "reward": space(np.float32, ()),
        "is_first": space(np.bool_, ()),
        "is_last": space(np.bool_, ()),
        "is_terminal": space(np.bool_, ()),
    }


def obs_scalar(obs: Dict[str, np.ndarray], key: str, default, dtype):
    if key not in obs:
        return np.asarray(default, dtype=dtype)
    value = np.asarray(obs[key])
    if value.shape == ():
        return value.astype(dtype, copy=False)
    return np.asarray(value.reshape(-1)[0], dtype=dtype)


class OnlineVADRSSMEncoder(ORIGINAL_ONLINE_RSSM):
    def __init__(self, obs_space, act_space, checkpoint_path: str, config):
        del obs_space
        if VAD_ARGS is None or RAW_ENV is None:
            raise RuntimeError("VAD runtime was not initialized before constructing the RSSM encoder.")
        self.vad_args = VAD_ARGS
        self.raw_env = RAW_ENV
        self.torch, self.vad_model, self.vad_cfg = build_vad(self.vad_args)
        latent_dim = vad_scene_latent_dim(self.vad_model, self.vad_args)
        rssm_obs_space = make_vad_rssm_obs_space(latent_dim)
        super().__init__(rssm_obs_space, act_space, checkpoint_path, config)
        self.base_meta = make_img_meta(self.vad_args, self.vad_cfg)
        self.prev_vad_bev = None
        self.prev_pose = None
        self.frame_index = 0
        print(
            "[vad_rssm_jax] "
            f"runtime={self.vad_args.vad_runtime} obs_key={self.vad_args.vad_obs_key} "
            f"latent_dim={latent_dim} components={self.vad_args.latent_components} "
            f"pool={self.vad_args.pool} camera_profile={self.vad_args.camera_profile}",
            flush=True,
        )

    def reset(self, action_shape):
        super().reset(action_shape)
        self.prev_vad_bev = None
        self.prev_pose = None
        self.frame_index = 0

    def encode(self, obs: Dict[str, np.ndarray], prev_action: np.ndarray) -> np.ndarray:
        is_first = bool(obs_scalar(obs, "is_first", self.frame_index == 0, np.bool_))
        if is_first:
            self.prev_vad_bev = None
            self.prev_pose = None

        images = stack_vad_images(obs, self.vad_args)
        pose = ego_pose_from_env(self.raw_env)
        can_bus = make_can_bus(pose, self.prev_pose)
        meta = make_frame_img_meta(
            self.base_meta,
            can_bus,
            Path("online_vad"),
            self.frame_index,
            self.prev_pose is not None,
        )
        prev_bev = None if self.vad_args.no_temporal_bev else self.prev_vad_bev
        vad_latent, next_bev = extract_vad_scene_latent(
            self.torch,
            self.vad_model,
            images,
            meta,
            self.vad_args,
            self.vad_cfg,
            prev_bev,
        )
        self.prev_vad_bev = None if self.vad_args.no_temporal_bev else next_bev
        self.prev_pose = pose
        self.frame_index += 1

        rssm_obs = {
            str(self.vad_args.vad_obs_key): vad_latent,
            "reward": obs_scalar(obs, "reward", 0.0, np.float32),
            "is_first": np.asarray(is_first, dtype=np.bool_),
            "is_last": obs_scalar(obs, "is_last", False, np.bool_),
            "is_terminal": obs_scalar(obs, "is_terminal", False, np.bool_),
        }
        return super().encode(rssm_obs, prev_action)


def patch_parse_args_for_jax_flags(plan_interval_steps: int) -> None:
    def parse_args():
        args, extra = ORIGINAL_PARSE_ARGS()
        args.plan_interval_steps = int(plan_interval_steps)
        return args, extra

    base.parse_args = parse_args


def main() -> None:
    global VAD_ARGS
    VAD_ARGS = consume_vad_args(sys.argv)
    eval_timestep = jax_loop.consume_int_flag(sys.argv, "eval_timestep", 0)
    plan_interval_steps = jax_loop.consume_int_flag(sys.argv, "plan_interval_steps", 5)
    jax_loop.EVAL_TIMESTEP = int(eval_timestep)
    jax_loop.PLAN_INTERVAL_STEPS = int(plan_interval_steps)

    patch_parse_args_for_jax_flags(plan_interval_steps)
    base.build_config = build_config_with_vad
    base.OnlineRSSMEncoder = OnlineVADRSSMEncoder
    base.load_planner = jax_loop.load_jax_planner
    base.plan_with_model = jax_loop.plan_with_jax_model
    base.main()


if __name__ == "__main__":
    main()
