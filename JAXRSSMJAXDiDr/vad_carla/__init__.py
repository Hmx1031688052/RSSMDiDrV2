"""CARLA/VAD integration helpers for the JAX RSSM pipeline."""

from .camera_setup import (
    VAD_CAMERA_KEYS,
    VAD_CAMERA_ORDER,
    VADCameraSpec,
    build_vad_observation_overlay,
    camera_intrinsic,
    install_vad_carla_patches,
    lidar2img_matrix,
)

__all__ = [
    "VAD_CAMERA_KEYS",
    "VAD_CAMERA_ORDER",
    "VADCameraSpec",
    "build_vad_observation_overlay",
    "camera_intrinsic",
    "install_vad_carla_patches",
    "lidar2img_matrix",
]
