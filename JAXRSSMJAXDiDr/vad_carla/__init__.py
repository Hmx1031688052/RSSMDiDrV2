"""CARLA/VAD integration helpers for the JAX RSSM pipeline."""

from .camera_setup import (
    DEFAULT_VAD_LIDAR_X,
    DEFAULT_VAD_LIDAR_Y,
    DEFAULT_VAD_LIDAR_Z,
    B2D_CAMERA_ORDER,
    VAD_CAMERA_KEYS,
    VAD_CAMERA_ORDER,
    VADCameraSpec,
    build_vad_observation_overlay,
    camera_intrinsic,
    install_vad_carla_patches,
    get_camera_keys,
    get_camera_order,
    lidar2img_matrix,
    lidar2img_matrices,
)

__all__ = [
    "DEFAULT_VAD_LIDAR_X",
    "DEFAULT_VAD_LIDAR_Y",
    "DEFAULT_VAD_LIDAR_Z",
    "B2D_CAMERA_ORDER",
    "VAD_CAMERA_KEYS",
    "VAD_CAMERA_ORDER",
    "VADCameraSpec",
    "build_vad_observation_overlay",
    "camera_intrinsic",
    "install_vad_carla_patches",
    "get_camera_keys",
    "get_camera_order",
    "lidar2img_matrix",
    "lidar2img_matrices",
]
