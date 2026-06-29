"""VAD-compatible six-camera setup for CARLA replay collection.

The VAD checkpoint was trained with nuScenes-style six camera embeddings. Keep
the camera order stable when exporting tensors:

    CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
    CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT

The helpers here live under JAXRSSMJAXDiDr so the VAD/RSSM pipeline stays
self-contained inside the JAX package.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class VADCameraSpec:
    name: str
    key: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0


VAD_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

VAD_CAMERA_SPECS = {
    "CAM_FRONT": VADCameraSpec("CAM_FRONT", "camera_front", 1.5, 0.0, 2.0, 0.0),
    "CAM_FRONT_RIGHT": VADCameraSpec("CAM_FRONT_RIGHT", "camera_front_right", 1.3, 0.45, 2.0, 55.0),
    "CAM_FRONT_LEFT": VADCameraSpec("CAM_FRONT_LEFT", "camera_front_left", 1.3, -0.45, 2.0, -55.0),
    "CAM_BACK": VADCameraSpec("CAM_BACK", "camera_back", -1.5, 0.0, 2.0, 180.0),
    "CAM_BACK_LEFT": VADCameraSpec("CAM_BACK_LEFT", "camera_back_left", -1.3, -0.45, 2.0, -125.0),
    "CAM_BACK_RIGHT": VADCameraSpec("CAM_BACK_RIGHT", "camera_back_right", -1.3, 0.45, 2.0, 125.0),
}

DEFAULT_VAD_LIDAR_X = 0.8
DEFAULT_VAD_LIDAR_Y = 0.0
DEFAULT_VAD_LIDAR_Z = 2.3
VAD_CAMERA_KEYS = tuple(VAD_CAMERA_SPECS[name].key for name in VAD_CAMERA_ORDER)
SIDECAR_PATH_BYTES = 256


def build_vad_observation_overlay(
    width: int = 1600,
    height: int = 900,
    fov: float = 70.0,
    sensor_tick: float = 0.1,
    include_birdeye: bool = True,
) -> Dict[str, object]:
    """Return a config overlay that enables six VAD cameras in CarDreamer."""

    overlay: Dict[str, object] = {}
    enabled = list(VAD_CAMERA_KEYS)
    enabled.append("collision")
    if include_birdeye:
        enabled.append("birdeye_wpt")
    overlay["env.observation.enabled"] = enabled
    overlay["dreamerv3.run.log_keys_video"] = ["camera_front", "birdeye_wpt"] if include_birdeye else ["camera_front"]
    for spec in VAD_CAMERA_SPECS.values():
        prefix = f"env.observation.{spec.key}"
        overlay[f"{prefix}.handler"] = "camera"
        overlay[f"{prefix}.blueprint"] = "sensor.camera.rgb"
        overlay[f"{prefix}.key"] = spec.key
        overlay[f"{prefix}.shape"] = [int(height), int(width), 3]
        overlay[f"{prefix}.transform.x"] = float(spec.x)
        overlay[f"{prefix}.transform.y"] = float(spec.y)
        overlay[f"{prefix}.transform.z"] = float(spec.z)
        overlay[f"{prefix}.transform.pitch"] = float(spec.pitch)
        overlay[f"{prefix}.transform.yaw"] = float(spec.yaw)
        overlay[f"{prefix}.transform.roll"] = float(spec.roll)
        overlay[f"{prefix}.attributes.image_size_x"] = int(width)
        overlay[f"{prefix}.attributes.image_size_y"] = int(height)
        overlay[f"{prefix}.attributes.fov"] = float(fov)
        overlay[f"{prefix}.attributes.sensor_tick"] = float(sensor_tick)
    return overlay


def install_vad_carla_patches(
    task: str,
    width: int = 1600,
    height: int = 900,
    fov: float = 70.0,
    sensor_tick: float = 0.1,
    include_birdeye: bool = True,
) -> None:
    """Patch CarDreamer at runtime to expose VAD cameras and camera rotation.

    This avoids adding new scripts or config files outside JAXRSSMJAXDiDr. The
    patch is intentionally local to the current Python process.
    """

    import carla
    import car_dreamer
    from car_dreamer.toolkit.observer.handlers import sensor_handlers

    if not getattr(sensor_handlers.SensorHandler, "_jaxrssm_vad_rotation_patch", False):
        original_init = sensor_handlers.SensorHandler.__init__

        def patched_init(self, world, config):
            super(sensor_handlers.SensorHandler, self).__init__(world, config)
            blueprint = self._world.get_blueprint(config.blueprint)
            transform = getattr(config, "transform", {})
            loc = {
                key: float(transform[key])
                for key in ("x", "y", "z")
                if key in transform
            }
            rot = {
                key: float(transform[key])
                for key in ("pitch", "yaw", "roll")
                if key in transform
            }
            self._transform = carla.Transform(carla.Location(**loc), carla.Rotation(**rot))
            if "attributes" in config:
                for attr_name, attr_value in config.attributes.items():
                    blueprint.set_attribute(attr_name, str(attr_value))
            self._blueprint = blueprint
            self._sensor = None
            self._data = None

        sensor_handlers.SensorHandler.__init__ = patched_init
        sensor_handlers.SensorHandler._jaxrssm_vad_rotation_patch = True
        sensor_handlers.SensorHandler._jaxrssm_vad_original_init = original_init

    if getattr(car_dreamer, "_jaxrssm_vad_config_patch", None) == task:
        return

    original_load = getattr(car_dreamer, "_jaxrssm_vad_original_load_task_configs", None)
    if original_load is None:
        original_load = car_dreamer.load_task_configs
        car_dreamer._jaxrssm_vad_original_load_task_configs = original_load

    overlay = build_vad_observation_overlay(width, height, fov, sensor_tick, include_birdeye)

    def patched_load_task_configs(task_name: str):
        config = original_load(task_name)
        if task_name == task:
            config = config.update(overlay)
        return config

    car_dreamer.load_task_configs = patched_load_task_configs
    car_dreamer._jaxrssm_vad_config_patch = task


def camera_intrinsic(width: int, height: int, fov: float) -> np.ndarray:
    """Pinhole intrinsic matrix for CARLA RGB camera attributes."""

    focal = float(width) / (2.0 * math.tan(math.radians(float(fov)) / 2.0))
    return np.asarray(
        [[focal, 0.0, float(width) / 2.0], [0.0, focal, float(height) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _rotation_yaw_pitch_roll(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return rz @ ry @ rx


def lidar2img_matrix(
    spec: VADCameraSpec,
    width: int = 1600,
    height: int = 900,
    fov: float = 70.0,
    scale: float = 1.0,
    lidar_x: float = DEFAULT_VAD_LIDAR_X,
    lidar_y: float = DEFAULT_VAD_LIDAR_Y,
    lidar_z: float = DEFAULT_VAD_LIDAR_Z,
) -> np.ndarray:
    """Approximate nuScenes-lidar to image projection for CARLA-mounted cameras.

    VAD uses a lidar/ego frame with x forward, y left, z up. CarDreamer local
    vehicle features use x forward, y right, z up, so this function flips the y
    axis before applying the CARLA camera pose. The camera mount is defined in
    CARLA ego coordinates, while VAD's projection starts at a pseudo top-lidar
    origin; subtract the lidar origin before building lidar-to-camera.
    """

    k = camera_intrinsic(width, height, fov).astype(np.float32)
    k[:2] *= float(scale)
    rot = _rotation_yaw_pitch_roll(spec.yaw, spec.pitch, spec.roll)
    forward = rot @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    right = rot @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    up = rot @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    lidar_t = np.asarray([float(lidar_x), float(lidar_y), float(lidar_z)], dtype=np.float32)
    cam_t = np.asarray([spec.x, spec.y, spec.z], dtype=np.float32) - lidar_t

    # Input point is [x_forward, y_left, z_up]. Convert to CARLA y-right.
    nusc_to_carla = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    world_to_cam = np.eye(4, dtype=np.float32)
    # Camera coordinates: x right, y down, z forward.
    world_to_cam[:3, :3] = np.stack([right, -up, forward], axis=0)
    world_to_cam[:3, 3] = -world_to_cam[:3, :3] @ cam_t

    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:3, :3] = k
    return viewpad @ world_to_cam @ nusc_to_carla


def iter_camera_specs(order: Iterable[str] = VAD_CAMERA_ORDER) -> Iterable[VADCameraSpec]:
    for name in order:
        yield VAD_CAMERA_SPECS[name]


def _decode_path_value(value) -> str:
    value = np.asarray(value)
    if value.dtype == np.uint8:
        flat = value.reshape(-1)
        zeros = np.flatnonzero(flat == 0)
        end = int(zeros[0]) if zeros.size else int(flat.size)
        return bytes(flat[:end].tolist()).decode("utf-8")
    if value.shape == ():
        item = value.item()
    else:
        item = value.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _chunk_replay_dir(chunk: Mapping[str, np.ndarray]) -> Path:
    for key in ("__replay_dir__", "_replay_dir"):
        if key in chunk:
            return Path(_decode_path_value(chunk[key]))
    for key in ("__chunk_path__", "_chunk_path"):
        if key in chunk:
            return Path(_decode_path_value(chunk[key])).parent
    return Path(".")


def _load_rgb_sidecar(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Replay sidecar image does not exist: {path}")
    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ModuleNotFoundError:
        pass

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Loading replay image sidecars requires Pillow or OpenCV.") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Failed to read sidecar image: {path}")
    return image[..., ::-1].astype(np.uint8)


def _select_camera_image(chunk: Mapping[str, np.ndarray], key: str, index: int) -> np.ndarray:
    if key in chunk:
        return np.asarray(chunk[key][index])
    path_key = f"{key}_path"
    if path_key not in chunk:
        raise KeyError(f"Replay chunk is missing VAD camera key `{key}` and sidecar key `{path_key}`")
    paths = np.asarray(chunk[path_key])
    value = paths if paths.ndim == 1 and paths.dtype == np.uint8 else paths[index]
    relpath = Path(_decode_path_value(value))
    if not relpath.is_absolute():
        relpath = _chunk_replay_dir(chunk) / relpath
    return _load_rgb_sidecar(relpath)


def select_camera_arrays(chunk: Mapping[str, np.ndarray], index: int) -> np.ndarray:
    """Stack RGB camera images in VAD camera order as [6, H, W, 3].

    Supports both legacy inline RGB arrays and nuScenes-style JPEG sidecars
    referenced by `<camera_key>_path` uint8 path-byte fields.
    """

    images = []
    for key in VAD_CAMERA_KEYS:
        images.append(_select_camera_image(chunk, key, index))
    return np.stack(images, axis=0)
