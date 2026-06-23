"""BEV image builders for Bench2Drive annotations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .features import as_float, world_to_ego_xy, yaw_to_rad


Color = Tuple[int, int, int]


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required for Bench2Drive BEV conversion.") from exc
    return cv2


def load_topdown_bev(image_path: str | Path, shape: Tuple[int, int]) -> np.ndarray:
    image_path = Path(image_path)
    height, width = int(shape[0]), int(shape[1])
    try:
        cv2 = _import_cv2()
    except ModuleNotFoundError:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("OpenCV or Pillow is required for top-down BEV image loading.") from exc
        if not image_path.exists():
            raise FileNotFoundError(f"Could not read top-down image: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((width, height), Image.BILINEAR)
            return np.asarray(image, dtype=np.uint8)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read top-down image: {image_path}")
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _points_from_hdmap_item(item: Mapping) -> list[Tuple[float, float]]:
    out = []
    for point in item.get("Points", []) or []:
        loc = point[0] if isinstance(point, (list, tuple)) and point else point
        if isinstance(loc, Mapping):
            out.append((as_float(loc.get("x")), as_float(loc.get("y"))))
        elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
            out.append((as_float(loc[0]), as_float(loc[1])))
    return out


def _load_hdmap_dict(path: Optional[str | Path]):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    if "arr" in data:
        arr = data["arr"]
        if arr.shape == ():
            return dict(arr.item())
        return dict(arr.tolist())
    return {key: data[key] for key in data.files}


class PrivilegedBEVRenderer:
    """Lightweight annotation-based BEV renderer.

    The image convention matches the CarDreamer-style local frame used by the
    rest of the project: ego x points upward in the image and positive local y
    points to the right.
    """

    def __init__(
        self,
        *,
        shape: Tuple[int, int] = (128, 128),
        obs_range: float = 32.0,
        ego_offset: float = 12.0,
        hdmap_path: Optional[str | Path] = None,
    ):
        self.shape = (int(shape[0]), int(shape[1]))
        self.obs_range = float(obs_range)
        self.ego_offset = float(ego_offset)
        self.hdmap = _load_hdmap_dict(hdmap_path)

    @property
    def pixels_per_meter(self) -> float:
        return float(self.shape[0]) / max(self.obs_range, 1e-6)

    def local_to_pixel(self, local_x: float, local_y: float) -> Tuple[int, int]:
        height, width = self.shape
        ppm = self.pixels_per_meter
        ego_px = width * 0.5
        ego_py = height * (1.0 - self.ego_offset / max(self.obs_range, 1e-6))
        px = int(round(ego_px + float(local_y) * ppm))
        py = int(round(ego_py - float(local_x) * ppm))
        return px, py

    def world_to_pixel(self, wx: float, wy: float, ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[int, int]:
        lx, ly = world_to_ego_xy(wx, wy, ego_x, ego_y, ego_yaw)
        return self.local_to_pixel(lx, ly)

    def render(self, annotation: Mapping, *, hdmap_path: Optional[str | Path] = None) -> np.ndarray:
        cv2 = _import_cv2()
        if hdmap_path is not None:
            self.hdmap = _load_hdmap_dict(hdmap_path)
        height, width = self.shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        ego_x = as_float(annotation.get("x"))
        ego_y = as_float(annotation.get("y"))
        ego_yaw = yaw_to_rad(annotation.get("theta"))
        self._draw_hdmap(cv2, image, ego_x, ego_y, ego_yaw)
        self._draw_actors(cv2, image, annotation, ego_x, ego_y, ego_yaw)
        return image

    def _draw_hdmap(self, cv2, image: np.ndarray, ego_x: float, ego_y: float, ego_yaw: float) -> None:
        if not self.hdmap:
            return
        for road in self.hdmap.values():
            if not isinstance(road, Mapping):
                continue
            for lane_id, items in road.items():
                if str(lane_id).lower().startswith("trigger"):
                    continue
                if not isinstance(items, Sequence):
                    continue
                for item in items:
                    if not isinstance(item, Mapping):
                        continue
                    points = _points_from_hdmap_item(item)
                    if len(points) < 2:
                        continue
                    pix = np.asarray(
                        [self.world_to_pixel(x, y, ego_x, ego_y, ego_yaw) for x, y in points],
                        dtype=np.int32,
                    )
                    in_frame = (
                        (pix[:, 0] >= -image.shape[1])
                        & (pix[:, 0] <= image.shape[1] * 2)
                        & (pix[:, 1] >= -image.shape[0])
                        & (pix[:, 1] <= image.shape[0] * 2)
                    )
                    if not in_frame.any():
                        continue
                    item_type = str(item.get("Type", "")).lower()
                    if item_type == "center":
                        color: Color = (80, 80, 80)
                        thickness = 2
                    elif "solid" in item_type or "broken" in item_type:
                        color = (180, 180, 180)
                        thickness = 1
                    else:
                        color = (95, 95, 95)
                        thickness = 1
                    cv2.polylines(image, [pix], False, color, thickness, lineType=cv2.LINE_AA)

    def _draw_actors(self, cv2, image: np.ndarray, annotation: Mapping, ego_x: float, ego_y: float, ego_yaw: float) -> None:
        for bbox in annotation.get("bounding_boxes", []) or []:
            if not isinstance(bbox, Mapping):
                continue
            cls = str(bbox.get("class", "")).lower()
            if cls not in {"ego_vehicle", "vehicle", "walker", "pedestrian"}:
                continue
            color: Color
            if cls == "ego_vehicle":
                color = (220, 40, 40)
            elif cls in {"walker", "pedestrian"}:
                color = (60, 120, 220)
            else:
                color = (30, 190, 75)
            poly = self._bbox_polygon_pixels(bbox, ego_x, ego_y, ego_yaw)
            if poly is not None:
                cv2.fillPoly(image, [poly], color, lineType=cv2.LINE_AA)

    def _bbox_polygon_pixels(self, bbox: Mapping, ego_x: float, ego_y: float, ego_yaw: float):
        if "world_cord" in bbox and bbox["world_cord"]:
            points = []
            for point in bbox["world_cord"]:
                if isinstance(point, Mapping):
                    points.append((as_float(point.get("x")), as_float(point.get("y"))))
                elif isinstance(point, Sequence) and len(point) >= 2:
                    points.append((as_float(point[0]), as_float(point[1])))
            if len(points) >= 4:
                return np.asarray(
                    [self.world_to_pixel(x, y, ego_x, ego_y, ego_yaw) for x, y in points[:4]],
                    dtype=np.int32,
                )

        center = bbox.get("center", bbox.get("location", None))
        if center is None:
            return None
        if isinstance(center, Mapping):
            cx, cy = as_float(center.get("x")), as_float(center.get("y"))
        elif isinstance(center, Sequence) and len(center) >= 2:
            cx, cy = as_float(center[0]), as_float(center[1])
        else:
            return None

        extent = bbox.get("extent", [2.25, 0.9, 0.8])
        if isinstance(extent, Mapping):
            ex, ey = as_float(extent.get("x"), 2.25), as_float(extent.get("y"), 0.9)
        elif isinstance(extent, Sequence) and len(extent) >= 2:
            ex, ey = as_float(extent[0], 2.25), as_float(extent[1], 0.9)
        else:
            ex, ey = 2.25, 0.9
        yaw = yaw_to_rad(0.0)
        rotation = bbox.get("rotation")
        if isinstance(rotation, Mapping):
            yaw = yaw_to_rad(rotation.get("yaw", 0.0))
        elif isinstance(rotation, Sequence) and len(rotation) >= 3:
            yaw = yaw_to_rad(rotation[2])
        corners = np.asarray([[ex, ey], [ex, -ey], [-ex, -ey], [-ex, ey]], dtype=np.float32)
        rot = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float32,
        )
        world = corners @ rot.T + np.asarray([cx, cy], dtype=np.float32)
        return np.asarray(
            [self.world_to_pixel(float(x), float(y), ego_x, ego_y, ego_yaw) for x, y in world],
            dtype=np.int32,
        )


def render_privileged_bev(
    annotation: Mapping,
    *,
    shape: Tuple[int, int] = (128, 128),
    hdmap_path: Optional[str | Path] = None,
    obs_range: float = 32.0,
    ego_offset: float = 12.0,
) -> np.ndarray:
    return PrivilegedBEVRenderer(shape=shape, obs_range=obs_range, ego_offset=ego_offset, hdmap_path=hdmap_path).render(annotation)
