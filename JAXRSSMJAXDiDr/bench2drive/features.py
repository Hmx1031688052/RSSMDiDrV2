"""Feature builders shared by Bench2Drive dataset conversion and agents."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from JAXRSSMJAXDiDr.data.polyplanner_targets import waypoints8_to_trajectory


NUM_WAYPOINTS = 8
NEIGHBOR_FIELDS_LOCAL = 11
NEIGHBOR_FIELDS_WORLD = 12


def as_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if not math.isfinite(out):
            return float(default)
        return out
    except (TypeError, ValueError):
        return float(default)


def yaw_to_rad(value, default: float = 0.0) -> float:
    yaw = as_float(value, default)
    if abs(yaw) > 2.0 * math.pi:
        yaw = math.radians(yaw)
    return float((yaw + math.pi) % (2.0 * math.pi) - math.pi)


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def world_to_ego_xy(wx: float, wy: float, ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[float, float]:
    dx = float(wx) - float(ego_x)
    dy = float(wy) - float(ego_y)
    cos_y = math.cos(float(ego_yaw))
    sin_y = math.sin(float(ego_yaw))
    return cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy


def ego_to_world_xy(x: float, y: float, ego_x: float, ego_y: float, ego_yaw: float) -> Tuple[float, float]:
    cos_y = math.cos(float(ego_yaw))
    sin_y = math.sin(float(ego_yaw))
    return float(ego_x) + cos_y * x - sin_y * y, float(ego_y) + sin_y * x + cos_y * y


def _vector_xy(value, default=(0.0, 0.0)) -> Tuple[float, float]:
    if isinstance(value, Mapping):
        return as_float(value.get("x"), default[0]), as_float(value.get("y"), default[1])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return as_float(value[0], default[0]), as_float(value[1], default[1])
    return float(default[0]), float(default[1])


def _vector_z(value, default: float = 0.0) -> float:
    if isinstance(value, Mapping):
        return as_float(value.get("z"), default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return as_float(value[2], default)
    return float(default)


def _rotation_yaw(value, default: float = 0.0) -> float:
    if isinstance(value, Mapping):
        return yaw_to_rad(value.get("yaw"), default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        return yaw_to_rad(value[2], default)
    return yaw_to_rad(value, default)


def _extent_lw(value) -> Tuple[float, float]:
    if isinstance(value, Mapping):
        x = as_float(value.get("x"), 2.25)
        y = as_float(value.get("y"), 0.9)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        x = as_float(value[0], 2.25)
        y = as_float(value[1], 0.9)
    else:
        x, y = 2.25, 0.9
    return max(0.1, 2.0 * x), max(0.1, 2.0 * y)


def extract_ego_arrays(annotations: Sequence[Mapping]) -> Dict[str, np.ndarray]:
    length = len(annotations)
    ego_x = np.zeros(length, dtype=np.float32)
    ego_y = np.zeros(length, dtype=np.float32)
    ego_yaw = np.zeros(length, dtype=np.float32)
    ego_speed = np.zeros(length, dtype=np.float32)
    ego_yawrate = np.zeros(length, dtype=np.float32)
    for idx, anno in enumerate(annotations):
        ego_x[idx] = as_float(anno.get("x"))
        ego_y[idx] = as_float(anno.get("y"))
        ego_yaw[idx] = yaw_to_rad(anno.get("theta"))
        ego_speed[idx] = max(0.0, as_float(anno.get("speed")))
        angular_velocity = anno.get("angular_velocity", None)
        ego_yawrate[idx] = math.radians(_vector_z(angular_velocity)) if angular_velocity is not None else 0.0

    missing_yaw = ~np.isfinite(ego_yaw)
    if missing_yaw.any() and length > 1:
        dx = np.gradient(ego_x.astype(np.float64))
        dy = np.gradient(ego_y.astype(np.float64))
        fallback = np.arctan2(dy, dx).astype(np.float32)
        ego_yaw[missing_yaw] = fallback[missing_yaw]

    return {
        "ego_x": ego_x,
        "ego_y": ego_y,
        "ego_yaw": ego_yaw,
        "ego_speed": ego_speed,
        "ego_yawrate": ego_yawrate,
    }


def build_actions(annotations: Sequence[Mapping]) -> np.ndarray:
    actions = np.zeros((len(annotations), 2), dtype=np.float32)
    for idx, anno in enumerate(annotations):
        throttle = np.clip(as_float(anno.get("throttle")), 0.0, 1.0)
        brake = np.clip(as_float(anno.get("brake")), 0.0, 1.0)
        steer = np.clip(as_float(anno.get("steer")), -1.0, 1.0)
        actions[idx, 0] = np.clip(3.0 * (throttle - brake), -3.0, 3.0)
        actions[idx, 1] = np.clip(-steer, -1.0, 1.0)
    return actions


def build_future_waypoints8(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_yaw: np.ndarray,
    *,
    waypoint_scale: float = 30.0,
    dt: float = 0.1,
    waypoint_interval: int = 5,
    num_waypoints: int = NUM_WAYPOINTS,
) -> np.ndarray:
    length = int(len(ego_x))
    out = np.zeros((length, int(num_waypoints) * 2), dtype=np.float32)
    if length == 0:
        return out
    for t in range(length):
        pts = []
        for k in range(int(num_waypoints)):
            future = t + (k + 1) * int(waypoint_interval)
            if future < length:
                fx = float(ego_x[future])
                fy = float(ego_y[future])
            else:
                remaining = length - 1
                lookback = min(5, max(1, length - 1))
                base = max(0, remaining - lookback)
                if remaining > base:
                    vx = (float(ego_x[remaining]) - float(ego_x[base])) / max(1, remaining - base) / max(float(dt), 1e-6)
                    vy = (float(ego_y[remaining]) - float(ego_y[base])) / max(1, remaining - base) / max(float(dt), 1e-6)
                else:
                    vx, vy = 0.0, 0.0
                extra_steps = future - remaining
                fx = float(ego_x[remaining]) + vx * extra_steps * float(dt)
                fy = float(ego_y[remaining]) + vy * extra_steps * float(dt)
                dx = fx - float(ego_x[remaining])
                dy = fy - float(ego_y[remaining])
                dist = math.hypot(dx, dy)
                if dist > float(waypoint_scale):
                    fx = float(ego_x[remaining]) + dx / dist * float(waypoint_scale)
                    fy = float(ego_y[remaining]) + dy / dist * float(waypoint_scale)
            lx, ly = world_to_ego_xy(fx, fy, float(ego_x[t]), float(ego_y[t]), float(ego_yaw[t]))
            pts.extend([lx / float(waypoint_scale), ly / float(waypoint_scale)])
        out[t] = np.clip(np.asarray(pts, dtype=np.float32), -1.0, 1.0)
    return out


def _iter_vehicle_boxes(annotation: Mapping) -> Iterable[Mapping]:
    for bbox in annotation.get("bounding_boxes", []) or []:
        cls = str(bbox.get("class", bbox.get("type", ""))).lower()
        if cls in {"vehicle", "car", "ego_vehicle"}:
            yield bbox


def _actor_id(bbox: Mapping) -> Optional[int]:
    value = bbox.get("id")
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _bbox_xy(bbox: Mapping) -> Tuple[float, float]:
    for key in ("center", "location"):
        if key in bbox:
            return _vector_xy(bbox[key])
    return 0.0, 0.0


def _bbox_yaw(bbox: Mapping) -> float:
    return _rotation_yaw(bbox.get("rotation", 0.0))


def _finite_difference_actor_states(annotations: Sequence[Mapping], dt: float) -> Dict[int, Dict[int, Tuple[float, float, float, float]]]:
    histories: Dict[int, List[Tuple[int, float, float, float]]] = {}
    for idx, anno in enumerate(annotations):
        for bbox in _iter_vehicle_boxes(anno):
            actor_id = _actor_id(bbox)
            if actor_id is None:
                continue
            x, y = _bbox_xy(bbox)
            yaw = _bbox_yaw(bbox)
            histories.setdefault(actor_id, []).append((idx, x, y, yaw))

    states: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    for actor_id, rows in histories.items():
        rows.sort(key=lambda row: row[0])
        for pos, (idx, x, y, yaw) in enumerate(rows):
            if len(rows) == 1:
                vx = vy = yawrate = 0.0
            elif pos == 0:
                nidx, nx, ny, nyaw = rows[pos + 1]
                scale = 1.0 / max((nidx - idx) * float(dt), 1e-6)
                vx, vy = (nx - x) * scale, (ny - y) * scale
                yawrate = wrap_angle(nyaw - yaw) * scale
            elif pos == len(rows) - 1:
                pidx, px, py, pyaw = rows[pos - 1]
                scale = 1.0 / max((idx - pidx) * float(dt), 1e-6)
                vx, vy = (x - px) * scale, (y - py) * scale
                yawrate = wrap_angle(yaw - pyaw) * scale
            else:
                pidx, px, py, pyaw = rows[pos - 1]
                nidx, nx, ny, nyaw = rows[pos + 1]
                scale = 1.0 / max((nidx - pidx) * float(dt), 1e-6)
                vx, vy = (nx - px) * scale, (ny - py) * scale
                yawrate = wrap_angle(nyaw - pyaw) * scale
            states.setdefault(idx, {})[actor_id] = (float(vx), float(vy), 0.0, float(yawrate))
    return states


def build_neighbor_features_from_annotations(
    annotations: Sequence[Mapping],
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_yaw: np.ndarray,
    *,
    neighbor_k: int = 8,
    neighbor_radius: float = 50.0,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    length = len(annotations)
    local = np.zeros((length, int(neighbor_k) * NEIGHBOR_FIELDS_LOCAL), dtype=np.float32)
    world = np.zeros((length, int(neighbor_k) * NEIGHBOR_FIELDS_WORLD), dtype=np.float32)
    diff_states = _finite_difference_actor_states(annotations, dt)

    for t, anno in enumerate(annotations):
        rows = []
        for bbox in _iter_vehicle_boxes(anno):
            cls = str(bbox.get("class", "")).lower()
            if cls == "ego_vehicle":
                continue
            actor_id = _actor_id(bbox)
            if actor_id is None:
                continue
            wx, wy = _bbox_xy(bbox)
            dx = wx - float(ego_x[t])
            dy = wy - float(ego_y[t])
            dist = math.hypot(dx, dy)
            if dist > float(neighbor_radius):
                continue
            yaw = _bbox_yaw(bbox)
            speed = as_float(bbox.get("speed"), 0.0)
            default_vx = speed * math.cos(yaw)
            default_vy = speed * math.sin(yaw)
            vx, vy, accel, yawrate = diff_states.get(t, {}).get(actor_id, (default_vx, default_vy, 0.0, 0.0))
            length_m, width_m = _extent_lw(bbox.get("extent"))
            lx, ly = world_to_ego_xy(wx, wy, float(ego_x[t]), float(ego_y[t]), float(ego_yaw[t]))
            vx_e = math.cos(float(ego_yaw[t])) * vx + math.sin(float(ego_yaw[t])) * vy
            vy_e = -math.sin(float(ego_yaw[t])) * vx + math.cos(float(ego_yaw[t])) * vy
            rel_yaw = wrap_angle(yaw - float(ego_yaw[t]))
            rows.append(
                (
                    dist,
                    [
                        1.0,
                        lx,
                        ly,
                        vx_e,
                        vy_e,
                        math.sin(rel_yaw),
                        math.cos(rel_yaw),
                        length_m,
                        width_m,
                        accel,
                        yawrate,
                    ],
                    [
                        1.0,
                        float(actor_id),
                        wx,
                        wy,
                        vx,
                        vy,
                        math.sin(yaw),
                        math.cos(yaw),
                        length_m,
                        width_m,
                        accel,
                        yawrate,
                    ],
                )
            )
        rows.sort(key=lambda row: row[0])
        for slot, (_, local_row, world_row) in enumerate(rows[: int(neighbor_k)]):
            local[t, slot * NEIGHBOR_FIELDS_LOCAL : (slot + 1) * NEIGHBOR_FIELDS_LOCAL] = np.asarray(local_row, dtype=np.float32)
            world[t, slot * NEIGHBOR_FIELDS_WORLD : (slot + 1) * NEIGHBOR_FIELDS_WORLD] = np.asarray(world_row, dtype=np.float32)
    return local, world


def sample_polyline_global_path(
    route_xy: Sequence[Tuple[float, float]],
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    *,
    num: int = 50,
    waypoint_scale: float = 30.0,
    lookback_m: float = 5.0,
    lookahead_m: float = 45.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    num = int(num)
    out = np.zeros((num, 2), dtype=np.float32)
    mask = np.zeros((num,), dtype=np.float32)
    if len(route_xy) < 2:
        return out, mask, 0.0

    xy = np.asarray(route_xy, dtype=np.float64)
    seg = xy[1:] - xy[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 1e-6:
        return out, mask, 0.0

    ego = np.asarray([ego_x, ego_y], dtype=np.float64)
    best_s = 0.0
    best_dist2 = float("inf")
    for idx, length in enumerate(seg_len):
        if length <= 1e-6:
            continue
        alpha = float(np.clip(np.dot(ego - xy[idx], seg[idx]) / max(float(length * length), 1e-12), 0.0, 1.0))
        proj = xy[idx] + alpha * seg[idx]
        dist2 = float(np.sum((ego - proj) ** 2))
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_s = float(cum[idx] + alpha * length)

    window_start = best_s - float(lookback_m)
    window_end = best_s + float(lookahead_m)
    for slot in range(num):
        target_s = best_s if num == 1 else window_start + (window_end - window_start) * slot / max(num - 1, 1)
        if target_s < 0.0 or target_s > total:
            continue
        j = max(0, min(int(np.searchsorted(cum, target_s, side="right") - 1), len(seg_len) - 1))
        denom = max(float(seg_len[j]), 1e-6)
        alpha = float(np.clip((target_s - float(cum[j])) / denom, 0.0, 1.0))
        wx, wy = xy[j] + alpha * seg[j]
        lx, ly = world_to_ego_xy(wx, wy, ego_x, ego_y, ego_yaw)
        out[slot] = [np.clip(lx / waypoint_scale, -1.0, 1.0), np.clip(ly / waypoint_scale, -1.0, 1.0)]
        mask[slot] = 1.0
    remaining = max(0.0, total - best_s)
    return out, mask, float(remaining)


def sample_route_waypoints8(
    route_xy: Sequence[Tuple[float, float]],
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    *,
    waypoint_scale: float = 30.0,
    num_waypoints: int = NUM_WAYPOINTS,
    ref_step_m: float = 5.0,
) -> np.ndarray:
    out = np.zeros((int(num_waypoints) * 2,), dtype=np.float32)
    if len(route_xy) < 2:
        return out
    xy = np.asarray(route_xy, dtype=np.float64)
    seg = xy[1:] - xy[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 1e-6:
        return out
    ego = np.asarray([ego_x, ego_y], dtype=np.float64)
    best_s = 0.0
    best_dist2 = float("inf")
    for idx, length in enumerate(seg_len):
        if length <= 1e-6:
            continue
        alpha = float(np.clip(np.dot(ego - xy[idx], seg[idx]) / max(float(length * length), 1e-12), 0.0, 1.0))
        proj = xy[idx] + alpha * seg[idx]
        dist2 = float(np.sum((ego - proj) ** 2))
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_s = float(cum[idx] + alpha * length)
    pts = []
    for slot in range(int(num_waypoints)):
        target_s = min(best_s + (slot + 1) * float(ref_step_m), total)
        j = max(0, min(int(np.searchsorted(cum, target_s, side="right") - 1), len(seg_len) - 1))
        denom = max(float(seg_len[j]), 1e-6)
        alpha = float(np.clip((target_s - float(cum[j])) / denom, 0.0, 1.0))
        wx, wy = xy[j] + alpha * seg[j]
        lx, ly = world_to_ego_xy(wx, wy, ego_x, ego_y, ego_yaw)
        pts.extend([lx / float(waypoint_scale), ly / float(waypoint_scale)])
    return np.clip(np.asarray(pts, dtype=np.float32), -1.0, 1.0)


def build_route_features(
    annotations: Sequence[Mapping],
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_yaw: np.ndarray,
    *,
    waypoint_scale: float = 30.0,
    num_global_points: int = 50,
) -> Dict[str, np.ndarray]:
    route = [(as_float(anno.get("x")), as_float(anno.get("y"))) for anno in annotations]
    length = len(annotations)
    global_path = np.zeros((length, int(num_global_points), 2), dtype=np.float32)
    global_mask = np.zeros((length, int(num_global_points)), dtype=np.float32)
    route_remaining = np.zeros((length, 1), dtype=np.float32)
    route_waypoints8 = build_future_waypoints8(
        ego_x,
        ego_y,
        ego_yaw,
        waypoint_scale=waypoint_scale,
    )
    for idx in range(length):
        gp, mask, remaining = sample_polyline_global_path(
            route,
            float(ego_x[idx]),
            float(ego_y[idx]),
            float(ego_yaw[idx]),
            num=num_global_points,
            waypoint_scale=waypoint_scale,
        )
        global_path[idx] = gp
        global_mask[idx] = mask
        route_remaining[idx, 0] = np.clip(remaining / 200.0, 0.0, 1.0)
    return {
        "route_waypoints8": route_waypoints8.astype(np.float32),
        "global_path_ego": global_path,
        "global_path_ego_mask": global_mask,
        "target_region": np.zeros((length, 1), dtype=np.float32),
        "route_remaining": route_remaining,
    }


def build_online_neighbor_features_from_carla(
    ego_actor,
    world,
    *,
    neighbor_k: int = 8,
    neighbor_radius: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    ego_tf = ego_actor.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = math.radians(float(ego_tf.rotation.yaw))
    ego_id = ego_actor.id
    rows = []
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego_id:
            continue
        tf = actor.get_transform()
        loc = tf.location
        dx = float(loc.x - ego_loc.x)
        dy = float(loc.y - ego_loc.y)
        dist = math.hypot(dx, dy)
        if dist > float(neighbor_radius):
            continue
        yaw = math.radians(float(tf.rotation.yaw))
        vel = actor.get_velocity()
        vx = float(vel.x)
        vy = float(vel.y)
        accel_vec = actor.get_acceleration()
        accel = float(accel_vec.x * math.cos(yaw) + accel_vec.y * math.sin(yaw))
        yawrate = math.radians(float(actor.get_angular_velocity().z))
        bb = actor.bounding_box.extent
        length_m = max(0.1, 2.0 * float(bb.x))
        width_m = max(0.1, 2.0 * float(bb.y))
        lx, ly = world_to_ego_xy(float(loc.x), float(loc.y), float(ego_loc.x), float(ego_loc.y), ego_yaw)
        vx_e = math.cos(ego_yaw) * vx + math.sin(ego_yaw) * vy
        vy_e = -math.sin(ego_yaw) * vx + math.cos(ego_yaw) * vy
        rel_yaw = wrap_angle(yaw - ego_yaw)
        rows.append(
            (
                dist,
                [1.0, lx, ly, vx_e, vy_e, math.sin(rel_yaw), math.cos(rel_yaw), length_m, width_m, accel, yawrate],
                [1.0, float(actor.id), float(loc.x), float(loc.y), vx, vy, math.sin(yaw), math.cos(yaw), length_m, width_m, accel, yawrate],
            )
        )
    rows.sort(key=lambda row: row[0])
    local = np.zeros((int(neighbor_k) * NEIGHBOR_FIELDS_LOCAL,), dtype=np.float32)
    world_out = np.zeros((int(neighbor_k) * NEIGHBOR_FIELDS_WORLD,), dtype=np.float32)
    for slot, (_, local_row, world_row) in enumerate(rows[: int(neighbor_k)]):
        local[slot * NEIGHBOR_FIELDS_LOCAL : (slot + 1) * NEIGHBOR_FIELDS_LOCAL] = np.asarray(local_row, dtype=np.float32)
        world_out[slot * NEIGHBOR_FIELDS_WORLD : (slot + 1) * NEIGHBOR_FIELDS_WORLD] = np.asarray(world_row, dtype=np.float32)
    return local, world_out


def trajectory_from_waypoints8(expert_waypoints8: np.ndarray, waypoint_scale: float = 30.0) -> np.ndarray:
    return waypoints8_to_trajectory(np.asarray(expert_waypoints8, dtype=np.float32), waypoint_scale=waypoint_scale)
