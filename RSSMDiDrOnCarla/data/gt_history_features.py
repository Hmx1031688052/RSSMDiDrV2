"""GT state-history features for planner baselines."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


EGO_HISTORY_FIELDS = (
    "valid",
    "rel_x",
    "rel_y",
    "sin_yaw",
    "cos_yaw",
    "speed",
    "yawrate",
    "action_acc",
    "action_steer",
)


def _array(chunk: Dict[str, np.ndarray], key: str, length: int, default: float = 0.0) -> np.ndarray:
    value = chunk.get(key)
    if value is None:
        return np.full((length,), default, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 0:
        return np.full((length,), float(value), dtype=np.float32)
    return value.reshape(value.shape[0], -1)[:length, 0].astype(np.float32)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _ego_history(chunk: Dict[str, np.ndarray], history_length: int, waypoint_scale: float) -> np.ndarray:
    length = len(np.asarray(chunk["expert_waypoints8"]))
    ego_x = _array(chunk, "ego_x", length)
    ego_y = _array(chunk, "ego_y", length)
    ego_yaw = _array(chunk, "ego_yaw", length)
    ego_speed = _array(chunk, "ego_speed", length)
    ego_yawrate = _array(chunk, "ego_yawrate", length)
    action = np.asarray(chunk.get("action", np.zeros((length, 2), dtype=np.float32)), dtype=np.float32)
    if action.ndim != 2 or action.shape[1] < 2:
        action = np.zeros((length, 2), dtype=np.float32)
    action = action[:length, :2]

    out = np.zeros((length, history_length, len(EGO_HISTORY_FIELDS)), dtype=np.float32)
    for t in range(length):
        yaw = float(ego_yaw[t])
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        for slot, src in enumerate(range(t - history_length + 1, t + 1)):
            if src < 0:
                continue
            dx = float(ego_x[src] - ego_x[t])
            dy = float(ego_y[src] - ego_y[t])
            rel_x = cos_y * dx + sin_y * dy
            rel_y = -sin_y * dx + cos_y * dy
            dyaw = float(_wrap_angle(np.asarray(ego_yaw[src] - ego_yaw[t], dtype=np.float32)))
            out[t, slot] = np.asarray(
                [
                    1.0,
                    rel_x / waypoint_scale,
                    rel_y / waypoint_scale,
                    math.sin(dyaw),
                    math.cos(dyaw),
                    float(ego_speed[src]) / 15.0,
                    float(ego_yawrate[src]) / 3.0,
                    float(action[src, 0]),
                    float(action[src, 1]),
                ],
                dtype=np.float32,
            )
    return out.reshape(length, -1)


def _normalize_neighbor_local(neighbor: np.ndarray, waypoint_scale: float) -> np.ndarray:
    out = np.asarray(neighbor, dtype=np.float32).copy()
    if out.ndim != 2:
        raise ValueError(f"Expected neighbor_vehicles_local [T, K*11], got {neighbor.shape}")
    if out.shape[1] % 11 != 0:
        raise ValueError(f"Expected neighbor feature dim divisible by 11, got {out.shape[1]}")
    view = out.reshape(out.shape[0], -1, 11)
    view[..., 1:3] /= float(waypoint_scale)
    view[..., 3:5] /= 15.0
    view[..., 7] /= 6.0
    view[..., 8] /= 3.0
    view[..., 9] /= 5.0
    view[..., 10] /= 3.0
    return view.reshape(out.shape[0], -1)


def _history_stack(values: np.ndarray, history_length: int) -> np.ndarray:
    length, dim = values.shape
    out = np.zeros((length, history_length, dim), dtype=np.float32)
    for t in range(length):
        for slot, src in enumerate(range(t - history_length + 1, t + 1)):
            if src >= 0:
                out[t, slot] = values[src]
    return out.reshape(length, history_length * dim)


def _neighbor_world_id_aligned_history(
    chunk: Dict[str, np.ndarray],
    history_length: int,
    waypoint_scale: float,
) -> np.ndarray:
    """Track current K neighbors backward by actor id and express them in current ego frame."""

    world = np.asarray(chunk["neighbor_vehicles_world"], dtype=np.float32)
    if world.ndim != 2:
        raise ValueError(f"Expected neighbor_vehicles_world [T, K*12], got {world.shape}")
    if world.shape[1] % 12 != 0:
        raise ValueError(f"Expected neighbor feature dim divisible by 12, got {world.shape[1]}")

    length = len(np.asarray(chunk["expert_waypoints8"]))
    world = world[:length].reshape(length, -1, 12)
    neighbor_k = world.shape[1]
    ego_x = _array(chunk, "ego_x", length)
    ego_y = _array(chunk, "ego_y", length)
    ego_yaw = _array(chunk, "ego_yaw", length)
    out = np.zeros((length, history_length, neighbor_k, 11), dtype=np.float32)

    id_to_slot = []
    for t in range(length):
        mapping = {}
        for slot, veh in enumerate(world[t]):
            if veh[0] > 0.5:
                mapping[int(round(float(veh[1])))] = slot
        id_to_slot.append(mapping)

    for t in range(length):
        current_ids = [int(round(float(veh[1]))) if veh[0] > 0.5 else 0 for veh in world[t]]
        yaw = float(ego_yaw[t])
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        for hist_slot, src in enumerate(range(t - history_length + 1, t + 1)):
            if src < 0:
                continue
            for current_slot, actor_id in enumerate(current_ids):
                if actor_id == 0:
                    continue
                src_slot = id_to_slot[src].get(actor_id)
                if src_slot is None:
                    continue

                veh = world[src, src_slot]
                dx = float(veh[2] - ego_x[t])
                dy = float(veh[3] - ego_y[t])
                rel_x = cos_y * dx + sin_y * dy
                rel_y = -sin_y * dx + cos_y * dy
                vx_e = cos_y * float(veh[4]) + sin_y * float(veh[5])
                vy_e = -sin_y * float(veh[4]) + cos_y * float(veh[5])
                veh_yaw = math.atan2(float(veh[6]), float(veh[7]))
                rel_yaw = float(_wrap_angle(np.asarray(veh_yaw - yaw, dtype=np.float32)))

                out[t, hist_slot, current_slot] = np.asarray(
                    [
                        1.0,
                        rel_x / waypoint_scale,
                        rel_y / waypoint_scale,
                        vx_e / 15.0,
                        vy_e / 15.0,
                        math.sin(rel_yaw),
                        math.cos(rel_yaw),
                        float(veh[8]) / 6.0,
                        float(veh[9]) / 3.0,
                        float(veh[10]) / 5.0,
                        float(veh[11]) / 3.0,
                    ],
                    dtype=np.float32,
                )

    return out.reshape(length, history_length * neighbor_k * 11)


def build_gt_history_features(
    chunk: Dict[str, np.ndarray],
    history_length: int = 10,
    waypoint_scale: float = 30.0,
    include_neighbors: bool = True,
    align_neighbor_ids: bool = True,
) -> np.ndarray:
    """Build `[T, D]` features from ego and neighbor GT state history.

    Ego history is expressed in the current ego frame. When available, neighbor
    history tracks the current nearest vehicles backward using actor ids from
    `neighbor_vehicles_world`; this avoids mixing different vehicles when the
    K-nearest ordering changes over time.
    """

    history_length = int(history_length)
    if history_length <= 0:
        raise ValueError(f"history_length must be positive, got {history_length}")

    ego = _ego_history(chunk, history_length=history_length, waypoint_scale=waypoint_scale)
    parts = [ego]
    if include_neighbors and align_neighbor_ids and "neighbor_vehicles_world" in chunk:
        parts.append(
            _neighbor_world_id_aligned_history(
                chunk,
                history_length=history_length,
                waypoint_scale=waypoint_scale,
            )
        )
    elif include_neighbors and "neighbor_vehicles_local" in chunk:
        length = len(ego)
        neighbor = np.asarray(chunk["neighbor_vehicles_local"], dtype=np.float32)[:length]
        neighbor = _normalize_neighbor_local(neighbor, waypoint_scale=waypoint_scale)
        parts.append(_history_stack(neighbor, history_length=history_length))
    return np.concatenate(parts, axis=-1).astype(np.float32)
