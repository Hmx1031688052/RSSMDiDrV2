"""Differentiable waypoint-to-control bridge in JAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .jax_didr_planner import JAXDiDrConfig


def normalized_acc_to_phys(action, acc_min=-3.0, acc_max=3.0):
    return (action[..., 0] + 1.0) * 0.5 * (float(acc_max) - float(acc_min)) + float(acc_min)


def apply_plan_sign(waypoints, plan_x_sign=1.0, plan_y_sign=1.0):
    sign = jnp.asarray([float(plan_x_sign), float(plan_y_sign)], dtype=waypoints.dtype)
    return waypoints * sign


def differentiable_pidpp(poses_xy, ego_speed, config: JAXDiDrConfig):
    """TorchDiDrOnCarla DifferentiablePIDPP port.

    Args:
      poses_xy: `[B, P, 2]` or `[B, M, P, 2]` ego-frame waypoints in meters.
      ego_speed: `[B, 1]` or `[B]` current speed in m/s.

    Returns:
      normalized action `[acc_norm, steer]` with shape `[B, 2]` or `[B, M, 2]`.
    """

    squeeze = poses_xy.ndim == 3
    if squeeze:
        poses_xy = poses_xy[:, None]
    batch, modes, poses, _ = poses_xy.shape
    speed = jnp.maximum(ego_speed.reshape((batch, -1))[:, 0], 0.0)

    origin = jnp.zeros((batch, modes, 1, 2), dtype=poses_xy.dtype)
    points = jnp.concatenate([origin, poses_xy], axis=2)
    seg = jnp.diff(points, axis=2)
    seg_len = jnp.maximum(jnp.linalg.norm(seg, axis=-1), 1e-4)

    speed_segments = jnp.clip(seg_len / max(float(config.waypoint_dt), 1e-6), 0.0, float(config.ctrl_target_speed_max))
    target_speed = speed_segments[:, :, : min(4, poses)].mean(axis=-1)
    acc_phys = jnp.clip(
        float(config.speed_kp) * (target_speed - speed[:, None]),
        float(config.ctrl_acc_min),
        float(config.ctrl_acc_max),
    )
    acc_norm = 2.0 * (acc_phys - float(config.ctrl_acc_min)) / max(
        float(config.ctrl_acc_max) - float(config.ctrl_acc_min), 1e-6
    ) - 1.0

    cumulative = jnp.cumsum(seg_len, axis=-1)
    base_ld = jnp.clip(
        float(config.lookahead_min) + float(config.lookahead_gain) * speed,
        float(config.lookahead_min),
        float(config.lookahead_max),
    )
    scales = jnp.asarray([0.75, 1.0, 1.35], dtype=poses_xy.dtype)
    lookahead = base_ld[:, None] * scales[None]
    errors = jnp.abs(cumulative[:, :, None, :] - lookahead[:, None, :, None])
    weights = jax.nn.softmax(-errors / max(float(config.ctrl_soft_lookup_temp), 1e-6), axis=-1)
    targets = (weights[..., None] * poses_xy[:, :, None]).sum(axis=-2)
    mix = jnp.asarray([0.25, 0.50, 0.25], dtype=poses_xy.dtype)
    target = (targets * mix[None, None, :, None]).sum(axis=2)

    dist2 = jnp.maximum(jnp.square(target[..., 0]) + jnp.square(target[..., 1]), 1e-6)
    curvature = 2.0 * target[..., 1] / dist2
    steer_angle = jnp.arctan(float(config.wheelbase) * curvature)
    steer = jnp.clip(
        float(config.steer_sign) * float(config.steer_gain) * steer_angle / max(float(config.max_steer_rad), 1e-6),
        -1.0,
        1.0,
    )
    action = jnp.stack([jnp.clip(acc_norm, -1.0, 1.0), steer], axis=-1)
    return action[:, 0] if squeeze else action

