from abc import abstractmethod

import math
import numpy as np
from gym import spaces

from .carla_base_env import CarlaBaseEnv
from .toolkit import BasePlanner, TTCCalculator, get_location_distance, get_vehicle_pos, get_vehicle_velocity
import numpy as np


class CarlaWptEnv(CarlaBaseEnv):
    """
    This is the base env for all waypoint following tasks.
    An ``ego_planner`` is required to provide waypoints for the ego vehicle.
    **DO NOT** instantiate this class directly.

    All envs that inherit from this class also inherits the following config parameters:

    * ``reward``: Reward configuration.

        * ``desired_speed``: Desired speed for the ego vehicle.
        * ``scales``: Dictionary of reward scales.

            * ``waypoint``: Reward for reaching waypoints.
            * ``speed``: Reward for speed.
            * ``collision``: Penalty for collision.
            * ``out_of_lane``: Penalty for going out of lane.
            * ``time``: Penalty for each time step.

    * ``terminal``: Terminal condition configuration.

        * ``time_limit``: Maximum number of time steps.
        * ``out_lane_thres``: Distance threshold for going out of lane.

    """

    @abstractmethod
    def get_ego_planner(self) -> BasePlanner:
        """
        Override this method to return the ego vehicle planner.
        The default behavior is to return self.ego_planner.
        """
        return self.ego_planner
    def _clear_waypoint_controller_cache(self):
        """Clear cached waypoint controller state.

        PolyPlannerController stores:
          - ego vehicle actor reference
          - PID integral state
          - steering low-pass state
          - previous target speed

        These must not survive across CARLA episode reset.
        """
        for attr in ("_poly_controller",):
            if hasattr(self, attr):
                delattr(self, attr)

    def _get_or_create_poly_controller(self):
        """Return a PolyPlannerController bound to the current ego vehicle.

        This protects against using a controller that still points to the
        previous episode's destroyed ego actor.
        """
        # from .toolkit.planner.poly_planner_controller import (
        #     PolyPlannerController,
        # )
        from .toolkit.planner.PIDPP import (
            PolyPlannerController,
        )

        vehicle = self.get_ego_vehicle()
        target_speed = float(getattr(self._config, "target_speed", 5.0))

        ctrl = getattr(self, "_poly_controller", None)
        old_vehicle = getattr(ctrl, "_vehicle", None) if ctrl is not None else None

        same_vehicle = (
            old_vehicle is not None
            and getattr(old_vehicle, "id", None) == getattr(vehicle, "id", None)
        )

        if ctrl is None or not same_vehicle:
            ctrl = PolyPlannerController(vehicle, target_speed=target_speed)
            self._poly_controller = ctrl

        return ctrl

    def register_extra_observations(self):
        super().register_extra_observations()

        # External expert trajectory from quintic polynomial planner (world frame)
        self.external_traj_world = None  # list of (x, y) world coords, set by set_expert_trajectory()

        # Register ego world position for post-processing expert trajectories.
        # Always registered regardless of planner_target.enable.
        self._observer.register_simple_handler(
            "ego_x", self._get_ego_x,
            spaces.Box(-np.inf, np.inf, (1,), np.float32),
        )
        self._observer.register_simple_handler(
            "ego_y", self._get_ego_y,
            spaces.Box(-np.inf, np.inf, (1,), np.float32),
        )
        self._observer.register_simple_handler(
            "ego_yaw", self._get_ego_yaw,
            spaces.Box(-np.pi, np.pi, (1,), np.float32),
        )
        self._observer.register_simple_handler(
            "ego_speed", self._get_ego_speed,
            spaces.Box(0.0, 30.0, (1,), np.float32),
        )
        self._observer.register_simple_handler(
            "ego_yawrate", self._get_ego_yawrate,
            spaces.Box(-3.0, 3.0, (1,), np.float32),
        )

        # Register neighbor vehicles in ego frame for collision-aware scoring.
        # K=8 vehicles, 11 fields each:
        #   [valid, x, y, vx, vy, sin_yaw, cos_yaw, L, W, accel, yaw_rate]
        self._neighbor_k = int(getattr(self._config, "neighbor_k", 8))
        self._neighbor_radius = float(getattr(self._config, "neighbor_radius", 50.0))
        self._observer.register_simple_handler(
            "neighbor_vehicles_local",
            self._get_neighbor_vehicles_local,
            spaces.Box(-np.inf, np.inf, (self._neighbor_k * 11,), np.float32),
        )
        # Same surrounding vehicles in world frame, with actor ids for stable
        # offline trajectory targets. Format per vehicle:
        # [valid, actor_id, x, y, vx, vy, sin_yaw, cos_yaw, L, W, accel, yaw_rate]
        self._observer.register_simple_handler(
            "neighbor_vehicles_world",
            self._get_neighbor_vehicles_world,
            spaces.Box(-np.inf, np.inf, (self._neighbor_k * 12,), np.float32),
        )
        self._observer.register_simple_handler(
            "target_region",
            self._get_target_region,
            spaces.Box(0.0, 1.0, (1,), np.float32),
        )
        self._observer.register_simple_handler(
            "route_remaining",
            self._get_route_remaining,
            spaces.Box(0.0, 1.0, (1,), np.float32),
        )

        cfg = getattr(self._config, "planner_target", None)
        if cfg is None or not bool(getattr(cfg, "enable", True)):
            return

        self._planner_num_waypoints = int(getattr(cfg, "num_waypoints", 8))
        self._planner_waypoint_scale = float(getattr(cfg, "waypoint_scale", 30.0))
        self._planner_waypoint_stride = int(getattr(cfg, "waypoint_stride", 5))
        self._planner_expert_source_dt = float(
            getattr(cfg, "expert_source_dt", 0.2))
        self._planner_expert_target_dt = float(
            getattr(cfg, "expert_target_dt", 0.5))
        self._planner_expert_min_first_x = float(
            getattr(cfg, "expert_min_first_x_m", -1000.0))
        self._planner_expert_max_clip_frac = float(
            getattr(cfg, "expert_max_clip_frac", 0.25))
        self._planner_reference_source = str(
            getattr(cfg, "reference_source", "csv_global"))
        self._planner_reference_step_m = float(
            getattr(cfg, "reference_step_m", 0.0))
        self._planner_global_path_num = int(
            getattr(cfg, "global_path_num_points", 50))
        self._planner_global_path_lookback = float(
            getattr(cfg, "global_path_lookback_m", 5.0))
        self._planner_global_path_lookahead = float(
            getattr(cfg, "global_path_lookahead_m", 45.0))

        shape = (self._planner_num_waypoints * 2,)
        space = spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)
        self._observer.register_simple_handler(
            "expert_waypoints8",
            self._get_expert_waypoints8_local,
            space,
        )
        self._observer.register_simple_handler(
            "route_waypoints8",
            self._get_route_waypoints8_local,
            space,
        )

        # Dense local-window global path in ego frame for point-to-polyline
        # distance computation. Samples a window around the ego's projection
        # onto the CSV route, with a validity mask for near-end-of-path cases.
        gp_shape = (self._planner_global_path_num, 2)
        gp_space = spaces.Box(low=-1.0, high=1.0, shape=gp_shape, dtype=np.float32)
        self._observer.register_simple_handler(
            "global_path_ego",
            self._get_global_path_ego,
            gp_space,
        )
        mask_shape = (self._planner_global_path_num,)
        mask_space = spaces.Box(low=0.0, high=1.0, shape=mask_shape, dtype=np.float32)
        self._observer.register_simple_handler(
            "global_path_ego_mask",
            self._get_global_path_ego_mask,
            mask_space,
        )
        # Cache mask across the two handler calls within one step.
        self._global_path_ego_mask_cache = None

    def _world_to_ego_xy(self, x, y):
        ego_tf = self.get_ego_vehicle().get_transform()
        ego_loc = ego_tf.location
        yaw = math.radians(float(ego_tf.rotation.yaw))
        dx = float(x) - float(ego_loc.x)
        dy = float(y) - float(ego_loc.y)
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return local_x, local_y

    def set_expert_trajectory(self, traj_world):
        """Set external expert trajectory from quintic polynomial planner.

        Args:
            traj_world: list of (x, y) tuples in world coordinates,
                        or None to clear and fall back to global route.
        """
        self.external_traj_world = traj_world

    def _get_target_region(self, env_state=None):
        del env_state
        return np.asarray([1.0 if self.is_destination_reached() else 0.0], dtype=np.float32)

    def _get_route_remaining(self, env_state=None):
        del env_state
        norm = 200.0
        reward_cfg = getattr(self._config, "reward", None)
        if reward_cfg is not None:
            try:
                norm = float(reward_cfg.get("route_remaining_norm", norm))
            except Exception:
                norm = float(getattr(reward_cfg, "route_remaining_norm", norm))
        norm = max(norm, 1.0)
        remaining = float(len(getattr(self, "waypoints", []))) / norm
        return np.asarray([np.clip(remaining, 0.0, 1.0)], dtype=np.float32)

    def _get_expert_waypoints8_local(self, env_state=None):
        """Sample 8 waypoints from external trajectory (preferred) or global route."""
        del env_state
        num = int(getattr(self, "_planner_num_waypoints", 8))
        scale = float(getattr(self, "_planner_waypoint_scale", 30.0))

        if (hasattr(self, "external_traj_world")
                and self.external_traj_world is not None
                and len(self.external_traj_world) >= num):
            waypoints = self._sample_trajectory_waypoints(
                self.external_traj_world, num, scale)
            if self._valid_expert_waypoints(waypoints, scale):
                return waypoints
        return self._sample_global_route_waypoints(num, scale)
    
    def _get_route_waypoints8_local(self, env_state=None):
        """Sample 8 normalized reference waypoints for GR-DPPO scoring.

        Prefer the full CSV global path when the planner exposes
        get_global_waypoints(). This avoids using self.waypoints, which is only
        a short local queue and is popped as the ego moves.
        """
        del env_state
        num = int(getattr(self, "_planner_num_waypoints", 8))
        scale = float(getattr(self, "_planner_waypoint_scale", 30.0))

        if getattr(self, "_planner_reference_source", "csv_global") == "csv_global":
            out = self._sample_csv_global_reference_waypoints(num, scale)
            if out is not None:
                return out

        return self._sample_global_route_waypoints(num, scale)


    def _sample_global_route_waypoints(self, num, scale):
        """Fallback: sample from the current local planner queue."""
        stride = int(getattr(self, "_planner_waypoint_stride", 5))
        if not hasattr(self, "waypoints") or len(self.waypoints) == 0:
            return np.zeros((num * 2,), dtype=np.float32)

        pts = []
        for i in range(num):
            idx = min((i + 1) * stride - 1, len(self.waypoints) - 1)
            wp = self.waypoints[idx]
            lx, ly = self._world_to_ego_xy(wp[0], wp[1])
            pts.extend([lx / scale, ly / scale])

        return np.clip(np.asarray(pts, dtype=np.float32), -1.0, 1.0)


    def _sample_csv_global_reference_waypoints(self, num, scale):
        """Sample route_waypoints8 from the full CSV global path.

        Returns normalized ego-frame [x1,y1,...,x8,y8], or None if the active
        planner is not backed by a CSV/global path.
        """
        try:
            planner = self.get_ego_planner()
            if not hasattr(planner, "get_global_waypoints"):
                return None
            global_path = list(planner.get_global_waypoints())
        except Exception:
            return None

        if len(global_path) < 2:
            return None

        xy = np.asarray(
            [(float(p[0]), float(p[1])) for p in global_path],
            dtype=np.float64,
        )

        seg = xy[1:] - xy[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        valid = seg_len > 1e-6

        if not np.any(valid):
            return np.zeros((num * 2,), dtype=np.float32)

        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        ego_tf = self.get_ego_vehicle().get_transform()
        ego_xy = np.array(
            [ego_tf.location.x, ego_tf.location.y],
            dtype=np.float64,
        )

        # Project current ego position onto the full CSV polyline.
        best_dist2 = float("inf")
        best_s = 0.0

        for i in range(len(seg_len)):
            if seg_len[i] <= 1e-6:
                continue

            v = seg[i]
            t = np.dot(ego_xy - xy[i], v) / max(np.dot(v, v), 1e-12)
            t = float(np.clip(t, 0.0, 1.0))

            proj = xy[i] + t * v
            dist2 = float(np.sum((ego_xy - proj) ** 2))

            if dist2 < best_dist2:
                best_dist2 = dist2
                best_s = float(cum[i] + t * seg_len[i])

        ref_step_m = float(getattr(self, "_planner_reference_step_m", 0.0))

        if ref_step_m <= 0.0:
            stride = int(getattr(self, "_planner_waypoint_stride", 5))
            median_ds = float(np.median(seg_len[valid]))
            ref_step_m = max(0.1, median_ds * max(1, stride))

        pts = []

        for k in range(num):
            target_s = min(best_s + (k + 1) * ref_step_m, cum[-1])

            j = int(np.searchsorted(cum, target_s, side="right") - 1)
            j = max(0, min(j, len(seg_len) - 1))

            denom = max(seg_len[j], 1e-6)
            alpha = float(np.clip((target_s - cum[j]) / denom, 0.0, 1.0))

            wx, wy = xy[j] + alpha * seg[j]
            lx, ly = self._world_to_ego_xy(wx, wy)
            pts.extend([lx / scale, ly / scale])

        return np.clip(np.asarray(pts, dtype=np.float32), -1.0, 1.0)

    def _get_global_path_ego(self, env_state=None):
        """Dense local-window global path in normalized ego frame, shape [N, 2].

        Projects ego onto the CSV global path, then samples N points evenly
        over a window [best_s - lookback, best_s + lookahead]. Points outside
        the valid arclength range are set to zero and flagged in the mask.
        The mask is cached so _get_global_path_ego_mask can return it without
        recomputing.
        """
        del env_state
        num = int(getattr(self, "_planner_global_path_num", 50))
        scale = float(getattr(self, "_planner_waypoint_scale", 15.0))
        lookback = float(getattr(self, "_planner_global_path_lookback", 5.0))
        lookahead = float(getattr(self, "_planner_global_path_lookahead", 45.0))

        mask = np.zeros(num, dtype=np.float32)
        out = np.zeros((num, 2), dtype=np.float32)

        try:
            planner = self.get_ego_planner()
            if not hasattr(planner, "get_global_waypoints"):
                self._global_path_ego_mask_cache = mask
                return out
            global_path = list(planner.get_global_waypoints())
        except Exception:
            self._global_path_ego_mask_cache = mask
            return out

        if len(global_path) < 2:
            self._global_path_ego_mask_cache = mask
            return out

        xy = np.asarray(
            [(float(p[0]), float(p[1])) for p in global_path],
            dtype=np.float64,
        )
        seg = xy[1:] - xy[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        valid_seg = seg_len > 1e-6
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = float(cum[-1])

        if total < 1e-6:
            self._global_path_ego_mask_cache = mask
            return out

        # Project ego onto the CSV polyline to find best_s.
        ego_tf = self.get_ego_vehicle().get_transform()
        ego_xy = np.array(
            [ego_tf.location.x, ego_tf.location.y], dtype=np.float64)
        best_s = 0.0
        best_dist2 = float("inf")
        for i in range(len(seg_len)):
            if seg_len[i] <= 1e-6:
                continue
            v = seg[i]
            t = np.dot(ego_xy - xy[i], v) / max(np.dot(v, v), 1e-12)
            t = float(np.clip(t, 0.0, 1.0))
            proj = xy[i] + t * v
            dist2 = float(np.sum((ego_xy - proj) ** 2))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_s = float(cum[i] + t * seg_len[i])

        # Sample num points over the local window.
        window_start = best_s - lookback
        window_end = best_s + lookahead

        for k in range(num):
            if num > 1:
                target_s = window_start + (window_end - window_start) * k / (num - 1)
            else:
                target_s = best_s

            # Out-of-range check
            if target_s < 0.0 or target_s > total:
                out[k] = [0.0, 0.0]
                mask[k] = 0.0
                continue

            mask[k] = 1.0
            j = max(0, min(
                int(np.searchsorted(cum, target_s, side="right")) - 1,
                len(seg_len) - 1,
            ))
            denom = max(float(seg_len[j]), 1e-6)
            alpha = float(np.clip((target_s - float(cum[j])) / denom, 0.0, 1.0))
            wx = float(xy[j, 0]) + alpha * float(seg[j, 0])
            wy = float(xy[j, 1]) + alpha * float(seg[j, 1])
            lx, ly = self._world_to_ego_xy(wx, wy)
            out[k, 0] = np.clip(lx / scale, -1.0, 1.0)
            out[k, 1] = np.clip(ly / scale, -1.0, 1.0)

        self._global_path_ego_mask_cache = mask
        return out

    def _get_global_path_ego_mask(self, env_state=None):
        """Validity mask for global_path_ego points, shape [N].

        Returns the mask cached by _get_global_path_ego, or all-zeros if the
        cache is stale (should not happen — the framework calls handlers in
        registration order).
        """
        del env_state
        cached = getattr(self, "_global_path_ego_mask_cache", None)
        if cached is not None:
            return cached
        num = int(getattr(self, "_planner_global_path_num", 50))
        return np.zeros(num, dtype=np.float32)

    def _valid_expert_waypoints(self, waypoints, scale):
        """Reject stale reset-time trajectories before they enter replay."""
        pts = np.asarray(waypoints, dtype=np.float32).reshape(-1, 2)
        if pts.size == 0 or not np.all(np.isfinite(pts)):
            return False

        clip_frac = float(np.mean(np.abs(pts) >= 0.999))
        max_clip_frac = float(getattr(
            self, "_planner_expert_max_clip_frac", 0.25))
        if clip_frac > max_clip_frac:
            return False

        # Optional guard for deployments where the planner marker is guaranteed
        # to start in front of the ego vehicle. Disabled by default because the
        # current PolyPlanner marker can include near/behind-ego fit points.
        min_first_x_m = float(getattr(
            self, "_planner_expert_min_first_x", -0.25))
        first_x_m = float(pts[0, 0]) * float(scale)
        if first_x_m < min_first_x_m:
            return False

        return True

    def _sample_trajectory_waypoints(self, traj_world, num, scale):
        """Sample num future waypoints by time from a world-frame trajectory.

        Trims trajectory prefix that projects behind the vehicle (ego-frame x < 0)
        before sampling, so all output waypoints are physically ahead of the vehicle.

        Args:
            traj_world: list of (x, y) world coordinates. Consecutive points are
                assumed to be spaced by planner_target.expert_source_dt seconds.
            num: number of waypoints to sample (typically 8).
            scale: normalization scale (meters).

        Returns:
            np.array of shape (num * 2,) with normalized ego-frame [x0, y0, x1, y1, ...].
        """
        n = len(traj_world)
        if n < 2:
            return np.zeros((num * 2,), dtype=np.float32)

        # Trim prefix that has fallen behind the vehicle
        skip = 0
        for i in range(n):
            lx, _ = self._world_to_ego_xy(traj_world[i][0], traj_world[i][1])
            if lx >= 0.0:
                skip = i
                break

        traj_trimmed = traj_world[skip:]
        n_trim = len(traj_trimmed)
        if n_trim < 2:
            return np.zeros((num * 2,), dtype=np.float32)

        source_dt = float(getattr(self, "_planner_expert_source_dt", 0.2))
        target_dt = float(getattr(self, "_planner_expert_target_dt", 0.5))
        source_dt = max(source_dt, 1e-6)
        target_dt = max(target_dt, 1e-6)

        max_time = (n_trim - 1) * source_dt
        pts = []
        for k in range(num):
            sample_time = min((k + 1) * target_dt, max_time)
            pos = sample_time / source_dt
            idx = int(math.floor(pos))
            idx = max(0, min(idx, n_trim - 2))
            t = float(pos - idx)
            t = max(0.0, min(1.0, t))
            x = traj_trimmed[idx][0] + t * (traj_trimmed[idx + 1][0] - traj_trimmed[idx][0])
            y = traj_trimmed[idx][1] + t * (traj_trimmed[idx + 1][1] - traj_trimmed[idx][1])
            lx, ly = self._world_to_ego_xy(x, y)
            pts.extend([lx / scale, ly / scale])
        return np.clip(np.asarray(pts, dtype=np.float32), -1.0, 1.0)

    def _get_ego_x(self, env_state=None):
        del env_state
        return np.array([self.get_ego_vehicle().get_location().x], dtype=np.float32)

    def _get_ego_y(self, env_state=None):
        del env_state
        return np.array([self.get_ego_vehicle().get_location().y], dtype=np.float32)

    def _get_ego_yaw(self, env_state=None):
        del env_state
        return np.array([math.radians(self.get_ego_vehicle().get_transform().rotation.yaw)],
                        dtype=np.float32)

    def _get_ego_speed(self, env_state=None):
        del env_state
        vel = self.get_ego_vehicle().get_velocity()
        return np.array([math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)], dtype=np.float32)

    def _get_ego_yawrate(self, env_state=None):
        del env_state
        ang = self.get_ego_vehicle().get_angular_velocity()
        return np.array([math.radians(ang.z)], dtype=np.float32)

    def _get_neighbor_vehicles_local(self, env_state=None):
        """Return K=8 nearest vehicles in ego frame, padded with zeros.

        Format: [K*11] flattened, each vehicle =
          [valid, x, y, vx, vy, sin_yaw, cos_yaw, L, W, accel, yaw_rate]
        """
        del env_state
        K = int(getattr(self, "_neighbor_k", 8))
        radius = float(getattr(self, "_neighbor_radius", 50.0))
        ego = self.get_ego_vehicle()
        ego_tf = ego.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        ego_id = ego.id

        vehicles = []
        try:
            actors = self._world.carla_world.get_actors().filter("vehicle.*")
        except Exception:
            actors = []

        for actor in actors:
            if actor.id == ego_id:
                continue
            a_tf = actor.get_transform()
            a_loc = a_tf.location
            dx = a_loc.x - ego_loc.x
            dy = a_loc.y - ego_loc.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > radius:
                continue

            # World -> ego frame (Carla: x forward, y right)
            lx = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
            ly = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy

            vel = actor.get_velocity()
            vx_w = vel.x
            vy_w = vel.y
            # Rotate velocity to ego frame
            vx_e = math.cos(ego_yaw) * vx_w + math.sin(ego_yaw) * vy_w
            vy_e = -math.sin(ego_yaw) * vx_w + math.cos(ego_yaw) * vy_w

            a_yaw = math.radians(a_tf.rotation.yaw)
            rel_yaw = a_yaw - ego_yaw  # relative heading

            bb = actor.bounding_box.extent
            L = 2.0 * bb.x  # full length
            W = 2.0 * bb.y  # full width

            # Longitudinal acceleration: project world-frame accel onto vehicle heading.
            accel_vec = actor.get_acceleration()
            acc_long = (accel_vec.x * math.cos(a_yaw)
                        + accel_vec.y * math.sin(a_yaw))

            # Yaw rate (CARLA angular velocity is in deg/s, convert to rad/s).
            ang_vel = actor.get_angular_velocity()
            yaw_rate = math.radians(ang_vel.z)

            vehicles.append((dist, [1.0, lx, ly, vx_e, vy_e,
                                   math.sin(rel_yaw), math.cos(rel_yaw),
                                   L, W, acc_long, yaw_rate]))

        # Sort by distance, take K nearest, pad with zeros
        vehicles.sort(key=lambda x: x[0])
        flat = np.zeros(K * 11, dtype=np.float32)
        for i, (_, vdata) in enumerate(vehicles[:K]):
            flat[i * 11 : (i + 1) * 11] = np.array(vdata, dtype=np.float32)
        return flat

    def _get_neighbor_vehicles_world(self, env_state=None):
        """Return K nearest vehicles in world frame, padded with zeros.

        This is intended for standalone VWM dataset construction. Actor ids make
        it possible to align each current neighbor with its future trajectory
        after the replay has been saved.
        """
        del env_state
        K = int(getattr(self, "_neighbor_k", 8))
        radius = float(getattr(self, "_neighbor_radius", 50.0))
        ego = self.get_ego_vehicle()
        ego_loc = ego.get_location()
        ego_id = ego.id

        vehicles = []
        try:
            actors = self._world.carla_world.get_actors().filter("vehicle.*")
        except Exception:
            actors = []

        for actor in actors:
            if actor.id == ego_id:
                continue
            a_tf = actor.get_transform()
            a_loc = a_tf.location
            dx = a_loc.x - ego_loc.x
            dy = a_loc.y - ego_loc.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > radius:
                continue

            vel = actor.get_velocity()
            yaw = math.radians(a_tf.rotation.yaw)
            bb = actor.bounding_box.extent
            accel_vec = actor.get_acceleration()
            acc_long = accel_vec.x * math.cos(yaw) + accel_vec.y * math.sin(yaw)
            ang_vel = actor.get_angular_velocity()

            vehicles.append(
                (
                    dist,
                    [
                        1.0,
                        float(actor.id),
                        float(a_loc.x),
                        float(a_loc.y),
                        float(vel.x),
                        float(vel.y),
                        math.sin(yaw),
                        math.cos(yaw),
                        2.0 * float(bb.x),
                        2.0 * float(bb.y),
                        float(acc_long),
                        math.radians(float(ang_vel.z)),
                    ],
                )
            )

        vehicles.sort(key=lambda x: x[0])
        flat = np.zeros(K * 12, dtype=np.float32)
        for i, (_, vdata) in enumerate(vehicles[:K]):
            flat[i * 12 : (i + 1) * 12] = np.array(vdata, dtype=np.float32)
        return flat

    def get_state(self):
        planner = self.get_ego_planner()

        if hasattr(planner, "get_global_waypoints"):
            ego_global_path = planner.get_global_waypoints()
        else:
            ego_global_path = list(planner.get_all_waypoints())

        return {
            "ego_waypoints": self.waypoints,
            "ego_global_path": ego_global_path,
            "timesteps": self._time_step,
        }

    def apply_control(self, action) -> None:
        """Apply action to ego vehicle.

        Waypoint mode (>2 dims): runs the PID+PP/LQR controller to convert
        the 16-dim waypoint plan into CARLA VehicleControl.

        Legacy mode (2 dims): direct [acc, steer] → VehicleControl.
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        if len(action) > 2:
            # Waypoint action: convert to VehicleControl via the controller.
            scale = float(getattr(self, "_planner_waypoint_scale", 30.0))
            control = self._get_or_create_poly_controller().run_step(action, scale)
        else:
            # Legacy 2-dim [acc, steer] action.
            control = self.get_vehicle_control(action)
        self.get_ego_vehicle().apply_control(control)

    def on_step(self) -> None:
        self.waypoints, self.planner_stats = self.get_ego_planner().run_step()
        self.num_completed = self.planner_stats["num_completed"]

    def reward(self):
        reward_scales = self._config.reward.scales
        ego = self.get_ego_vehicle()
        ego_location = np.array([*get_vehicle_pos(ego)])
        ego_velocity = np.array([*get_vehicle_velocity(ego)])
        speed_norm = np.linalg.norm(ego_velocity)

        # Reward for reaching waypoints
        r_waypoints = 0.0
        if self.num_completed > 0:
            r_waypoints = reward_scales["waypoint"]

        # Reward for speed
        r_speed = 0.0
        speed_parallel = 0.0
        speed_perpendicular = 0.0
        if len(self.waypoints) > 0:
            # compute the wpt line direction
            next_waypoint = self.waypoints[0]
            next_location = np.array([next_waypoint[0], next_waypoint[1]])
            yaw_radius = next_waypoint[2] * np.pi / 180
            waypoint_direction = np.array([np.cos(yaw_radius), np.sin(yaw_radius)])

            # compute the perpendicular direction
            goal_offset = next_location - ego_location
            perp_direction = goal_offset - np.dot(goal_offset, waypoint_direction) * waypoint_direction
            perp_direction_norm = np.linalg.norm(perp_direction)
            if perp_direction_norm > 0.05:
                perp_direction = perp_direction / perp_direction_norm
            else:
                perp_direction = np.array([0.0, 0.0])

            # compute the speed reward
            desired_speed = self._config.reward.desired_speed
            speed_parallel = np.dot(ego_velocity, waypoint_direction)
            speed_perpendicular = np.abs(np.dot(ego_velocity, perp_direction))
            r_speed = (desired_speed - np.abs(speed_parallel - desired_speed) - 2 * min(speed_perpendicular, 0.5)) * reward_scales["speed"]

        # Reward for collision
        r_collision = 0.0
        if reward_scales["collision"] > 0 and self.is_collision():
            r_collision = -reward_scales["collision"] * np.abs(speed_norm)

        # Reward for going out of lane
        r_out_of_lane = 0.0
        if len(self.waypoints) > 0:
            dist = perp_direction_norm
            if dist > 0.5:
                r_out_of_lane = -reward_scales["out_of_lane"] * (dist - 0.5)

        # Reward for reaching the destination
        r_destination = 0.0
        if self.is_destination_reached():
            r_destination = reward_scales["destination_reached"]

        # Time penalty
        time_penalty = -reward_scales["time"]

        # Total reward
        total_reward = r_waypoints + r_speed + r_collision + r_out_of_lane + r_destination + time_penalty
        # total_reward = r_destination
        ttc = TTCCalculator.get_ttc(ego, self._world.carla_world, self._world.carla_map)

        info = {
            **self.planner_stats,
            "ego_x": ego_location[0],
            "ego_y": ego_location[1],
            "speed_parallel": speed_parallel,
            "speed_perpendicular": speed_perpendicular,
            "speed_norm": speed_norm,
            "wpt_dis": self.get_wpt_dist(ego_location),
            "r_waypoints": r_waypoints,
            "r_speed": r_speed,
            "r_collision": r_collision,
            "r_out_of_lane": r_out_of_lane,
            "ttc": ttc,
        }

        return total_reward, info

    def is_destination_reached(self):
        return len(self.waypoints) <= 50

    def get_terminal_conditions(self):
        terminal_config = self._config.terminal
        ego_location = get_vehicle_pos(self.get_ego_vehicle())
        
        warmup_steps = int(getattr(terminal_config, "out_lane_grace_steps", 15))
        out_of_lane = False
        if self._time_step > warmup_steps:
            out_of_lane = self.get_wpt_dist(ego_location) > terminal_config.out_lane_thres
        conds = {
            "is_collision": self.is_collision(),
            "time_exceeded": self._time_step > terminal_config.time_limit,
            "out_of_lane": out_of_lane,
            "destination_reached": self.is_destination_reached(),
        }
        return conds

    # def get_wpt_dist(self, ego_location):
    #     if len(self.waypoints) == 0:
    #         return 0
    #     else:
    #         return get_location_distance(ego_location, self.waypoints[0])
    import numpy as np

    def point_to_segment_distance(self, p, a, b):
        ab = b - a
        denom = np.dot(ab, ab)
        if denom < 1e-9:
            return np.linalg.norm(p - a)
        t = np.dot(p - a, ab) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    def get_wpt_dist(self, ego_location):
        if len(self.waypoints) == 0:
            return 0.0

        p = np.array([ego_location[0], ego_location[1]], dtype=float)
        pts = [np.array([w[0], w[1]], dtype=float) for w in self.waypoints[:15]]

        best = np.linalg.norm(p - pts[0])
        for i in range(len(pts) - 1):
            d = self.point_to_segment_distance(p, pts[i], pts[i + 1])
            if d < best:
                best = d
        return float(best)
