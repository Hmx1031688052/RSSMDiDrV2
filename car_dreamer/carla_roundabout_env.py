from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import get_vehicle_pos
import carla
import math
import numpy as np
from .toolkit import get_vehicle_velocity

class CarlaRoundaboutEnv(CarlaWptFixedEnv):
    """
    Vehicle passes the roundabout and avoid collision.

    **Provided Tasks**: ``carla_roundabout``
    """

    def on_reset(self) -> None:
        super().on_reset()
        self._last_action = np.zeros(2, dtype=np.float32)
        self._current_action = np.zeros(2, dtype=np.float32)
        self._prev_waypoint_count = len(self.waypoints)

    def apply_control(self, action) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size == 1:
            action = np.array([action[0], 0.0], dtype=np.float32)
        self._last_action = self._current_action.copy()
        self._current_action = action[:2].copy()
        super().apply_control(action)

    def on_step(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(vehicle)
            if (y < 0.0 and x < -39.8) or y < -47.2 or y > 46.0 or x > 44.8:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
        self._update_spectator_follow()
        # self._debug_draw_waypoints_and_lateral(life_time=0.15)
        super().on_step()
    
    def _update_spectator_follow(self):
        spectator = self._world._world.get_spectator()  # carla.World
        ego_tf = self.get_ego_vehicle().get_transform()
        loc = ego_tf.location
        rot = ego_tf.rotation

        yaw = math.radians(rot.yaw)

        back_dist = 8.0
        height = 3.0

        cam_loc = carla.Location(
            x=loc.x - back_dist * math.cos(yaw),
            y=loc.y - back_dist * math.sin(yaw),
            z=loc.z + height,
        )

        cam_rot = carla.Rotation(
            pitch=-15.0,
            yaw=rot.yaw,
            roll=0.0,
        )

        spectator.set_transform(carla.Transform(cam_loc, cam_rot))

    def reward(self):
        total_reward, info = super().reward()
        reward_cfg = self._config.reward
        scales = reward_cfg.scales
        ego = self.get_ego_vehicle()
        ego_location = np.array([*get_vehicle_pos(ego)], dtype=np.float32)
        ego_yaw_deg = float(ego.get_transform().rotation.yaw)

        prev_waypoint_count = getattr(self, "_prev_waypoint_count", len(self.waypoints))
        current_waypoint_count = len(self.waypoints)
        waypoint_delta = max(prev_waypoint_count - current_waypoint_count, 0)
        route_progress = float(self.planner_stats.get("travel_distance", 0.0))
        if waypoint_delta > 0:
            route_progress += float(waypoint_delta) * float(reward_cfg.get("progress_waypoint_bonus", 0.0))
        route_progress = float(np.clip(route_progress, 0.0, float(reward_cfg.get("progress_clip", 2.0))))
        r_progress = float(scales.get("progress", 0.0)) * route_progress
        self._prev_waypoint_count = current_waypoint_count

        lateral_error = float(
            np.clip(
                self.get_wpt_dist(ego_location),
                0.0,
                float(reward_cfg.get("lateral_clip", 3.0)),
            )
        )
        r_lateral = -float(scales.get("lateral", 0.0)) * lateral_error

        if len(self.waypoints) > 0:
            ref_yaw_deg = float(self.waypoints[0][2])
        else:
            ref_yaw_deg = ego_yaw_deg
        heading_delta_deg = ((ego_yaw_deg - ref_yaw_deg + 180.0) % 360.0) - 180.0
        heading_error = abs(heading_delta_deg) / 180.0
        r_heading = -float(scales.get("heading", 0.0)) * heading_error

        r_collision = -float(scales.get("collision", 0.0)) if self.is_collision() else 0.0
        ttc, ttc_risk = self._quadratic_neighbor_ttc_risk(ego)
        r_ttc = -float(scales.get("ttc", 0.0)) * ttc_risk

        acc_delta = float(abs(self._current_action[0] - self._last_action[0]))
        steer_delta = float(abs(self._current_action[1] - self._last_action[1]))
        r_acc_jerk = -float(scales.get("acc_jerk", 0.0)) * acc_delta
        r_steer_jerk = -float(scales.get("steer_jerk", 0.0)) * steer_delta

        # Base CarlaWptEnv.reward() already includes waypoint/speed efficiency,
        # collision, out-of-lane, destination, and time terms. Roundabout adds
        # only dense traffic-interaction safety here to avoid double counting.
        total_reward += r_ttc
        info.update(
            {
                "route_progress": route_progress,
                "lateral_error": lateral_error,
                "heading_error": heading_error,
                "ttc": ttc,
                "ttc_risk": ttc_risk,
                "r_progress": r_progress,
                "r_lateral": r_lateral,
                "r_heading": r_heading,
                "r_collision": r_collision,
                "r_ttc": r_ttc,
                "acc_delta": acc_delta,
                "steer_delta": steer_delta,
                "r_acc_jerk": r_acc_jerk,
                "r_steer_jerk": r_steer_jerk,
            }
        )
        # print('info: ', info)
        return total_reward, info

    def _quadratic_neighbor_ttc_risk(self, ego):
        reward_cfg = self._config.reward
        threshold = float(reward_cfg.get("ttc_threshold", 3.0))
        safe_radius = float(reward_cfg.get("ttc_safe_radius", 5.0))
        max_distance = float(reward_cfg.get("ttc_max_distance", 50.0))
        threshold = max(threshold, 1e-6)
        safe_radius = max(safe_radius, 1e-6)

        try:
            neighbors = self._get_neighbor_vehicles_local().reshape(-1, 11)
        except Exception:
            return 0.0, 0.0
        valid = neighbors[:, 0] > 0.5
        if not np.any(valid):
            return 0.0, 0.0

        ego_tf = ego.get_transform()
        ego_yaw = math.radians(float(ego_tf.rotation.yaw))
        ego_vel = ego.get_velocity()
        ego_vx = math.cos(ego_yaw) * float(ego_vel.x) + math.sin(ego_yaw) * float(ego_vel.y)
        ego_vy = -math.sin(ego_yaw) * float(ego_vel.x) + math.cos(ego_yaw) * float(ego_vel.y)

        min_ttc = float("inf")
        max_risk = 0.0
        for veh in neighbors[valid]:
            px = float(veh[1])
            py = float(veh[2])
            current_dist = math.sqrt(px * px + py * py)
            if current_dist > max_distance:
                continue

            vehicle_radius = 0.5 * math.sqrt(float(veh[7]) ** 2 + float(veh[8]) ** 2)
            radius = max(safe_radius, vehicle_radius + 1.5)
            if current_dist <= radius:
                min_ttc = 0.0
                max_risk = 1.0
                continue

            vx = float(veh[3]) - ego_vx
            vy = float(veh[4]) - ego_vy
            a = vx * vx + vy * vy
            b = 2.0 * (px * vx + py * vy)
            c = px * px + py * py - radius * radius
            if a <= 1e-8:
                continue
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                continue
            sqrt_disc = math.sqrt(disc)
            roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
            positive_roots = [root for root in roots if root >= 0.0]
            if not positive_roots:
                continue
            ttc = min(positive_roots)
            if ttc < min_ttc:
                min_ttc = ttc
            if ttc < threshold:
                risk = (threshold - ttc) / threshold
                max_risk = max(max_risk, float(np.clip(risk * risk, 0.0, 1.0)))

        if not math.isfinite(min_ttc):
            return 0.0, max_risk
        return float(min_ttc), float(max_risk)

    def _debug_draw_waypoints_and_lateral(self, life_time=0.1):
        if len(self.waypoints) == 0:
            return

        world = self._world._world
        debug = world.debug
        ego = self.get_ego_vehicle()
        ego_loc = ego.get_transform().location

        z = ego_loc.z + 0.5

        prev_loc = None
        for i, wpt in enumerate(self.waypoints[:20]):
            loc = carla.Location(x=float(wpt[0]), y=float(wpt[1]), z=z)

            color = carla.Color(0, 255, 0) if i == 0 else carla.Color(0, 180, 255)
            debug.draw_point(loc, size=0.08, color=color, life_time=life_time)

            if i < 10:
                debug.draw_string(
                    loc + carla.Location(z=0.15),
                    str(i),
                    draw_shadow=False,
                    color=color,
                    life_time=life_time,
                )

            if prev_loc is not None:
                debug.draw_line(
                    prev_loc,
                    loc,
                    thickness=0.03,
                    color=carla.Color(0, 180, 255),
                    life_time=life_time,
                )
            prev_loc = loc
