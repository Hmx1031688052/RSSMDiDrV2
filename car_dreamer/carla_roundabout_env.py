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

        acc_delta = float(abs(self._current_action[0] - self._last_action[0]))
        steer_delta = float(abs(self._current_action[1] - self._last_action[1]))
        r_acc_jerk = -float(scales.get("acc_jerk", 0.0)) * acc_delta
        r_steer_jerk = -float(scales.get("steer_jerk", 0.0)) * steer_delta

        # total_reward += (
        #     # r_progress
        #     r_lateral
        #     + r_heading
        #     + r_collision
        #     + r_acc_jerk
        #     + r_steer_jerk
        # )
        info.update(
            {
                "route_progress": route_progress,
                "lateral_error": lateral_error,
                "heading_error": heading_error,
                "r_progress": r_progress,
                "r_lateral": r_lateral,
                "r_heading": r_heading,
                "r_collision": r_collision,
                "acc_delta": acc_delta,
                "steer_delta": steer_delta,
                "r_acc_jerk": r_acc_jerk,
                "r_steer_jerk": r_steer_jerk,
            }
        )
        # print('info: ', info)
        return total_reward, info
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
