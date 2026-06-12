import math

import carla
import numpy as np

from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedPathPlanner, get_vehicle_pos, get_vehicle_velocity


class CarlaStaticObstacleEnv(CarlaWptEnv):
    """
    Straight-road benchmark with one static obstacle ahead of the ego vehicle.

    Expert modes:
      - left:  bypass through the left adjacent lane
      - right: bypass through the right adjacent lane
      - stop:  brake and stop safely before the obstacle

    This env keeps the action interface identical to the rest of CarDreamer.
    The intended usage is to let Dreamer run the rollout/replay shell, while a
    collector-side wrapper replaces Dreamer actions with rule expert actions.
    """

    MODES = ("left", "right", "stop")
    MODE_TO_ID = {name: idx for idx, name in enumerate(MODES)}

    def __init__(self, config):
        super().__init__(config)
        self.ego = None
        self.obstacle = None
        self.ego_planner = None
        self.waypoints = []
        self.planner_stats = {
            "num_completed": 0,
            "num_obsolete": 0,
            "travel_distance": 0.0,
        }
        self.num_completed = 0

        self.expert_mode = "left"
        self._mode_queue = []

        self._passed_obstacle = False
        self._stop_success = False
        self._stop_hold_steps = 0
        self._awarded_pass_bonus = False
        self._awarded_stop_bonus = False
        self._obstacle_y = None

    def on_reset(self) -> None:
        self._passed_obstacle = False
        self._stop_success = False
        self._stop_hold_steps = 0
        self._awarded_pass_bonus = False
        self._awarded_stop_bonus = False

        self._spawn_ego()
        self.expert_mode = self._sample_mode()
        self._spawn_obstacle()
        self._rebuild_planner(self.expert_mode)
        self._update_spectator()

    def apply_control(self, action) -> None:
        ego_control = self.get_vehicle_control(action)
        self.ego.apply_control(ego_control)
        self._lock_obstacle()

    def on_step(self) -> None:
        self._lock_obstacle()
        self._update_mode_state()
        # self._update_spectator()   # 调试时可以先关掉
        # self._debug_draw_future_waypoints(life_time=0.25)
        # ego_loc = self.ego.get_location()
        # ego_vel = self.ego.get_velocity()
        # speed = (ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2) ** 0.5

        # if self.obstacle is not None:
        #     obs_loc = self.obstacle.get_location()
        #     dx = ego_loc.x - obs_loc.x
        #     dy = ego_loc.y - obs_loc.y
        #     dist = (dx * dx + dy * dy) ** 0.5
        #     print(
        #         f"[DBG] mode={self.expert_mode} "
        #         f"ego=({ego_loc.x:.2f},{ego_loc.y:.2f}) "
        #         f"obs=({obs_loc.x:.2f},{obs_loc.y:.2f}) "
        #         f"speed={speed:.2f} dist={dist:.2f}",
        #         flush=True,
        #     )
        # else:
        #     print(
        #         f"[DBG] mode={self.expert_mode} "
        #         f"ego=({ego_loc.x:.2f},{ego_loc.y:.2f}) "
        #         f"speed={speed:.2f} obstacle=None",
        #         flush=True,
        #     )

        super().on_step()

    def reward(self):
        total_reward, info = super().reward()
        reward_scales = self._config.reward.scales

        r_pass_obstacle = 0.0
        if self.expert_mode in ("left", "right") and self._passed_obstacle and not self._awarded_pass_bonus:
            r_pass_obstacle = reward_scales.get("pass_obstacle", 0.0)
            self._awarded_pass_bonus = True

        r_stop_success = 0.0
        if self.expert_mode == "stop" and self._stop_success and not self._awarded_stop_bonus:
            r_stop_success = reward_scales.get("stop_success", 0.0)
            self._awarded_stop_bonus = True

        total_reward += r_pass_obstacle + r_stop_success
        info.update(
            {
                "expert_mode_id": float(self.MODE_TO_ID[self.expert_mode]),
                "expert_mode_left": float(self.expert_mode == "left"),
                "expert_mode_right": float(self.expert_mode == "right"),
                "expert_mode_stop": float(self.expert_mode == "stop"),
                "dist_to_obstacle": self._distance_to_obstacle(),
                "passed_obstacle": float(self._passed_obstacle),
                "stop_hold_steps": float(self._stop_hold_steps),
                "stop_success": float(self._stop_success),
                "r_pass_obstacle": r_pass_obstacle,
                "r_stop_success": r_stop_success,
            }
        )
        return total_reward, info

    def is_destination_reached(self):
        if self.expert_mode == "stop":
            return self._stop_success
        return super().is_destination_reached()

    def get_terminal_conditions(self):
        conds = super().get_terminal_conditions()
        ego_x = self.ego.get_location().x
        terminal_cfg = self._config.terminal

        conds["x_out_of_bound"] = bool(
            ego_x < terminal_cfg.left_lane_boundry or ego_x > terminal_cfg.right_lane_boundry
        )
        conds["stop_success"] = bool(self._stop_success)
        # For bypass experts, passing the obstacle should award success bonus but
        # should not immediately terminate the episode, otherwise the controller
        # has no chance to stabilize and merge back to the center lane.
        conds["passed_obstacle"] = False
        conds["out_of_lane"] = bool(conds["out_of_lane"] or conds["x_out_of_bound"])
        return conds

    def get_expert_action(self, raw_action_env=None):
        """
        Return expert action in ENV ACTION SPACE, not normalized Dreamer space.
        Current code assumes continuous action = [acc, steer].
        """
        del raw_action_env

        steer = self._compute_pure_pursuit_steer()
        target_speed = self._get_target_speed()
        speed = np.linalg.norm(np.array(get_vehicle_velocity(self.ego), np.float32))

        expert_cfg = self._config.expert
        acc = expert_cfg.speed_kp * (target_speed - speed)

        acc = np.clip(
            acc,
            self._config.action.continuous_acc[0],
            self._config.action.continuous_acc[1],
        )
        steer = np.clip(
            steer,
            self._config.action.continuous_steer[0],
            self._config.action.continuous_steer[1],
        )

        return np.array([acc, steer], dtype=np.float32)

    def _spawn_ego(self):
        ego_src = self._config.lane_start_point
        ego_transform = carla.Transform(
            carla.Location(x=ego_src[0], y=ego_src[1], z=ego_src[2]),
            carla.Rotation(yaw=ego_src[3]),
        )
        ego_bp = self._make_vehicle_blueprint(role_name="hero", color="49,8,8")
        self.ego = self._world.spawn_actor(transform=ego_transform, blueprint=ego_bp)

    def _spawn_obstacle(self):
        lane_x = float(getattr(self._config, "obstacle_lane_x", self._config.lane_center_x))
        y = float(self._config.obstacle_spawn_y)
        z = float(self._config.lane_start_point[2])

        obstacle_transform = carla.Transform(
            carla.Location(x=lane_x, y=y, z=z),
            carla.Rotation(yaw=float(self._config.lane_start_point[3])),
        )
        obstacle_bp = self._make_vehicle_blueprint(role_name="obstacle", color="80,80,80")
        self.obstacle = self._world.spawn_actor(transform=obstacle_transform, blueprint=obstacle_bp)
        self._obstacle_y = y
        self._lock_obstacle()

    def _make_vehicle_blueprint(self, role_name="hero", color=None):
        bp = self._world.get_blueprint("vehicle.audi*", {"number_of_wheels": "4"})
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", role_name)
        if color is not None and bp.has_attribute("color"):
            bp.set_attribute("color", color)
        return bp

    def _sample_mode(self):
        expert_cfg = self._config.expert

        if getattr(expert_cfg, "fixed_mode", None) in self.MODES:
            return expert_cfg.fixed_mode

        sampler = getattr(expert_cfg, "mode_sampler", "cycle")
        modes = list(getattr(expert_cfg, "modes", list(self.MODES)))

        if sampler == "cycle":
            if not self._mode_queue:
                self._mode_queue = modes.copy()
                np.random.shuffle(self._mode_queue)
            return self._mode_queue.pop()

        probs = getattr(expert_cfg, "mode_probs", None)
        if probs is not None:
            return np.random.choice(modes, p=np.asarray(probs, dtype=np.float64))
        return np.random.choice(modes)

    def _rebuild_planner(self, mode):
        route = self._build_mode_path(mode)
        use_road_waypoints = [False] * (len(route) - 1)

        self.ego_planner = FixedPathPlanner(
            vehicle=self.ego,
            vehicle_path=route,
            use_road_waypoints=use_road_waypoints,
        )
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]
    def _debug_draw_future_waypoints(self, life_time=0.25):
        if not self.waypoints:
            return

        world = self._world._world
        route_color = carla.Color(0, 255, 0) if self.expert_mode == "left" else carla.Color(0, 0, 255)

        max_draw = min(len(self.waypoints), 30)
        for i in range(max_draw):
            x, y, z = self.waypoints[i]
            loc = carla.Location(float(x), float(y), float(z) + 0.3)
            world.debug.draw_point(loc, size=0.08, color=route_color, life_time=life_time)
            world.debug.draw_string(
                loc + carla.Location(z=0.15),
                str(i),
                draw_shadow=False,
                color=route_color,
                life_time=life_time,
            )
            if i > 0:
                px, py, pz = self.waypoints[i - 1]
                prev = carla.Location(float(px), float(py), float(pz) + 0.3)
                world.debug.draw_line(prev, loc, thickness=0.03, color=route_color, life_time=life_time)

        target_index = min(int(self._config.expert.lookahead_index), len(self.waypoints) - 1)
        tx, ty, tz = self.waypoints[target_index]
        target_loc = carla.Location(float(tx), float(ty), float(tz) + 0.7)
        world.debug.draw_point(target_loc, size=0.14, color=carla.Color(255, 0, 0), life_time=life_time)
        world.debug.draw_string(
            target_loc + carla.Location(z=0.15),
            f"T{target_index}",
            draw_shadow=False,
            color=carla.Color(255, 0, 0),
            life_time=life_time,
        )
        
    def _sample_cubic_bezier(self, p0, p1, p2, p3, num_samples):
        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)
        p2 = np.asarray(p2, dtype=np.float32)
        p3 = np.asarray(p3, dtype=np.float32)

        points = []
        for i in range(num_samples + 1):
            t = i / float(num_samples)
            omt = 1.0 - t
            pt = (
                (omt ** 3) * p0
                + 3.0 * (omt ** 2) * t * p1
                + 3.0 * omt * (t ** 2) * p2
                + (t ** 3) * p3
            )
            points.append([float(pt[0]), float(pt[1]), float(pt[2])])
        return points
    
    def _build_dense_bypass_path(self, side_x):
        route_cfg = self._config.route

        ego_loc = self.ego.get_location()
        z = float(ego_loc.z)
        p0 = np.array([float(ego_loc.x), float(ego_loc.y), z], dtype=np.float32)

        y_obs = float(self._obstacle_y)
        forward_sign = -1.0 if y_obs < float(ego_loc.y) else 1.0

        front_offset = float(getattr(route_cfg, "bezier_front_offset", 10.0))
        return_offset = float(getattr(route_cfg, "return_front_offset", 8.0))
        straight_after_return = float(getattr(route_cfg, "straight_after_return", 12.0))

        num_samples_1 = int(getattr(route_cfg, "bezier_samples_seg1", 15))
        num_samples_2 = int(getattr(route_cfg, "bezier_samples_seg2", 15))
        num_samples_3 = int(getattr(route_cfg, "bezier_samples_seg3", 15))
        num_samples_4 = int(getattr(route_cfg, "bezier_samples_seg4", 10))

        center_x = float(self._config.lane_center_x)

        pm = np.array([float(side_x), y_obs, z], dtype=np.float32)
        p2 = np.array([float(side_x), y_obs + forward_sign * front_offset, z], dtype=np.float32)
        p3 = np.array([center_x, p2[1] + forward_sign * return_offset, z], dtype=np.float32)
        p4 = np.array([center_x, p3[1] + forward_sign * straight_after_return, z], dtype=np.float32)

        lane_change_dx = abs(float(side_x) - float(ego_loc.x))
        ctrl_long_1 = max(3.0, 1.2 * lane_change_dx)
        ctrl_long_2 = max(4.0, 1.0 * lane_change_dx)
        ctrl_long_3 = max(4.0, 1.0 * abs(center_x - float(side_x)))

        p1 = np.array([p0[0], p0[1] + forward_sign * ctrl_long_1, z], dtype=np.float32)
        p2_ctrl = np.array([pm[0], pm[1] - forward_sign * (0.6 * ctrl_long_1), z], dtype=np.float32)
        seg1 = self._sample_cubic_bezier(p0, p1, p2_ctrl, pm, num_samples_1)

        q1 = np.array([pm[0], pm[1] + forward_sign * (0.6 * ctrl_long_2), z], dtype=np.float32)
        q2 = np.array([p2[0], p2[1] - forward_sign * ctrl_long_2, z], dtype=np.float32)
        seg2 = self._sample_cubic_bezier(pm, q1, q2, p2, num_samples_2)

        r1 = np.array([p2[0], p2[1] + forward_sign * (0.6 * ctrl_long_3), z], dtype=np.float32)
        r2 = np.array([p3[0], p3[1] - forward_sign * ctrl_long_3, z], dtype=np.float32)
        seg3 = self._sample_cubic_bezier(p2, r1, r2, p3, num_samples_3)

        seg4 = []
        for i in range(num_samples_4 + 1):
            t = i / float(num_samples_4)
            pt = p3 * (1.0 - t) + p4 * t
            seg4.append([float(pt[0]), float(pt[1]), float(pt[2])])

        dense_route = seg1 + seg2[1:] + seg3[1:] + seg4[1:]
        return dense_route


    def _build_mode_path(self, mode):
        cfg = self._config

        left_x = float(cfg.left_lane_x)
        right_x = float(cfg.right_lane_x)
        center_x = float(cfg.lane_center_x)
        z = float(cfg.lane_start_point[2])
        y_start = float(cfg.lane_start_point[1])
        y_stop = float(self._obstacle_y) + float(self._config.route.stop_buffer)

        if mode == "left":
            return self._build_dense_bypass_path(left_x)

        if mode == "right":
            return self._build_dense_bypass_path(right_x)

        if mode == "stop":
            return [
                [center_x, y_start, z],
                [center_x, y_stop, z],
            ]

        raise ValueError(f"Unsupported mode: {mode}")

    def _lock_obstacle(self):
        if self.obstacle is None:
            return
        self.obstacle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.obstacle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        self.obstacle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        )

    def _update_mode_state(self):
        _, ego_y = get_vehicle_pos(self.ego)
        speed = np.linalg.norm(np.array(get_vehicle_velocity(self.ego), np.float32))
        route_cfg = self._config.route
        expert_cfg = self._config.expert
        forward_sign = self._get_forward_sign(float(self._config.lane_start_point[1]), float(self._obstacle_y))

        longitudinal_progress = forward_sign * (ego_y - self._obstacle_y)
        if longitudinal_progress >= float(route_cfg.pass_threshold):
            self._passed_obstacle = True

        if self.expert_mode == "stop":
            close_enough = self._distance_to_obstacle() <= float(expert_cfg.stop_success_dist)
            before_obstacle = forward_sign * (self._obstacle_y - ego_y) >= 0.0
            slow_enough = speed <= float(expert_cfg.stop_success_speed)

            if close_enough and before_obstacle and slow_enough:
                self._stop_hold_steps += 1
            else:
                self._stop_hold_steps = 0

            self._stop_success = self._stop_hold_steps >= int(expert_cfg.stop_hold_steps)

    def _distance_to_obstacle(self):
        x, ego_y = get_vehicle_pos(self.ego)
        # print('xxxyyyyyy',x,ego_y)
        forward_sign = self._get_forward_sign(float(self._config.lane_start_point[1]), float(self._obstacle_y))
        return float(max(0.0, forward_sign * (self._obstacle_y - ego_y)))


    def _get_forward_sign(self, start_y, target_y):
        return -1.0 if float(target_y) < float(start_y) else 1.0

    def _get_target_speed(self):
        expert_cfg = self._config.expert
        dist = self._distance_to_obstacle()

        if self.expert_mode == "stop":
            if self._stop_success:
                return 0.0
            if dist <= float(expert_cfg.stop_trigger_dist):
                return 0.0
            return float(expert_cfg.cruise_speed)

        if not self._passed_obstacle:
            if dist <= float(expert_cfg.slowdown_dist):
                return float(expert_cfg.bypass_speed)
            return float(expert_cfg.cruise_speed)

        return float(expert_cfg.post_bypass_speed)

    def _compute_pure_pursuit_steer(self):
        if len(self.waypoints) == 0:
            return 0.0

        expert_cfg = self._config.expert
        ego_transform = self.ego.get_transform()
        ego_loc = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)

        front_indices = []
        for i, (wx, wy, _) in enumerate(self.waypoints):
            dx = wx - ego_loc.x
            dy = wy - ego_loc.y
            local_x = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
            if local_x > 0.5:
                front_indices.append(i)

        if not front_indices:
            return 0.0

        base_idx = front_indices[0]
        target_index = min(base_idx + int(expert_cfg.lookahead_index), len(self.waypoints) - 1)
        target_x, target_y, _ = self.waypoints[target_index]

        dx = target_x - ego_loc.x
        dy = target_y - ego_loc.y

        local_x = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
        local_y = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy

        lookahead = max(1e-3, math.sqrt(local_x**2 + local_y**2))
        curvature = 2.0 * local_y / (lookahead**2)
        steer = math.atan(float(expert_cfg.wheel_base) * curvature)
        steer *= float(expert_cfg.steer_gain)
        # print(
        #         f"[PP] mode={self.expert_mode} ego=({ego_loc.x:.2f},{ego_loc.y:.2f}) "
        #         f"yaw={ego_transform.rotation.yaw:.2f} "
        #         f"target=({target_x:.2f},{target_y:.2f}) "
        #         f"local_x={local_x:.2f} local_y={local_y:.2f} steer={steer:.3f}",
        #         flush=True,
        #     )
        return -steer


    # def _update_spectator(self):
    #     spectator = self._world._world.get_spectator()
    #     ego_tf = self.ego.get_transform()
    #     follow = carla.Transform(
    #         carla.Location(
    #             x=ego_tf.location.x,
    #             y=ego_tf.location.y - 10.0,
    #             z=22.0,
    #         ),
    #         carla.Rotation(
    #             pitch=-65.0,
    #             yaw=ego_tf.rotation.yaw,
    #             roll=0.0,
    #         ),
    #     )
    #     spectator.set_transform(follow)
    def _update_spectator(self):
        spectator = self._world._world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=9.0, y=95.0, z=50.0),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            )
        )
