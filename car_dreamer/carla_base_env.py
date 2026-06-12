from abc import abstractmethod
from typing import Dict, Tuple

import carla
import gym
import numpy as np
from gym import spaces

from .toolkit import EnvMonitorOpenCV, Observer, WorldManager

import json
import socket
import threading
import time


class CarlaBaseEnv(gym.Env):
    def __init__(self, config):
        self._config = config

        self._monitor = EnvMonitorOpenCV(self._config)
        self._world = WorldManager(self._config)
        self._world.on_reset(self.on_reset)
        self._world.on_step(self.on_step)
        self._observer = Observer(self._world, self._config.observation)
        self.register_extra_observations()

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # ===== ROS2 expert UDP receiver =====
        self._expert_udp_host = getattr(self._config, "expert_udp_host", "127.0.0.1")
        self._expert_udp_port = int(getattr(self._config, "expert_udp_port", 5005))
        self._expert_udp_enabled = bool(getattr(self._config, "expert_udp_enabled", True))

        self._latest_ros2_ctrl = None
        self._expert_lock = threading.Lock()
        self._expert_stop_event = threading.Event()
        self._expert_thread = None
        self._expert_sock = None

        if self._expert_udp_enabled:
            self._start_expert_udp_receiver()

    @abstractmethod
    def on_reset(self) -> None:
        """
        Override this method to perform additional reset operations.
        Specifically, you can spawn actors and plan routes here.
        """
        pass

    @abstractmethod
    def apply_control(self, action) -> None:
        """
        Override this method to apply control to actors.
        This method will be called before the simulator ticks.
        """
        pass

    @abstractmethod
    def on_step(self) -> None:
        """
        Override this method to perform additional operations at each step.
        Specifically, you can update the planner and the route here.
        This method will be called after the simulator ticks.
        """
        pass

    @abstractmethod
    def reward(self) -> Tuple[float, Dict]:
        """
        Override this method to define the reward function.
        """
        pass

    @abstractmethod
    def get_terminal_conditions(self) -> Dict[str, bool]:
        """
        Override this method to define the terminal condition.
        If one of the keys in the returned dictionary gives True, the episode will be terminated.
        """
        pass

    def get_ego_vehicle(self) -> carla.Actor:
        """
        Override this method to return the ego vehicle.
        The default behavior is to return self.ego
        """
        return self.ego

    def get_state(self) -> Dict:
        """Return the environment state. Implement this method to define the env state."""
        return self._state

    def register_extra_observations(self):
        """Hook for subclasses to register additional replay-only observations.

        These observations can be used as policy targets or scorer inputs while
        being excluded from the world-model encoder/decoder via
        config.policy_target_keys.
        """
        pass

    def _get_action_space(self):
        # Waypoint-action mode: action IS the 16-dim waypoint plan in [-1, 1].
        pt_cfg = getattr(self._config, "planner_target", None)
        use_wpt = (
            pt_cfg is not None
            and bool(getattr(pt_cfg, "enable", False))
            and bool(getattr(pt_cfg, "use_waypoint_action", False))
        )
        if use_wpt:
            num_wp = int(getattr(pt_cfg, "num_waypoints", 8))
            wdim = int(getattr(pt_cfg, "waypoint_dim", 2))
            dim = num_wp * wdim
            return spaces.Box(
                low=-1.0, high=1.0, shape=(dim,), dtype=np.float32,
            )

        action_config = self._config.action
        if action_config.discrete:
            self.n_steer = len(action_config.discrete_steer)
            self.n_acc = len(action_config.discrete_acc)
            return spaces.Discrete(self.n_steer * self.n_acc)
        else:
            return spaces.Box(
                low=np.array([action_config.continuous_acc[0], action_config.continuous_steer[0]]),
                high=np.array([action_config.continuous_acc[1], action_config.continuous_steer[1]]),
                dtype=np.float32,
            )

    def _get_observation_space(self):
        return self._observer.get_observation_space()

    def reset(self):
        print("[CARLA] Reset environment")

        self._clear_ros2_ctrl_cache()

        self._observer.destroy()
        self._world.reset()
        self._observer.reset(self.get_ego_vehicle())
        for _ in range(2):
            self._world._world.tick()
        self._time_step = 0

        print("[CARLA] Environment reset")
        self.obs, _ = self._observer.get_observation(self.get_state())
        return self.obs

    def get_vehicle_control(self, action):
        """
        Convert actions in the action space to vehicle control in CARLA.

        Handles both legacy 2-dim [acc, steer] and waypoint-mode where
        CarlaWptEnv.apply_control routes to the controller instead.
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        action_config = self._config.action
        if action_config.discrete:
            acc = action_config.discrete_acc[action // self.n_steer]
            steer = action_config.discrete_steer[action % self.n_steer]
        else:
            acc = float(action[0])
            steer = float(action[1])

        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)

        return carla.VehicleControl(
            throttle=float(throttle),
            steer=float(-steer),
            brake=float(brake),
        )

    def _is_terminal(self):
        terminal_conds = self.get_terminal_conditions()
        terminal = False
        for k, v in terminal_conds.items():
            if v:
                print(f"[CARLA] Terminal condition triggered: {k}")
                terminal = True
            terminal_conds[k] = np.array([v], dtype=np.bool_)
        if terminal:
            terminal_conds["episode_timesteps"] = self._time_step
        terminal_conds["terminal"] = terminal
        return terminal, terminal_conds

    def step(self, action):
        self.apply_control(action)
        self._world.step()
        self._time_step += 1

        env_state = self.get_state()
        is_terminal, terminal_conds = self._is_terminal()
        self.obs, obs_info = self._observer.get_observation(env_state)
        reward, reward_info = self.reward()

        info = {
            **env_state,
            **terminal_conds,
            **obs_info,
            **reward_info,
            "action": action,
        }
        if self._config.eval:
            info = {f"eval_{k}": v for k, v in info.items()}
            self.obs = {**self.obs, **info}
        if self._config.display.enable:
            self._render(self.obs, info)
        return (self.obs, reward, is_terminal, info)

    def is_collision(self):
        """
        Check if the ego vehicle is in collision.
        You must include 'collsion' in observation.names to use this method.
        """
        return self.obs["collision"][0] > 0
        # return self.obs["collision"][0] < 0


    def _render(self, obs, info):
        self._monitor.render(obs, info)

    # =========================================================
    # ROS2 expert UDP interface
    # =========================================================
    def _start_expert_udp_receiver(self):
        if self._expert_thread is not None:
            return

        self._expert_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._expert_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._expert_sock.bind((self._expert_udp_host, self._expert_udp_port))
        self._expert_sock.settimeout(0.2)

        self._expert_thread = threading.Thread(
            target=self._expert_udp_loop,
            name="expert_udp_receiver",
            daemon=True,
        )
        self._expert_thread.start()

        print(
            f"[CARLA] Expert UDP receiver started at "
            f"{self._expert_udp_host}:{self._expert_udp_port}"
        )

    def _expert_udp_loop(self):
        while not self._expert_stop_event.is_set():
            try:
                data, _addr = self._expert_sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception:
                continue

            try:
                payload = json.loads(data.decode("utf-8"))
                ctrl = {
                    "steer_deg": float(payload.get("steer_deg", 0.0)),
                    "v_des_mps": float(payload.get("v_des_mps", 0.0)),
                    "brk_pressure": float(payload.get("brk_pressure", 0.0)),
                    "ad_mode": float(payload.get("ad_mode", 1.0)),
                    "turn_light": float(payload.get("turn_light", 0.0)),
                    "ts": float(payload.get("ts", time.time())),
                }
            except Exception:
                continue

            with self._expert_lock:
                self._latest_ros2_ctrl = ctrl

    def _clear_ros2_ctrl_cache(self):
        with self._expert_lock:
            self._latest_ros2_ctrl = None

    def get_ros2_ctrl(self, max_age_s=None):
        with self._expert_lock:
            ctrl = None if self._latest_ros2_ctrl is None else dict(self._latest_ros2_ctrl)

        if ctrl is None:
            return None

        if max_age_s is not None:
            ts = ctrl.get("ts", None)
            if ts is None:
                return None
            if (time.time() - float(ts)) > float(max_age_s):
                return None

        return ctrl

    def get_ego_speed(self):
        ego = self.get_ego_vehicle()
        if ego is None:
            return None
        try:
            vel = ego.get_velocity()
            return float(np.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))
        except Exception:
            return None

    def close(self):
        try:
            self._observer.destroy()
        except Exception:
            pass

        try:
            self._expert_stop_event.set()
        except Exception:
            pass

        try:
            if self._expert_sock is not None:
                self._expert_sock.close()
        except Exception:
            pass

        try:
            if self._expert_thread is not None and self._expert_thread.is_alive():
                self._expert_thread.join(timeout=1.0)
        except Exception:
            pass
