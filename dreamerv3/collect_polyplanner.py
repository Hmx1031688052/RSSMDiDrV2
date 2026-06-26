import datetime
import math
import pathlib
import time
import warnings

import embodied
import numpy as np
import ruamel.yaml as yaml

import carla
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray

import car_dreamer
from collect_utils import make_replay, wrap_env
from embodied.envs import from_gym


warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


# ============================================================
# 与环境动作空间对齐
# action = [acc, steer]
#   acc   : m/s^2, range [-3, 3]
#   steer : range [-1, 1]
#
# CarlaBaseEnv.get_vehicle_control(action) 内部会执行:
#   throttle = max(acc, 0) / 3
#   brake    = max(-acc, 0) / 3
#   control.steer = -steer
# ============================================================
ENV_ACC_MIN = -3.0
ENV_ACC_MAX = 3.0
ENV_STEER_MIN = -1.0
ENV_STEER_MAX = 1.0

# 规则专家给来的方向盘角度映射上限
RULE_STEER_MAX_DEG = 120.0

# ROS2 topics
OBS_TOPIC = "/obs_info"
CTRL_TOPIC = "/ctrl_info"
GLOBAL_TOPIC = "/global_info"
RESET_TOPIC = "/reset_info"
TRAJ_TOPIC = "/traj_best_vis"

SUR_RADIUS_M = 60.0
SUR_MAX_N = 30
SEND_SIZE_IN_CM = True


def get_speed_mps(v: carla.Vector3D) -> float:
    return float(np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z))


class PIDController:
    def __init__(
        self,
        kp,
        ki,
        kd,
        dt,
        output_limits=(ENV_ACC_MIN, ENV_ACC_MAX),
        integrator_limits=(-10.0, 10.0),
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.min_out = float(output_limits[0])
        self.max_out = float(output_limits[1])
        self.min_int = float(integrator_limits[0])
        self.max_int = float(integrator_limits[1])
        self.integral = 0.0
        self.prev_err = 0.0
        self.inited = False

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0
        self.inited = False

    def step(self, target, current):
        err = float(target - current)
        if not self.inited:
            self.prev_err = err
            self.inited = True

        self.integral += err * self.dt
        self.integral = max(self.min_int, min(self.max_int, self.integral))

        derr = (err - self.prev_err) / max(1e-6, self.dt)
        self.prev_err = err

        out = self.kp * err + self.ki * self.integral + self.kd * derr
        return max(self.min_out, min(self.max_out, out))


def clip_env_action(action):
    acc = float(np.clip(action[0], ENV_ACC_MIN, ENV_ACC_MAX))
    steer = float(np.clip(action[1], ENV_STEER_MIN, ENV_STEER_MAX))
    return np.array([acc, steer], dtype=np.float32)


def world_traj_to_ego_waypoints(traj_world, ego, num_wp=8, scale=30.0,
                                 source_dt=0.2, target_dt=0.5):
    """Convert world-frame trajectory to normalized ego-frame waypoints.

    Mirrors CarlaWptEnv._sample_trajectory_waypoints() so that waypoints
    injected by the collector policy are byte-identical to what the env's
    observation handler would produce — except they are synchronous with
    the current ROS2 planning cycle (no one-step delay).

    Args:
        traj_world: list of (x, y) world coordinates, consecutive points
                    spaced by source_dt seconds.
        ego: CARLA Vehicle actor.
        num_wp: number of waypoints (default 8, matching expert_waypoints8).
        scale: normalization scale in meters.
        source_dt: time interval [s] between consecutive trajectory points.
        target_dt: desired time interval [s] between output waypoints.

    Returns:
        np.array of shape (num_wp * 2,) with normalized ego-frame coords,
        or all-zeros if traj_world is None / too short.
    """
    if traj_world is None or len(traj_world) < 2:
        return np.zeros((num_wp * 2,), dtype=np.float32)

    ego_loc = ego.get_location()
    ego_yaw = math.radians(ego.get_transform().rotation.yaw)
    cos_y = math.cos(ego_yaw)
    sin_y = math.sin(ego_yaw)

    def _to_ego(x, y):
        dx = float(x) - float(ego_loc.x)
        dy = float(y) - float(ego_loc.y)
        lx = cos_y * dx + sin_y * dy
        ly = -sin_y * dx + cos_y * dy
        return lx, ly

    n = len(traj_world)
    source_dt = max(source_dt, 1e-6)
    target_dt = max(target_dt, 1e-6)

    # Trim trajectory prefix that has already fallen behind the vehicle.
    # The ROS2 planner generates the trajectory based on a slightly stale
    # vehicle state.  By the time we receive it the vehicle has moved forward,
    # so the first few trajectory points project to negative ego-frame x
    # (behind the vehicle).  Sampling waypoints from behind produces
    # physically impossible path curvature that confuses the LQR+FF controller.
    skip = 0
    for i in range(n):
        lx, _ = _to_ego(traj_world[i][0], traj_world[i][1])
        if lx >= 0.0:
            skip = i
            break

    if skip >= n - 2:
        return np.zeros((num_wp * 2,), dtype=np.float32)

    traj_trimmed = traj_world[skip:]
    n_trim = len(traj_trimmed)
    max_time = (n_trim - 1) * source_dt

    pts = []
    for k in range(num_wp):
        sample_time = min((k + 1) * target_dt, max_time)
        pos = sample_time / source_dt
        idx = int(math.floor(pos))
        idx = max(0, min(idx, n_trim - 2))
        t = float(pos - idx)
        t = max(0.0, min(1.0, t))
        wx = traj_trimmed[idx][0] + t * (traj_trimmed[idx + 1][0] - traj_trimmed[idx][0])
        wy = traj_trimmed[idx][1] + t * (traj_trimmed[idx + 1][1] - traj_trimmed[idx][1])
        lx, ly = _to_ego(wx, wy)
        pts.extend([lx / scale, ly / scale])
    return np.clip(np.asarray(pts, dtype=np.float32), -1.0, 1.0)


def ros_cmd_to_env_action(
    cmd,
    ego: carla.Vehicle,
    speed_pid: PIDController,
    steer_max_deg: float = RULE_STEER_MAX_DEG,
    use_brake_pressure: bool = True,
):
    """
    /ctrl_info -> env continuous action [acc, steer]

    协议:
      msg.orientation.x         = steer_cmd_deg
      msg.orientation.y         = v_des_mps
      msg.orientation.z         = brk_pressure (0..25)
      msg.orientation.w         = ADmode
      msg.linear_acceleration.x = turn_light
    """
    steer_deg, v_des_mps, brk_pressure, ADmode, turn_light = cmd
    del turn_light

    if ADmode < 0.0:
        return np.array([ENV_ACC_MIN, 0.0], dtype=np.float32)

    v_now = get_speed_mps(ego.get_velocity())
    acc_cmd = speed_pid.step(target=float(v_des_mps), current=float(v_now))

    if use_brake_pressure and brk_pressure > 1e-6:
        brake_ratio = float(np.clip(brk_pressure / 25.0, 0.0, 1.0))
        brake_acc = -brake_ratio * abs(ENV_ACC_MIN)
        acc_cmd = min(acc_cmd, brake_acc)

    acc_env = float(np.clip(acc_cmd, ENV_ACC_MIN, ENV_ACC_MAX))

    steer_carla = float(np.clip(float(steer_deg) / max(1e-6, steer_max_deg), -1.0, 1.0))
    steer_env = -steer_carla
    steer_env = float(np.clip(steer_env, ENV_STEER_MIN, ENV_STEER_MAX))

    return np.array([acc_env, steer_env], dtype=np.float32)


def collect_surrounding_vehicles(
    world: carla.World,
    ego: carla.Actor,
    radius_m: float = SUR_RADIUS_M,
    max_vehicles: int = SUR_MAX_N,
):
    ego_loc = ego.get_location()
    vehs = world.get_actors().filter("vehicle.*")

    results = []
    for v in vehs:
        if v.id == ego.id:
            continue

        loc = v.get_location()
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > radius_m:
            continue

        vel = v.get_velocity()
        speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
        yaw_deg = float(v.get_transform().rotation.yaw)

        bb = v.bounding_box
        length_m = float(bb.extent.x * 2.0)
        width_m = float(bb.extent.y * 2.0)

        results.append(
            {
                "id": int(v.id),
                "type": 1,
                "x": float(loc.x),
                "y": float(loc.y),
                "speed": float(speed),
                "yaw_deg": float(yaw_deg),
                "length_m": float(length_m),
                "width_m": float(width_m),
                "dist": float(dist),
            }
        )

    results.sort(key=lambda d: d["dist"])
    return results[:max_vehicles]


def build_marker_array_from_surrounding(sur_list, send_size_in_cm=True):
    msg = MarkerArray()
    for ob in sur_list:
        m = Marker()
        m.type = int(ob["type"])
        m.id = int(ob["id"])

        m.pose.position.x = float(ob["x"])
        m.pose.position.y = float(ob["y"])
        m.pose.position.z = float(ob["speed"])

        m.pose.orientation.x = float(ob["yaw_deg"])

        if send_size_in_cm:
            m.scale.x = float(ob["length_m"] * 100.0)
            m.scale.y = float(ob["width_m"] * 100.0)
        else:
            m.scale.x = float(ob["length_m"])
            m.scale.y = float(ob["width_m"])

        msg.markers.append(m)
    return msg


def build_global_info_imu_from_ego(ego: carla.Vehicle) -> Imu:
    g = Imu()

    tf = ego.get_transform()
    vel = ego.get_velocity()
    acc = ego.get_acceleration()
    ang = ego.get_angular_velocity()
    ctl = ego.get_control()

    g.orientation.x = float(tf.location.x)
    g.orientation.y = float(tf.location.y)
    g.orientation.z = float(tf.rotation.yaw)
    g.orientation.w = float(math.sqrt(vel.x * vel.x + vel.y * vel.y))

    g.linear_acceleration.x = float(vel.y)
    g.linear_acceleration.y = float(ang.z)
    g.linear_acceleration.z = float(acc.x)

    g.angular_velocity.x = float(ctl.steer)

    g.orientation_covariance[0] = 0.0
    g.orientation_covariance[1] = 0.0
    g.orientation_covariance[2] = float(tf.rotation.yaw)
    g.orientation_covariance[3] = 1.0

    return g


class CarlaRos2ExpertBridge(Node):
    def __init__(
        self,
        obs_topic=OBS_TOPIC,
        ctrl_topic=CTRL_TOPIC,
        global_topic=GLOBAL_TOPIC,
        reset_topic=RESET_TOPIC,
        traj_topic=TRAJ_TOPIC,
        cmd_timeout_s=0.25,
    ):
        super().__init__("carla_ros2_expert_bridge")

        self.pub_obs = self.create_publisher(MarkerArray, obs_topic, 10)
        self.pub_global = self.create_publisher(Imu, global_topic, 10)
        self.pub_reset = self.create_publisher(Bool, reset_topic, 10)
        self.sub_cmd = self.create_subscription(Imu, ctrl_topic, self._on_ctrl, 10)
        self.sub_traj = self.create_subscription(Marker, traj_topic, self._on_traj, 10)

        self.latest = None
        self.last_t = 0.0
        self.cmd_timeout_s = float(cmd_timeout_s)
        self.ignore_traj_until = 0.0

        # Trajectory data from quintic polynomial planner
        self.latest_traj_world = None   # list of (x, y) world coords
        self.last_traj_t = 0.0

    def _on_ctrl(self, msg: Imu):
        steer = float(msg.orientation.x)
        v_des = float(msg.orientation.y)
        brk_p = float(msg.orientation.z)
        adm = float(msg.orientation.w)
        turn = float(msg.linear_acceleration.x)
        self.latest = (steer, v_des, brk_p, adm, turn)
        self.last_t = time.time()

    def _on_traj(self, msg: Marker):
        """Receive best trajectory from /traj_best_vis (Marker::LINE_STRIP)."""
        now = time.time()
        if now < self.ignore_traj_until:
            return
        if not msg.points:
            return
        traj = [(float(p.x), float(p.y)) for p in msg.points]
        self.latest_traj_world = traj
        self.last_traj_t = now

    def clear_latest_control(self):
        self.latest = None
        self.last_t = 0.0
        self.latest_traj_world = None
        self.last_traj_t = 0.0

    def get_latest_control(self):
        if self.latest is None:
            return None
        if (time.time() - self.last_t) > self.cmd_timeout_s:
            return None
        return self.latest

    def get_latest_trajectory(self):
        """Return latest trajectory as list of (x, y) world coords, or None if stale."""
        if self.latest_traj_world is None:
            return None
        if (time.time() - self.last_traj_t) > self.cmd_timeout_s:
            return None
        return self.latest_traj_world

    def send_reset(self):
        self.clear_latest_control()
        # Drop stale planner markers that were published just before ROS reset
        # propagation. Otherwise they can be transformed in the next episode's
        # ego frame and saturate expert_waypoints8 at the clip boundary.
        self.ignore_traj_until = time.time() + max(0.75, self.cmd_timeout_s)
        msg = Bool()
        msg.data = True
        self.pub_reset.publish(msg)

    def publish_env(self, raw_env):
        base_env = raw_env.unwrapped if hasattr(raw_env, "unwrapped") else raw_env

        ego = getattr(base_env, "ego", None)
        if ego is None:
            print(f"[publish_env] base_env has no ego. attrs={dir(base_env)[:120]}", flush=True)
            return

        world_mgr = getattr(base_env, "_world", None)
        if world_mgr is None:
            world_mgr = getattr(base_env, "world", None)

        if world_mgr is None:
            print(f"[publish_env] base_env has no _world/world. attrs={dir(base_env)[:120]}", flush=True)
            return

        carla_world = getattr(world_mgr, "carla_world", None)
        if carla_world is None and hasattr(world_mgr, "_world"):
            carla_world = world_mgr._world

        if carla_world is None and hasattr(world_mgr, "get_actors"):
            carla_world = world_mgr

        if carla_world is None:
            print(
                f"[publish_env] world_mgr has no carla_world/_world/get_actors. attrs={dir(world_mgr)[:120]}",
                flush=True,
            )
            return

        global_msg = build_global_info_imu_from_ego(ego)
        global_msg.header.frame_id = "map"
        global_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_global.publish(global_msg)

        sur_list = collect_surrounding_vehicles(
            world=carla_world,
            ego=ego,
            radius_m=SUR_RADIUS_M,
            max_vehicles=SUR_MAX_N,
        )
        obs_msg = build_marker_array_from_surrounding(
            sur_list,
            send_size_in_cm=SEND_SIZE_IN_CM,
        )
        stamp = self.get_clock().now().to_msg()
        for mk in obs_msg.markers:
            mk.header.frame_id = "map"
            mk.header.stamp = stamp
        self.pub_obs.publish(obs_msg)


class Ros2ExpertCollectorPolicy:
    def __init__(
        self,
        raw_envs,
        ros_bridge: CarlaRos2ExpertBridge,
        act_key="action",
        steer_max_deg=RULE_STEER_MAX_DEG,
        use_brake_pressure=True,
        spin_timeout_sec=0.0,
    ):
        self.raw_envs = list(raw_envs)
        self.bridge = ros_bridge
        self.act_key = act_key
        self.steer_max_deg = float(steer_max_deg)
        self.use_brake_pressure = bool(use_brake_pressure)
        self.spin_timeout_sec = float(spin_timeout_sec)

        self.speed_pids = [
            PIDController(
                kp=1.0,
                ki=0.0,
                kd=0.0,
                dt=0.1,
                output_limits=(ENV_ACC_MIN, ENV_ACC_MAX),
                integrator_limits=(-10.0, 10.0),
            )
            for _ in self.raw_envs
        ]

        self.timeout_count = 0
        self.last_env_actions = [
            np.array([0.0, 0.0], dtype=np.float32) for _ in self.raw_envs
        ]

    def __call__(self, obs, state=None, **kwargs):
        del state, kwargs

        batch_size = len(self.raw_envs)
        outs = self._empty_policy(batch_size)

        is_first = np.asarray(
            obs.get("is_first", np.zeros((batch_size,), dtype=np.bool_))
        ).reshape(-1)

        override_actions = []
        current_env_actions = []
        # Collect fresh waypoints to override the delayed ones in obs
        override_expert_waypoints = []

        for i, env in enumerate(self.raw_envs):
            if i < len(is_first) and bool(is_first[i]):
                self.speed_pids[i].reset()
                self.bridge.send_reset()

            try:
                fps = 1.0 / float(env._world._settings.fixed_delta_seconds)
                self.speed_pids[i].dt = 1.0 / max(fps, 1e-6)
            except Exception:
                pass

            self.bridge.publish_env(env)
            rclpy.spin_once(self.bridge, timeout_sec=self.spin_timeout_sec)

            cmd = self.bridge.get_latest_control()

            if cmd is None:
                env_action = np.array([ENV_ACC_MIN, 0.0], dtype=np.float32)
                self.timeout_count += 1
                if self.timeout_count <= 5 or self.timeout_count % 20 == 0:
                    print(
                        f"[ROS2 Expert] /ctrl_info timeout, fallback action={env_action.tolist()} "
                        f"(count={self.timeout_count})",
                        flush=True,
                    )
            else:
                env_action = ros_cmd_to_env_action(
                    cmd=cmd,
                    ego=env.ego,
                    speed_pid=self.speed_pids[i],
                    steer_max_deg=self.steer_max_deg,
                    use_brake_pressure=self.use_brake_pressure,
                )

            # ── Fix: inject synchronous expert waypoints ──
            # Previously the ROS2 trajectory was only stashed on the env via
            # set_expert_trajectory(), which meant the observation handler would
            # only pick it up on the *next* env.step() — causing a one-step
            # delay between obs_t and expert_waypoints8_t.  Now we convert the
            # trajectory to normalized ego-frame waypoints right here and put
            # them directly into outs, so they are synchronous with obs_t.
            traj = self.bridge.get_latest_trajectory()
            if traj is not None and len(traj) >= 2:
                wps = world_traj_to_ego_waypoints(
                    traj, env.ego,
                    num_wp=8,
                    scale=30.0,
                    source_dt=0.2,
                    target_dt=0.5,
                )
                override_expert_waypoints.append(wps)
            else:
                # Fall back to whatever the observation handler produced (may be
                # delayed or fallback route, but better than all-zeros).
                override_expert_waypoints.append(
                    obs.get("expert_waypoints8",
                            np.zeros((8 * 2,), dtype=np.float32))[i]
                    if "expert_waypoints8" in obs
                    else np.zeros((8 * 2,), dtype=np.float32)
                )

            # Keep env-side stash for visualisation / fallback (harmless)
            if hasattr(env, "set_expert_trajectory"):
                env.set_expert_trajectory(traj)

            env_action = clip_env_action(env_action)
            current_env_actions.append(env_action)
            override_actions.append(self._env_to_policy_action(env_action, env))

        self.last_env_actions = current_env_actions
        outs[self.act_key] = np.asarray(override_actions, dtype=np.float32)
        # Override delayed expert_waypoints8 with synchronous version.
        # The Driver merges outs on top of obs, so this field wins.
        outs["expert_waypoints8"] = np.stack(override_expert_waypoints, axis=0)
        return outs, None

    def _empty_policy(self, batch_size):
        shape = self.raw_envs[0].action_space.shape
        return {self.act_key: np.zeros((batch_size,) + shape, dtype=np.float32)}

    def _env_to_policy_action(self, env_action, env):
        space = env.action_space
        if hasattr(space, "n"):
            raise NotImplementedError("当前实现只支持连续动作空间。")

        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        env_action = np.asarray(env_action, dtype=np.float32)

        policy_action = 2.0 * (env_action - low) / (high - low + 1e-6) - 1.0
        return np.clip(policy_action, -1.0, 1.0)


def _last_bool_from_source(src, key):
    if key not in src:
        return None
    value = np.asarray(src[key])
    if value.size == 0:
        return None
    return bool(value.reshape(-1)[-1])


def _is_success_episode(ep, ep_info):
    for src in (ep_info, ep):
        for key in ("destination_reached", "is_success"):
            result = _last_bool_from_source(src, key)
            if result is not None:
                return result
    return False


def _pick_timestep_value(src, key, t):
    if key not in src:
        return None
    arr = np.asarray(src[key])

    if arr.ndim == 0:
        return np.array(arr, copy=True)

    if arr.shape[0] <= t:
        return None

    return np.array(arr[t], copy=True)

def _episode_length(ep):
    for value in ep.values():
        arr = np.asarray(value)
        if arr.ndim > 0:
            return int(arr.shape[0])
    return 0

def _episode_step(ep, t):
    step = {}
    for key, value in ep.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] > t:
            step[key] = np.array(arr[t], copy=True)
    return step

def _replay_step_defaults():
    return {
        "destination_reached": np.array([False], dtype=np.bool_),
        "is_success": np.array([False], dtype=np.bool_),
        "out_of_lane": np.array([False], dtype=np.bool_),
        "time_exceeded": np.array([False], dtype=np.bool_),
        "is_collision": np.array([False], dtype=np.bool_),
    }
def _ensure_bool_vector(value):
    value = np.asarray(value, dtype=np.bool_)
    if value.shape == ():
        value = value.reshape(1)
    return value
def main(argv=None):
    yaml_path = pathlib.Path(__file__).resolve().with_name("dreamerv3.yaml")
    model_configs = yaml.YAML(typ="safe").load(
        embodied.Path(str(yaml_path)).read()
    )
    base = embodied.Config({"dreamerv3": model_configs["defaults"]})
    base = base.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=["carla_static_obstacle"]).parse_known(argv)
    quick, other = embodied.Flags(base).parse_known(other)

    raw_env, env_config = car_dreamer.create_task(parsed.task[0], argv)
    config = quick.update(env_config)
    config = embodied.Flags(config).parse(other)
    cfg = config.dreamerv3

    logdir = embodied.Path(cfg.logdir)
    logdir.mkdirs()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config.save(str(logdir / f"expert_collect_{timestamp}.yaml"))

    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    replay, replay_dir = make_replay(cfg, logdir)
    print(f"Replay dir: {replay_dir}")

    env = from_gym.FromGym(raw_env)
    env = wrap_env(env, cfg)
    env = embodied.BatchEnv([env], parallel=False)

    if not rclpy.ok():
        rclpy.init(args=None)

    ros_bridge = CarlaRos2ExpertBridge(
        obs_topic=OBS_TOPIC,
        ctrl_topic=CTRL_TOPIC,
        global_topic=GLOBAL_TOPIC,
        reset_topic=RESET_TOPIC,
        traj_topic=TRAJ_TOPIC,
        cmd_timeout_s=0.25,
    )

    policy = Ros2ExpertCollectorPolicy(
        raw_envs=[raw_env],
        ros_bridge=ros_bridge,
        act_key="action",
        steer_max_deg=RULE_STEER_MAX_DEG,
        use_brake_pressure=True,
        spin_timeout_sec=0.0,
    )

    episodes_done = {"count": 0}
    saved_episodes = {"count": 0}
    target_episodes = int(config.env.expert_collection.episodes)

    def on_step(tran, _, worker):
        del tran, worker
        step.increment()

    def on_episode(ep, ep_info, worker):
        del worker
        episodes_done["count"] += 1

        success = _is_success_episode(ep, ep_info)
        episode_return = float(np.asarray(ep["reward"]).sum())

        if success:
            defaults = _replay_step_defaults()

            for t in range(_episode_length(ep)):
                step_data = _episode_step(ep, t)

                if "is_first" in step_data:
                    step_data["episode_start"] = np.array(step_data["is_first"], copy=True)
                    step_data["reset_export"] = np.array(step_data["is_first"], copy=True)

                # ep contains the reset transition; ep_info starts at env steps.
                info_t = t - 1
                for key, default in defaults.items():
                    value = None
                    if info_t >= 0:
                        value = _pick_timestep_value(ep_info, key, info_t)
                        if value is None:
                            value = _pick_timestep_value(ep, key, info_t)

                    if value is None:
                        value = np.array(default, copy=True)

                    step_data[key] = _ensure_bool_vector(value)


                replay.add(step_data, 0)

            saved_episodes["count"] += 1

        last_env_action = policy.last_env_actions[0]
        logger.add(
            {
                "episodes": episodes_done["count"],
                "saved_episodes": saved_episodes["count"],
                "success": float(success),
                "episode_return": episode_return,
                "last_acc_env": float(last_env_action[0]),
                "last_steer_env": float(last_env_action[1]),
            },
            prefix="expert_collect",
        )
        logger.write()

        print(
            f"[ROS2 Expert Collect] episode={episodes_done['count']} "
            f"success={success} saved={saved_episodes['count']} "
            f"return={episode_return:.3f} "
            f"replay_size={len(replay)}",
            flush=True,
        )


    driver = embodied.Driver(env)
    driver.on_step(on_step)
    driver.on_episode(on_episode)

    try:
        while episodes_done["count"] < target_episodes:
            driver(policy, steps=100)

        replay.save(wait=True)
        logger.write()
    finally:
        try:
            ros_bridge.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
