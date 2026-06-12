"""
PIDPP.py

Drop-in waypoint controller for the current CarDreamer/CARLA pipeline.

Input interface is intentionally aligned with the existing
poly_planner_controller.PolyPlannerController:

    ctrl = PIDPPController(vehicle, target_speed=5.0)
    control = ctrl.run_step(waypoints_ego, waypoint_scale)

where waypoints_ego is either shape (16,) or (8, 2), normalized to [-1, 1].
The 8 waypoints are assumed to be time-parameterized expert/planner outputs:

    p1..p8 = t+0.5s, t+1.0s, ..., t+4.0s

Lateral control: Pure Pursuit (PP)
Longitudinal control: PID speed tracking
Target velocity: terminal-horizon style, close to original planner usage
                 trajF2C.v[40] at t ~= 4.0s, approximated from the last
                 waypoint segment and lightly blended with the previous one.

Place this file at:
    car_dreamer/toolkit/planner/PIDPP.py

Then import either PIDPPController or the compatibility alias
PolyPlannerController from this module.
"""

import math

import numpy as np

try:
    import carla
except ImportError:  # Allows static analysis outside a CARLA runtime.
    carla = None


# -----------------------------------------------------------------------------
# Vehicle / controller constants
# -----------------------------------------------------------------------------


class _VehicleParams:
    # Same wheelbase as the existing poly_planner_controller.py / hpp vehicle model.
    L: float = 2.85


VP = _VehicleParams()

# The expert waypoints are sampled every 0.5 s: 8 points -> 4.0 s horizon.
WAYPOINT_DT: float = 0.5
DENSE_DT: float = 0.1
NUM_DENSE_POINTS: int = 41  # 0.0s..4.0s at 0.1s resolution, matches traj index 40.

# Longitudinal PID, matching the existing poly planner controller style.
PID_P: float = 1.0
PID_I: float = 0.0002
PID_D: float = 0.0
PID_DT: float = 0.1
A_MAX: float = 3.0
A_MIN: float = -3.0

# Low-speed system speed limit requested by the user.
DEFAULT_TARGET_SPEED: float = 5.0
TARGET_SPEED_MAX: float = 5.0
TARGET_SPEED_FILTER_ALPHA: float = 0.4

# Optional safety cap from path curvature.  The primary speed still comes from
# terminal waypoint spacing, but this prevents obviously unsafe acceleration into
# a noisy sharp turn. Set to False if you want terminal-speed-only behavior.
USE_CURVATURE_SPEED_CAP: bool = True
A_LAT_MAX: float = 1.5

# Pure Pursuit settings for <=5 m/s.
LOOKAHEAD_BASE: float = 1.5
LOOKAHEAD_GAIN: float = 0.45
LOOKAHEAD_MIN: float = 2.0
LOOKAHEAD_MAX: float = 6.0

# CARLA steer limit used in the current controller.
STEER_MAX_DEG: float = 42.0
STEER_MAX_RAD: float = math.radians(STEER_MAX_DEG)
STEER_FILTER_ALPHA: float = 0.55

# Waypoint sanitization / outlier repair.
MAX_SEGMENT_FACTOR: float = 2.4   # max allowed segment ~= v_max * dt * factor
MAX_LATERAL_ABS: float = 12.0     # meters
MAX_BACKWARD_X: float = 1.0       # meters behind ego
MAX_HEADING_REVERSAL_COS: float = -0.65
MIN_STEP_FOR_DIRECTION: float = 0.05


class PIDController:
    """Small PID with integral clamp, outputting desired acceleration."""

    def __init__(self, kp, ki, kd, dt, out_min=-1.0, out_max=1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.out_min = float(out_min)
        self.out_max = float(out_max)
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    def step(self, setpoint, measurement):
        error = float(setpoint) - float(measurement)
        if not self._initialized:
            self._prev_error = error
            self._initialized = True

        self._integral += error * self.dt
        self._integral = float(np.clip(self._integral, -5.0, 5.0))

        derivative = (error - self._prev_error) / max(self.dt, 1e-6)
        self._prev_error = error

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(out, self.out_min, self.out_max))


class PIDPPController:
    """
    PID longitudinal + Pure Pursuit lateral waypoint controller.

    This class is a drop-in replacement for the existing PolyPlannerController:
      - __init__(vehicle, target_speed=5.0)
      - set_target_speed(speed_ms)
      - reset()
      - run_step(waypoints_ego, waypoint_scale=30.0) -> carla.VehicleControl
    """

    def __init__(self, vehicle, target_speed=DEFAULT_TARGET_SPEED):
        self._vehicle = vehicle
        self._target_speed = float(target_speed)
        self._speed_limit = float(np.clip(target_speed, 0.0, TARGET_SPEED_MAX))
        if self._speed_limit <= 1e-6:
            self._speed_limit = TARGET_SPEED_MAX

        self._speed_pid = PIDController(
            kp=PID_P,
            ki=PID_I,
            kd=PID_D,
            dt=PID_DT,
            out_min=A_MIN,
            out_max=A_MAX,
        )

        self._low_speed_counter = 0
        self.debug = {}

    # ------------------------------------------------------------------
    # Public API, aligned with the existing pipeline
    # ------------------------------------------------------------------

    def set_target_speed(self, speed_ms):
        self._target_speed = float(speed_ms)
        self._speed_limit = float(np.clip(speed_ms, 0.0, TARGET_SPEED_MAX))
        if self._speed_limit <= 1e-6:
            self._speed_limit = TARGET_SPEED_MAX

    def reset(self):
        self._speed_pid.reset()
        self._low_speed_counter = 0
        self.debug = {}
        for name in (
            "_target_speed_prev",
            "_steer_prev_deg",
            "_last_good_waypoints",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def run_step(self, waypoints_ego, waypoint_scale=30.0):
        """
        Convert network/policy waypoints to CARLA VehicleControl.

        Args:
            waypoints_ego:
                Shape (16,) or (8, 2). Values are normalized by waypoint_scale,
                exactly like the current planner target pipeline.
            waypoint_scale:
                Meters per normalized unit. Should be the same value as
                planner_target.waypoint_scale.

        Returns:
            carla.VehicleControl
        """
        if carla is None:
            raise RuntimeError("PIDPPController requires the CARLA Python package at runtime.")

        pts_raw = self._decode_waypoints(waypoints_ego, waypoint_scale)
        pts_clean = self._sanitize_waypoints(pts_raw)
        pts_with_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), pts_clean])

        # Smooth to 41 points at 0.1s spacing. This mirrors the planner horizon
        # used by trajF2C.v[40] and gives PP a dense path to track.
        pts_dense = self._smooth_path_time_cubic(pts_with_origin, NUM_DENSE_POINTS)

        current_speed = self._get_speed_ms()
        target_speed = self._estimate_target_speed(pts_clean, pts_dense)

        steer_rad = self._pure_pursuit_steer(pts_dense, current_speed)
        steer_deg = math.degrees(steer_rad)
        steer_deg = float(np.clip(steer_deg, -STEER_MAX_DEG, STEER_MAX_DEG))

        if not hasattr(self, "_steer_prev_deg"):
            self._steer_prev_deg = steer_deg
        steer_deg = (
            STEER_FILTER_ALPHA * steer_deg
            + (1.0 - STEER_FILTER_ALPHA) * self._steer_prev_deg
        )
        self._steer_prev_deg = steer_deg

        # Longitudinal PID: target speed -> desired acceleration -> throttle/brake.
        a_des = self._speed_pid.step(target_speed, current_speed)

        # Same low-speed anti-windup idea as the existing controller.
        if current_speed < 1.0:
            self._low_speed_counter += 1
            if self._low_speed_counter > 5:
                self._speed_pid.reset()
                self._low_speed_counter = 0
        else:
            self._low_speed_counter = 0

        a_des = float(np.clip(a_des, A_MIN, A_MAX))
        if abs(a_des) < 0.05:
            a_des = 0.0

        throttle, brake = self._accel_to_control(a_des)

        # Ego frame uses y positive left. Pure Pursuit positive steer means left.
        # In this CARLA pipeline, normalized VehicleControl.steer is inverted in
        # The env applies steer as -steer_deg / STEER_MAX_DEG, so keep the
        # same sign convention as the existing PolyPlannerController.
        steer_carla = -steer_deg / STEER_MAX_DEG

        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.steer = float(np.clip(steer_carla, -1.0, 1.0))
        control.brake = float(brake)
        control.hand_brake = False
        control.manual_gear_shift = False

        self.debug = {
            "current_speed": float(current_speed),
            "target_speed": float(target_speed),
            "accel_des": float(a_des),
            "steer_deg_pp": float(steer_deg),
            "steer_carla": float(control.steer),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "num_waypoints": int(len(pts_clean)),
        }
        return control

    # ------------------------------------------------------------------
    # Waypoint decoding, outlier removal / repair, smoothing
    # ------------------------------------------------------------------

    def _decode_waypoints(self, waypoints_ego, waypoint_scale):
        arr = np.asarray(waypoints_ego, dtype=np.float64).reshape(-1)
        num = len(arr) // 2
        if num < 1:
            return self._fallback_waypoints(8)

        arr = arr[: num * 2]
        pts = arr.reshape(num, 2) * float(waypoint_scale)
        if not np.isfinite(pts).all():
            return self._fallback_waypoints(num)
        return pts

    def _fallback_waypoints(self, num=8):
        speed = float(np.clip(self._target_speed, 0.0, self._speed_limit))
        xs = (np.arange(num, dtype=np.float64) + 1.0) * WAYPOINT_DT * speed
        ys = np.zeros(num, dtype=np.float64)
        return np.stack([xs, ys], axis=-1)

    def _sanitize_waypoints(self, pts):
        """
        Remove/repair abnormal waypoint jumps while preserving the 0.5s slots.

        Because target speed is inferred from the temporal waypoint spacing, this
        function does not simply delete points. Instead it clamps or replaces
        outliers with a plausible continuation so p1..p8 still mean 0.5..4.0s.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] == 0 or not np.isfinite(pts).all():
            return self._fallback_waypoints(8)

        n = pts.shape[0]
        speed_for_bounds = max(0.5, self._speed_limit)
        max_step = max(1.5, speed_for_bounds * WAYPOINT_DT * MAX_SEGMENT_FACTOR)
        horizon_max = max(8.0, speed_for_bounds * WAYPOINT_DT * n * MAX_SEGMENT_FACTOR)

        cleaned = np.zeros_like(pts)
        prev = np.zeros(2, dtype=np.float64)
        last_good_step = np.array([speed_for_bounds * WAYPOINT_DT, 0.0], dtype=np.float64)

        for i, raw in enumerate(pts):
            p = np.asarray(raw, dtype=np.float64).copy()
            bad = False

            if not np.isfinite(p).all():
                bad = True
            else:
                if np.linalg.norm(p) > horizon_max:
                    bad = True
                if p[0] < -MAX_BACKWARD_X:
                    bad = True

            if bad:
                p = prev + last_good_step

            p[1] = float(np.clip(p[1], -MAX_LATERAL_ABS, MAX_LATERAL_ABS))

            step = p - prev
            step_norm = float(np.linalg.norm(step))

            # Reject an abrupt reversal relative to the previous reliable step.
            prev_norm = float(np.linalg.norm(last_good_step))
            if step_norm > MIN_STEP_FOR_DIRECTION and prev_norm > MIN_STEP_FOR_DIRECTION:
                cos_angle = float(np.dot(step, last_good_step) / (step_norm * prev_norm))
                if cos_angle < MAX_HEADING_REVERSAL_COS:
                    p = prev + last_good_step
                    step = p - prev
                    step_norm = float(np.linalg.norm(step))

            # Clamp physically implausible segment length for <=5m/s data.
            if step_norm > max_step:
                p = prev + step / max(step_norm, 1e-6) * max_step
                step = p - prev
                step_norm = float(np.linalg.norm(step))

            # Do not allow large backward progression along ego x.
            if p[0] < prev[0] - 0.5:
                p[0] = prev[0] - 0.5
                step = p - prev
                step_norm = float(np.linalg.norm(step))

            cleaned[i] = p
            if step_norm > MIN_STEP_FOR_DIRECTION:
                last_good_step = step
            else:
                # If the planner is stopping, keep a decaying tiny step rather
                # than inventing a large one.
                last_good_step = 0.5 * last_good_step
            prev = p

        if not np.isfinite(cleaned).all():
            return self._fallback_waypoints(n)

        self._last_good_waypoints = cleaned.copy()
        return cleaned

    @staticmethod
    def _smooth_path_time_cubic(pts_with_origin, num_points=NUM_DENSE_POINTS):
        """Cubic Hermite time-spline through p0..p8, sampled densely.

        This is equivalent to fitting a smooth parametric curve x(t), y(t) over
        the 4s horizon. It avoids requiring scipy and preserves the 0.5s time
        semantics of the expert waypoints.
        """
        pts = np.asarray(pts_with_origin, dtype=np.float64).reshape(-1, 2)
        n = pts.shape[0]
        if n == 0:
            return np.zeros((num_points, 2), dtype=np.float64)
        if n == 1:
            return np.tile(pts[0], (num_points, 1))
        if n == 2:
            u = np.linspace(0.0, 1.0, num_points)
            return pts[0] + u[:, None] * (pts[1] - pts[0])

        t = np.arange(n, dtype=np.float64) * WAYPOINT_DT
        tang = np.zeros_like(pts)
        tang[0] = (pts[1] - pts[0]) / WAYPOINT_DT
        tang[-1] = (pts[-1] - pts[-2]) / WAYPOINT_DT
        tang[1:-1] = (pts[2:] - pts[:-2]) / (2.0 * WAYPOINT_DT)

        # Tangent limiting reduces cubic overshoot around noisy waypoints.
        for i in range(n):
            if i == 0:
                local_step = np.linalg.norm(pts[1] - pts[0]) / WAYPOINT_DT
            elif i == n - 1:
                local_step = np.linalg.norm(pts[-1] - pts[-2]) / WAYPOINT_DT
            else:
                s0 = np.linalg.norm(pts[i] - pts[i - 1]) / WAYPOINT_DT
                s1 = np.linalg.norm(pts[i + 1] - pts[i]) / WAYPOINT_DT
                local_step = max(s0, s1)
            max_tangent = max(0.5, 2.5 * local_step)
            tang_norm = float(np.linalg.norm(tang[i]))
            if tang_norm > max_tangent:
                tang[i] *= max_tangent / max(tang_norm, 1e-6)

        t_new = np.linspace(t[0], t[-1], num_points)
        out = np.zeros((num_points, 2), dtype=np.float64)

        for j, tj in enumerate(t_new):
            idx = int(np.searchsorted(t, tj, side="right") - 1)
            idx = max(0, min(idx, n - 2))
            dt = t[idx + 1] - t[idx]
            u = (tj - t[idx]) / max(dt, 1e-6)
            u2 = u * u
            u3 = u2 * u

            h00 = 2.0 * u3 - 3.0 * u2 + 1.0
            h10 = u3 - 2.0 * u2 + u
            h01 = -2.0 * u3 + 3.0 * u2
            h11 = u3 - u2

            out[j] = (
                h00 * pts[idx]
                + h10 * dt * tang[idx]
                + h01 * pts[idx + 1]
                + h11 * dt * tang[idx + 1]
            )

        out[0] = np.array([0.0, 0.0], dtype=np.float64)
        return out

    # ------------------------------------------------------------------
    # Lateral: Pure Pursuit
    # ------------------------------------------------------------------

    def _pure_pursuit_steer(self, pts_ego_dense, current_speed):
        pts = np.asarray(pts_ego_dense, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 2 or not np.isfinite(pts).all():
            return 0.0

        path_len = self._path_length(pts)
        if path_len < 1e-3:
            return 0.0

        lookahead = LOOKAHEAD_BASE + LOOKAHEAD_GAIN * max(0.0, float(current_speed))
        lookahead = float(np.clip(lookahead, LOOKAHEAD_MIN, LOOKAHEAD_MAX))
        lookahead = min(lookahead, path_len)

        target = self._point_at_arclength(pts, lookahead)
        x_t, y_t = float(target[0]), float(target[1])
        dist2 = x_t * x_t + y_t * y_t
        if dist2 < 1e-6:
            return 0.0

        # If the selected point is very close or slightly behind due to noise,
        # use the farthest forward available point instead.
        if x_t < 0.1:
            forward = pts[pts[:, 0] > 0.1]
            if len(forward) > 0:
                target = forward[min(len(forward) - 1, 5)]
                x_t, y_t = float(target[0]), float(target[1])
                dist2 = x_t * x_t + y_t * y_t

        curvature = 2.0 * y_t / max(dist2, 1e-6)
        steer_rad = math.atan(VP.L * curvature)
        return float(np.clip(steer_rad, -STEER_MAX_RAD, STEER_MAX_RAD))

    @staticmethod
    def _path_length(pts):
        if len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    @staticmethod
    def _point_at_arclength(pts, distance):
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if distance <= 0.0:
            return pts[0]
        if distance >= cum[-1]:
            return pts[-1]

        idx = int(np.searchsorted(cum, distance, side="right") - 1)
        idx = max(0, min(idx, len(seg) - 1))
        ds = max(seg[idx], 1e-6)
        ratio = (distance - cum[idx]) / ds
        return pts[idx] + ratio * (pts[idx + 1] - pts[idx])

    # ------------------------------------------------------------------
    # Longitudinal: target speed from terminal 4s waypoint speed + PID
    # ------------------------------------------------------------------

    def _estimate_target_speed(self, pts_clean, pts_dense=None):
        """Estimate target velocity close to C++ trajF2C.v[40] usage.

        Since the 8 points are sampled at 0.5s intervals, segment speed is:
            v_i = ||p_i - p_{i-1}|| / 0.5

        The original planner used trajF2C.v[40] where dt=0.1s, i.e. roughly
        the t+4.0s terminal velocity. From only 8 positions, the closest robust
        approximation is the last interval speed, lightly blended with the
        previous interval:
            v_target_raw = 0.8 * v[3.5,4.0] + 0.2 * v[3.0,3.5]
        """
        pts = np.asarray(pts_clean, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 1 or not np.isfinite(pts).all():
            target = float(np.clip(self._target_speed, 0.0, self._speed_limit))
        else:
            pts_time = np.vstack([np.zeros((1, 2), dtype=np.float64), pts])
            dists = np.linalg.norm(np.diff(pts_time, axis=0), axis=1)
            speeds = dists / max(WAYPOINT_DT, 1e-6)
            speeds = np.clip(speeds, 0.0, self._speed_limit)

            target = float(speeds[-1])
            if len(speeds) >= 2:
                target = 0.8 * float(speeds[-1]) + 0.2 * float(speeds[-2])

        if USE_CURVATURE_SPEED_CAP and pts_dense is not None:
            kappa_max = self._estimate_max_curvature(pts_dense)
            if kappa_max > 1e-6:
                v_curve = math.sqrt(A_LAT_MAX / (kappa_max + 1e-4))
                target = min(target, v_curve)

        target = float(np.clip(target, 0.0, self._speed_limit))

        if not hasattr(self, "_target_speed_prev"):
            self._target_speed_prev = target
        target = (
            TARGET_SPEED_FILTER_ALPHA * target
            + (1.0 - TARGET_SPEED_FILTER_ALPHA) * self._target_speed_prev
        )
        self._target_speed_prev = target
        return float(np.clip(target, 0.0, self._speed_limit))

    @staticmethod
    def _estimate_max_curvature(pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3 or not np.isfinite(pts).all():
            return 0.0

        kappas = []
        for i in range(1, len(pts) - 1):
            p0 = pts[i - 1]
            p1 = pts[i]
            p2 = pts[i + 1]
            a = float(np.linalg.norm(p1 - p0))
            b = float(np.linalg.norm(p2 - p1))
            c = float(np.linalg.norm(p2 - p0))
            if a < 1e-4 or b < 1e-4 or c < 1e-4:
                continue
            cross = float((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
            kappa = abs(2.0 * cross) / max(a * b * c, 1e-9)
            if np.isfinite(kappa):
                kappas.append(kappa)
        if not kappas:
            return 0.0
        # Use a high percentile rather than a hard max to ignore one-point noise.
        return float(np.percentile(kappas, 90.0))

    @staticmethod
    def _accel_to_control(a_des):
        if a_des >= 0.0:
            return min(float(a_des) / A_MAX, 1.0), 0.0
        return 0.0, min(abs(float(a_des)) / abs(A_MIN), 1.0)

    def _get_speed_ms(self):
        vel = self._vehicle.get_velocity()
        return float(math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))


# Compatibility alias:
# This lets the rest of the existing pipeline keep the variable/class name
# PolyPlannerController after only changing the import module to PIDPP.
PolyPlannerController = PIDPPController
