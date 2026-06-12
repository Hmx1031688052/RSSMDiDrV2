"""
LQR+FF+PID controller matching the C++ poly_planner_onsite control logic.

This is the NumPy/CARLA inference-time controller.  It shares the exact same
mathematical formulas as the JAX differentiable controller in
dreamerv3/jax_controller.py, so that training gradients align with inference
behaviour.

Lateral control:  FF (bicycle-model feedforward) + LQR feedback
Longitudinal control: PID speed tracking → desired acceleration

Reference: poly_planner_onsite (4).hpp
  - control_FF:  lines 1674-1678
  - control_LQR: lines 1708-1712
  - Error computation at preview indices 2,4,6,8,10 with weights G0..G4
  - control_brake / PID: lines 958-993

Usage (drop-in replacement for the existing pipeline):
    ctrl = PolyPlannerController(vehicle, target_speed=5.0)
    control = ctrl.run_step(waypoints_ego_16, waypoint_scale=30.0)
"""

import math
from collections import deque

import numpy as np

try:
    import carla
except ImportError:
    carla = None


# ══════════════════════════════════════════════════════════════════════════════
# Vehicle parameters — poly_planner_onsite.hpp lines 37-43
# ══════════════════════════════════════════════════════════════════════════════

class _VehicleParams:
    m: float = 1890.0          # total mass [kg]
    R: float = 0.33            # tire radius [m]
    Iz: float = 3800.0         # yaw moment of inertia [kg·m²]
    L: float = 2.85            # wheelbase [m]
    lf: float = 1.62           # CoG to front axle [m]
    lr: float = 1.23           # CoG to rear axle [m]
    Cf: float = 110000.0       # front cornering stiffness [N/rad]
    Cr: float = 108000.0       # rear cornering stiffness [N/rad]


VP = _VehicleParams()

# ══════════════════════════════════════════════════════════════════════════════
# Control gains — poly_planner_onsite.hpp lines 47-50, 189, 639
# ══════════════════════════════════════════════════════════════════════════════

K_FF: float = 1.0
K_FB: float = 1.0
PRE_N: int = 2
PREVIEW_WEIGHTS = (0.2, 0.3, 0.3, 0.1, 0.1)   # G0..G4
LQR_K = (50.0142, 0.1093, 1.2018, 0.0013)       # [ephi, dephi, ed, ded]

PID_P: float = 1.0
PID_I: float = 0.0002
PID_D: float = 0.0
PID_DT: float = 0.1

A_MAX: float = 3.0
A_MIN: float = -3.0

STEER_MAX_DEG: float = 42.0            # physical steer limit at wheels (C++ line 645)
STEER_MAX_RAD: float = math.radians(STEER_MAX_DEG)

# Normalization divisor for Dreamer action-space steer output.
# MUST equal the collector's RULE_STEER_MAX_DEG so that the prev_action
# fed to the RSSM is in the same scale as the expert action in replay.
STEER_ACTION_NORM_DEG: float = 120.0   # matches collect_polyplanner.RULE_STEER_MAX_DEG

WAYPOINT_DT: float = 0.5
NUM_DENSE: int = 41
DENSE_DT: float = 0.1

# Default target speed
DEFAULT_TARGET_SPEED: float = 5.0
TARGET_SPEED_MAX: float = 8.0
TARGET_SPEED_FILTER_ALPHA: float = 0.4
STEER_FILTER_ALPHA: float = 0.55

# Waypoint sanitization
MAX_LATERAL_ABS: float = 12.0
MAX_BACKWARD_X: float = 1.0


class PIDController:
    """Minimal PID with integral clamping (matching SpeedControlPID in hpp)."""

    def __init__(self, kp, ki, kd, dt, out_min=-1.0, out_max=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.out_min = out_min
        self.out_max = out_max
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
        self._integral = max(-5.0, min(5.0, self._integral))
        deriv = (error - self._prev_error) / max(self.dt, 1e-6)
        self._prev_error = error
        out = self.kp * error + self.ki * self._integral + self.kd * deriv
        return max(self.out_min, min(self.out_max, out))


class PolyPlannerController:
    """LQR+FF+PID waypoint controller matching C++ poly_planner_onsite.

    Converts ego-frame waypoints to CARLA VehicleControl using the same
    FF+LQR lateral controller and PID longitudinal controller as the C++
    expert planner.
    """

    def __init__(self, vehicle, target_speed=DEFAULT_TARGET_SPEED):
        self._vehicle = vehicle
        self._target_speed = float(target_speed)

        self._speed_pid = PIDController(
            kp=PID_P, ki=PID_I, kd=PID_D, dt=PID_DT,
            out_min=A_MIN, out_max=A_MAX,
        )
        self._low_speed_counter = 0
        self.debug = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_target_speed(self, speed_ms):
        self._target_speed = float(speed_ms)

    def reset(self):
        self._speed_pid.reset()
        self._low_speed_counter = 0
        for attr in ("_target_speed_prev", "_steer_prev_deg", "_last_good_waypoints"):
            if hasattr(self, attr):
                delattr(self, attr)

    def run_step(self, waypoints_ego, waypoint_scale=30.0):
        """Convert network/policy waypoints to CARLA VehicleControl.

        Args:
            waypoints_ego:  Shape (16,) or (8, 2), normalized to [-1, 1].
            waypoint_scale: Meters per normalized unit.

        Returns:
            carla.VehicleControl
        """
        if carla is None:
            raise RuntimeError(
                "PolyPlannerController requires CARLA at runtime."
            )

        # ── 1. Decode & sanitize waypoints ──
        pts_raw = self._decode_waypoints(waypoints_ego, waypoint_scale)
        pts_clean = self._sanitize_waypoints(pts_raw)
        pts_with_origin = np.vstack(
            [np.zeros((1, 2), dtype=np.float64), pts_clean]
        )

        # ── 2. Dense interpolation (Catmull-Rom → 41 pts) ──
        pts_dense = self._catmull_rom_dense(pts_with_origin, NUM_DENSE)

        # ── 3. Ego state ──
        vx = self._get_speed_ms()
        yawrate_deg = self._vehicle.get_angular_velocity().z
        yawrate = math.radians(yawrate_deg)

        # ── 4. Compute errors at C++ preview points ──
        preview_idx = [2, 4, 6, 8, 10]
        preview_idx = [i for i in preview_idx if i < len(pts_dense)]
        edL, dedL, ephi0, dephi0, rowL = self._compute_weighted_errors(
            pts_dense, preview_idx, vx, yawrate,
        )

        # ── 5. Lateral: FF + LQR ──
        steer_ff_rad = self._control_ff(vx, rowL)
        steer_fb_deg = self._control_lqr(edL, dedL, ephi0, dephi0, LQR_K)
        steer_fb_rad = math.radians(steer_fb_deg)
        steer_total_rad = K_FF * steer_ff_rad + K_FB * steer_fb_rad
        steer_total_deg = math.degrees(steer_total_rad)
        steer_total_deg = float(np.clip(steer_total_deg, -STEER_MAX_DEG, STEER_MAX_DEG))

        # Low-pass filter (C++ doesn't have this, but suppresses waypoint noise)
        if not hasattr(self, "_steer_prev_deg"):
            self._steer_prev_deg = steer_total_deg
        steer_total_deg = (
            STEER_FILTER_ALPHA * steer_total_deg
            + (1.0 - STEER_FILTER_ALPHA) * self._steer_prev_deg
        )
        self._steer_prev_deg = steer_total_deg

        # ── 6. Longitudinal: PID speed tracking ──
        target_speed = self._estimate_target_speed(pts_clean)
        a_des = self._speed_pid.step(target_speed, vx)

        if vx < 1.0:
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
        steer_carla = -steer_total_deg / STEER_ACTION_NORM_DEG

        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.steer = float(np.clip(steer_carla, -1.0, 1.0))
        control.brake = float(brake)
        control.hand_brake = False
        control.manual_gear_shift = False

        self.debug = {
            "current_speed": float(vx),
            "target_speed": float(target_speed),
            "accel_des": float(a_des),
            "steer_ff_deg": float(math.degrees(steer_ff_rad)),
            "steer_fb_deg": float(steer_fb_deg),
            "steer_total_deg": float(steer_total_deg),
            "edL": float(edL),
            "ephi0": float(ephi0),
            "rowL": float(rowL),
        }
        return control

    # ------------------------------------------------------------------
    # Waypoint helpers
    # ------------------------------------------------------------------

    def _decode_waypoints(self, waypoints_ego, scale):
        arr = np.asarray(waypoints_ego, dtype=np.float64).reshape(-1)
        num = len(arr) // 2
        arr = arr[: num * 2]
        pts = arr.reshape(num, 2) * float(scale)
        if not np.isfinite(pts).all():
            return self._fallback_waypoints(num)
        return pts

    def _fallback_waypoints(self, num=8):
        speed = float(np.clip(self._target_speed, 0.0, TARGET_SPEED_MAX))
        xs = (np.arange(num, dtype=np.float64) + 1.0) * WAYPOINT_DT * speed
        return np.stack([xs, np.zeros(num, dtype=np.float64)], axis=-1)

    def _sanitize_waypoints(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        n = pts.shape[0]
        speed_for_bounds = max(0.5, self._target_speed)
        max_step = max(1.5, speed_for_bounds * WAYPOINT_DT * 2.4)

        cleaned = np.zeros_like(pts)
        prev = np.zeros(2, dtype=np.float64)
        last_good_step = np.array(
            [speed_for_bounds * WAYPOINT_DT, 0.0], dtype=np.float64
        )
        for i in range(n):
            p = pts[i].copy()
            # Basic validity
            if not np.isfinite(p).all() or p[0] < -MAX_BACKWARD_X:
                p = prev + last_good_step
            p[1] = float(np.clip(p[1], -MAX_LATERAL_ABS, MAX_LATERAL_ABS))
            step = p - prev
            step_norm = float(np.linalg.norm(step))
            if step_norm > max_step:
                p = prev + step / max(step_norm, 1e-6) * max_step
                step = p - prev
                step_norm = float(np.linalg.norm(step))
            if p[0] < prev[0] - 0.5:
                p[0] = prev[0] - 0.5
            cleaned[i] = p
            prev = p
            if step_norm > 0.05:
                last_good_step = step
            else:
                last_good_step = 0.5 * last_good_step
        return cleaned

    # ------------------------------------------------------------------
    # Catmull-Rom interpolation (matches JAX version)
    # ------------------------------------------------------------------

    @staticmethod
    def _catmull_rom_dense(pts, num_out=NUM_DENSE):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        n = pts.shape[0]
        if n < 2:
            return np.tile(pts[0] if n == 1 else [0, 0], (num_out, 1))
        if n == 2:
            u = np.linspace(0.0, 1.0, num_out)
            return pts[0] + u[:, None] * (pts[1] - pts[0])

        t = np.arange(n, dtype=np.float64) * WAYPOINT_DT
        tang = np.zeros_like(pts)
        tang[0] = (pts[1] - pts[0]) / WAYPOINT_DT
        tang[-1] = (pts[-1] - pts[-2]) / WAYPOINT_DT
        tang[1:-1] = (pts[2:] - pts[:-2]) / (2.0 * WAYPOINT_DT)

        # Tangent magnitude limiting
        for i in range(n):
            if i == 0:
                local_step = np.linalg.norm(pts[1] - pts[0]) / WAYPOINT_DT
            elif i == n - 1:
                local_step = np.linalg.norm(pts[-1] - pts[-2]) / WAYPOINT_DT
            else:
                s0 = np.linalg.norm(pts[i] - pts[i - 1]) / WAYPOINT_DT
                s1 = np.linalg.norm(pts[i + 1] - pts[i]) / WAYPOINT_DT
                local_step = max(s0, s1)
            max_t = max(0.5, 2.5 * local_step)
            tn = float(np.linalg.norm(tang[i]))
            if tn > max_t:
                tang[i] *= max_t / max(tn, 1e-6)

        t_new = np.linspace(t[0], t[-1], num_out)
        out = np.zeros((num_out, 2), dtype=np.float64)
        for j, tj in enumerate(t_new):
            idx = int(np.searchsorted(t, tj, side="right") - 1)
            idx = max(0, min(idx, n - 2))
            dt_seg = t[idx + 1] - t[idx]
            u = (tj - t[idx]) / max(dt_seg, 1e-6)
            u2, u3 = u * u, u2 * u
            h00 = 2.0 * u3 - 3.0 * u2 + 1.0
            h10 = (u3 - 2.0 * u2 + u) * dt_seg
            h01 = -2.0 * u3 + 3.0 * u2
            h11 = (u3 - u2) * dt_seg
            out[j] = (
                h00 * pts[idx]
                + h10 * tang[idx]
                + h01 * pts[idx + 1]
                + h11 * tang[idx + 1]
            )
        out[0] = np.array([0.0, 0.0], dtype=np.float64)
        return out

    # ------------------------------------------------------------------
    # Error computation — matches C++ ErrCalcuModule
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_weighted_errors(pts_dense, preview_idx, vx, yawrate):
        """Compute weighted lateral/heading errors and curvature.

        Matches poly_planner_onsite.hpp lines 622-636.
        """
        ed_vals, ded_vals = [], []
        ephi_vals, dephi_vals, row_vals = [], [], []

        for idx in preview_idx:
            if idx >= len(pts_dense):
                continue
            px, py = pts_dense[idx]
            # Next point for heading
            next_i = min(idx + 1, len(pts_dense) - 1)
            nx, ny = pts_dense[next_i]
            ref_h = math.atan2(ny - py, nx - px)

            # ephi = reference heading (vehicle heading = 0 in ego frame)
            ephi = ref_h
            sin_h, cos_h = math.sin(ref_h), math.cos(ref_h)

            # ed = -px*sin(ref_h) + py*cos(ref_h)
            ed = -px * sin_h + py * cos_h

            # ded = vx * sin(ephi)
            ded = vx * math.sin(ephi)

            # Curvature (3-point Menger)
            kappa = PolyPlannerController._curvature_at(pts_dense, idx)

            # dephi = vx * kappa - yawrate
            dephi = vx * kappa - yawrate

            ed_vals.append(ed)
            ded_vals.append(ded)
            ephi_vals.append(ephi)
            dephi_vals.append(dephi)
            row_vals.append(kappa)

        # Fallback
        if not ed_vals:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Weighted sum (G0..G4)
        w = PREVIEW_WEIGHTS[:len(ed_vals)]
        w_sum = sum(w)
        w = [g / w_sum for g in w]

        edL = sum(g * v for g, v in zip(w, ed_vals))
        dedL = sum(g * v for g, v in zip(w, ded_vals))
        ephi0 = ephi_vals[0]
        dephi0 = dephi_vals[0]
        rowL = sum(g * v for g, v in zip(w, row_vals))

        return edL, dedL, ephi0, dephi0, rowL

    @staticmethod
    def _curvature_at(pts, idx):
        """3-point Menger curvature at pts[idx]."""
        n = len(pts)
        i0 = max(idx - 1, 0)
        i1 = idx
        i2 = min(idx + 1, n - 1)
        if i2 - i0 < 2:
            return 0.0
        p0, p1, p2 = pts[i0], pts[i1], pts[i2]
        a = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        b = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        c = math.hypot(p2[0] - p0[0], p2[1] - p0[1])
        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            return 0.0
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - \
                (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(cross) < 1e-12:
            return 0.0
        return 4.0 * cross / (a * b * c)

    # ------------------------------------------------------------------
    # Lateral: FF + LQR — matches C++ lines 1674-1712
    # ------------------------------------------------------------------

    @staticmethod
    def _control_ff(vx, curvature):
        """Feedforward steer angle [rad] from desired curvature (bicycle model).

        Matches poly_planner_onsite.hpp lines 1674-1678.
        """
        if vx < 1e-2:
            return 0.0
        vx2 = vx * vx
        num = (VP.Cf * VP.Cr * (VP.lf + VP.lr) ** 2
               + (VP.Cr * VP.lr - VP.Cf * VP.lf) * VP.m * vx2)
        den = VP.Cf * VP.Cr * (VP.lf + VP.lr)
        if abs(den) < 1e-12:
            return 0.0
        return curvature * num / den

    @staticmethod
    def _control_lqr(ed, ded, ephi, dephi, K):
        """LQR feedback steer angle [deg].

        Matches poly_planner_onsite.hpp lines 1708-1712.
          steer_deg = -(ephi*K[0] + dephi*K[1]*10.0 + ed*K[2] + ded*K[3])
        """
        return -(ephi * K[0] + dephi * K[1] * 10.0 + ed * K[2] + ded * K[3])

    # ------------------------------------------------------------------
    # Longitudinal: PID speed tracking
    # ------------------------------------------------------------------

    def _estimate_target_speed(self, pts_clean):
        """Target speed from terminal waypoint spacing (matching trajF2C.v[40])."""
        pts = np.asarray(pts_clean, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 1 or not np.isfinite(pts).all():
            return float(np.clip(self._target_speed, 0.0, TARGET_SPEED_MAX))

        pts_time = np.vstack([np.zeros((1, 2), dtype=np.float64), pts])
        dists = np.linalg.norm(np.diff(pts_time, axis=0), axis=1)
        speeds = np.clip(dists / WAYPOINT_DT, 0.0, TARGET_SPEED_MAX)

        target = float(speeds[-1])
        if len(speeds) >= 2:
            target = 0.8 * float(speeds[-1]) + 0.2 * float(speeds[-2])

        target = float(np.clip(target, 0.0, TARGET_SPEED_MAX))

        # Low-pass filter
        if not hasattr(self, "_target_speed_prev"):
            self._target_speed_prev = target
        target = (
            TARGET_SPEED_FILTER_ALPHA * target
            + (1.0 - TARGET_SPEED_FILTER_ALPHA) * self._target_speed_prev
        )
        self._target_speed_prev = target
        return float(np.clip(target, 0.0, TARGET_SPEED_MAX))

    @staticmethod
    def _accel_to_control(a_des):
        """Desired acceleration → throttle [0-1] / brake [0-1]."""
        if a_des >= 0.0:
            return min(a_des / A_MAX, 1.0), 0.0
        return 0.0, min(abs(a_des) / abs(A_MIN), 1.0)

    def _get_speed_ms(self):
        vel = self._vehicle.get_velocity()
        return float(math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))
