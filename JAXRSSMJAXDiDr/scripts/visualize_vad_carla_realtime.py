"""Visualize live VAD perception outputs on CARLA six-camera observations."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import sys
import time
from typing import Dict

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from JAXRSSMJAXDiDr.scripts.export_vad_latents import (
    DEFAULT_VAD_LIDAR_X,
    DEFAULT_VAD_LIDAR_Y,
    DEFAULT_VAD_LIDAR_Z,
    build_vad,
    make_can_bus,
    make_frame_img_meta,
    make_img_meta,
    normalize_and_resize,
    normalize_vad_runtime_args,
)
from JAXRSSMJAXDiDr.vad_carla import install_vad_carla_patches
from JAXRSSMJAXDiDr.vad_carla.camera_setup import get_camera_keys, get_camera_order


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="carla_roundabout")
    parser.add_argument("--vad_runtime", choices=("official", "b2d"), default="b2d")
    parser.add_argument("--vad_root", default=str(ROOT / "VAD"))
    parser.add_argument("--vad_model", choices=("tiny", "base"), default="tiny")
    parser.add_argument("--vad_checkpoint", required=True)
    parser.add_argument("--checkpoint_strict", action="store_true")
    parser.add_argument(
        "--b2d_root",
        default=str(ROOT / "Bench2DriveZoo-uniad-vad" / "Bench2DriveZoo-uniad-vad"),
    )
    parser.add_argument("--b2d_config", default="adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--camera_width", type=int, default=1600)
    parser.add_argument("--camera_height", type=int, default=900)
    parser.add_argument("--camera_fov", type=float, default=70.0)
    parser.add_argument("--camera_profile", choices=("nusc", "b2d"), default="b2d")
    parser.add_argument("--vad_lidar_x", type=float, default=DEFAULT_VAD_LIDAR_X)
    parser.add_argument("--vad_lidar_y", type=float, default=DEFAULT_VAD_LIDAR_Y)
    parser.add_argument("--vad_lidar_z", type=float, default=DEFAULT_VAD_LIDAR_Z)
    parser.add_argument("--auto_calibrate_vad", action="store_true")
    parser.add_argument("--calib_lidar_x_values", default="0.0,0.4,0.8,1.0,1.2")
    parser.add_argument("--calib_lidar_z_values", default="1.8,2.0,2.3,2.5,2.7")
    parser.add_argument("--sensor_tick", type=float, default=0.1)
    parser.add_argument(
        "--pretrained_norm",
        action="store_true",
        default=False,
        help="Use the legacy BGR/std=1 normalization path instead of cfg.img_norm_cfg.",
    )
    parser.add_argument("--no_pretrained_norm", dest="pretrained_norm", action="store_false")
    parser.add_argument("--score_thresh", type=float, default=0.35)
    parser.add_argument("--map_score_thresh", type=float, default=0.35)
    parser.add_argument("--bev_size", type=int, default=640)
    parser.add_argument("--front_width", type=int, default=640)
    parser.add_argument("--surround_width", type=int, default=960)
    parser.add_argument("--vad_every", type=int, default=1)
    parser.add_argument("--no_temporal_bev", action="store_true")
    parser.add_argument("--vad_debug", action="store_true")
    parser.add_argument("--vad_debug_every", type=int, default=50)
    parser.add_argument("--vad_debug_overlay", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0, help="0 means run until q/Esc.")
    parser.add_argument("--action_acc", type=float, default=0.0)
    parser.add_argument("--action_steer", type=float, default=0.0)
    parser.add_argument("--manual_action", action="store_true")
    parser.add_argument("--csv_control", action="store_true", help="Use CARLA BasicAgent to follow a CSV route.")
    parser.add_argument("--route_csv", default="", help="Route CSV with x,y columns. Defaults to task env.route_csv.")
    parser.add_argument("--csv_x_col", default="x")
    parser.add_argument("--csv_y_col", default="y")
    parser.add_argument("--csv_target_speed", type=float, default=20.0, help="BasicAgent target speed in km/h.")
    parser.add_argument("--csv_min_waypoint_dist", type=float, default=0.5)
    parser.add_argument("--csv_respect_traffic_lights", action="store_true")
    parser.add_argument("--csv_respect_vehicles", action="store_true")
    parser.add_argument("--csv_draw_route", action="store_true")
    parser.add_argument("--cmd_timeout_s", type=float, default=0.25)
    parser.add_argument("--spin_timeout_sec", type=float, default=0.0)
    parser.add_argument("--steer_max_deg", type=float, default=120.0)
    parser.add_argument("--no_brake_pressure", action="store_true")
    parser.add_argument("--obs_topic", default="/obs_info")
    parser.add_argument("--ctrl_topic", default="/ctrl_info")
    parser.add_argument("--global_topic", default="/global_info")
    parser.add_argument("--reset_topic", default="/reset_info")
    parser.add_argument("--traj_topic", default="/traj_best_vis")
    parser.add_argument("--window", default="VAD CARLA realtime")
    args, rest = parser.parse_known_args(argv)
    return args, rest


def load_collect_polyplanner_runtime():
    """Load dreamerv3/collect_polyplanner.py without importing dreamerv3."""

    dreamerv3_dir = ROOT / "dreamerv3"
    if str(dreamerv3_dir) not in sys.path:
        sys.path.insert(0, str(dreamerv3_dir))

    module_name = "_dreamerv3_collect_polyplanner"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = dreamerv3_dir / "collect_polyplanner.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load collect_polyplanner from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def init_ros2_expert(args):
    cp = load_collect_polyplanner_runtime()
    import rclpy

    if not rclpy.ok():
        rclpy.init(args=None)

    bridge = cp.CarlaRos2ExpertBridge(
        obs_topic=args.obs_topic,
        ctrl_topic=args.ctrl_topic,
        global_topic=args.global_topic,
        reset_topic=args.reset_topic,
        traj_topic=args.traj_topic,
        cmd_timeout_s=args.cmd_timeout_s,
    )
    speed_pid = cp.PIDController(
        kp=1.0,
        ki=0.0,
        kd=0.0,
        dt=0.1,
        output_limits=(cp.ENV_ACC_MIN, cp.ENV_ACC_MAX),
        integrator_limits=(-10.0, 10.0),
    )
    return cp, rclpy, bridge, speed_pid


def reset_ros2_expert(bridge, speed_pid):
    speed_pid.reset()
    bridge.send_reset()


def import_box_type(args):
    if getattr(args, "vad_runtime", "official") == "b2d":
        from mmcv.core.bbox import get_box_type

        box_type_3d, _ = get_box_type("LiDAR")
        return box_type_3d
    try:
        from mmdet3d.core import LiDARInstance3DBoxes
    except Exception:
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
    return LiDARInstance3DBoxes


def stack_vad_images(obs: Dict[str, np.ndarray], args) -> np.ndarray:
    camera_keys = get_camera_keys(args.camera_profile)
    missing = [key for key in camera_keys if key not in obs]
    if missing:
        raise KeyError(f"Observation is missing VAD camera keys: {missing}")
    return np.stack([np.asarray(obs[key]) for key in camera_keys], axis=0)


def ego_pose_from_env(env) -> dict:
    ego = env.get_ego_vehicle() if hasattr(env, "get_ego_vehicle") else env.ego
    tf = ego.get_transform()
    vel = ego.get_velocity()
    acc = ego.get_acceleration()
    ang = ego.get_angular_velocity()
    # CARLA/CarDreamer world: x-forward, y-right. VAD/nuScenes BEV:
    # x-forward, y-left. Convert y, yaw and yaw-rate before building can_bus.
    yaw = -np.radians(float(tf.rotation.yaw))
    speed = float(np.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))
    return {
        "x": float(tf.location.x),
        "y": -float(tf.location.y),
        "yaw": float(yaw),
        "patch_angle": float(np.degrees(yaw) % 360.0),
        "speed": speed,
        "vx": float(vel.x),
        "vy": -float(vel.y),
        "acceleration": [float(acc.x), -float(acc.y), float(acc.z)],
        "angular_velocity": [
            -float(np.radians(ang.x)),
            float(np.radians(ang.y)),
            -float(np.radians(ang.z)),
        ],
        "yawrate": -float(np.radians(ang.z)),
    }


def to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def run_vad_frame(torch, model, images, meta, args, cfg, prev_bev):
    img_np = normalize_and_resize(images, args, cfg)
    return run_vad_tensor(torch, model, img_np, meta, args, prev_bev)


def run_vad_tensor(torch, model, img_np, meta, args, prev_bev):
    img = torch.from_numpy(img_np[None]).to(args.device)
    with torch.no_grad():
        feats = model.extract_feat(img=img, img_metas=[meta])
        outs = model.pts_bbox_head(feats, [meta], prev_bev=prev_bev)
        bbox_list = model.pts_bbox_head.get_bboxes(outs, [meta], rescale=False)
    result = decode_bbox_list(bbox_list[0], outs)
    return result["bev_embed"], result


def decode_bbox_list(item, outs):
    if len(item) == 8:
        boxes, scores, labels, trajs, map_bboxes, map_scores, map_labels, map_pts = item
        trajs_cls = None
    elif len(item) >= 9:
        boxes, scores, labels, trajs, trajs_cls, map_bboxes, map_scores, map_labels, map_pts = item[:9]
    else:
        raise ValueError(f"Unexpected VAD bbox output length: {len(item)}")
    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "trajs": trajs,
        "trajs_cls": trajs_cls,
        "map_bboxes": map_bboxes,
        "map_scores": map_scores,
        "map_labels": map_labels,
        "map_pts": map_pts,
        # Perception latent for downstream planners. VAD returns [HW, B, D].
        "bev_embed": outs.get("bev_embed"),
    }


def point_to_bev(point, pc_range, size):
    x_min, y_min = float(pc_range[0]), float(pc_range[1])
    x_max, y_max = float(pc_range[3]), float(pc_range[4])
    x, y = float(point[0]), float(point[1])
    # x-forward -> image upward; y-left -> image leftward.
    px = int((y_max - y) / max(y_max - y_min, 1e-6) * (size - 1))
    py = int((x_max - x) / max(x_max - x_min, 1e-6) * (size - 1))
    return px, py


def draw_polyline(canvas, pts, pc_range, color, thickness=1, closed=False):
    import cv2

    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return
    pixels = [point_to_bev(pt, pc_range, canvas.shape[0]) for pt in pts]
    for p0, p1 in zip(pixels[:-1], pixels[1:]):
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
    if closed:
        cv2.line(canvas, pixels[-1], pixels[0], color, thickness, cv2.LINE_AA)


def box_corners_bev(box):
    x, y = float(box[0]), float(box[1])
    dx = max(float(box[3]), 0.1)
    dy = max(float(box[4]), 0.1)
    yaw = float(box[6]) if len(box) > 6 else 0.0
    local = np.asarray(
        [[dx / 2, dy / 2], [dx / 2, -dy / 2], [-dx / 2, -dy / 2], [-dx / 2, dy / 2]],
        dtype=np.float32,
    )
    rot = np.asarray(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float32,
    )
    return local @ rot.T + np.asarray([x, y], dtype=np.float32)


def draw_agent_trajs(canvas, center, trajs, pc_range, color):
    trajs = np.asarray(trajs, dtype=np.float32)
    if trajs.size == 0:
        return

    # [12] -> [1, 6, 2]
    if trajs.ndim == 1:
        trajs = trajs.reshape(1, -1, 2)

    # [6, 12] -> [6, 6, 2]; [T, 2] -> [1, T, 2]
    elif trajs.ndim == 2:
        if trajs.shape[-1] == 2:
            trajs = trajs[None]
        else:
            trajs = trajs.reshape(trajs.shape[0], -1, 2)
    elif trajs.ndim != 3:
        return

    for mode in trajs[:3]:
        pts = np.cumsum(mode, axis=0)
        pts += np.asarray(center[:2], dtype=np.float32)
        draw_polyline(canvas, pts, pc_range, color, thickness=1)


def render_bev(result, cfg, args):
    import cv2

    size = int(args.bev_size)
    pc_range = np.asarray(getattr(cfg, "point_cloud_range", [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]))
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)

    for meters in range(-30, 31, 10):
        p0 = point_to_bev((pc_range[0], meters), pc_range, size)
        p1 = point_to_bev((pc_range[3], meters), pc_range, size)
        cv2.line(canvas, p0, p1, (225, 225, 225), 1)
    for meters in range(-10, 16, 5):
        p0 = point_to_bev((meters, pc_range[1]), pc_range, size)
        p1 = point_to_bev((meters, pc_range[4]), pc_range, size)
        cv2.line(canvas, p0, p1, (225, 225, 225), 1)

    if result is not None:
        map_pts = to_numpy(result["map_pts"])
        map_scores = to_numpy(result["map_scores"])
        if map_pts is not None and map_scores is not None:
            for pts, score in zip(map_pts, map_scores):
                if float(score) >= args.map_score_thresh:
                    draw_polyline(canvas, pts, pc_range, (80, 170, 80), thickness=1)

        boxes = to_numpy(result["boxes"])
        scores = to_numpy(result["scores"])
        labels = to_numpy(result["labels"])
        trajs = to_numpy(result["trajs"])
        if boxes is not None and scores is not None:
            for idx, (box, score) in enumerate(zip(boxes, scores)):
                if float(score) < args.score_thresh:
                    continue
                corners = box_corners_bev(box)
                draw_polyline(canvas, corners, pc_range, (30, 80, 220), thickness=2, closed=True)
                center_px = point_to_bev(box[:2], pc_range, size)
                label = int(labels[idx]) if labels is not None and idx < len(labels) else -1
                cv2.putText(
                    canvas,
                    f"{label}:{float(score):.2f}",
                    (center_px[0] + 3, center_px[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )
                if trajs is not None and idx < len(trajs):
                    draw_agent_trajs(canvas, box[:2], trajs[idx], pc_range, (220, 120, 20))

    ego_px = point_to_bev((0.0, 0.0), pc_range, size)
    cv2.circle(canvas, ego_px, 5, (0, 0, 0), -1)
    cv2.putText(canvas, "BEV x-forward y-left", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    return canvas


def vad_image_scale(args):
    if getattr(args, "vad_runtime", "official") == "b2d":
        return 0.8
    return 0.4 if args.vad_model == "tiny" else 0.8


def vad_ground_z(args):
    return -float(getattr(args, "vad_lidar_z", DEFAULT_VAD_LIDAR_Z))


def project_debug_grid(meta, cfg, args, cam_idx, nx=9, ny=17):
    pc_range = np.asarray(getattr(cfg, "point_cloud_range", [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]), dtype=np.float32)
    xs = np.linspace(float(pc_range[0]), float(pc_range[3]), nx, dtype=np.float32)
    ys = np.linspace(float(pc_range[1]), float(pc_range[4]), ny, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full(xx.size, vad_ground_z(args), dtype=np.float32)
    pts = np.stack(
        [xx.reshape(-1), yy.reshape(-1), zz, np.ones(xx.size, dtype=np.float32)],
        axis=0,
    )
    lidar2img = np.asarray(meta["lidar2img"][cam_idx], dtype=np.float32)
    proj = lidar2img @ pts
    depth = proj[2]
    uv = proj[:2] / np.maximum(depth, 1e-6)
    scale = vad_image_scale(args)
    scaled_w = int(args.camera_width * scale)
    scaled_h = int(args.camera_height * scale)
    valid = (
        (depth > 1e-3)
        & (uv[0] >= 0.0)
        & (uv[0] < float(scaled_w))
        & (uv[1] >= 0.0)
        & (uv[1] < float(scaled_h))
    )
    return uv[:, valid].T, int(valid.sum()), int(valid.size), (scaled_w, scaled_h)


def projection_coverage_summary(meta, cfg, args):
    items = []
    for cam_idx, cam_name in enumerate(get_camera_order(args.camera_profile)):
        _, visible, total, _ = project_debug_grid(meta, cfg, args, cam_idx)
        items.append(f"{cam_name}:{visible}/{total}")
    return " ".join(items)


def print_vad_debug(step, obs, img_np, meta, result, cfg, args, pose, prev_pose):
    if not args.vad_debug:
        return
    interval = max(int(args.vad_debug_every), 1)
    if step != 0 and step % interval != 0:
        return

    raw_shape = np.asarray(obs["camera_front"]).shape
    can_bus = np.asarray(meta["can_bus"], dtype=np.float32)
    front_raw = np.asarray(obs["camera_front"], dtype=np.float32)
    boxes = to_numpy(result["boxes"]) if result is not None else None
    scores = to_numpy(result["scores"]) if result is not None else None
    map_scores = to_numpy(result["map_scores"]) if result is not None else None
    bev_embed = result.get("bev_embed") if result is not None else None
    bev_shape = tuple(bev_embed.shape) if bev_embed is not None and hasattr(bev_embed, "shape") else None
    norm_mode = "legacy_bgr_std1" if args.pretrained_norm else "config_img_norm_cfg"
    det_count = int(np.sum(scores >= args.score_thresh)) if scores is not None else 0
    map_count = int(np.sum(map_scores >= args.map_score_thresh)) if map_scores is not None else 0
    total_boxes = int(len(boxes)) if boxes is not None else 0
    total_maps = int(len(map_scores)) if map_scores is not None else 0
    top_det_scores = []
    top_map_scores = []
    if scores is not None and len(scores):
        top_det_scores = np.sort(np.asarray(scores, dtype=np.float32).reshape(-1))[-5:][::-1].round(3).tolist()
    if map_scores is not None and len(map_scores):
        top_map_scores = np.sort(np.asarray(map_scores, dtype=np.float32).reshape(-1))[-5:][::-1].round(3).tolist()
    prev_flag = prev_pose is not None and not args.no_temporal_bev
    print(
        "[VAD Debug] "
        f"step={step} model={args.vad_model} prev_bev={prev_flag} "
        f"profile={args.camera_profile} norm={norm_mode} bev_shape={bev_shape} "
        f"vad_lidar=({args.vad_lidar_x:.2f},{args.vad_lidar_y:.2f},{args.vad_lidar_z:.2f}) "
        f"raw_front_shape={tuple(raw_shape)} tensor_shape={tuple(img_np.shape)} "
        f"meta_ori={meta['ori_shape'][0]} meta_img={meta['img_shape'][0]} meta_pad={meta['pad_shape'][0]}",
        flush=True,
    )
    print(
        "[VAD Debug] "
        f"front_rgb_mean={front_raw.mean(axis=(0, 1)).round(1).tolist()} "
        f"tensor_mean={float(img_np.mean()):.2f} tensor_std={float(img_np.std()):.2f} "
        f"can_bus_xy=({can_bus[0]:.3f},{can_bus[1]:.3f}) "
        f"speed={can_bus[7]:.3f} ang_vel={can_bus[13:16].round(3).tolist()} "
        f"yaw_rad={can_bus[-2]:.3f} delta_yaw_deg={can_bus[-1]:.3f} "
        f"ego=({pose['x']:.2f},{pose['y']:.2f},{np.degrees(pose['yaw']):.1f}deg)",
        flush=True,
    )
    print(
        "[VAD Debug] "
        f"projection_grid={projection_coverage_summary(meta, cfg, args)} "
        f"dets={det_count}/{total_boxes}@{args.score_thresh:.2f} "
        f"maps={map_count}/{total_maps}@{args.map_score_thresh:.2f} "
        f"top_det={top_det_scores} top_map={top_map_scores}",
        flush=True,
    )


def parse_float_list(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f"Expected at least one float value, got {text!r}")
    return values


def mean_top_scores(values, k):
    if values is None:
        return 0.0
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    arr = np.sort(arr)[-min(int(k), arr.size):]
    return float(arr.mean())


def vad_result_confidence(result):
    if result is None:
        return 0.0, 0.0, 0.0
    scores = to_numpy(result["scores"])
    map_scores = to_numpy(result["map_scores"])
    det_score = mean_top_scores(scores, 5)
    map_score = mean_top_scores(map_scores, 5)
    det_bonus = 0.0 if scores is None else 0.01 * float(np.sum(np.asarray(scores) >= 0.15))
    map_bonus = 0.0 if map_scores is None else 0.005 * float(np.sum(np.asarray(map_scores) >= 0.15))
    total = 0.7 * det_score + 0.3 * map_score + det_bonus + map_bonus
    return float(total), float(det_score), float(map_score)


def clone_args_with_lidar(args, lidar_x, lidar_z):
    clone = argparse.Namespace(**vars(args))
    clone.vad_lidar_x = float(lidar_x)
    clone.vad_lidar_z = float(lidar_z)
    return clone


def auto_calibrate_vad_geometry(torch, model, img_np, pose, args, cfg, box_type_3d):
    x_values = parse_float_list(args.calib_lidar_x_values)
    z_values = parse_float_list(args.calib_lidar_z_values)
    trials = []
    best = None
    print(
        "[VAD Calib] sweeping pseudo-lidar origin "
        f"x={x_values} z={z_values}",
        flush=True,
    )
    for lidar_x in x_values:
        for lidar_z in z_values:
            trial_args = clone_args_with_lidar(args, lidar_x, lidar_z)
            trial_meta = make_img_meta(trial_args, cfg)
            trial_meta["box_type_3d"] = box_type_3d
            can_bus = make_can_bus(pose, None)
            meta = make_frame_img_meta(trial_meta, can_bus, Path(args.task), 0, False)
            try:
                trial_prev_bev, result = run_vad_tensor(torch, model, img_np, meta, trial_args, None)
                score, det_score, map_score = vad_result_confidence(result)
            except Exception as exc:
                print(f"[VAD Calib] x={lidar_x:.2f} z={lidar_z:.2f} failed: {exc}", flush=True)
                continue
            trials.append((score, det_score, map_score, lidar_x, lidar_z, trial_meta, trial_prev_bev, result, meta))
            print(
                "[VAD Calib] "
                f"x={lidar_x:.2f} z={lidar_z:.2f} score={score:.3f} "
                f"det_top={det_score:.3f} map_top={map_score:.3f}",
                flush=True,
            )
            if best is None or score > best[0]:
                best = trials[-1]

    if best is None:
        raise RuntimeError("VAD auto calibration failed for all candidates.")

    score, det_score, map_score, lidar_x, lidar_z, base_meta, prev_bev, result, meta = best
    args.vad_lidar_x = float(lidar_x)
    args.vad_lidar_z = float(lidar_z)
    print(
        "[VAD Calib] selected "
        f"x={lidar_x:.2f} z={lidar_z:.2f} score={score:.3f} "
        f"det_top={det_score:.3f} map_top={map_score:.3f}",
        flush=True,
    )
    return base_meta, prev_bev, result, meta


SURROUND_LAYOUT = (
    ("camera_front_left", "FRONT LEFT"),
    ("camera_front", "FRONT"),
    ("camera_front_right", "FRONT RIGHT"),
    ("camera_back_left", "BACK LEFT"),
    ("camera_back", "BACK"),
    ("camera_back_right", "BACK RIGHT"),
)


def _render_camera_tile(obs, key, label, tile_size, args=None, meta=None, cfg=None, cam_idx=None):
    import cv2

    if key not in obs:
        raise KeyError(f"Observation is missing camera key: {key}")
    tile_w, tile_h = tile_size
    image = np.asarray(obs[key])
    image = cv2.resize(image, (tile_w, tile_h))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if args is not None and meta is not None and cfg is not None and cam_idx is not None and args.vad_debug_overlay:
        points, _, _, scaled_shape = project_debug_grid(meta, cfg, args, cam_idx)
        scaled_w, scaled_h = scaled_shape
        for u, v in points:
            px = int(round(float(u) * tile_w / max(scaled_w, 1)))
            py = int(round(float(v) * tile_h / max(scaled_h, 1)))
            if 0 <= px < tile_w and 0 <= py < tile_h:
                cv2.circle(image, (px, py), 2, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (tile_w, 24), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def render_surround(obs, args, meta=None, cfg=None):
    import cv2

    camera_keys = get_camera_keys(args.camera_profile)
    panel_w = max(3, int(getattr(args, "surround_width", 0) or args.front_width))
    tile_w = max(1, panel_w // 3)
    front = np.asarray(obs["camera_front"])
    h, w = front.shape[:2]
    tile_h = max(1, int(h * tile_w / max(w, 1)))
    tiles = [
        _render_camera_tile(
            obs,
            key,
            label,
            (tile_w, tile_h),
            args=args,
            meta=meta,
            cfg=cfg,
            cam_idx=camera_keys.index(key) if key in camera_keys else None,
        )
        for key, label in SURROUND_LAYOUT
    ]
    top = np.concatenate(tiles[:3], axis=1)
    bottom = np.concatenate(tiles[3:], axis=1)
    return np.concatenate([top, bottom], axis=0)


def make_visual(surround, bev):
    import cv2

    if bev.shape[0] != surround.shape[0]:
        target_h = surround.shape[0]
        target_w = max(1, int(bev.shape[1] * target_h / max(bev.shape[0], 1)))
        bev = cv2.resize(bev, (target_w, target_h))
    return np.concatenate([surround, bev], axis=1)


def make_env_action(env, args):
    space = env.action_space
    if hasattr(space, "n"):
        return 0
    shape = tuple(getattr(space, "shape", ()) or ())
    action = np.zeros(shape, dtype=np.float32)
    if action.size >= 2:
        action = action.reshape(-1)
        action[0] = float(args.action_acc)
        action[1] = float(args.action_steer)
        return action.reshape(shape).astype(np.float32)
    return action


def get_config_value(config, key, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except Exception:
            pass
    return getattr(config, key, default)


def resolve_route_csv(env, args):
    route_csv = args.route_csv or get_config_value(getattr(env, "_config", None), "route_csv", "")
    if not route_csv:
        return None
    path = Path(route_csv).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        ROOT / path,
        ROOT.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_csv_route_points(path, x_col="x", y_col="y"):
    points = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or x_col not in reader.fieldnames or y_col not in reader.fieldnames:
            raise ValueError(f"CSV route must contain columns {x_col!r} and {y_col!r}: {path}")
        for row in reader:
            if row.get(x_col, "") == "" or row.get(y_col, "") == "":
                continue
            points.append((float(row[x_col]), float(row[y_col])))
    if len(points) < 2:
        raise ValueError(f"CSV route needs at least two points: {path}")
    return points


def route_points_from_env_planner(env):
    if not hasattr(env, "get_ego_planner"):
        return []
    planner = env.get_ego_planner()
    if not hasattr(planner, "get_global_waypoints"):
        return []
    return [(float(wp[0]), float(wp[1])) for wp in planner.get_global_waypoints()]


def get_fixed_delta_seconds(env):
    candidates = []
    try:
        candidates.append(env._world._settings.fixed_delta_seconds)
    except Exception:
        pass
    try:
        candidates.append(env.get_ego_vehicle().get_world().get_settings().fixed_delta_seconds)
    except Exception:
        pass
    for value in candidates:
        try:
            value = float(value)
            if value > 0:
                return value
        except Exception:
            pass
    return 0.05


def make_basic_agent_route(env, points, args):
    import carla
    from car_dreamer.toolkit.planner.agents.navigation.local_planner import RoadOption

    ego = env.get_ego_vehicle() if hasattr(env, "get_ego_vehicle") else env.ego
    carla_map = ego.get_world().get_map()
    plan = []
    last_loc = None
    min_dist = max(float(args.csv_min_waypoint_dist), 0.0)
    for x, y in points:
        loc = carla.Location(x=float(x), y=float(y), z=0.1)
        waypoint = carla_map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            continue
        wp_loc = waypoint.transform.location
        if last_loc is not None and wp_loc.distance(last_loc) < min_dist:
            continue
        plan.append((waypoint, RoadOption.LANEFOLLOW))
        last_loc = wp_loc
    if len(plan) < 2:
        raise RuntimeError("Could not project enough CSV points onto CARLA driving lanes.")
    return plan


def draw_basic_agent_route(env, plan):
    import carla

    try:
        world = env.get_ego_vehicle().get_world()
        color = carla.Color(0, 180, 255)
        for (wp0, _), (wp1, _) in zip(plan[:-1], plan[1:]):
            p0 = wp0.transform.location + carla.Location(z=0.35)
            p1 = wp1.transform.location + carla.Location(z=0.35)
            world.debug.draw_line(p0, p1, thickness=0.08, color=color, life_time=30.0)
    except Exception as exc:
        print(f"[CSV Control] route debug draw skipped: {exc}", flush=True)


def init_csv_basic_agent(env, args):
    from car_dreamer.toolkit.planner.agents.navigation.basic_agent import BasicAgent

    route_path = resolve_route_csv(env, args)
    if route_path is not None:
        if not route_path.exists():
            raise FileNotFoundError(f"CSV route not found: {route_path}")
        points = load_csv_route_points(route_path, args.csv_x_col, args.csv_y_col)
        route_label = str(route_path)
    else:
        points = route_points_from_env_planner(env)
        route_label = "env planner"

    if not points:
        raise RuntimeError(
            "CSV control needs --route_csv or a task config with env.route_csv/global waypoints."
        )

    plan = make_basic_agent_route(env, points, args)
    ego = env.get_ego_vehicle() if hasattr(env, "get_ego_vehicle") else env.ego
    carla_map = ego.get_world().get_map()
    opt_dict = {
        "target_speed": float(args.csv_target_speed),
        "dt": get_fixed_delta_seconds(env),
        "ignore_traffic_lights": not bool(args.csv_respect_traffic_lights),
        "ignore_vehicles": not bool(args.csv_respect_vehicles),
    }
    agent = BasicAgent(
        ego,
        target_speed=float(args.csv_target_speed),
        opt_dict=opt_dict,
        map_inst=carla_map,
    )
    agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
    if args.csv_draw_route:
        draw_basic_agent_route(env, plan)
    print(
        "[CSV Control] BasicAgent route loaded "
        f"source={route_label} raw_points={len(points)} projected_waypoints={len(plan)} "
        f"target_speed={args.csv_target_speed:.1f}km/h",
        flush=True,
    )
    return agent


def vehicle_control_to_env_action(control, env):
    acc = 3.0 * float(control.throttle) - 3.0 * float(control.brake)
    steer = -float(control.steer)
    action = np.asarray([acc, steer], dtype=np.float32)
    space = getattr(env, "action_space", None)
    if space is not None and hasattr(space, "low") and hasattr(space, "high"):
        low = np.asarray(space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(space.high, dtype=np.float32).reshape(-1)
        if low.size >= 2 and high.size >= 2:
            action = np.clip(action, low[:2], high[:2]).astype(np.float32)
    return action


def csv_basic_agent_action(agent, env):
    if agent.done():
        return np.asarray([-3.0, 0.0], dtype=np.float32)
    control = agent.run_step()
    return vehicle_control_to_env_action(control, env)


def update_speed_pid_dt(env, speed_pid):
    try:
        fixed_delta = float(env._world._settings.fixed_delta_seconds)
        speed_pid.dt = max(fixed_delta, 1e-6)
    except Exception:
        pass


def ros2_expert_action(cp, rclpy, bridge, speed_pid, env, args, timeout_count):
    update_speed_pid_dt(env, speed_pid)
    bridge.publish_env(env)
    rclpy.spin_once(bridge, timeout_sec=args.spin_timeout_sec)

    cmd = bridge.get_latest_control()
    if cmd is None:
        env_action = np.array([cp.ENV_ACC_MIN, 0.0], dtype=np.float32)
        timeout_count += 1
        if timeout_count <= 5 or timeout_count % 20 == 0:
            print(
                f"[VAD ROS2 Expert] /ctrl_info timeout, fallback action={env_action.tolist()} "
                f"(count={timeout_count})",
                flush=True,
            )
    else:
        env_action = cp.ros_cmd_to_env_action(
            cmd=cmd,
            ego=env.ego,
            speed_pid=speed_pid,
            steer_max_deg=args.steer_max_deg,
            use_brake_pressure=not args.no_brake_pressure,
        )

    traj = bridge.get_latest_trajectory()
    if hasattr(env, "set_expert_trajectory"):
        env.set_expert_trajectory(traj)

    return cp.clip_env_action(env_action), timeout_count


def main(argv=None):
    args, rest = parse_args(argv)
    args = normalize_vad_runtime_args(args)

    install_vad_carla_patches(
        task=args.task,
        width=args.camera_width,
        height=args.camera_height,
        fov=args.camera_fov,
        sensor_tick=args.sensor_tick,
        include_birdeye=True,
        camera_profile=args.camera_profile,
    )

    import cv2
    import car_dreamer

    torch, model, cfg = build_vad(args)
    box_type_3d = import_box_type(args)
    base_meta = make_img_meta(args, cfg)
    base_meta["box_type_3d"] = box_type_3d
    print(
        "[VAD Realtime] "
        f"camera_profile={args.camera_profile} "
        f"camera_order={', '.join(get_camera_order(args.camera_profile))}",
        flush=True,
    )

    env, _ = car_dreamer.create_task(args.task, rest)
    obs = env.reset()
    prev_bev = None
    prev_pose = None
    latest_result = None
    manual_action = make_env_action(env, args)
    ros2 = None
    csv_agent = None
    vad_calibrated = not bool(args.auto_calibrate_vad)
    timeout_count = 0
    if args.csv_control:
        if args.manual_action:
            print("[CSV Control] --csv_control enabled; ignoring --manual_action.", flush=True)
        csv_agent = init_csv_basic_agent(env, args)
    elif not args.manual_action:
        ros2 = init_ros2_expert(args)
        _, _, bridge, speed_pid = ros2
        reset_ros2_expert(bridge, speed_pid)
        print(
            "[VAD ROS2 Expert] enabled topics "
            f"obs={args.obs_topic} ctrl={args.ctrl_topic} traj={args.traj_topic}",
            flush=True,
        )
    step = 0

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    try:
        while args.max_steps <= 0 or step < args.max_steps:
            images = stack_vad_images(obs, args)
            pose = ego_pose_from_env(env)

            if csv_agent is not None:
                action = csv_basic_agent_action(csv_agent, env)
            elif ros2 is not None:
                cp, rclpy, bridge, speed_pid = ros2
                action, timeout_count = ros2_expert_action(
                    cp, rclpy, bridge, speed_pid, env, args, timeout_count)
            else:
                action = manual_action

            if step % max(int(args.vad_every), 1) == 0:
                can_bus_prev_pose = None if args.no_temporal_bev else prev_pose
                can_bus = make_can_bus(pose, can_bus_prev_pose)
                meta = make_frame_img_meta(
                    base_meta,
                    can_bus,
                    Path(args.task),
                    step,
                    can_bus_prev_pose is not None,
                )
                img_np = normalize_and_resize(images, args, cfg)
                try:
                    if not vad_calibrated:
                        base_meta, next_prev_bev, latest_result, meta = auto_calibrate_vad_geometry(
                            torch, model, img_np, pose, args, cfg, box_type_3d)
                        vad_calibrated = True
                    else:
                        vad_prev_bev = None if args.no_temporal_bev else prev_bev
                        next_prev_bev, latest_result = run_vad_tensor(
                            torch, model, img_np, meta, args, vad_prev_bev)
                    prev_bev = None if args.no_temporal_bev else next_prev_bev
                    print_vad_debug(step, obs, img_np, meta, latest_result, cfg, args, pose, prev_pose)
                    prev_pose = pose
                except Exception as exc:
                    cv2.destroyAllWindows()
                    raise RuntimeError(
                        "VAD realtime inference failed. Check checkpoint/config compatibility "
                        "and the CARLA camera/calibration/meta input contract."
                    ) from exc

            surround = render_surround(obs, args, meta=base_meta, cfg=cfg)
            bev = render_bev(latest_result, cfg, args)
            cv2.imshow(args.window, make_visual(surround, bev))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            obs, _, done, _ = env.step(action)
            step += 1
            if done:
                obs = env.reset()
                prev_bev = None
                prev_pose = None
                latest_result = None
                timeout_count = 0
                vad_calibrated = not bool(args.auto_calibrate_vad)
                if csv_agent is not None:
                    csv_agent = init_csv_basic_agent(env, args)
                if ros2 is not None:
                    _, _, bridge, speed_pid = ros2
                    reset_ros2_expert(bridge, speed_pid)
            time.sleep(0.001)
    finally:
        cv2.destroyAllWindows()
        if ros2 is not None:
            _, rclpy, bridge, _ = ros2
            try:
                bridge.destroy_node()
            except Exception:
                pass
            if rclpy.ok():
                rclpy.shutdown()
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
