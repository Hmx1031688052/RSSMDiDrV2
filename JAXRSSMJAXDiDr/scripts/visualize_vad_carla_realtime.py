"""Visualize live VAD perception outputs on CARLA six-camera observations."""

from __future__ import annotations

import argparse
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
    build_vad,
    make_can_bus,
    make_frame_img_meta,
    make_img_meta,
    normalize_and_resize,
)
from JAXRSSMJAXDiDr.vad_carla import install_vad_carla_patches
from JAXRSSMJAXDiDr.vad_carla.camera_setup import VAD_CAMERA_KEYS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="carla_roundabout")
    parser.add_argument("--vad_root", default=str(ROOT / "VAD"))
    parser.add_argument("--vad_model", choices=("tiny", "base"), default="tiny")
    parser.add_argument("--vad_checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--camera_width", type=int, default=1600)
    parser.add_argument("--camera_height", type=int, default=900)
    parser.add_argument("--camera_fov", type=float, default=70.0)
    parser.add_argument("--sensor_tick", type=float, default=0.1)
    parser.add_argument("--pretrained_norm", action="store_true", default=True)
    parser.add_argument("--no_pretrained_norm", dest="pretrained_norm", action="store_false")
    parser.add_argument("--score_thresh", type=float, default=0.35)
    parser.add_argument("--map_score_thresh", type=float, default=0.35)
    parser.add_argument("--bev_size", type=int, default=640)
    parser.add_argument("--front_width", type=int, default=640)
    parser.add_argument("--surround_width", type=int, default=960)
    parser.add_argument("--vad_every", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means run until q/Esc.")
    parser.add_argument("--action_acc", type=float, default=0.0)
    parser.add_argument("--action_steer", type=float, default=0.0)
    parser.add_argument("--manual_action", action="store_true")
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


def import_box_type():
    try:
        from mmdet3d.core import LiDARInstance3DBoxes
    except Exception:
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
    return LiDARInstance3DBoxes


def stack_vad_images(obs: Dict[str, np.ndarray]) -> np.ndarray:
    missing = [key for key in VAD_CAMERA_KEYS if key not in obs]
    if missing:
        raise KeyError(f"Observation is missing VAD camera keys: {missing}")
    return np.stack([np.asarray(obs[key]) for key in VAD_CAMERA_KEYS], axis=0)


def ego_pose_from_env(env) -> dict:
    ego = env.get_ego_vehicle() if hasattr(env, "get_ego_vehicle") else env.ego
    tf = ego.get_transform()
    vel = ego.get_velocity()
    ang = ego.get_angular_velocity()
    yaw = np.radians(float(tf.rotation.yaw))
    speed = float(np.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))
    return {
        "x": float(tf.location.x),
        "y": float(tf.location.y),
        "yaw": float(yaw),
        "patch_angle": float(np.degrees(yaw) % 360.0),
        "speed": speed,
        "yawrate": float(np.radians(ang.z)),
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
    img = torch.from_numpy(img_np[None]).to(args.device)
    with torch.no_grad():
        feats = model.extract_feat(img=img, img_metas=[meta])
        outs = model.pts_bbox_head(feats, [meta], prev_bev=prev_bev)
        bbox_list = model.pts_bbox_head.get_bboxes(outs, [meta], rescale=False)
    return outs.get("bev_embed"), decode_bbox_list(bbox_list[0], outs)


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
        "ego_fut_preds": outs.get("ego_fut_preds"),
    }


def point_to_bev(point, pc_range, size):
    x_min, y_min = float(pc_range[0]), float(pc_range[1])
    x_max, y_max = float(pc_range[3]), float(pc_range[4])
    x, y = float(point[0]), float(point[1])
    px = int((y - y_min) / max(y_max - y_min, 1e-6) * (size - 1))
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
    if trajs.ndim == 1:
        trajs = trajs.reshape(1, -1, 2)
    elif trajs.ndim == 2:
        trajs = trajs[None]
    for mode in trajs[:3]:
        pts = np.cumsum(mode.reshape(-1, 2), axis=0) + np.asarray(center[:2], dtype=np.float32)
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

        ego_fut = to_numpy(result["ego_fut_preds"])
        if ego_fut is not None:
            plans = np.asarray(ego_fut)[0]
            for plan in plans[:3]:
                pts = np.cumsum(plan.reshape(-1, 2), axis=0)
                draw_polyline(canvas, pts, pc_range, (50, 50, 255), thickness=2)

    ego_px = point_to_bev((0.0, 0.0), pc_range, size)
    cv2.circle(canvas, ego_px, 5, (0, 0, 0), -1)
    cv2.putText(canvas, "BEV x-forward y-left", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    return canvas


SURROUND_LAYOUT = (
    ("camera_front_left", "FRONT LEFT"),
    ("camera_front", "FRONT"),
    ("camera_front_right", "FRONT RIGHT"),
    ("camera_back_left", "BACK LEFT"),
    ("camera_back", "BACK"),
    ("camera_back_right", "BACK RIGHT"),
)


def _render_camera_tile(obs, key, label, tile_size):
    import cv2

    if key not in obs:
        raise KeyError(f"Observation is missing camera key: {key}")
    tile_w, tile_h = tile_size
    image = np.asarray(obs[key])
    image = cv2.resize(image, (tile_w, tile_h))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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


def render_surround(obs, args):
    import cv2

    panel_w = max(3, int(getattr(args, "surround_width", 0) or args.front_width))
    tile_w = max(1, panel_w // 3)
    front = np.asarray(obs["camera_front"])
    h, w = front.shape[:2]
    tile_h = max(1, int(h * tile_w / max(w, 1)))
    tiles = [
        _render_camera_tile(obs, key, label, (tile_w, tile_h))
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

    install_vad_carla_patches(
        task=args.task,
        width=args.camera_width,
        height=args.camera_height,
        fov=args.camera_fov,
        sensor_tick=args.sensor_tick,
        include_birdeye=True,
    )

    import cv2
    import car_dreamer

    torch, model, cfg = build_vad(args)
    box_type_3d = import_box_type()
    base_meta = make_img_meta(args, cfg)
    base_meta["box_type_3d"] = box_type_3d

    env, _ = car_dreamer.create_task(args.task, rest)
    obs = env.reset()
    prev_bev = None
    prev_pose = None
    latest_result = None
    manual_action = make_env_action(env, args)
    ros2 = None
    timeout_count = 0
    if not args.manual_action:
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
            images = stack_vad_images(obs)
            pose = ego_pose_from_env(env)

            if ros2 is not None:
                cp, rclpy, bridge, speed_pid = ros2
                action, timeout_count = ros2_expert_action(
                    cp, rclpy, bridge, speed_pid, env, args, timeout_count)
            else:
                action = manual_action

            if step % max(int(args.vad_every), 1) == 0:
                can_bus = make_can_bus(pose, prev_pose)
                meta = make_frame_img_meta(base_meta, can_bus, Path(args.task), step, prev_pose is not None)
                try:
                    prev_bev, latest_result = run_vad_frame(torch, model, images, meta, args, cfg, prev_bev)
                    prev_pose = pose
                except Exception as exc:
                    cv2.destroyAllWindows()
                    raise RuntimeError(
                        "VAD realtime inference failed. Check checkpoint/config compatibility "
                        "and whether this model requires extra planning inputs."
                    ) from exc

            surround = render_surround(obs, args)
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
