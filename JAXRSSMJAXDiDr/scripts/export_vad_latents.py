"""Export compact VAD BEV latents from six-camera CARLA replay chunks.

The script uses a VAD checkpoint as a frozen perception encoder and writes a
single RSSM-friendly observation key, `vad_scene_latent`, to new NPZ chunks.
It intentionally exports encoder BEV features instead of VAD's final ego plan.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Dict

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from JAXRSSMJAXDiDr.vad_carla.camera_setup import (
    VAD_CAMERA_KEYS,
    VAD_CAMERA_ORDER,
    iter_camera_specs,
    lidar2img_matrix,
    select_camera_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vad_root", default=str(ROOT / "VAD"))
    parser.add_argument("--vad_model", choices=("tiny", "base"), default="tiny")
    parser.add_argument("--vad_checkpoint", required=True)
    parser.add_argument("--camera_width", type=int, default=1600)
    parser.add_argument("--camera_height", type=int, default=900)
    parser.add_argument("--camera_fov", type=float, default=70.0)
    parser.add_argument("--output_key", default="vad_scene_latent")
    parser.add_argument("--pool", choices=("mean", "mean_std"), default="mean_std")
    parser.add_argument(
        "--latent_components",
        default="bev,agent,map,ego",
        help="Comma-separated components for output_key. Choices: bev, agent, map, ego. "
        "Use 'bev' to reproduce the earlier BEV-only latent.",
    )
    parser.add_argument("--pretrained_norm", action="store_true", default=True)
    parser.add_argument("--no_pretrained_norm", dest="pretrained_norm", action="store_false")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit_chunks", type=int, default=0)
    return parser.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        chunk = {key: np.asarray(data[key]) for key in data.files}
    chunk["__chunk_path__"] = np.asarray(str(path))
    chunk["__replay_dir__"] = np.asarray(str(path.parent))
    return chunk


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    drop_keys = set(VAD_CAMERA_KEYS)
    drop_keys.update(f"{key}_path" for key in VAD_CAMERA_KEYS)
    compact = {
        key: value
        for key, value in arrays.items()
        if key not in drop_keys and not key.startswith("__")
    }
    np.savez_compressed(path, **compact)


def import_vad_runtime(vad_root: Path):
    if str(vad_root) not in sys.path:
        sys.path.insert(0, str(vad_root))
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model

    plugin_dir = vad_root / "projects" / "mmdet3d_plugin"
    if str(plugin_dir.parent) not in sys.path:
        sys.path.insert(0, str(plugin_dir.parent))
    importlib.import_module("projects.mmdet3d_plugin")
    return torch, Config, load_checkpoint, build_model


def build_vad(args: argparse.Namespace):
    vad_root = Path(args.vad_root)
    torch, Config, load_checkpoint, build_model = import_vad_runtime(vad_root)
    config_name = "VAD_tiny_stage_2.py" if args.vad_model == "tiny" else "VAD_base_stage_2.py"
    cfg = Config.fromfile(str(vad_root / "projects" / "configs" / "VAD" / config_name))
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.vad_checkpoint, map_location="cpu")
    model.to(args.device)
    model.eval()
    return torch, model, cfg


def normalize_and_resize(images: np.ndarray, args: argparse.Namespace, cfg) -> np.ndarray:
    import cv2

    scale = 0.4 if args.vad_model == "tiny" else 0.8
    out = []
    if args.pretrained_norm:
        # VAD docs note that released checkpoints were trained with this BGR,
        # std=1 normalization. CARLA CameraHandler returns RGB, so flip first.
        mean = np.asarray([103.530, 116.280, 123.675], dtype=np.float32)
        std = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        to_bgr = True
    else:
        mean = np.asarray(cfg.img_norm_cfg["mean"], dtype=np.float32)
        std = np.asarray(cfg.img_norm_cfg["std"], dtype=np.float32)
        to_bgr = bool(cfg.img_norm_cfg.get("to_rgb", False)) is False

    for image in images:
        arr = np.asarray(image, dtype=np.float32)
        if to_bgr:
            arr = arr[..., ::-1]
        resized = cv2.resize(arr, (int(arr.shape[1] * scale), int(arr.shape[0] * scale)))
        resized = (resized - mean) / std
        pad_h = int(np.ceil(resized.shape[0] / 32.0) * 32)
        pad_w = int(np.ceil(resized.shape[1] / 32.0) * 32)
        padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        out.append(padded.transpose(2, 0, 1))
    return np.stack(out, axis=0)


def make_img_meta(args: argparse.Namespace, cfg) -> dict:
    scale = 0.4 if args.vad_model == "tiny" else 0.8
    scaled_h = int(args.camera_height * scale)
    scaled_w = int(args.camera_width * scale)
    pad_h = int(np.ceil(scaled_h / 32.0) * 32)
    pad_w = int(np.ceil(scaled_w / 32.0) * 32)
    lidar2img = [
        lidar2img_matrix(spec, args.camera_width, args.camera_height, args.camera_fov, scale=scale)
        for spec in iter_camera_specs(VAD_CAMERA_ORDER)
    ]
    return {
        "filename": list(VAD_CAMERA_ORDER),
        # Match VAD's official pipeline after RandomScaleImageMultiViewImage
        # and PadMultiViewImage: ori_shape is scaled, img_shape/pad_shape are padded.
        "ori_shape": [(scaled_h, scaled_w, 3)] * 6,
        "img_shape": [(pad_h, pad_w, 3)] * 6,
        "pad_shape": [(pad_h, pad_w, 3)] * 6,
        "lidar2img": lidar2img,
        "can_bus": np.zeros((18,), dtype=np.float32),
        "box_type_3d": None,
        "scene_token": "carla",
        "sample_idx": 0,
        "prev_idx": "",
        "next_idx": "",
    }


def _step_scalar(chunk: Dict[str, np.ndarray], key: str, index: int) -> float:
    if key not in chunk:
        raise KeyError(f"Replay chunk is missing required ego field: {key}")
    value = np.asarray(chunk[key][index])
    return float(value.reshape(-1)[0])


def _optional_step_scalar(
    chunk: Dict[str, np.ndarray],
    key: str,
    index: int,
    default: float = 0.0,
) -> float:
    if key not in chunk:
        return float(default)
    value = np.asarray(chunk[key][index])
    return float(value.reshape(-1)[0])


def _step_bool(chunk: Dict[str, np.ndarray], key: str, index: int) -> bool:
    if key not in chunk:
        return False
    value = np.asarray(chunk[key][index])
    return bool(np.any(value))


def _is_sequence_start(chunk: Dict[str, np.ndarray], index: int) -> bool:
    if index == 0:
        return True
    return (
        _step_bool(chunk, "is_first", index)
        or _step_bool(chunk, "episode_start", index)
        or _step_bool(chunk, "reset_export", index)
    )


def _patch_angle_deg(yaw_rad: float) -> float:
    return float(np.degrees(yaw_rad) % 360.0)


def _ego_pose_from_chunk(chunk: Dict[str, np.ndarray], index: int) -> dict:
    # CarDreamer/CARLA world uses x-forward, y-right. VAD/nuScenes BEV
    # expects x-forward, y-left, so convert global y, yaw and yaw-rate here.
    yaw = -_step_scalar(chunk, "ego_yaw", index)
    return {
        "x": _step_scalar(chunk, "ego_x", index),
        "y": -_step_scalar(chunk, "ego_y", index),
        "yaw": yaw,
        "patch_angle": _patch_angle_deg(yaw),
        "speed": _optional_step_scalar(chunk, "ego_speed", index),
        "yawrate": -_optional_step_scalar(chunk, "ego_yawrate", index),
    }


def make_can_bus(cur_pose: dict, prev_pose: dict | None) -> np.ndarray:
    """Build VAD-style can_bus for direct transformer calls.

    VAD's dataset/detector passes absolute ego pose into img_meta and then
    converts translation and rotation to frame deltas before BEV temporal
    fusion. This exporter bypasses that detector wrapper, so the delta fields
    are computed here.
    """

    can_bus = np.zeros((18,), dtype=np.float32)
    if prev_pose is not None:
        can_bus[0] = float(cur_pose["x"] - prev_pose["x"])
        can_bus[1] = float(cur_pose["y"] - prev_pose["y"])
        can_bus[-1] = float(cur_pose["patch_angle"] - prev_pose["patch_angle"])

    yaw = float(cur_pose["yaw"])
    half_yaw = 0.5 * yaw
    # Quaternion order matches pyquaternion/nuscenes: w, x, y, z.
    can_bus[3:7] = np.asarray(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float32,
    )
    can_bus[12] = float(cur_pose["yawrate"])
    can_bus[13] = float(cur_pose["speed"] * np.cos(yaw))
    can_bus[14] = float(cur_pose["speed"] * np.sin(yaw))
    can_bus[-2] = float(np.radians(cur_pose["patch_angle"]))
    return can_bus


def make_frame_img_meta(
    base_meta: dict,
    can_bus: np.ndarray,
    path: Path,
    index: int,
    has_prev: bool,
) -> dict:
    meta = dict(base_meta)
    # VAD's temporal BEV path passes can_bus[-1] directly to
    # torchvision.transforms.functional.rotate, whose older versions reject
    # numpy scalar angles. Keep the metadata as plain Python floats.
    meta["can_bus"] = [float(x) for x in np.asarray(can_bus).reshape(-1)]
    meta["scene_token"] = path.stem
    meta["sample_idx"] = int(index)
    meta["prev_idx"] = int(index - 1) if has_prev else ""
    meta["next_idx"] = int(index + 1)
    return meta


def pool_tokens(tokens: np.ndarray, mode: str) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens[0]
    tokens = tokens.reshape((-1, tokens.shape[-1]))
    mean = tokens.mean(axis=0)
    if mode == "mean":
        return mean.astype(np.float32)
    std = tokens.std(axis=0)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def latent_components_arg(value: str) -> tuple[str, ...]:
    components = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {"bev", "agent", "map", "ego"}
    unknown = [part for part in components if part not in allowed]
    if unknown:
        raise ValueError(f"Unknown latent components {unknown}; allowed={sorted(allowed)}")
    if not components:
        raise ValueError("At least one latent component is required.")
    return components


def extract_vad_tokens(torch, model, feats, img_metas, prev_bev):
    """Run VAD perception transformer and expose clean internal query tokens."""

    head = model.pts_bbox_head
    bs = feats[0].shape[0]
    dtype = feats[0].dtype
    object_query_embeds = head.query_embedding.weight.to(dtype)
    if head.map_query_embed_type == "all_pts":
        map_query_embeds = head.map_query_embedding.weight.to(dtype)
    elif head.map_query_embed_type == "instance_pts":
        map_pts_embeds = head.map_pts_embedding.weight.unsqueeze(0)
        map_instance_embeds = head.map_instance_embedding.weight.unsqueeze(1)
        map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)
    else:
        raise ValueError(f"Unsupported map_query_embed_type={head.map_query_embed_type!r}")

    bev_queries = head.bev_embedding.weight.to(dtype)
    bev_mask = torch.zeros((bs, head.bev_h, head.bev_w), device=bev_queries.device).to(dtype)
    bev_pos = head.positional_encoding(bev_mask).to(dtype)
    outputs = head.transformer(
        feats,
        bev_queries,
        object_query_embeds,
        map_query_embeds,
        head.bev_h,
        head.bev_w,
        grid_length=(head.real_h / head.bev_h, head.real_w / head.bev_w),
        bev_pos=bev_pos,
        reg_branches=head.reg_branches if head.with_box_refine else None,
        cls_branches=head.cls_branches if head.as_two_stage else None,
        map_reg_branches=head.map_reg_branches if head.with_box_refine else None,
        map_cls_branches=head.map_cls_branches if head.as_two_stage else None,
        img_metas=img_metas,
        prev_bev=prev_bev,
    )
    bev_embed, agent_hs, _, _, map_hs, _, _ = outputs
    # VAD transformer returns [HW, B, D] for BEV and [L, Q, B, D] for query states.
    bev_tokens = bev_embed.permute(1, 0, 2)
    agent_tokens = agent_hs[-1].permute(1, 0, 2)
    map_tokens = map_hs[-1].permute(1, 0, 2)
    ego_tokens = head.ego_query.weight.unsqueeze(0).repeat(bs, 1, 1)
    return {
        "bev": bev_tokens,
        "agent": agent_tokens,
        "map": map_tokens,
        "ego": ego_tokens,
    }, bev_tokens


def export_chunk(path: Path, out_path: Path, torch, model, cfg, args: argparse.Namespace) -> None:
    chunk = load_npz(path)
    length = len(chunk["action"]) if "action" in chunk else len(next(iter(chunk.values())))
    latents = []
    prev_bev = None
    prev_pose = None
    base_meta = make_img_meta(args, cfg)
    components = latent_components_arg(args.latent_components)
    with torch.no_grad():
        for index in range(length):
            if _is_sequence_start(chunk, index):
                prev_bev = None
                prev_pose = None

            cur_pose = _ego_pose_from_chunk(chunk, index)
            can_bus = make_can_bus(cur_pose, prev_pose)
            meta = make_frame_img_meta(base_meta, can_bus, path, index, prev_pose is not None)
            images = select_camera_arrays(chunk, index)
            img_np = normalize_and_resize(images, args, cfg)
            img = torch.from_numpy(img_np[None]).to(args.device)
            feats = model.extract_feat(img=img, img_metas=[meta])
            token_dict, prev_bev = extract_vad_tokens(torch, model, feats, [meta], prev_bev)
            pooled = [
                pool_tokens(token_dict[name].detach().cpu().numpy(), args.pool)
                for name in components
            ]
            latents.append(np.concatenate(pooled, axis=0).astype(np.float32))
            prev_pose = cur_pose
    chunk[args.output_key] = np.stack(latents, axis=0).astype(np.float32)
    save_npz(out_path, chunk)


def main() -> None:
    args = parse_args()
    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir)
    paths = sorted(replay_dir.glob("*.npz"))
    if args.limit_chunks:
        paths = paths[: int(args.limit_chunks)]
    if not paths:
        raise FileNotFoundError(f"No replay chunks found in {replay_dir}")

    torch, model, cfg = build_vad(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vad_export] model={args.vad_model} checkpoint={args.vad_checkpoint}")
    print(f"[vad_export] cameras={', '.join(VAD_CAMERA_ORDER)}")
    print(f"[vad_export] latent_components={args.latent_components} pool={args.pool}")
    for idx, path in enumerate(paths, 1):
        out_path = output_dir / path.name
        print(f"[vad_export] {idx}/{len(paths)} {path.name} -> {out_path}")
        export_chunk(path, out_path, torch, model, cfg, args)
    print(f"[vad_export] wrote {len(paths)} chunks to {output_dir}")


if __name__ == "__main__":
    main()
