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
        return {key: np.asarray(data[key]) for key in data.files}


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


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
    lidar2img = [
        lidar2img_matrix(spec, args.camera_width, args.camera_height, args.camera_fov, scale=scale)
        for spec in iter_camera_specs(VAD_CAMERA_ORDER)
    ]
    return {
        "filename": list(VAD_CAMERA_ORDER),
        "ori_shape": [(args.camera_height, args.camera_width, 3)] * 6,
        "img_shape": [(int(args.camera_height * scale), int(args.camera_width * scale), 3)] * 6,
        "pad_shape": [(int(np.ceil(args.camera_height * scale / 32.0) * 32), int(np.ceil(args.camera_width * scale / 32.0) * 32), 3)] * 6,
        "lidar2img": lidar2img,
        "can_bus": np.zeros((18,), dtype=np.float32),
        "box_type_3d": None,
        "scene_token": "carla",
        "sample_idx": 0,
        "prev_idx": "",
        "next_idx": "",
    }


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
    meta = make_img_meta(args, cfg)
    components = latent_components_arg(args.latent_components)
    with torch.no_grad():
        for index in range(length):
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
