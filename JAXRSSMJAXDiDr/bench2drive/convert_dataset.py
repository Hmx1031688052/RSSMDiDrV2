"""Convert extracted Bench2Drive clips into Dreamer row-format replay chunks."""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from JAXRSSMJAXDiDr.bench2drive.bev import PrivilegedBEVRenderer, load_topdown_bev
from JAXRSSMJAXDiDr.bench2drive.features import (
    build_actions,
    build_future_waypoints8,
    build_neighbor_features_from_annotations,
    build_route_features,
    extract_ego_arrays,
    trajectory_from_waypoints8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b2d_root", required=True, help="Root containing extracted Bench2Drive clips.")
    parser.add_argument("--split_json", default=None, help="Optional Bench2Drive split json listing clip tarballs.")
    parser.add_argument("--hdmap_dir", default=None, help="Directory containing Town*_HD_map.npz or Town*_lanemarkings.npz files.")
    parser.add_argument("--output_replay_dir", required=True)
    parser.add_argument("--bev_mode", choices=("none", "topdown_rgb", "privileged_renderer"), default="topdown_rgb")
    parser.add_argument("--bev_shape", nargs=2, type=int, default=(128, 128), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--chunk_length", type=int, default=64)
    parser.add_argument("--waypoint_scale", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--waypoint_interval", type=int, default=5)
    parser.add_argument("--neighbor_k", type=int, default=8)
    parser.add_argument("--neighbor_radius", type=float, default=50.0)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--max_frames_per_clip", type=int, default=None)
    parser.add_argument("--skip_missing_bev", action="store_true")
    parser.add_argument("--summary_path", default=None)
    return parser.parse_args()


def load_json_gz(path: Path) -> Mapping:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def split_stems(split_json: Optional[str | Path]) -> Optional[List[str]]:
    if split_json is None:
        return None
    split_path = Path(split_json)
    data = json.loads(split_path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        names = list(data.keys())
    elif isinstance(data, Sequence):
        names = list(data)
    else:
        raise ValueError(f"Unsupported split json shape: {split_path}")
    stems = []
    for name in names:
        stem = str(name)
        if stem.endswith(".tar.gz"):
            stem = stem[:-7]
        elif stem.endswith(".tgz"):
            stem = stem[:-4]
        stems.append(stem)
    return stems


def _candidate_paths_for_stem(root: Path, stem: str) -> Iterable[Path]:
    yield root / stem
    match = re.match(r"(?P<scenario>.+?)_(?P<town>Town[^_]+)_Route(?P<route>[^_]+)_Weather(?P<weather>[^_]+)$", stem)
    if not match:
        return
    scenario = match.group("scenario")
    town = match.group("town")
    route = match.group("route")
    weather = match.group("weather")
    yield root / scenario / f"{town}_Route{route}_Weather{weather}"
    yield root / scenario / f"{town}_Weather{weather}_Route{route}"
    yield root / scenario / f"{town}_route{route}_weather{weather}"
    yield root / scenario / f"{town}_weather{weather}_route{route}"


def discover_clip_dirs(root: str | Path, split_json: Optional[str | Path], max_clips: Optional[int]) -> List[Path]:
    root = Path(root)
    stems = split_stems(split_json)
    found: List[Path] = []
    if stems is None:
        for anno_dir in sorted(root.rglob("anno")):
            if anno_dir.is_dir() and any(anno_dir.glob("*.json.gz")):
                found.append(anno_dir.parent)
                if max_clips is not None and len(found) >= int(max_clips):
                    break
        return found

    all_anno_dirs = None
    for stem in stems:
        candidates = [path for path in _candidate_paths_for_stem(root, stem) if (path / "anno").is_dir()]
        if not candidates:
            if all_anno_dirs is None:
                all_anno_dirs = [path.parent for path in sorted(root.rglob("anno")) if path.is_dir()]
            candidates = [path for path in all_anno_dirs if stem in str(path).replace("\\", "/")]
        if candidates:
            found.append(candidates[0])
        if max_clips is not None and len(found) >= int(max_clips):
            break
    return found


def town_from_clip_or_annotation(clip_dir: Path, annotations: Sequence[Mapping]) -> Optional[str]:
    text = str(clip_dir)
    match = re.search(r"(Town[^_\\/]+)", text)
    if match:
        return match.group(1)
    for anno in annotations:
        town = anno.get("town")
        if town:
            return str(town)
    return None


def find_hdmap_path(hdmap_dir: Optional[str | Path], town: Optional[str]) -> Optional[Path]:
    if not hdmap_dir or not town:
        return None
    root = Path(hdmap_dir)
    candidates = [
        root / f"{town}_HD_map.npz",
        root / f"{town}_lanemarkings.npz",
        root / f"{town}_HD_map.npz".replace("Town", "Town"),
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(root.glob(f"{town}*.npz"))
    return matches[0] if matches else None


def sorted_annotation_paths(clip_dir: Path, max_frames: Optional[int]) -> List[Path]:
    paths = sorted((clip_dir / "anno").glob("*.json.gz"))
    if max_frames is not None:
        paths = paths[: int(max_frames)]
    return paths


def topdown_path_for_annotation(clip_dir: Path, anno_path: Path) -> Path:
    frame_stem = anno_path.stem.split(".")[0]
    folder = clip_dir / "camera" / "rgb_top_down"
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = folder / f"{frame_stem}{suffix}"
        if candidate.exists():
            return candidate
    return folder / f"{frame_stem}.jpg"


def build_bev_array(
    clip_dir: Path,
    anno_paths: Sequence[Path],
    annotations: Sequence[Mapping],
    *,
    bev_mode: str,
    bev_shape: Tuple[int, int],
    hdmap_path: Optional[Path],
    skip_missing: bool,
) -> Optional[np.ndarray]:
    if bev_mode == "none":
        return None
    frames = []
    if bev_mode == "privileged_renderer":
        renderer = PrivilegedBEVRenderer(shape=bev_shape, hdmap_path=hdmap_path)
    else:
        renderer = None
    for anno_path, anno in zip(anno_paths, annotations):
        if bev_mode == "topdown_rgb":
            image_path = topdown_path_for_annotation(clip_dir, anno_path)
            try:
                frame = load_topdown_bev(image_path, bev_shape)
            except FileNotFoundError:
                if not skip_missing:
                    raise
                frame = np.zeros((bev_shape[0], bev_shape[1], 3), dtype=np.uint8)
        elif bev_mode == "privileged_renderer":
            frame = renderer.render(anno)
        else:
            raise ValueError(f"Unknown bev_mode: {bev_mode}")
        frames.append(frame.astype(np.uint8))
    return np.stack(frames, axis=0).astype(np.uint8)


def chunk_filename(clip_index: int, chunk_index: int, length: int) -> str:
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{clip_index:08d}-{chunk_index:08d}-{int(length)}.npz"


def make_replay_arrays(
    annotations: Sequence[Mapping],
    *,
    waypoint_scale: float,
    dt: float,
    waypoint_interval: int,
    neighbor_k: int,
    neighbor_radius: float,
) -> Dict[str, np.ndarray]:
    ego = extract_ego_arrays(annotations)
    action = build_actions(annotations)
    expert_waypoints8 = build_future_waypoints8(
        ego["ego_x"],
        ego["ego_y"],
        ego["ego_yaw"],
        waypoint_scale=waypoint_scale,
        dt=dt,
        waypoint_interval=waypoint_interval,
    )
    trajectory = trajectory_from_waypoints8(expert_waypoints8, waypoint_scale=waypoint_scale)
    neighbor_local, neighbor_world = build_neighbor_features_from_annotations(
        annotations,
        ego["ego_x"],
        ego["ego_y"],
        ego["ego_yaw"],
        neighbor_k=neighbor_k,
        neighbor_radius=neighbor_radius,
        dt=dt,
    )
    route = build_route_features(annotations, ego["ego_x"], ego["ego_y"], ego["ego_yaw"], waypoint_scale=waypoint_scale)
    length = len(annotations)
    arrays = {
        **ego,
        **route,
        "neighbor_vehicles_local": neighbor_local,
        "neighbor_vehicles_world": neighbor_world,
        "action": action.astype(np.float32),
        "reward": np.zeros((length,), dtype=np.float32),
        "is_first": np.zeros((length,), dtype=bool),
        "is_last": np.zeros((length,), dtype=bool),
        "is_terminal": np.zeros((length,), dtype=bool),
        "expert_waypoints8": expert_waypoints8.astype(np.float32),
        "future_ego_waypoints8": expert_waypoints8.astype(np.float32),
        "trajectory": trajectory.astype(np.float32),
    }
    if length:
        arrays["is_first"][0] = True
        arrays["is_last"][-1] = True
        arrays["is_terminal"][-1] = True
        arrays["target_region"][-1, 0] = 1.0
    return arrays


def validate_arrays(arrays: Mapping[str, np.ndarray], clip_dir: Path) -> None:
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.dtype.kind in {"f", "c"} and not np.isfinite(arr).all():
            raise ValueError(f"{clip_dir}: non-finite values in {key}")
    required = ("action", "expert_waypoints8", "trajectory", "ego_x", "ego_y", "ego_yaw")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"{clip_dir}: missing required arrays {missing}")


def write_chunks(
    arrays: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    clip_index: int,
    chunk_length: int,
    clip_name: str,
) -> List[Dict[str, object]]:
    length = int(len(np.asarray(arrays["action"])))
    summaries = []
    chunk_length = max(1, int(chunk_length))
    for chunk_index, start in enumerate(range(0, length, chunk_length)):
        end = min(start + chunk_length, length)
        out = {}
        for key, value in arrays.items():
            value = np.asarray(value)
            if value.ndim > 0 and value.shape[0] == length:
                out[key] = value[start:end]
            else:
                out[key] = value
        out["is_first"] = np.asarray(out["is_first"]).copy()
        out["is_last"] = np.asarray(out["is_last"]).copy()
        out["is_terminal"] = np.asarray(out["is_terminal"]).copy()
        out["is_first"][0] = True
        out["is_last"][-1] = True
        out["is_terminal"][-1] = bool(end == length)
        out["source_clip"] = np.asarray(clip_name)
        out["source_start"] = np.int32(start)
        filename = chunk_filename(clip_index, chunk_index, end - start)
        path = output_dir / filename
        np.savez_compressed(path, **out)
        summaries.append({"path": str(path), "clip": clip_name, "start": int(start), "length": int(end - start)})
    return summaries


def convert_clip(clip_dir: Path, output_dir: Path, args: argparse.Namespace, clip_index: int) -> Dict[str, object]:
    anno_paths = sorted_annotation_paths(clip_dir, args.max_frames_per_clip)
    if not anno_paths:
        raise FileNotFoundError(f"No anno/*.json.gz files under {clip_dir}")
    annotations = [load_json_gz(path) for path in anno_paths]
    town = town_from_clip_or_annotation(clip_dir, annotations)
    hdmap_path = find_hdmap_path(args.hdmap_dir, town)
    arrays = make_replay_arrays(
        annotations,
        waypoint_scale=float(args.waypoint_scale),
        dt=float(args.dt),
        waypoint_interval=int(args.waypoint_interval),
        neighbor_k=int(args.neighbor_k),
        neighbor_radius=float(args.neighbor_radius),
    )
    bev = build_bev_array(
        clip_dir,
        anno_paths,
        annotations,
        bev_mode=args.bev_mode,
        bev_shape=(int(args.bev_shape[0]), int(args.bev_shape[1])),
        hdmap_path=hdmap_path,
        skip_missing=bool(args.skip_missing_bev),
    )
    if bev is not None:
        arrays["b2d_bev"] = bev
    validate_arrays(arrays, clip_dir)
    chunks = write_chunks(
        arrays,
        output_dir,
        clip_index=clip_index,
        chunk_length=int(args.chunk_length),
        clip_name=str(clip_dir),
    )
    return {
        "clip": str(clip_dir),
        "frames": len(annotations),
        "chunks": chunks,
        "town": town,
        "hdmap_path": str(hdmap_path) if hdmap_path else None,
        "bev_mode": args.bev_mode,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_replay_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_dirs = discover_clip_dirs(args.b2d_root, args.split_json, args.max_clips)
    if not clip_dirs:
        raise FileNotFoundError(f"No extracted Bench2Drive clips found under {args.b2d_root}")

    summaries = []
    for clip_index, clip_dir in enumerate(clip_dirs):
        print(f"[b2d_convert] {clip_index + 1}/{len(clip_dirs)} {clip_dir}", flush=True)
        try:
            summaries.append(convert_clip(clip_dir, output_dir, args, clip_index))
        except Exception as exc:
            print(f"[b2d_convert] ERROR {clip_dir}: {exc}", file=sys.stderr, flush=True)
            raise

    summary = {
        "b2d_root": str(Path(args.b2d_root)),
        "split_json": str(args.split_json) if args.split_json else None,
        "output_replay_dir": str(output_dir),
        "bev_mode": args.bev_mode,
        "bev_shape": [int(args.bev_shape[0]), int(args.bev_shape[1])],
        "chunk_length": int(args.chunk_length),
        "clips": len(summaries),
        "chunks": int(sum(len(item["chunks"]) for item in summaries)),
        "summaries": summaries,
    }
    summary_path = Path(args.summary_path) if args.summary_path else output_dir / "bench2drive_conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[b2d_convert] Wrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
