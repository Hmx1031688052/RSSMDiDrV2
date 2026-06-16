"""Collect PolyPlanner replay with VAD/nuScenes-style six RGB cameras.

All VAD-specific collection logic lives under JAXRSSMJAXDiDr. This script
patches CarDreamer in-process so the legacy task config exposes six cameras
without permanently modifying car_dreamer/configs/*.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dreamerv3") not in sys.path:
    sys.path.insert(0, str(ROOT / "dreamerv3"))

from JAXRSSMJAXDiDr.vad_carla import install_vad_carla_patches


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", default="carla_roundabout")
    parser.add_argument("--vad_camera_width", type=int, default=1600)
    parser.add_argument("--vad_camera_height", type=int, default=900)
    parser.add_argument("--vad_camera_fov", type=float, default=70.0)
    parser.add_argument("--vad_sensor_tick", type=float, default=0.1)
    parser.add_argument("--vad_no_birdeye", action="store_true")
    args, rest = parser.parse_known_args(argv)
    return args, rest


def main(argv=None) -> None:
    args, rest = parse_args(argv)
    install_vad_carla_patches(
        task=args.task,
        width=args.vad_camera_width,
        height=args.vad_camera_height,
        fov=args.vad_camera_fov,
        sensor_tick=args.vad_sensor_tick,
        include_birdeye=not args.vad_no_birdeye,
    )
    if "--task" not in rest:
        rest = ["--task", args.task, *rest]

    from dreamerv3 import collect_polyplanner

    print("[vad_collect] Enabled VAD camera keys in nuScenes order:")
    print("[vad_collect] CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT")
    print(
        "[vad_collect] camera="
        f"{args.vad_camera_width}x{args.vad_camera_height} fov={args.vad_camera_fov} tick={args.vad_sensor_tick}"
    )
    collect_polyplanner.main(rest)


if __name__ == "__main__":
    main()
