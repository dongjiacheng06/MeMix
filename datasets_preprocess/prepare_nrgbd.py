#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_root = Path(os.environ.get("DATA_ROOT", str(repo_root / "data")))
    parser = argparse.ArgumentParser(
        description="Validate Neural RGB-D layout for MeMix mv_recon evaluation."
    )
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Base data directory containing neural_rgbd/.",
    )
    parser.add_argument(
        "--nrgbd-root",
        default=None,
        help="Override Neural RGB-D root (default: <data-root>/neural_rgbd).",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Optional single scene to validate.",
    )
    return parser.parse_args()


def resolve_root(args: argparse.Namespace) -> Path:
    if args.nrgbd_root:
        return Path(args.nrgbd_root)
    return Path(args.data_root) / "neural_rgbd"


def main() -> None:
    args = parse_args()
    root = resolve_root(args)
    if not root.is_dir():
        raise FileNotFoundError(f"NRGBD root not found: {root}")

    if args.scene:
        scene_dirs = [root / args.scene]
    else:
        scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not scene_dirs:
        raise FileNotFoundError(f"No NRGBD scenes found under: {root}")

    validated = 0
    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"NRGBD scene not found: {scene_dir}")
        images_dir = scene_dir / "images"
        depth_dir = scene_dir / "depth"
        poses_path = scene_dir / "poses.txt"
        focal_path = scene_dir / "focal.txt"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing images dir: {images_dir}")
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Missing depth dir: {depth_dir}")
        if not poses_path.exists():
            raise FileNotFoundError(f"Missing poses.txt: {poses_path}")

        num_images = len(list(images_dir.glob("img*.png")))
        num_depths = len(list(depth_dir.glob("depth*.png")))
        has_focal = focal_path.exists()
        print(
            f"{scene_dir.name}: images={num_images} depth={num_depths} "
            f"poses={'yes'} focal={'yes' if has_focal else 'no'}"
        )
        if num_images == 0 or num_depths == 0:
            raise RuntimeError(f"Scene is empty or incomplete: {scene_dir}")
        validated += 1

    print(f"Validated {validated} NRGBD scene(s) under {root}")


if __name__ == "__main__":
    main()
