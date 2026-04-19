#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_root = Path(os.environ.get("DATA_ROOT", str(repo_root / "data")))
    parser = argparse.ArgumentParser(
        description="Prepare canonical full-sequence KITTI depth-selection data for MeMix video-depth evaluation."
    )
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Base data directory containing kitti/.",
    )
    parser.add_argument(
        "--kitti-root",
        default=None,
        help="Override KITTI root (default: <data-root>/kitti).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional output root pointing at the val_selection_cropped layout. "
            "Defaults to <kitti-root>/depth_selection/val_selection_cropped."
        ),
    )
    return parser.parse_args()


def resolve_root(args: argparse.Namespace) -> Path:
    if args.kitti_root:
        return Path(args.kitti_root)
    return Path(args.data_root) / "kitti"


def find_annotated_root(kitti_root: Path) -> Path:
    candidates = [kitti_root / "data_depth_annotated", kitti_root]
    for candidate in candidates:
        if (candidate / "val").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find KITTI annotated depth data. "
        "Expected 'val/' under kitti/ or kitti/data_depth_annotated/."
    )


def rebuild_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_frame(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main() -> None:
    args = parse_args()
    kitti_root = resolve_root(args)
    if not kitti_root.exists():
        raise FileNotFoundError(f"KITTI root not found: {kitti_root}")

    annotated_root = find_annotated_root(kitti_root)
    depth_dirs = sorted((annotated_root / "val").glob("*/proj_depth/groundtruth/image_02"))
    if not depth_dirs:
        raise FileNotFoundError(
            f"No KITTI depth maps found under: {annotated_root}/val/*/proj_depth/groundtruth/image_02"
        )

    output_root = (
        Path(args.output_root)
        if args.output_root
        else kitti_root / "depth_selection" / "val_selection_cropped"
    )
    depth_out_root = output_root / "groundtruth_depth_gathered_full"
    image_out_root = output_root / "image_gathered_full"

    for depth_dir in depth_dirs:
        drive = depth_dir.parents[2].name
        depth_files = sorted(depth_dir.glob("*.png"))
        new_depth_dir = depth_out_root / f"{drive}_02"
        new_image_dir = image_out_root / f"{drive}_02"
        rebuild_dir(new_depth_dir)
        rebuild_dir(new_image_dir)

        linked = 0
        for depth_file in depth_files:
            date = "_".join(drive.split("_")[:3])
            image_file = kitti_root / date / drive / "image_02" / "data" / depth_file.name
            if not image_file.exists():
                print(f"Skipping missing image file: {image_file}")
                continue
            symlink_frame(depth_file, new_depth_dir / depth_file.name)
            symlink_frame(image_file, new_image_dir / image_file.name)
            linked += 1

        print(f"{drive}: linked {linked} annotated RGB/depth pairs")

    print(f"Prepared canonical KITTI eval data at: {output_root}")


if __name__ == "__main__":
    main()
