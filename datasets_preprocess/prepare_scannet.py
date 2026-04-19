#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_root = Path(os.environ.get("DATA_ROOT", str(repo_root / "data")))
    parser = argparse.ArgumentParser(
        description="Prepare canonical full-sequence ScanNet data for MeMix relpose evaluation."
    )
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Base data directory containing scannetv2/.",
    )
    parser.add_argument(
        "--scannet-root",
        default=None,
        help="Override ScanNet root (default: <data-root>/scannetv2).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root. Defaults to the input ScanNet root.",
    )
    return parser.parse_args()


def resolve_root(args: argparse.Namespace) -> Path:
    if args.scannet_root:
        return Path(args.scannet_root)
    return Path(args.data_root) / "scannetv2"


def numeric_stem(path: Path) -> int:
    return int(path.stem)


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
    root = resolve_root(args)
    if not root.is_dir():
        raise FileNotFoundError(f"ScanNet root not found: {root}")

    output_root = Path(args.output_root) if args.output_root else root
    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not scene_dirs:
        raise FileNotFoundError(f"No ScanNet scenes found under: {root}")

    for scene_dir in scene_dirs:
        color_files = sorted((scene_dir / "color").glob("*.jpg"), key=numeric_stem)
        depth_files = sorted((scene_dir / "depth").glob("*.png"), key=numeric_stem)
        pose_files = sorted((scene_dir / "pose").glob("*.txt"), key=numeric_stem)
        total_frames = min(len(color_files), len(depth_files), len(pose_files))
        if total_frames == 0:
            print(f"Skipping {scene_dir.name}: missing color/depth/pose files")
            continue

        scene_out_root = output_root / scene_dir.name
        color_out = scene_out_root / "color_full"
        depth_out = scene_out_root / "depth_full"
        rebuild_dir(color_out)
        rebuild_dir(depth_out)

        pose_rows = []
        for out_idx in range(total_frames):
            symlink_frame(
                color_files[out_idx], color_out / f"frame_{out_idx:06d}.jpg"
            )
            symlink_frame(
                depth_files[out_idx], depth_out / f"frame_{out_idx:06d}.png"
            )
            pose = np.loadtxt(pose_files[out_idx]).reshape(-1)
            pose_rows.append(" ".join(map(str, pose.tolist())))

        pose_out = scene_out_root / "pose_full.txt"
        pose_out.write_text(
            "\n".join(pose_rows) + ("\n" if pose_rows else ""),
            encoding="utf-8",
        )
        print(f"{scene_dir.name}: total={total_frames} -> {color_out} and {pose_out.name}")


if __name__ == "__main__":
    main()
