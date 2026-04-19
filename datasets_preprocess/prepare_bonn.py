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
        description="Prepare canonical full-sequence Bonn RGB-D data for MeMix video-depth evaluation."
    )
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Base data directory containing bonn/.",
    )
    parser.add_argument(
        "--bonn-root",
        default=None,
        help="Override Bonn root (default: <data-root>/bonn/rgbd_bonn_dataset).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root. Defaults to the input Bonn root.",
    )
    return parser.parse_args()


def resolve_root(args: argparse.Namespace) -> Path:
    if args.bonn_root:
        return Path(args.bonn_root)
    return Path(args.data_root) / "bonn" / "rgbd_bonn_dataset"


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
        raise FileNotFoundError(f"Bonn root not found: {root}")

    output_root = Path(args.output_root) if args.output_root else root
    seq_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not seq_dirs:
        raise FileNotFoundError(f"No Bonn sequences found under: {root}")

    for seq_dir in seq_dirs:
        rgb_files_all = sorted((seq_dir / "rgb").glob("*.png"))
        depth_files_all = sorted((seq_dir / "depth").glob("*.png"))
        gt_path = seq_dir / "groundtruth.txt"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing Bonn groundtruth: {gt_path}")

        gt = np.loadtxt(gt_path)
        if gt.ndim == 1:
            gt = gt[None, :]

        total_frames = min(len(rgb_files_all), len(depth_files_all), int(gt.shape[0]))
        if total_frames == 0:
            print(f"Skipping {seq_dir.name}: no complete RGB/depth/pose triplets")
            continue

        seq_out_dir = output_root / seq_dir.name
        rgb_out = seq_out_dir / "rgb_full"
        depth_out = seq_out_dir / "depth_full"
        rebuild_dir(rgb_out)
        rebuild_dir(depth_out)

        for out_idx in range(total_frames):
            symlink_frame(rgb_files_all[out_idx], rgb_out / f"frame_{out_idx:06d}.png")
            symlink_frame(
                depth_files_all[out_idx], depth_out / f"frame_{out_idx:06d}.png"
            )

        np.savetxt(seq_out_dir / "groundtruth_full.txt", gt[:total_frames])
        print(
            f"{seq_dir.name}: rgb={total_frames} depth={total_frames} poses={total_frames}"
        )


if __name__ == "__main__":
    main()
