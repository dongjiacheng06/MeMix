#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_root = Path(os.environ.get("DATA_ROOT", str(repo_root / "data")))
    parser = argparse.ArgumentParser(
        description="Validate Sintel layout for MeMix relpose/video-depth evaluation."
    )
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Base data directory containing sintel/.",
    )
    parser.add_argument(
        "--sintel-root",
        default=None,
        help="Override Sintel root (default: <data-root>/sintel/training).",
    )
    return parser.parse_args()


def resolve_root(args: argparse.Namespace) -> Path:
    if args.sintel_root:
        return Path(args.sintel_root)
    return Path(args.data_root) / "sintel" / "training"


def main() -> None:
    args = parse_args()
    root = resolve_root(args)
    final_root = root / "final"
    depth_root = root / "depth"
    cam_root = root / "camdata_left"
    for required in (final_root, depth_root, cam_root):
        if not required.is_dir():
            raise FileNotFoundError(f"Missing Sintel directory: {required}")

    seq_dirs = sorted(path for path in final_root.iterdir() if path.is_dir())
    if not seq_dirs:
        raise FileNotFoundError(f"No Sintel sequences found under: {final_root}")

    for seq_dir in seq_dirs:
        seq = seq_dir.name
        depth_dir = depth_root / seq
        cam_dir = cam_root / seq
        if not depth_dir.is_dir() or not cam_dir.is_dir():
            raise FileNotFoundError(
                f"Missing Sintel companion dirs for {seq}: depth={depth_dir} cam={cam_dir}"
            )
        img_count = len(list(seq_dir.glob("*.png")))
        depth_count = len(list(depth_dir.glob("*.dpt")))
        cam_count = len(list(cam_dir.glob("*.cam")))
        print(f"{seq}: rgb={img_count} depth={depth_count} cam={cam_count}")
        if img_count == 0 or depth_count == 0 or cam_count == 0:
            raise RuntimeError(f"Sintel sequence is incomplete: {seq}")

    print(f"Validated {len(seq_dirs)} Sintel sequences under {root}")


if __name__ == "__main__":
    main()
