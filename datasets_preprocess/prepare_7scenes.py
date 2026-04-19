#!/usr/bin/env python3
"""
Register Kinect depth maps to the RGB camera for 7-Scenes.

Adapted from the CUT3R / visloc preprocessing script, but exposed here with a
small CLI so the release tree stays self-contained.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np

FOCAL_LENGTH = 525.0
DEPTH_FOCAL_LENGTH = 585.0
IMG_W = 640
IMG_H = 480

D_TO_RGB = np.array(
    [
        [
            9.9996518012567637e-01,
            2.6765126468950343e-03,
            -7.9041012313000904e-03,
            -2.5558943178152542e-02,
        ],
        [
            -2.7409311281316700e-03,
            9.9996302803027592e-01,
            -8.1504520778013286e-03,
            1.0109636268061706e-04,
        ],
        [
            7.8819942130445332e-03,
            8.1718328771890631e-03,
            9.9993554558014031e-01,
            2.0318321729487039e-03,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

DEFAULT_SCENES = [
    "chess",
    "fire",
    "heads",
    "office",
    "pumpkin",
    "redkitchen",
    "stairs",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_root = Path(os.environ.get("DATA_ROOT", str(repo_root / "data")))
    parser = argparse.ArgumentParser(
        description="Prepare 7-Scenes by generating *.depth.proj.png files."
    )
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Base data directory containing 7scenes/.",
    )
    parser.add_argument(
        "--seven-scenes-root",
        default=None,
        help="Override 7-Scenes root (default: <data-root>/7scenes).",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=7,
        help="Number of scenes to process in parallel.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help="Subset of 7-Scenes scene names to process.",
    )
    return parser.parse_args()


def resolve_root(args: argparse.Namespace) -> Path:
    if args.seven_scenes_root:
        return Path(args.seven_scenes_root)
    return Path(args.data_root) / "7scenes"


def process_scene(scene_dir: Path) -> None:
    try:
        from skimage import io
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "prepare_7scenes.py requires scikit-image at runtime. "
            "Install it from requirements.txt before running preprocessing."
        ) from exc

    def process_split(split_file: str) -> None:
        split_path = scene_dir / split_file
        if not split_path.exists():
            print(f"Warning: {split_path} not found, skipping")
            return

        split = split_path.read_text().splitlines()
        seqs = ["seq-" + line.strip()[8:].zfill(2) for line in split]

        for seq in seqs:
            seq_dir = scene_dir / seq
            if not seq_dir.is_dir():
                print(f"Warning: {seq_dir} not found, skipping")
                continue

            depth_files = sorted(
                f
                for f in os.listdir(seq_dir)
                if f.endswith("depth.png") and not f.endswith("depth.proj.png")
            )
            for idx, depth_name in enumerate(depth_files):
                if idx % 500 == 0:
                    print(f"  {scene_dir.name}/{seq}: {idx}/{len(depth_files)}")

                depth = io.imread(seq_dir / depth_name).astype(np.float32) / 1000.0
                d_h, d_w = depth.shape[:2]

                eye_coords = np.zeros((4, d_h, d_w), dtype=np.float32)
                eye_coords[0] = 0.5 + np.dstack([np.arange(0, d_w)] * d_h)[0].T
                eye_coords[1] = 0.5 + np.dstack([np.arange(0, d_h)] * d_w)[0]

                eye_coords = eye_coords.reshape(4, -1)
                depth_flat = depth.reshape(-1)

                mask = (depth_flat > 0) & (depth_flat < 100)
                eye_coords = eye_coords[:, mask]
                depth_flat = depth_flat[mask]

                eye_coords[0] -= d_w / 2
                eye_coords[1] -= d_h / 2
                eye_coords[0:2] /= DEPTH_FOCAL_LENGTH
                eye_coords[0] *= depth_flat
                eye_coords[1] *= depth_flat
                eye_coords[2] = depth_flat
                eye_coords[3] = 1.0

                eye_coords = np.matmul(D_TO_RGB, eye_coords)

                depth_proj = eye_coords[2]
                eye_coords[0] /= depth_proj
                eye_coords[1] /= depth_proj
                eye_coords[0:2] *= FOCAL_LENGTH
                eye_coords[0] += IMG_W / 2
                eye_coords[1] += IMG_H / 2

                registered_depth = np.ones((IMG_H, IMG_W), dtype=np.float32) * 2e3
                for point_idx in range(eye_coords.shape[1]):
                    x = round(float(eye_coords[0, point_idx]))
                    y = round(float(eye_coords[1, point_idx]))
                    z = float(eye_coords[2, point_idx])
                    if 0 <= x < IMG_W and 0 <= y < IMG_H:
                        registered_depth[y, x] = min(registered_depth[y, x], z)

                registered_depth[registered_depth > 1e3] = 0
                registered_depth = (1000.0 * registered_depth).astype(np.uint16)
                out_path = seq_dir / depth_name.replace("depth.png", "depth.proj.png")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    io.imsave(out_path, registered_depth)

    process_split("TrainSplit.txt")
    process_split("TestSplit.txt")


def main() -> None:
    args = parse_args()
    root = resolve_root(args)
    if not root.is_dir():
        raise FileNotFoundError(f"7-Scenes root not found: {root}")

    scene_paths = [root / name for name in args.scenes]
    existing = [path for path in scene_paths if path.is_dir()]
    if not existing:
        raise FileNotFoundError(f"No requested 7-Scenes scenes found under: {root}")

    try:
        from joblib import Parallel, delayed
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "prepare_7scenes.py requires joblib at runtime. "
            "Install it from requirements.txt before running preprocessing."
        ) from exc

    print(f"Processing {len(existing)} scenes from {root}")
    Parallel(n_jobs=args.num_jobs)(delayed(process_scene)(scene_path) for scene_path in existing)
    print("Done!")


if __name__ == "__main__":
    main()
