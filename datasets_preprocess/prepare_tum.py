#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def read_file_list(filename: Path) -> dict[float, list[str]]:
    data = filename.read_text(encoding="utf-8")
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    entries = [
        [value.strip() for value in line.split(" ") if value.strip() != ""]
        for line in lines
        if line and not line.startswith("#")
    ]
    return {float(entry[0]): entry[1:] for entry in entries if len(entry) > 1}


def associate(
    first_list: dict[float, list[str]],
    second_list: dict[float, list[str]],
    offset: float,
    max_difference: float,
) -> list[tuple[float, float]]:
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    potential = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential.sort()
    matches = []
    for _, a, b in potential:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_root = Path(os.environ.get("DATA_ROOT", str(repo_root / "data")))
    parser = argparse.ArgumentParser(
        description="Prepare canonical full-sequence TUM data for MeMix relpose evaluation."
    )
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Base data directory containing tum/.",
    )
    parser.add_argument(
        "--tum-root",
        default=None,
        help="Override TUM root (default: <data-root>/tum).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root. Defaults to the input TUM root.",
    )
    parser.add_argument(
        "--max-diff",
        type=float,
        default=0.02,
        help="Max timestamp difference when associating rgb and groundtruth.",
    )
    return parser.parse_args()


def resolve_root(args: argparse.Namespace) -> Path:
    if args.tum_root:
        return Path(args.tum_root)
    return Path(args.data_root) / "tum"


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
    tum_root = resolve_root(args)
    if not tum_root.is_dir():
        raise FileNotFoundError(f"TUM root not found: {tum_root}")

    output_root = Path(args.output_root) if args.output_root else tum_root
    seq_dirs = sorted(path for path in tum_root.iterdir() if path.is_dir())
    if not seq_dirs:
        raise FileNotFoundError(f"No TUM sequences found under: {tum_root}")

    for seq_dir in seq_dirs:
        rgb_txt = seq_dir / "rgb.txt"
        gt_txt = seq_dir / "groundtruth.txt"
        if not rgb_txt.exists() or not gt_txt.exists():
            print(f"Skipping {seq_dir.name}: missing rgb.txt or groundtruth.txt")
            continue

        first_list = read_file_list(rgb_txt)
        second_list = read_file_list(gt_txt)
        matches = associate(first_list, second_list, 0.0, args.max_diff)
        if not matches:
            print(f"Skipping {seq_dir.name}: no associated rgb/pose pairs found")
            continue

        seq_out_root = output_root / seq_dir.name
        rgb_out = seq_out_root / "rgb_full"
        rebuild_dir(rgb_out)

        gt_rows = []
        for out_idx, (rgb_stamp, gt_stamp) in enumerate(matches):
            src_frame = seq_dir / first_list[rgb_stamp][0]
            suffix = src_frame.suffix or ".png"
            dst_frame = rgb_out / f"frame_{out_idx:06d}{suffix}"
            symlink_frame(src_frame, dst_frame)
            gt_rows.append([gt_stamp] + second_list[gt_stamp])

        gt_out = seq_out_root / "groundtruth_full.txt"
        with gt_out.open("w", encoding="utf-8") as handle:
            for pose in gt_rows:
                handle.write(" ".join(map(str, pose)) + "\n")

        print(
            f"{seq_dir.name}: matched={len(matches)} -> {rgb_out} and {gt_out.name}"
        )


if __name__ == "__main__":
    main()
