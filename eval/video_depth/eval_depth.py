import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from eval.protocol_config import apply_runtime_slice, resolve_runtime_slice
from eval.video_depth.metadata import (
    bonn_depth_dir,
    dataset_metadata,
    resolve_dataset_metadata,
)
from eval.video_depth.tools import depth_evaluation, group_by_directory


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="nyu", choices=list(dataset_metadata.keys())
    )
    parser.add_argument(
        "--eval_protocol",
        type=str,
        default="short",
        choices=("short", "long"),
        help="Use the short paper subset or the long-sequence protocol.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=0,
        help="Sequence length for the selected protocol. Short mode uses fixed paper lengths.",
    )
    parser.add_argument(
        "--align",
        type=str,
        default="scale&shift",
        choices=["scale&shift", "scale", "metric"],
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=None,
        help="Optional starting frame index before runtime slicing.",
    )
    parser.add_argument(
        "--pose_eval_stride",
        type=int,
        default=None,
        help="Runtime frame stride after start_frame.",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=None,
        help="Runtime frame cap after start_frame and stride (0 means no limit).",
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=0,
        help="skip sequences with fewer frames (0 means no limit)",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="optional sequence subset for sequence datasets like Bonn/Sintel",
    )
    return parser


def read_sintel_depth(filename):
    tag_float = 202021.25
    with open(filename, "rb") as handle:
        check = np.fromfile(handle, dtype=np.float32, count=1)[0]
        assert (
            check == tag_float
        ), f"depth_read:: wrong tag in {filename}: expected {tag_float}, got {check}"
        width = np.fromfile(handle, dtype=np.int32, count=1)[0]
        height = np.fromfile(handle, dtype=np.int32, count=1)[0]
        size = width * height
        assert 1 < size < 100000000, (
            f"depth_read:: invalid size for {filename}: width={width}, height={height}"
        )
        depth = np.fromfile(handle, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def read_bonn_depth(filename):
    depth_png = np.asarray(Image.open(filename))
    assert np.max(depth_png) > 255
    depth = depth_png.astype(np.float64) / 5000.0
    depth[depth_png == 0] = -1.0
    return depth


def read_kitti_depth(filename):
    depth_png = np.array(Image.open(filename), dtype=int)
    assert np.max(depth_png) > 255
    depth = depth_png.astype(float) / 256.0
    depth[depth_png == 0] = -1.0
    return depth


def resolve_runtime_args(args):
    runtime_slice = resolve_runtime_slice(
        "video_depth",
        args.eval_dataset,
        args.eval_protocol,
        args.seq_len,
        args.start_frame,
        args.pose_eval_stride,
        args.max_views,
    )
    args.seq_len = runtime_slice["seq_len"] or 0
    args.start_frame = runtime_slice["start_frame"]
    args.pose_eval_stride = runtime_slice["pose_eval_stride"]
    args.max_views = runtime_slice["max_views"]
    return args


def collect_prediction_groups(output_dir):
    pred_paths = sorted(glob.glob(f"{output_dir}/*/frame_*.npy"))
    return group_by_directory(pred_paths)


def collect_sintel_gt(metadata, seq_list):
    depth_root = Path(metadata["depth_root"])
    return {
        seq: sorted(glob.glob(str(depth_root / seq / "*.dpt")))
        for seq in seq_list
    }


def collect_bonn_gt(metadata, seq_list):
    depth_root = metadata["depth_root"]
    return {
        seq: sorted(glob.glob(os.path.join(bonn_depth_dir(depth_root, seq), "*.png")))
        for seq in seq_list
    }


def collect_kitti_gt(metadata, seq_list=None):
    all_depth = sorted(glob.glob(str(Path(metadata["depth_root"]) / "*" / "*.png")))
    grouped = {key: sorted(value) for key, value in group_by_directory(all_depth).items()}
    if seq_list is None:
        return grouped
    return {seq: grouped.get(seq, []) for seq in seq_list}


def evaluate_one_sequence(pd_paths, gt_paths, depth_reader, args, *, max_depth, post_clip_max=None):
    gt_paths = apply_runtime_slice(
        gt_paths,
        start_frame=args.start_frame,
        pose_eval_stride=args.pose_eval_stride,
        max_views=args.max_views,
    )
    effective_len = min(len(pd_paths), len(gt_paths))
    if args.min_views and effective_len < args.min_views:
        return None
    if effective_len == 0:
        return None
    if len(pd_paths) != len(gt_paths):
        print(
            f"Aligning sequence lengths: pred {len(pd_paths)} vs gt {len(gt_paths)} -> {effective_len}"
        )
    pd_paths = pd_paths[:effective_len]
    gt_paths = gt_paths[:effective_len]

    gt_depth = np.stack([depth_reader(path) for path in gt_paths], axis=0)
    pr_depth = np.stack(
        [
            cv2.resize(
                np.load(path),
                (gt_depth.shape[2], gt_depth.shape[1]),
                interpolation=cv2.INTER_CUBIC,
            )
            for path in pd_paths
        ],
        axis=0,
    )

    if args.align == "scale&shift":
        depth_results, _, _, _ = depth_evaluation(
            pr_depth,
            gt_depth,
            max_depth=max_depth,
            align_with_lad2=True,
            use_gpu=True,
            post_clip_max=post_clip_max,
        )
    elif args.align == "scale":
        depth_results, _, _, _ = depth_evaluation(
            pr_depth,
            gt_depth,
            max_depth=max_depth,
            align_with_scale=True,
            use_gpu=True,
            post_clip_max=post_clip_max,
        )
    else:
        depth_results, _, _, _ = depth_evaluation(
            pr_depth,
            gt_depth,
            max_depth=max_depth,
            metric_scale=True,
            use_gpu=True,
            post_clip_max=post_clip_max,
        )
    return depth_results


def finalize_metrics(gathered_depth_metrics, output_dir, align, dataset_name):
    if not gathered_depth_metrics:
        print(
            f"No valid {dataset_name} depth metrics collected. "
            "Check predictions and runtime slicing constraints."
        )
        return

    average_metrics = {
        key: np.average(
            [metrics[key] for metrics in gathered_depth_metrics],
            weights=[metrics["valid_pixels"] for metrics in gathered_depth_metrics],
        )
        for key in gathered_depth_metrics[0].keys()
        if key != "valid_pixels"
    }
    depth_log_path = f"{output_dir}/result_{align}.json"
    print("Average depth evaluation metrics:", average_metrics)
    with open(depth_log_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(average_metrics))


def main(args):
    args = resolve_runtime_args(args)
    metadata = resolve_dataset_metadata(
        args.eval_dataset, args.eval_protocol, args.seq_len
    )

    if args.eval_dataset == "sintel":
        seq_list = args.seq_list if args.seq_list is not None else metadata.get("seq_list", [])
        grouped_pred_depth = collect_prediction_groups(args.output_dir)
        grouped_gt_depth = collect_sintel_gt(metadata, seq_list)
        gathered_depth_metrics = []
        for seq in tqdm(seq_list):
            pd_paths = sorted(grouped_pred_depth.get(seq, []))
            gt_paths = grouped_gt_depth.get(seq, [])
            if not pd_paths:
                print(f"Missing predictions for {seq}, skipping.")
                continue
            depth_results = evaluate_one_sequence(
                pd_paths,
                gt_paths,
                read_sintel_depth,
                args,
                max_depth=70,
                post_clip_max=70,
            )
            if depth_results is None:
                print(f"Skipping {seq}: empty/short sequence after runtime slicing.")
                continue
            gathered_depth_metrics.append(depth_results)

        finalize_metrics(gathered_depth_metrics, args.output_dir, args.align, "Sintel")
        return

    if args.eval_dataset == "bonn":
        seq_list = args.seq_list if args.seq_list is not None else metadata.get("seq_list", [])
        grouped_pred_depth = collect_prediction_groups(args.output_dir)
        grouped_gt_depth = collect_bonn_gt(metadata, seq_list)
        gathered_depth_metrics = []
        for seq in tqdm(seq_list):
            pd_paths = sorted(grouped_pred_depth.get(seq, []))
            gt_paths = grouped_gt_depth.get(seq, [])
            if not pd_paths:
                print(f"Missing predictions for {seq}, skipping.")
                continue
            depth_results = evaluate_one_sequence(
                pd_paths,
                gt_paths,
                read_bonn_depth,
                args,
                max_depth=70,
            )
            if depth_results is None:
                print(f"Skipping {seq}: empty/short sequence after runtime slicing.")
                continue
            gathered_depth_metrics.append(depth_results)

        finalize_metrics(gathered_depth_metrics, args.output_dir, args.align, "Bonn")
        return

    if args.eval_dataset == "kitti":
        seq_list = args.seq_list
        grouped_pred_depth = collect_prediction_groups(args.output_dir)
        grouped_gt_depth = collect_kitti_gt(metadata, seq_list)
        sequence_keys = (
            sorted(seq_list) if seq_list is not None else sorted(grouped_gt_depth.keys())
        )
        gathered_depth_metrics = []
        for seq in tqdm(sequence_keys):
            pd_paths = sorted(grouped_pred_depth.get(seq, []))
            gt_paths = grouped_gt_depth.get(seq, [])
            if not pd_paths:
                print(f"Missing predictions for {seq}, skipping.")
                continue
            depth_results = evaluate_one_sequence(
                pd_paths,
                gt_paths,
                read_kitti_depth,
                args,
                max_depth=None,
            )
            if depth_results is None:
                print(f"Skipping {seq}: empty/short sequence after runtime slicing.")
                continue
            gathered_depth_metrics.append(depth_results)

        finalize_metrics(gathered_depth_metrics, args.output_dir, args.align, "KITTI")
        return

    raise NotImplementedError(
        f"Depth evaluation is only implemented for kitti/bonn/sintel, got {args.eval_dataset}."
    )


if __name__ == "__main__":
    parsed_args = get_args_parser().parse_args()
    main(parsed_args)
