from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(_REPO_ROOT / "data")))

SINTEL_SEQ_LIST = [
    "alley_2",
    "ambush_4",
    "ambush_5",
    "ambush_6",
    "cave_2",
    "cave_4",
    "market_2",
    "market_5",
    "market_6",
    "shaman_3",
    "sleeping_1",
    "sleeping_2",
    "temple_2",
    "temple_3",
]

BONN_SEQ_LIST = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]

SHORT_SEQ_LENGTHS = {
    "tum": 90,
    "scannet": 90,
    "kitti": 110,
    "bonn": 110,
    "sintel": 50,
}

LONG_SEQ_LENGTHS = {
    "tum": [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "scannet": [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    "kitti": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "bonn": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
}

RUNTIME_PRESETS = {
    "relpose": {
        "tum": {
            "short": {"start_frame": 0, "pose_eval_stride": 3, "max_views": 90},
            "long": {"start_frame": 0, "pose_eval_stride": 1},
        },
        "scannet": {
            "short": {"start_frame": 0, "pose_eval_stride": 3, "max_views": 90},
            "long": {"start_frame": 0, "pose_eval_stride": 1},
        },
        "sintel": {
            "short": {"start_frame": 0, "pose_eval_stride": 1, "max_views": 50},
            "long": {"start_frame": 0, "pose_eval_stride": 1},
        },
    },
    "video_depth": {
        "kitti": {
            "short": {"start_frame": 0, "pose_eval_stride": 1, "max_views": 110},
            "long": {"start_frame": 0, "pose_eval_stride": 1},
        },
        "bonn": {
            "short": {"start_frame": 30, "pose_eval_stride": 1, "max_views": 110},
            "long": {"start_frame": 30, "pose_eval_stride": 1},
        },
        "sintel": {
            "short": {"start_frame": 0, "pose_eval_stride": 1, "max_views": 50},
            "long": {"start_frame": 0, "pose_eval_stride": 1},
        },
    },
}


def normalize_seq_len(seq_len: int | None) -> int | None:
    if seq_len is None:
        return None
    seq_len = int(seq_len)
    if seq_len <= 0:
        return None
    return seq_len


def normalize_runtime_int(value: int | None, *, allow_zero: bool) -> int | None:
    if value is None:
        return None
    value = int(value)
    minimum = 0 if allow_zero else 1
    if value < minimum:
        label = ">= 0" if allow_zero else ">= 1"
        raise ValueError(f"Expected runtime parameter {label}, got {value}.")
    return value


def apply_runtime_slice(
    items,
    *,
    start_frame: int = 0,
    pose_eval_stride: int = 1,
    max_views: int = 0,
):
    start_frame = normalize_runtime_int(start_frame, allow_zero=True) or 0
    pose_eval_stride = normalize_runtime_int(pose_eval_stride, allow_zero=False) or 1
    max_views = normalize_runtime_int(max_views, allow_zero=True) or 0

    sliced = list(items)[start_frame::pose_eval_stride]
    if max_views > 0:
        sliced = sliced[:max_views]
    return sliced


def resolve_runtime_slice(
    task: str,
    eval_dataset: str,
    eval_protocol: str,
    seq_len: int | None,
    start_frame: int | None = None,
    pose_eval_stride: int | None = None,
    max_views: int | None = None,
):
    if eval_protocol not in ("short", "long"):
        raise ValueError(f"Unknown eval protocol: {eval_protocol}")

    seq_len = normalize_seq_len(seq_len)
    start_frame = normalize_runtime_int(start_frame, allow_zero=True)
    pose_eval_stride = normalize_runtime_int(pose_eval_stride, allow_zero=False)
    max_views = normalize_runtime_int(max_views, allow_zero=True)

    preset = RUNTIME_PRESETS.get(task, {}).get(eval_dataset, {}).get(eval_protocol, {})
    default_start = int(preset.get("start_frame", 0))
    default_stride = int(preset.get("pose_eval_stride", 1))

    if eval_protocol == "short":
        default_max_views = preset.get("max_views")
        if default_max_views is None:
            default_max_views = SHORT_SEQ_LENGTHS.get(eval_dataset, 0)
        if (
            seq_len is not None
            and seq_len != default_max_views
            and start_frame is None
            and pose_eval_stride is None
            and max_views is None
        ):
            raise ValueError(
                f"Short protocol for '{eval_dataset}' expects seq_len={default_max_views}; "
                f"got {seq_len}. Use --eval_protocol long or explicit runtime slicing."
            )
    else:
        default_max_views = seq_len
        if default_max_views is None and max_views is None:
            if eval_dataset == "sintel":
                raise ValueError(
                    "Long protocol for 'sintel' requires --seq_len or --max_views."
                )
            if eval_dataset in LONG_SEQ_LENGTHS:
                raise ValueError(
                    f"Long protocol for '{eval_dataset}' requires --seq_len or --max_views."
                )
            raise ValueError(
                f"Long protocol is not supported for dataset '{eval_dataset}'."
            )

    resolved = {
        "seq_len": seq_len,
        "start_frame": default_start if start_frame is None else start_frame,
        "pose_eval_stride": (
            default_stride if pose_eval_stride is None else pose_eval_stride
        ),
        "max_views": default_max_views if max_views is None else max_views,
    }
    return resolved
