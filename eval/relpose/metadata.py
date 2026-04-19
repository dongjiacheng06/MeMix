import glob
import os

from tqdm import tqdm

from eval.protocol_config import BONN_SEQ_LIST, DATA_ROOT, SINTEL_SEQ_LIST


dataset_metadata = {
    "davis": {
        "img_path": str(DATA_ROOT / "davis" / "DAVIS" / "JPEGImages" / "480p"),
        "mask_path": str(DATA_ROOT / "davis" / "DAVIS" / "masked_images" / "480p"),
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: os.path.join(mask_path, seq),
        "skip_condition": None,
        "process_func": None,
    },
    "kitti": {
        "img_path": str(
            DATA_ROOT
            / "kitti"
            / "depth_selection"
            / "val_selection_cropped"
            / "image_gathered_full"
        ),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: None,
        "traj_format": None,
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_kitti(args, img_path),
    },
    "bonn": {
        "img_path": str(DATA_ROOT / "bonn" / "rgbd_bonn_dataset"),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "rgb_full"
        ),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, f"rgbd_bonn_{seq}", "groundtruth_full.txt"
        ),
        "traj_format": "tum",
        "seq_list": BONN_SEQ_LIST,
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_bonn(args, img_path),
    },
    "nyu": {
        "img_path": str(DATA_ROOT / "nyu-v2" / "val" / "nyu_images"),
        "mask_path": None,
        "process_func": lambda args, img_path: process_nyu(args, img_path),
    },
    "scannet": {
        "img_path": str(DATA_ROOT / "scannetv2"),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "color_full"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "pose_full.txt"
        ),
        "traj_format": "replica",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_scannet(args, img_path),
    },
    "tum": {
        "img_path": str(DATA_ROOT / "tum"),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq, "rgb_full"),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(
            img_path, seq, "groundtruth_full.txt"
        ),
        "traj_format": "tum",
        "seq_list": None,
        "full_seq": True,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": None,
    },
    "sintel": {
        "img_path": str(DATA_ROOT / "sintel" / "training" / "final"),
        "anno_path": str(DATA_ROOT / "sintel" / "training" / "camdata_left"),
        "mask_path": None,
        "dir_path_func": lambda img_path, seq: os.path.join(img_path, seq),
        "gt_traj_func": lambda img_path, anno_path, seq: os.path.join(anno_path, seq),
        "traj_format": None,
        "seq_list": SINTEL_SEQ_LIST,
        "full_seq": False,
        "mask_path_seq_func": lambda mask_path, seq: None,
        "skip_condition": None,
        "process_func": lambda args, img_path: process_sintel(args, img_path),
    },
}


def resolve_dataset_metadata(eval_dataset, eval_protocol="short", seq_len=None):
    if eval_dataset not in dataset_metadata:
        raise KeyError(f"Unknown relpose dataset: {eval_dataset}")

    metadata = dict(dataset_metadata[eval_dataset])
    metadata["eval_protocol"] = eval_protocol
    metadata["seq_len"] = seq_len

    if eval_dataset == "sintel":
        metadata["seq_list"] = list(SINTEL_SEQ_LIST)
        metadata["full_seq"] = False
    elif eval_dataset == "bonn":
        metadata["seq_list"] = list(BONN_SEQ_LIST)
        metadata["full_seq"] = False

    return metadata


def resolve_sequence_list(args, metadata):
    if args.seq_list is not None:
        return sorted(args.seq_list)

    if args.full_seq or metadata.get("full_seq", False):
        img_path = metadata["img_path"]
        return sorted(
            seq
            for seq in os.listdir(img_path)
            if os.path.isdir(os.path.join(img_path, seq))
        )

    return sorted(metadata.get("seq_list", []))


def process_kitti(args, img_path):
    for directory in tqdm(sorted(glob.glob(f"{img_path}/*"))):
        filelist = sorted(glob.glob(f"{directory}/*.png"))
        save_dir = f"{args.output_dir}/{os.path.basename(directory)}"
        yield filelist, save_dir


def process_bonn(args, img_path):
    if args.full_seq:
        for directory in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{directory}/rgb_full/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(directory))}"
            yield filelist, save_dir
        return

    seq_list = list(BONN_SEQ_LIST) if args.seq_list is None else args.seq_list
    for seq in tqdm(seq_list):
        filelist = sorted(glob.glob(f"{img_path}/rgbd_bonn_{seq}/rgb_full/*.png"))
        save_dir = f"{args.output_dir}/{seq}"
        yield filelist, save_dir


def process_nyu(args, img_path):
    filelist = sorted(glob.glob(f"{img_path}/*.png"))
    save_dir = f"{args.output_dir}"
    yield filelist, save_dir


def process_scannet(args, img_path):
    seq_list = sorted(glob.glob(f"{img_path}/*"))
    for seq in tqdm(seq_list):
        color_dir = os.path.join(seq, "color_full")
        filelist = sorted(glob.glob(f"{color_dir}/*.jpg"))
        save_dir = f"{args.output_dir}/{os.path.basename(seq)}"
        yield filelist, save_dir


def process_sintel(args, img_path):
    if args.full_seq:
        for directory in tqdm(sorted(glob.glob(f"{img_path}/*/"))):
            filelist = sorted(glob.glob(f"{directory}/*.png"))
            save_dir = f"{args.output_dir}/{os.path.basename(os.path.dirname(directory))}"
            yield filelist, save_dir
        return

    for seq in tqdm(SINTEL_SEQ_LIST):
        filelist = sorted(glob.glob(f"{img_path}/{seq}/*.png"))
        save_dir = f"{args.output_dir}/{seq}"
        yield filelist, save_dir
