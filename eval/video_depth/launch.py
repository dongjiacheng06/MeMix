import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import math
import cv2
import numpy as np
import torch
import torch.distributed as dist
import argparse
import json

from copy import deepcopy
from eval.video_depth.metadata import (
    dataset_metadata,
    resolve_dataset_metadata,
    resolve_sequence_list,
)
from eval.protocol_config import apply_runtime_slice, resolve_runtime_slice
from eval.video_depth.utils import save_depth_maps
from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r
import time
from tqdm import tqdm

MODEL_VARIANTS = (
    "cut",
    "ttt",
    "ttsa",
    "cut_memix",
    "ttt_memix",
    "ttsa_memix",
)


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
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
        "--model_variant",
        type=str,
        default=None,
        choices=MODEL_VARIANTS,
        help="select one of the 6 public model variants",
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument(
        "--start_frame",
        type=int,
        default=None,
        help="Optional starting frame index before runtime slicing.",
    )
    parser.add_argument(
        "--pose_eval_stride",
        default=None,
        type=int,
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
        help="skip sequences with fewer frames after stride (0 means no limit)",
    )
    parser.add_argument(
        "--no_recurrent",
        action="store_false",
        dest="use_recurrent",
        help="disable recurrent inference (uses more memory)",
    )
    parser.add_argument(
        "--no_state_update",
        action="store_true",
        default=False,
        help="disable state/memory updates during inference",
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )
    parser.add_argument(
        "--mom_num_patches",
        type=int,
        default=None,
        help="MoM: number of state patches (None keeps checkpoint default)",
    )
    parser.add_argument(
        "--mom_topk",
        type=int,
        default=None,
        help="MoM: top-k patches to update per step (None keeps checkpoint default)",
    )
    parser.add_argument(
        "--mom_anchor_idx",
        type=int,
        default=None,
        help="MoM: anchor patch index (None keeps checkpoint default)",
    )
    parser.add_argument(
        "--mom_update_anchor",
        action="store_true",
        default=False,
        help="MoM: allow anchor patch updates",
    )
    parser.add_argument(
        "--mom_track_topk",
        action="store_true",
        default=False,
        help="MoM: record topk patch frequency",
    )
    parser.add_argument(
        "--mom_beta_gate",
        action="store_true",
        default=False,
        help="MoM: enable beta gate for state updates",
    )
    parser.add_argument(
        "--mom_pose_sparse_read",
        action="store_true",
        default=False,
        help="MoM: enable pose sparse read from state",
    )
    return parser


def eval_pose_estimation(args, model, save_dir=None):
    metadata = resolve_dataset_metadata(
        args.eval_dataset, args.eval_protocol, args.seq_len
    )
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):
    from dust3r.inference import inference, inference_recurrent

    metadata = resolve_dataset_metadata(
        args.eval_dataset, args.eval_protocol, args.seq_len
    )
    anno_path = metadata.get("anno_path", None)
    seq_list = resolve_sequence_list(args, metadata)

    if save_dir is None:
        save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        load_img_size = args.size
        assert load_img_size == 512
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                filelist = sorted(
                    os.path.join(dir_path, name) for name in os.listdir(dir_path)
                )
                filelist = apply_runtime_slice(
                    filelist,
                    start_frame=args.start_frame,
                    pose_eval_stride=args.pose_eval_stride,
                    max_views=args.max_views,
                )
                if args.min_views and len(filelist) < args.min_views:
                    print(
                        f"Skipping {seq}: only {len(filelist)} frames (< {args.min_views})"
                    )
                    continue

                views = prepare_input(
                    filelist,
                    [True for _ in filelist],
                    size=load_img_size,
                    update=not args.no_state_update,
                    crop=not args.no_crop,
                )
                start = time.time()
                infer_fn = inference_recurrent if args.use_recurrent else inference
                outputs, _ = infer_fn(views, model, device)
                end = time.time()
                fps = len(filelist) / (end - start)

                (
                    colors,
                    pts3ds_self,
                    pts3ds_other,
                    conf_self,
                    conf_other,
                    cam_dict,
                    pr_poses,
                ) = prepare_output(outputs)

                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_depth_maps(pts3ds_self, f"{save_dir}/{seq}", conf_self=conf_self)

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception
    if getattr(model, "mom_track_topk", False):
        counts = getattr(model, "mom_topk_counter", None)
        if counts is not None:
            counts_tensor = counts.to(device=distributed_state.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
            if distributed_state.is_main_process:
                counts_list = counts_tensor.detach().cpu().tolist()
                stats = {
                    "mom_num_patches": int(
                        getattr(model, "mom_num_patches", len(counts_list))
                    ),
                    "mom_topk": int(getattr(model, "mom_topk", 0)),
                    "mom_anchor_idx": int(getattr(model, "mom_anchor_idx", 0)),
                    "mom_update_anchor": bool(
                        getattr(model, "mom_update_anchor", False)
                    ),
                    "counts": counts_list,
                }
                out_path = os.path.join(save_dir, "mom_topk_counts.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
    return None, None, None


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
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
    add_path_to_dust3r(args.weights)
    from dust3r.utils.image import load_images_for_eval as load_images
    from dust3r.post_process import estimate_focal_knowing_depth
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.camera import pose_encoding_to_camera

    args.no_crop = True

    def prepare_input(
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
    ):
        images = load_images(img_paths, size=size, crop=crop)
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"],
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(True).unsqueeze(0),
                    "ray_mask": torch.tensor(False).unsqueeze(0),
                    "update": torch.tensor(bool(update)).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                views.append(view)
        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"]
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan)
                    ),
                    "ray_map": (
                        raymaps[k]
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan)
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"])
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                    "update": torch.tensor(bool(img_mask[i] and update)).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            # repeat input for 'revisit' times
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    def prepare_output(outputs, revisit=1):
        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        pts3ds_self = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
        pts3ds_other = [
            output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
        ]
        conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
        conf_other = [output["conf"].cpu() for output in outputs["pred"]]
        pts3ds_self = torch.cat(pts3ds_self, 0)
        pr_poses = [
            pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
            for pred in outputs["pred"]
        ]
        pr_poses = torch.cat(pr_poses, 0)

        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        colors = [0.5 * (output["rgb"][0] + 1.0) for output in outputs["pred"]]
        cam_dict = {
            "focal": focal.cpu().numpy(),
            "pp": pp.cpu().numpy(),
        }
        return (
            colors,
            pts3ds_self,
            pts3ds_other,
            conf_self,
            conf_other,
            cam_dict,
            pr_poses,
        )

    model = ARCroco3DStereo.from_pretrained(args.weights)
    if args.model_variant is not None:
        if not hasattr(model, "set_variant"):
            raise AttributeError("Loaded model does not support --model_variant")
        model.set_variant(args.model_variant)
    if args.mom_num_patches is not None:
        model.mom_num_patches = max(1, int(args.mom_num_patches))
    if args.mom_topk is not None:
        model.mom_topk = int(args.mom_topk)
    if args.mom_anchor_idx is not None:
        model.mom_anchor_idx = int(args.mom_anchor_idx)
    if args.mom_update_anchor:
        model.mom_update_anchor = True
    if args.mom_track_topk:
        model.mom_track_topk = True
        model.mom_topk_counter = None
    if args.mom_beta_gate:
        model.mom_beta_gate = True
    if args.mom_pose_sparse_read:
        model.mom_pose_sparse_read = True
    eval_pose_estimation(args, model, save_dir=args.output_dir)
