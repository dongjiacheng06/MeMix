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
from eval.relpose.metadata import (
    dataset_metadata,
    resolve_dataset_metadata,
    resolve_sequence_list,
)
from eval.protocol_config import apply_runtime_slice, resolve_runtime_slice
from eval.relpose.utils import *

from accelerate import PartialState
from add_ckpt_path import add_path_to_dust3r

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
    parser.add_argument("--shuffle", action="store_true", default=False)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume evaluation by skipping sequences that already have metric files",
    )

    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--freeze_state", action="store_true", default=False)
    parser.add_argument("--solve_pose", action="store_true", default=False)
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
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        bug = False
        for seq in tqdm(seqs):
            try:
                if args.resume and os.path.exists(f"{save_dir}/{seq}_eval_metric.txt"):
                    print(f"Skipping {seq}: found existing metric file.")
                    continue
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
                    crop=not args.no_crop,
                    revisit=args.revisit,
                    update=not args.freeze_state,
                )
                infer_fn = inference_recurrent if args.use_recurrent else inference
                outputs, _ = infer_fn(views, model, device)

                (
                    colors,
                    pts3ds_self,
                    pts3ds_other,
                    conf_self,
                    conf_other,
                    cam_dict,
                    pr_poses,
                ) = prepare_output(
                    outputs, revisit=args.revisit, solve_pose=args.solve_pose
                )

                pred_traj = get_tum_poses(pr_poses)
                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_tum_poses(pr_poses, f"{save_dir}/{seq}/pred_traj.txt")
                save_focals(cam_dict, f"{save_dir}/{seq}/pred_focal.txt")
                save_intrinsics(cam_dict, f"{save_dir}/{seq}/pred_intrinsics.txt")
                # save_depth_maps(pts3ds_self,f'{save_dir}/{seq}', conf_self=conf_self)
                # save_conf_maps(conf_self,f'{save_dir}/{seq}')
                # save_rgb_imgs(colors,f'{save_dir}/{seq}')

                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)

                if args.eval_dataset == "sintel":
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        skip=args.start_frame,
                        stride=args.pose_eval_stride,
                        num_frames=len(filelist),
                    )
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        traj_format=traj_format,
                        skip=args.start_frame,
                        stride=args.pose_eval_stride,
                        num_frames=len(filelist),
                    )
                else:
                    gt_traj = None

                if gt_traj is not None:
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{save_dir}/{seq}_eval_metric.txt",
                    )
                    plot_trajectory(
                        pred_traj, gt_traj, title=seq, filename=f"{save_dir}/{seq}.png"
                    )
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0
                    bug = True

                ate_list.append(ate)
                rpe_trans_list.append(rpe_trans)
                rpe_rot_list.append(rpe_rot)

                # Write to error log after each sequence
                with open(error_log_path, "a") as f:
                    f.write(
                        f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
                    )
                    f.write(f"{ate:.5f}\n")
                    f.write(f"{rpe_trans:.5f}\n")
                    f.write(f"{rpe_rot:.5f}\n")

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

    distributed_state.wait_for_everyone()
    if getattr(model, "mom_track_topk", False):
        counts = getattr(model, "mom_topk_counter", None)
        if counts is not None:
            counts_local = counts.to(device=distributed_state.device)
            rank_stats = {
                "mom_num_patches": int(
                    getattr(model, "mom_num_patches", counts_local.shape[0])
                ),
                "mom_topk": int(getattr(model, "mom_topk", 0)),
                "mom_anchor_idx": int(getattr(model, "mom_anchor_idx", 0)),
                "mom_update_anchor": bool(getattr(model, "mom_update_anchor", False)),
                "counts": counts_local.detach().cpu().tolist(),
            }
            rank_out_path = os.path.join(
                save_dir, f"mom_topk_counts_rank{distributed_state.process_index}.json"
            )
            with open(rank_out_path, "w", encoding="utf-8") as f:
                json.dump(rank_stats, f, indent=2)

            counts_total = counts_local.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts_total, op=dist.ReduceOp.SUM)
            if distributed_state.is_main_process:
                total_stats = {
                    "mom_num_patches": int(
                        getattr(model, "mom_num_patches", counts_total.shape[0])
                    ),
                    "mom_topk": int(getattr(model, "mom_topk", 0)),
                    "mom_anchor_idx": int(getattr(model, "mom_anchor_idx", 0)),
                    "mom_update_anchor": bool(
                        getattr(model, "mom_update_anchor", False)
                    ),
                    "counts": counts_total.detach().cpu().tolist(),
                }
                out_path = os.path.join(save_dir, "mom_topk_counts.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(total_stats, f, indent=2)

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if distributed_state.is_main_process:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            # Copy the error log from each process to the main error log
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())
            f.write(
                f"Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )

    return avg_ate, avg_rpe_trans, avg_rpe_rot


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    runtime_slice = resolve_runtime_slice(
        "relpose",
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
    from dust3r.utils.geometry import weighted_procrustes, geotrf

    args.no_crop = False

    def recover_cam_params(pts3ds_self, pts3ds_other, conf_self, conf_other):
        B, H, W, _ = pts3ds_self.shape
        pp = (
            torch.tensor([W // 2, H // 2], device=pts3ds_self.device)
            .float()
            .repeat(B, 1)
            .reshape(B, 1, 2)
        )
        focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

        pts3ds_self = pts3ds_self.reshape(B, -1, 3)
        pts3ds_other = pts3ds_other.reshape(B, -1, 3)
        conf_self = conf_self.reshape(B, -1)
        conf_other = conf_other.reshape(B, -1)
        # weighted procrustes
        c2w = weighted_procrustes(
            pts3ds_self,
            pts3ds_other,
            torch.log(conf_self) * torch.log(conf_other),
            use_weights=True,
            return_T=True,
        )
        return c2w, focal, pp.reshape(B, 2)

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
                    "update": torch.tensor(True).unsqueeze(0),
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
                    "update": torch.tensor(img_mask[i]).unsqueeze(0),
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

    def prepare_output(outputs, revisit=1, solve_pose=False):
        valid_length = len(outputs["pred"]) // revisit
        outputs["pred"] = outputs["pred"][-valid_length:]
        outputs["views"] = outputs["views"][-valid_length:]

        if solve_pose:
            pts3ds_self = [
                output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
            ]
            pts3ds_other = [
                output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
            ]
            conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
            conf_other = [output["conf"].cpu() for output in outputs["pred"]]
            pr_poses, focal, pp = recover_cam_params(
                torch.cat(pts3ds_self, 0),
                torch.cat(pts3ds_other, 0),
                torch.cat(conf_self, 0),
                torch.cat(conf_other, 0),
            )
            pts3ds_self = torch.cat(pts3ds_self, 0)
        else:

            pts3ds_self = [
                output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
            ]
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
            focal = estimate_focal_knowing_depth(
                pts3ds_self, pp, focal_mode="weiszfeld"
            )

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
