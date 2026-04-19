import os
import sys
from pathlib import Path
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import argparse
import numpy as np
import os.path as osp
import inspect
import importlib.util
import sysconfig
import torch.distributed as dist
import json

_stdlib = sysconfig.get_path("stdlib")
if _stdlib and _stdlib not in sys.path:
    sys.path.insert(0, _stdlib)

if importlib.util.find_spec("unittest") is None or importlib.util.find_spec(
    "unittest.main"
) is None:
    raise RuntimeError(
        "Python stdlib unittest is not importable. "
        f"stdlib={_stdlib} sys.path[0]={sys.path[0]!r}"
    )

try:
    import torch
except Exception as exc:
    import traceback

    print(f"torch import failed: {exc!r}", file=sys.stderr, flush=True)
    print(f"sys.executable={sys.executable}", file=sys.stderr, flush=True)
    print(f"sys.path={sys.path}", file=sys.stderr, flush=True)
    print(f"stdlib={_stdlib}", file=sys.stderr, flush=True)
    print(
        f"email spec={importlib.util.find_spec('email')}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"email.charset spec={importlib.util.find_spec('email.charset')}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"unittest spec={importlib.util.find_spec('unittest')}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"unittest.main spec={importlib.util.find_spec('unittest.main')}",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc()
    raise
import open3d as o3d
from torch.utils.data import DataLoader
from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(_REPO_ROOT / "data")))
MODEL_VARIANTS = (
    "cut",
    "ttt",
    "ttsa",
    "cut_memix",
    "ttt_memix",
    "ttsa_memix",
)


def _restore_full_cpu_affinity() -> None:
    """Undo accidental single-core pinning inherited from launcher/runtime."""
    try:
        if hasattr(os, "sched_setaffinity"):
            ncpu = os.cpu_count()
            if ncpu and ncpu > 1:
                os.sched_setaffinity(0, set(range(ncpu)))
    except Exception:
        # Best effort only; do not fail evaluation if affinity is unsupported.
        pass


def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "7scenes", "NRGBD"],
        help="dataset to evaluate",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="restrict evaluation to a single scene/sequence root",
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        default=None,
        help="restrict 7-Scenes evaluation to a single sequence like seq-01",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default=None,
        choices=MODEL_VARIANTS,
        help="select one of the 6 public model variants",
    )
    parser.add_argument(
        "--kf_every",
        type=int,
        default=200,
        help="keyframe stride for full-video evaluation",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=0,
        help="max frames per sequence (0 means no limit)",
    )
    parser.add_argument(
        "--recurrent_memory_mode",
        type=str,
        default="stream",
        choices=["stream", "legacy"],
        help=(
            "Memory policy for recurrent inference: "
            "'stream' uploads one view at a time; "
            "'legacy' keeps old full-batch upload behavior."
        ),
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
        "--mom_track_topk",
        action="store_true",
        default=False,
        help="MoM: record topk patch frequency",
    )
    parser.add_argument(
        "--mom_trace_update_gate",
        action="store_true",
        default=False,
        help="MoM: export per-frame update-gate trace (frame x patch-bin).",
    )
    parser.add_argument(
        "--skip_metrics_eval",
        action="store_true",
        default=False,
        help=(
            "Skip ICP + Acc/Comp/NC metrics (faster; still exports scene npy/trace/"
            "optional GLB)."
        ),
    )
    parser.add_argument(
        "--export_glb",
        action="store_true",
        default=False,
        help="Export fused 3D scene (.glb) per sequence.",
    )
    parser.add_argument(
        "--glb_as_pointcloud",
        action="store_true",
        default=False,
        help="When exporting GLB, save as point cloud instead of mesh.",
    )
    parser.add_argument(
        "--glb_conf_thresh",
        type=float,
        default=1.5,
        help="Confidence threshold for GLB point filtering.",
    )
    parser.add_argument(
        "--glb_cam_size",
        type=float,
        default=0.03,
        help="Camera frustum size in exported GLB.",
    )
    parser.add_argument(
        "--glb_hide_cams",
        action="store_true",
        default=False,
        help="Do not include camera frustums in exported GLB.",
    )
    parser.add_argument(
        "--glb_no_cam_texture",
        action="store_true",
        default=False,
        help="Render camera frustums without embedding per-frame image textures.",
    )
    parser.add_argument(
        "--glb_cam_color_mode",
        type=str,
        default="default",
        choices=["default", "ordered_green"],
        help="Camera color mode for GLB export.",
    )
    parser.add_argument(
        "--glb_max_points",
        type=int,
        default=0,
        help=(
            "Maximum number of points exported to GLB point cloud "
            "(0 disables downsampling)."
        ),
    )
    parser.add_argument(
        "--glb_sample_seed",
        type=int,
        default=0,
        help="Random seed used when downsampling GLB points.",
    )
    return parser


def _ordered_green_cam_colors(num_cams: int):
    if num_cams <= 0:
        return []
    start = np.array([0xB7, 0xF5, 0xB7], dtype=np.float32)  # #B7F5B7
    end = np.array([0x0B, 0x5E, 0x20], dtype=np.float32)  # #0B5E20
    if num_cams == 1:
        return [tuple(start.astype(np.uint8).tolist())]
    alpha = np.linspace(0.0, 1.0, num_cams, dtype=np.float32)[:, None]
    colors = start[None, :] * (1.0 - alpha) + end[None, :] * alpha
    return [tuple(np.clip(np.round(c), 0, 255).astype(np.uint8).tolist()) for c in colors]


def _build_glb_pointcloud_inputs(
    pts_all: np.ndarray,
    images_all: np.ndarray,
    glb_mask: np.ndarray,
    max_points: int,
    sample_seed: int,
):
    nviews = int(pts_all.shape[0])
    valid_flat_indices = [np.flatnonzero(glb_mask[i].reshape(-1)) for i in range(nviews)]
    valid_counts = np.array([idx.size for idx in valid_flat_indices], dtype=np.int64)
    total_valid = int(valid_counts.sum())

    if total_valid <= 0:
        empty_pts = [np.empty((0, 3), dtype=np.float32) for _ in range(nviews)]
        empty_cols = [np.empty((0, 3), dtype=np.float32) for _ in range(nviews)]
        empty_mask = [np.zeros((0,), dtype=bool) for _ in range(nviews)]
        return empty_pts, empty_cols, empty_mask

    if max_points > 0 and total_valid > max_points:
        expected = valid_counts.astype(np.float64) * (float(max_points) / float(total_valid))
        keep_counts = np.floor(expected).astype(np.int64)
        remain = int(max_points - int(keep_counts.sum()))
        if remain > 0:
            frac = expected - keep_counts
            order = np.argsort(frac)[::-1]
            for idx in order:
                if remain <= 0:
                    break
                if keep_counts[idx] < valid_counts[idx]:
                    keep_counts[idx] += 1
                    remain -= 1
        if remain > 0:
            room = valid_counts - keep_counts
            order = np.argsort(room)[::-1]
            for idx in order:
                if remain <= 0:
                    break
                avail = int(room[idx])
                if avail <= 0:
                    continue
                take = min(avail, remain)
                keep_counts[idx] += take
                remain -= take
    else:
        keep_counts = valid_counts

    rng = np.random.default_rng(int(sample_seed))
    glb_pts = []
    glb_cols = []
    glb_masks = []

    for i in range(nviews):
        flat_idx = valid_flat_indices[i]
        keep = int(min(max(0, keep_counts[i]), flat_idx.size))
        if keep <= 0:
            glb_pts.append(np.empty((0, 3), dtype=np.float32))
            glb_cols.append(np.empty((0, 3), dtype=np.float32))
            glb_masks.append(np.zeros((0,), dtype=bool))
            continue
        if keep < flat_idx.size:
            chosen = rng.choice(flat_idx, size=keep, replace=False)
        else:
            chosen = flat_idx
        pts_i = pts_all[i].reshape(-1, 3)[chosen].astype(np.float32, copy=False)
        cols_i = images_all[i].reshape(-1, 3)[chosen].astype(np.float32, copy=False)
        glb_pts.append(pts_i)
        glb_cols.append(cols_i)
        glb_masks.append(np.ones((pts_i.shape[0],), dtype=bool))

    return glb_pts, glb_cols, glb_masks


def main(args):
    _restore_full_cpu_affinity()
    if args.scene_name and args.dataset == "all":
        raise ValueError("--scene_name requires --dataset to be 7scenes or NRGBD")
    if args.seq_name and args.dataset != "7scenes":
        raise ValueError("--seq_name is only valid with --dataset 7scenes")
    add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import SevenScenes, NRGBD
    from eval.mv_recon.utils import accuracy, completion
    convert_scene_output_to_glb = None

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError
    datasets_all = {}
    if args.dataset in ("all", "7scenes"):
        datasets_all["7scenes"] = SevenScenes(
            split="test",
            ROOT=str(DATA_ROOT / "7scenes"),
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf_every,
            max_views=args.max_views,
            test_id=args.scene_name,
            seq_id=args.seq_name,
        )  # 20),
    if args.dataset in ("all", "NRGBD"):
        nrgbd_root = DATA_ROOT / "neural_rgbd"
        if not nrgbd_root.exists():
            raise FileNotFoundError(
                f"NRGBD dataset root not found: {nrgbd_root}"
            )
        datasets_all["NRGBD"] = NRGBD(
            split="test",
            ROOT=str(nrgbd_root),
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=args.kf_every,
            max_views=args.max_views,
            test_id=args.scene_name,
        )

    timeout_env = os.environ.get("MOM3R_DIST_TIMEOUT") or os.environ.get(
        "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT"
    )
    timeout_seconds = None
    if timeout_env:
        try:
            timeout_seconds = int(timeout_env)
        except ValueError:
            print(
                f"Invalid distributed timeout '{timeout_env}', ignoring.",
                file=sys.stderr,
                flush=True,
            )
            timeout_seconds = None
    kwargs_handlers = []
    if timeout_seconds is not None and timeout_seconds > 0:
        kwargs_handlers.append(
            InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_seconds))
        )
    accelerator = (
        Accelerator(kwargs_handlers=kwargs_handlers)
        if kwargs_handlers
        else Accelerator()
    )
    device = accelerator.device
    model_name = args.model_name
    if model_name in ("ours", "cut3r", "ours_random"):
        if model_name == "ours_random":
            import dust3r.model_random_update
            from dust3r.model_random_update import ARCroco3DStereo
            # Sync MOM_K with mom_topk if provided
            if args.mom_topk is not None:
                dust3r.model_random_update.MOM_K = int(args.mom_topk)
        else:
            from dust3r.model import ARCroco3DStereo
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy

        model = ARCroco3DStereo.from_pretrained(args.weights).to(device)
        if args.model_variant is not None:
            if not hasattr(model, "set_variant"):
                raise AttributeError("Loaded model does not support --model_variant")
            model.set_variant(args.model_variant)
        model.float()  # Ensure float32 for RoPE
        if args.mom_num_patches is not None:
            model.mom_num_patches = max(1, int(args.mom_num_patches))
        if args.mom_topk is not None:
            model.mom_topk = int(args.mom_topk)
        if args.mom_anchor_idx is not None:
            model.mom_anchor_idx = int(args.mom_anchor_idx)
        if args.mom_update_anchor:
            model.mom_update_anchor = True
        if args.mom_beta_gate:
            model.mom_beta_gate = True
        if args.mom_pose_sparse_read:
            model.mom_pose_sparse_read = True
        if args.mom_track_topk:
            model.mom_track_topk = True
            model.mom_topk_counter = None
        if args.mom_trace_update_gate:
            model.mom_trace_update_gate = True
            if hasattr(model, "reset_mom_update_trace"):
                model.reset_mom_update_trace()
        model.eval()
    else:
        raise NotImplementedError
    os.makedirs(args.output_dir, exist_ok=True)
    supports_recurrent_stream_args = False
    if hasattr(model, "forward_recurrent"):
        try:
            recurrent_sig = inspect.signature(model.forward_recurrent)
            params = recurrent_sig.parameters
            supports_recurrent_stream_args = (
                "stream_views" in params and "offload_preds_to_cpu" in params
            )
        except (TypeError, ValueError):
            supports_recurrent_stream_args = False

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    mode_audit_logged = False
    stream_support_logged = False

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    if args.mom_trace_update_gate and hasattr(
                        model, "reset_mom_update_trace"
                    ):
                        model.reset_mom_update_trace()
                    ignore_keys = set(
                        [
                            "depthmap",
                            "dataset",
                            "label",
                            "instance",
                            "idx",
                            "true_shape",
                            "rng",
                        ]
                    )

                    if model_name == "ours" or model_name == "cut3r":
                        revisit = args.revisit
                        update = not args.freeze
                        if revisit > 1:
                            # repeat input for 'revisit' times
                            new_views = []
                            for r in range(revisit):
                                for i in range(len(batch)):
                                    new_view = deepcopy(batch[i])
                                    new_view["idx"] = [
                                        (r * len(batch) + i)
                                        for _ in range(len(batch[i]["idx"]))
                                    ]
                                    new_view["instance"] = [
                                        str(r * len(batch) + i)
                                        for _ in range(len(batch[i]["instance"]))
                                    ]
                                    if r > 0:
                                        if not update:
                                            new_view["update"] = torch.zeros_like(
                                                batch[i]["update"]
                                            ).bool()
                                    new_views.append(new_view)
                            batch = new_views
                        use_recurrent = (
                            hasattr(model, "forward_recurrent") and len(batch) > 50
                        )
                        use_stream_recurrent = (
                            use_recurrent
                            and args.recurrent_memory_mode == "stream"
                            and supports_recurrent_stream_args
                        )
                        if (
                            use_recurrent
                            and args.recurrent_memory_mode == "stream"
                            and not supports_recurrent_stream_args
                            and not stream_support_logged
                        ):
                            print(
                                "WARNING: recurrent stream mode requested but the "
                                "loaded model.forward_recurrent does not accept "
                                "stream arguments; falling back to legacy behavior.",
                                file=sys.stderr,
                                flush=True,
                            )
                            stream_support_logged = True
                        if not mode_audit_logged:
                            print(
                                "Recurrent mode audit: "
                                f"recurrent_memory_mode={args.recurrent_memory_mode}, "
                                f"use_recurrent={use_recurrent}, "
                                f"batch_len={len(batch)}",
                                flush=True,
                            )
                            mode_audit_logged = True
                        if not use_stream_recurrent:
                            for view in batch:
                                for name in view.keys():  # pseudo_focal
                                    if name in ignore_keys:
                                        continue
                                    if isinstance(view[name], tuple) or isinstance(
                                        view[name], list
                                    ):
                                        view[name] = [
                                            x.to(device, non_blocking=True)
                                            for x in view[name]
                                        ]
                                    else:
                                        view[name] = view[name].to(
                                            device, non_blocking=True
                                        )
                        with torch.cuda.amp.autocast(enabled=False):
                            start = time.time()
                            if use_recurrent:
                                if supports_recurrent_stream_args:
                                    preds, batch = model.forward_recurrent(
                                        batch,
                                        device,
                                        stream_views=use_stream_recurrent,
                                        offload_preds_to_cpu=use_stream_recurrent,
                                    )
                                else:
                                    preds, batch = model.forward_recurrent(batch, device)
                            else:
                                output = model(batch)
                                preds, batch = output.ress, output.views
                            end = time.time()
                        valid_length = len(preds) // revisit
                        preds = preds[-valid_length:]
                        batch = batch[-valid_length:]
                        fps = len(batch) / (end - start)
                        print(
                            f"Finished reconstruction for {name_data} {data_idx+1}/{len(dataset)}, FPS: {fps:.2f}"
                        )
                        # continue
                        fps_all.append(fps)
                        time_all.append(end - start)

                        # Evaluation -- move data to CPU to free GPU memory
                        _eval_threads = int(os.environ.get("EVAL_OMP_THREADS", "24"))
                        os.environ["OMP_NUM_THREADS"] = str(_eval_threads)
                        try:
                            import ctypes
                            ctypes.CDLL('libgomp.so.1').omp_set_num_threads(_eval_threads)
                        except Exception:
                            pass
                        print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")

                        # Move preds/batch tensors to CPU before evaluation
                        for _p in preds:
                            for _k, _v in _p.items():
                                if isinstance(_v, torch.Tensor):
                                    _p[_k] = _v.cpu()
                        for _b in batch:
                            for _k, _v in _b.items():
                                if isinstance(_v, torch.Tensor):
                                    _b[_k] = _v.cpu()
                                elif isinstance(_v, (list, tuple)):
                                    _b[_k] = [x.cpu() if isinstance(x, torch.Tensor) else x for x in _v]
                        torch.cuda.empty_cache()

                        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                            criterion.get_all_pts3d_t(batch, preds)
                        )
                        pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                            monitoring["pred_scale"],
                            monitoring["gt_scale"],
                            monitoring["pred_shift_z"],
                            monitoring["gt_shift_z"],
                        )

                        in_camera1 = None
                        pts_all = []
                        pts_gt_all = []
                        images_all = []
                        masks_all = []
                        conf_all = []
                        cams2world_all = []
                        focals_all = []

                        for j, view in enumerate(batch):
                            if in_camera1 is None:
                                in_camera1 = view["camera_pose"][0].cpu()

                            image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                            mask = view["valid_mask"].cpu().numpy()[0]

                            # pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
                            pts = pred_pts[j].cpu().numpy()[0]
                            conf = preds[j]["conf"].cpu().data.numpy()[0]
                            # mask = mask & (conf > 1.8)

                            pts_gt = gt_pts[j].detach().cpu().numpy()[0]
                            camera_pose = view["camera_pose"][0].cpu().numpy()
                            focal = float(view["camera_intrinsics"][0, 0, 0].cpu().numpy())

                            H, W = image.shape[:2]
                            cx = W // 2
                            cy = H // 2
                            l, t = cx - 112, cy - 112
                            r, b = cx + 112, cy + 112
                            image = image[t:b, l:r]
                            mask = mask[t:b, l:r]
                            pts = pts[t:b, l:r]
                            pts_gt = pts_gt[t:b, l:r]
                            conf = conf[t:b, l:r]

                            #### Align predicted 3D points to the ground truth
                            pts[..., -1] += gt_shift_z.cpu().numpy().item()
                            pts = geotrf(in_camera1, pts)

                            pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                            pts_gt = geotrf(in_camera1, pts_gt)

                            images_all.append((image[None, ...] + 1.0) / 2.0)
                            pts_all.append(pts[None, ...])
                            pts_gt_all.append(pts_gt[None, ...])
                            masks_all.append(mask[None, ...])
                            conf_all.append(conf[None, ...])
                            cams2world_all.append(camera_pose)
                            focals_all.append(focal)

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)
                    conf_all = np.concatenate(conf_all, axis=0)
                    cams2world_all = np.asarray(cams2world_all, dtype=np.float32)
                    focals_all = np.asarray(focals_all, dtype=np.float32)

                    scene_id = view["label"][0].rsplit("/", 1)[0]

                    save_params = {}

                    save_params["images_all"] = images_all
                    save_params["pts_all"] = pts_all
                    save_params["pts_gt_all"] = pts_gt_all
                    save_params["masks_all"] = masks_all

                    np.save(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                        save_params,
                    )
                    if args.mom_trace_update_gate:
                        trace_base = scene_id.replace("/", "_")
                        trace_tensor = None
                        if hasattr(model, "get_mom_update_trace"):
                            trace_tensor = model.get_mom_update_trace()
                        if trace_tensor is None:
                            trace_np = np.zeros((0, 0), dtype=np.float32)
                        else:
                            trace_np = trace_tensor.detach().cpu().numpy().astype(
                                np.float32, copy=False
                            )
                        trace_path = os.path.join(
                            save_path, f"{trace_base}_update_trace.npy"
                        )
                        np.save(trace_path, trace_np)
                        trace_meta = {
                            "scene_id": scene_id,
                            "scene_key": trace_base,
                            "dataset": name_data,
                            "model_name": model_name,
                            "args_model_name": args.model_name,
                            "kf_every": int(args.kf_every),
                            "max_views_arg": int(args.max_views),
                            "processed_views": int(len(batch)),
                            "trace_frames": int(trace_np.shape[0]),
                            "trace_patch_bins": int(
                                trace_np.shape[1] if trace_np.ndim == 2 else 0
                            ),
                            "mom_num_patches": int(
                                getattr(model, "mom_num_patches", 0)
                            ),
                            "mom_topk": int(getattr(model, "mom_topk", 0)),
                            "mom_anchor_idx": int(
                                getattr(model, "mom_anchor_idx", 0)
                            ),
                            "mom_update_anchor": bool(
                                getattr(model, "mom_update_anchor", False)
                            ),
                            "mom_beta_gate": bool(
                                getattr(model, "mom_beta_gate", False)
                            ),
                        }
                        meta_path = os.path.join(
                            save_path, f"{trace_base}_update_trace_meta.json"
                        )
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(trace_meta, f, indent=2)

                    if args.export_glb:
                        if convert_scene_output_to_glb is None:
                            try:
                                from viser_utils import convert_scene_output_to_glb
                            except ModuleNotFoundError as exc:
                                raise ModuleNotFoundError(
                                    "GLB export requested but `viser_utils` is not available "
                                    "inside MeMix. Run without --export_glb or vendor the "
                                    "visualization helper into this repo."
                                ) from exc
                        conf_thresh = float(args.glb_conf_thresh)
                        glb_mask = (masks_all > 0) & (conf_all >= conf_thresh)
                        save_name = (
                            f"{scene_id.replace('/', '_')}_"
                            f"{'pointcloud' if args.glb_as_pointcloud else 'mesh'}"
                        )

                        if bool(args.glb_no_cam_texture):
                            cam_h, cam_w = images_all.shape[1:3]
                            blank = np.zeros((cam_h, cam_w, 3), dtype=np.float32)
                            cam_imgs = [blank] * int(images_all.shape[0])
                        else:
                            cam_imgs = [images_all[i] for i in range(int(images_all.shape[0]))]

                        cam_colors = None
                        if str(args.glb_cam_color_mode) == "ordered_green":
                            cam_colors = _ordered_green_cam_colors(len(cam_imgs))

                        glb_pts = pts_all
                        glb_cols = images_all
                        glb_masks = glb_mask
                        if bool(args.glb_as_pointcloud):
                            glb_pts, glb_cols, glb_masks = _build_glb_pointcloud_inputs(
                                pts_all=pts_all,
                                images_all=images_all,
                                glb_mask=glb_mask,
                                max_points=int(args.glb_max_points),
                                sample_seed=int(args.glb_sample_seed),
                            )

                        convert_scene_output_to_glb(
                            outdir=save_path,
                            imgs=cam_imgs,
                            pts3d=glb_pts,
                            mask=glb_masks,
                            focals=focals_all,
                            cams2world=cams2world_all,
                            cam_size=float(args.glb_cam_size),
                            show_cam=not bool(args.glb_hide_cams),
                            cam_color=cam_colors,
                            as_pointcloud=bool(args.glb_as_pointcloud),
                            transparent_cams=bool(args.glb_no_cam_texture),
                            silent=False,
                            save_name=save_name,
                            point_colors=glb_cols,
                            cam_imgs=cam_imgs,
                        )

                    if not args.skip_metrics_eval:
                        if "DTU" in name_data:
                            threshold = 100
                        else:
                            threshold = 0.1

                        pts_all_masked = pts_all[masks_all > 0]
                        pts_gt_all_masked = pts_gt_all[masks_all > 0]
                        images_all_masked = images_all[masks_all > 0]

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(
                            pts_all_masked.reshape(-1, 3)
                        )
                        pcd.colors = o3d.utility.Vector3dVector(
                            images_all_masked.reshape(-1, 3)
                        )
                        o3d.io.write_point_cloud(
                            os.path.join(
                                save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                            ),
                            pcd,
                        )

                        pcd_gt = o3d.geometry.PointCloud()
                        pcd_gt.points = o3d.utility.Vector3dVector(
                            pts_gt_all_masked.reshape(-1, 3)
                        )
                        pcd_gt.colors = o3d.utility.Vector3dVector(
                            images_all_masked.reshape(-1, 3)
                        )
                        o3d.io.write_point_cloud(
                            os.path.join(
                                save_path, f"{scene_id.replace('/', '_')}-gt.ply"
                            ),
                            pcd_gt,
                        )

                        trans_init = np.eye(4)

                        reg_p2p = o3d.pipelines.registration.registration_icp(
                            pcd,
                            pcd_gt,
                            threshold,
                            trans_init,
                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        )

                        transformation = reg_p2p.transformation

                        pcd = pcd.transform(transformation)
                        pcd.estimate_normals()
                        pcd_gt.estimate_normals()

                        gt_normal = np.asarray(pcd_gt.normals)
                        pred_normal = np.asarray(pcd.normals)

                        acc, acc_med, nc1, nc1_med = accuracy(
                            pcd_gt.points, pcd.points, gt_normal, pred_normal
                        )
                        comp, comp_med, nc2, nc2_med = completion(
                            pcd_gt.points, pcd.points, gt_normal, pred_normal
                        )
                        print(
                            f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                        )
                        print(
                            f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                            file=open(log_file, "a"),
                        )

                        acc_all += acc
                        comp_all += comp
                        nc1_all += nc1
                        nc2_all += nc2

                        acc_all_med += acc_med
                        comp_all_med += comp_med
                        nc1_all_med += nc1_med
                        nc2_all_med += nc2_med

                    # release cuda memory
                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()
            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                to_write = ""
                # Copy the error log from each process to the main error log
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.3f} | "
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)

    if getattr(model, "mom_track_topk", False):
        counts = getattr(model, "mom_topk_counter", None)
        if counts is not None:
            counts_local = counts.to(device=accelerator.device)
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
                args.output_dir, f"mom_topk_counts_rank{accelerator.process_index}.json"
            )
            with open(rank_out_path, "w", encoding="utf-8") as f:
                json.dump(rank_stats, f, indent=2)

            counts_total = counts_local.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts_total, op=dist.ReduceOp.SUM)
            if accelerator.is_main_process:
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
                out_path = os.path.join(args.output_dir, "mom_topk_counts.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(total_stats, f, indent=2)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
