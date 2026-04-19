import sys
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
)  # noqa
from einops import rearrange

inf = float("inf")
from accelerate.logging import get_logger

printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        mom_num_patches=768,
        mom_topk=1,
        mom_anchor_idx=0,
        mom_update_anchor=True,
        mom_track_topk=False,
        mom_beta_gate=False,
        mom_ttsa_gate=False,
        mom_pose_sparse_read=False,
        mom_trace_update_gate=False,
        model_variant=None,
        mom_dual_bank=False,
        mom_topk_img=None,
        mom_topk_pose=None,
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.mom_num_patches = mom_num_patches
        self.mom_topk = mom_topk
        self.mom_anchor_idx = mom_anchor_idx
        self.mom_update_anchor = mom_update_anchor
        self.mom_track_topk = mom_track_topk
        self.mom_beta_gate = mom_beta_gate
        self.mom_ttsa_gate = mom_ttsa_gate
        self.mom_pose_sparse_read = mom_pose_sparse_read
        self.mom_trace_update_gate = mom_trace_update_gate
        self.model_variant = model_variant
        # Deprecated no-op fields kept in signature for backward compatibility.
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        )
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True
    VARIANT_SPECS = OrderedDict(
        [
            ("cut", {"mom_topk": 768, "mom_beta_gate": False, "mom_ttsa_gate": False}),
            ("ttt", {"mom_topk": 768, "mom_beta_gate": True, "mom_ttsa_gate": False}),
            ("ttsa", {"mom_topk": 768, "mom_beta_gate": False, "mom_ttsa_gate": True}),
            (
                "cut_memix",
                {"mom_topk": 708, "mom_beta_gate": False, "mom_ttsa_gate": False},
            ),
            (
                "ttt_memix",
                {"mom_topk": 708, "mom_beta_gate": True, "mom_ttsa_gate": False},
            ),
            (
                "ttsa_memix",
                {"mom_topk": 708, "mom_beta_gate": False, "mom_ttsa_gate": True},
            ),
        ]
    )
    LEGACY_VARIANT_RENAMES = {
        "mom": "cut_memix",
        "momttt": "ttt_memix",
        "ttsa_mom": "ttsa_memix",
    }

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            )
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim)
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.mom_num_patches = max(1, int(config.mom_num_patches))
        self.mom_topk = int(config.mom_topk)
        self.mom_anchor_idx = config.mom_anchor_idx
        self.mom_update_anchor = bool(config.mom_update_anchor)
        self.mom_track_topk = bool(getattr(config, "mom_track_topk", False))
        self.mom_beta_gate = bool(getattr(config, "mom_beta_gate", False))
        self.mom_ttsa_gate = bool(getattr(config, "mom_ttsa_gate", False))
        self.mom_pose_sparse_read = bool(
            getattr(config, "mom_pose_sparse_read", False)
        )
        self.mom_trace_update_gate = bool(
            getattr(config, "mom_trace_update_gate", False)
        )
        self.model_variant = getattr(config, "model_variant", None)
        self.mom_topk_counter = None
        self.mom_update_trace_steps = []
        self.mom_update_trace_patch_bins = 0
        self.prev_feat_i = None
        self.prev_new_state_feat = None
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.set_freeze(config.freeze)
        if self.model_variant is not None:
            self.set_variant(self.model_variant)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    @classmethod
    def _normalize_variant_name(cls, name):
        if name is None:
            return None
        variant = str(name).strip().lower()
        if variant in cls.VARIANT_SPECS:
            return variant
        if variant in cls.LEGACY_VARIANT_RENAMES:
            replacement = cls.LEGACY_VARIANT_RENAMES[variant]
            raise ValueError(
                f"Legacy model variant '{variant}' is no longer supported. "
                f"Use '{replacement}' instead."
            )
        allowed = ", ".join(cls.VARIANT_SPECS.keys())
        raise ValueError(f"Unsupported model variant '{name}'. Allowed: {allowed}")

    def set_variant(self, name):
        variant = self._normalize_variant_name(name)
        spec = self.VARIANT_SPECS[variant]
        self.mom_topk = int(spec["mom_topk"])
        self.mom_beta_gate = bool(spec["mom_beta_gate"])
        self.mom_ttsa_gate = bool(spec["mom_ttsa_gate"])
        self.model_variant = variant
        if hasattr(self, "config") and self.config is not None:
            self.config.mom_topk = self.mom_topk
            self.config.mom_beta_gate = self.mom_beta_gate
            self.config.mom_ttsa_gate = self.mom_ttsa_gate
            self.config.model_variant = self.model_variant
        return self

    def _reset_temporal_update_state(self):
        self.prev_feat_i = None
        self.prev_new_state_feat = None

    def _needs_cross_attn_state(self, i):
        return (
            i > 0
            and self.mom_topk > 0
            and (bool(self.mom_beta_gate) or bool(self.mom_ttsa_gate))
        )

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        )
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(
                selected_ray_maps, selected_shapes_ray
            )
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(
        self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, return_attn=False
    ):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        raw_img_feat = f_img
        # MoM sparse write is applied at frame-level in _apply_state_update.
        f_img = self.decoder_embed(f_img)
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1)
            pos_img = torch.cat([pos_pose, pos_img], dim=1)
        pose_attn_mask = self._build_pose_attn_mask(
            f_state, raw_img_feat, f_img.shape[1]
        )
        final_output.append((f_state, f_img))
        attention_maps = []
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                if return_attn:
                    f_state, _, self_attn_state, cross_attn_state = checkpoint(
                        blk_state,
                        *final_output[-1][::+1],
                        pos_state,
                        pos_img,
                        return_attn,
                        use_reentrant=not self.fixed_input_length,
                    )
                    f_img, _, self_attn_img, cross_attn_img = checkpoint(
                        blk_img,
                        *final_output[-1][::-1],
                        pos_img,
                        pos_state,
                        return_attn,
                        pose_attn_mask,
                        use_reentrant=not self.fixed_input_length,
                    )
                else:
                    f_state, _ = checkpoint(
                        blk_state,
                        *final_output[-1][::+1],
                        pos_state,
                        pos_img,
                        use_reentrant=not self.fixed_input_length,
                    )
                    f_img, _ = checkpoint(
                        blk_img,
                        *final_output[-1][::-1],
                        pos_img,
                        pos_state,
                        False,
                        pose_attn_mask,
                        use_reentrant=not self.fixed_input_length,
                    )
            else:
                if return_attn:
                    f_state, _, self_attn_state, cross_attn_state = blk_state(
                        *final_output[-1][::+1],
                        pos_state,
                        pos_img,
                        return_attn=True,
                    )
                    f_img, _, self_attn_img, cross_attn_img = blk_img(
                        *final_output[-1][::-1],
                        pos_img,
                        pos_state,
                        return_attn=True,
                        attn_mask=pose_attn_mask,
                    )
                else:
                    f_state, _ = blk_state(
                        *final_output[-1][::+1], pos_state, pos_img
                    )
                    f_img, _ = blk_img(
                        *final_output[-1][::-1],
                        pos_img,
                        pos_state,
                        False,
                        pose_attn_mask,
                    )

            final_output.append((f_state, f_img))
            if return_attn:
                attention_maps.append(
                    (self_attn_state, cross_attn_state, self_attn_img, cross_attn_img)
                )
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        if return_attn:
            return zip(*final_output), zip(*attention_maps)
        return zip(*final_output)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat)
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        return_attn=False,
    ):
        if return_attn:
            (new_state_feat, dec), (
                self_attn_state,
                cross_attn_state,
                self_attn_img,
                cross_attn_img,
            ) = self._decoder(
                state_feat,
                state_pos,
                current_feat,
                current_pos,
                pose_feat,
                pose_pos,
                return_attn=return_attn,
            )
            new_state_feat = new_state_feat[-1]
            return (
                new_state_feat,
                dec,
                self_attn_state,
                cross_attn_state,
                self_attn_img,
                cross_attn_img,
            )
        new_state_feat, dec = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def _mom_patch_gate(self, state_feat, feat_i, track=True, topk=None):
        if topk is None:
            topk = self.mom_topk
        num_tokens = state_feat.shape[1]
        num_patches = min(self.mom_num_patches, num_tokens)
        if track and self.mom_track_topk:
            if (
                self.mom_topk_counter is None
                or self.mom_topk_counter.numel() != num_patches
                or self.mom_topk_counter.device != state_feat.device
            ):
                self.mom_topk_counter = torch.zeros(
                    num_patches, device=state_feat.device, dtype=torch.long
                )
        if topk <= 0:
            return state_feat.new_zeros(state_feat.shape[0], num_tokens, 1)
        if num_patches <= 1 or topk >= num_patches:
            if track and self.mom_track_topk:
                self.mom_topk_counter += state_feat.shape[0]
            return state_feat.new_ones(state_feat.shape[0], num_tokens, 1)

        img_feat = self.decoder_embed(feat_i)
        img_feat_mean = self._get_img_level_feat(img_feat)
        score_token = (state_feat * img_feat_mean).sum(dim=-1)
        chunks = torch.chunk(score_token, num_patches, dim=1)
        scores = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)

        eligible = scores.shape[1]
        anchor_idx = self.mom_anchor_idx
        if not self.mom_update_anchor and anchor_idx is not None:
            anchor_idx = int(anchor_idx)
            if 0 <= anchor_idx < scores.shape[1]:
                scores = scores.clone()
                scores[:, anchor_idx] = float("inf") # use inf to exclude from lowest K
                eligible -= 1
        k = min(int(topk), eligible)
        if k <= 0:
            return state_feat.new_zeros(state_feat.shape[0], num_tokens, 1)

        topk_idx = torch.topk(scores, k=k, dim=1, largest=False).indices
        if track and self.mom_track_topk:
            if (
                self.mom_topk_counter is None
                or self.mom_topk_counter.numel() != num_patches
                or self.mom_topk_counter.device != topk_idx.device
            ):
                self.mom_topk_counter = torch.zeros(
                    num_patches, device=topk_idx.device, dtype=torch.long
                )
            counts = torch.bincount(topk_idx.reshape(-1), minlength=num_patches)
            self.mom_topk_counter += counts
        gate = state_feat.new_zeros(state_feat.shape[0], num_tokens, 1)
        start = 0
        for p, chunk in enumerate(chunks):
            end = start + chunk.shape[1]
            if start == end:
                continue
            selected = (topk_idx == p).any(dim=1)
            if selected.any():
                gate[selected, start:end, 0] = 1.0
            start = end
        return gate

    def _mom_patch_gate_from_decoded(self, state_feat, img_feat, track=True, topk=None):
        if topk is None:
            topk = self.mom_topk
        num_tokens = state_feat.shape[1]
        num_patches = min(self.mom_num_patches, num_tokens)
        if track and self.mom_track_topk:
            if (
                self.mom_topk_counter is None
                or self.mom_topk_counter.numel() != num_patches
                or self.mom_topk_counter.device != state_feat.device
            ):
                self.mom_topk_counter = torch.zeros(
                    num_patches, device=state_feat.device, dtype=torch.long
                )
        if topk <= 0:
            return state_feat.new_zeros(state_feat.shape[0], num_tokens, 1)
        if num_patches <= 1 or topk >= num_patches:
            if track and self.mom_track_topk:
                self.mom_topk_counter += state_feat.shape[0]
            return state_feat.new_ones(state_feat.shape[0], num_tokens, 1)

        img_feat_mean = self._get_img_level_feat(img_feat)
        score_token = (state_feat * img_feat_mean).sum(dim=-1)
        chunks = torch.chunk(score_token, num_patches, dim=1)
        scores = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)

        eligible = scores.shape[1]
        anchor_idx = self.mom_anchor_idx
        if not self.mom_update_anchor and anchor_idx is not None:
            anchor_idx = int(anchor_idx)
            if 0 <= anchor_idx < scores.shape[1]:
                scores = scores.clone()
                scores[:, anchor_idx] = float("inf")  # use inf to exclude from lowest K
                eligible -= 1
        k = min(int(topk), eligible)
        if k <= 0:
            return state_feat.new_zeros(state_feat.shape[0], num_tokens, 1)

        topk_idx = torch.topk(scores, k=k, dim=1, largest=False).indices
        if track and self.mom_track_topk:
            if (
                self.mom_topk_counter is None
                or self.mom_topk_counter.numel() != num_patches
                or self.mom_topk_counter.device != topk_idx.device
            ):
                self.mom_topk_counter = torch.zeros(
                    num_patches, device=topk_idx.device, dtype=torch.long
                )
            counts = torch.bincount(topk_idx.reshape(-1), minlength=num_patches)
            self.mom_topk_counter += counts
        gate = state_feat.new_zeros(state_feat.shape[0], num_tokens, 1)
        start = 0
        for p, chunk in enumerate(chunks):
            end = start + chunk.shape[1]
            if start == end:
                continue
            selected = (topk_idx == p).any(dim=1)
            if selected.any():
                gate[selected, start:end, 0] = 1.0
            start = end
        return gate

    def _build_pose_attn_mask(self, state_feat, feat_i, num_img_tokens):
        if not self.mom_pose_sparse_read or not self.pose_head_flag:
            return None
        if self.mom_topk <= 0:
            return None
        patch_gate = self._mom_patch_gate(state_feat, feat_i, track=False)
        key_mask = patch_gate.squeeze(-1) > 0
        if key_mask.numel() == 0:
            return None
        key_counts = key_mask.sum(dim=1)
        if (key_counts == 0).any():
            return None
        if key_mask.all():
            return None
        batch_size, num_keys = key_mask.shape
        attn_mask = state_feat.new_zeros((batch_size, num_img_tokens, num_keys))
        neg_inf = torch.finfo(attn_mask.dtype).min
        attn_mask[:, 0, :] = (~key_mask).to(attn_mask.dtype) * neg_inf
        return attn_mask

    def _compute_beta_gate(self, cross_attn_state):
        if not cross_attn_state:
            return None
        attn_list = [attn for attn in cross_attn_state if attn is not None]
        if not attn_list:
            return None
        stacked = torch.stack(attn_list, dim=0)
        state_query_img_key = stacked.mean(dim=(0, 2, 4))
        return torch.sigmoid(state_query_img_key)[..., None]

    def _compute_ttsa_gate(
        self, i, update_mask, feat_i, new_state_feat, cross_attn_state
    ):
        if (
            i == 0
            or self.prev_feat_i is None
            or self.prev_new_state_feat is None
            or not cross_attn_state
        ):
            gate_ttsa = update_mask
        else:
            state_change = (new_state_feat - self.prev_new_state_feat).norm(dim=-1)
            state_change = state_change.squeeze(0)
            state_change_normalized = state_change / state_change.mean()
            temporal_mask = torch.sigmoid(state_change_normalized - 1.5)

            feat_i_norm = feat_i / feat_i.norm(dim=-1, keepdim=True)
            prev_feat_norm = self.prev_feat_i / self.prev_feat_i.norm(
                dim=-1, keepdim=True
            )
            feat_dissimilarity = 1.0 - (feat_i_norm * prev_feat_norm).sum(dim=-1)

            cross_att = rearrange(
                torch.cat(cross_attn_state, dim=0),
                "l h nstate nimg -> 1 nstate nimg (l h)",
            )[:, :, 1:, :]
            attention_magnitude = cross_att.mean(dim=-1).abs()
            interaction = attention_magnitude * feat_dissimilarity.unsqueeze(1)
            spatial_signal = interaction.max(dim=-1)[0].squeeze(0)
            spatial_mask = torch.sigmoid(spatial_signal)
            gate_ttsa = (temporal_mask * spatial_mask).unsqueeze(0).unsqueeze(-1)

        self.prev_feat_i = feat_i.clone().detach()
        self.prev_new_state_feat = new_state_feat.clone().detach()
        return gate_ttsa

    def reset_mom_update_trace(self):
        self.mom_update_trace_steps = []
        self.mom_update_trace_patch_bins = 0

    def get_mom_update_trace(self):
        if not self.mom_update_trace_steps:
            return None
        return torch.stack(self.mom_update_trace_steps, dim=0)

    def _record_mom_update_gate(self, gate):
        if not bool(getattr(self, "mom_trace_update_gate", False)):
            return
        if gate is None:
            return
        gate_step = gate.detach()
        if gate_step.ndim == 3 and gate_step.shape[-1] == 1:
            gate_step = gate_step[..., 0]
        if gate_step.ndim != 2 or gate_step.numel() == 0:
            return
        gate_step = gate_step.float()
        num_tokens = int(gate_step.shape[1])
        num_patches = min(int(self.mom_num_patches), num_tokens)
        if num_patches <= 0:
            return
        chunks = torch.chunk(gate_step, num_patches, dim=1)
        # [B, P] -> mean over batch so each frame stores one value per patch bin.
        patch_vals = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1).mean(
            dim=0
        )
        self.mom_update_trace_steps.append(patch_vals.cpu())
        self.mom_update_trace_patch_bins = num_patches

    def _apply_state_update(
        self,
        state_feat,
        new_state_feat,
        feat_i_for_gate,
        update_mask,
        beta_gate=None,
        ttsa_gate=None,
        topk=None,
        track=True,
    ):
        # Unified write gate: view update mask * patch gate * optional beta gate * optional TTSA gate.
        patch_gate = self._mom_patch_gate(
            new_state_feat,
            feat_i_for_gate,
            track=track,
            topk=topk,
        )
        gate = update_mask * patch_gate
        if beta_gate is not None:
            gate = gate * beta_gate
        if ttsa_gate is not None:
            gate = gate * ttsa_gate
        self._record_mom_update_gate(gate)
        return new_state_feat * gate + state_feat * (1 - gate)

    def _forward_encoder(self, views):
        self._reset_temporal_update_state()
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        )

    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
    ):
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        return_attn = self._needs_cross_attn_state(i)
        if return_attn:
            (
                new_state_feat,
                dec,
                _self_attn_state,
                cross_attn_state,
                _self_attn_img,
                _cross_attn_img,
            ) = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
                return_attn=True,
            )
            beta_gate = (
                self._compute_beta_gate(cross_attn_state)
                if self.mom_beta_gate
                else None
            )
        else:
            new_state_feat, dec = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )
            beta_gate = None
            cross_attn_state = None
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape_i, pos=pos_i)
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update  # if don't update, then whatever img_mask
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        if self.mom_topk <= 0:
            update_mask = torch.zeros_like(update_mask)
        ttsa_gate = None
        if self.mom_ttsa_gate:
            ttsa_gate = self._compute_ttsa_gate(
                i, update_mask, feat_i, new_state_feat, cross_attn_state
            )
        state_feat = self._apply_state_update(
            state_feat,
            new_state_feat,
            feat_i,
            update_mask,
            beta_gate=beta_gate,
            ttsa_gate=ttsa_gate,
        )
        mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
        return res, (state_feat, mem)

    def _forward_impl(self, views, ret_state=False):
        self._reset_temporal_update_state()
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = None
        if ret_state:
            all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            return_attn = self._needs_cross_attn_state(i)
            if return_attn:
                (
                    new_state_feat,
                    dec,
                    _self_attn_state,
                    cross_attn_state,
                    _self_attn_img,
                    _cross_attn_img,
                ) = self._recurrent_rollout(
                    state_feat,
                    state_pos,
                    feat_i,
                    pos_i,
                    pose_feat_i,
                    pose_pos_i,
                    init_state_feat,
                    img_mask=views[i]["img_mask"],
                    reset_mask=views[i]["reset"],
                    update=views[i].get("update", None),
                    return_attn=True,
                )
                beta_gate = (
                    self._compute_beta_gate(cross_attn_state)
                    if self.mom_beta_gate
                    else None
                )
            else:
                new_state_feat, dec = self._recurrent_rollout(
                    state_feat,
                    state_pos,
                    feat_i,
                    pos_i,
                    pose_feat_i,
                    pose_pos_i,
                    init_state_feat,
                    img_mask=views[i]["img_mask"],
                    reset_mask=views[i]["reset"],
                    update=views[i].get("update", None),
                )
                beta_gate = None
                cross_attn_state = None
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            if self.mom_topk <= 0:
                update_mask = torch.zeros_like(update_mask)
            ttsa_gate = None
            if self.mom_ttsa_gate:
                ttsa_gate = self._compute_ttsa_gate(
                    i, update_mask, feat_i, new_state_feat, cross_attn_state
                )
            state_feat = self._apply_state_update(
                state_feat,
                new_state_feat,
                feat_i,
                update_mask,
                beta_gate=beta_gate,
                ttsa_gate=ttsa_gate,
            )
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            if ret_state:
                all_state_args.append(
                    (state_feat, state_pos, init_state_feat, mem, init_mem)
                )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False):
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    def inference_step(
        self, view, state_feat, state_pos, init_state_feat, mem, init_mem
    ):
        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []
        for j in range(batch_size):
            assert view["ray_mask"][j]
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)
            raymaps.append(raymap)
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(
                        view["ray_map"].shape[0], 1
                    ),
                )[[j]]
            )

        raymaps = torch.cat(raymaps, dim=0)
        shape = torch.cat(shapes, dim=0).to(raymaps.device)
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shapes)

        feat_i = feat_ls[-1]
        pos_i = pos
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        assert len(dec) == self.dec_depth + 1
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape, pos=pos_i)
        return res, view

    def forward_recurrent(
        self,
        views,
        device,
        ret_state=False,
        stream_views=False,
        offload_preds_to_cpu=False,
    ):
        if not isinstance(device, torch.device):
            device = torch.device(device)

        def _to_device(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.to(device, non_blocking=True)
            if isinstance(value, list):
                return [_to_device(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_to_device(v) for v in value)
            return value

        def _detach_to_cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, list):
                return [_detach_to_cpu(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_detach_to_cpu(v) for v in value)
            if isinstance(value, dict):
                return {k: _detach_to_cpu(v) for k, v in value.items()}
            return value

        ress = []
        all_state_args = [] if ret_state else None
        self._reset_temporal_update_state()
        for i, view in enumerate(views):
            if stream_views:
                model_view = {
                    "img": _to_device(view["img"]),
                    "img_mask": _to_device(view["img_mask"]),
                    "ray_mask": _to_device(view["ray_mask"]),
                    "ray_map": _to_device(view["ray_map"]),
                    "reset": _to_device(view["reset"]),
                    "update": _to_device(view.get("update", None)),
                }
                if "true_shape" in view:
                    model_view["true_shape"] = _to_device(view["true_shape"])
            else:
                model_view = view

            batch_size = model_view["img"].shape[0]
            img_mask = model_view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = model_view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = model_view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = model_view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                model_view["true_shape"].unsqueeze(0)
                if "true_shape" in model_view
                else torch.tensor(model_view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                if ret_state:
                    all_state_args.append(
                        (state_feat, state_pos, init_state_feat, mem, init_mem)
                    )

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            return_attn = self._needs_cross_attn_state(i)
            if return_attn:
                (
                    new_state_feat,
                    dec,
                    _self_attn_state,
                    cross_attn_state,
                    _self_attn_img,
                    _cross_attn_img,
                ) = self._recurrent_rollout(
                    state_feat,
                    state_pos,
                    feat_i,
                    pos_i,
                    pose_feat_i,
                    pose_pos_i,
                    init_state_feat,
                    img_mask=model_view["img_mask"],
                    reset_mask=model_view["reset"],
                    update=model_view.get("update", None),
                    return_attn=True,
                )
                beta_gate = (
                    self._compute_beta_gate(cross_attn_state)
                    if self.mom_beta_gate
                    else None
                )
            else:
                new_state_feat, dec = self._recurrent_rollout(
                    state_feat,
                    state_pos,
                    feat_i,
                    pos_i,
                    pose_feat_i,
                    pose_pos_i,
                    init_state_feat,
                    img_mask=model_view["img_mask"],
                    reset_mask=model_view["reset"],
                    update=model_view.get("update", None),
                )
                beta_gate = None
                cross_attn_state = None
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            if offload_preds_to_cpu:
                ress.append(_detach_to_cpu(res))
            else:
                ress.append(res)
            img_mask = model_view["img_mask"]
            update = model_view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            if self.mom_topk <= 0:
                update_mask = torch.zeros_like(update_mask)
            ttsa_gate = None
            if self.mom_ttsa_gate:
                ttsa_gate = self._compute_ttsa_gate(
                    i, update_mask, feat_i, new_state_feat, cross_attn_state
                )
            state_feat = self._apply_state_update(
                state_feat,
                new_state_feat,
                feat_i,
                update_mask,
                beta_gate=beta_gate,
                ttsa_gate=ttsa_gate,
            )
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = model_view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            if ret_state:
                all_state_args.append(
                    (state_feat, state_pos, init_state_feat, mem, init_mem)
                )
        if ret_state:
            return ress, views, all_state_args
        return ress, views


if __name__ == "__main__":
    print(ARCroco3DStereo.mro())
    cfg = ARCroco3DStereoConfig(
        state_size=256,
        pos_embed="RoPE100",
        rgb_head=True,
        pose_head=True,
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        pose_mode=("exp", -inf, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )
    ARCroco3DStereo(cfg)
