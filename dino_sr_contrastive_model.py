"""
DINO-SR + Contrastive Head — model, augmentations, training loop
================================================================

Fork of dino_sr_ablation.py. Adds, per SPEC.md:

  * ProjectionMLP (64 -> 256 -> 128, BN+GELU, L2-norm) in parallel with the
    existing prototype head. Shared backbone.
  * Soft-label contrastive loss: MSE between
        target = cos-sim of teacher prototype-softmax rows  (detached)
        pred   = cos-sim of student projection-embedding rows
    with the diagonal masked out. Optional per-batch entropy gate that keeps
    samples below the batch-median teacher entropy.
  * Lambda warmup: 0 for `warmup_epochs`, linear ramp over `ramp_epochs`,
    then the target.
  * PolarThetaRoll: random circular shift along the polar theta axis,
    applied AFTER polar transform and AFTER center mask, per view.
  * ThetaCircularConv2d: wraps nn.Conv2d to use circular padding on theta
    (rows) and zero on r (cols). Recursively swapped into the ResNet trunk
    at model construction. Unit test supplied.
  * Removes Cartesian RandomRotation. Rotation is now handled exclusively
    by theta-roll in polar space.
  * Per-step logging of L_DINO, L_contrastive, total, lambda_effective,
    prototype-usage entropy, teacher softmax entropy (mean, p10, p90),
    student projection embedding norm (mean / std), and the fraction of
    the batch kept by the entropy gate.

The contrastive branch is purely additive. With
`contrastive_lambda=0 + theta_roll_aug=False`, the backbone /
student_projector / prototypes receive exactly the gradient they would
receive from the pure-DINO loss on the identical (rotation-less) aug
pipeline; the projection MLP drifts only due to its own BN and optimizer
state and never influences the DINO path.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from dino_sr_fixed import (
    CenterMask, ProjectionHead, Prototypes,
    get_teacher_temp, get_teacher_momentum, _is_bn_or_bias_param,
    VisionTransformerTiny,
)

# Reuse PolarTransform + PolarMaskLeft from the ablation fork so we stay
# compatible with the rest of the pipeline (eval_all, scorecard, etc.).
from dino_sr_ablation import PolarTransform, PolarMaskLeft, CenterOnCOM


# =========================================================================
# 1. THETA-CIRCULAR CONV
# =========================================================================

class ThetaCircularConv2d(nn.Module):
    """Conv2d with circular padding on theta (rows) and zero on r (cols).

    Expects tensors laid out as (N, C, theta=H, r=W), matching PolarTransform
    output.

    nn.Conv2d takes a single `padding_mode`, so we pad manually with F.pad
    (circular on rows, zero on cols) and hand a padding=0 Conv2d the result.
    The inner Conv2d keeps nn.Conv2d's weight init / parameter registration
    exactly, so state_dict shapes and optimizer grouping behave as usual.

    Used in place of every nn.Conv2d in the L1 ResNet trunk so the backbone
    is equivariant to theta-roll (up to stride). See
    theta_roll_equivariance_test for the interior-r-col tolerance check.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1, groups: int = 1,
                 bias: bool = True):
        super().__init__()
        if isinstance(padding, int):
            pad_theta = pad_r = padding
        else:
            pad_theta, pad_r = int(padding[0]), int(padding[1])
        self.pad_theta = int(pad_theta)
        self.pad_r = int(pad_r)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_r > 0:
            x = F.pad(x, (self.pad_r, self.pad_r, 0, 0),
                      mode='constant', value=0.0)
        if self.pad_theta > 0:
            x = F.pad(x, (0, 0, self.pad_theta, self.pad_theta),
                      mode='circular')
        return self.conv(x)


def replace_conv2d_circular_theta(module: nn.Module) -> int:
    """Recursively swap every nn.Conv2d child for a ThetaCircularConv2d
    wrapper with identical attributes and weights. Returns the count
    of replacements.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            wrapper = ThetaCircularConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
            )
            with torch.no_grad():
                wrapper.conv.weight.copy_(child.weight)
                if child.bias is not None:
                    wrapper.conv.bias.copy_(child.bias)
            setattr(module, name, wrapper)
            n += 1
        else:
            n += replace_conv2d_circular_theta(child)
    return n


# =========================================================================
# 2. ENCODER
# =========================================================================

LAYER_OUT_DIMS = {1: 64, 2: 128, 3: 256, 4: 512}


class PlainSequentialEncoder(nn.Sequential):
    def __init__(self, *args, use_maxpool=True, out_dim=128):
        super().__init__(*args)
        self.use_maxpool = use_maxpool
        self.out_dim = out_dim

    def forward(self, x, return_feature_map: bool = False):
        for m in self:
            x = m(x)
        if return_feature_map:
            return x
        if self.use_maxpool:
            # Mathematically identical to adaptive_max_pool2d(x, 1) on the
            # forward path, but uses Tensor.max(dim=) which has a
            # deterministic backward (selects the first argmax, scatters
            # gradient to that one index). adaptive_max_pool2d_backward_cuda
            # uses atomic adds and is non-deterministic.
            return x.flatten(2).max(dim=2).values
        return F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)


def create_encoder_resnet18_variableN(n_layers: int = 1,
                                       use_maxpool: bool = True,
                                       circular_theta: bool = True) -> PlainSequentialEncoder:
    assert n_layers in (1, 2, 3, 4)
    resnet = models.resnet18(weights=None)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                             padding=3, bias=False)
    layer_map = {1: resnet.layer1, 2: resnet.layer2,
                 3: resnet.layer3, 4: resnet.layer4}
    out_dim = LAYER_OUT_DIMS[n_layers]
    modules = [resnet.conv1, resnet.bn1, resnet.relu]
    for i in range(1, n_layers + 1):
        modules.append(layer_map[i])
    encoder = PlainSequentialEncoder(*modules,
                                     use_maxpool=use_maxpool,
                                     out_dim=out_dim)
    if circular_theta:
        replace_conv2d_circular_theta(encoder)
    return encoder


def encoder_total_theta_stride(n_layers: int = 1) -> int:
    """Total theta-axis downsampling factor for the L{n_layers} trunk.

    conv1 is stride=2. layer1 is stride=1 (BasicBlock default). Each
    subsequent layer adds a stride of 2. Used by the equivariance test
    to un-roll feature maps for comparison.
    """
    stride = 2  # conv1
    # layer1 is stride 1
    for i in range(2, n_layers + 1):
        stride *= 2
    return stride


# =========================================================================
# 3. HEADS
# =========================================================================

class ProjectionMLP(nn.Module):
    """Contrastive head: in_dim -> hidden -> out, BN + GELU, L2-normed output.

    Parallel to the existing ProjectionHead -> Prototypes path. Shares the
    backbone feature `feat` with the prototype head; only the head weights
    differ.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        return F.normalize(x, dim=-1, p=2)


# =========================================================================
# 4. POLAR THETA-ROLL
# =========================================================================

class PolarThetaRoll(nn.Module):
    """Per-sample independent random circular roll along the polar theta axis.

    Expects input shape (B, C, H=theta, W=r) or (C, H, W).

    Each sample in the batch gets its own integer shift drawn uniformly from
    [-shift_range//2, +shift_range//2] inclusive. Shift 0 is allowed, and if
    `shift_range=0` the module is a deterministic no-op (use for config a
    where theta_roll_aug=False).

    The last-draw shift amounts are exposed via `self.last_shifts` so the
    training loop can assert student/teacher asymmetry on the first batch.
    """

    def __init__(self, shift_range: int = 192):
        super().__init__()
        self.shift_range = int(max(0, shift_range))
        # last_shifts: CPU 1-D long tensor of per-sample shifts, updated
        # on every forward. Stored on CPU so the training loop can compare
        # without device transfer overhead.
        self.register_buffer("last_shifts", torch.zeros(0, dtype=torch.long),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shift_range == 0:
            # Keep last_shifts consistent so assertion code still sees zeros.
            B = x.shape[0] if x.dim() == 4 else 1
            self.last_shifts = torch.zeros(B, dtype=torch.long)
            return x
        single = (x.dim() == 3)
        if single:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape
        half = self.shift_range // 2
        # randint high is exclusive; +1 so positive endpoint is included.
        shifts = torch.randint(-half, half + 1, (B,), device=x.device)
        # torch.roll can't take per-sample shifts directly; use index_select
        # or a per-sample loop. Vectorize via broadcasted gather on theta.
        idx = (torch.arange(H, device=x.device)[None, :]
               - shifts[:, None]) % H          # (B, H)
        idx = idx[:, None, :, None].expand(B, C, H, W)
        out = torch.gather(x, 2, idx)
        self.last_shifts = shifts.detach().cpu()
        return out.squeeze(0) if single else out


# =========================================================================
# 5. AUGMENTATIONS
# =========================================================================

_AUG_NAMES = {"hflip", "vflip", "colorjitter", "blur", "centermask"}


def get_contrastive_transforms(disable=None,
                               center_mask_radius: int = 15,
                               center_crop_size: int = 140,
                               polar_size: int = 192,
                               polar_mask_cols: int = 30,
                               theta_roll_aug: bool = True,
                               theta_shift_range: int | None = None,
                               theta_shift_range_student: int | None = None,
                               theta_shift_range_teacher: int | None = None,
                               com_centering: bool = False,
                               com_search_radius_factor: float = 2.0,
                               # Per-augmentation params (used only
                               # when the matching tag is NOT in
                               # `disable`).  Defaults match the
                               # previous hardcoded values.
                               cj_brightness: float = 0.2,
                               cj_contrast:   float = 0.2,
                               blur_kernel_max: int = 5,
                               blur_sigma_max:  float = 0.3):
    """Build student / teacher aug pipelines for the contrastive run.

    Pipeline order:
      Cartesian aug (no rotation)
        -> Resize(polar_size)
        -> PolarTransform(polar_size)
        -> PolarMaskLeft(polar_mask_cols)
        -> PolarThetaRoll(theta_shift_range)   [if theta_roll_aug]

    Cartesian RandomRotation is intentionally omitted per spec 7 -- rotation
    is handled exclusively by PolarThetaRoll now.

    Asymmetric theta-roll (matches the older validated pipeline where the
    student got RandomRotation(360) and the teacher got RandomRotation(15)):
      - `theta_shift_range_student`: student's theta-roll range. If not
        provided, falls back to `theta_shift_range`, otherwise to
        `polar_size` (full +/- 180 degrees).
      - `theta_shift_range_teacher`: teacher's theta-roll range. If not
        provided, falls back to `theta_shift_range`, otherwise to
        `max(2, round(polar_size * 15 / 180))` (roughly +/- 15 degrees,
        the original teacher Cartesian-rotation magnitude).

    Pass `theta_shift_range` to force both views to the same range (the
    symmetric setup used in the initial 3x50-epoch run).
    """
    if disable is None:
        disable = set()
    elif isinstance(disable, str):
        disable = {disable}
    else:
        disable = set(disable)
    unknown = disable - _AUG_NAMES
    if unknown:
        raise ValueError(f"Unknown aug names: {unknown}")

    # Resolve per-view ranges.
    if theta_shift_range_student is None:
        theta_shift_range_student = (theta_shift_range
                                      if theta_shift_range is not None
                                      else polar_size)
    if theta_shift_range_teacher is None:
        if theta_shift_range is not None:
            theta_shift_range_teacher = theta_shift_range
        else:
            # ~+/-15 degrees: polar_size rows span 360 degrees, so
            # 15 / 180 * polar_size rows is the total range.
            theta_shift_range_teacher = max(2, int(round(polar_size * 15 / 180)))

    student_ops, teacher_ops = [], []
    student_ops.append(T.CenterCrop(center_crop_size))
    # COM centering BEFORE any flipping/jitter, so the COM lookup uses the
    # raw (cropped) intensities.  Historically the search radius was
    # ``com_search_radius_factor * center_mask_radius`` — a coupling that
    # forced ``center_mask_radius > 0`` even when the equivalent low-r
    # masking was being done in polar space via ``polar_mask_cols``.
    # Now: if ``center_mask_radius == 0`` we fall back to deriving the
    # search radius from ``polar_mask_cols / 2`` (the cart-frame radius
    # the polar mask actually covers), so the GUI can pass 0 for the
    # legacy field without silently breaking COM.
    if com_centering:
        eff_beam_r = (center_mask_radius if center_mask_radius > 0
                      else max(1, polar_mask_cols // 2))
        com_search_radius_post_crop = int(
            com_search_radius_factor * eff_beam_r)
        student_ops.append(CenterOnCOM(search_radius=com_search_radius_post_crop))
        teacher_ops.append(T.CenterCrop(center_crop_size))
        teacher_ops.append(CenterOnCOM(search_radius=com_search_radius_post_crop))
        # teacher ops continue below (we already added CenterCrop+CenterOnCOM here)
        teacher_added_crop = True
    else:
        teacher_added_crop = False
    if "hflip" not in disable:
        student_ops.append(T.RandomHorizontalFlip(p=0.5))
    if "vflip" not in disable:
        student_ops.append(T.RandomVerticalFlip(p=0.5))
    if "colorjitter" not in disable:
        student_ops.append(T.ColorJitter(
            brightness=float(cj_brightness),
            contrast=float(cj_contrast)))
    if "blur" not in disable:
        # Kernel size must be ODD; clamp + force odd.
        kmax = max(3, int(blur_kernel_max))
        if kmax % 2 == 0: kmax += 1
        smax = max(0.11, float(blur_sigma_max))
        student_ops.append(T.GaussianBlur(
            kernel_size=(3, kmax), sigma=(0.1, smax)))
    student_ops.append(T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR,
                                 antialias=True))

    if not teacher_added_crop:
        teacher_ops.append(T.CenterCrop(center_crop_size))
    teacher_ops.append(T.Resize(polar_size,
                                 interpolation=InterpolationMode.BILINEAR,
                                 antialias=True))

    # Polar transform (LAST geometric step in Cartesian).
    student_ops.append(PolarTransform(output_size=polar_size))
    teacher_ops.append(PolarTransform(output_size=polar_size))
    if polar_mask_cols > 0:
        student_ops.append(PolarMaskLeft(k_cols=polar_mask_cols))
        teacher_ops.append(PolarMaskLeft(k_cols=polar_mask_cols))

    # theta-roll is per-view: build two independent modules so each view
    # gets its own RNG state AND its own range.
    student_roll = PolarThetaRoll(
        shift_range=theta_shift_range_student if theta_roll_aug else 0)
    teacher_roll = PolarThetaRoll(
        shift_range=theta_shift_range_teacher if theta_roll_aug else 0)
    student_ops.append(student_roll)
    teacher_ops.append(teacher_roll)

    return T.Compose(student_ops), T.Compose(teacher_ops), student_roll, teacher_roll


def get_cartesian_transforms(disable=None,
                              center_mask_radius: int = 15,
                              center_crop_size: int = 140,
                              polar_size: int = 192,       # used as resize target
                              rotation_range_student: float = 180.0,
                              rotation_range_teacher: float = 15.0):
    """Cartesian-only aug pipeline (no PolarTransform, no theta-roll, and the
    backbone should be built with `circular_theta=False`).

    Pipeline order (student):
      CenterCrop -> RandomRotation(+-180 deg) -> hflip / vflip / ColorJitter / Blur
      -> Resize(polar_size) -> CenterMask(radius)

    CenterMask is LAST so the beam disk stays sharp (no interpolation drift
    from the preceding rotation). The teacher gets the same sequence with
    RandomRotation(+-15) and no flips / jitter / blur.

    Returns dummy `PolarThetaRoll(shift_range=0)` modules in the student /
    teacher slots so the training loop's asymmetry-check path works
    unchanged (both views report zero shifts).
    """
    if disable is None:
        disable = set()
    elif isinstance(disable, str):
        disable = {disable}
    else:
        disable = set(disable)
    unknown = disable - _AUG_NAMES
    if unknown:
        raise ValueError(f"Unknown aug names: {unknown}")

    student_ops = [
        T.CenterCrop(center_crop_size),
        T.RandomRotation(rotation_range_student,
                          interpolation=InterpolationMode.BILINEAR, expand=False),
    ]
    if "hflip" not in disable:
        student_ops.append(T.RandomHorizontalFlip(p=0.5))
    if "vflip" not in disable:
        student_ops.append(T.RandomVerticalFlip(p=0.5))
    if "colorjitter" not in disable:
        student_ops.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    if "blur" not in disable:
        student_ops.append(T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 0.3)))
    student_ops.append(T.Resize(polar_size,
                                 interpolation=InterpolationMode.BILINEAR,
                                 antialias=True))
    if center_mask_radius > 0 and "centermask" not in disable:
        student_ops.append(CenterMask(radius=center_mask_radius))

    teacher_ops = [
        T.CenterCrop(center_crop_size),
        T.RandomRotation(rotation_range_teacher,
                          interpolation=InterpolationMode.BILINEAR, expand=False),
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR,
                 antialias=True),
    ]
    if center_mask_radius > 0 and "centermask" not in disable:
        teacher_ops.append(CenterMask(radius=center_mask_radius))

    # Dummy rolls (shift_range=0) for training-loop asymmetry-check compat.
    student_roll = PolarThetaRoll(shift_range=0)
    teacher_roll = PolarThetaRoll(shift_range=0)
    return T.Compose(student_ops), T.Compose(teacher_ops), student_roll, teacher_roll


# =========================================================================
# 6. MODEL
# =========================================================================

@torch.no_grad()
def sinkhorn_knopp(logits: torch.Tensor,
                    epsilon: float = 0.05,
                    n_iters: int = 3) -> torch.Tensor:
    """Sinkhorn-Knopp normalization on teacher prototype logits.

    Produces a doubly-stochastic-ish soft-assignment matrix Q (B, K)
    where rows sum to 1 (per-sample assignment) and column marginals
    are pushed toward uniform 1/K (equipartition across prototypes).
    This is the SwAV / DINOv2-with-SK / MSN target — replaces DINO's
    centering+sharpening recipe with an explicit balance constraint.

    Args:
        logits  : (B, K) raw teacher prototype logits
        epsilon : entropic regularizer (smaller = sharper Q)
        n_iters : number of Sinkhorn iterations (3-5 is standard)

    Returns:
        Q : (B, K) soft assignment, rows sum to 1
    """
    Q = (logits / float(epsilon)).exp()                     # (B, K)
    # Numerical stability — start from a global rescale.
    Q = Q / Q.sum().clamp_min(1e-12)
    B, K = Q.shape
    for _ in range(int(n_iters)):
        # Column normalization → each prototype gets total mass 1/K.
        Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        Q = Q / float(K)
        # Row normalization → each sample sums to 1/B.
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        Q = Q / float(B)
    # Final rescale so each row sums to 1 (per-sample distribution).
    Q = Q * float(B)
    return Q


class ContrastiveDINOModel(nn.Module):
    """DINO-SR L1 backbone + prototype head + contrastive projection head.

    Optional additional centroid loss (`use_centroid_loss=True`) adds a
    learnable (K, contrastive_out) centroid matrix in contrastive-embedding
    space. The centroid loss pulls each sample's embedding toward a
    softly-weighted centroid (weights = teacher prototype softmax) and
    margin-repels the centroid matrix's own off-diagonal pairs. This is
    additive to the pairwise contrastive loss; set `lam_cent=0` to disable.

    `target_mode` selects the teacher target distribution:
        - "dino"     : center + softmax(/temp_teacher)  [default]
        - "sinkhorn" : Sinkhorn-Knopp normalization (SwAV / DINOv2-SK style).
                       Centering buffer is unused. `sinkhorn_eps` and
                       `sinkhorn_iters` control the SK iteration.
    """

    def __init__(self,
                 n_layers: int = 1,
                 num_prototypes: int = 10,
                 proj_out_dim: int = 256,
                 contrastive_hidden: int = 256,
                 contrastive_out: int = 128,
                 use_maxpool: bool = True,
                 w_ent: float = 0.0,
                 center_momentum: float = 0.9,
                 T0: float = 0.07, Tfin: float = 0.07,
                 EMA0: float = 0.990, EMAfin: float = 0.999,
                 warmup_frac: float = 0.2,
                 circular_theta: bool = True,
                 use_centroid_loss: bool = False,
                 conf_weight_gamma: float = 0.0,
                 architecture: str = "resnet",
                 vit_img_size: int = 192,
                 vit_patch_size: int = 16,
                 vit_embed_dim: int = 192,
                 vit_depth: int = 12,
                 vit_num_heads: int = 3,
                 target_mode: str = "dino",
                 sinkhorn_eps: float = 0.05,
                 sinkhorn_iters: int = 3):
        super().__init__()
        assert architecture in ("resnet", "vit"), f"unknown architecture {architecture!r}"
        self.architecture = architecture
        self.n_layers = n_layers
        if architecture == "vit":
            self.student_encoder = VisionTransformerTiny(
                img_size=vit_img_size, patch_size=vit_patch_size,
                in_chans=1, embed_dim=vit_embed_dim,
                depth=vit_depth, num_heads=vit_num_heads,
            )
            # ViT trunk lacks a true conv trunk, so there's nothing for
            # circular-theta to wrap. Caller should set circular_theta=False
            # when architecture=vit. We don't silently alter user's input;
            # the flag just becomes a no-op.
        else:
            self.student_encoder = create_encoder_resnet18_variableN(
                n_layers=n_layers, use_maxpool=use_maxpool,
                circular_theta=circular_theta)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        in_dim = self.student_encoder.out_dim
        # DINO heads (unchanged from ablation model).
        self.student_projector = ProjectionHead(in_dim, 512, proj_out_dim)
        self.teacher_projector = ProjectionHead(in_dim, 512, proj_out_dim)
        self.prototypes = Prototypes(num_prototypes, proj_out_dim)
        # Contrastive head (student-only; teacher only provides soft labels
        # via its prototype softmax, not via the contrastive projection).
        self.student_contrastive = ProjectionMLP(in_dim,
                                                  contrastive_hidden,
                                                  contrastive_out)
        # Optional per-class centroid bank for the centroid loss. Same
        # dimensionality as the contrastive embedding.
        self.use_centroid_loss = bool(use_centroid_loss)
        if self.use_centroid_loss:
            self.contrastive_centroids = nn.Parameter(
                torch.randn(num_prototypes, contrastive_out)
                / math.sqrt(max(contrastive_out, 1)))
        else:
            self.register_parameter("contrastive_centroids", None)
        self.register_buffer("center", torch.zeros(1, num_prototypes))
        self.center_momentum = float(center_momentum)
        self.T0, self.Tfin = T0, Tfin
        self.EMA0, self.EMAfin = EMA0, EMAfin
        self.warmup_frac, self.w_ent = warmup_frac, w_ent
        self.conf_weight_gamma = float(conf_weight_gamma)
        if target_mode not in ("dino", "sinkhorn"):
            raise ValueError(
                f"target_mode must be 'dino' or 'sinkhorn', got {target_mode!r}")
        self.target_mode = target_mode
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        for p in self.teacher_projector.parameters():
            p.requires_grad = False

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------
    def forward(self, x_student, x_teacher,
                temp_student: float = 0.1, temp_teacher: float = 0.06):
        feat_s = self.student_encoder(x_student)
        proj_s = self.student_projector(feat_s)
        logits_s = self.prototypes(proj_s)
        embed_s = self.student_contrastive(feat_s)  # already L2-normed

        self.teacher_encoder.eval()
        self.teacher_projector.eval()
        with torch.no_grad():
            feat_t = self.teacher_encoder(x_teacher)
            proj_t = self.teacher_projector(feat_t)
            raw_t = self.prototypes(proj_t)

        if self.target_mode == "sinkhorn":
            # SwAV / DINOv2-SK style — equipartition via Sinkhorn-Knopp.
            # Centering buffer is unused; sinkhorn_eps replaces temp_teacher
            # as the sharpness control, sinkhorn_iters as the balance
            # iterations. We expose the SK-input under `pt` for downstream
            # losses that consume teacher softmax (contrastive, centroid,
            # supcon, cluster1d) so they don't need to branch.
            pt = sinkhorn_knopp(raw_t, epsilon=self.sinkhorn_eps,
                                  n_iters=self.sinkhorn_iters)
        else:
            tl = raw_t - self.center
            pt = F.softmax(tl / temp_teacher, dim=-1)                   # (B, K)
        lps = F.log_softmax(logits_s / temp_student, dim=-1)            # (B, K)
        per_sample = -(pt.detach() * lps).sum(dim=-1)                   # (B,)
        if self.conf_weight_gamma > 0:
            # Weight each sample's KL by teacher peak prob^gamma, then
            # renormalize so loss magnitude stays comparable to unweighted.
            w = pt.detach().max(dim=-1).values ** self.conf_weight_gamma
            loss_dino = (w * per_sample).sum() / w.sum().clamp_min(1e-6)
        else:
            loss_dino = per_sample.mean()
        if self.w_ent > 0:
            ps = F.softmax(logits_s / temp_student, dim=-1).mean(dim=0)
            loss_dino = loss_dino - self.w_ent * -(ps * ps.clamp_min(1e-6).log()).sum()
        return {
            "loss_dino": loss_dino,
            "pt": pt,
            "raw_teacher_logits": raw_t,
            "student_embed": embed_s,
            "logits_s": logits_s,
        }

    @staticmethod
    def contrastive_loss(student_embed: torch.Tensor,
                          teacher_softmax: torch.Tensor,
                          entropy_gate: bool = False):
        """Soft-label contrastive MSE.

        target[i,j] = cos(pt_i, pt_j)   -- teacher prototype softmax, detached
        pred[i,j]   = <embed_i, embed_j>  -- student contrastive embedding

        Loss is MSE over off-diagonal pairs. If entropy_gate=True, only
        samples with teacher softmax entropy below the batch median contribute
        (per-batch quantile). Returns (loss, frac_used).
        """
        B = student_embed.shape[0]
        device = student_embed.device
        pt_n = F.normalize(teacher_softmax.detach(), dim=-1)
        target = pt_n @ pt_n.t()
        pred = student_embed @ student_embed.t()

        mask = 1.0 - torch.eye(B, device=device, dtype=pred.dtype)
        if entropy_gate:
            ent = -(teacher_softmax * teacher_softmax.clamp_min(1e-12).log()).sum(-1)
            keep = (ent < ent.median()).to(pred.dtype)
            mask = mask * keep[:, None] * keep[None, :]
            frac_used = keep.mean().item()
        else:
            frac_used = 1.0

        denom = mask.sum().clamp_min(1.0)
        loss = ((pred - target) ** 2 * mask).sum() / denom
        return loss, frac_used

    @staticmethod
    def centroid_loss(student_embed: torch.Tensor,
                       teacher_softmax: torch.Tensor,
                       centroids: torch.Tensor,
                       margin: float = 0.3):
        """Additive centroid loss.

        L_intra: soft pull — each sample's target is a teacher-softmax-
          weighted combination of centroids, L2-normed. 1 - cos(z, tgt).
        L_inter: margin-repel between centroid pairs, squared hinge on
          cos-sim exceeding `margin`.

        Returns (l_intra, l_inter) so the training loop can log them
        separately and combine with chosen weights.
        """
        c_norm = F.normalize(centroids, dim=-1)
        pt = teacher_softmax.detach()
        tgt = pt @ c_norm                              # (B, D)
        tgt = F.normalize(tgt, dim=-1)
        l_intra = (1.0 - (student_embed * tgt).sum(-1)).mean()
        K = c_norm.size(0)
        sim = c_norm @ c_norm.t()
        off_diag = 1.0 - torch.eye(K, device=sim.device, dtype=sim.dtype)
        denom = max(K * (K - 1), 1)
        l_inter = (F.relu(sim - margin) ** 2 * off_diag).sum() / denom
        return l_intra, l_inter

    # -------------------------------------------------------------
    # Teacher update / center / buffer sync (unchanged from ablation)
    # -------------------------------------------------------------
    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996):
        for sp, tp in zip(self.student_encoder.parameters(),
                          self.teacher_encoder.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)
        for sp, tp in zip(self.student_projector.parameters(),
                          self.teacher_projector.parameters()):
            tp.data.mul_(momentum).add_(sp.data, alpha=1 - momentum)

    @torch.no_grad()
    def sync_teacher_bn_buffers(self):
        for (_, ms), (_, mt) in zip(self.student_encoder.named_modules(),
                                     self.teacher_encoder.named_modules()):
            if isinstance(ms, nn.modules.batchnorm._BatchNorm):
                mt.running_mean.copy_(ms.running_mean)
                mt.running_var.copy_(ms.running_var)
        for (_, ms), (_, mt) in zip(self.student_projector.named_modules(),
                                     self.teacher_projector.named_modules()):
            if isinstance(ms, nn.modules.batchnorm._BatchNorm):
                mt.running_mean.copy_(ms.running_mean)
                mt.running_var.copy_(ms.running_var)

    @torch.no_grad()
    def update_center(self, raw_teacher_logits: torch.Tensor):
        bc = raw_teacher_logits.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + bc * (1 - self.center_momentum)

    def _num_prototypes(self) -> int:
        return self.prototypes.prototypes.size(0)


# =========================================================================
# 7. LAMBDA WARMUP
# =========================================================================

def lambda_schedule(epoch: int, target: float,
                    warmup_epochs: int = 20,
                    ramp_epochs: int = 10) -> float:
    """0 for epoch < warmup_epochs, linear ramp to target over ramp_epochs,
    then target. `epoch` is 0-indexed."""
    if target <= 0:
        return 0.0
    if epoch < warmup_epochs:
        return 0.0
    if epoch < warmup_epochs + ramp_epochs:
        frac = (epoch - warmup_epochs + 1) / max(1, ramp_epochs)
        return float(target * frac)
    return float(target)


# =========================================================================
# 8. THETA-ROLL EQUIVARIANCE UNIT TEST
# =========================================================================

@torch.no_grad()
def theta_roll_equivariance_test(encoder: PlainSequentialEncoder,
                                  x: torch.Tensor,
                                  k_theta: int,
                                  total_stride: int,
                                  r_edge: int = 3):
    """Verify the encoder is (up to stride) equivariant to theta-roll.

    Forwards `x` and `roll(x, k_theta, dim=-2)` through the convolutional
    trunk (pre-pool), un-rolls the second feature map by k_theta //
    total_stride, and returns (interior_max_abs_diff, full_max_abs_diff).

    r_edge: number of edge r-columns to skip on each side when computing the
    interior diff. Zero-padding on r produces legitimate drift there; the
    interior should be essentially bit-exact (< 1e-6 at float32, much
    smaller if the stride-division is exact).
    """
    encoder.eval()
    feat1 = encoder(x, return_feature_map=True)
    x_rolled = torch.roll(x, k_theta, dims=-2)
    feat2 = encoder(x_rolled, return_feature_map=True)
    k_feat = k_theta // total_stride
    feat2_aligned = torch.roll(feat2, -k_feat, dims=-2)
    diff = (feat1 - feat2_aligned).abs()
    full = float(diff.max().item())
    if diff.size(-1) > 2 * r_edge:
        interior = diff[..., r_edge: diff.size(-1) - r_edge].max().item()
    else:
        interior = full
    return float(interior), full


# =========================================================================
# 9. TRAINING LOOP
# =========================================================================

def _teacher_softmax_entropy_stats(pt: torch.Tensor):
    """Return (mean, p10, p90) of per-sample teacher softmax entropy."""
    ent = -(pt * pt.clamp_min(1e-12).log()).sum(-1)
    vals = ent.detach().cpu().numpy()
    return float(vals.mean()), float(np.percentile(vals, 10)), float(np.percentile(vals, 90))


class _IndexedDataset(torch.utils.data.Dataset):
    """Wraps a base dataset so __getitem__ returns (pattern, dataset_index).
    Used to look up per-pattern radial profiles by index in the SupCon loss.
    """
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        return item, idx
    def __getattr__(self, name):
        # Forward attribute access (e.g., .vmax, .resize) to the base dataset.
        if name.startswith("_") or name in ("base",):
            raise AttributeError(name)
        return getattr(self.base, name)


def prototype_repel_loss(prototype_weights: torch.Tensor,
                          threshold: float = 0.5):
    """Hinge-style prototype-prototype repulsion. Penalizes pairs of prototype
    vectors whose cosine similarity exceeds `threshold`. Pairs already below
    the threshold incur no penalty -- well-separated prototypes are left alone.

    Maps to the "prototype scattering" / "anti-partial-collapse" idea
    (cf. ProPos TPAMI 2022; arXiv:2510.20108). Compatible with DINO centering:
    centering keeps the per-prototype usage non-zero, repulsion ensures the
    alive prototypes are distinct directions.

    prototype_weights: (K, D) -- raw prototype matrix (will be L2-normed
                                  internally, no in-place change)
    Returns: (loss, mean_off_diag_cos, max_off_diag_cos)
    """
    K = prototype_weights.shape[0]
    p = F.normalize(prototype_weights, dim=1)
    cos = p @ p.t()                                       # (K, K)
    eye = torch.eye(K, device=cos.device, dtype=cos.dtype)
    off = cos - eye                                       # zero on diagonal
    excess = F.relu(off - threshold)                      # only the > threshold portion
    loss = (excess ** 2).sum() / max(K * (K - 1), 1)
    with torch.no_grad():
        # Stats over off-diagonal only.
        flat = (cos * (1 - eye)).flatten()
        mean_off = float(flat.sum().item() / max(K * (K - 1), 1))
        max_off = float((cos - 2.0 * eye).max().item())
    return loss, mean_off, max_off


def supcon_radial_loss(embeds: torch.Tensor,
                        radials: torch.Tensor,
                        tau_pos: float, tau_neg: float,
                        temperature: float = 0.1):
    """Supervised contrastive loss with positive/negative pairs gated by
    cosine similarity of (mean-centered, log-normalized) radial profiles.

    embeds: (B, D)  -- expected L2-normed (model returns normed embeddings)
    radials: (B, R) -- mean-centered log-normalized profiles
    Returns: (loss, n_pos_per_anchor_mean, n_neg_per_anchor_mean,
              n_abstain_per_anchor_mean, frac_anchors_with_pos)
    """
    B = embeds.size(0)
    device = embeds.device
    eye = torch.eye(B, device=device, dtype=embeds.dtype)

    # Normalize radials for cosine (they are mean-centered, not unit).
    rad_n = F.normalize(radials, dim=1)
    rad_sim = rad_n @ rad_n.t()                                # (B, B) cosine

    pos_mask = (rad_sim > tau_pos).to(embeds.dtype) * (1.0 - eye)
    neg_mask = (rad_sim < tau_neg).to(embeds.dtype) * (1.0 - eye)

    n_pos = pos_mask.sum(dim=1)                                # (B,)
    n_neg = neg_mask.sum(dim=1)
    n_abst = (B - 1) - n_pos - n_neg

    # Embedding cosine / temperature.
    z_sim = (embeds @ embeds.t()) / temperature                # (B, B)
    z_sim_max = z_sim.max(dim=1, keepdim=True).values.detach()
    z_sim_stable = z_sim - z_sim_max

    # Denominator includes positives AND negatives only (no self, no abstain).
    valid_mask = pos_mask + neg_mask                           # (B, B)
    exp_sim = torch.exp(z_sim_stable) * valid_mask
    denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = z_sim_stable - torch.log(denom)                 # (B, B)

    # Mean log-prob over positives, per anchor.
    valid_anchors = (n_pos > 0).to(embeds.dtype)
    mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / n_pos.clamp(min=1.0)
    loss = -(mean_log_prob_pos * valid_anchors).sum() \
            / valid_anchors.sum().clamp_min(1e-12)

    return (loss,
            float(n_pos.mean().item()),
            float(n_neg.mean().item()),
            float(n_abst.mean().item()),
            float(valid_anchors.mean().item()))


def cluster1d_loss(p_student: torch.Tensor,
                    radials: torch.Tensor,
                    margin: float = 0.4,
                    min_cluster_mass: float = 1.0):
    """Physics-gated CLUSTERING loss on 1D radial profiles.

    Two terms:
        L_intra: each pattern's 1D profile pulled toward the soft centroid
                 of whichever cluster it's most likely in. Centroid is
                 detached -- gradient flows through p only (assignment).
        L_inter: pairwise centroid cosines pushed below `margin` (hinge).
                 Gradient flows through batch centroids -> p -> encoder/head.

    Together they say: members of a cluster share 1D shape; clusters'
    1D shapes differ. Acts on the prototype-assignment head directly,
    unlike supcon_radial_loss which acts on the embedding.

    p_student: (B, K) softmax(student_prototype_logits / temp).
    radials:   (B, n_bins) precomputed mean-centered 1D profiles.
    margin:    hinge for L_inter (centroid cos pairs above this contribute).
    min_cluster_mass: cluster c with sum_i(p[i,c]) below this is excluded
                     from L_inter (its centroid is too noisy to push).

    Returns: (l_intra, l_inter, mean_off_diag_cos, max_off_diag_cos)
    """
    B, K = p_student.shape
    eps = 1e-8

    N_c = p_student.sum(dim=0)                                    # (K,)
    r_bar = (p_student.t() @ radials) / (N_c.unsqueeze(-1) + eps) # (K, n_bins)

    rad_n = F.normalize(radials, dim=-1)
    rb_n_target = F.normalize(r_bar.detach(), dim=-1)
    cos_iC = rad_n @ rb_n_target.t()                              # (B, K)
    l_intra = -(p_student * cos_iC).sum(dim=-1).mean()

    rb_n = F.normalize(r_bar, dim=-1)
    C = rb_n @ rb_n.t()                                           # (K, K)

    eye = torch.eye(K, device=p_student.device, dtype=p_student.dtype)
    off_diag = 1.0 - eye
    # Smooth mass weighting on the inter pressure (replaces the
    # binary `min_cluster_mass` cliff): pairs where BOTH prototypes
    # carry significant mass push apart strongly; pairs where either
    # is under-used barely push at all. This lets weak prototypes
    # drift toward similar centroids without being kicked away,
    # which causes them to alias / merge naturally over training
    # (passive K-shrinkage).
    N_c_norm = N_c / N_c.sum().clamp_min(eps)                     # (K,) sum = 1
    mass_w   = N_c_norm.unsqueeze(0) * N_c_norm.unsqueeze(1)      # (K, K)
    mask     = off_diag * mass_w
    mask_sum = mask.sum().clamp_min(eps)
    l_inter  = (F.relu(C - margin) * mask).sum() / mask_sum
    # Keep `valid` for the diagnostic mean/max-off-diag stats below
    # so callers that read those fields don't change behaviour.
    valid = (N_c >= min_cluster_mass).to(p_student.dtype)

    with torch.no_grad():
        n_off = off_diag.sum().clamp_min(1.0)
        mean_off = float((C * off_diag).sum().item() / float(n_off.item()))
        C_for_max = C - 2.0 * eye
        max_off = float(C_for_max.max().item())

    return l_intra, l_inter, mean_off, max_off


def pair_assignment_loss(p_a: torch.Tensor,
                            p_b: torch.Tensor,
                            y: torch.Tensor,
                            entropy_reg: float = 0.0):
    """Pair-supervised loss on the prototype-assignment simplex.

    Inputs
    ------
    p_a, p_b : (P, K) softmax(student_prototype_logits / temp) for
               the two members of each labelled pair.
    y        : (P,)   ±1 — +1 for "same physical phase", -1 for
                       "different".
    entropy_reg : optional non-negative entropy regulariser on the
                  pair members. Without this, sparse labels can collapse
                  `p` toward one-hot just to satisfy the dot products.

    Loss formulation
    ----------------
    For each pair (i, j, y):
        L_pair_ij = -y * <p_i, p_j>             # cosine in simplex
    averaged over labelled pairs in the batch. Lower = better.

      y=+1  -> minimise -<p_i, p_j>  (push the two assignments onto the
                                      same prototype face)
      y=-1  -> minimise +<p_i, p_j>  (push them onto different faces)

    The dot product <p_i, p_j> = sum_c p_i[c] p_j[c] is bounded in
    [0, 1]; cosine on the simplex without re-normalisation. Sufficient
    in practice and avoids the normalisation having to backprop through
    a square root for a tiny effect.

    Optional entropy regulariser:
        L_ent = -mean(H(p_i) + H(p_j))    # add entropy_reg * L_ent
    with H(p) = -sum_c p[c] log(p[c] + eps).  Higher entropy (softer
    assignment) is *encouraged*: this counteracts the pair loss's pull
    toward one-hot on labelled samples.

    Returns
    -------
    L_pair  : scalar tensor (gradient flows through p_a, p_b)
    n_used  : int — number of valid pairs (so the caller can log this
              and detect zero-pair batches without dividing by zero in
              the loss).
    """
    eps = 1e-12
    P, K = p_a.shape
    if P == 0 or p_a.shape != p_b.shape or y.numel() != P:
        # No valid pairs — return a zero scalar that still has a grad
        # graph attached (so the optimiser doesn't choke on a Python 0).
        z = (p_a.sum() * 0.0) + (p_b.sum() * 0.0)
        return z, 0

    dot = (p_a * p_b).sum(dim=-1)                       # (P,)
    y = y.to(dot.dtype)
    L = -(y * dot).mean()

    if entropy_reg > 0.0:
        H_a = -(p_a * (p_a + eps).log()).sum(dim=-1)    # (P,)
        H_b = -(p_b * (p_b + eps).log()).sum(dim=-1)
        L_ent = -(H_a + H_b).mean()                     # negate so adding
                                                          # this term encourages
                                                          # higher entropy
        L = L + float(entropy_reg) * L_ent

    return L, int(P)


def train_contrastive(model: ContrastiveDINOModel,
                      dataset,
                      epochs: int = 50,
                      lr: float = 3e-4,
                      weight_decay: float = 1e-6,
                      device=None,
                      outdir: str = "./",
                      save_every: int = 10,
                      seed: int = 42,
                      batch_size: int = 128,
                      student_aug=None, teacher_aug=None,
                      student_roll=None, teacher_roll=None,
                      contrastive_lambda: float = 0.2,
                      contrastive_warmup_epochs: int = 20,
                      contrastive_ramp_epochs: int = 10,
                      contrastive_entropy_gate: bool = False,
                      centroid_lambda: float = 0.0,
                      centroid_margin: float = 0.3,
                      centroid_inter_weight: float = 0.5,
                      lam_spatial: float = 0.0,
                      scan_shape=None,
                      # Gated 4-neighbor spatial loss thresholds.
                      # When supcon_radials is provided, the spatial
                      # Potts loss is GATED by 1D radial agreement
                      # between scan-neighbor pairs:
                      #   pull (1 - <p_i,p_j>) when cos(R_i,R_j) > tau_pos
                      #   push   <p_i,p_j>    when cos(R_i,R_j) < tau_neg
                      # In the dead band [tau_neg, tau_pos] the pair
                      # contributes nothing (uncertain physics ->
                      # don't impose anything). Without radials,
                      # falls back to the unconditional Potts loss.
                      spatial_tau_pos: float = 0.5,
                      spatial_tau_neg: float = 0.0,
                      grad_clip: float = 1.0,
                      temp_student: float = 0.1,
                      conf_weight_auto: "dict | None" = None,
                      stop_at_epoch: "int | None" = None,
                      supcon_radials: "np.ndarray | None" = None,
                      supcon_tau_pos: float = 0.5,
                      supcon_tau_neg: float = 0.0,
                      supcon_lambda: float = 0.0,
                      supcon_temperature: float = 0.1,
                      proto_repel_lambda: float = 0.0,
                      proto_repel_threshold: float = 0.5,
                      cluster1d_lambda: float = 0.0,
                      # Per-term weights for cluster1d.  When BOTH are
                      # < 0, fall back to `cluster1d_lambda` for both
                      # (backward-compat with callers that still pass
                      # the unified knob).  Setting `intra > 0,
                      # inter = 0` is the "let weak prototypes drift /
                      # die" recipe — keeps physics-baked intra
                      # without anti-shrinkage centroid-repel.
                      cluster1d_lambda_intra: float = -1.0,
                      cluster1d_lambda_inter: float = -1.0,
                      cluster1d_margin: float = 0.4,
                      cluster1d_min_cluster_mass: float = 1.0,
                      cluster1d_warmup_epochs: int = 0,
                      cluster1d_ramp_epochs: int = 0,
                      # ---- pair-supervised assignment loss (pre-train labels)
                      pair_idx_a: "np.ndarray | None" = None,
                      pair_idx_b: "np.ndarray | None" = None,
                      pair_y:     "np.ndarray | None" = None,
                      lambda_pair: float = 0.0,
                      pair_entropy_reg: float = 0.0,
                      pair_per_batch: int = 32,
                      # Restrict the DataLoader to a deterministic
                      # random subset (used by Phase-B fine-tune).
                      # Pair-loss endpoints are NOT restricted — they
                      # always pull from the full dataset by flat
                      # scan index.
                      subsample_n: "int | None" = None):
    """Forked from train_ablation. The contrastive branch is purely
    additive: when contrastive_lambda=0 and theta_roll_aug=False (i.e.,
    student_roll.shift_range == teacher_roll.shift_range == 0), the
    backbone / student_projector / prototypes receive exactly the gradient
    they would under the pure-DINO pipeline.
    """
    # ----- determinism (best-effort) -----
    # CUBLAS_WORKSPACE_CONFIG is read by cuBLAS the first time it does a
    # GEMM; setting it here works as long as no cuBLAS op has fired yet in
    # this process. (run_config does CPU-only work before train_contrastive,
    # so we are still safe.)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as _e:
        print(f"[determinism] use_deterministic_algorithms note: {_e}",
              flush=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(outdir, exist_ok=True)
    model = model.to(device)

    # If radial-gated SupCon or 1D-cluster loss is requested, wrap dataset
    # to also return idx so we can look up the precomputed radials per batch.
    use_supcon = supcon_radials is not None and supcon_lambda > 0
    # Resolve cluster1d intra / inter weights.  Sentinel value -1
    # means "fall back to the unified `cluster1d_lambda`".  Both
    # explicitly set: use them.
    if cluster1d_lambda_intra < 0:
        cluster1d_lambda_intra = float(cluster1d_lambda)
    if cluster1d_lambda_inter < 0:
        cluster1d_lambda_inter = float(cluster1d_lambda)
    use_cluster1d = (supcon_radials is not None
                     and (cluster1d_lambda_intra > 0
                          or cluster1d_lambda_inter > 0))
    use_radials = use_supcon or use_cluster1d
    # KEEP a reference to the underlying dataset BEFORE we wrap it for
    # _IndexedDataset, so the pair-loss path can fetch raw patterns by
    # scan index without going through the (idx, x) wrapper.
    _pair_raw_dataset = dataset
    if use_radials:
        radials_tensor = torch.from_numpy(np.asarray(supcon_radials,
                                                       dtype=np.float32))
        dataset = _IndexedDataset(dataset)

    # Pair-supervised assignment loss (pre-train + fine-tune labels).
    use_pair = (pair_idx_a is not None and pair_idx_b is not None
                and pair_y    is not None and lambda_pair > 0)
    if use_pair:
        pair_idx_a_t = torch.as_tensor(np.asarray(pair_idx_a),
                                          dtype=torch.long, device=device)
        pair_idx_b_t = torch.as_tensor(np.asarray(pair_idx_b),
                                          dtype=torch.long, device=device)
        pair_y_t = torch.as_tensor(np.asarray(pair_y),
                                      dtype=torch.float32, device=device)
        n_pairs_total = int(pair_y_t.numel())
        if n_pairs_total == 0:
            use_pair = False
            print("[pair-loss] zero labelled pairs — disabled.", flush=True)
        else:
            print(f"[pair-loss] enabled: {n_pairs_total} pairs, "
                  f"lambda_pair={lambda_pair}, "
                  f"pair_per_batch={pair_per_batch}, "
                  f"pair_entropy_reg={pair_entropy_reg}", flush=True)
    else:
        n_pairs_total = 0

    # DataLoader with deterministic generator.
    g = torch.Generator(); g.manual_seed(seed)
    if subsample_n is not None and int(subsample_n) > 0 \
            and int(subsample_n) < len(dataset):
        # Fine-tune path: only iterate over a deterministic random
        # subset of the dataset. The dataset itself is unchanged, so
        # pair-loss can still index ANY scan position via
        # `_pair_raw_dataset[i]`.
        n_total = len(dataset)
        n_take = int(subsample_n)
        idxs = torch.randperm(n_total, generator=g)[:n_take].tolist()
        from torch.utils.data import SubsetRandomSampler
        sampler = SubsetRandomSampler(idxs, generator=g)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            num_workers=0, pin_memory=True, sampler=sampler)
        print(f"[finetune] DataLoader restricted to {n_take}/"
              f"{n_total} patterns (SubsetRandomSampler, "
              f"seed={seed}). Pair-loss endpoints unrestricted.",
              flush=True)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, generator=g)

    # Optimizer / scheduler.
    no_decay, decay = _is_bn_or_bias_param(model)
    optimizer = torch.optim.AdamW([
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay,    "weight_decay": weight_decay}], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    # CSV log.
    log_path = os.path.join(outdir, "training_log.csv")
    cols = ["epoch", "avg_loss", "avg_loss_dino", "avg_loss_contrastive",
            "avg_loss_centroid_intra", "avg_loss_centroid_inter",
            "lambda_eff", "lambda_cent_eff",
            "effK", "active_classes", "hard_K_active",
            # FCM-style fuzzy validity (Option C): per-sample peakiness.
            #   pc  in [1/K, 1] : 1 = hard one-hot, 1/K = uniform
            #   mpc in [0, 1]   : (K*pc - 1) / (K - 1), normalized
            #   keff_sample     : mean exp(H(p_i)), per-pattern effective K
            "pc", "mpc", "keff_sample",
            "teacher_ent_mean", "teacher_ent_p10", "teacher_ent_p90",
            "student_embed_norm_mean", "student_embed_norm_std",
            "gate_frac_used", "avg_conf",
            "avg_loss_supcon", "supcon_n_pos", "supcon_n_neg", "supcon_n_abst",
            "avg_loss_repel", "repel_mean_off", "repel_max_off",
            "avg_loss_cluster1d_intra", "avg_loss_cluster1d_inter",
            "cluster1d_mean_off", "cluster1d_max_off",
            "lambda_cluster1d_eff",
            "lambda_cluster1d_intra_eff", "lambda_cluster1d_inter_eff",
            "avg_loss_spatial",
            "avg_loss_spatial_pull", "avg_loss_spatial_push",
            "frac_pairs_pull", "frac_pairs_push",
            "lambda_spatial_eff", "spatial_tile_y0", "spatial_tile_x0",
            "avg_loss_pair", "pair_n_used", "lambda_pair_eff"]
    cf = open(log_path, "w", newline='')
    wr = csv.writer(cf); wr.writerow(cols); cf.flush()

    # Assertion buffer: run the asymmetry check exactly once, on the first
    # batch of the first epoch, to detect RNG-sharing bugs. Any student/teacher
    # pair with equal theta-roll shifts is reported but does not hard-fail
    # (1-in-range collision is legitimate for a single sample; assertion
    # still fires if ALL samples in the batch match).
    asymmetry_checked = False
    asymmetry_report = None

    K = model._num_prototypes()

    # Prepare spatial 4-neighbor Potts loss with Option B "deterministic
    # tile walk" coverage: a single tile of ~batch_size patterns is too
    # small to cover the scan, so we sweep a different tile each epoch
    # in row-major order with stride = tile_size (no overlap).  After
    # min(epochs, n_tiles) epochs every distinct tile location has been
    # seen at least once.  When supcon_radials is provided, each scan-
    # neighbor pair is gated by 1D radial agreement (gated-Potts);
    # without radials, falls back to unconditional Potts.
    use_spatial = (lam_spatial > 0.0 and scan_shape is not None
                    and teacher_aug is not None)
    spatial_tile_locs = []      # list of (y0, x0) for the walk
    spatial_ph = spatial_pw = 0
    spatial_radials_for_gate = None
    spatial_use_gate = False
    if use_spatial:
        Ny, Nx = scan_shape
        pa = min(batch_size, Ny * Nx)
        spatial_ph = int(math.sqrt(pa * Ny / max(Nx, 1)))
        spatial_pw = pa // max(spatial_ph, 1)
        spatial_ph = min(spatial_ph, Ny)
        spatial_pw = min(spatial_pw, Nx)
        # Build deterministic tile-walk: row-major sweep, stride = tile size.
        ny_grid = max(1, (Ny - spatial_ph) // max(spatial_ph, 1) + 1)
        nx_grid = max(1, (Nx - spatial_pw) // max(spatial_pw, 1) + 1)
        for yi in range(ny_grid):
            for xi in range(nx_grid):
                y0 = min(yi * spatial_ph, max(0, Ny - spatial_ph))
                x0 = min(xi * spatial_pw, max(0, Nx - spatial_pw))
                spatial_tile_locs.append((y0, x0))
        # 1D-radial gate uses the same precomputed radial profiles as
        # cluster1d / supcon. If absent, fall back to unconditional Potts.
        spatial_use_gate = (supcon_radials is not None)
        if spatial_use_gate:
            spatial_radials_for_gate = torch.from_numpy(np.asarray(
                supcon_radials, dtype=np.float32))
        gate_msg = (f"GATED tau_pos={spatial_tau_pos} tau_neg={spatial_tau_neg}"
                    if spatial_use_gate else
                    "UNCONDITIONAL (no radials provided)")
        print(f"[L_spatial] tile {spatial_ph}x{spatial_pw} = "
              f"{spatial_ph * spatial_pw} positions, "
              f"{len(spatial_tile_locs)} tile locations sweeping the scan "
              f"(stride={spatial_ph}x{spatial_pw}, no overlap). "
              f"gate: {gate_msg}", flush=True)

    t_start = time.perf_counter()
    eiter = range(epochs)
    if _HAS_TQDM:
        eiter = _tqdm(eiter, desc=os.path.basename(outdir), ncols=100,
                      leave=False, mininterval=5)

    for epoch in eiter:
        model.train()
        tt = get_teacher_temp(epoch, epochs, model.T0, model.Tfin, model.warmup_frac)
        mm = get_teacher_momentum(epoch, epochs, model.EMA0, model.EMAfin)
        lam_eff = lambda_schedule(epoch, contrastive_lambda,
                                   contrastive_warmup_epochs,
                                   contrastive_ramp_epochs)
        lam_cent_eff = lambda_schedule(epoch, centroid_lambda,
                                        contrastive_warmup_epochs,
                                        contrastive_ramp_epochs)
        use_centroid = (centroid_lambda > 0.0
                         and model.contrastive_centroids is not None)

        tot_loss = tot_dino = tot_cont = 0.0
        tot_cent_intra = tot_cent_inter = 0.0
        n_seen = 0
        sp_sum = None       # accumulated SOFT teacher prob per prototype
        hard_sum = None     # accumulated HARD argmax counts per prototype
        conf_sum = 0.0
        ent_mean_sum = ent_p10_sum = ent_p90_sum = 0.0
        # Fuzzy-validity (FCM-style) accumulators.
        #   pc  : partition coefficient = mean_i sum_k p_ik^2
        #         (1 = hard one-hot, 1/K = uniform)
        #   keff_sample_sum : per-sample exp(H(p_i)), summed over samples
        #         (mean is the average effective # prototypes per pattern;
        #          the existing population-level `effK` is exp(H(mean p))
        #          which is a very different number — see project memory).
        pc_sum = 0.0
        keff_sample_sum = 0.0
        norm_mean_sum = 0.0
        norm_sq_sum = 0.0
        gate_frac_sum = 0.0
        n_batches = 0

        # Per-batch SupCon stats (averaged over the epoch)
        supcon_loss_sum = 0.0
        supcon_n_pos_sum = 0.0
        supcon_n_neg_sum = 0.0
        supcon_n_abst_sum = 0.0
        supcon_n_batches = 0
        # Per-batch prototype-repel stats
        repel_loss_sum = 0.0
        repel_mean_off_sum = 0.0
        repel_max_off_running = 0.0
        repel_n_batches = 0
        # Per-batch 1D-cluster loss stats
        c1d_intra_sum = 0.0
        c1d_inter_sum = 0.0
        c1d_mean_off_sum = 0.0
        c1d_max_off_running = -1.0
        c1d_n_batches = 0
        # Per-batch pair-supervised assignment loss stats
        pair_loss_sum = 0.0
        pair_n_used_sum = 0
        pair_n_batches  = 0
        # Per-batch spatial-loss stats (gated pull/push 4-neighbor Potts)
        spatial_loss_sum = 0.0
        spatial_pull_sum = 0.0
        spatial_push_sum = 0.0
        spatial_n_batches = 0
        # Per-epoch tile pick + augmentation + gate precomputation.  The
        # tile sweeps the scan deterministically in row-major order
        # (Option B): each epoch sees a different tile location so over
        # min(epochs, n_tiles) epochs every part of the scan is exercised.
        spx = None
        sps = None
        spatial_gate_pull = None    # (right, down) tensors of pull-weights
        spatial_gate_push = None    # (right, down) tensors of push-weights
        spatial_tile_y0 = spatial_tile_x0 = -1
        spatial_frac_pull = 0.0
        spatial_frac_push = 0.0
        if use_spatial and spatial_tile_locs:
            spatial_tile_y0, spatial_tile_x0 = spatial_tile_locs[
                epoch % len(spatial_tile_locs)]
            ph, pw = spatial_ph, spatial_pw
            si = [r * Nx + c
                  for r in range(spatial_tile_y0, spatial_tile_y0 + ph)
                  for c in range(spatial_tile_x0, spatial_tile_x0 + pw)
                  if r * Nx + c < len(_pair_raw_dataset)]
            sb = torch.stack([
                (_pair_raw_dataset[i][0]
                 if isinstance(_pair_raw_dataset[i], (list, tuple))
                 else _pair_raw_dataset[i])
                for i in si]).to(device).float()
            if sb.dim() == 3:
                sb = sb.unsqueeze(1)
            with torch.no_grad():
                spx = teacher_aug(sb)
            an = len(si)
            sps = (ph, min(pw, an // max(ph, 1)))
            spx = spx[:sps[0] * sps[1]]
            # Precompute the 1D-radial agreement gate for the tile edges.
            if spatial_use_gate:
                tile_idx_t = torch.tensor(
                    si[:sps[0] * sps[1]], dtype=torch.long)
                rad_tile = spatial_radials_for_gate[tile_idx_t]
                rad_n = F.normalize(rad_tile, dim=-1)
                rad_2d = rad_n.view(sps[0], sps[1], -1)
                c_right = (rad_2d[:, :-1] * rad_2d[:, 1:]).sum(-1)
                c_down  = (rad_2d[:-1, :] * rad_2d[1:, :]).sum(-1)
                # Pull when 1D strongly agrees; push when strongly disagrees.
                pull_r = F.relu(c_right - spatial_tau_pos).to(device)
                pull_d = F.relu(c_down  - spatial_tau_pos).to(device)
                push_r = F.relu(spatial_tau_neg - c_right).to(device)
                push_d = F.relu(spatial_tau_neg - c_down ).to(device)
                spatial_gate_pull = (pull_r, pull_d)
                spatial_gate_push = (push_r, push_d)
                # Diagnostics: fraction of edges in each regime
                n_total = c_right.numel() + c_down.numel()
                n_pull  = int((c_right > spatial_tau_pos).sum().item()) \
                        + int((c_down  > spatial_tau_pos).sum().item())
                n_push  = int((c_right < spatial_tau_neg).sum().item()) \
                        + int((c_down  < spatial_tau_neg).sum().item())
                spatial_frac_pull = n_pull / max(1, n_total)
                spatial_frac_push = n_push / max(1, n_total)
        for batch in dataloader:
            if use_radials and isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_raw, batch_idx = batch
                # Some custom datasets return tuples too -- unwrap if needed.
                x = x_raw[0] if isinstance(x_raw, (list, tuple)) else x_raw
            else:
                x = (batch[0] if isinstance(batch, (list, tuple)) else batch)
                batch_idx = None
            x = x.to(device, non_blocking=True).float()
            bs = x.size(0)
            with torch.no_grad():
                xs = student_aug(x.clone())
                xt = teacher_aug(x.clone())
                if (not asymmetry_checked) and (student_roll is not None) \
                        and (teacher_roll is not None) \
                        and student_roll.shift_range > 0:
                    ss = student_roll.last_shifts
                    ts = teacher_roll.last_shifts
                    if ss.numel() > 0 and ts.numel() > 0 and ss.numel() == ts.numel():
                        equal = (ss == ts)
                        n_eq = int(equal.sum().item())
                        if n_eq == ss.numel():
                            asymmetry_report = (
                                f"CRITICAL: student and teacher theta-roll "
                                f"shifts identical for every sample "
                                f"({n_eq}/{ss.numel()}). RNG-sharing bug.")
                            print(f"\n[ASYMMETRY-CHECK] {asymmetry_report}",
                                  flush=True)
                        elif n_eq > 0:
                            asymmetry_report = (
                                f"OK: student/teacher theta-roll differ for "
                                f"{ss.numel() - n_eq}/{ss.numel()} samples "
                                f"(first-batch collision count {n_eq} is "
                                f"consistent with independent RNG over "
                                f"shift_range={student_roll.shift_range}).")
                            print(f"\n[ASYMMETRY-CHECK] {asymmetry_report}",
                                  flush=True)
                        else:
                            asymmetry_report = (
                                f"OK: student/teacher theta-roll differ for "
                                f"all {ss.numel()} samples.")
                            print(f"\n[ASYMMETRY-CHECK] {asymmetry_report}",
                                  flush=True)
                        with open(os.path.join(outdir, "asymmetry_check.txt"),
                                   "w") as f_asym:
                            f_asym.write(asymmetry_report + "\n")
                            f_asym.write(f"student shifts: {ss.tolist()}\n")
                            f_asym.write(f"teacher shifts: {ts.tolist()}\n")
                    asymmetry_checked = True

            optimizer.zero_grad(set_to_none=True)
            out = model(xs, xt, temp_student=temp_student, temp_teacher=tt)
            loss_dino = out["loss_dino"]
            pt = out["pt"]
            embed_s = out["student_embed"]
            raw_t = out["raw_teacher_logits"]
            loss_cont, frac_used = ContrastiveDINOModel.contrastive_loss(
                embed_s, pt, entropy_gate=contrastive_entropy_gate)
            loss = loss_dino + lam_eff * loss_cont
            # Radial-gated SupCon (between-sample, physics-supervised).
            if use_supcon and batch_idx is not None:
                rad_b = radials_tensor[batch_idx].to(device, non_blocking=True)
                l_supcon, n_pos_b, n_neg_b, n_abst_b, frac_w_pos = \
                    supcon_radial_loss(embed_s, rad_b,
                                        tau_pos=supcon_tau_pos,
                                        tau_neg=supcon_tau_neg,
                                        temperature=supcon_temperature)
                # Only add if at least some anchors had positives -- avoids
                # NaNs when a batch happens to have no positive pairs at all.
                if frac_w_pos > 0:
                    loss = loss + supcon_lambda * l_supcon
                with torch.no_grad():
                    supcon_loss_sum += float(l_supcon.item()) * bs
                    supcon_n_pos_sum += n_pos_b * bs
                    supcon_n_neg_sum += n_neg_b * bs
                    supcon_n_abst_sum += n_abst_b * bs
                    supcon_n_batches += bs
            # 1D-cluster loss: physics-gated CLUSTERING (not embedding).
            # Acts on student soft assignments + per-cluster 1D centroids.
            # Intra and inter terms are weighted independently so the
            # user can set inter=0 (lets weak prototypes drift / die
            # naturally; aligns hard_K_active with the data-driven
            # natural K).
            if use_cluster1d and batch_idx is not None:
                lam_c1d_intra_eff = lambda_schedule(
                    epoch, cluster1d_lambda_intra,
                    cluster1d_warmup_epochs,
                    cluster1d_ramp_epochs)
                lam_c1d_inter_eff = lambda_schedule(
                    epoch, cluster1d_lambda_inter,
                    cluster1d_warmup_epochs,
                    cluster1d_ramp_epochs)
                # Backward-compat field for downstream readers that
                # still expect a single "lam_c1d_eff".
                lam_c1d_eff = max(lam_c1d_intra_eff, lam_c1d_inter_eff)
                if (lam_c1d_intra_eff > 0
                        or lam_c1d_inter_eff > 0):
                    rad_b_c1d = (radials_tensor[batch_idx]
                                  .to(device, non_blocking=True))
                    p_s = F.softmax(out["logits_s"] / temp_student, dim=-1)
                    l_c1d_intra, l_c1d_inter, c1d_mean_off, c1d_max_off = \
                        cluster1d_loss(
                            p_s, rad_b_c1d,
                            margin=cluster1d_margin,
                            min_cluster_mass=cluster1d_min_cluster_mass)
                    loss = (loss
                            + lam_c1d_intra_eff * l_c1d_intra
                            + lam_c1d_inter_eff * l_c1d_inter)
                    with torch.no_grad():
                        c1d_intra_sum += float(l_c1d_intra.item()) * bs
                        c1d_inter_sum += float(l_c1d_inter.item()) * bs
                        c1d_mean_off_sum += c1d_mean_off * bs
                        c1d_max_off_running = max(c1d_max_off_running, c1d_max_off)
                        c1d_n_batches += bs
            # Pair-supervised assignment loss (pre-train + fine-tune labels).
            # Each batch: sample up to `pair_per_batch` random pairs from
            # the labelled set, fetch the raw patterns, run the same
            # student aug pipeline + encoder + projector + prototypes, and
            # apply pair_assignment_loss on the resulting p ∈ Δ^K.
            if use_pair and lambda_pair > 0:
                n_take = min(int(pair_per_batch), n_pairs_total)
                if n_take > 0:
                    sel = torch.randperm(n_pairs_total,
                                            device=device)[:n_take]
                    a_idx = pair_idx_a_t[sel]
                    b_idx = pair_idx_b_t[sel]
                    y_sel = pair_y_t[sel]
                    # Fetch raw patterns from the underlying dataset
                    # (pre-_IndexedDataset wrapping). This is CPU-side
                    # but cheap (`pair_per_batch` is small).
                    all_idx = torch.cat([a_idx, b_idx]).cpu().tolist()
                    pair_xs = []
                    for i in all_idx:
                        try:
                            pair_xs.append(_pair_raw_dataset[int(i)])
                        except Exception:
                            pair_xs.append(None)
                    pair_xs = [t for t in pair_xs if t is not None]
                    if len(pair_xs) == 2 * n_take:
                        pair_x = torch.stack(pair_xs).to(
                            device, non_blocking=True).float()
                        # Apply the same student aug pipeline so the
                        # forward is consistent with the rest of the
                        # batch's student view.
                        pair_x = student_aug(pair_x)
                        feat_p = model.student_projector(
                            model.student_encoder(pair_x))
                        logits_p = model.prototypes(feat_p)
                        p_all = F.softmax(
                            logits_p / temp_student, dim=-1)
                        p_a = p_all[:n_take]
                        p_b = p_all[n_take:]
                        l_pair, n_used_p = pair_assignment_loss(
                            p_a, p_b, y_sel,
                            entropy_reg=pair_entropy_reg)
                        loss = loss + float(lambda_pair) * l_pair
                        with torch.no_grad():
                            pair_loss_sum  += float(l_pair.item()) * bs
                            pair_n_used_sum += int(n_used_p) * bs
                            pair_n_batches += bs
            # Prototype-prototype repulsion (anti-partial-collapse).
            if proto_repel_lambda > 0:
                proto_w = model.prototypes.prototypes
                l_repel, mean_off, max_off = prototype_repel_loss(
                    proto_w, threshold=proto_repel_threshold)
                loss = loss + proto_repel_lambda * l_repel
                with torch.no_grad():
                    repel_loss_sum += float(l_repel.item()) * bs
                    repel_mean_off_sum += mean_off * bs
                    repel_max_off_running = max(repel_max_off_running, max_off)
                    repel_n_batches += bs
            # Additive centroid loss (disabled if centroid_lambda == 0 or
            # the model was built without contrastive_centroids).
            if use_centroid:
                l_intra, l_inter = ContrastiveDINOModel.centroid_loss(
                    embed_s, pt, model.contrastive_centroids,
                    margin=centroid_margin)
                l_cent = l_intra + centroid_inter_weight * l_inter
                loss = loss + lam_cent_eff * l_cent
            else:
                l_intra = torch.zeros((), device=device)
                l_inter = torch.zeros((), device=device)
            # Spatial 4-neighbor Potts regularizer on the per-epoch tile.
            # Gated by 1D radial agreement (when supcon_radials is provided):
            #   pull = (1 - <p_i,p_j>) weighted by ReLU(c_ij - tau_pos)
            #   push =     <p_i,p_j>   weighted by ReLU(tau_neg - c_ij)
            # Pairs in the dead band [tau_neg, tau_pos] contribute nothing.
            # Without radials, falls back to unconditional Potts (legacy).
            if use_spatial and spx is not None:
                sp_feat = model.student_projector(model.student_encoder(spx))
                sp_logits = model.prototypes(sp_feat)
                sp_probs = F.softmax(sp_logits / temp_student, dim=-1)
                sp_2d = sp_probs.view(sps[0], sps[1], K)
                p_dot_r = (sp_2d[:, :-1] * sp_2d[:, 1:]).sum(-1)
                p_dot_d = (sp_2d[:-1, :] * sp_2d[1:, :]).sum(-1)
                if (spatial_gate_pull is not None
                        and spatial_gate_push is not None):
                    pull_r, pull_d = spatial_gate_pull
                    push_r, push_d = spatial_gate_push
                    eps_g = 1e-8
                    pull_loss = (
                        (pull_r * (1.0 - p_dot_r)).sum()
                            / pull_r.sum().clamp_min(eps_g)
                      + (pull_d * (1.0 - p_dot_d)).sum()
                            / pull_d.sum().clamp_min(eps_g)
                    ) * 0.5
                    push_loss = (
                        (push_r * p_dot_r).sum()
                            / push_r.sum().clamp_min(eps_g)
                      + (push_d * p_dot_d).sum()
                            / push_d.sum().clamp_min(eps_g)
                    ) * 0.5
                    spatial_loss = pull_loss + push_loss
                    pull_val = float(pull_loss.detach().item())
                    push_val = float(push_loss.detach().item())
                else:
                    spatial_loss = ((1.0 - p_dot_r).mean()
                                       + (1.0 - p_dot_d).mean()) * 0.5
                    pull_val = float(spatial_loss.detach().item())
                    push_val = 0.0
                loss = loss + lam_spatial * spatial_loss
                with torch.no_grad():
                    spatial_loss_sum += float(spatial_loss.detach().item()) * bs
                    spatial_pull_sum += pull_val * bs
                    spatial_push_sum += push_val * bs
                    spatial_n_batches += bs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model.update_teacher(momentum=mm)
            model.sync_teacher_bn_buffers()
            # DINO-style centering only — Sinkhorn target replaces this
            # mechanism with equipartition iterations on the raw logits.
            if getattr(model, "target_mode", "dino") == "dino":
                model.update_center(raw_t)

            with torch.no_grad():
                tot_loss += loss.item() * bs
                tot_dino += loss_dino.item() * bs
                tot_cont += loss_cont.item() * bs
                tot_cent_intra += float(l_intra.detach().item()) * bs
                tot_cent_inter += float(l_inter.detach().item()) * bs
                n_seen += bs
                sp_sum = pt.sum(dim=0).cpu() if sp_sum is None else sp_sum + pt.sum(dim=0).cpu()
                # Hard K_active: count how many prototypes win the
                # argmax for at least one pattern this epoch. Unlike
                # `sp_sum > 0` (which is almost always K), this can
                # actually drop below K when prototypes effectively
                # alias / die.
                hard_assigns = pt.argmax(dim=-1)                      # (B,)
                hb = torch.bincount(hard_assigns,
                                       minlength=K).cpu()
                hard_sum = hb if hard_sum is None else hard_sum + hb
                conf_sum += pt.max(dim=-1).values.mean().item() * bs
                em, ep10, ep90 = _teacher_softmax_entropy_stats(pt)
                ent_mean_sum += em; ent_p10_sum += ep10; ent_p90_sum += ep90
                # FCM partition coefficient + per-sample effective-K (Option C).
                # Both are sample-level; we accumulate sums-over-samples and
                # divide by n_seen (not n_batches) so the result is correct
                # even with variable batch sizes.
                with torch.no_grad():
                    pc_b = (pt ** 2).sum(dim=-1)                    # (B,)
                    ent_b = -(pt.clamp_min(1e-12).log() * pt).sum(dim=-1)  # (B,)
                pc_sum += pc_b.sum().item()
                keff_sample_sum += ent_b.exp().sum().item()
                nrm = embed_s.norm(dim=-1)
                norm_mean_sum += nrm.mean().item()
                norm_sq_sum += (nrm ** 2).mean().item()
                gate_frac_sum += frac_used
                n_batches += 1

        scheduler.step()

        pb = (sp_sum / n_seen).clamp(1e-12, 1.0)
        effK = math.exp(-(pb * pb.log()).sum().item())
        active_classes = int((sp_sum > 0).sum().item())
        hard_K_active = (int((hard_sum > 0).sum().item())
                          if hard_sum is not None else 0)
        avg_loss = tot_loss / max(n_seen, 1)
        avg_dino = tot_dino / max(n_seen, 1)
        avg_cont = tot_cont / max(n_seen, 1)
        avg_cent_intra = tot_cent_intra / max(n_seen, 1)
        avg_cent_inter = tot_cent_inter / max(n_seen, 1)
        avg_conf = conf_sum / max(n_seen, 1)
        # Fuzzy-validity epoch averages (Option C).
        pc = pc_sum / max(n_seen, 1)
        mpc = (K * pc - 1.0) / max(K - 1, 1)
        keff_sample = keff_sample_sum / max(n_seen, 1)
        ent_mean = ent_mean_sum / max(n_batches, 1)
        ent_p10  = ent_p10_sum  / max(n_batches, 1)
        ent_p90  = ent_p90_sum  / max(n_batches, 1)
        norm_mean = norm_mean_sum / max(n_batches, 1)
        norm_mean_sq = norm_sq_sum / max(n_batches, 1)
        norm_std = math.sqrt(max(norm_mean_sq - norm_mean ** 2, 0.0))
        gate_frac = gate_frac_sum / max(n_batches, 1)

        if supcon_n_batches > 0:
            avg_supcon = supcon_loss_sum / supcon_n_batches
            avg_supcon_pos = supcon_n_pos_sum / supcon_n_batches
            avg_supcon_neg = supcon_n_neg_sum / supcon_n_batches
            avg_supcon_abst = supcon_n_abst_sum / supcon_n_batches
        else:
            avg_supcon = avg_supcon_pos = avg_supcon_neg = avg_supcon_abst = 0.0
        if repel_n_batches > 0:
            avg_repel = repel_loss_sum / repel_n_batches
            avg_mean_off = repel_mean_off_sum / repel_n_batches
        else:
            avg_repel = avg_mean_off = 0.0
        if c1d_n_batches > 0:
            avg_c1d_intra = c1d_intra_sum / c1d_n_batches
            avg_c1d_inter = c1d_inter_sum / c1d_n_batches
            avg_c1d_mean_off = c1d_mean_off_sum / c1d_n_batches
        else:
            avg_c1d_intra = avg_c1d_inter = avg_c1d_mean_off = 0.0
        # lambda effective for cluster1d at this epoch.  We log the
        # MAX of intra/inter under the legacy column so downstream
        # readers get a sensible scalar; if you need per-term values
        # they're in `_train_kwargs.json`.
        if use_cluster1d:
            _lam_intra_eff = lambda_schedule(
                epoch, cluster1d_lambda_intra,
                cluster1d_warmup_epochs, cluster1d_ramp_epochs)
            _lam_inter_eff = lambda_schedule(
                epoch, cluster1d_lambda_inter,
                cluster1d_warmup_epochs, cluster1d_ramp_epochs)
            lam_c1d_for_log = max(_lam_intra_eff, _lam_inter_eff)
        else:
            _lam_intra_eff = _lam_inter_eff = 0.0
            lam_c1d_for_log = 0.0
        # Spatial-loss per-epoch averages (zero when not used).
        if spatial_n_batches > 0:
            avg_spatial = spatial_loss_sum / spatial_n_batches
            avg_spatial_pull = spatial_pull_sum / spatial_n_batches
            avg_spatial_push = spatial_push_sum / spatial_n_batches
        else:
            avg_spatial = avg_spatial_pull = avg_spatial_push = 0.0
        lam_spatial_eff = float(lam_spatial) if use_spatial else 0.0
        if pair_n_batches > 0:
            avg_pair_loss = pair_loss_sum / pair_n_batches
            avg_pair_n_used = pair_n_used_sum / pair_n_batches
        else:
            avg_pair_loss = 0.0
            avg_pair_n_used = 0
        lam_pair_for_log = float(lambda_pair) if use_pair else 0.0
        wr.writerow([
            epoch + 1,
            f"{avg_loss:.4f}", f"{avg_dino:.4f}", f"{avg_cont:.4f}",
            f"{avg_cent_intra:.4f}", f"{avg_cent_inter:.4f}",
            f"{lam_eff:.4f}", f"{lam_cent_eff:.4f}",
            f"{effK:.2f}", active_classes, hard_K_active,
            f"{pc:.4f}", f"{mpc:.4f}", f"{keff_sample:.2f}",
            f"{ent_mean:.4f}", f"{ent_p10:.4f}", f"{ent_p90:.4f}",
            f"{norm_mean:.4f}", f"{norm_std:.4f}",
            f"{gate_frac:.4f}", f"{avg_conf:.4f}",
            f"{avg_supcon:.4f}", f"{avg_supcon_pos:.2f}",
            f"{avg_supcon_neg:.2f}", f"{avg_supcon_abst:.2f}",
            f"{avg_repel:.4f}", f"{avg_mean_off:.4f}",
            f"{repel_max_off_running:.4f}",
            f"{avg_c1d_intra:.4f}", f"{avg_c1d_inter:.4f}",
            f"{avg_c1d_mean_off:.4f}", f"{c1d_max_off_running:.4f}",
            f"{lam_c1d_for_log:.4f}",
            f"{_lam_intra_eff:.4f}", f"{_lam_inter_eff:.4f}",
            f"{avg_spatial:.4f}",
            f"{avg_spatial_pull:.4f}", f"{avg_spatial_push:.4f}",
            f"{spatial_frac_pull:.3f}", f"{spatial_frac_push:.3f}",
            f"{lam_spatial_eff:.4f}",
            spatial_tile_y0, spatial_tile_x0,
            f"{avg_pair_loss:.4f}", f"{avg_pair_n_used:.1f}",
            f"{lam_pair_for_log:.4f}",
        ])
        cf.flush()

        # Auto-weight detection: at the configured epoch, decide whether to
        # enable the conf_weight loss based on the observed avg_conf. The
        # decision flips model.conf_weight_gamma in place (read fresh on each
        # forward pass), so it takes effect from the next batch onward.
        if (conf_weight_auto is not None
                and (epoch + 1) == int(conf_weight_auto.get("detect_at_epoch", 5))
                and float(model.conf_weight_gamma) == 0.0):
            threshold    = float(conf_weight_auto.get("threshold", 0.85))
            gamma_target = float(conf_weight_auto.get("gamma_target", 1.0))
            enable = bool(avg_conf > threshold)
            new_gamma = gamma_target if enable else 0.0
            model.conf_weight_gamma = new_gamma
            decision = {
                "detect_at_epoch": int(conf_weight_auto.get("detect_at_epoch", 5)),
                "threshold": threshold,
                "gamma_target": gamma_target,
                "avg_conf_observed": float(avg_conf),
                "decision": "enable_weight" if enable else "skip_weight",
                "gamma_set": new_gamma,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(os.path.join(outdir, "auto_weight_decision.json"), "w") as _fh:
                json.dump(decision, _fh, indent=2)
            print(f"\n[AUTO-WEIGHT] ep{decision['detect_at_epoch']}: "
                  f"avg_conf={avg_conf:.4f}, threshold={threshold:.2f} -> "
                  f"{'ENABLE' if enable else 'SKIP'} weight "
                  f"(gamma={new_gamma:.2f})", flush=True)

        if _HAS_TQDM and hasattr(eiter, "set_postfix"):
            cent_bit = f" Ci={avg_cent_intra:.3f}" if use_centroid else ""
            eiter.set_postfix_str(
                f"L={avg_loss:.3f} Ld={avg_dino:.3f} Lc={avg_cont:.4f}{cent_bit} "
                f"lam={lam_eff:.3f} effK={effK:.2f} hardK={hard_K_active} "
                f"pc={pc:.2f} kS={keff_sample:.1f} |z|={norm_mean:.3f}"
            )

        # Checkpointing (latest / periodic / best-by-loss).
        train_config_snapshot = {
            "vmax": getattr(dataset, "vmax", None),
            "resize": getattr(dataset, "resize", 192),
            "batch_size": batch_size,
            "seed": seed,
            "contrastive_lambda": contrastive_lambda,
            "contrastive_warmup_epochs": contrastive_warmup_epochs,
            "contrastive_ramp_epochs": contrastive_ramp_epochs,
            "contrastive_entropy_gate": contrastive_entropy_gate,
            "centroid_lambda": centroid_lambda,
            "centroid_margin": centroid_margin,
            "centroid_inter_weight": centroid_inter_weight,
            "use_centroid_loss": bool(use_centroid),
            "theta_shift_range": (student_roll.shift_range
                                   if student_roll is not None else 0),
        }

        def mkc():
            return {
                "epoch": epoch + 1,
                "architecture": getattr(model, "architecture", "resnet"),
                "n_layers": model.n_layers,
                "num_prototypes": K,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "teacher_temp": tt,
                "center": model.center.detach().cpu(),
                "train_config": dict(train_config_snapshot),
                "metrics": {
                    "loss": avg_loss,
                    "loss_dino": avg_dino,
                    "loss_contrastive": avg_cont,
                    "lambda_eff": lam_eff,
                    "eff_k": effK,
                    "active_classes": active_classes,
                    "pc": pc,
                    "mpc": mpc,
                    "keff_sample": keff_sample,
                    "teacher_ent_mean": ent_mean,
                    "teacher_ent_p10": ent_p10,
                    "teacher_ent_p90": ent_p90,
                    "student_embed_norm_mean": norm_mean,
                    "student_embed_norm_std": norm_std,
                    "gate_frac_used": gate_frac,
                    "avg_conf": avg_conf,
                },
            }

        torch.save(mkc(), os.path.join(outdir, "latest.pth"))
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            torch.save(mkc(), os.path.join(outdir, f"ckpt_ep{epoch+1}.pth"))
        # best by loss (pragmatic default when no scorecard callback is
        # wired up). Downstream eval will rank configs itself.
        if epoch == 0 or avg_loss < getattr(train_contrastive, "_best_loss", float("inf")):
            train_contrastive._best_loss = avg_loss
            torch.save(mkc(), os.path.join(outdir, "best.pth"))

        # Early stop after the requested epoch's row is written. Used to
        # implement the probe-and-commit pipeline where the "probe" must
        # share the LR / temperature schedule of a full main-length run.
        if stop_at_epoch is not None and (epoch + 1) >= int(stop_at_epoch):
            print(f"[train] stop_at_epoch={stop_at_epoch} reached "
                  f"(stopped at ep{epoch+1})", flush=True)
            break

    cf.close()

    # Clear the function attribute so successive runs start fresh.
    if hasattr(train_contrastive, "_best_loss"):
        delattr(train_contrastive, "_best_loss")

    return model, {
        "train_time_s": time.perf_counter() - t_start,
        "train_time_per_epoch_s": (time.perf_counter() - t_start) / max(epochs, 1),
        "asymmetry_report": asymmetry_report,
    }


# =========================================================================
# 10. CHECKPOINT LOADING
# =========================================================================

def load_contrastive_checkpoint(path: str, device=None,
                                 num_prototypes: int | None = None, **kw):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    nl = ckpt.get("n_layers", 1)
    if num_prototypes is None:
        num_prototypes = ckpt.get("num_prototypes", 10)
    # Detect whether the checkpoint was trained with centroid loss by looking
    # for the contrastive_centroids parameter in the state dict. Also detect
    # whether it used circular-theta conv by looking for a ThetaCircularConv2d
    # signature (conv.weight inside a wrapper).
    has_centroid = any(k.startswith("contrastive_centroids") for k in sd.keys())
    uses_circular = any(".conv.weight" in k for k in sd.keys())
    kw.setdefault("use_centroid_loss", has_centroid)
    kw.setdefault("circular_theta", uses_circular)
    # conf_weight_gamma is a scalar training-time flag; only matters at train
    # time, so loading eval-only is unaffected regardless of value.
    kw.setdefault("conf_weight_gamma", 0.0)
    # Architecture detection. Older checkpoints hardcode
    # "architecture": "resnet" regardless of how they were actually trained,
    # so trust the state_dict keys over the header. A ViT checkpoint always
    # has "student_encoder.cls_token" and "student_encoder.patch_embed.proj.*".
    has_cls = "student_encoder.cls_token" in sd
    has_patch_embed = any(k.startswith("student_encoder.patch_embed.proj.")
                           for k in sd.keys())
    if has_cls and has_patch_embed:
        # Extract ViT hyperparameters from the state_dict shapes so we don't
        # have to trust the checkpoint header.
        embed_dim = int(sd["student_encoder.cls_token"].shape[-1])
        pe_w = sd["student_encoder.patch_embed.proj.weight"]
        patch_size = int(pe_w.shape[-1])
        depth = sum(1 for k in sd.keys()
                    if k.startswith("student_encoder.blocks.")
                    and k.endswith(".norm1.weight"))
        # Standard ViT-Tiny/Small/Base head-dim convention: head_dim = 64.
        # num_heads isn't recoverable from the state_dict (qkv.weight has
        # shape (3*embed_dim, embed_dim) for any head count), so fall back
        # to this convention. Override via kw if you trained with a
        # non-standard head-dim.
        num_heads = max(1, embed_dim // 64)
        # pos_embed shape is (1, num_patches + 1, embed_dim); use it to
        # recover img_size.
        num_patches = int(sd["student_encoder.pos_embed"].shape[1]) - 1
        grid = int(round(num_patches ** 0.5))
        img_size = grid * patch_size
        kw["architecture"] = "vit"
        kw["circular_theta"] = False
        kw.setdefault("vit_img_size", img_size)
        kw.setdefault("vit_patch_size", patch_size)
        kw.setdefault("vit_embed_dim", embed_dim)
        kw.setdefault("vit_depth", depth)
        kw.setdefault("vit_num_heads", num_heads)
        print(f"  [arch] detected ViT from state_dict: "
              f"img={img_size} patch={patch_size} dim={embed_dim} "
              f"depth={depth} heads={num_heads}", flush=True)
    model = ContrastiveDINOModel(n_layers=nl, num_prototypes=num_prototypes, **kw)
    model = model.to(device)
    r = model.load_state_dict(sd, strict=False)
    if r.missing_keys:
        print(f"  WARNING: Missing keys: {r.missing_keys}")
    if r.unexpected_keys:
        print(f"  WARNING: Unexpected keys: {r.unexpected_keys}")
    model.eval()
    et = ckpt.get("teacher_temp", 0.06)
    if torch.is_tensor(et):
        et = et.item()
    return model, et, ckpt.get("metrics", {}), ckpt.get("train_config", {})
