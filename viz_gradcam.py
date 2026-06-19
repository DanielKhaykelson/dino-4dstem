"""
viz_gradcam.py — GradCAM over the prototype head for the contrastive model.

For each prototype c we compute:

  * GradCAM in polar space: d(logit_c) / d(last-conv activation) over the
    backbone trunk, global-avg-pool over channels to get a (θ, r) attention
    map at the input resolution.
  * GradCAM in Cartesian space: invert the PolarTransform on the polar cam
    so the attention lives on the detector-pixel geometry the user actually
    sees.
  * Per-prototype mean diffraction pattern (weighted by teacher softmax
    confidence on that prototype).
  * 10 representative samples for the prototype, each with their own
    polar + Cartesian GradCAM overlay.

Outputs under `runs/<sample>/<config>/eval/gradcam/`:

  - fig_gradcam_summary.png        one row per prototype, class-average view
  - fig_gradcam_p{c}.png           for c in 0..K-1: class avg + 10 samples

Reuses `inference.npz` so it doesn't re-run the scan. GradCAM runs on the
STUDENT path (which is what receives gradient during training and therefore
reflects what the backbone learned to attend to). Teacher/student weights
are EMA-coupled so the difference is small.

Usage:
    python viz_gradcam.py --sample Na007b --config config_c_K6_asym
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

from data import SAMPLES, LoadPRZ
from dino_sr_contrastive_model import (
    load_contrastive_checkpoint,
    PolarTransform, PolarMaskLeft,
)


# =========================================================================
# 1. Eval preprocessing (matches infer_scan)
# =========================================================================

def build_polar_preproc(polar_size: int = 192, polar_mask_cols: int = 30,
                         center_crop_size: int = 140,
                         center_mask_radius: int = 0):
    """Eval/attribution polar pre-processor.

    Mirrors the training-time get_contrastive_transforms order:
       CenterCrop  →  Resize  →  CenterMask (if r > 0)  →  PolarTransform
                                                        →  PolarMaskLeft

    The CenterMask step was previously absent here, so models trained
    with center_mask_radius > 0 saw an OOD input at inference (the
    central beam intact) and produced an artificial CAM donut at
    the centre.  Pass center_mask_radius equal to whatever the training
    used for a faithful eval.
    """
    ops = [
        T.CenterCrop(center_crop_size),
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR,
                  antialias=True),
    ]
    if int(center_mask_radius) > 0:
        try:
            from dino_sr_fixed import CenterMask
            ops.append(CenterMask(radius=int(center_mask_radius)))
        except Exception:
            pass
    ops += [
        PolarTransform(output_size=polar_size),
        PolarMaskLeft(k_cols=polar_mask_cols),
    ]
    return T.Compose(ops)


def build_cart_preproc(polar_size: int = 192, center_crop_size: int = 140,
                         center_mask_radius: int = 0):
    """Eval cart pre-processor used for the displayed underlay.  Mirrors
    the training pipeline up to (but not including) the polar transform,
    so the visual matches what the model actually saw."""
    ops = [
        T.CenterCrop(center_crop_size),
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR,
                  antialias=True),
    ]
    if int(center_mask_radius) > 0:
        try:
            from dino_sr_fixed import CenterMask
            ops.append(CenterMask(radius=int(center_mask_radius)))
        except Exception:
            pass
    return T.Compose(ops)


# =========================================================================
# 2. GradCAM hook
# =========================================================================

class GradCAM:
    """Vanilla GradCAM on a chosen conv module (the last module of the
    PlainSequentialEncoder trunk). Hooks the forward output for activations
    and the backward grad_output for gradients, then takes the per-channel
    global-average-pooled gradient as the channel weight.
    """

    def __init__(self, model, target_module: nn.Module):
        self.model = model
        self.target = target_module
        self.acts = None
        self.grads = None
        self._fh = target_module.register_forward_hook(self._fwd)
        self._bh = target_module.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, o):
        self.acts = o

    def _bwd(self, _m, _gi, go):
        self.grads = go[0]

    def close(self):
        self._fh.remove()
        self._bh.remove()

    def __call__(self, x_polar: torch.Tensor, target_class: int) -> torch.Tensor:
        """x_polar: (1, 1, H, W). Returns (H, W) cam in [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        # Student path — it's what backprop has been shaping.
        feat = self.model.student_encoder(x_polar)
        proj = self.model.student_projector(feat)
        logits = self.model.prototypes(proj)
        score = logits[0, target_class]
        score.backward()
        A = self.acts.detach()
        G = self.grads.detach()
        alpha = G.mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)
        cam = (alpha * A).sum(dim=1)                    # (1, Hf, Wf)
        cam = F.relu(cam)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1).clamp_min(1e-12)
        cam = cam / cam_max
        cam = F.interpolate(cam.unsqueeze(1),
                             size=x_polar.shape[-2:],
                             mode="bilinear", align_corners=False).squeeze()
        return cam  # (H, W)


def integrated_gradients(model, x_input: torch.Tensor, target_class: int,
                          n_steps: int = 50,
                          baseline: torch.Tensor | None = None) -> torch.Tensor:
    """Integrated Gradients attribution at the INPUT pixel level.

    IG_c(x) = (x - x') * mean_{alpha=0..1} d F_c(x' + alpha*(x - x')) / dx

    Returns a (H, W) heat map in [0, 1] obtained by taking |IG| summed over
    the channel dimension and normalizing.

    Baseline defaults to zero, which is appropriate here because the inputs
    have already been beam-masked (the low-r region is zero), so the zero
    baseline is "no signal anywhere" rather than "a pattern minus its beam."

    Student path is used, matching the GradCAM target.
    """
    if baseline is None:
        baseline = torch.zeros_like(x_input)
    # Riemann sum: alphas at the midpoints of n_steps equal intervals.
    alphas = (torch.arange(n_steps, device=x_input.device, dtype=x_input.dtype)
              + 0.5) / n_steps                                     # (n,)
    grad_sum = torch.zeros_like(x_input)
    for a in alphas:
        x_interp = (baseline + a * (x_input - baseline)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        feat = model.student_encoder(x_interp)
        proj = model.student_projector(feat)
        logits = model.prototypes(proj)
        score = logits[0, target_class]
        g = torch.autograd.grad(score, x_interp, retain_graph=False)[0]
        grad_sum = grad_sum + g
    avg_grad = grad_sum / n_steps
    ig = (x_input - baseline) * avg_grad                           # (1, C, H, W)
    # Channel-sum of |IG|, per-image normalize to [0, 1].
    heat = ig.abs().sum(dim=1).squeeze()                            # (H, W)
    heat = heat / heat.max().clamp_min(1e-12)
    return heat


# =========================================================================
# 3. Polar -> Cartesian inverse warp (for display)
# =========================================================================

def polar_cam_to_cartesian(cam_polar: torch.Tensor,
                             output_size: int = 192) -> torch.Tensor:
    """Invert the PolarTransform: given a cam in polar coords (rows=theta,
    cols=r, both in [0, 1] normalized space), sample it at every Cartesian
    pixel in a `output_size x output_size` image.

    The forward PolarTransform used by training maps Cartesian (x, y) in
    [-1, 1] to polar row=theta, col=r in [0, 1] via:
        (x, y) = (r * cos(theta), r * sin(theta))
    so the inverse is (theta, r) = (atan2(y, x), sqrt(x^2 + y^2)).

    Pixels with r > 1 (outside the inscribed circle the forward transform
    reached) get zeroed.
    """
    device = cam_polar.device
    dtype = cam_polar.dtype
    H = W = output_size
    y = torch.linspace(-1.0, 1.0, H, dtype=dtype, device=device)
    x = torch.linspace(-1.0, 1.0, W, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(xx ** 2 + yy ** 2)
    theta = torch.atan2(yy, xx)                        # [-pi, pi]
    theta_norm = (theta + math.pi) / (2 * math.pi)     # [0, 1]
    # grid_sample convention: grid[..., 0] is the input's x-axis (cols=r),
    # grid[..., 1] is the input's y-axis (rows=theta). Both in [-1, 1] when
    # align_corners=True.
    r_grid = r * 2.0 - 1.0
    r_grid = r_grid.clamp(-1.0, 1.0)
    theta_grid = theta_norm * 2.0 - 1.0
    grid = torch.stack([r_grid, theta_grid], dim=-1).unsqueeze(0)
    cam4 = cam_polar.unsqueeze(0).unsqueeze(0)
    cart = F.grid_sample(cam4, grid, mode="bilinear",
                          padding_mode="zeros", align_corners=True).squeeze()
    mask = (r <= 1.0).to(dtype=dtype)
    return cart * mask


# =========================================================================
# 4. Display helpers
# =========================================================================

def _beam_mask(H: int, W: int, radius: int):
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    return ((yy - cy) ** 2 + (xx - cx) ** 2) > radius ** 2


def _clip_log1p(arr: np.ndarray, mask: np.ndarray | None = None,
                pct_lo: float = 2.0, pct_hi: float = 99.5):
    ref = arr[mask] if mask is not None else arr.ravel()
    if ref.size == 0:
        return arr
    lo = np.percentile(ref, pct_lo)
    hi = np.percentile(ref, pct_hi)
    clipped = np.clip(arr, lo, hi)
    if mask is not None:
        clipped = clipped * mask
    return np.log1p(clipped - lo)


def _overlay(ax, base: np.ndarray, cam: np.ndarray, title: "str | None",
              alpha: float = 0.45, cmap_base: str = "inferno",
              cmap_cam: str = "jet", aspect: str = "auto"):
    ax.imshow(base, cmap=cmap_base, aspect=aspect)
    ax.imshow(cam, cmap=cmap_cam, alpha=alpha, aspect=aspect)
    if title:
        ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


# =========================================================================
# 5. Driver
# =========================================================================

def _pick_top_samples(soft_probs: np.ndarray, assigns: np.ndarray,
                       class_id: int, n: int = 10) -> np.ndarray:
    mask = (assigns == class_id)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.array([], dtype=int)
    scores = soft_probs[idx, class_id]
    return idx[np.argsort(-scores)[:n]]


def run(sample: str, config: str, n_samples_per_proto: int = 10,
         beam_mask_radius: int = 40, device=None,
         ckpt_path: str | None = None,
         run_dir: "str | None" = None):
    """If `run_dir` is given, use it directly; otherwise fall back to the
    legacy convention `runs/<sample>/<config>/`.
    """
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    if run_dir is None:
        run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "runs", sample, config)

    # Read TRAINING-time mask values from run_summary.json so the displayed
    # beam mask reflects what the model actually saw, not a hardcoded 40 px.
    rs_path = os.path.join(run_dir, "run_summary.json")
    train_mask_r = 15
    train_polar_mask_cols = 45
    train_polar_size = 192
    train_center_crop = 140
    if os.path.exists(rs_path):
        import json as _json
        with open(rs_path) as _rsf:
            _rs = _json.load(_rsf)
        _rcfg = _rs.get("cfg", {})
        train_mask_r = int(_rcfg.get("center_mask_radius", 15))
        train_polar_mask_cols = int(_rcfg.get("polar_mask_cols", 45))
        train_polar_size = int(_rcfg.get("polar_size", 192))
        train_center_crop = int(_rcfg.get("center_crop_size", 140))
    # Convert to RAW (256-frame) pixels for display.
    SCALE_192_TO_RAW = train_center_crop / train_polar_size
    eff_polar_r_raw = (train_polar_mask_cols / train_polar_size) * (
        train_polar_size / 2.0) * SCALE_192_TO_RAW
    eff_cart_r_raw = train_mask_r * SCALE_192_TO_RAW
    actual_beam_mask_radius = max(eff_cart_r_raw, eff_polar_r_raw)
    # Override the function-arg value with the auto-detected one when run_summary exists.
    beam_mask_radius = max(int(round(actual_beam_mask_radius)), 1)
    print(f"[gradcam] mask: cart={eff_cart_r_raw:.1f}px  "
          f"polar_eff={eff_polar_r_raw:.1f}px  "
          f"display={beam_mask_radius}px (raw)", flush=True)

    data = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft_probs = data["soft_probs"]
    assigns = data["assigns"]
    embeds = data["embeds"]              # student L2-normed embeddings
    K = soft_probs.shape[1]

    # Load model (student weights will serve the gradcam forward path).
    # For transfer-inference configs, the checkpoint lives at the *source*
    # training dir, not in this config's folder.
    if ckpt_path is None:
        ckpt_path = os.path.join(run_dir, "best.pth")
    model, eval_temp, _, _ = load_contrastive_checkpoint(ckpt_path, device=device)
    # Unfreeze the encoder so gradients flow (training loop froze teacher).
    for p in model.student_encoder.parameters():
        p.requires_grad_(True)
    for p in model.student_projector.parameters():
        p.requires_grad_(True)
    for p in model.prototypes.parameters():
        p.requires_grad_(True)
    model.eval()  # BN stays in eval (uses running stats) — deterministic.

    # Target module for GradCAM: last module in the trunk (layer1 for L1).
    # PlainSequentialEncoder IS an nn.Sequential whose last child is the
    # final residual layer block. Hook that.
    last_mod = list(model.student_encoder.children())[-1]
    cam_tool = GradCAM(model, last_mod)

    # Preprocessing -- use training-time mask values, not defaults.
    polar_pre = build_polar_preproc(polar_size=train_polar_size,
                                      polar_mask_cols=train_polar_mask_cols,
                                      center_crop_size=train_center_crop)
    cart_pre = build_cart_preproc(polar_size=train_polar_size,
                                    center_crop_size=train_center_crop)

    # Output dir.
    out_dir = os.path.join(run_dir, "eval", "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    # Pre-compute class-average raw Cartesian patterns (confidence-weighted).
    avg_polar_by_c = {}
    avg_cart_by_c = {}
    cam_polar_avg_by_c = {}
    cam_cart_avg_by_c = {}
    ig_polar_avg_by_c = {}
    ig_cart_avg_by_c = {}

    for c in range(K):
        top = _pick_top_samples(soft_probs, assigns, c,
                                 n=max(200, n_samples_per_proto))
        if top.size == 0:
            continue
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0).astype(np.float32)
        weights = soft_probs[top, c].astype(np.float32)
        wavg = (patterns * weights[:, None, None]).sum(0) / (weights.sum() + 1e-12)
        # Feed the average through the polar + Cartesian preprocessing so
        # it lives in the same coord frames as the model's inputs.
        wavg_t = torch.from_numpy(wavg).unsqueeze(0).to(device)  # (1, H0, W0)
        # LoadPRZ.__getitem__ does resize + rescale_like_vmax; mimic here so
        # the class-average pattern enters the pipeline at vmax-normalized
        # [0, 1] dynamic range like training samples did.
        vmax = cfg["vmax"]
        wavg_norm = wavg.astype(np.float32) / max(float(vmax), 1e-6)
        wavg_norm = np.clip(wavg_norm, 0.0, 1.0)
        x_full = torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0)  # (1, 1, H0, W0)
        x_full = F.interpolate(x_full, size=(192, 192), mode="bilinear",
                                align_corners=False).to(device)
        # Cartesian (post-crop + resize): for display only.
        x_cart = cart_pre(x_full)              # (1, 1, 192, 192)
        x_polar = polar_pre(x_full)            # (1, 1, 192, 192)
        # GradCAM + Integrated Gradients on the class average.
        with torch.enable_grad():
            x_p = x_polar.detach().requires_grad_(True)
            cam_p = cam_tool(x_p, target_class=c)       # (192, 192)
            ig_p = integrated_gradients(model, x_polar.detach(),
                                         target_class=c, n_steps=50)
        cam_c = polar_cam_to_cartesian(cam_p).detach().cpu().numpy()
        ig_c = polar_cam_to_cartesian(ig_p).detach().cpu().numpy()
        avg_polar_by_c[c] = x_polar[0, 0].detach().cpu().numpy()
        avg_cart_by_c[c] = x_cart[0, 0].detach().cpu().numpy()
        cam_polar_avg_by_c[c] = cam_p.detach().cpu().numpy()
        cam_cart_avg_by_c[c] = cam_c
        ig_polar_avg_by_c[c] = ig_p.detach().cpu().numpy()
        ig_cart_avg_by_c[c] = ig_c

    # ================================================================
    # Summary figures (paper-quality): class-avg attribution for all prototypes
    #   Two parallel figures, each Cartesian only, 3 cols (raw, +overlay, only):
    #     fig_gradcam_summary.png  - GradCAM
    #     fig_ig_summary.png       - Integrated Gradients
    # ================================================================
    counts = np.bincount(assigns, minlength=K)
    order = np.argsort(-counts)
    PANEL_S = 3.0
    H_cart = avg_cart_by_c[order[0]].shape[-1]
    bm_cart = _beam_mask(H_cart, H_cart, radius=int(beam_mask_radius * H_cart / 512))

    def _summary_figure(by_c: dict, title: str, fname: str, cmap_cam: str = "turbo"):
        fig, axes = plt.subplots(K, 3, figsize=(3 * PANEL_S, K * PANEL_S),
                                  squeeze=False)
        col_titles_summary = ["pattern (Cartesian)", "+ overlay", "attribution only"]
        for j, t in enumerate(col_titles_summary):
            axes[0, j].set_title(t, fontsize=12, pad=8)
        for row_idx, c in enumerate(order):
            if c not in avg_polar_by_c:
                for j in range(3):
                    axes[row_idx, j].set_axis_off()
                continue
            cart = avg_cart_by_c[c]
            attr = by_c[c]
            disp_cart = _clip_log1p(cart, mask=bm_cart)
            axes[row_idx, 0].imshow(disp_cart, cmap="inferno", aspect="equal")
            _overlay(axes[row_idx, 1], disp_cart, attr,
                      title=None, alpha=0.55, cmap_cam=cmap_cam, aspect="equal")
            axes[row_idx, 2].imshow(attr, cmap=cmap_cam, aspect="equal")
            axes[row_idx, 0].set_ylabel(f"p{c}\n(N={int(counts[c])})",
                                         fontsize=11, rotation=0,
                                         labelpad=42, ha="right", va="center")
            for j in range(3):
                axes[row_idx, j].set_xticks([]); axes[row_idx, j].set_yticks([])
        fig.suptitle(title, fontsize=13, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        fig.savefig(os.path.join(out_dir, fname),
                     dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    _summary_figure(cam_cart_avg_by_c,
                     "GradCAM on class-average diffraction patterns "
                     "(per prototype, Cartesian)",
                     "fig_gradcam_summary.png", cmap_cam="turbo")
    _summary_figure(ig_cart_avg_by_c,
                     "Integrated Gradients on class-average diffraction patterns "
                     "(per prototype, Cartesian)",
                     "fig_ig_summary.png", cmap_cam="magma")

    # ================================================================
    # Per-prototype figure (paper-quality)
    #   - Class average (top row) + 5 representative samples
    #   - 3 columns: raw Cartesian, GradCAM-on-Cartesian overlay, GradCAM heatmap
    #   - Higher DPI, bigger panels, no polar-or-IG clutter
    # ================================================================
    n_per_proto_paper = min(int(n_samples_per_proto), 5)
    PANEL = 3.4  # inches per panel
    for c in order:
        if c not in avg_polar_by_c:
            continue
        top = _pick_top_samples(soft_probs, assigns, c, n=n_per_proto_paper)
        n_rows = 1 + len(top)

        # Pre-compute attribution maps for all rows so we can build both the
        # GradCAM and IG figures in one pass (avoids re-doing forward/back).
        sample_attr = []                 # list of (cart_disp, cam_c, ig_c, idx, p)
        for idx in top:
            tns = dataset[int(idx)]
            x_full = tns.to(device).float().unsqueeze(0)
            x_cart = cart_pre(x_full)
            x_polar = polar_pre(x_full)
            with torch.enable_grad():
                x_p = x_polar.detach().requires_grad_(True)
                cam_p_sample = cam_tool(x_p, target_class=c).detach().cpu().numpy()
                ig_p_sample = integrated_gradients(
                    model, x_polar.detach(), target_class=c, n_steps=50
                ).detach().cpu().numpy()
            cam_c_sample = polar_cam_to_cartesian(
                torch.from_numpy(cam_p_sample).to(device)).cpu().numpy()
            ig_c_sample = polar_cam_to_cartesian(
                torch.from_numpy(ig_p_sample).to(device)).cpu().numpy()
            cart_img = x_cart[0, 0].detach().cpu().numpy()
            sample_attr.append((
                _clip_log1p(cart_img, mask=bm_cart),
                cam_c_sample, ig_c_sample,
                int(idx), float(soft_probs[idx, c]),
            ))

        cart = avg_cart_by_c[c]
        disp_cart = _clip_log1p(cart, mask=bm_cart)
        cam_c_avg = cam_cart_avg_by_c[c]
        ig_c_avg = ig_cart_avg_by_c[c]

        def _proto_figure(attr_avg, attr_samples_idx: int, title: str,
                          fname: str, cmap_cam: str):
            """attr_samples_idx: index into sample_attr tuple of the per-sample
            map (1=cam_c_sample, 2=ig_c_sample)."""
            fig, axes = plt.subplots(n_rows, 3,
                                      figsize=(3 * PANEL, n_rows * PANEL),
                                      squeeze=False)
            col_titles_paper = ["pattern (Cartesian)", "+ overlay",
                                 "attribution only"]
            for j, t in enumerate(col_titles_paper):
                axes[0, j].set_title(t, fontsize=12, pad=8)

            # Row 0: class average.
            axes[0, 0].imshow(disp_cart, cmap="inferno", aspect="equal")
            _overlay(axes[0, 1], disp_cart, attr_avg,
                      title=None, alpha=0.55, cmap_cam=cmap_cam, aspect="equal")
            axes[0, 2].imshow(attr_avg, cmap=cmap_cam, aspect="equal")
            axes[0, 0].set_ylabel(f"class avg\n(N={int(counts[c])})",
                                   fontsize=11, rotation=0, labelpad=44,
                                   ha="right", va="center")
            for j in range(3):
                axes[0, j].set_xticks([]); axes[0, j].set_yticks([])

            for r_off, sa in enumerate(sample_attr):
                cart_disp_s = sa[0]
                attr_s = sa[attr_samples_idx]
                idx = sa[3]; pval = sa[4]
                r = 1 + r_off
                axes[r, 0].imshow(cart_disp_s, cmap="inferno", aspect="equal")
                _overlay(axes[r, 1], cart_disp_s, attr_s,
                          title=None, alpha=0.55, cmap_cam=cmap_cam,
                          aspect="equal")
                axes[r, 2].imshow(attr_s, cmap=cmap_cam, aspect="equal")
                axes[r, 0].set_ylabel(
                    f"sample i={idx}\np(c)={pval:.2f}",
                    fontsize=10, rotation=0, labelpad=44,
                    ha="right", va="center")
                for j in range(3):
                    axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            fig.suptitle(title, fontsize=13, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.985])
            fig.savefig(os.path.join(out_dir, fname),
                         dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        _proto_figure(
            cam_c_avg, attr_samples_idx=1,
            title=(f"Prototype p{c} GradCAM  —  class average + "
                   f"{len(top)} top-confidence samples "
                   f"(N_class={int(counts[c])})"),
            fname=f"fig_gradcam_p{c}.png", cmap_cam="turbo")
        _proto_figure(
            ig_c_avg, attr_samples_idx=2,
            title=(f"Prototype p{c} Integrated Gradients  —  class average + "
                   f"{len(top)} top-confidence samples "
                   f"(N_class={int(counts[c])})"),
            fname=f"fig_ig_p{c}.png", cmap_cam="magma")
        print(f"[attr] wrote p{c} GradCAM + IG ({len(top)} samples, paper layout)",
              flush=True)

    cam_tool.close()
    print(f"[gradcam] done. See {out_dir}/")

    # ================================================================
    # Class-quality figures: 5 closest + 5 farthest from class centroid
    # in embedding cosine space, alongside the class average. Sanity-checks
    # whether each cluster is tight (farthest still look like the centroid)
    # or heterogeneous (farthest look very different).
    # ================================================================
    quality_dir = os.path.join(run_dir, "eval", "class_quality")
    os.makedirs(quality_dir, exist_ok=True)

    def _proc(idx):
        tns = dataset[int(idx)]
        x_full = tns.to(device).float().unsqueeze(0)
        x_cart = cart_pre(x_full)
        cart_img = x_cart[0, 0].detach().cpu().numpy()
        return _clip_log1p(cart_img, mask=bm_cart)

    n_show_q = 5
    PANEL_Q = 3.2
    for c in order:
        if c not in avg_polar_by_c:
            continue
        mask = (assigns == c)
        if int(mask.sum()) < n_show_q + 1:
            continue
        idx_in_class = np.where(mask)[0]
        embeds_c = embeds[mask]
        centroid = embeds_c.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        cos_to_centroid = embeds_c @ centroid              # cosine similarity
        order_close_to_far = np.argsort(-cos_to_centroid)  # high cos first
        closest_local = order_close_to_far[:n_show_q]
        farthest_local = order_close_to_far[-n_show_q:][::-1]   # most-far first
        closest_global = idx_in_class[closest_local]
        farthest_global = idx_in_class[farthest_local]
        closest_cos = cos_to_centroid[closest_local]
        farthest_cos = cos_to_centroid[farthest_local]

        # within-class cosine distribution stats
        cos_p10 = float(np.percentile(cos_to_centroid, 10))
        cos_p50 = float(np.percentile(cos_to_centroid, 50))
        cos_p90 = float(np.percentile(cos_to_centroid, 90))

        n_rows = 1 + n_show_q
        fig, axes = plt.subplots(n_rows, 2,
                                  figsize=(2 * PANEL_Q, n_rows * PANEL_Q),
                                  squeeze=False)
        axes[0, 0].set_title("closest to centroid", fontsize=12, pad=8)
        axes[0, 1].set_title("farthest from centroid", fontsize=12, pad=8)

        # Row 0: class average shown in both columns for reference.
        cart_avg = avg_cart_by_c[c]
        disp_avg = _clip_log1p(cart_avg, mask=bm_cart)
        for j in range(2):
            axes[0, j].imshow(disp_avg, cmap="inferno", aspect="equal")
            axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[0, 0].set_ylabel(
            f"class avg\n(N={int(mask.sum())})\n"
            f"cos p10/p50/p90:\n{cos_p10:.3f}/{cos_p50:.3f}/{cos_p90:.3f}",
            fontsize=9, rotation=0, labelpad=58, ha="right", va="center")

        for r in range(n_show_q):
            disp_c = _proc(closest_global[r])
            disp_f = _proc(farthest_global[r])
            axes[r + 1, 0].imshow(disp_c, cmap="inferno", aspect="equal")
            axes[r + 1, 1].imshow(disp_f, cmap="inferno", aspect="equal")
            for j in range(2):
                axes[r + 1, j].set_xticks([])
                axes[r + 1, j].set_yticks([])
            axes[r + 1, 0].set_ylabel(
                f"#{r+1} cos={closest_cos[r]:.3f}\n"
                f"i={int(closest_global[r])}  p(c)={soft_probs[closest_global[r], c]:.2f}",
                fontsize=9, rotation=0, labelpad=58, ha="right", va="center")
            axes[r + 1, 1].set_ylabel(
                f"#{r+1} cos={farthest_cos[r]:.3f}\n"
                f"i={int(farthest_global[r])}  p(c)={soft_probs[farthest_global[r], c]:.2f}",
                fontsize=9, rotation=0, labelpad=58, ha="right", va="center")
        fig.suptitle(
            f"Class-quality: prototype p{c}  —  centroid + "
            f"{n_show_q} closest / {n_show_q} farthest in embedding cosine space",
            fontsize=12, y=0.998)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(os.path.join(quality_dir, f"fig_class_quality_p{c}.png"),
                     dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[class-quality] wrote p{c} (cos p10/p50/p90 = "
              f"{cos_p10:.3f}/{cos_p50:.3f}/{cos_p90:.3f})", flush=True)
    print(f"[class-quality] done. See {quality_dir}/")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="Na007b")
    ap.add_argument("--config", default="config_c_K6_asym")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--beam-mask-radius", type=int, default=40,
                    help="Beam-disk mask radius in RAW detector pixels "
                         "(scaled to display size internally).")
    args = ap.parse_args(argv)
    run(args.sample, args.config,
        n_samples_per_proto=args.n_samples,
        beam_mask_radius=args.beam_mask_radius)


if __name__ == "__main__":
    main()
