"""viz_paper_attribution.py -- paper-quality GradCAM + Integrated Gradients
figure showing what physics each prototype learns.

Layout (single PNG per run):
    K rows x 5 cols
        col 1: class-mean diffraction (Cartesian, log)
        col 2: GradCAM on class mean    (overlay)
        col 3: IG on class mean         (overlay)
        col 4: one representative sample (raw)
        col 5: GradCAM on that sample    (overlay)

Improvements over the existing viz_gradcam:
    - Cartesian only (Polar mostly unused on the paper page).
    - Gaussian-smoothed attribution maps (sigma ~ 2 px) so per-sample heat
      maps stop looking like static.
    - Global vmin/vmax across the column so a brighter prototype isn't
      visually overwhelming a quieter one.
    - 1 figure (not K+2 figures) -- easier to read on a single page.
    - White background, large fonts, big panels.

Outputs:
    <run_dir>/eval/paper_attribution/fig_paper_attribution.png
    <run_dir>/eval/paper_attribution/fig_paper_attribution_p{c}.png
        (per-prototype panel with class avg + 3 sample rows)

Usage:
    python viz_paper_attribution.py --run-dir runs/_winner_followup/Na007b_K6_50ep
"""
from __future__ import annotations
import argparse, math, os, sys, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

from data import SAMPLES, LoadPRZ
from dino_sr_contrastive_model import (
    load_contrastive_checkpoint, PolarTransform, PolarMaskLeft,
)
from viz_gradcam import (
    GradCAM, integrated_gradients, polar_cam_to_cartesian,
    build_polar_preproc, build_cart_preproc,
    resolve_prototype_ids, dense_target,
)


def _gaussian_blur(arr: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    if sigma <= 0:
        return arr
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr.astype(np.float32), sigma=sigma)
    except Exception:
        return arr


def _norm01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo + 1e-12:
        return a * 0.0
    return (a - lo) / (hi - lo)


def _log_disp(cart: np.ndarray, beam_mask: np.ndarray,
               pct_lo=2.0, pct_hi=99.5) -> np.ndarray:
    ref = cart[beam_mask]
    if ref.size == 0:
        return cart
    lo, hi = np.percentile(ref, pct_lo), np.percentile(ref, pct_hi)
    clipped = np.clip(cart, lo, hi) * beam_mask
    return np.log1p(clipped - lo)


def _beam_mask(H: int, W: int, radius: float):
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    return ((yy - cy) ** 2 + (xx - cx) ** 2) > radius ** 2


def _read_train_cfg(run_dir: str):
    rs_path = os.path.join(run_dir, "run_summary.json")
    out = dict(mask_r=15, polar_mask_cols=45, polar_size=192, center_crop=140)
    if os.path.exists(rs_path):
        with open(rs_path) as f:
            rs = json.load(f)
        rcfg = rs.get("cfg", {})
        out["mask_r"] = int(rcfg.get("center_mask_radius", 15))
        out["polar_mask_cols"] = int(rcfg.get("polar_mask_cols", 45))
        out["polar_size"] = int(rcfg.get("polar_size", 192))
        out["center_crop"] = int(rcfg.get("center_crop_size", 140))
    return out


def run(run_dir: str, sample: str, n_samples_per_proto: int = 3,
         attribution_sigma: float = 2.0, ig_steps: int = 50,
         device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]
    K = soft_probs.shape[1]

    tcfg = _read_train_cfg(run_dir)
    polar_pre = build_polar_preproc(polar_size=tcfg["polar_size"],
                                      polar_mask_cols=tcfg["polar_mask_cols"],
                                      center_crop_size=tcfg["center_crop"])
    cart_pre = build_cart_preproc(polar_size=tcfg["polar_size"],
                                    center_crop_size=tcfg["center_crop"])

    # effective beam-mask radius in display space
    SCALE = tcfg["center_crop"] / tcfg["polar_size"]
    eff_polar_r = (tcfg["polar_mask_cols"] / tcfg["polar_size"]) * (
                    tcfg["polar_size"] / 2.0) * SCALE
    eff_cart_r = tcfg["mask_r"] * SCALE
    bm_r = max(eff_polar_r, eff_cart_r)

    # model
    ckpt_path = os.path.join(run_dir, "best.pth")
    model, _, _, _ = load_contrastive_checkpoint(ckpt_path, device=device)
    for p in model.student_encoder.parameters():
        p.requires_grad_(True)
    for p in model.student_projector.parameters():
        p.requires_grad_(True)
    for p in model.prototypes.parameters():
        p.requires_grad_(True)
    model.eval()
    last_mod = list(model.student_encoder.children())[-1]
    cam_tool = GradCAM(model, last_mod)
    # Map dense class ids (0..K-1) -> original prototype indices so GradCAM/IG
    # attribute the *actual* trained prototype, not an inactive one.
    orig_ids = resolve_prototype_ids(run_dir, model, dataset, device,
                                     polar_pre=polar_pre)
    print(f"[paper-attr] dense->prototype mapping: {orig_ids}", flush=True)

    # confidence-weighted class means + their attribution
    avg_cart = {}
    cam_cart = {}
    ig_cart = {}
    counts = np.bincount(assigns, minlength=K)
    order = np.argsort(-counts)

    print(f"[paper-attr] computing class-average GradCAM/IG for K={K}", flush=True)
    for c in range(K):
        idx = np.where(assigns == c)[0]
        if idx.size == 0:
            continue
        scores = soft_probs[idx, c]
        top = idx[np.argsort(-scores)[:max(200, n_samples_per_proto)]]
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0).astype(np.float32)
        w = soft_probs[top, c].astype(np.float32)
        wavg = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
        wavg_norm = np.clip(wavg / max(float(cfg["vmax"]), 1e-6), 0.0, 1.0)
        x_full = (torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0)
                   .to(device).float())
        x_full = F.interpolate(x_full, size=(192, 192), mode="bilinear",
                                align_corners=False)
        x_cart = cart_pre(x_full); x_polar = polar_pre(x_full)
        with torch.enable_grad():
            xp = x_polar.detach().requires_grad_(True)
            cam_p = cam_tool(xp, target_class=dense_target(orig_ids, c))
            ig_p = integrated_gradients(model, x_polar.detach(),
                                         target_class=dense_target(orig_ids, c), n_steps=ig_steps)
        avg_cart[c] = x_cart[0, 0].detach().cpu().numpy()
        cam_cart[c] = polar_cam_to_cartesian(cam_p).detach().cpu().numpy()
        ig_cart[c] = polar_cam_to_cartesian(ig_p).detach().cpu().numpy()

    # Apply Gaussian smoothing for paper-quality look
    cam_cart = {c: _gaussian_blur(v, attribution_sigma) for c, v in cam_cart.items()}
    ig_cart = {c: _gaussian_blur(v, attribution_sigma) for c, v in ig_cart.items()}

    H = avg_cart[order[0]].shape[-1]
    bm_disp = _beam_mask(H, H, bm_r * H / 192.0)

    # =================================================================
    # FIGURE 1: K rows x 5 cols summary (class avg attribution + 1 sample)
    # =================================================================
    PANEL = 2.6
    sample_for_proto = {}     # cache: class -> (raw cart, cam_cart smoothed)

    for c in order:
        if c not in avg_cart:
            continue
        idx = np.where(assigns == c)[0]
        scores = soft_probs[idx, c]
        top1 = idx[np.argsort(-scores)[:1]]
        if len(top1) == 0:
            continue
        i0 = int(top1[0])
        tns = dataset[i0]
        x_full = tns.to(device).float().unsqueeze(0)
        x_cart_s = cart_pre(x_full)
        x_polar_s = polar_pre(x_full)
        with torch.enable_grad():
            xp = x_polar_s.detach().requires_grad_(True)
            cam_ps = cam_tool(xp, target_class=dense_target(orig_ids, c)).detach().cpu().numpy()
        cam_cs = polar_cam_to_cartesian(
            torch.from_numpy(cam_ps).to(device)).cpu().numpy()
        sample_for_proto[c] = (
            x_cart_s[0, 0].detach().cpu().numpy(),
            _gaussian_blur(cam_cs, attribution_sigma),
            i0, float(soft_probs[i0, c]),
        )

    fig, axes = plt.subplots(K, 5, figsize=(5 * PANEL, K * PANEL),
                              squeeze=False)
    col_titles = ["class average", "+ GradCAM", "+ IG",
                  "exemplar sample", "+ GradCAM"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12, pad=10)
    for r, c in enumerate(order):
        if c not in avg_cart:
            for j in range(5):
                axes[r, j].set_axis_off()
            continue
        disp_avg = _log_disp(avg_cart[c], bm_disp)
        # col 1: class avg
        axes[r, 0].imshow(disp_avg, cmap="inferno")
        # col 2: GradCAM overlay
        axes[r, 1].imshow(disp_avg, cmap="inferno")
        axes[r, 1].imshow(_norm01(cam_cart[c]), cmap="turbo", alpha=0.50)
        # col 3: IG overlay
        axes[r, 2].imshow(disp_avg, cmap="inferno")
        axes[r, 2].imshow(_norm01(ig_cart[c]), cmap="magma", alpha=0.55)
        # col 4 + 5: sample
        if c in sample_for_proto:
            cart_s, cam_s, i0, p_s = sample_for_proto[c]
            disp_s = _log_disp(cart_s, bm_disp)
            axes[r, 3].imshow(disp_s, cmap="inferno")
            axes[r, 4].imshow(disp_s, cmap="inferno")
            axes[r, 4].imshow(_norm01(cam_s), cmap="turbo", alpha=0.50)
            axes[r, 4].set_xlabel(f"i={i0}  p={p_s:.2f}",
                                    fontsize=8, labelpad=2)
        axes[r, 0].set_ylabel(f"p{c}\nN={int(counts[c])}",
                                fontsize=11, rotation=0, labelpad=42,
                                ha="right", va="center")
        for j in range(5):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])

    fig.suptitle(f"Paper attribution -- {sample} -- "
                  f"each row: prototype's learned 'physics' "
                  f"(GradCAM/IG on class-avg + an exemplar)",
                  fontsize=13, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_dir = os.path.join(run_dir, "eval", "paper_attribution")
    os.makedirs(out_dir, exist_ok=True)
    fname1 = os.path.join(out_dir, "fig_paper_attribution.png")
    fig.savefig(fname1, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[paper-attr] wrote {fname1}", flush=True)

    # =================================================================
    # FIGURE 2 (per prototype): class avg + N samples, 3 cols (raw, GC, IG)
    # =================================================================
    PANEL2 = 3.0
    n_per = max(1, int(n_samples_per_proto))
    for c in order:
        if c not in avg_cart:
            continue
        idx = np.where(assigns == c)[0]
        scores = soft_probs[idx, c]
        top = idx[np.argsort(-scores)[:n_per]]
        rows = []
        for i in top:
            i = int(i)
            x_full = dataset[i].to(device).float().unsqueeze(0)
            x_cart_s = cart_pre(x_full)
            x_polar_s = polar_pre(x_full)
            with torch.enable_grad():
                xp = x_polar_s.detach().requires_grad_(True)
                cam_ps = cam_tool(xp, target_class=dense_target(orig_ids, c)).detach().cpu().numpy()
                ig_ps = integrated_gradients(model, x_polar_s.detach(),
                                              target_class=dense_target(orig_ids, c),
                                              n_steps=ig_steps).detach().cpu().numpy()
            cam_cs = _gaussian_blur(polar_cam_to_cartesian(
                torch.from_numpy(cam_ps).to(device)).cpu().numpy(),
                attribution_sigma)
            ig_cs = _gaussian_blur(polar_cam_to_cartesian(
                torch.from_numpy(ig_ps).to(device)).cpu().numpy(),
                attribution_sigma)
            rows.append((x_cart_s[0, 0].detach().cpu().numpy(),
                          cam_cs, ig_cs, i, float(soft_probs[i, c])))

        n_rows = 1 + len(rows)
        fig, axes = plt.subplots(n_rows, 3,
                                  figsize=(3 * PANEL2, n_rows * PANEL2),
                                  squeeze=False)
        for j, t in enumerate(["pattern (Cartesian)", "+ GradCAM", "+ IG"]):
            axes[0, j].set_title(t, fontsize=12, pad=8)
        # Row 0: class average
        disp_avg = _log_disp(avg_cart[c], bm_disp)
        axes[0, 0].imshow(disp_avg, cmap="inferno")
        axes[0, 1].imshow(disp_avg, cmap="inferno")
        axes[0, 1].imshow(_norm01(cam_cart[c]), cmap="turbo", alpha=0.50)
        axes[0, 2].imshow(disp_avg, cmap="inferno")
        axes[0, 2].imshow(_norm01(ig_cart[c]), cmap="magma", alpha=0.55)
        axes[0, 0].set_ylabel(f"class avg\nN={int(counts[c])}",
                                fontsize=11, rotation=0, labelpad=42,
                                ha="right", va="center")
        for j in range(3):
            axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        # Sample rows
        for r_off, (cart_r, cam_r, ig_r, i, p_v) in enumerate(rows):
            r = 1 + r_off
            disp_r = _log_disp(cart_r, bm_disp)
            axes[r, 0].imshow(disp_r, cmap="inferno")
            axes[r, 1].imshow(disp_r, cmap="inferno")
            axes[r, 1].imshow(_norm01(cam_r), cmap="turbo", alpha=0.50)
            axes[r, 2].imshow(disp_r, cmap="inferno")
            axes[r, 2].imshow(_norm01(ig_r), cmap="magma", alpha=0.55)
            axes[r, 0].set_ylabel(f"i={i}\np(c)={p_v:.2f}",
                                    fontsize=10, rotation=0, labelpad=42,
                                    ha="right", va="center")
            for j in range(3):
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        fig.suptitle(f"Prototype p{c} -- learned attention "
                      f"(class average + {len(rows)} samples)",
                      fontsize=13, y=0.997)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        fname2 = os.path.join(out_dir, f"fig_paper_attribution_p{c}.png")
        fig.savefig(fname2, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[paper-attr] wrote {fname2}", flush=True)

    cam_tool.close()
    print(f"[paper-attr] done -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=2.0,
                     help="Gaussian sigma (px) for attribution smoothing")
    ap.add_argument("--ig-steps", type=int, default=50)
    args = ap.parse_args()
    run(args.run_dir, args.sample,
        n_samples_per_proto=args.n_samples,
        attribution_sigma=args.sigma,
        ig_steps=args.ig_steps)


if __name__ == "__main__":
    main()
