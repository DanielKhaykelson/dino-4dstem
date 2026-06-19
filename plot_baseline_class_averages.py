"""plot_baseline_class_averages.py -- post-process for the cepstral+NMF
baseline. Reads the saved inference.npz, computes per-class average
diffraction (raw Cartesian, log1p, beam-mask r<15px) and writes
class_averages/p{c}.png next to the baseline outputs.

Usage:
    python plot_baseline_class_averages.py --run-dir runs/_baselines/cepstral_nmf_Na007b_K6_v5.0 --sample Na007b --vmax 5
"""
from __future__ import annotations
import argparse, os, sys
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

H = 192
MASK_R = 15
CENTER_CROP = 140
N_TOP = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--vmax", type=float, default=None)
    args = ap.parse_args()

    cfg = SAMPLES[args.sample]
    vmax = args.vmax if args.vmax is not None else cfg["vmax"]
    inf = np.load(os.path.join(args.run_dir, "inference.npz"))
    assigns = inf["assigns"]
    W = inf["W"]              # (N, K) NMF weights
    K = W.shape[1]
    counts = np.bincount(assigns, minlength=K)
    print(f"[baseline-avg] K={K}  counts={counts.tolist()}", flush=True)

    ds = LoadPRZ(cfg["path"], resize=H, vmax=vmax)
    cart_pre = T.Compose([
        T.CenterCrop(CENTER_CROP),
        T.Resize(H, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    cy = cx = H / 2.0
    yy, xx = np.ogrid[:H, :H]
    bm = ((yy - cy) ** 2 + (xx - cx) ** 2) > MASK_R ** 2

    out_dir = os.path.join(args.run_dir, "class_averages")
    os.makedirs(out_dir, exist_ok=True)

    # Combined panel for quick glance
    fig_panel, axes_panel = plt.subplots(2, 4, figsize=(14, 7))
    axes_panel = axes_panel.flatten()

    for c in range(K):
        idx = np.where(assigns == c)[0]
        if idx.size == 0:
            print(f"  p{c}: empty", flush=True)
            axes_panel[c].set_axis_off()
            continue
        # weight by NMF weight on this cluster's primary component (use the
        # cluster centroid in W-space distance to pick top-N)
        # Simpler & faithful: top-N by Euclidean distance from cluster mean in W
        mean_W = W[idx].mean(axis=0)
        dists = np.linalg.norm(W[idx] - mean_W[None, :], axis=1)
        order = np.argsort(dists)
        top = idx[order[:min(N_TOP, len(idx))]]

        patterns = np.stack([ds.get_raw(int(i)) for i in top], 0).astype(np.float32)
        avg = patterns.mean(0)
        avg_norm = np.clip(avg / float(vmax), 0.0, 1.0)
        x = torch.from_numpy(avg_norm).unsqueeze(0).unsqueeze(0).float()
        x = F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)
        x_cart = cart_pre(x)[0, 0].cpu().numpy()
        ref = (x_cart * bm).flatten(); ref = ref[ref > 0]
        if ref.size:
            lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
            disp = np.log1p(np.clip(x_cart, lo, hi) - lo) * bm
        else:
            disp = x_cart * bm

        # individual PNG
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
        ax.set_title(f"{args.sample} (cepstral+NMF) p{c} N={int(counts[c])}",
                      fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        out = os.path.join(out_dir, f"p{c}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # combined panel
        ax = axes_panel[c]
        ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
        ax.set_title(f"p{c}  N={int(counts[c])}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        print(f"  p{c}: N={counts[c]}  -> {out}", flush=True)

    for j in range(K, len(axes_panel)):
        axes_panel[j].set_axis_off()
    fig_panel.suptitle(
        f"{args.sample} cepstral+NMF+kmeans K={K}  (class averages, top-{N_TOP})",
        fontsize=12)
    fig_panel.tight_layout()
    panel_out = os.path.join(args.run_dir, "fig_class_averages_grid.png")
    fig_panel.savefig(panel_out, dpi=180, bbox_inches="tight",
                       facecolor="white")
    plt.close(fig_panel)
    print(f"[baseline-avg] grid -> {panel_out}", flush=True)


if __name__ == "__main__":
    main()
