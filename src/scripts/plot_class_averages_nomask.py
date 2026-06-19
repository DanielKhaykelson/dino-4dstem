"""plot_class_averages_nomask.py -- render per-prototype class averages
WITHOUT the central beam-mask disk (so the central beam halo is visible).

Same pipeline as viz_paper_outputs but skips the beam_mask multiplication.
Saves to <run_dir>/eval/class_averages_nomask/p{c}.png so the original
masked versions are preserved.

Run on multiple (run_dir, sample) pairs in one shot via a small driver.
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
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
CENTER_CROP = 140


def render_one(run_dir, sample, vmax=None, n_top_avg=300):
    cfg = SAMPLES[sample]
    if vmax is None:
        vmax = cfg["vmax"]
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]
    K = soft_probs.shape[1]
    counts = np.bincount(assigns, minlength=K)

    ds = LoadPRZ(cfg["path"], resize=H, vmax=vmax)
    cart_pre = T.Compose([
        T.CenterCrop(CENTER_CROP),
        T.Resize(H, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    out_dir = os.path.join(run_dir, "eval", "class_averages_nomask")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{sample}] {run_dir}  vmax={vmax}  K={K}  counts={counts.tolist()}",
          flush=True)
    for c in range(K):
        idx = np.where(assigns == c)[0]
        if idx.size == 0:
            print(f"  p{c}: empty", flush=True); continue
        scores = soft_probs[idx, c]
        top_avg = idx[np.argsort(-scores)[:min(n_top_avg, len(idx))]]
        patterns = np.stack([ds.get_raw(int(i)) for i in top_avg], 0).astype(np.float32)
        w = soft_probs[top_avg, c].astype(np.float32)
        wavg = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
        wavg_norm = np.clip(wavg / float(vmax), 0.0, 1.0)
        x = torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0).float()
        x = F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)
        x_cart = cart_pre(x)[0, 0].cpu().numpy()
        # NO beam mask. Percentile-clip on the WHOLE image (central beam
        # included) so the dynamic range still reveals the spots.
        ref = x_cart.flatten()
        lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
        disp = np.log1p(np.clip(x_cart, lo, hi) - lo)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
        ax.set_title(f"{sample}  p{c}  N={int(counts[c])}  (no mask)",
                      fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"p{c}.png"), dpi=200,
                     bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  p{c}: N={counts[c]}  -> nomask png", flush=True)


def render_aggregate(run_dir, sample, include=None, exclude=None,
                       vmax=None, n_top=2000):
    cfg = SAMPLES[sample]
    if vmax is None:
        vmax = cfg["vmax"]
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]; assigns = inf["assigns"]
    K = soft_probs.shape[1]
    if include is not None:
        keep = sorted(set(include))
    else:
        excl = set(exclude or [])
        keep = sorted(c for c in range(K) if c not in excl)
    keep_arr = np.array(keep, dtype=np.int64)
    in_keep = np.isin(assigns, keep_arr)
    idx_keep = np.where(in_keep)[0]
    if idx_keep.size == 0:
        return
    sp_keep = soft_probs[idx_keep][:, keep_arr]
    conf = sp_keep.max(axis=1)
    n_top_use = min(n_top, idx_keep.size)
    order = np.argsort(-conf)[:n_top_use]
    chosen = idx_keep[order]

    ds = LoadPRZ(cfg["path"], resize=H, vmax=vmax)
    patterns = np.stack([ds.get_raw(int(i)) for i in chosen], 0).astype(np.float32)
    w = conf[order].astype(np.float32)
    wavg = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
    wavg_norm = np.clip(wavg / float(vmax), 0.0, 1.0)
    x = torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0).float()
    x = F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)
    cart_pre = T.Compose([
        T.CenterCrop(CENTER_CROP),
        T.Resize(H, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    x_cart = cart_pre(x)[0, 0].cpu().numpy()
    ref = x_cart.flatten()
    lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
    disp = np.log1p(np.clip(x_cart, lo, hi) - lo)
    out_dir = os.path.join(run_dir, "eval", "class_averages_nomask")
    os.makedirs(out_dir, exist_ok=True)
    if include is not None:
        tag = "only_p" + "_p".join(str(c) for c in keep)
    else:
        tag = "no_p" + "_p".join(str(c) for c in sorted(set(exclude or [])))
    out_png = os.path.join(out_dir, f"aggregate_{tag}.png")
    n_kept = int(in_keep.sum()); pct = 100.0 * n_kept / len(assigns)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
    ax.set_title(f"{sample} aggregate over {keep} (no mask)\n"
                  f"({n_kept}/{len(assigns)} = {pct:.1f}%)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  aggregate {tag} -> {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--exclude", type=int, nargs="*", default=None)
    ap.add_argument("--include", type=int, nargs="*", default=None)
    args = ap.parse_args()
    render_one(args.run_dir, args.sample, vmax=args.vmax)
    if args.exclude is not None or args.include is not None:
        render_aggregate(args.run_dir, args.sample,
                          include=args.include, exclude=args.exclude,
                          vmax=args.vmax)


if __name__ == "__main__":
    main()
