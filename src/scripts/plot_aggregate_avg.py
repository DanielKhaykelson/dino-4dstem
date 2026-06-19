"""plot_aggregate_avg.py -- compute an aggregate class average over a
chosen subset of prototype IDs (e.g., "all except p1").

Usage:
    python plot_aggregate_avg.py --run-dir runs/_paper_master/Na007b_K6/transfer/Na007a --sample Na007a --exclude 1
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import argparse, os, sys, json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="prototype IDs to exclude (e.g. 1)")
    ap.add_argument("--include", type=int, nargs="*", default=None,
                    help="prototype IDs to include (overrides --exclude)")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--n-top", type=int, default=2000,
                    help="confidence-weighted top-N within the kept set")
    args = ap.parse_args()

    cfg = SAMPLES[args.sample]
    vmax = args.vmax if args.vmax is not None else cfg["vmax"]
    inf_path = os.path.join(args.run_dir, "eval", "inference.npz")
    inf = np.load(inf_path)
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]
    K = soft_probs.shape[1]

    if args.include is not None:
        keep = sorted(set(args.include))
    else:
        excluded = set(args.exclude)
        keep = sorted(c for c in range(K) if c not in excluded)
    keep_arr = np.array(keep, dtype=np.int64)
    print(f"[agg-avg] K={K}  keep={keep}  exclude={list(set(range(K))-set(keep))}",
          flush=True)
    print(f"[agg-avg] vmax={vmax}", flush=True)

    in_keep = np.isin(assigns, keep_arr)
    idx_keep = np.where(in_keep)[0]
    if idx_keep.size == 0:
        sys.exit("no patterns matching keep set")
    # Confidence weight = max softmax over the KEPT prototypes only
    sp_keep = soft_probs[idx_keep][:, keep_arr]
    conf = sp_keep.max(axis=1)
    n_top = min(args.n_top, idx_keep.size)
    order = np.argsort(-conf)[:n_top]
    chosen = idx_keep[order]
    print(f"[agg-avg] using top-{n_top}/{idx_keep.size} kept patterns "
          f"(by max softmax over kept protos)", flush=True)

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
    cy = cx = H / 2.0
    yy, xx = np.ogrid[:H, :H]
    bm = ((yy - cy) ** 2 + (xx - cx) ** 2) > MASK_R ** 2
    ref = (x_cart * bm).flatten(); ref = ref[ref > 0]
    if ref.size:
        lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
        disp = np.log1p(np.clip(x_cart, lo, hi) - lo) * bm
    else:
        disp = x_cart * bm

    avg_dir = os.path.join(args.run_dir, "eval", "class_averages")
    os.makedirs(avg_dir, exist_ok=True)
    keep_tag = "p" + "_p".join(str(c) for c in keep)
    excl_tag = "no_p" + "_p".join(str(c) for c in sorted(set(args.exclude))) \
                if not args.include else f"only_{keep_tag}"
    out_png = os.path.join(avg_dir, f"aggregate_{excl_tag}.png")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
    n_kept = int(in_keep.sum())
    pct = 100.0 * n_kept / len(assigns)
    ax.set_title(
        f"{args.sample} aggregate class avg over {keep}\n"
        f"({n_kept}/{len(assigns)} = {pct:.1f}% of patterns)",
        fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[agg-avg] wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
