"""overlay_classes_on_haadf.py -- overlay a set of prototype IDs (single
color, alpha-blended) on the virtual HAADF image of a sample, plus produce
an unmasked aggregate average diffraction of those classes.

Usage:
    python overlay_classes_on_haadf.py \\
        --run-dir runs/_mgphi_tilt/NBED001a_K8_30ep_v5 \\
        --sample MgPhi_tilt_NBED001a \\
        --classes 0 2 6 \\
        --vmax 5
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
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from data import SAMPLES
import plot_class_averages_nomask as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--classes", type=int, nargs="+", required=True)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--color", default="#d62728",
                    help="hex color for the overlay (default red)")
    args = ap.parse_args()

    cfg = SAMPLES[args.sample]
    Ny, Nx = cfg["scan_shape"]
    inf = np.load(os.path.join(args.run_dir, "eval", "inference.npz"))
    assigns = inf["assigns"]
    mask = np.isin(assigns, args.classes).reshape(Ny, Nx)
    haadf_path = os.path.join(args.run_dir, "eval", "virtual", "HAADF.npy")
    if not os.path.exists(haadf_path):
        sys.exit(f"missing HAADF.npy at {haadf_path}")
    haadf = np.load(haadf_path)

    # ----- overlay figure -----
    out_dir = os.path.join(args.run_dir, "eval", "virtual")
    os.makedirs(out_dir, exist_ok=True)
    cls_tag = "_".join(f"p{c}" for c in args.classes)
    aspect = Nx / max(Ny, 1)
    if aspect > 1:
        fig_h = 5.0; fig_w = min(13, fig_h * aspect)
    else:
        fig_w = 5.0; fig_h = min(12, fig_w / aspect)

    lo, hi = np.percentile(haadf, 1), np.percentile(haadf, 99)
    haadf_disp = np.clip(haadf, lo, hi)

    # build a colored RGBA overlay
    from matplotlib.colors import to_rgb
    rgb = to_rgb(args.color)
    overlay = np.zeros((Ny, Nx, 4), dtype=np.float32)
    overlay[mask, 0] = rgb[0]; overlay[mask, 1] = rgb[1]
    overlay[mask, 2] = rgb[2]; overlay[mask, 3] = float(args.alpha)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(haadf_disp, cmap="gray", aspect="equal",
               interpolation="nearest")
    ax.imshow(overlay, aspect="equal", interpolation="nearest")
    ax.set_title(f"{args.sample}  HAADF + {{{','.join(f'p{c}' for c in args.classes)}}} "
                  f"({int(mask.sum())}/{Ny * Nx} pixels = "
                  f"{100*mask.sum()/(Ny*Nx):.1f}%)",
                  fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out_overlay = os.path.join(out_dir, f"fig_HAADF_overlay_{cls_tag}.png")
    fig.savefig(out_overlay, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_overlay}", flush=True)

    # ----- unmasked aggregate average via shared helper -----
    M.render_aggregate(args.run_dir, args.sample,
                        include=args.classes, vmax=args.vmax)


if __name__ == "__main__":
    main()
