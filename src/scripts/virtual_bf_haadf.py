"""virtual_bf_haadf.py -- compute BF and HAADF virtual-detector images
from raw 4DSTEM cubes for the MgPhi tilt samples.

Per scan position (y, x) in a (Ny, Nx, H, W) cube:
    BF   = sum_{(ky, kx) in disk r<r_BF}        I(ky, kx)
    HAADF = sum_{(ky, kx) in annulus r_in<r<r_out} I(ky, kx)

Radii proportional to the diffraction-frame size (H = W) so this
generalises to other cubes:
    r_BF      = 0.06 * H       (central beam + bleeding halo)
    r_HAADF_in= 0.18 * H       (just outside the BF disk)
    r_HAADF_out=0.45 * H       (avoids the corner aperture clip)

Output (per sample):
    <run_dir>/eval/virtual/
        fig_BF.png
        fig_HAADF.png
        fig_BF_HAADF.png       (side-by-side)
        BF.npy   HAADF.npy
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from data import SAMPLES


def _open_lazy(path):
    if path.lower().endswith((".prz", ".npz")):
        base, _ = os.path.splitext(path)
        cand = base + ".cube.npy"
        if os.path.exists(cand):
            return np.load(cand, mmap_mode="r")
        arr = np.load(path, allow_pickle=True, mmap_mode="r")
        return arr["data"]
    return np.load(path, mmap_mode="r")


def _radial_mask(H, W, r_in, r_out):
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (r2 >= r_in ** 2) & (r2 < r_out ** 2)


def compute_virtual(sample, out_dir,
                     bf_frac=0.06, haadf_in_frac=0.18, haadf_out_frac=0.45):
    cfg = SAMPLES[sample]
    cube = _open_lazy(cfg["path"])
    Ny, Nx, H, W = cube.shape
    print(f"[{sample}] cube={cube.shape}  scan={Ny}x{Nx}  frame={H}x{W}",
          flush=True)
    r_BF = bf_frac * H
    r_in = haadf_in_frac * H
    r_out = haadf_out_frac * H
    bf_mask = _radial_mask(H, W, 0, r_BF)
    haadf_mask = _radial_mask(H, W, r_in, r_out)
    print(f"  r_BF={r_BF:.1f}  r_HAADF=[{r_in:.1f}, {r_out:.1f}]  "
          f"BF pixels={bf_mask.sum()}  HAADF pixels={haadf_mask.sum()}",
          flush=True)

    BF = np.zeros((Ny, Nx), dtype=np.float64)
    HA = np.zeros((Ny, Nx), dtype=np.float64)

    t0 = time.perf_counter()
    for y in range(Ny):
        block = np.asarray(cube[y]).astype(np.float32)  # (Nx, H, W)
        BF[y] = (block * bf_mask).sum(axis=(1, 2))
        HA[y] = (block * haadf_mask).sum(axis=(1, 2))
    dt = time.perf_counter() - t0
    print(f"  computed BF + HAADF in {dt:.1f}s "
          f"({dt * 1000.0 / (Ny * Nx):.2f} ms/pattern)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "BF.npy"), BF)
    np.save(os.path.join(out_dir, "HAADF.npy"), HA)

    aspect = Nx / max(Ny, 1)
    if aspect > 1:
        fig_h = 4.5; fig_w = min(13, fig_h * aspect)
    else:
        fig_w = 4.5; fig_h = min(12, fig_w / aspect)

    for label, img in [("BF", BF), ("HAADF", HA)]:
        # display with mild percentile clip
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        disp = np.clip(img, lo, hi)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(disp, cmap="gray", aspect="equal",
                        interpolation="nearest")
        ax.set_title(f"{sample}  virtual {label}  "
                      f"({Ny}x{Nx})", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"fig_{label}.png")
        fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {out_png}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(min(13, 2 * fig_w + 1), fig_h))
    for ax, (label, img) in zip(axes, [("BF", BF), ("HAADF", HA)]):
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        ax.imshow(np.clip(img, lo, hi), cmap="gray", aspect="equal",
                   interpolation="nearest")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{sample}  virtual BF / HAADF "
                  f"(BF r<{r_BF:.0f}px, HAADF {r_in:.0f}-{r_out:.0f}px on "
                  f"{H}x{W} frame)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_BF_HAADF.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="append", required=True,
                    help="sample registry key (repeatable)")
    ap.add_argument("--run-dir", action="append", required=True,
                    help="run dir under which to save eval/virtual/ "
                         "(repeatable, paired with --sample)")
    ap.add_argument("--bf-frac", type=float, default=0.06)
    ap.add_argument("--haadf-in-frac", type=float, default=0.18)
    ap.add_argument("--haadf-out-frac", type=float, default=0.45)
    args = ap.parse_args()
    if len(args.sample) != len(args.run_dir):
        sys.exit("--sample and --run-dir must come in the same number of pairs")
    for s, r in zip(args.sample, args.run_dir):
        compute_virtual(s, os.path.join(r, "eval", "virtual"),
                          bf_frac=args.bf_frac,
                          haadf_in_frac=args.haadf_in_frac,
                          haadf_out_frac=args.haadf_out_frac)


if __name__ == "__main__":
    main()
