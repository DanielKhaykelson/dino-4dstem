"""make_ablation_overlays.py — Generate class-map-on-BF/HAADF overlays for
every (sample, condition) in the K=6 ablation pipeline.

Reads from `_paper_ablation_K6/<sample>/<label>/eval/inference.npz` and
produces:
    fig_overlay_BF.png    (class map + virtual BF, 200 nm scale bar)
    fig_overlay_HAADF.png (class map + virtual HAADF)
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SAMPLES

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "runs", "_paper_ablation_K6")

# scan-pixel size in nm per sample (from earlier metadata reads / standard
# 115k mag NaPHI = 3.55 nm/step; Na007a/b/006a at 14k mag = 18.86 nm/step;
# IMC = 5 nm/step; EuInAs ~ 5 nm/step). Approximate; user can refine.
SCAN_STEP_NM = {
    "Na007a": 18.86, "Na007b": 18.86, "Na006a": 18.86,
    "EuInAs_B100": 5.0,
    "IMC_50nm_SI2": 5.0, "IMC_150nm_SI5": 5.0,
}
SCALE_BAR_NM = 200.0


def _virtual_bf_haadf(cube_path: str, mask_r: int = 15):
    """sum-inside-disk and sum-outside-disk per scan position.
    cube_path may be the .prz path or the .cube.npy sidecar (auto-detect).
    """
    npy = cube_path + ".npy" if cube_path.endswith(".prz") else cube_path
    npy_alt = cube_path[:-4] + ".cube.npy" if cube_path.endswith(".prz") else None
    if os.path.exists(npy):
        cube = np.load(npy, mmap_mode="r")
    elif npy_alt and os.path.exists(npy_alt):
        cube = np.load(npy_alt, mmap_mode="r")
    else:
        z = np.load(cube_path, allow_pickle=True)
        cube = z["data"]
    Nx, Ny, H, W = cube.shape
    yy, xx = np.ogrid[:H, :W]
    disk = (yy - H/2)**2 + (xx - W/2)**2 <= mask_r ** 2
    bf = np.empty((Nx, Ny), dtype=np.float64)
    haadf = np.empty((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        slab = np.asarray(cube[i], dtype=np.float64)
        s = slab.sum(axis=(-1, -2))
        bf_i = (slab * disk).sum(axis=(-1, -2))
        bf[i] = bf_i
        haadf[i] = s - bf_i
    return bf, haadf


def _overlay_figure(bf: np.ndarray, class_map: np.ndarray, title: str,
                     scan_step_nm: float, K: int, out_path: str,
                     base_label: str = "BF"):
    bf_n = (bf - bf.min()) / (bf.max() - bf.min() + 1e-12)
    base_colors = plt.get_cmap("tab10").colors[:K]
    cmap_classes = ListedColormap(base_colors, name=f"K{K}")
    norm = BoundaryNorm(np.arange(K + 1) - 0.5, K)
    aspect = bf.shape[0] / bf.shape[1]
    fig_w = 7.5
    fig_h = max(4.0, fig_w * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(bf_n, cmap="gray", interpolation="nearest", aspect="equal")
    ov = ax.imshow(class_map, cmap=cmap_classes, norm=norm, alpha=0.55,
                    interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    sb_px = SCALE_BAR_NM / scan_step_nm
    sb_x0, sb_y0 = 4, bf.shape[0] - 8
    ax.add_patch(Rectangle((sb_x0, sb_y0), sb_px, 1.5,
                            color="white", ec="black", lw=0.5))
    ax.text(sb_x0 + sb_px / 2, sb_y0 + 4,
             f"{int(SCALE_BAR_NM)} nm",
             ha="center", va="top", color="white", fontsize=11,
             bbox=dict(facecolor="black", edgecolor="none", pad=2))
    cb = fig.colorbar(ov, ax=ax, fraction=0.046, pad=0.02,
                       ticks=range(K))
    cb.set_label("class id", fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process(sample: str, label: str):
    sd = os.path.join(ROOT, sample, label)
    inf_path = os.path.join(sd, "eval", "inference.npz")
    if not os.path.exists(inf_path):
        print(f"[skip] {sample}/{label}: no inference.npz")
        return
    cfg = SAMPLES[sample]
    Nx, Ny = cfg["scan_shape"]
    inf = np.load(inf_path)
    assigns = inf["assigns"]
    K = int(inf["K_original"])
    class_map = assigns.reshape(Nx, Ny)
    cube_path = cfg["path"]

    # mask_r per sample default; resolve from data.py SAMPLES if present
    mask_r = int(cfg.get("center_mask_radius", 15))

    print(f"[{sample}/{label}] computing virtual BF/HAADF (mask_r={mask_r})...",
          flush=True)
    bf, haadf = _virtual_bf_haadf(cube_path, mask_r=mask_r)

    step_nm = SCAN_STEP_NM.get(sample, 5.0)
    title_prefix = f"{sample} {label}  K={K}"
    _overlay_figure(bf, class_map,
                     title=f"{title_prefix}  --  class map on virtual BF",
                     scan_step_nm=step_nm, K=K,
                     out_path=os.path.join(sd, "eval", "fig_overlay_BF.png"),
                     base_label="BF")
    _overlay_figure(haadf, class_map,
                     title=f"{title_prefix}  --  class map on virtual HAADF",
                     scan_step_nm=step_nm, K=K,
                     out_path=os.path.join(sd, "eval", "fig_overlay_HAADF.png"),
                     base_label="HAADF")
    print(f"  wrote {sd}/eval/fig_overlay_BF.png and fig_overlay_HAADF.png",
          flush=True)


def main():
    if not os.path.isdir(ROOT):
        print(f"ROOT does not exist: {ROOT}"); sys.exit(1)
    SAMPLES_LIST = ["EuInAs_B100", "IMC_150nm_SI5", "Na007b", "IMC_50nm_SI2",
                     "Na007a", "Na006a"]
    for s in SAMPLES_LIST:
        for label in ("vanilla", "weight"):
            try:
                process(s, label)
            except Exception as e:
                print(f"[FAIL] {s}/{label}: {e!r}")


if __name__ == "__main__":
    main()
