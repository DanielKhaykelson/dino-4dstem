"""Boris Fig 3: IMC clustering. Rows SI3/SI4/SI5; cols = HAADF (scan ROI) | DINO |
NMF+k-means | NMF+agglomerative | NMF+GMM. HDBSCAN and FCM go to the SI (see
boris_nmf_extra.py). Plain clustering maps, distinct colours; no interpretation.
  python tools/boris_fig3.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib import image as mpimg
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import matplotlib as mpl
from gui_app.crystallinity_panel import _radial_mean_var

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits")
NAMES = ["SI3", "SI4", "SI5"]
ROLE = {"SI3": "interface", "SI4": "needles", "SI5": "interface (magnified)"}
NMPX = {"SI3": 44.0, "SI4": 44.0, "SI5": 16.0}   # nm per scan pixel
BARNM = {"SI3": 1000, "SI4": 1000, "SI5": 500}
COLS = [("kmeans", "NMF + k-means"), ("aglo", "NMF + agglomerative"), ("gmm", "NMF + GMM")]
INV = 0.00185; KMAX = 0.35; FOV = {"SI3": 187, "SI4": 160, "SI5": 160}


def spot_rank(name, dino):
    """{class: rank} ordered by median azimuthal spottiness (low->high); classes
    with no measurable grain rank lowest."""
    try:
        z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    except Exception:
        return {int(c): i for i, c in enumerate(sorted(np.unique(dino)))}
    cls, gsum, gcnt, vac = z["cls"], z["gsum"], z["gcnt"], z["vac"]
    H = int(z["H"]); cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1
    hi = min(int(KMAX / INV), FOV[name])
    spot = np.full(gsum.shape[0], np.nan)
    for g in range(gsum.shape[0]):
        if vac[g]:
            continue
        m, v, _ = _radial_mean_var(gsum[g] / max(gcnt[g], 1), (cyx, cyx), beam_px=beam)
        seg = m[lo:hi]; vseg = v[lo:hi]
        if seg.size < 5 or seg.sum() <= 0:
            continue
        spot[g] = np.percentile(np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None), 90)
    cval = {}
    for c in np.unique(dino):
        gv = spot[cls == c]; gv = gv[np.isfinite(gv)]
        cval[int(c)] = float(np.median(gv)) if gv.size else -np.inf
    order = sorted(np.unique(dino).tolist(), key=lambda c: cval[int(c)])
    return {int(c): r for r, c in enumerate(order)}


def discrete(ax, lab, rank=None, cmap="tab20b"):
    uni = sorted(np.unique(lab).tolist()); has_noise = -1 in uni
    base = mpl.colormaps.get_cmap(cmap).resampled(max(len([u for u in uni if u != -1]), 1))
    cols = [(0.72, 0.72, 0.72) if u == -1 else base(i - (1 if has_noise else 0)) for i, u in enumerate(uni)]
    lut = rank if rank is not None else {u: i for i, u in enumerate(uni)}
    ax.imshow(np.vectorize(lut.get)(lab).astype(float), cmap=ListedColormap(cols),
              interpolation="nearest", vmin=0, vmax=max(len(uni) - 1, 1))
    ax.set_xticks([]); ax.set_yticks([])


ncol = 2 + len(COLS)
_LET = "abcdefghijklmnopqrstuvwxyz"
def _pl(ax, idx):
    ax.text(0.06, 0.95, _LET[idx], transform=ax.transAxes, fontsize=15, fontweight="bold",
            va="top", ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.14", fc="black", alpha=0.55, ec="none"), zorder=20)
def _scalebar(ax, H, W, n):
    """Clear, Figure-4-style bar: thick white bar + bold black-outlined white label."""
    bpx = BARNM[n] / NMPX[n]; x0 = W * 0.05; y0 = H - H * 0.07
    ax.add_patch(Rectangle((x0, y0), bpx, max(2.5, H * 0.038), color="white", ec="black", lw=0.5, zorder=26))
    lbl = f"{BARNM[n] // 1000} µm" if BARNM[n] >= 1000 else f"{BARNM[n]} nm"
    ax.text(x0, y0 - H * 0.012, lbl, color="white", ha="left", va="bottom",
            fontsize=9.5, fontweight="bold", zorder=26,
            path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])
fig = Figure(figsize=(2.05 * ncol, 2.15 * 3 + 0.5), facecolor="white")
for ri, n in enumerate(NAMES):
    z = np.load(os.path.join(FIGS, f"boris_nmf_cache_{n}.npz"))
    # col 0: HAADF with ROI
    ax = fig.add_subplot(3, ncol, ri * ncol + 1); ax.set_xticks([]); ax.set_yticks([])
    try:
        ax.imshow(mpimg.imread(os.path.join(OUT, f"haadf_{n}.png")))
    except Exception:
        ax.text(0.5, 0.5, "HAADF", ha="center", va="center")
    ax.set_ylabel(ROLE[n], fontsize=9.5, fontweight="bold", rotation=90, labelpad=6, va="center")
    _pl(ax, ri * ncol + 0)
    if ri == 0: ax.set_title("HAADF (scan ROI)", fontsize=10)
    # col 1: DINO (colours ordered by azimuthal spottiness)
    ax = fig.add_subplot(3, ncol, ri * ncol + 2); discrete(ax, z["dino"]); _pl(ax, ri * ncol + 1)
    _scalebar(ax, z["dino"].shape[0], z["dino"].shape[1], n)
    if ri == 0: ax.set_title("DINO", fontsize=10, fontweight="bold")
    # cols 2..: NMF variants (scale bar on every map column, 2 to last)
    for ci, (key, title) in enumerate(COLS):
        ax = fig.add_subplot(3, ncol, ri * ncol + 3 + ci); discrete(ax, z[key]); _pl(ax, ri * ncol + 2 + ci)
        _scalebar(ax, z[key].shape[0], z[key].shape[1], n)
        if ri == 0: ax.set_title(title, fontsize=10)
fig.suptitle("IMC clustering. For each field of view (rows): the HAADF with the 4D-STEM scan region, the DINO class map, and NMF clustered by "
             "k-means, agglomerative, and Gaussian-mixture. Colours are arbitrary cluster labels.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "fig3_clustering.png"); fig.savefig(p, dpi=170, facecolor="white")
print(f"wrote {p}", flush=True)
