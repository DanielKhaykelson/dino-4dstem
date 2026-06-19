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
import matplotlib as mpl

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits")
NAMES = ["SI3", "SI4", "SI5"]
ROLE = {"SI3": "overview", "SI4": "needles", "SI5": "interface"}
COLS = [("kmeans", "NMF + k-means"), ("aglo", "NMF + agglomerative"), ("gmm", "NMF + GMM")]


def discrete(ax, lab):
    uni = sorted(np.unique(lab).tolist()); has_noise = -1 in uni
    base = mpl.colormaps.get_cmap("tab20").resampled(max(len([u for u in uni if u != -1]), 1))
    cols = [(0.72, 0.72, 0.72) if u == -1 else base(i - (1 if has_noise else 0)) for i, u in enumerate(uni)]
    lut = {u: i for i, u in enumerate(uni)}
    ax.imshow(np.vectorize(lut.get)(lab).astype(float), cmap=ListedColormap(cols),
              interpolation="nearest", vmin=0, vmax=max(len(uni) - 1, 1))
    ax.set_xticks([]); ax.set_yticks([])


ncol = 2 + len(COLS)
fig = Figure(figsize=(2.05 * ncol, 2.15 * 3 + 0.5), facecolor="white")
for ri, n in enumerate(NAMES):
    z = np.load(os.path.join(FIGS, f"boris_nmf_cache_{n}.npz"))
    # col 0: HAADF with ROI
    ax = fig.add_subplot(3, ncol, ri * ncol + 1); ax.set_xticks([]); ax.set_yticks([])
    try:
        ax.imshow(mpimg.imread(os.path.join(OUT, f"haadf_{n}.png")))
    except Exception:
        ax.text(0.5, 0.5, "HAADF", ha="center", va="center")
    ax.set_ylabel(f"{n}\n{ROLE[n]}", fontsize=11, fontweight="bold", rotation=0, labelpad=20, va="center")
    if ri == 0: ax.set_title("HAADF (scan ROI)", fontsize=10)
    # col 1: DINO
    ax = fig.add_subplot(3, ncol, ri * ncol + 2); discrete(ax, z["dino"])
    if ri == 0: ax.set_title("DINO", fontsize=10, fontweight="bold")
    # cols 2..: NMF variants
    for ci, (key, title) in enumerate(COLS):
        ax = fig.add_subplot(3, ncol, ri * ncol + 3 + ci); discrete(ax, z[key])
        if ri == 0: ax.set_title(title, fontsize=10)
fig.suptitle("IMC clustering. For each field of view (rows): the HAADF with the 4D-STEM scan region, the DINO class map, and NMF clustered by "
             "k-means, agglomerative, and Gaussian-mixture. Colours are arbitrary cluster labels.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "fig3_clustering.png"); fig.savefig(p, dpi=170, facecolor="white")
print(f"wrote {p}", flush=True)
