"""Boris SI figure: the two NMF clusterings moved out of main Fig 3 — HDBSCAN and
fuzzy c-means — for the three IMC samples. Shows they give yet different (and, for
HDBSCAN, collapsed) partitions, i.e. NMF is sensitive to the clustering choice.
  python tools/boris_nmf_extra.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
import matplotlib as mpl

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits")
NAMES = ["SI3", "SI4", "SI5"]; ROLE = {"SI3": "interface", "SI4": "needles", "SI5": "magnified interface"}
COLS = [("dino", "DINO (reference)"), ("hdbscan", "NMF + HDBSCAN"), ("fcm", "NMF + fuzzy c-means")]


def discrete(ax, lab):
    uni = sorted(np.unique(lab).tolist()); has_noise = -1 in uni
    base = mpl.colormaps.get_cmap("tab20").resampled(max(len([u for u in uni if u != -1]), 1))
    cols = [(0.72, 0.72, 0.72) if u == -1 else base(i - (1 if has_noise else 0)) for i, u in enumerate(uni)]
    lut = {u: i for i, u in enumerate(uni)}
    ax.imshow(np.vectorize(lut.get)(lab).astype(float), cmap=ListedColormap(cols),
              interpolation="nearest", vmin=0, vmax=max(len(uni) - 1, 1))
    ax.set_xticks([]); ax.set_yticks([])


fig = Figure(figsize=(2.15 * 3 + 0.7, 2.15 * 3 + 0.6), facecolor="white")
row_axes = []
for ri, n in enumerate(NAMES):
    z = np.load(os.path.join(FIGS, f"boris_nmf_cache_{n}.npz"))
    for ci, (key, title) in enumerate(COLS):
        ax = fig.add_subplot(3, 3, ri * 3 + ci + 1); discrete(ax, z[key])
        if ci == 0: row_axes.append((ROLE[n], ax))
        if ri == 0: ax.set_title(title, fontsize=10)
fig.tight_layout(rect=[0.13, 0, 1, 0.98]); FigureCanvasAgg(fig)
# row labels centred in the left gutter (vertical, clear of both frame and canvas edge)
for lbl, ax in row_axes:
    bb = ax.get_position(); yc = (bb.y0 + bb.y1) / 2
    fig.text(bb.x0 - 0.045, yc, lbl, rotation=90, ha="center", va="center", fontsize=12, fontweight="bold")
p = os.path.join(OUT, "nmf_hdbscan_fcm.png"); fig.savefig(p, dpi=170, facecolor="white")
print(f"wrote {p}", flush=True)
