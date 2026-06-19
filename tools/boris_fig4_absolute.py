"""Boris Fig 4, alternative: ONE absolute vmin/vmax across all three samples AND
both methods (full min->max of every class B, no percentile clipping, not per
sample), so the rows are directly comparable -- a low-crystallinity field really
reads darker than a high-crystallinity one. Built from the cached class averages
(boris_classavg_*.npz); no cube pass.
  python tools/boris_fig4_absolute.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits")
NAMES = ["SI3", "SI4", "SI5"]; ROLE = {"SI3": "overview", "SI4": "needles", "SI5": "interface"}

DATA = {}; allB = []
for n in NAMES:
    z = np.load(os.path.join(FIGS, f"boris_classavg_{n}.npz"))
    dino = z["dino"]; kmeans = z["kmeans"]
    Bd = {int(k.split("_")[-1]): float(z[k]) for k in z.files if k.startswith("dino_B_")}
    Bn = {int(k.split("_")[-1]): float(z[k]) for k in z.files if k.startswith("nmf_B_")}
    DATA[n] = dict(dino=dino, kmeans=kmeans, Bd=Bd, Bn=Bn)
    allB += list(Bd.values()) + list(Bn.values())

vmin, vmax = float(min(allB)), float(max(allB))   # ABSOLUTE full range, shared everywhere
cmap = mpl.cm.get_cmap("YlOrBr_r").copy()          # brown = low (less ordered), yellow = high


def paint(lab2d, Bm):
    return np.vectorize(lambda c: Bm[int(c)])(lab2d).astype(float)


fig = Figure(figsize=(8.6, 3.1 * 3 + 0.5), facecolor="white")
for ri, n in enumerate(NAMES):
    d = DATA[n]
    for ci, (lab, Bm, title) in enumerate([(d["dino"], d["Bd"], "DINO"), (d["kmeans"], d["Bn"], "NMF (k-means)")]):
        ax = fig.add_subplot(3, 2, ri * 2 + ci + 1); ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(paint(lab, Bm), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if ri == 0: ax.set_title(title, fontsize=11, fontweight="bold")
        if ci == 0: ax.set_ylabel(f"{n}\n{ROLE[n]}", fontsize=11, fontweight="bold", rotation=0, labelpad=20, va="center")
fig.subplots_adjust(right=0.88)
cax = fig.add_axes([0.90, 0.15, 0.02, 0.7]); fig.colorbar(im, cax=cax, label="crystallinity (2D Bragg excess B)")
fig.suptitle("Crystallinity on ONE absolute scale (full range, no clipping; brown = less ordered, yellow = more crystalline).\n"
             "Same colour = same B across all three samples and both methods, so the SI5 interface reads genuinely lower than SI3/SI4.", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 0.88, 0.93]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "fig4_crystallinity_absolute.png"); fig.savefig(p, dpi=170, facecolor="white")
print(f"wrote {p}  (absolute B scale {vmin:.3f}-{vmax:.3f})", flush=True)
