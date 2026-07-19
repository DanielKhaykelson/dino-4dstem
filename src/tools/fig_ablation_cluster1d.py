"""SI ablation: the 1-D radial clustering loss (cluster1d_lambda 0 vs 0.1).
Rows = EuInAs (layered) and IMC_SI5; cols = loss off vs on. Class maps from
inference.npz (tab20), annotated with active-class count, effective-K, and the
intra/inter feature-distance ratio (higher = better separated classes). Writes to
BorisEdits + latest_review.
  python src/tools/fig_ablation_cluster1d.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
import matplotlib as mpl

MC = "runs/ModelComparisons"; OUT = "docs/paper/draft_v2/figs/BorisEdits"; REVIEW = "docs/paper/draft_v2/figs/latest_review"
SHAPE = {"EuInAs": (66, 396), "IMC_SI5": (128, 128)}
CELLS = {
    "EuInAs": ("03_cluster1d/EuInAs_c1d_off", "03_cluster1d/EuInAs_c1d_on"),
    "IMC_SI5": ("03_cluster1d/IMC_SI5_c1d_off", "02_resnet_depth/IMC_SI5_L1"),
}
_LET = "abcdefghijklmnop"


def load(cell):
    z = np.load(os.path.join(MC, cell, "eval", "inference.npz"), allow_pickle=True)
    return z["assigns"].astype(int), json.load(open(os.path.join(MC, cell, "eval", "metrics.json")))


def classmap(ax, cell, sample, title, idx):
    a, m = load(cell); ny, nx = SHAPE[sample]; asg = a.reshape(ny, nx)
    uni = sorted(set(asg.ravel().tolist())); lut = {u: i for i, u in enumerate(uni)}
    cmap = ListedColormap([mpl.colormaps.get_cmap("tab20").resampled(max(len(uni), 1))(i) for i in range(len(uni))])
    ax.imshow(np.vectorize(lut.get)(asg).astype(float), cmap=cmap, interpolation="nearest",
              vmin=0, vmax=max(len(uni) - 1, 1)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"K={m.get('K_active')},  eff-K={m.get('effective_K',0):.1f},  intra/inter={m.get('intra_over_inter',0):.2f}", fontsize=8.5)
    ax.text(0.04, 0.93, _LET[idx], transform=ax.transAxes, fontsize=15, fontweight="bold", va="top",
            ha="left", color="white", bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"), zorder=20)


fig = Figure(figsize=(8.4, 6.6), facecolor="white")
gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0], hspace=0.32, wspace=0.12)
for ri, s in enumerate(["EuInAs", "IMC_SI5"]):
    off, on = CELLS[s]
    classmap(fig.add_subplot(gs[ri, 0]), off, s, f"{s}: 1-D loss off", ri * 2 + 0)
    classmap(fig.add_subplot(gs[ri, 1]), on, s, f"{s}: 1-D loss on", ri * 2 + 1)
fig.suptitle("Effect of the 1-D radial clustering loss. With the loss on, EuInAs classes track the\n"
             "layer structure more cleanly (higher intra/inter separation, fewer leaked classes).", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "fig_ablation_cluster1d.png"); fig.savefig(p, dpi=170, facecolor="white")
import shutil; shutil.copy(p, os.path.join(REVIEW, "fig_ablation_cluster1d.png"))
print("wrote fig_ablation_cluster1d.png", flush=True)
