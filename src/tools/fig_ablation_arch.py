"""SI architecture-ablation figures from runs/ModelComparisons/.
 (1) ViT vs ResNet (matched shallow encoder): ViT collapses to a few classes,
     the convolutional encoder keeps the real structure.
 (2) ResNet encoder depth L1..L4: shallow (L1/L2) keeps structure, deep (L3/L4)
     collapses, which is why a shallow trunk is used.
Class maps rendered from inference.npz (tab20), annotated with effective-K and the
number of active classes. Writes to BorisEdits + latest_review.
  python src/tools/fig_ablation_arch.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
import matplotlib as mpl

MC = "runs/ModelComparisons"; OUT = "docs/paper/draft_v2/figs/BorisEdits"; REVIEW = "docs/paper/draft_v2/figs/latest_review"
SHAPE = {"Na007b": (126, 100), "IMC_SI5": (128, 128)}


def load(cell):
    z = np.load(os.path.join(MC, cell, "eval", "inference.npz"), allow_pickle=True)
    a = z["assigns"].astype(int)
    m = json.load(open(os.path.join(MC, cell, "eval", "metrics.json")))
    return a, m


def sample_of(cell):
    return "Na007b" if "Na007b" in cell else "IMC_SI5"


_LET = "abcdefghijklmnop"
def classmap(ax, cell, title, idx=None):
    a, m = load(cell); ny, nx = SHAPE[sample_of(cell)]
    asg = a.reshape(ny, nx)
    uni = sorted(set(asg.ravel().tolist())); lut = {u: i for i, u in enumerate(uni)}
    cmap = ListedColormap([mpl.colormaps.get_cmap("tab20").resampled(max(len(uni), 1))(i) for i in range(len(uni))])
    ax.imshow(np.vectorize(lut.get)(asg).astype(float), cmap=cmap, interpolation="nearest",
              vmin=0, vmax=max(len(uni) - 1, 1)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"active classes = {m.get('K_active')},  eff-K = {m.get('effective_K',0):.1f}", fontsize=8.5)
    if idx is not None:
        ax.text(0.05, 0.95, _LET[idx], transform=ax.transAxes, fontsize=15, fontweight="bold", va="top",
                ha="left", color="white", bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"), zorder=20)


# ---------- Fig A: ViT vs ResNet (matched L1 depth) ----------
figA = Figure(figsize=(7.2, 7.2), facecolor="white")
for ri, s in enumerate(["Na007b", "IMC_SI5"]):
    classmap(figA.add_subplot(2, 2, ri * 2 + 1), f"01_vit_vs_resnet/vit_{s}", f"{s}: ViT encoder", ri * 2 + 0)
    classmap(figA.add_subplot(2, 2, ri * 2 + 2), f"02_resnet_depth/{s}_L1", f"{s}: ResNet (shallow)", ri * 2 + 1)
figA.suptitle("ViT versus convolutional encoder (matched shallow depth). The ViT encoder collapses to a few\n"
              "classes; the convolutional encoder recovers the full domain structure.", fontsize=10)
figA.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(figA)
pA = os.path.join(OUT, "fig_ablation_vit_vs_resnet.png"); figA.savefig(pA, dpi=170, facecolor="white")
import shutil; shutil.copy(pA, os.path.join(REVIEW, "fig_ablation_vit_vs_resnet.png"))

# ---------- Fig B: ResNet depth L1..L4 ----------
figB = Figure(figsize=(12.5, 6.4), facecolor="white")
for ri, s in enumerate(["Na007b", "IMC_SI5"]):
    for ci, L in enumerate([1, 2, 3, 4]):
        classmap(figB.add_subplot(2, 4, ri * 4 + ci + 1), f"02_resnet_depth/{s}_L{L}", f"{s}: depth L{L}", ri * 4 + ci)
figB.suptitle("Convolutional encoder depth (L1 to L4). Shallow encoders (L1, L2) retain the domain structure,\n"
              "while deeper encoders (L3, L4) collapse to fewer, coarser classes; a shallow trunk is therefore used.", fontsize=10)
figB.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(figB)
pB = os.path.join(OUT, "fig_ablation_depth.png"); figB.savefig(pB, dpi=170, facecolor="white")
shutil.copy(pB, os.path.join(REVIEW, "fig_ablation_depth.png"))
print("wrote fig_ablation_vit_vs_resnet.png and fig_ablation_depth.png", flush=True)
