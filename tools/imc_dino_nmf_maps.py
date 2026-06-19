"""Main-text Fig 3: the three IMC fields of view (SI3/SI4/SI5), each as the DINO
class map next to the matched classical polar-NMF class map. Shows side by side
that DINO yields coherent domains while NMF (clustered to the same K) gives a
different, fragmented partition -- the maps themselves, no ARI bars (those move to
SI). Discrete colours per map (tab20).

  python tools/imc_dino_nmf_maps.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
import matplotlib as mpl
try:
    from sklearn.metrics import adjusted_rand_score as ARI
except Exception:
    ARI = None

OUT = "docs/explainer/figs"; REVIEW = "docs/paper/draft_v2/figs/latest_review"; NY = NX = 128
RUN = {"SI3": "runs/_gui/IMC_SI3_m097k60", "SI4": "runs/_gui/IMC_SI4_m097_k60",
       "SI5": "runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60"}
NAMES = ["SI3", "SI4", "SI5"]
ROLE = {"SI3": "overview (matrix + needles)", "SI4": "needles", "SI5": "needle / matrix interface"}


def discrete(ax, lab2d, title):
    uni = sorted(np.unique(lab2d).tolist()); lut = {u: k for k, u in enumerate(uni)}
    cmap = ListedColormap([mpl.colormaps.get_cmap("tab20").resampled(max(len(uni), 1))(k) for k in range(len(uni))])
    ax.imshow(np.vectorize(lut.get)(lab2d).astype(float), cmap=cmap, interpolation="nearest",
              vmin=0, vmax=max(len(uni) - 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)


fig = Figure(figsize=(7.2, 10.0), facecolor="white")
for ri, n in enumerate(NAMES):
    dino = np.load(os.path.join(RUN[n], "eval", "inference.npz"))["assigns"].astype(int).reshape(NY, NX)
    nmf = np.load(os.path.join(OUT, f"polarnmf_labels_IMC_{n}.npy")).astype(int)
    K = len(np.unique(dino))
    ari = ARI(dino.ravel(), nmf.ravel()) if ARI else float("nan")
    axd = fig.add_subplot(3, 2, ri * 2 + 1); discrete(axd, dino, "DINO" if ri == 0 else "")
    axd.set_ylabel(f"{n}\n{ROLE[n]}", fontsize=10, fontweight="bold")
    axn = fig.add_subplot(3, 2, ri * 2 + 2); discrete(axn, nmf, "classical polar-NMF" if ri == 0 else "")
    axn.text(0.5, -0.06, f"ARI vs DINO = {ari:.2f}", transform=axn.transAxes, ha="center",
             va="top", fontsize=8, color="#C0392B")
fig.suptitle("IMC: self-supervised (DINO) vs classical polar-NMF class maps, matched class count K, for the three fields of view.\n"
             "DINO yields coherent crystallization-state domains; NMF (clustered to the same K) gives a different, fragmented partition (low ARI; per-algorithm spread in Fig. S11).",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_dino_nmf_maps.png"); fig.savefig(p, dpi=170, facecolor="white")
import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "imc_dino_nmf_maps.png"))
print("wrote imc_dino_nmf_maps.png", flush=True)
