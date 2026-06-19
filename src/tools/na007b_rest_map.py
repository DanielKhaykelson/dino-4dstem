"""Na007b non-Line (rest) cluster MAPS: one image per method, all rest clusters
in DISTINCT colours overlaid on HAADF (GUI gray). DINO uses the existing trained
model's inference (runs/_gui/Na007b_k60_m097_vmax2/eval/inference.npz) — no
retraining. NMF = polar+theta-shift loadings (K=6). No cube pass needed."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.cm as cmx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch

F = r"docs/explainer/figs"; RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
Ny, Nx = 126, 100
asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int).reshape(Ny, Nx)
flake = (asg != 0); dino_line_mask = np.isin(asg, [1, 8]) & flake
ph = np.load(f"{F}/na007b_cryst.npz.npy")[0].reshape(Ny, Nx)
HA = np.load(f"{F}/na007b_HAADF_gui.npy")
W = np.load(f"{F}/na007b_polar_W.npy"); KNMF = 6
nmf = KMeans(KNMF, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(W)).reshape(Ny, Nx)
ov = [((nmf == c) & dino_line_mask).sum() / max((nmf == c).sum(), 1) for c in range(KNMF)]
nmf_line_c = int(np.argmax(ov))

dino_rest = [c for c in sorted(np.unique(asg[flake])) if c not in (0, 1, 8) and ((asg == c) & flake).sum() > 30]
nmf_rest = [c for c in range(KNMF) if c != nmf_line_c and ((nmf == c) & flake).sum() > 30]
lo, hi = np.percentile(HA, 1), np.percentile(HA, 99); bg = np.clip(HA, lo, hi)
COLS = cmx.get_cmap("tab10").colors + cmx.get_cmap("Set2").colors   # 18 distinct colours

def panel(ax, masks, prefix, title):
    ax.imshow(bg, cmap="gray", aspect="equal", interpolation="nearest")
    o = np.zeros((Ny, Nx, 4), np.float32); handles = []
    for i, (c, m) in enumerate(masks):
        col = COLS[i % len(COLS)]
        o[m, :3] = col[:3]; o[m, 3] = 0.7
        handles.append(Patch(facecolor=col, edgecolor="white", lw=0.4,
                             label=f"{prefix}{c}  n={int(m.sum())}  p/h={np.nanmedian(ph[m]):.2f}"))
    ax.imshow(o, aspect="equal", interpolation="nearest"); ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85, handlelength=1.0)

dmasks = [(c, (asg == c) & flake) for c in dino_rest]
nmasks = [(c, (nmf == c) & flake & ~dino_line_mask) for c in nmf_rest]
fig = Figure(figsize=(10, 6.2), facecolor="white")
panel(fig.add_subplot(1, 2, 1), dmasks, "D", f"DINO non-Line clusters ({len(dmasks)})")
panel(fig.add_subplot(1, 2, 2), nmasks, "N", f"NMF (polar+θ-shift) non-Line clusters ({len(nmasks)})")
fig.suptitle("Na007b non-Line (rest) clusters on HAADF — distinct colour per cluster", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
fig.savefig(f"{F}/na007b_rest_clusters_haadf.png", dpi=160, facecolor="white")
print(f"DINO rest={dino_rest}; NMF rest={nmf_rest}; wrote na007b_rest_clusters_haadf.png", flush=True)
