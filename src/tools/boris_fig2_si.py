"""Boris Fig 2 SI: NaPHI (Na007b) -- show ALL classes for DINO and for NMF, and
what each class represents (its average diffraction). One cube pass accumulates a
class-average pattern for every DINO class and every NMF (K=6, polar+theta-shift)
cluster -- identical clustering to main Fig 2. Two SI panels written to BorisEdits:
  fig2si_dino_classes.png  -- DINO class map (all classes) + per-class avg diffraction
  fig2si_nmf_classes.png   -- NMF  class map (all classes) + per-class avg diffraction
  python tools/boris_fig2_si.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib as mpl

F = "docs/explainer/figs"; OUT = "docs/paper/draft_v2/figs/BorisEdits"
CUBE = "D:/DINOSR/data/Na007b_nbed.cube.npy"; RUN = "runs/_gui/Na007b_k60_m097_vmax2"
Ny, Nx = 126, 100; KNMF = 6; PXNM = 16.0
os.makedirs(OUT, exist_ok=True)

asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int).reshape(Ny, Nx)
flake = (asg != 0)
W = np.load(f"{F}/na007b_polar_W.npy")
nmf = KMeans(KNMF, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(W)).reshape(Ny, Nx)
BF = np.load(f"{F}/na007b_HAADF_gui.npy")

# ---- one cube pass: average diffraction per DINO class and per NMF cluster ----
cube = open_lazy_cube(CUBE, scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
dino_ids = sorted(np.unique(asg).tolist())
dsum = {c: np.zeros((H, Wd)) for c in dino_ids}; dcnt = {c: 0 for c in dino_ids}
nsum = {c: np.zeros((H, Wd)) for c in range(KNMF)}; ncnt = {c: 0 for c in range(KNMF)}
print("cube pass...", flush=True); t0 = time.time()
for rx in range(Ny):
    blk = np.asarray(cube[rx], np.float32)
    for ry in range(Nx):
        pat = blk[ry]
        dsum[int(asg[rx, ry])] += pat; dcnt[int(asg[rx, ry])] += 1
        if flake[rx, ry]:
            nsum[int(nmf[rx, ry])] += pat; ncnt[int(nmf[rx, ry])] += 1
    if rx % 40 == 0: print(f"  row {rx} ({time.time()-t0:.0f}s)", flush=True)
davg = {c: dsum[c] / max(dcnt[c], 1) for c in dino_ids}
navg = {c: nsum[c] / max(ncnt[c], 1) for c in range(KNMF)}


def diff(ax, m, ec):
    a = np.log1p(np.clip(m, 0, None)); h = a.shape[0]; a = a[h // 2 - 130:h // 2 + 130, h // 2 - 130:h // 2 + 130]
    o = a[a > 0]
    ax.imshow(a, cmap="inferno", vmax=(np.percentile(o, 99.5) if o.size else 1), interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(ec); s.set_linewidth(3)


def scalebar(ax, dark=True):
    sb = 500.0 / PXNM; c = "black" if dark else "white"
    ax.add_patch(Rectangle((4, Ny - 7), sb, 2.2, color=c, ec="black", lw=0.4))
    ax.text(4 + sb / 2, Ny - 9, "500 nm", color=c, ha="center", va="bottom", fontsize=7)


def panel(name, lab2d, ids, avg, cnt, fname, descr):
    base = mpl.colormaps.get_cmap("tab20").resampled(max(len(ids), 1))
    col = {c: base(i)[:3] for i, c in enumerate(ids)}
    # class map (every class its colour; off-flake stays light grey for DINO class 0)
    img = np.ones((Ny, Nx, 3), np.float32) * 0.93
    for c in ids:
        if name == "DINO" and c == 0:  # class 0 = vacuum / off-flake background
            continue
        img[lab2d == c] = col[c]
    ncols = 6; nrows = int(np.ceil(len(ids) / ncols))
    fig = Figure(figsize=(2.6 + 1.55 * ncols, max(3.0, 1.7 * nrows + 0.9)), facecolor="white")
    gs = fig.add_gridspec(nrows, ncols + 2, wspace=0.12, hspace=0.32)
    axm = fig.add_subplot(gs[:, 0:2]); axm.imshow(img, aspect="equal", interpolation="nearest")
    axm.set_xticks([]); axm.set_yticks([]); scalebar(axm, dark=True)
    axm.set_title(f"{name} class map (all {len([c for c in ids if not (name=='DINO' and c==0)])} classes)", fontsize=10, fontweight="bold")
    for i, c in enumerate(ids):
        ax = fig.add_subplot(gs[i // ncols, 2 + i % ncols]); diff(ax, avg[c], col[c])
        frac = 100.0 * cnt[c] / (Ny * Nx)
        lbl = f"class {c}" + ("  (vacuum)" if name == "DINO" and c == 0 else "")
        ax.set_title(f"{lbl}\n{frac:.0f}% of scan", fontsize=8)
    fig.suptitle(f"NaPHI (Na007b): every {name} class and its average diffraction pattern.  {descr}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
    p = os.path.join(OUT, fname); fig.savefig(p, dpi=165, facecolor="white"); print(f"wrote {p}", flush=True)


panel("DINO", asg, dino_ids, davg, dcnt, "fig2si_dino_classes.png",
      "Class colours match the main-text map; the Line classes are the ones whose average shows the extra sharp reflections.")
panel("NMF", nmf, list(range(KNMF)), navg, ncnt, "fig2si_nmf_classes.png",
      "NMF (polar+theta-shift, K=6) splits the flake mainly by peak-vs-halo weighting rather than into the same continuum of states.")
print("DONE", flush=True)
