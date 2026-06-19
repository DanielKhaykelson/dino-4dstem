"""Quick view: per-pixel ACOM corr, n_peaks, peak/halo ratio maps for SI3/SI4
(+ DINO class map) from imc_acom_fullpx_{name}.npz. Vacuum-masked by halo<0.02."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl
OUT = "docs/paper/draft_v2/figs"
RUNS = {"SI3": "runs/_gui/IMC_SI3_m097k60", "SI4": "runs/_gui/IMC_SI4_m097_k60"}
fig = Figure(figsize=(15, 7.5), facecolor="white")
for ri, name in enumerate(RUNS):
    z = np.load(os.path.join(OUT, f"imc_acom_fullpx_{name}.npz"))
    Ny, Nx = z["scan"]; corr = z["corr"]; npk = z["n_peaks"].astype(float)
    ratio = z["ratio"]; halo = z["halo"]
    samp = halo >= 0.02
    asg = np.load(os.path.join(RUNS[name], "eval", "inference.npz"))["assigns"].astype(int)
    def M(v):
        m = np.full(Ny * Nx, np.nan); m[samp] = v[samp]; return m.reshape(Ny, Nx)
    cols = [("tab20", asg.reshape(Ny, Nx).astype(float), "DINO classes", False),
            ("inferno", M(npk), f"n_peaks (med {np.median(npk[samp]):.0f})", True),
            ("viridis", M(np.clip(ratio, 0, np.nanpercentile(ratio[samp], 98))), "peak/halo ratio", True),
            ("magma", M(np.where(corr > 0, corr, np.nan)), f"ACOM corr ({100*np.mean(corr[samp]>0):.0f}% idx)", True)]
    for ci, (cm, mp, t, mask) in enumerate(cols):
        ax = fig.add_subplot(2, 4, ri * 4 + ci + 1)
        cmap = mpl.cm.get_cmap(cm).copy()
        if mask: cmap.set_bad("#222")
        im = ax.imshow(mp, cmap=cmap, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0: ax.set_ylabel(name, fontsize=12, fontweight="bold")
        if ri == 0: ax.set_title(t, fontsize=9)
        else: ax.set_title(t, fontsize=9)
        if mask: fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Per-pixel stride=1 ACOM + crystallinity, SI3/SI4 (vacuum halo<0.02 masked) — note n_peaks~190 = shot-noise dominated", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "fullpx_maps_SI34.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "fullpx_maps_SI34.png"))
print("wrote fullpx_maps_SI34.png", flush=True)
