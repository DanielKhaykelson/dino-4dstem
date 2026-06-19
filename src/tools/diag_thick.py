"""Diagnose the thick-amorphous false-peak hypothesis. From grain_acom_v2_{name}.npz:
 row1: scatter n_peaks vs grain thickness (scat), color = crystallinity ratio.
       If n_peaks rises with thickness even at ratio~0 -> thick amorphous makes
       false peaks. If high-npk points are all high-ratio -> n_peaks = crystallinity.
 row2: grain-avg pattern (log) of the 3 THICKEST grains with ratio<0.15 (thick
       'amorphous') with their blob count + ACOM corr — are they smooth halo (good)
       or speckle that got peaks (bad)?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
OUT = "docs/paper/draft_v2/figs"; NAMES = ["SI3", "SI4", "SI5"]
fig = Figure(figsize=(15, 8.5), facecolor="white")
for ri, name in enumerate(NAMES):
    z = np.load(os.path.join(OUT, f"grain_acom_v2_{name}.npz"))
    npk = z["npk"]; ratio = z["ratio"]; gscat = z["gscat"]; vac = z["vac"]; corr = z["corr"]
    gsum = z["gsum"]; gcnt = z["gcnt"]; H = int(z["H"]); cyx = (H - 1) / 2.0
    m = ~vac
    ax = fig.add_subplot(2, 3, ri + 1)
    sc = ax.scatter(gscat[m], npk[m], c=np.clip(ratio[m], 0, 2), cmap="viridis", s=28, edgecolor="k", lw=0.3)
    ax.set_xlabel("grain thickness (median scat)"); ax.set_ylabel("n_peaks")
    ax.set_title(f"{name}: n_peaks vs thickness (color=crystallinity)", fontsize=9)
    fig.colorbar(sc, ax=ax, fraction=0.046)
    # thickest low-crystallinity grains
    cand = np.where(m & (ratio < 0.15))[0]
    cand = cand[np.argsort(-gscat[cand])][:3]
    for j, g in enumerate(cand):
        ax = fig.add_subplot(2, 9, 9 + ri * 3 + j + 1)
        avg = gsum[g] / max(gcnt[g], 1)
        cr = slice(int(cyx) - 130, int(cyx) + 130)
        ax.imshow(np.log1p(np.clip(avg[cr, cr], 0, None)), cmap="inferno")
        ax.set_title(f"{name} thick-amorph\nscat={gscat[g]:.0e} npk={int(npk[g])}\nratio={ratio[g]:.2f} corr={corr[g]:.0f}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    print(f"[{name}] thick-amorph grains (ratio<.15) npk: {[int(npk[g]) for g in cand]} "
          f"corr: {[round(float(corr[g]),1) for g in cand]}", flush=True)
fig.suptitle("Thick-amorphous false-peak test: does n_peaks track THICKNESS (bad) or CRYSTALLINITY (good)? + thickest amorphous grain patterns", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "diag_thick.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "diag_thick.png"))
print("wrote diag_thick.png", flush=True)
