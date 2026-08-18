"""Same as imc_class_boxplots.py but with a THIRD column: the radial peak/halo
ratio chi. Each box = distribution of that descriptor over the grains of one DINO
class; classes ordered by median; eta^2 annotated. Writes a separate figure
(imc_class_boxplots_3col.png) to latest_review + explainer.
  python src/tools/imc_class_boxplots_3col.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")
INV = 0.00185; KMAX = 0.35; FOV = {"SI3": 187, "SI4": 160, "SI5": 160}; NAMES = ["SI3", "SI4", "SI5"]
CLR = {"SI3": "#2E86C1", "SI4": "#28B463", "SI5": "#CA6F1E"}
DISP = {"SI3": "interface", "SI4": "needles", "SI5": "magnified interface"}


def eta2(v, lab):
    v = np.asarray(v, float); lab = np.asarray(lab); ok = np.isfinite(v); v, lab = v[ok], lab[ok]
    if v.size < 5: return np.nan
    gm = v.mean(); sst = ((v - gm) ** 2).sum()
    ssb = sum(((v[lab == c].mean() - gm) ** 2) * (lab == c).sum() for c in set(lab.tolist()))
    return float(ssb / (sst + 1e-12))


def per_grain(name):
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    cls, vac, gsum, gcnt = z["cls"], z["vac"], z["gsum"], z["gcnt"]
    H = int(z["H"]); cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1
    hi = min(int(KMAX / INV), FOV[name])
    spot, B, chi, lab = [], [], [], []
    for g in range(gsum.shape[0]):
        if vac[g]: continue
        avg = gsum[g] / max(gcnt[g], 1)
        m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
        seg = m[lo:hi]; vseg = v[lo:hi]
        if seg.size < 5 or seg.sum() <= 0: continue
        halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); pk = np.clip(seg - halo, 0, None)
        cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None)
        yy, xx = np.indices(avg.shape); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
        hf = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1]); band = (rr >= lo) & (rr <= hi)
        spot.append(float(np.percentile(cv, 90)))
        B.append(float(np.clip(avg[band] - hf[band], 0, None).sum() / (hf[band].sum() + 1e-9)))
        chi.append(float((pk / np.clip(halo, 1e-9, None)).max()))
        lab.append(int(cls[g]))
    return np.array(spot), np.array(B), np.array(chi), np.array(lab)


PARAMS = ["spottiness", "Bragg excess B", "peak/halo χ"]
fig = Figure(figsize=(18.5, 3.3 * 3), facecolor="white")
for ri, name in enumerate(NAMES):
    spot, B, chi, lab = per_grain(name)
    data = {"spottiness": spot, "Bragg excess B": B, "peak/halo χ": chi}
    for ci, pname in enumerate(PARAMS):
        v = data[pname]; e2 = eta2(v, lab)
        order = sorted(set(lab.tolist()), key=lambda c: np.median(v[lab == c]))
        groups = [v[lab == c] for c in order]
        ax = fig.add_subplot(3, 3, ri * 3 + ci + 1)
        bp = ax.boxplot(groups, showfliers=False, widths=0.62, patch_artist=True, medianprops=dict(color="k"))
        for patch in bp["boxes"]: patch.set_facecolor(CLR[name]); patch.set_alpha(0.55)
        for xi, gg in enumerate(groups):
            ax.scatter(np.full(len(gg), xi + 1) + np.linspace(-0.13, 0.13, len(gg)), gg,
                       s=9, color="#333", alpha=0.55, zorder=3, linewidths=0)
        ax.set_xticklabels([f"c{c}\n(n={len(v[lab==c])})" for c in order], fontsize=6.5)
        ax.set_ylabel(f"per-grain {pname}", fontsize=9)
        ax.set_title(f"{DISP[name]}: {pname} by DINO class (ordered)   η²={e2:.2f}", fontsize=9.5, color=CLR[name])
fig.tight_layout(); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_class_boxplots_3col.png"); fig.savefig(p, dpi=160, facecolor="white")
import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "imc_class_boxplots_3col.png"))
print("wrote imc_class_boxplots_3col.png", flush=True)
