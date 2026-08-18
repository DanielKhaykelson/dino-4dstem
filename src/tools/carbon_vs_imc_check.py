"""Can we tell amorphous CARBON support from IMC, diffraction-only?
Discriminator: amorphous carbon's main halo is d~2.1A (~257 px, OFF our frame), so
bare carbon is ~featureless in-window; a clear d~4.5A ring is an organic (IMC)
signature. We (1) overlay radial profiles of the THICKEST (high-scatter) vs
THINNEST (low-scatter) SI3 grains to see who shows the 4.5A ring, and (2) report,
for the chosen matrix grain (SI3 g54): its ring d-spacing, scatter percentile, and
whether it is spatially ADJACENT to crystalline (high-spottiness) needle grains
(which are unambiguously IMC) -- a spatial argument that the matrix is IMC too.

  python tools/carbon_vs_imc_check.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")
INV = 0.00185; KMAX = 0.35; FOV = {"SI3": 187}; NY = NX = 128
name = "SI3"
z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
gid, gsum, gcnt, vac, gscat = z["gid"], z["gsum"], z["gcnt"], z["vac"], z["gscat"]; H = int(z["H"])
cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV[name])
rr = np.arange(lo, hi); dd = 1.0 / (rr * INV)

rows = {}
for g in range(gsum.shape[0]):
    if vac[g] or gcnt[g] < 40:
        continue
    avg = gsum[g] / max(gcnt[g], 1)
    m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
    seg = m[lo:hi]; vseg = v[lo:hi]
    if seg.size < 5 or seg.sum() <= 0:
        continue
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); pk = np.clip(seg - halo, 0, None)
    cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None)
    rstar = lo + int(np.argmax(pk))
    # halo-ring prominence at 4.5A band (105-130 px)
    band = (rr >= 105) & (rr <= 130)
    bump = float((pk[band] / np.clip(halo[band], 1e-9, None)).max())
    rows[g] = dict(seg=seg, prof=seg / (seg.mean() + 1e-9), spot=float(np.percentile(cv, 90)),
                   gscat=float(gscat[g]), n=int(gcnt[g]), rstar=rstar, d=1.0/(rstar*INV), bump=bump)

gl = list(rows)
scats = np.array([rows[g]["gscat"] for g in gl])
thick = [gl[i] for i in np.argsort(-scats)[:4]]
thin = [gl[i] for i in np.argsort(scats)[:4]]

# spatial adjacency: which grains touch grain 54, and are they crystalline?
gidmap = gid.reshape(NY, NX)
def neighbors(g):
    foot = gidmap == g
    dil = ndimage.binary_dilation(foot, iterations=2) & ~foot
    touch = set(int(x) for x in np.unique(gidmap[dil]) if x >= 0 and x != g and x in rows)
    return touch
g54 = 54 if 54 in rows else thick[0]
touch = neighbors(g54)
print(f"[SI3] matrix grain g={g54}: ring d={rows[g54]['d']:.2f}A  4.5A-bump={rows[g54]['bump']:.2f}  "
      f"scat={rows[g54]['gscat']:.0f} (pctile {100*np.mean(scats<rows[g54]['gscat']):.0f}%)  n={rows[g54]['n']}", flush=True)
print(f"   touching grains spottiness: {sorted([round(rows[t]['spot'],2) for t in touch], reverse=True)}", flush=True)
print(f"   -> adjacent to crystalline needle (spot>1.0)? {'YES' if any(rows[t]['spot']>1.0 for t in touch) else 'no'}", flush=True)
print("thick grains: d/bump/scat =", [(round(rows[g]['d'],2), round(rows[g]['bump'],2), int(rows[g]['gscat'])) for g in thick], flush=True)
print("thin  grains: d/bump/scat =", [(round(rows[g]['d'],2), round(rows[g]['bump'],2), int(rows[g]['gscat'])) for g in thin], flush=True)

fig = Figure(figsize=(12, 4.6), facecolor="white")
ax = fig.add_subplot(1, 2, 1)
for g in thick:
    ax.plot(dd, rows[g]["prof"], color="#1F618D", lw=1.4, alpha=0.85)
for g in thin:
    ax.plot(dd, rows[g]["prof"], color="#C0392B", lw=1.4, alpha=0.85)
ax.axvspan(4.2, 4.8, color="#A9DFBF", alpha=0.35); ax.text(4.5, ax.get_ylim()[1]*0.97, " IMC organic halo ~4.5Å", fontsize=8, color="#1E7A34", va="top", ha="center")
ax.axvline(3.4, color="#E08A1E", ls="--", lw=1.2); ax.text(3.4, ax.get_ylim()[1]*0.78, " a-C (002) ~3.4Å\n(in-frame)", fontsize=7, color="#B5650F", va="top")
ax.axvline(2.1, color="#999", ls=":", lw=1); ax.text(2.1, ax.get_ylim()[1]*0.6, " a-C ~2.1Å\n(off-frame)", fontsize=7, color="#777", va="top")
ax.invert_xaxis(); ax.set_xlabel("d-spacing (Å)"); ax.set_ylabel("norm. radial intensity")
ax.set_title("interface: thick/high-scatter grains (blue) vs thinnest grains (red)\nIMC ring sits at 4.3-4.7Å, distinct from a-C (002) at 3.4Å", fontsize=9)
ax.plot([], [], color="#1F618D", label="thick (high scatter)"); ax.plot([], [], color="#C0392B", label="thin (low scatter)"); ax.legend(fontsize=8)
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(dd, rows[g54]["prof"], color="#1F618D", lw=2, label=f"matrix grain g{g54}")
ax2.axvspan(4.2, 4.8, color="#A9DFBF", alpha=0.35); ax2.axvline(3.4, color="#E08A1E", ls="--", lw=1.2)
ax2.text(3.4, ax2.get_ylim()[1]*0.5, " a-C(002) 3.4Å", fontsize=7, color="#B5650F")
ax2.invert_xaxis(); ax2.set_xlabel("d-spacing (Å)"); ax2.set_ylabel("norm. radial intensity")
ax2.set_title(f"chosen matrix grain g{g54}: ring d={rows[g54]['d']:.1f}Å, 4.5Å-bump={rows[g54]['bump']:.2f}\n"
              f"scatter pctile {100*np.mean(scats<rows[g54]['gscat']):.0f}%, adjacent to needle: "
              f"{'yes' if any(rows[t]['spot']>1.0 for t in touch) else 'no'}", fontsize=9); ax2.legend(fontsize=8)
fig.tight_layout(); FigureCanvasAgg(fig)
p = os.path.join(OUT, "carbon_vs_imc_check.png"); fig.savefig(p, dpi=160, facecolor="white")
import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "carbon_vs_imc_check.png"))
print("wrote carbon_vs_imc_check.png", flush=True)
