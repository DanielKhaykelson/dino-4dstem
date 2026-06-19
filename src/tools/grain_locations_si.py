"""SI figure: where the three representative diffraction patterns in Fig. 5 come
from. For each (less-ordered matrix, crystallization front: SI5; highly-ordered
needle: SI4) we mark the selected grain's footprint on the scan (HAADF-like
scattered-intensity map) and show its grain-average diffraction pattern beneath,
so each Fig. 5 pattern is traceable to a location. Selection matches Fig. 5
(extreme-spottiness tail, largest grain).

  python tools/grain_locations_si.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")
INV = 0.00185; KMAX = 0.35; FOV = {"SI3": 187, "SI4": 160, "SI5": 160}; NY = NX = 128


def grain_table(name):
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    gid, gsum, gcnt, vac, gscat = z["gid"], z["gsum"], z["gcnt"], z["vac"], z["gscat"]; H = int(z["H"])
    cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV[name])
    rows = []
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
        rows.append(dict(g=g, spot=float(np.percentile(cv, 90)), n=int(gcnt[g]),
                         gscat=float(gscat[g]), avg=avg, cyx=cyx, beam=beam, rstar=lo + int(np.argmax(pk))))
    return gid, rows


def pick(rows, lowest, frac=0.35, min_n=60):
    """On-sample (above-median scatter), so we never pick a thin hole/carbon region."""
    med = np.median([d["gscat"] for d in rows])
    cand = [d for d in rows if d["gscat"] >= med and d["n"] >= min_n] or [d for d in rows if d["gscat"] >= med] or rows
    s = sorted(cand, key=lambda d: d["spot"]); k = max(1, int(len(s) * frac))
    return max(s[:k] if lowest else s[-k:], key=lambda d: d["n"])


gid3, r3 = grain_table("SI3"); gid5, r5 = grain_table("SI5"); gid4, r4 = grain_table("SI4")
SEL = [("less-ordered matrix grain (SI3)", "SI3", gid3, pick(r3, True)),
       ("crystallization-front grain (SI5)", "SI5", gid5, pick(r5, False)),
       ("highly-ordered needle grain (SI4)", "SI4", gid4, pick(r4, False))]
COL = ["#39FF14", "#FFD24D", "#FF3B3B"]


def scatmap(name):
    z = np.load(os.path.join(FIGS, f"imc_glassorder_{name}.npz"))
    return np.log1p(np.clip(z["scat"].reshape(NY, NX), 0, None))


fig = Figure(figsize=(11, 7.6), facecolor="white")
for ci, (label, name, gid, sel) in enumerate(SEL):
    # location map
    ax = fig.add_subplot(2, 3, ci + 1); ax.set_xticks([]); ax.set_yticks([])
    bg = scatmap(name); ax.imshow(bg, cmap="gray", interpolation="nearest")
    foot = (gid == sel["g"]).reshape(NY, NX).astype(float)
    ov = np.zeros((NY, NX, 4)); rgb = tuple(int(COL[ci][i:i+2], 16) / 255 for i in (1, 3, 5))
    ov[foot > 0] = (*rgb, 0.95)
    ax.imshow(ov, interpolation="nearest")
    ys, xs = np.where(foot > 0)
    if len(xs):
        ax.add_patch(Circle((xs.mean(), ys.mean()), 9, fill=False, ec=COL[ci], lw=1.6))
    ax.set_title(label, fontsize=10, fontweight="bold", color="#21295C")
    ax.set_xlabel(f"{sel['n']} px   spottiness {sel['spot']:.1f}", fontsize=8.5)
    # diffraction pattern
    cyx = sel["cyx"]; cr = slice(int(cyx) - 145, int(cyx) + 145)
    p = sel["avg"][cr, cr].astype(np.float32); pc = (p.shape[0] - 1) / 2.0
    yy, xx = np.indices(p.shape); mask = np.sqrt((yy - pc) ** 2 + (xx - pc) ** 2) > sel["beam"]
    loq, hq = np.percentile(p[mask], 2), np.percentile(p[mask], 99.5)
    d = np.log1p(np.clip(p, loq, hq) * mask - loq)
    axd = fig.add_subplot(2, 3, ci + 4); axd.imshow(d, cmap="inferno"); axd.set_xticks([]); axd.set_yticks([])
    axd.add_patch(Circle((pc, pc), sel["rstar"], fill=False, ec="#39FF14", lw=1.1, ls=(0, (5, 3))))
    axd.set_title("its grain-average diffraction", fontsize=9)
    for s in axd.spines.values():
        s.set_edgecolor(COL[ci]); s.set_linewidth(2)
fig.suptitle("Provenance of the Fig. 5 diffraction patterns: location of each selected grain on the scan (top, scattered-intensity / HAADF-like map; "
             "coloured footprint) and its grain-average diffraction (bottom). Same grains and ordering as Fig. 5.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "grain_locations_si.png"); fig.savefig(p, dpi=170, facecolor="white")
import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "grain_locations_si.png"))
print(f"wrote grain_locations_si.png  grains: matrix g={SEL[0][3]['g']} front g={SEL[1][3]['g']} needle g={SEL[2][3]['g']}", flush=True)
