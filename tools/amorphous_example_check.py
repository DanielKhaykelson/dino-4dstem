"""Find a genuine less-ordered (amorphous / lightly crystalline) IMC example that
is ON the sample, not a thin spot / hole / carbon near a needle. The earlier Fig.5
'matrix' pick used the lowest-spottiness grain, which biases toward LOW-SCATTER
regions (holes). Here we require decent scattered intensity (on-sample) and then
take the least-spotty grains, preferring SI3. Also reports whether even the
least-spotty on-sample grain still has appreciable crystallinity (i.e. the blob is
already partly crystalline).

  python tools/amorphous_example_check.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")
INV = 0.00185; KMAX = 0.35; FOV = {"SI3": 187, "SI4": 160, "SI5": 160}; NY = NX = 128


def table(name):
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
        yy, xx = np.indices(avg.shape); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
        hf = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1]); band = (rr >= lo) & (rr <= hi)
        rows.append(dict(g=g, spot=float(np.percentile(cv, 90)),
                         B=float(np.clip(avg[band] - hf[band], 0, None).sum() / (hf[band].sum() + 1e-9)),
                         gscat=float(gscat[g]), n=int(gcnt[g]), avg=avg, cyx=cyx, beam=beam, rstar=lo + int(np.argmax(pk))))
    return gid, rows, np.median([r["gscat"] for r in rows])


def disp(avg, cyx, beam):
    cr = slice(int(cyx) - 145, int(cyx) + 145); p = avg[cr, cr].astype(np.float32); pc = (p.shape[0] - 1) / 2.0
    yy, xx = np.indices(p.shape); mask = np.sqrt((yy - pc) ** 2 + (xx - pc) ** 2) > beam
    loq, hq = np.percentile(p[mask], 2), np.percentile(p[mask], 99.5)
    return np.log1p(np.clip(p, loq, hq) * mask - loq), pc


# diagnostics: was the old pick a low-scatter outlier?
for name in ["SI3", "SI5"]:
    gid, rows, medscat = table(name)
    onsample = [r for r in rows if r["gscat"] >= medscat]            # require above-median thickness
    least = min(onsample, key=lambda r: r["spot"])
    s = sorted(rows, key=lambda r: r["spot"]); k = max(1, int(len(s) * 0.35))
    oldpick = max(s[:k], key=lambda r: r["n"])                       # replicate old Fig.5 'matrix' pick
    pct = 100 * np.mean([r["gscat"] < oldpick["gscat"] for r in rows])
    print(f"[{name}] grains={len(rows)} medscat={medscat:.0f} | OLD matrix pick g={oldpick['g']} "
          f"spot={oldpick['spot']:.2f} gscat={oldpick['gscat']:.0f} (scatter pctile {pct:.0f}%)  "
          f"| least-spotty ON-SAMPLE g={least['g']} spot={least['spot']:.2f} B={least['B']:.2f} gscat={least['gscat']:.0f} n={least['n']}", flush=True)

# figure: top 4 least-spotty ON-SAMPLE SI3 grains (genuine less-ordered IMC) + locations
name = "SI3"; gid, rows, medscat = table(name)
cand = sorted([r for r in rows if r["gscat"] >= medscat and r["n"] >= 60], key=lambda r: r["spot"])[:4]
scat = np.load(os.path.join(FIGS, f"imc_glassorder_{name}.npz"))["scat"].reshape(NY, NX)
bg = np.log1p(np.clip(scat, 0, None))
fig = Figure(figsize=(13, 6.6), facecolor="white")
for ci, r in enumerate(cand):
    d, pc = disp(r["avg"], r["cyx"], r["beam"])
    ax = fig.add_subplot(2, 4, ci + 1); ax.imshow(d, cmap="inferno"); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Circle((pc, pc), r["rstar"], fill=False, ec="#39FF14", lw=1.0, ls=(0, (5, 3))))
    ax.set_title(f"SI3 grain {r['g']}\nspot={r['spot']:.2f}  B={r['B']:.2f}", fontsize=9)
    ax.set_xlabel(f"thickness(scat)={r['gscat']:.0f}  {r['n']}px", fontsize=8)
    axm = fig.add_subplot(2, 4, ci + 5); axm.imshow(bg, cmap="gray"); axm.set_xticks([]); axm.set_yticks([])
    foot = (gid == r["g"]).reshape(NY, NX)
    ov = np.zeros((NY, NX, 4)); ov[foot] = (1, 0.1, 0.1, 0.95); axm.imshow(ov)
    ys, xs = np.where(foot)
    if len(xs): axm.add_patch(Circle((xs.mean(), ys.mean()), 9, fill=False, ec="#FF3B3B", lw=1.5))
    axm.set_title("location on HAADF-like map", fontsize=8)
fig.suptitle("Genuine less-ordered IMC examples (SI3): least-spotty grains that are ON the sample (above-median scattered intensity), "
             "not thin holes/carbon. Smooth halo + decent thickness = amorphous/lightly-crystalline matrix.", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "amorphous_example_SI3.png"); fig.savefig(p, dpi=160, facecolor="white")
import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "amorphous_example_SI3.png"))
print("wrote amorphous_example_SI3.png", flush=True)
