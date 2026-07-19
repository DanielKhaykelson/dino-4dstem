"""50nm vs 150nm IMC: do the 50nm films index to the same alpha-IMC structure, and
do they reach the same point on the order axis? Overlays the most-ordered class-average
radial profile of each field on a q=1/d axis with the alpha.cif reflections (kinematical),
and prints the max class-median spottiness so 'needle reached?' is quantitative.
  python src/tools/imc_50_vs_150_compare.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
CIF = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/cifs/alpha.cif"
INV = 0.00185; QMIN, QMAX = 0.10, 0.33


def cif_refs(dmin=3.0, dmax=9.0):
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    s = Structure.from_file(CIF)
    pat = XRDCalculator(wavelength="CuKa").get_pattern(s, two_theta_range=(2, 40))
    out = []
    for d, i, h in zip(pat.d_hkls, pat.y, pat.hkls):
        if dmin <= d <= dmax:
            out.append((1.0 / d, i))
    mx = max(i for _, i in out)
    return [(q, i / mx) for q, i in out]


def most_ordered_profile(name):
    """q, normalized class-average radial profile of the most-ordered class, and that
    class's median PER-GRAIN azimuthal spottiness (the meaningful 'needle-reached' metric)."""
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    cls, vac, gsum, gcnt, gscat = z["cls"], z["vac"], z["gsum"], z["gcnt"], z["gscat"]
    H = int(z["H"]); cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1
    hi = min(int(0.35 / INV), 160, H // 2 - 2)

    def grain_spot(g):
        avg = gsum[g] / max(gcnt[g], 1); m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
        seg = m[lo:hi]; vs = v[lo:hi]
        if seg.size < 5 or seg.sum() <= 0:
            return np.nan
        return float(np.percentile(np.sqrt(np.clip(vs, 0, None)) / np.clip(seg, 1e-9, None), 90))

    med = np.median(gscat[~vac]); cmed = {}
    for c in sorted(set(cls[~vac].tolist())):
        idx = [g for g in range(gsum.shape[0]) if cls[g] == c and not vac[g] and gscat[g] >= med]
        if len(idx) < 2:
            continue
        sps = [grain_spot(g) for g in idx]; sps = [s for s in sps if np.isfinite(s)]
        if sps:
            cmed[c] = (float(np.median(sps)), idx)
    cbest = max(cmed, key=lambda c: cmed[c][0]); med_spot, idx = cmed[cbest]
    avg = sum(gsum[g] for g in idx) / max(sum(gcnt[g] for g in idx), 1)
    m, _, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
    r = np.arange(lo, hi); q = r * INV; prof = m[lo:hi]
    return q, prof / np.nanmax(prof), med_spot


refs = cif_refs()
SERIES = [
    ("150 nm needle (SI4)", "SI4", "#01665e", 2.4),
    ("50 nm SI-005",        "50nm_SI5", "#d6604d", 1.8),
    ("50 nm SI-001",        "50nm_SI1", "#f4a582", 1.6),
    ("50 nm SI-002",        "50nm_SI2", "#92c5de", 1.6),
    ("50 nm SI-003",        "50nm_SI3", "#4393c3", 1.6),
]
fig = Figure(figsize=(9.4, 5.2), facecolor="white"); ax = fig.add_subplot(111)
for q0, i0 in refs:
    if QMIN <= q0 <= QMAX:
        ax.vlines(q0, 0, 0.10 + 0.30 * i0, color="0.6", lw=0.8 + 1.6 * i0, alpha=0.5, zorder=0)
print("most-ordered class spottiness (needle reached if > ~4):")
for lab, name, col, lw in SERIES:
    q, prof, cv = most_ordered_profile(name)
    ax.plot(q, prof, lw=lw, color=col, label=f"{lab}  (needle-class spottiness {cv:.1f})")
    print(f"  {lab}: {cv:.2f}")
ax.set_xlim(QMIN, QMAX); ax.set_ylim(0, 1.25)
ax.set_xlabel("q = 1/d  (Å⁻¹)"); ax.set_ylabel("most-ordered class profile (norm.)")
ax.set_title("50 nm vs 150 nm indomethacin: same α reflections, comparable maximum order (field-to-field variation)",
             fontsize=10)
secax = ax.secondary_xaxis("top", functions=(lambda q: 1.0 / np.clip(q, 1e-6, None),
                                             lambda d: 1.0 / np.clip(d, 1e-6, None)))
secax.set_xlabel("d-spacing (Å)")
ax.legend(fontsize=8, loc="upper right")
ax.text(0.01, 0.97, "grey sticks: α-IMC reflections from alpha.cif (kinematical); calibration 0.00185 Å⁻¹/px",
        transform=ax.transAxes, fontsize=7, color="0.4", va="top")
fig.tight_layout(); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_50_vs_150_compare.png"); fig.savefig(p, dpi=170, facecolor="white")
import shutil; shutil.copy(p, os.path.join(REVIEW, "imc_50_vs_150_compare.png"))
print("wrote imc_50_vs_150_compare.png", flush=True)
