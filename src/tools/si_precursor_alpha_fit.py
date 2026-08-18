"""SI figure: the indomethacin precursor (least-ordered on-sample class) and a mature
needle both index to the alpha-IMC structure. Experimental azimuthally-averaged radial
profiles on a d-spacing axis, with the alpha reflections COMPUTED from alpha.cif
(pymatgen, kinematical) overlaid as intensity-scaled sticks. Shows the experimental
peaks fall on the alpha reflection positions, and doubles as a calibration check.
  python src/tools/si_precursor_alpha_fit.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from gui_app.crystallinity_panel import _radial_mean_var

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
CIF = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/cifs/alpha.cif"
INV = 0.00185  # A^-1 per detector pixel
DMIN, DMAX = 3.0, 9.0


def reflections_from_cif(cif, dmin=DMIN, dmax=DMAX):
    """Return [(d_A, rel_intensity, 'hkl'), ...] computed from the CIF (kinematical)."""
    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        s = Structure.from_file(cif)
        # Wide 2-theta range so we capture the low-angle (large-d) organic reflections.
        pat = XRDCalculator(wavelength="CuKa").get_pattern(s, two_theta_range=(2, 40))
        peaks = []
        for d, inten, hkls in zip(pat.d_hkls, pat.y, pat.hkls):
            if dmin <= d <= dmax:
                h, k, l = hkls[0]["hkl"]
                peaks.append((float(d), float(inten), f"{{{abs(int(h))}{abs(int(k))}{abs(int(l))}}}"))
        # merge reflections within 0.03 A (degenerate at our resolution): keep strongest label
        peaks.sort(key=lambda t: -t[0]); merged = []
        for d, i, lab in peaks:
            if merged and abs(merged[-1][0] - d) < 0.04:
                d0, i0, lab0 = merged[-1]
                merged[-1] = (d0, i0 + i, lab0 if i0 >= i else lab)
            else:
                merged.append((d, i, lab))
        if merged:
            mx = max(i for _, i, _ in merged)
            return [(d, i / mx, lab) for d, i, lab in merged], "computed from alpha.cif (pymatgen, kinematical)"
    except Exception as e:
        print("pymatgen CIF compute failed:", e)
    verified = [(7.4, 0.5, "{022}"), (6.0, 0.4, "{003}"), (4.79, 1.0, "{102}/{112}"), (3.90, 0.6, "{103}")]
    return verified, "verified alpha-IMC assignments (fallback)"


refs, src = reflections_from_cif(CIF)
print("reflection source:", src)
print("reflections (d, I, hkl):", [(round(d, 2), round(i, 2), lab) for d, i, lab in refs])


def class_avg_profiles(name, pick):
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    cls, vac, gsum, gcnt, gscat = z["cls"], z["vac"], z["gsum"], z["gcnt"], z["gscat"]
    H = int(z["H"]); cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1
    hi = min(int(0.35 / INV), 187 if name == "SI3" else 160)
    med = np.median(gscat[~vac])
    sp = {}
    for c in sorted(set(cls[~vac].tolist())):
        idx = [g for g in range(gsum.shape[0]) if cls[g] == c and not vac[g] and gscat[g] >= med]
        if len(idx) < 2:
            continue
        avg = sum(gsum[g] for g in idx) / max(sum(gcnt[g] for g in idx), 1)
        m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam); seg = m[lo:hi]; vs = v[lo:hi]
        cv = np.percentile(np.sqrt(np.clip(vs, 0, None)) / np.clip(seg, 1e-9, None), 90)
        sp[c] = (cv, m, lo, hi)
    order = sorted(sp, key=lambda c: sp[c][0])
    c = order[0] if pick == "low" else order[-1]
    cv, m, lo, hi = sp[c]
    # q in A^-1 directly from the pixel radius (0.00185/px, full 512-px detector; no crop applied here)
    r = np.arange(lo, hi); q = r * INV
    prof = m[lo:hi]
    return q, prof / np.nanmax(prof), c, cv


qP, pP, cP, svP = class_avg_profiles("SI3", "low")    # precursor (least ordered, thick)
qN, pN, cN, svN = class_avg_profiles("SI4", "high")   # mature needle
QMIN, QMAX = 0.10, 0.33   # A^-1 (covers {022} ~0.136 through ~{105})

fig = Figure(figsize=(9.2, 4.9), facecolor="white"); ax = fig.add_subplot(111)
# calculated alpha reflections as intensity-scaled sticks, plotted in q = 1/d
for d0, i0, lab in refs:
    q0 = 1.0 / d0
    if not (QMIN <= q0 <= QMAX):
        continue
    ax.vlines(q0, 0, 0.18 + 0.82 * i0, color="#b2182b", lw=1.0 + 2.2 * i0, alpha=0.5, zorder=0)
    if i0 > 0.45:
        ax.text(q0, 0.20 + 0.82 * i0, f"{lab}\n{d0:.2f}Å", ha="center", va="bottom", fontsize=7, color="#b2182b")
ax.plot(qP, pP, lw=2.0, color="#8c510a", label=f"precursor (interface c{cP}, spottiness {svP:.2f})")
ax.plot(qN, pN, lw=2.0, color="#01665e", label=f"mature needle (needles c{cN}, spottiness {svN:.2f})")
ax.set_xlim(QMIN, QMAX); ax.set_ylim(0, 1.35)
ax.set_xlabel("q = 1/d  (Å⁻¹)"); ax.set_ylabel("azimuthally-averaged intensity (norm.)")
ax.set_title("")
secax = ax.secondary_xaxis("top", functions=(lambda q: 1.0 / np.clip(q, 1e-6, None),
                                             lambda d: 1.0 / np.clip(d, 1e-6, None)))
secax.set_xlabel("d-spacing (Å)")
ax.legend(fontsize=8, loc="upper right")
ax.text(0.01, 0.97, "calibration 0.00185 Å⁻¹/px", transform=ax.transAxes, fontsize=7, color="0.4", va="top")
ax.text(0.99, 0.02, src, transform=ax.transAxes, fontsize=7, color="#b2182b", ha="right")
fig.tight_layout(); FigureCanvasAgg(fig)
p = os.path.join(OUT, "si_precursor_alpha_fit.png"); fig.savefig(p, dpi=170, facecolor="white")
import shutil; shutil.copy(p, os.path.join(REVIEW, "si_precursor_alpha_fit.png"))
print(f"wrote si_precursor_alpha_fit.png  precursor=SI3 c{cP} needle=SI4 c{cN}", flush=True)
