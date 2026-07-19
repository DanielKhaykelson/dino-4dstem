"""2x4 panel: class-average diffraction for four representative DINO classes of
SI4 (row 1, needles) and SI5 (row 2, interface, magnified), ordered left->right by
increasing crystalline order (weighted composite of B + chi + spottiness). Panels
share a common intensity scale within each copy. Emits a log and a linear copy.
  python src/tools/si45_class_diffraction.py
"""
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
INV_ANG = 0.00185; KMAX = 0.35
ROWS = [("SI4", "needles"), ("SI5", "interface (mag.)")]
FOV = {"SI4": 160, "SI5": 160}
MIN_PX = 400            # only use classes with enough grain pixels for a clean average
NCOL = 4                # representative classes per row
CROP = 150             # half-width of the displayed reciprocal window (raw px)


def descriptors(avg, H, beam, lo, hi):
    cyx = (H - 1) / 2.0
    m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
    seg = m[lo:hi]; vseg = v[lo:hi]
    if seg.size < 5 or seg.sum() <= 0:
        return 0.0, 0.0, 0.0
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
    chi = float((np.clip(seg - halo, 0, None) / np.clip(halo, 1e-9, None)).max())
    spot = float(np.percentile(np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None), 90))
    yy, xx = np.indices((H, H)); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
    halo_full = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1])
    band = (rr >= lo) & (rr <= hi)
    bragg = float(np.clip(avg[band] - halo_full[band], 0, None).sum() / (halo_full[band].sum() + 1e-9))
    return bragg, chi, spot


def masked_crop(avg):
    """Return the cropped central-beam-masked pattern and the matching boolean mask
    (both cropped to the CROP window)."""
    H = avg.shape[0]; c = (H - 1) / 2.0; beam = max(8, round(0.11 * H))
    yy, xx = np.indices(avg.shape); mask = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) > beam
    cr = slice(int(c) - CROP, int(c) + CROP)
    return (avg.astype(np.float32) * mask)[cr, cr], mask[cr, cr]


def _recipbar(ax, H, bnm=2.0):
    bpx = bnm / (INV_ANG * 10.0); x0, y = 9, H - 20
    ax.add_patch(Rectangle((x0 - 2, y - 2), bpx + 4, 8, facecolor="black", alpha=0.5, ec="none", zorder=24))
    ax.add_patch(Rectangle((x0, y), bpx, 3.2, color="white", ec="black", lw=0.5, zorder=26))
    ax.text(x0 + bpx / 2, y - 2.5, "2 nm⁻¹", color="white", ha="center", va="bottom",
            fontsize=9, fontweight="bold", zorder=27, bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"))


# ---- gather four representative classes per row, ordered by crystalline order ----
rows_data = []
for region, role in ROWS:
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{region}.npz"))
    cls, gsum, gcnt = z["cls"], z["gsum"], z["gcnt"]; H = int(z["H"])
    beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV_ANG), FOV[region])
    classes = sorted(set(cls.tolist()))
    avg = {}; desc = {}
    for c in classes:
        sel = cls == c
        if int(gcnt[sel].sum()) < MIN_PX:
            continue
        a = gsum[sel].sum(0) / max(gcnt[sel].sum(), 1)
        avg[c] = a; desc[c] = np.array(descriptors(a, H, beam, lo, hi))
    keys = list(avg.keys()); D = np.array([desc[c] for c in keys])
    mu = D.mean(0); sd = D.std(0) + 1e-9
    comp = ((D - mu) / sd).mean(1)                       # weighted composite z per class
    order = np.argsort(comp)                             # ascending order of order
    pick = np.unique(np.linspace(0, len(order) - 1, NCOL).round().astype(int))
    chosen = [keys[order[p]] for p in pick]
    rows_data.append((region, role, chosen, avg, {c: desc[c] for c in chosen}))
    print(f"[{region}] classes {chosen}  spot={[round(float(desc[c][2]),2) for c in chosen]}", flush=True)


def build(log):
    # common intensity scale across all 8 panels
    pieces = []
    for region, role, chosen, avg, dd in rows_data:
        for c in chosen:
            crop, cmask = masked_crop(avg[c]); pieces.append(crop[cmask])
    allpix = np.concatenate(pieces)
    lo_g, hi_g = np.percentile(allpix, 2), np.percentile(allpix, 99.6)

    fig = Figure(figsize=(4 * NCOL + 0.6, 8.4), facecolor="white")
    for ri, (region, role, chosen, avg, dd) in enumerate(rows_data):
        for ci, c in enumerate(chosen):
            ax = fig.add_subplot(2, NCOL, ri * NCOL + ci + 1); ax.set_xticks([]); ax.set_yticks([])
            crop, _ = masked_crop(avg[c])
            if log:
                img = np.log1p(np.clip(crop, lo_g, hi_g) - lo_g); vmax = np.log1p(hi_g - lo_g)
                ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax)
            else:
                ax.imshow(crop, cmap="inferno", vmin=0, vmax=hi_g)
            _recipbar(ax, crop.shape[0])
            ax.set_title(f"class {c}   spottiness {dd[c][2]:.2f}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{region}\n({role})", fontsize=13, fontweight="bold", labelpad=8)
    fig.suptitle("Class-average diffraction, four representative DINO classes per field, ordered left→right by increasing "
                 "crystalline order.  Top: SI4 (needles).  Bottom: SI5 (interface, magnified).  Common intensity scale.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); FigureCanvasAgg(fig)
    fname = "si45_class_diffraction" + ("" if log else "_linear") + ".png"
    for d in (OUT, REVIEW):
        fig.savefig(os.path.join(d, fname), dpi=170, facecolor="white")
    print("wrote", fname, flush=True)


build(True); build(False)
