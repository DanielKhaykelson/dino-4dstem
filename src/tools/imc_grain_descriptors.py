"""Which ROTATION-INVARIANT parameter makes grains of one DINO class alike and
different from other classes? From grain_acom_v2_{name}.npz (per-grain average
patterns, vmax=2-clipped), per non-vacuum grain we measure a small battery of
rotation-invariant diffraction descriptors and ask, per sample, how much of each
descriptor's variance the DINO class label explains (eta^2):

  bragg   = 2D Bragg excess  = sum_band max(I - halo2d, 0) / sum_band halo2d
            (counts discrete spots AND sharp rings; ~0 for diffuse halo)
  chi_r   = radial peak/halo of the azimuthal-mean profile m(r) (SNIP halo)
            -- the OLD 1D ordering proxy; blind to spots (azimuth-averaged)
  spot    = azimuthal spottiness = 90th-pct of CV_theta(r)=sqrt(v)/m over the
            ring band (high = spots/coarse grain, low = smooth powder ring / halo)
  thick   = grain scattered intensity (gscat), a thickness proxy
  orient  = ACOM in-plane angle (control: rotation-variant -> eta^2 ~ 0)

Outputs: imc_grain_descriptors.png (eta^2 bars + chi_r-vs-spot feature space
colored by class) and imc_class_order.json (per-class median 'bragg' = the
corrected crystallization-ladder ordering for the Grad-CAM atlas).

  python tools/imc_grain_descriptors.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

OUT = "docs/explainer/figs"; FIGS = "docs/paper/draft_v2/figs"
REVIEW = os.path.join(FIGS, "latest_review")
NAMES = ["SI3", "SI4", "SI5"]
INV_ANG = 0.00185; KMAX = 0.35
FOV = {"SI3": 187, "SI4": 160, "SI5": 160}        # model field-of-view radius (raw px) from training crop
DESCS = ["bragg", "chi_r", "spot", "thick", "orient"]
DLAB = {"bragg": "Bragg excess\n(2D, radial+azim.)", "chi_r": "radial peak/halo\n(1D, azim-avg)",
        "spot": "azimuthal\nspottiness", "thick": "thickness\n(scat)", "orient": "ACOM orientation\n(control)"}


def eta2(v, lab):
    v = np.asarray(v, float); lab = np.asarray(lab)
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    if v.size < 5:
        return np.nan
    gm = v.mean(); sst = ((v - gm) ** 2).sum()
    ssb = sum(((v[lab == c].mean() - gm) ** 2) * (lab == c).sum() for c in set(lab.tolist()))
    return float(ssb / (sst + 1e-12))


def descriptors(avg, H, beam, lo, hi):
    cyx = (H - 1) / 2.0
    m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
    seg = m[lo:hi]; vseg = v[lo:hi]
    if seg.size < 5 or seg.sum() <= 0:
        return 0.0, 0.0, 0.0
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
    pk = np.clip(seg - halo, 0, None)
    chi_r = float((pk / np.clip(halo, 1e-9, None)).max())
    cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None)
    spot = float(np.percentile(cv, 90))
    # 2D Bragg excess: build a radially-symmetric halo image, integrate I-halo
    yy, xx = np.indices((H, H)); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
    halo_full = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1])
    band = (rr >= lo) & (rr <= hi)
    bragg = float(np.clip(avg[band] - halo_full[band], 0, None).sum() / (halo_full[band].sum() + 1e-9))
    return bragg, chi_r, spot


D = {}; order_index = {}
for name in NAMES:
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    gsum, gcnt, cls, vac = z["gsum"], z["gcnt"], z["cls"], z["vac"]
    gscat, orient, H = z["gscat"], z["orient"], int(z["H"])
    beam = max(8, round(0.11 * H)); lo = beam + 1
    hi = min(int(KMAX / INV_ANG), FOV[name])          # clip band to model field-of-view
    G = gsum.shape[0]
    bragg = np.zeros(G); chi_r = np.zeros(G); spot = np.zeros(G)
    for g in range(G):
        if vac[g]:
            bragg[g] = chi_r[g] = spot[g] = np.nan; continue
        bragg[g], chi_r[g], spot[g] = descriptors(gsum[g] / max(gcnt[g], 1), H, beam, lo, hi)
    m = ~vac
    vals = dict(bragg=bragg, chi_r=chi_r, spot=spot, thick=gscat.astype(float), orient=orient)
    e2 = {d: eta2(vals[d][m] if d != "orient" else vals[d], cls[m] if d != "orient" else cls) for d in DESCS}
    # orient eta2 only over indexed grains (finite)
    e2["orient"] = eta2(orient[m], cls[m])
    D[name] = dict(vals=vals, cls=cls, m=m, e2=e2)
    # per-class corrected ordering = median 2D Bragg excess
    oc = {}
    for c in sorted(set(cls[m].tolist())):
        sel = m & (cls == c)
        oc[int(c)] = float(np.nanmedian(bragg[sel])) if sel.any() else 0.0
    order_index[name] = oc
    print(f"[{name}] eta2 " + "  ".join(f"{d}={e2[d]:.2f}" for d in DESCS), flush=True)

json.dump(order_index, open(os.path.join(FIGS, "imc_class_order.json"), "w"), indent=2)

# ---- figure: eta2 bars per sample (which descriptor the DINO class encodes) ----
fig = Figure(figsize=(15, 4.4), facecolor="white")
for ri, name in enumerate(NAMES):
    e2 = D[name]["e2"]
    ax = fig.add_subplot(1, 3, ri + 1)
    colb = ["#1F618D", "#5DADE2", "#28B463", "#CA6F1E", "#909497"]
    ax.bar(range(len(DESCS)), [e2[d] for d in DESCS], color=colb)
    ax.set_xticks(range(len(DESCS))); ax.set_xticklabels([DLAB[d] for d in DESCS], fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("η² explained by DINO class")
    ax.set_title(f"{name}: which parameter the DINO class encodes", fontsize=10)
    for xi, d in enumerate(DESCS):
        ax.text(xi, e2[d] + 0.02, f"{e2[d]:.2f}", ha="center", fontsize=8)
fig.suptitle("Which rotation-invariant parameter makes grains of a DINO class alike? η² = fraction of each grain-descriptor's variance explained by the class label.\n"
             "Azimuthal spottiness (and the 2D Bragg excess that includes it) label the class; absolute orientation does NOT.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.90]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_grain_descriptors.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(REVIEW, "imc_grain_descriptors.png"))
print("wrote imc_grain_descriptors.png + imc_class_order.json", flush=True)
