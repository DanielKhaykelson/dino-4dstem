"""Self-contained Figure-4-style panel for an extra IMC region (SI6, SI2, SI1).
One cube pass: grain segmentation (connected DINO classes >=30 px) -> grain-average
diffraction + per-pixel scattered intensity; Otsu vacuum gate; per-grain azimuthal
spottiness / 2D Bragg excess B / radial peak-halo chi. Output = class map (discrete,
ordered by spottiness, Spectral_r) + the three descriptor maps (inferno, per-panel
2-98 pct + 3*MAD cap), with a real-space scale bar. Also caches grain_acom_v2_<name>.npz.
  python src/tools/si_extra_fig4.py SI6
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib as mpl
try:
    from skimage.filters import threshold_otsu
except Exception:
    def threshold_otsu(x, nbins=256):
        h, e = np.histogram(x, nbins); c = (e[:-1] + e[1:]) / 2
        w1 = np.cumsum(h); w2 = np.cumsum(h[::-1])[::-1]
        m1 = np.cumsum(h * c) / np.clip(w1, 1, None)
        m2 = (np.cumsum((h * c)[::-1]) / np.clip(w2[::-1], 1, None))[::-1]
        v = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
        return c[:-1][np.argmax(v)]

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
INV = 0.00185; KMAX = 0.35; VMAX = 50.0; NY = NX = 128; MINPIX = 30  # VMAX high: ring-band spots reach ~12; the central beam is masked out of the descriptor band, so clipping at 2 only destroyed real crystallinity contrast
D = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM"
D50 = r"D:/DINOSR/data/231228-IMC50nm-0p2apersec-anneal-70c-60min"
EXTRA = {
    "SI6": dict(run="runs/_gui/IMC_SI6_2_3", cube=f"{D}/SI-006/Survey_CH2_1_nbed.cube.npy", nm_px=44.0, role="needle (independent)"),
    "SI2": dict(run="runs/_gui/IMC_SI2",     cube=f"{D}/SI-002/Survey_CH2_1_nbed.cube.npy", nm_px=16.0, role="thin-film crystals"),
    "SI1": dict(run="runs/_gui/IMC_SI1",     cube=f"{D}/SI-001/Survey_CH2_0_1_nbed.cube.npy", nm_px=44.0, role="thin-film crystals"),
    "50nm_SI1": dict(run="runs/_gui/IMC50_SI1", cube=f"{D50}/SI-001/Survey_CH2_1_nbed.cube.npy",   nm_px=44.0, role="50nm thin film"),
    "50nm_SI2": dict(run="runs/_gui/IMC50_SI2", cube=f"{D50}/SI-002/Survey_CH2_1_nbed.cube.npy",   nm_px=44.0, role="50nm thin film"),
    "50nm_SI3": dict(run="runs/_gui/IMC50_SI3", cube=f"{D50}/SI-003/Survey_CH2_1_nbed.cube.npy",   nm_px=44.0, role="50nm thin film"),
    "50nm_SI4": dict(run="runs/_gui/IMC50_SI4", cube=f"{D50}/SI-004/Survey_CH2_0_1_nbed.cube.npy", nm_px=44.0, role="50nm thin film"),
    "50nm_SI5": dict(run="runs/_gui/IMC50_SI5", cube=f"{D50}/SI-005/Survey_CH2_1_nbed.cube.npy",   nm_px=44.0, role="50nm thin film"),
}
PARAMS = ["spottiness", "Bragg excess B", "peak/halo χ"]


def descriptors(avg, cyx, beam, lo, hi):
    m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam); seg = m[lo:hi]; vs = v[lo:hi]
    if seg.size < 5 or seg.sum() <= 0:
        return np.nan, np.nan, np.nan
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); pk = np.clip(seg - halo, 0, None)
    chi = float((pk / np.clip(halo, 1e-9, None)).max())
    cv = np.sqrt(np.clip(vs, 0, None)) / np.clip(seg, 1e-9, None); spot = float(np.percentile(cv, 90))
    yy, xx = np.indices(avg.shape); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
    hf = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1]); band = (rr >= lo) & (rr <= hi)
    B = float(np.clip(avg[band] - hf[band], 0, None).sum() / (hf[band].sum() + 1e-9))
    return spot, B, chi


def run(name):
    c = EXTRA[name]
    asg = np.load(os.path.join(c["run"], "eval", "inference.npz"))["assigns"].astype(int)
    asgmap = asg.reshape(NY, NX)
    # grain segmentation: connected components per class, >=MINPIX
    gid = np.full(NY * NX, -1, np.int32); grains = []
    for k in sorted(set(asg.tolist())):
        lab, n = ndimage.label(asgmap == k)
        for gi in range(1, n + 1):
            mm = (lab == gi).ravel()
            if mm.sum() >= MINPIX:
                gid[mm] = len(grains); grains.append(int(k))
    G = len(grains)
    cube = open_lazy_cube(c["cube"], scan_shape=(NY, NX)); _, _, H, Wd = cube.shape
    cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), 160)
    gsum = np.zeros((G, H, Wd), np.float32); gcnt = np.zeros(G); scat = np.zeros(NY * NX)
    yy, xx = np.indices((H, Wd)); rr0 = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2); ring = rr0 > beam
    for rx in range(NY):
        blk = np.clip(np.asarray(cube[rx], np.float32), 0, VMAX)
        for ry in range(NX):
            p = blk[ry]; scat[rx * NX + ry] = float(p[ring].sum())
            g = gid[rx * NX + ry]
            if g >= 0: gsum[g] += p; gcnt[g] += 1
    # NO total-scatter vacuum gate: gscat conflates THICKNESS with crystallinity, so a
    # thin crystalline needle has low gscat and was wrongly thrown away. Instead measure
    # every grain; a grain counts as "off-sample" only if it has no measurable ring
    # signal at all (descriptors NaN). A very low absolute floor removes true holes.
    SCAT_FLOOR = 2000.0
    spot = np.full(G, np.nan); B = np.full(G, np.nan); chi = np.full(G, np.nan); vac = np.zeros(G, bool); gscat = np.zeros(G)
    for g in range(G):
        gscat[g] = float(np.median(scat[gid == g]))
        spot[g], B[g], chi[g] = descriptors(gsum[g] / max(gcnt[g], 1), cyx, beam, lo, hi)
        if (not np.isfinite(spot[g])) or gscat[g] < SCAT_FLOOR:
            vac[g] = True
    cls = np.array(grains)
    np.savez(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"), gid=gid, vac=vac, gscat=gscat,
             cls=cls, gate=SCAT_FLOOR, gsum=gsum, gcnt=gcnt, H=H)
    print(f"[{name}] grains={G} off-sample={int(vac.sum())} painted={int((~vac).sum())} (floor={SCAT_FLOOR:.0f})", flush=True)

    import shutil
    DATA = {"spottiness": spot, "Bragg excess B": B, "peak/halo χ": chi}
    uni = sorted(set(asg.tolist()))

    # ---- PER-GRAIN maps (each grain painted by its own measured value) ----
    def paint_grain(vals):
        mm = np.full(NY * NX, np.nan)
        for g in range(G):
            if not vac[g] and np.isfinite(vals[g]): mm[gid == g] = vals[g]
        return mm.reshape(NY, NX)
    # class order by median on-sample per-grain spottiness
    cval = {}
    for k in uni:
        gv = [spot[g] for g in range(G) if cls[g] == k and not vac[g] and np.isfinite(spot[g])]
        cval[k] = float(np.median(gv)) if gv else -np.inf
    order = sorted(uni, key=lambda k: cval[k]); rank = {k: i for i, k in enumerate(order)}; nC = len(uni)
    clsmap = np.vectorize(lambda v: rank[int(v)])(asgmap).astype(float)

    _LET = "abcdefgh"
    def _pl(ax, idx, dark=True):
        ax.text(0.06, 0.95, _LET[idx], transform=ax.transAxes, fontsize=15, fontweight="bold", va="top",
                ha="left", color=("white" if dark else "black"),
                bbox=dict(boxstyle="round,pad=0.15", fc=("black" if dark else "white"), alpha=0.55, ec="none"), zorder=30)
    fig = Figure(figsize=(13, 3.6), facecolor="white")
    ax = fig.add_subplot(1, 4, 1); ax.set_xticks([]); ax.set_yticks([])
    dcmap = ListedColormap([mpl.colormaps.get_cmap("Spectral_r").resampled(nC)(i) for i in range(nC)])
    ax.imshow(clsmap, cmap=dcmap, interpolation="nearest", vmin=0, vmax=max(nC - 1, 1))
    ax.set_title("DINO classes\n(ordered by spottiness)", fontsize=10); _pl(ax, 0)
    barpx = 1000.0 / c["nm_px"]
    ax.add_patch(Rectangle((6, NY - 9), barpx, 2.4, color="white", ec="black", lw=0.5))
    ax.text(6 + barpx / 2, NY - 11, "1 µm", color="white", ha="center", va="bottom", fontsize=8)
    for ci, pname in enumerate(PARAMS):
        mp = paint_grain(DATA[pname]); ax = fig.add_subplot(1, 4, ci + 2); ax.set_xticks([]); ax.set_yticks([]); _pl(ax, ci + 1)
        px = mp[np.isfinite(mp)]
        if px.size > 20:
            vmin, vmax = np.percentile(px, [2, 98]); med = float(np.median(px)); mad = float(np.median(np.abs(px - med))) * 1.4826
            if mad > 0: vmax = min(vmax, med + 3 * mad)
        else:
            vmin, vmax = (float(px.min()), float(px.max())) if px.size else (0.0, 1.0)
        if vmax <= vmin: vmax = vmin + 1e-6
        cmap = mpl.cm.get_cmap("inferno").copy(); cmap.set_bad("#222")
        im = ax.imshow(mp, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); ax.set_title(pname, fontsize=10)
    fig.suptitle(f"IMC {name} ({c['role']}): per-grain crystallinity  [vmax={int(VMAX)}, {c['nm_px']:.0f} nm/px]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); FigureCanvasAgg(fig)
    p = os.path.join(OUT, f"fig4_extra_{name}.png"); fig.savefig(p, dpi=170, facecolor="white")
    shutil.copy(p, os.path.join(REVIEW, f"fig4_extra_{name}.png")); print(f"wrote fig4_extra_{name}.png", flush=True)

    # ---- PER-CLASS boxplots (matches imc_class_boxplots_3col) ----
    def eta2(v, lab):
        v = np.asarray(v, float); lab = np.asarray(lab); ok = np.isfinite(v); v, lab = v[ok], lab[ok]
        if v.size < 5: return np.nan
        gm = v.mean(); sst = ((v - gm) ** 2).sum()
        ssb = sum(((v[lab == k].mean() - gm) ** 2) * (lab == k).sum() for k in set(lab.tolist()))
        return float(ssb / (sst + 1e-12))
    lab_g = np.array([cls[g] for g in range(G)])
    onsamp = ~vac
    figb = Figure(figsize=(15, 4.0), facecolor="white")
    for ci, pname in enumerate(PARAMS):
        v = DATA[pname]; ok = onsamp & np.isfinite(v)
        e2 = eta2(v[ok], lab_g[ok])
        ks = sorted(set(lab_g[ok].tolist()), key=lambda k: np.median(v[ok & (lab_g == k)]))
        groups = [v[ok & (lab_g == k)] for k in ks]
        ax = figb.add_subplot(1, 3, ci + 1); _pl(ax, ci, dark=False)
        bp = ax.boxplot(groups, showfliers=False, widths=0.62, patch_artist=True, medianprops=dict(color="k"))
        for patch in bp["boxes"]: patch.set_facecolor("#CA6F1E"); patch.set_alpha(0.55)
        for xi, gg in enumerate(groups):
            ax.scatter(np.full(len(gg), xi + 1) + np.linspace(-0.13, 0.13, len(gg)), gg, s=10, color="#333", alpha=0.6, linewidths=0, zorder=3)
        ax.set_xticklabels([f"c{k}\n(n={len(v[ok&(lab_g==k)])})" for k in ks], fontsize=7)
        ax.set_ylabel(f"per-grain {pname}", fontsize=9)
        ax.set_title(f"{name}: {pname} by DINO class   η²={e2:.2f}", fontsize=9.5, color="#CA6F1E")
    figb.suptitle(f"IMC {name} ({c['role']}): per-class distribution of per-grain crystallization descriptors (on-sample grains, ordered by median).", fontsize=10)
    figb.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(figb)
    pb = os.path.join(OUT, f"boxplots_extra_{name}.png"); figb.savefig(pb, dpi=160, facecolor="white")
    shutil.copy(pb, os.path.join(REVIEW, f"boxplots_extra_{name}.png")); print(f"wrote boxplots_extra_{name}.png", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "SI6")
