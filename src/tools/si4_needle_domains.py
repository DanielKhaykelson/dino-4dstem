"""Per-region 'domain' figures for the SI: panel (a) the region's DINO class map
(coloured exactly as Figure 3) with 3 chosen grains marked, panels (b-d) their
grain-average diffraction. Clean (no ticks); Figure-4 real-space scale bar on (a),
a 2 nm^-1 reciprocal scale bar on (b-d). Emits BOTH log and linear stretch.
Regions: SI3, SI4, SI5.
  python src/tools/si4_needle_domains.py
"""
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib as mpl
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
INV_ANG = 0.00185; KMAX = 0.35
FOV = {"SI3": 187, "SI4": 160, "SI5": 160}      # model field-of-view radius (raw px)
COLORS = ["#00e5ff", "#ff29d6", "#1b7d34"]   # domain marker colours (cyan, magenta, dark green)
# region -> (grains [(tag, grain_index)], nm_px, bar_nm, title, output stub[, data_region])
# grains are listed in domain order 1->3; where an order gradient is intended they are
# pre-sorted by increasing azimuthal spottiness.
REGIONS = {
    "SI4": ([("1", 17), ("2", 45), ("3", 59)], 44.0, 1000, "needle domains (SI4)", "si4_needle_domains"),
    # partly-ordered -> more ordered -> most ordered (avoid fully-amorphous background)
    "SI3": ([("1", 26), ("2", 27), ("3", 5)],  44.0, 1000, "needle domains (SI3, left needle)", "si3_domains"),
    "SI5": ([("1", 12), ("2", 23), ("3", 37)], 16.0, 500,  "domains (SI5, right side)", "si5_domains"),
    # SAME DINO class (class 2), three grains, orientation/mosaic varies (sorted by spottiness 0.83/1.08/4.14)
    "SAME": ([("1", 24), ("2", 23), ("3", 21)], 16.0, 500, "one class, three grains (SI5)", "si5_sameclass_grains", "SI5"),
    # THREE different DINO classes at three locations (class 9/2/3 = grains 69/23/33)
    "DIFF": ([("1", 33), ("2", 23), ("3", 69)], 16.0, 500, "three classes (SI5)", "si5_diffclass_grains", "SI5"),
}
# ordering of the a->b->c panels per figure: "spot" = azimuthal spottiness only,
# "composite" = weighted z-scored mean of B + chi + spottiness.
ORDER_MODE = {"DIFF": "spot"}


VMAX_LIN = 0.4   # fixed linear clip (the cached grain-avgs are vmax=50-scaled, so this is the
                 # equivalent of the GUI 'vmax=2' look on this data: spots saturate, faint ring visible)


def disp(avg, log=True):
    """Central-beam-masked display. log=True -> percentile log stretch; else a fixed
    linear clip at VMAX_LIN (as scaled in the model / GUI 'vmax=2')."""
    H = avg.shape[0]; c = (H - 1) / 2.0; beam = max(8, round(0.11 * H))
    yy, xx = np.indices(avg.shape); mask = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) > beam
    p = avg.astype(np.float32) * mask
    cr = slice(int(c) - 150, int(c) + 150)
    if log:
        pm = p[mask]; lo, hi = np.percentile(pm, 2), np.percentile(pm, 99.5)
        return np.log1p(np.clip(p, lo, hi) - lo * mask)[cr, cr]
    return np.clip(p, 0.0, VMAX_LIN)[cr, cr]


def _recipbar(ax, H, bnm=2.0):
    INVnm = 0.0185; bpx = bnm / INVnm
    x0, y = 9, H - 20
    ax.add_patch(Rectangle((x0 - 2, y - 2), bpx + 4, 8, facecolor="black", alpha=0.5, ec="none", zorder=24))
    ax.add_patch(Rectangle((x0, y), bpx, 3.2, color="white", ec="black", lw=0.5, zorder=26))
    ax.text(x0 + bpx / 2, y - 2.5, f"{int(bnm)} nm⁻¹", color="white", ha="center", va="bottom",
            fontsize=9, fontweight="bold", zorder=27, bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"))


def _lbl(ax, s):
    ax.text(0.04, 0.95, s, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left",
            color="white", bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"), zorder=30)


# ---- alpha-indomethacin [100] hkl overlay for the Figure-4 SI4 (needle) row ----
INV_CAL = 0.001809                     # A^-1/px (refined SI4 calibration = 0.0181 nm^-1/px)
ALPHA_CIF = "D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/cifs/alpha.cif"
_ROW_HKL = [(0, -1, 2), (0, 1, -2), (0, -1, 3), (0, 1, -3), (0, -2, 4), (0, 2, -4)]  # 012->013->024 row
try:
    import itertools as _it
    from scipy.ndimage import maximum_filter as _maxf
    from pymatgen.core import Structure as _Struct
    _st = _Struct.from_file(ALPHA_CIF); _Bm = _st.lattice.reciprocal_lattice_crystallographic.matrix
    _tv = np.array([1., 0., 0.]) @ _st.lattice.matrix; _tv /= np.linalg.norm(_tv)
    _e1 = np.cross(_tv, [0., 1., 0.]); _e1 /= np.linalg.norm(_e1); _e2 = np.cross(_tv, _e1)
    _RF = []
    for _h, _k, _l in _it.product(range(-9, 10), repeat=3):
        if _h != 0 or (_h, _k, _l) == (0, 0, 0) or (_l == 0 and _k % 2 != 0):
            continue
        _g = np.array([_h, _k, _l]) @ _Bm
        if 0.087 < np.linalg.norm(_g) < 0.40:
            _RF.append((np.array([_g @ _e1, _g @ _e2]), (_h, _k, _l)))
    _PV = np.array([r[0] for r in _RF]); _HKL_OK = True
except Exception:
    _HKL_OK = False


def _hkl_str(h):
    return ''.join((str(abs(x)) + '̅' if x < 0 else str(x)) for x in h)


def _detect(av):
    H = av.shape[0]; cc = (H - 1) / 2.0
    yy, xx = np.indices(av.shape); r = np.sqrt((yy - cc) ** 2 + (xx - cc) ** 2)
    prof = np.bincount(r.astype(int).ravel(), av.ravel()) / np.maximum(np.bincount(r.astype(int).ravel()), 1)
    wk = av.astype(float).copy(); wk[r < 48] = 0
    pk = (wk == _maxf(wk, 9)) & (wk > np.percentile(wk[r > 48], 99.0)) & (r > 48) & (r < 150)
    ys, xs = np.where(pk); val = wk[ys, xs]
    O = np.c_[(xs - cc) * INV_CAL, -(ys - cc) * INV_CAL]
    sharp = val > 3.0 * prof[np.clip(np.hypot(xs - cc, ys - cc).astype(int), 0, len(prof) - 1)]
    return O[sharp][np.argsort(-val[sharp])][:16], O[np.argsort(-val)]


def _fitrot(O):
    best = (-1, 0.0)
    for a in np.deg2rad(np.arange(0, 360, 0.5)):
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]); PR = _PV @ R.T
        s = O[:8]; sc = (sum(np.min(np.linalg.norm(PR - o, axis=1)) < 0.02 for o in s)
                         - 0.4 * sum(np.min(np.linalg.norm(PR - o, axis=1)) for o in s))
        if sc > best[0]: best = (sc, a)
    return best[1]


def si4_hkl_rotations(gsum, gcnt):
    """{grain: rotation} for the SI4 needle grains; 59 shares 17's orientation (same domain)."""
    r17 = _fitrot(_detect(gsum[17] / max(gcnt[17], 1))[0])
    r45 = _fitrot(_detect(gsum[45] / max(gcnt[45], 1))[0])
    return {17: r17, 59: r17, 45: r45}


def overlay_hkl(ax, av, a):
    """White alpha[100] hkl labels on a diffraction panel (disp crop is 300 px, centre 150)."""
    Os, Oall = _detect(av); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]); PR = _PV @ R.T
    lab = []
    for T in _ROW_HKL:
        g = np.array(T) @ _Bm; pt = np.array([g @ _e1, g @ _e2]) @ R.T
        j = int(np.argmin(np.linalg.norm(Oall - pt, axis=1)))
        if np.linalg.norm(Oall[j] - pt) < 0.022:
            lab.append((Oall[j], T))
    for o in Os:
        j = int(np.argmin(np.linalg.norm(PR - o, axis=1)))
        if np.linalg.norm(PR[j] - o) < 0.028 and not any(np.linalg.norm(o - p) < 0.02 for p, _ in lab):
            lab.append((o, _RF[j][1]))
    placed = []
    for o, h in lab:
        px, py = o[0] / INV_CAL, -o[1] / INV_CAL; rn = np.hypot(px, py); ux, uy = px / rn, py / rn
        for off in (14, 22, 30, 8, -14):
            lx, ly = 150 + px + ux * off, 150 + py + uy * off
            if not (6 < lx < 294 and 6 < ly < 294):
                continue
            if ly > 260 and lx < 150:                    # keep clear of the 2 nm^-1 scale bar
                continue
            if all(np.hypot(lx - a2, ly - b2) > 20 for a2, b2 in placed):
                ax.text(lx, ly, _hkl_str(h), color="white", fontsize=8.0, fontweight="bold",
                        ha="center", va="center", zorder=28); placed.append((lx, ly)); break


def descriptors(avg, H, beam, lo, hi):
    """Return the three crystallinity descriptors (B, chi, spot) of one pattern:
    2D Bragg excess, radial peak/halo ratio, azimuthal spottiness (90th-pct CV)."""
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


def composite_order(grains, data_region, gsum, gcnt, cls, vac, by="composite"):
    """Order the chosen grains by crystalline order. by="spot": azimuthal spottiness
    only. by="composite": weighted mean of the three descriptors (B, chi, spot), each
    z-scored against the region's grain population. Returns grains re-tagged 1..3 in
    ascending order plus the per-grain scores (composite_z, grain, [B, chi, spot])."""
    H = gsum.shape[1]; beam = max(8, round(0.11 * H)); lo = beam + 1
    hi = min(int(KMAX / INV_ANG), FOV.get(data_region, 160))
    # population statistics over all non-vacuum grains
    G = gsum.shape[0]; pop = np.full((G, 3), np.nan)
    for g in range(G):
        if not vac[g]:
            pop[g] = descriptors(gsum[g] / max(gcnt[g], 1), H, beam, lo, hi)
    mu = np.nanmean(pop, axis=0); sd = np.nanstd(pop, axis=0) + 1e-9
    scored = []
    for tag, g in grains:
        d = np.array(descriptors(gsum[g] / max(gcnt[g], 1), H, beam, lo, hi))
        z = (d - mu) / sd
        scored.append((float(z.mean()), g, d))          # equal-weight composite z-score
    key = (lambda t: t[2][2]) if by == "spot" else (lambda t: t[0])
    scored.sort(key=key)
    ordered = [(str(i + 1), g) for i, (s, g, d) in enumerate(scored)]
    return ordered, scored


def render_row(axes, region, log, letters="abcd", side_title=False, title_override=None, index_hkl=False):
    """Draw one domain row into the four supplied axes [class_map, d1, d2, d3].
    letters = the four panel labels; side_title places the headline as a rotated
    y-label on the class map (Fig 3/4 style) instead of above it; title_override
    replaces the region's default headline."""
    entry = REGIONS[region]
    grains, nmpx, barnm, title, stub = entry[:5]
    if title_override is not None:
        title = title_override
    data_region = entry[5] if len(entry) > 5 else region
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{data_region}.npz"))
    gid, cls, gsum, gcnt, vac = z["gid"], z["cls"], z["gsum"], z["gcnt"], z["vac"]
    dino = np.load(os.path.join(FIGS, f"boris_nmf_cache_{data_region}.npz"))["dino"]
    NY, NX = dino.shape; gm = gid.reshape(NY, NX)

    by = ORDER_MODE.get(region, "composite")
    grains, scored = composite_order(grains, data_region, gsum, gcnt, cls, vac, by=by)
    if log:
        print(f"[{region}] order by '{by}' (composite z, B, chi, spot):", flush=True)
        for tag, (s, g, d) in zip("abc", scored):
            print(f"   {tag} grain {g}: composite={s:+.2f}  B={d[0]:.3f} chi={d[1]:.2f} spot={d[2]:.2f}", flush=True)

    ax = axes[0]; ax.set_xticks([]); ax.set_yticks([])
    uni = sorted(np.unique(dino).tolist()); has_noise = -1 in uni
    base = mpl.colormaps.get_cmap("tab20b").resampled(max(len([u for u in uni if u != -1]), 1))
    cols = [(0.72, 0.72, 0.72) if u == -1 else base(i - (1 if has_noise else 0)) for i, u in enumerate(uni)]
    lut = {u: i for i, u in enumerate(uni)}
    ax.imshow(np.vectorize(lut.get)(dino).astype(float), cmap=ListedColormap(cols),
              interpolation="nearest", vmin=0, vmax=max(len(uni) - 1, 1))
    for i, (tag, g) in enumerate(grains):
        col = COLORS[i]
        ax.contour((gm == g).astype(float), levels=[0.5], colors=[col], linewidths=2.0)
        ys, xs = np.where(gm == g)
        ax.text(xs.mean(), ys.mean(), tag, color="black", fontsize=11, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.28", fc=col, ec="white", lw=1.0), zorder=20)
    if side_title:
        ax.set_ylabel(title, fontsize=12, fontweight="bold", rotation=90, labelpad=8, va="center")
    else:
        ax.set_title(title, fontsize=10)
    x0, y = 4, NY - 7; bpx = barnm / nmpx
    ax.add_patch(Rectangle((x0 - 1.5, y - 1.0), bpx + 3, 4.5, facecolor="black", alpha=0.5, ec="none", zorder=24))
    ax.add_patch(Rectangle((x0, y), bpx, max(2.2, NY * 0.018), color="white", ec="black", lw=0.5, zorder=26))
    lab = f"{barnm // 1000} µm" if barnm >= 1000 else f"{barnm} nm"
    ax.text(x0 + bpx / 2, y - 2.5, lab, color="white", ha="center", va="bottom", fontsize=10, fontweight="bold",
            zorder=27, bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"))
    _lbl(ax, letters[0])
    hkl_rot = si4_hkl_rotations(gsum, gcnt) if (index_hkl and _HKL_OK) else {}
    for i, (tag, g) in enumerate(grains):
        col = COLORS[i]
        ax = axes[i + 1]; ax.set_xticks([]); ax.set_yticks([])
        db = disp(gsum[g] / max(gcnt[g], 1), log=log); ax.imshow(db, cmap="inferno")
        if g in hkl_rot:
            overlay_hkl(ax, gsum[g] / max(gcnt[g], 1), hkl_rot[g])
        _recipbar(ax, db.shape[0])
        for s in ax.spines.values():
            s.set_color(col); s.set_linewidth(3.5)
        _lbl(ax, letters[i + 1])
        ax.set_title(f"domain {tag}", fontsize=10, color=col, fontweight="bold")
    return stub


def build(region, log):
    fig = Figure(figsize=(14, 3.7), facecolor="white")
    axes = [fig.add_subplot(1, 4, k + 1) for k in range(4)]
    stub = render_row(axes, region, log, letters="abcd", side_title=False)
    fig.tight_layout(); FigureCanvasAgg(fig)
    fname = stub + ("" if log else "_linear") + ".png"
    fig.savefig(os.path.join(OUT, fname), dpi=185, facecolor="white")
    shutil.copy(os.path.join(OUT, fname), os.path.join(REVIEW, fname))
    print("wrote", fname, flush=True)


def build_combined(top, bottom, stub, log):
    """Two domain rows stacked in one canvas: continuous labels a-h, side titles."""
    fig = Figure(figsize=(14, 7.3), facecolor="white")
    top_axes = [fig.add_subplot(2, 4, k + 1) for k in range(4)]
    bot_axes = [fig.add_subplot(2, 4, 4 + k + 1) for k in range(4)]
    render_row(top_axes, top, log, letters="abcd", side_title=True, title_override="needle domains", index_hkl=True)
    render_row(bot_axes, bottom, log, letters="efgh", side_title=True, title_override="interface domains")
    fig.tight_layout(); FigureCanvasAgg(fig)
    fname = stub + ("" if log else "_linear") + ".png"
    fig.savefig(os.path.join(OUT, fname), dpi=185, facecolor="white")
    shutil.copy(os.path.join(OUT, fname), os.path.join(REVIEW, fname))
    print("wrote", fname, flush=True)


if __name__ == "__main__":
    for _r in ["SI4", "SI3", "SI5", "SAME", "DIFF"]:
        for _log in [True, False]:
            build(_r, _log)
    # combined figure: needles (SI4) above interface classes (SI5), labels a-h
    for _log in [True, False]:
        build_combined("SI4", "DIFF", "si4needle_si5class_combined", _log)
