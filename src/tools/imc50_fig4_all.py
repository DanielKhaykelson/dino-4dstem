"""Figure-4-style combined panel for the 50nm IMC films: rows = the unique 50nm
fields, columns = DINO class map (ordered by spottiness) + the three per-grain
descriptor maps (azimuthal spottiness, 2D Bragg excess B, radial peak/halo chi).
Same rendering as the main Figure 4 (imc_param_maps / si_extra_fig4). Reuses the
cached grain_acom_v2_50nm_SI*.npz, so no cube pass.
  python src/tools/imc50_fig4_all.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib as mpl

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
D50 = r"D:/DINOSR/data/231228-IMC50nm-0p2apersec-anneal-70c-60min"
INV = 0.00185; KMAX = 0.35; NY = NX = 128; NMPX = 44.0
# SI-003 == SI-004 (duplicate raw data); show the four unique fields.
ROWS = [("SI-001", "50nm_SI1", f"{D50}/SI-001/Survey_CH2_1_nbed.cube.npy"),
        ("SI-002", "50nm_SI2", f"{D50}/SI-002/Survey_CH2_1_nbed.cube.npy"),
        ("SI-003", "50nm_SI3", f"{D50}/SI-003/Survey_CH2_1_nbed.cube.npy"),
        ("SI-005", "50nm_SI5", f"{D50}/SI-005/Survey_CH2_1_nbed.cube.npy")]
PARAMS = ["azimuthal spottiness", "2D Bragg excess B", "radial peak/halo χ"]
_LET = "abcdefghijklmnopqrstuvwxyz"


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


def load(name):
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    gid, vac, cls, gsum, gcnt = z["gid"], z["vac"], z["cls"], z["gsum"], z["gcnt"]
    H = int(z["H"]); cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), 160)
    G = gsum.shape[0]
    spot = np.full(G, np.nan); B = np.full(G, np.nan); chi = np.full(G, np.nan)
    for g in range(G):
        if vac[g]:
            continue
        spot[g], B[g], chi[g] = descriptors(gsum[g] / max(gcnt[g], 1), cyx, beam, lo, hi)
    return dict(gid=gid, vac=vac, cls=cls, spot=spot, B=B, chi=chi)


def virtual_haadf(name, cube_path):
    """Annular dark-field (virtual HAADF): integrated intensity beyond the central beam,
    per scan position. Cached to haadf_<name>.npy so re-runs are instant."""
    cache = os.path.join(FIGS, f"vhaadf_{name}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    from data import open_lazy_cube
    cube = open_lazy_cube(cube_path, scan_shape=(NY, NX))
    H, W = cube.shape[2], cube.shape[3]; cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H))
    yy, xx = np.indices((H, W)); ring = (np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2) > beam)
    haadf = np.zeros(NY * NX, np.float64)
    for rx in range(NY):
        blk = np.asarray(cube[rx], np.float32)            # (NX, H, W)
        haadf[rx * NX:(rx + 1) * NX] = (blk * ring).reshape(blk.shape[0], -1).sum(1)
        if rx % 32 == 0:
            print(f"   [{name}] HAADF row {rx}/{NY}", flush=True)
    haadf = haadf.reshape(NY, NX); np.save(cache, haadf)
    return haadf


def paint(gid, vac, vals):
    mm = np.full(NY * NX, np.nan)
    for g in range(len(vals)):
        if not vac[g] and np.isfinite(vals[g]):
            mm[gid == g] = vals[g]
    return mm.reshape(NY, NX)


def _pl(ax, idx, dark=True):
    ax.text(0.06, 0.95, _LET[idx], transform=ax.transAxes, fontsize=15, fontweight="bold", va="top", ha="left",
            color=("white" if dark else "black"),
            bbox=dict(boxstyle="round,pad=0.15", fc=("black" if dark else "white"), alpha=0.55, ec="none"), zorder=30)


nrow = len(ROWS); ncol = 5
fig = Figure(figsize=(15.5, 3.2 * nrow), facecolor="white")
for ri, (role, name, cube_path) in enumerate(ROWS):
    d = load(name); gid, vac, cls, spot = d["gid"], d["vac"], d["cls"], d["spot"]
    DATA = {"azimuthal spottiness": spot, "2D Bragg excess B": d["B"], "radial peak/halo χ": d["chi"]}
    # ---- col 1: virtual HAADF ----
    haadf = virtual_haadf(name, cube_path)
    ax = fig.add_subplot(nrow, ncol, ri * ncol + 1); ax.set_xticks([]); ax.set_yticks([])
    lo_, hi_ = np.percentile(haadf, [1, 99])
    ax.imshow(np.clip(haadf, lo_, hi_), cmap="gray", interpolation="nearest")
    if ri == 0:
        ax.set_title("virtual HAADF", fontsize=10)
    ax.set_ylabel(f"{role}", fontsize=11, fontweight="bold", rotation=90, labelpad=8, va="center")
    _pl(ax, ri * ncol + 0, dark=False)
    barpx = 1000.0 / NMPX
    ax.add_patch(Rectangle((6, NY - 9), barpx, 2.4, color="white", ec="black", lw=0.5))
    ax.text(6 + barpx / 2, NY - 11, "1 µm", color="white", ha="center", va="bottom", fontsize=7)
    # per-pixel class map from grain ids -> class, recoloured by class-median spottiness
    on = ~vac & np.isfinite(spot)
    uni = sorted(set(cls[on].tolist()))
    cval = {k: (np.median(spot[on & (cls == k)]) if (on & (cls == k)).any() else -np.inf) for k in uni}
    order = sorted(uni, key=lambda k: cval[k]); rank = {k: i for i, k in enumerate(order)}; nC = max(len(uni), 1)
    clsmap = np.full(NY * NX, np.nan)
    for g in range(len(cls)):
        if gid is not None:
            sel = gid == g
            if sel.any() and cls[g] in rank:
                clsmap[sel] = rank[cls[g]]
    clsmap = clsmap.reshape(NY, NX)
    ax = fig.add_subplot(nrow, ncol, ri * ncol + 2); ax.set_xticks([]); ax.set_yticks([])
    dcmap = ListedColormap([mpl.colormaps.get_cmap("Spectral_r").resampled(nC)(i) for i in range(nC)]); dcmap.set_bad("#e8e8e8")
    ax.imshow(clsmap, cmap=dcmap, interpolation="nearest", vmin=0, vmax=max(nC - 1, 1))
    if ri == 0:
        ax.set_title("DINO classes\n(ordered by spottiness)", fontsize=10)
    _pl(ax, ri * ncol + 1)
    for ci, pname in enumerate(PARAMS):
        mp = paint(gid, vac, DATA[pname]); ax = fig.add_subplot(nrow, ncol, ri * ncol + ci + 3)
        ax.set_xticks([]); ax.set_yticks([]); _pl(ax, ri * ncol + ci + 2)
        px = mp[np.isfinite(mp)]
        if px.size > 20:
            vmin, vmax = np.percentile(px, [2, 98]); md = float(np.median(px)); mad = float(np.median(np.abs(px - md))) * 1.4826
            if mad > 0:
                vmax = min(vmax, md + 3 * mad)
        else:
            vmin, vmax = (float(px.min()), float(px.max())) if px.size else (0.0, 1.0)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        cmap = mpl.cm.get_cmap("inferno").copy(); cmap.set_bad("#222")
        im = ax.imshow(mp, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if ri == 0:
            ax.set_title(pname, fontsize=10)
fig.suptitle("50 nm indomethacin films: per-grain crystallinity (rows = unique fields; SI-003 ≡ SI-004 duplicate omitted)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "fig4_imc50_all.png"); fig.savefig(p, dpi=170, facecolor="white")
import shutil; shutil.copy(p, os.path.join(REVIEW, "fig4_imc50_all.png"))
print("wrote fig4_imc50_all.png", flush=True)
