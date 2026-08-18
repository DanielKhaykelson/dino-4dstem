"""Per-FRAME companion to Fig. 4: cols 2-4 (azimuthal spottiness, 2D Bragg excess B,
radial peak/halo chi) computed on EVERY single diffraction frame (no grain/class
averaging), so each scan pixel gets its own value. Col 1 is identical to Fig. 4
(discrete DINO classes, coloured Spectral_r in azimuthal-spottiness order). Noisier
than the per-class / per-grain versions by construction. Heat colormap, per-image
2-98 percentile clip + heavy-tail cap. Writes to latest_review + explainer + BorisEdits.
  python src/tools/imc_param_maps_perframe.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
import matplotlib as mpl

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")
BORIS = os.path.join(FIGS, "BorisEdits")
INV = 0.00185; KMAX = 0.35; VMAX = 2.0; NY = NX = 128
BIN = int(os.environ.get("PERFRAME_BIN", "1"))   # 2x2 detector binning to fight shot noise
FOV = {"SI3": 187, "SI4": 160, "SI5": 160}


def bin2(a, b):
    if b <= 1:
        return a
    H = (a.shape[0] // b) * b
    return a[:H, :H].reshape(H // b, b, H // b, b).mean((1, 3))
RUN = {"SI3": "runs/_gui/IMC_SI3_m097k60", "SI4": "runs/_gui/IMC_SI4_m097_k60",
       "SI5": "runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60"}
PATH = {"SI3": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
        "SI4": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
        "SI5": r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"}
NAMES = ["SI3", "SI4", "SI5"]
DISP = {"SI3": "interface", "SI4": "needles", "SI5": "magnified interface"}
PARAMS = [("spot", "azimuthal spottiness"), ("B", "2D Bragg excess B"), ("chi", "radial peak/halo χ")]


def frame_descriptors(avg, cyx, beam, lo, hi, rr, band):
    """spottiness, B, chi on one frame; NaN if unmeasurable."""
    m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
    seg = m[lo:hi]; vseg = v[lo:hi]
    if seg.size < 5 or seg.sum() <= 0:
        return np.nan, np.nan, np.nan
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
    pk = np.clip(seg - halo, 0, None)
    chi = float((pk / np.clip(halo, 1e-9, None)).max())
    cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None)
    spot = float(np.percentile(cv, 90))
    hf = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1])
    B = float(np.clip(avg[band] - hf[band], 0, None).sum() / (hf[band].sum() + 1e-9))
    return spot, B, chi


def class_spot_rank(name, asg):
    """{class: rank} by median grain spottiness (same ordering as Fig. 4 col 1)."""
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    cls, gsum, gcnt, vac = z["cls"], z["gsum"], z["gcnt"], z["vac"]
    H = int(z["H"]); cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1
    hi = min(int(KMAX / INV), FOV[name])
    spot = np.full(gsum.shape[0], np.nan)
    for g in range(gsum.shape[0]):
        if vac[g]:
            continue
        m, v, _ = _radial_mean_var(gsum[g] / max(gcnt[g], 1), (cyx, cyx), beam_px=beam)
        s = m[lo:hi]; vs = v[lo:hi]
        if s.size < 5 or s.sum() <= 0:
            continue
        spot[g] = np.percentile(np.sqrt(np.clip(vs, 0, None)) / np.clip(s, 1e-9, None), 90)
    cval = {}
    for c in np.unique(asg):
        gv = spot[cls == c]; gv = gv[np.isfinite(gv)]
        cval[int(c)] = float(np.median(gv)) if gv.size else -np.inf
    order = sorted(np.unique(asg).tolist(), key=lambda c: cval[int(c)])
    return {int(c): r for r, c in enumerate(order)}


DATA = {}
for name in NAMES:
    t0 = time.time()
    asg = np.load(os.path.join(RUN[name], "eval", "inference.npz"))["assigns"].astype(int).reshape(NY, NX)
    cache = os.path.join(FIGS, f"_perframe_cache_{name}_bin{BIN}.npz")
    if os.path.exists(cache):
        zz = np.load(cache); spot, B, chi = zz["spot"], zz["B"], zz["chi"]
        print(f"[{name}] loaded per-frame cache {os.path.basename(cache)}", flush=True)
    else:
        cube = open_lazy_cube(PATH[name], scan_shape=(NY, NX)); _, _, H, _ = cube.shape
        Hb = (H // BIN) if BIN > 1 else H; invb = INV * BIN; fovb = FOV[name] // BIN if BIN > 1 else FOV[name]
        cyx = (Hb - 1) / 2.0; beam = max(8, round(0.11 * Hb)); lo = beam + 1; hi = min(int(KMAX / invb), fovb)
        yy, xx = np.indices((Hb, Hb)); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2); band = (rr >= lo) & (rr <= hi)
        spot = np.full((NY, NX), np.nan); B = np.full((NY, NX), np.nan); chi = np.full((NY, NX), np.nan)
        for ry in range(NY):
            blk = np.clip(np.asarray(cube[ry], np.float32), 0, VMAX)
            for rx in range(NX):
                spot[ry, rx], B[ry, rx], chi[ry, rx] = frame_descriptors(bin2(blk[rx], BIN), cyx, beam, lo, hi, rr, band)
            if ry % 32 == 0: print(f"[{name}] row {ry}/{NY} ({time.time()-t0:.0f}s)", flush=True)
        np.savez(cache, spot=spot, B=B, chi=chi)
    DATA[name] = dict(asg=asg, spot=spot, B=B, chi=chi, rank=class_spot_rank(name, asg))
    print(f"[{name}] done ({time.time()-t0:.0f}s)", flush=True)

fig = Figure(figsize=(13, 3.5 * 3), facecolor="white")
for ri, name in enumerate(NAMES):
    d = DATA[name]; asg = d["asg"]; rank = d["rank"]; nC = max(len(rank), 1)
    ax = fig.add_subplot(3, 4, ri * 4 + 1); ax.set_xticks([]); ax.set_yticks([])
    dcmap = ListedColormap([mpl.colormaps.get_cmap("Spectral_r").resampled(nC)(k) for k in range(nC)])
    ax.imshow(np.vectorize(lambda c: rank[int(c)])(asg).astype(float), cmap=dcmap,
              interpolation="nearest", vmin=0, vmax=max(nC - 1, 1))
    ax.set_ylabel(DISP[name], fontsize=12, fontweight="bold", rotation=90, labelpad=8, va="center")
    if ri == 0: ax.set_title("DINO classes\n(discrete, ordered by spottiness)", fontsize=10)
    for ci, (key, lab) in enumerate(PARAMS):
        mp = d[key]; ax = fig.add_subplot(3, 4, ri * 4 + ci + 2); ax.set_xticks([]); ax.set_yticks([])
        px = mp[np.isfinite(mp)]
        if px.size > 20:
            vmin, vmax = np.percentile(px, [2, 98])
            med = float(np.median(px)); mad = float(np.median(np.abs(px - med))) * 1.4826
            if mad > 0: vmax = min(vmax, med + 3 * mad)
        elif px.size:
            vmin, vmax = float(px.min()), float(px.max())
        else:
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin: vmax = vmin + 1e-6
        im = ax.imshow(mp, cmap="inferno", interpolation="nearest", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if ri == 0: ax.set_title(lab + ("\n(per frame, 2×2 binned)" if BIN > 1 else "\n(per frame)"), fontsize=10)
fig.tight_layout(); FigureCanvasAgg(fig)
fname = f"imc_param_maps_heat_perframe_bin{BIN}.png" if BIN > 1 else "imc_param_maps_heat_perframe.png"
for d in (OUT, REVIEW, BORIS):
    fig.savefig(os.path.join(d, fname), dpi=160, facecolor="white")
print(f"wrote {fname} (latest_review + explainer + BorisEdits)", flush=True)
