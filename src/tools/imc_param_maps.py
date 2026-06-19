"""Continuous parameter maps: the full DINO class map of each IMC sample, coloured
by a rotation-invariant descriptor VALUE (spottiness / 2D Bragg excess B / radial
peak/halo chi). Every pixel is coloured (no white): grain pixels take their own
grain value (within-class detail); all other pixels take their DINO class's median
grain value (classes with no measurable grain fall to the low/amorphous end).
Rows = SI3/SI4/SI5; col 1 = discrete DINO classes (full map), cols 2-4 = the
continuous parameters. FOV-clipped, vmax=2 grain averages.

  python tools/imc_param_maps.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import matplotlib as mpl
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")
INV = 0.00185; KMAX = 0.35; FOV = {"SI3": 187, "SI4": 160, "SI5": 160}; NY = NX = 128
RUN = {"SI3": "runs/_gui/IMC_SI3_m097k60", "SI4": "runs/_gui/IMC_SI4_m097_k60",
       "SI5": "runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60"}
NAMES = ["SI3", "SI4", "SI5"]


def per_grain(name):
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    gid, cls, vac, gsum, gcnt = z["gid"], z["cls"], z["vac"], z["gsum"], z["gcnt"]
    H = int(z["H"]); cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1
    hi = min(int(KMAX / INV), FOV[name])
    G = gsum.shape[0]; spot = np.full(G, np.nan); B = np.full(G, np.nan); chi = np.full(G, np.nan)
    for g in range(G):
        if vac[g]:
            continue
        avg = gsum[g] / max(gcnt[g], 1)
        m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
        seg = m[lo:hi]; vseg = v[lo:hi]
        if seg.size < 5 or seg.sum() <= 0:
            continue
        halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); pk = np.clip(seg - halo, 0, None)
        chi[g] = (pk / np.clip(halo, 1e-9, None)).max()
        cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None); spot[g] = np.percentile(cv, 90)
        yy, xx = np.indices(avg.shape); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
        hf = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1]); band = (rr >= lo) & (rr <= hi)
        B[g] = np.clip(avg[band] - hf[band], 0, None).sum() / (hf[band].sum() + 1e-9)
    return gid, cls, dict(spot=spot, B=B, chi=chi)


def fullmap(name, key, gid, cls, vals, grain_detail=True):
    """Per-class value: every pixel = its DINO class's median grain value
    (grainless classes -> global min / amorphous end). If grain_detail, grain
    pixels are overridden with their own value for within-class structure."""
    asg = np.load(os.path.join(RUN[name], "eval", "inference.npz"))["assigns"].astype(int)
    v = vals[key]; gmin = np.nanmin(v)
    classmed = {}
    for c in np.unique(asg):
        gv = v[(cls == c)] if (cls == c).any() else np.array([np.nan])
        gv = gv[np.isfinite(gv)]
        classmed[int(c)] = float(np.median(gv)) if gv.size else gmin
    out = np.array([classmed[int(c)] for c in asg], float)        # one value per class
    if grain_detail:
        ig = np.where(gid >= 0)[0]; gv = v[gid[ig]]; ok = np.isfinite(gv)
        out[ig[ok]] = gv[ok]
    return out.reshape(NY, NX), asg.reshape(NY, NX)


PARAMS = [("spot", "azimuthal spottiness"), ("B", "2D Bragg excess B"), ("chi", "radial peak/halo χ")]
DATA = {name: per_grain(name) for name in NAMES}
import shutil; os.makedirs(REVIEW, exist_ok=True)


def render(grain_detail, fname, suptitle, cmap_name="viridis", diverging=False):
    fig = Figure(figsize=(13, 3.5 * 3), facecolor="white")
    for ri, name in enumerate(NAMES):
        gid, cls, vals = DATA[name]
        _, asg = fullmap(name, "B", gid, cls, vals, grain_detail)
        ax = fig.add_subplot(3, 4, ri * 4 + 1); ax.set_xticks([]); ax.set_yticks([])
        uni = sorted(set(asg.ravel().tolist())); lut = {u: k for k, u in enumerate(uni)}
        dcmap = ListedColormap([mpl.colormaps.get_cmap("tab20").resampled(max(len(uni), 1))(k) for k in range(len(uni))])
        ax.imshow(np.vectorize(lut.get)(asg).astype(float), cmap=dcmap, interpolation="nearest",
                  vmin=0, vmax=max(len(uni) - 1, 1))
        ax.set_ylabel(name, fontsize=13, fontweight="bold", rotation=0, labelpad=20, va="center")
        if ri == 0: ax.set_title("DINO classes\n(discrete)", fontsize=10)
        for ci, (key, lab) in enumerate(PARAMS):
            mp, _ = fullmap(name, key, gid, cls, vals, grain_detail)
            ax = fig.add_subplot(3, 4, ri * 4 + ci + 2); ax.set_xticks([]); ax.set_yticks([])
            uniq = np.unique(mp[np.isfinite(mp)])     # span the displayed (per-class) values
            vmin, vmax = (np.percentile(uniq, [8, 92]) if uniq.size > 4 else (uniq.min(), uniq.max())) \
                if uniq.size else (0.0, 1.0)
            if vmax <= vmin: vmax = vmin + 1e-6
            if diverging:
                vc = float(np.median(uniq)); vc = min(max(vc, vmin + 1e-6), vmax - 1e-6)
                norm = TwoSlopeNorm(vcenter=vc, vmin=vmin, vmax=vmax)
                im = ax.imshow(mp, cmap=cmap_name, norm=norm, interpolation="nearest")
            else:
                im = ax.imshow(mp, cmap=cmap_name, interpolation="nearest", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if ri == 0: ax.set_title(lab + "\n(continuous)", fontsize=10)
    fig.suptitle(suptitle, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(fig)
    p = os.path.join(OUT, fname); fig.savefig(p, dpi=160, facecolor="white")
    shutil.copy(p, os.path.join(REVIEW, fname)); print(f"wrote {fname}", flush=True)


_PC = ("DINO class maps coloured PER CLASS by rotation-invariant descriptor value (each whole class = its median grain value; one colour per class; "
       "colour scale = 8–92 percentile of class values so mid classes are distinguishable).  Col 1 = the discrete DINO classes; cols 2–4 recolour the SAME "
       "classes by azimuthal spottiness, 2D Bragg excess B, and radial peak/halo χ.  The classes form a graded amorphous→crystalline sequence (low=amorphous "
       "halo, high=discrete α Bragg spots) that varies coherently in space and peaks along the needle morphology (esp. SI4).")
render(False, "imc_param_maps_perclass.png", _PC + "  [viridis]")
render(False, "imc_param_maps_heat.png", _PC + "  [heat: dark=amorphous, bright=crystalline]", cmap_name="inferno")
render(False, "imc_param_maps_bwr.png",
       _PC + "  [blue–white–red diverging, centred at the median class value: blue=below-median (more amorphous), red=above-median (more crystalline)]",
       cmap_name="RdBu_r", diverging=True)
render(True, "imc_param_maps.png",
       "DINO class maps coloured by rotation-invariant descriptor value (per-pixel: grain pixels = own value, others = class median).\n"
       "The crystallinity/texture parameters vary GRADUALLY in space and peak along the needle morphology; the discrete classes (col 1) stratify this continuum.  [viridis]")


def render_shared(grain_detail, fname, suptitle, cmap_name="inferno", pct=None):
    """Like render(), but each descriptor column uses ONE vmin/vmax pooled across
    all three samples, so the colour is directly comparable between SI3/SI4/SI5
    (one shared colorbar per column). SI5 (interface) therefore reads genuinely
    lower than the needle fields rather than being stretched to fill its own row.
    pct=(lo,hi): clip the shared limits to those percentiles of the POOLED values
    (clips the low and high ends so the low/mid range is resolved); None = full
    min/max."""
    lim = {}
    for key, _ in PARAMS:
        pool = []
        for name in NAMES:
            gid, cls, vals = DATA[name]
            mp, _ = fullmap(name, key, gid, cls, vals, grain_detail)
            pool.append(mp[np.isfinite(mp)].ravel())
        pool = np.concatenate(pool)
        if pct is not None:
            lim[key] = tuple(float(x) for x in np.percentile(pool, pct))
        else:
            lim[key] = (float(np.nanmin(pool)), float(np.nanmax(pool)))
    fig = Figure(figsize=(13, 3.5 * 3), facecolor="white")
    ims = {}
    for ri, name in enumerate(NAMES):
        gid, cls, vals = DATA[name]
        _, asg = fullmap(name, "B", gid, cls, vals, grain_detail)
        ax = fig.add_subplot(3, 4, ri * 4 + 1); ax.set_xticks([]); ax.set_yticks([])
        uni = sorted(set(asg.ravel().tolist())); lut = {u: k for k, u in enumerate(uni)}
        dcmap = ListedColormap([mpl.colormaps.get_cmap("tab20").resampled(max(len(uni), 1))(k) for k in range(len(uni))])
        ax.imshow(np.vectorize(lut.get)(asg).astype(float), cmap=dcmap, interpolation="nearest",
                  vmin=0, vmax=max(len(uni) - 1, 1))
        ax.set_ylabel(name, fontsize=13, fontweight="bold", rotation=0, labelpad=20, va="center")
        if ri == 0: ax.set_title("DINO classes\n(discrete)", fontsize=10)
        for ci, (key, lab) in enumerate(PARAMS):
            mp, _ = fullmap(name, key, gid, cls, vals, grain_detail)
            ax = fig.add_subplot(3, 4, ri * 4 + ci + 2); ax.set_xticks([]); ax.set_yticks([])
            vmin, vmax = lim[key]
            if vmax <= vmin: vmax = vmin + 1e-6
            im = ax.imshow(mp, cmap=cmap_name, interpolation="nearest", vmin=vmin, vmax=vmax)
            ims[ci] = im
            # identical limits on every row -> a colorbar per panel keeps the grid
            # uniform and makes the shared scale obvious (same ticks down each column)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if ri == 0: ax.set_title(lab + "\n(shared scale across SI3/SI4/SI5)", fontsize=10)
    fig.suptitle(suptitle, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); FigureCanvasAgg(fig)
    p = os.path.join(REVIEW, fname); fig.savefig(p, dpi=160, facecolor="white")
    shutil.copy(p, os.path.join(OUT, fname))
    bp = "docs/paper/draft_v2/figs/BorisEdits"; os.makedirs(bp, exist_ok=True)
    shutil.copy(p, os.path.join(bp, fname)); print(f"wrote {fname}", flush=True)


render_shared(False, "imc_param_maps_heat_shared.png",
              "DINO class maps coloured by rotation-invariant descriptor VALUE, each descriptor on ONE shared scale across all three samples\n"
              "(one colorbar per column), so colour is directly comparable between SI3/SI4/SI5.  [heat: dark=amorphous, bright=crystalline]")
render_shared(False, "imc_param_maps_heat_shared_clip.png",
              "DINO class maps coloured by descriptor VALUE on ONE shared scale across all three samples, clipped to the 8-92 percentile of the POOLED values\n"
              "(low and high ends clipped) so the low/mid crystallinity range is resolved and the SI5 interface is comparable.  [heat: dark=amorphous, bright=crystalline]",
              pct=(8, 92))
