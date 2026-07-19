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
from matplotlib.patches import Rectangle
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


def _class_order(name, asg, cls, vals, key="spot"):
    """Rank the DINO classes by a descriptor (median value); grainless classes rank
    lowest. Returns {class_label: rank} with rank 0 = lowest value."""
    v = vals[key]; cval = {}
    for c in np.unique(asg):
        gv = v[cls == c]; gv = gv[np.isfinite(gv)]
        cval[int(c)] = float(np.median(gv)) if gv.size else -np.inf  # no grain -> low end
    order = sorted(np.unique(asg).tolist(), key=lambda c: cval[int(c)])
    return {int(c): r for r, c in enumerate(order)}


_LET = "abcdefghijklmnopqrstuvwxyz"


_DISP = {"SI3": "interface", "SI4": "needles", "SI5": "interface (magnified)"}
NMPX = {"SI3": 44.0, "SI4": 44.0, "SI5": 16.0}   # real-space step, nm per scan pixel
BARNM = {"SI3": 1000, "SI4": 1000, "SI5": 500}   # scale-bar length, nm


def _scalebar(ax, name, H, W):
    nmpx = NMPX[name]; bnm = BARNM[name]; bpx = bnm / nmpx
    lab = f"{bnm // 1000} µm" if bnm >= 1000 else f"{bnm} nm"
    x0 = 4; y = H - 7
    # dark semi-transparent backing strip behind the bar so it reads on any colour
    ax.add_patch(Rectangle((x0 - 1.5, y - 1.0), bpx + 3, 4.5, facecolor="black", alpha=0.5, ec="none", zorder=24))
    ax.add_patch(Rectangle((x0, y), bpx, max(2.2, H * 0.018), color="white", ec="black", lw=0.5, zorder=26))
    ax.text(x0 + bpx / 2, y - 2.5, lab, color="white", ha="center", va="bottom", fontsize=10, fontweight="bold",
            zorder=27, bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"))


def _pl(ax, idx):
    ax.text(0.05, 0.96, _LET[idx], transform=ax.transAxes, fontsize=16, fontweight="bold",
            va="top", ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.16", fc="black", alpha=0.55, ec="none"), zorder=20)


def render(grain_detail, fname, suptitle, cmap_name="viridis", diverging=False, pclip=(5, 95)):
    fig = Figure(figsize=(13, 3.5 * 3), facecolor="white")
    for ri, name in enumerate(NAMES):
        gid, cls, vals = DATA[name]
        _, asg = fullmap(name, "B", gid, cls, vals, grain_detail)
        ax = fig.add_subplot(3, 4, ri * 4 + 1); ax.set_xticks([]); ax.set_yticks([]); _pl(ax, ri * 4)
        # discrete classes coloured on the SAME inferno scale as the descriptor panels,
        # in azimuthal-spottiness order -> yellower class = more crystalline.
        rank = _class_order(name, asg, cls, vals, key="spot"); nC = max(len(rank), 1)
        dcmap = ListedColormap([mpl.colormaps.get_cmap("inferno")(0.06 + 0.94 * (k / max(nC - 1, 1))) for k in range(nC)])
        im0 = ax.imshow(np.vectorize(lambda c: rank[int(c)])(asg).astype(float),
                        cmap=dcmap, interpolation="nearest", vmin=0, vmax=max(nC - 1, 1))
        ax.set_ylabel(_DISP[name], fontsize=11, fontweight="bold", rotation=90, labelpad=8, va="center")
        _scalebar(ax, name, asg.shape[0], asg.shape[1])
        cb = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.02); cb.set_ticks([])
        cb.set_label("more crystalline", fontsize=8)
        cb.ax.annotate("", xy=(0.5, 0.99), xytext=(0.5, 0.02), xycoords="axes fraction",
                       arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4), zorder=6)
        if ri == 0: ax.set_title("DINO classes\n(ordered by crystallinity)", fontsize=10)
        for ci, (key, lab) in enumerate(PARAMS):
            mp, _ = fullmap(name, key, gid, cls, vals, grain_detail)
            ax = fig.add_subplot(3, 4, ri * 4 + ci + 2); ax.set_xticks([]); ax.set_yticks([]); _pl(ax, ri * 4 + ci + 1)
            px = mp[np.isfinite(mp)]                   # AREA-weighted: noise in tiny classes won't set the scale
            if px.size > 20:
                vmin, vmax = np.percentile(px, list(pclip))
                med = float(np.median(px)); mad = float(np.median(np.abs(px - med))) * 1.4826
                if mad > 0: vmax = min(vmax, med + 3 * mad)
            elif px.size:
                vmin, vmax = float(px.min()), float(px.max())
            else:
                vmin, vmax = 0.0, 1.0
            if vmax <= vmin: vmax = vmin + 1e-6
            if diverging:
                vc = float(np.median(px)) if px.size else 0.5; vc = min(max(vc, vmin + 1e-6), vmax - 1e-6)
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


_PC = ("DINO class maps coloured PER CLASS by rotation-invariant descriptor value (each whole class = its median grain value).  Col 1 = the discrete DINO "
       "classes, coloured by their crystallinity ORDER (median 2D Bragg excess B; dark=least ordered, bright=most ordered).  Cols 2–4 recolour the SAME classes "
       "by azimuthal spottiness, 2D Bragg excess B, and radial peak/halo χ; each panel is scaled per image and clipped to the 2–98 percentile of its pixels so "
       "noise (e.g. unstable χ in near-amorphous regions) does not blow out the scale.  The classes form a graded amorphous→crystalline sequence (low=amorphous "
       "halo, high=discrete α Bragg spots) that varies coherently in space and peaks along the needle morphology (esp. SI4).")
render(False, "imc_param_maps_perclass.png", _PC + "  [viridis]")
render(False, "imc_param_maps_heat.png", _PC + "  [heat: dark=amorphous, bright=crystalline]", cmap_name="inferno")
render(False, "imc_param_maps_bwr.png",
       _PC + "  [blue–white–red diverging, centred at the median class value: blue=below-median (more amorphous), red=above-median (more crystalline)]",
       cmap_name="RdBu_r", diverging=True)
render(True, "imc_param_maps.png",
       "DINO class maps coloured by rotation-invariant descriptor value (per-pixel: grain pixels = own value, others = class median).\n"
       "The crystallinity/texture parameters vary GRADUALLY in space and peak along the needle morphology; the discrete classes (col 1) stratify this continuum.  [viridis]")
# SI companion to Fig. 4: cols 2-4 measured PER GRAIN (within-class detail), not per class. Col 1 identical to Fig. 4.
render(True, "imc_param_maps_heat_pergrain.png",
       "As in Fig. 4 but cols 2–4 are coloured PER GRAIN (each grain its own descriptor value; within-class detail) rather than per class; col 1 is the same DINO "
       "class map (discrete, ordered by spottiness). Each panel is scaled per image and clipped to suppress noise.  [heat: dark=amorphous, bright=crystalline]",
       cmap_name="inferno")
shutil.copy(os.path.join(OUT, "imc_param_maps_heat_pergrain.png"),
            os.path.join(FIGS, "BorisEdits", "imc_param_maps_heat_pergrain.png"))
print("copied imc_param_maps_heat_pergrain.png -> BorisEdits", flush=True)


def render_shared(grain_detail, fname, suptitle, cmap_name="inferno", pct=None, lowest_vmax=False):
    """Like render(), but each descriptor column uses ONE vmin/vmax pooled across
    all three samples, so the colour is directly comparable between SI3/SI4/SI5
    (one shared colorbar per column). SI5 (interface) therefore reads genuinely
    lower than the needle fields rather than being stretched to fill its own row.
    pct=(lo,hi): clip the shared limits to those percentiles of the POOLED values
    (clips the low and high ends so the low/mid range is resolved); None = full
    min/max.
    lowest_vmax=True: per column, set the shared vmax to the LOWEST of the three
    samples' individual vmax (and vmin to the lowest of their vmin). The
    least-crystalline sample then spans the full colormap and the brighter samples
    saturate at the top -- best for comparing the low end. If pct is given, each
    sample's vmin/vmax is its pct percentile; otherwise its full min/max."""
    lim = {}
    for key, _ in PARAMS:
        per_sample = []          # each sample's own (vmin, vmax)
        pool = []                # all displayed values pooled
        for name in NAMES:
            gid, cls, vals = DATA[name]
            mp, _ = fullmap(name, key, gid, cls, vals, grain_detail)
            v = mp[np.isfinite(mp)].ravel(); pool.append(v)
            per_sample.append(tuple(float(x) for x in np.percentile(v, pct)) if pct is not None
                              else (float(np.nanmin(v)), float(np.nanmax(v))))
        if lowest_vmax:
            # shared scale anchored to the least-crystalline sample
            lim[key] = (min(lo for lo, _ in per_sample), min(hi for _, hi in per_sample))
        elif pct is not None:
            lim[key] = tuple(float(x) for x in np.percentile(np.concatenate(pool), pct))
        else:
            lim[key] = (float(np.nanmin(np.concatenate(pool))), float(np.nanmax(np.concatenate(pool))))
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
        ax.set_ylabel(_DISP[name], fontsize=11, fontweight="bold", rotation=90, labelpad=8, va="center")
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
render_shared(False, "imc_param_maps_heat_shared_lowvmax.png",
              "DINO class maps on ONE shared scale per descriptor, with vmax set to the LOWEST of the three samples' vmax (anchored to the least-crystalline sample).\n"
              "The low-crystallinity field (SI5) spans the full colour range; SI3/SI4 saturate at the top.  [heat: dark=amorphous, bright=crystalline]",
              lowest_vmax=True)
