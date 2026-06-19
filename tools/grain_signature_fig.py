"""Pretty SI figure (one per IMC sample): what diffraction feature makes grains of
one DINO class alike, and separates classes. For each of a few classes (rows,
ordered amorphous->crystalline) we show: several INDIVIDUAL grain-average patterns
(they share a texture but the spots sit at different angles), the class average,
and an 'unrolled first-ring' strip = intensity vs azimuth at the dominant ring for
each grain. A smooth ring -> flat strip (amorphous / fine powder); discrete spots
-> punctate strip at grain-specific angles (resolved crystallite). No scatter
plots. vmax=2, log display.

  python tools/grain_signature_fig.py            # all three
  python tools/grain_signature_fig.py SI5
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import map_coordinates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
import matplotlib as mpl
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"
REVIEW = os.path.join(FIGS, "latest_review")
INV_ANG = 0.00185; KMAX = 0.35; NGRAIN = 4; NTH = 360
FOV = {"SI3": 187, "SI4": 160, "SI5": 160}        # model field-of-view radius (raw px) from training crop
CLR = {"SI3": "#2E86C1", "SI4": "#28B463", "SI5": "#CA6F1E"}


def grain_avg(z, g):
    return z["gsum"][g] / max(z["gcnt"][g], 1)


def radial(avg, cyx, beam, lo, hi):
    m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
    seg = m[lo:hi]; vseg = v[lo:hi]
    if seg.size < 5 or seg.sum() <= 0:
        return None
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
    pk = np.clip(seg - halo, 0, None)
    chi = float((pk / np.clip(halo, 1e-9, None)).max())
    cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None)
    spot = float(np.percentile(cv, 90))
    rstar = lo + int(np.argmax(pk))
    return dict(seg=seg, halo=halo, chi=chi, spot=spot, rstar=rstar)


def disp_patch(avg, cyx, beam, cr):
    """Beam-masked, percentile-stretched log display so the ring/spots show."""
    patch = avg[cr, cr].astype(np.float32)
    n = patch.shape[0]; pc = (n - 1) / 2.0
    yy, xx = np.indices((n, n)); mask = np.sqrt((yy - pc) ** 2 + (xx - pc) ** 2) > beam
    ref = patch[mask]
    if ref.size == 0:
        return np.log1p(np.clip(patch, 0, None))
    lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
    return np.log1p(np.clip(patch, lo, hi) * mask - lo)


def unroll(avg, cyx, rstar, band=2):
    th = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
    rr = np.arange(rstar - band, rstar + band + 1)
    R, T = np.meshgrid(rr, th)
    ys = cyx + R * np.sin(T); xs = cyx + R * np.cos(T)
    samp = map_coordinates(avg, [ys.ravel(), xs.ravel()], order=1, mode="nearest")
    prof = samp.reshape(NTH, len(rr)).mean(1)
    return prof / (prof.mean() + 1e-9)        # >1 = above-ring-mean (spot)


def build(name):
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    cls = z["cls"]; vac = z["vac"]; orient = z["orient"]; H = int(z["H"])
    cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1
    hi = min(int(KMAX / INV_ANG), FOV[name])          # clip band to model field-of-view
    gids = [g for g in range(z["gsum"].shape[0]) if not vac[g]]
    feats = {}
    for g in gids:
        f = radial(grain_avg(z, g), cyx, beam, lo, hi)
        if f:
            feats[g] = f
    gids = [g for g in gids if g in feats]
    by = {}
    for g in gids:
        by.setdefault(int(cls[g]), []).append(g)
    cand = {c: gg for c, gg in by.items() if len(gg) >= 3}
    spotmed = {c: np.median([feats[g]["spot"] for g in gg]) for c, gg in cand.items()}
    order = sorted(cand, key=lambda c: spotmed[c])
    if len(order) > 4:                          # span the range: keep endpoints + 2 inside
        idx = np.linspace(0, len(order) - 1, 4).round().astype(int)
        order = [order[i] for i in sorted(set(idx))]
    nrow = len(order)
    HALF = 145; s0 = int(cyx) - HALF; cr = slice(s0, int(cyx) + HALF)
    ccrop = cyx - s0                       # beam centre in crop coordinates

    R1 = round((1.0 / 7.4) / INV_ANG)      # first alpha reflection d=7.4A (innermost)

    def ring(ax, rstar):                   # principal ring (green) + first alpha ring (cyan)
        ax.add_patch(Circle((ccrop, ccrop), rstar, fill=False, ec="#39FF14",
                            lw=0.9, ls=(0, (4, 3)), alpha=0.85))
        ax.add_patch(Circle((ccrop, ccrop), R1, fill=False, ec="#19D3F3",
                            lw=0.7, ls=(0, (1, 2)), alpha=0.8))

    fig = Figure(figsize=(15, 2.5 * nrow + 0.8), facecolor="white")
    gs = fig.add_gridspec(nrow, NGRAIN + 2, width_ratios=[1] * NGRAIN + [1.05, 2.4],
                          hspace=0.28, wspace=0.08)
    cmapI = mpl.cm.get_cmap("inferno")
    col = CLR[name]
    for ri, c in enumerate(order):
        gg = sorted(cand[c], key=lambda g: -z["gcnt"][g])[:NGRAIN]
        cavg = np.mean([grain_avg(z, g) for g in cand[c]], 0)
        rstar = radial(cavg, cyx, beam, lo, hi)["rstar"]
        for ci in range(NGRAIN):
            ax = fig.add_subplot(gs[ri, ci]); ax.set_xticks([]); ax.set_yticks([])
            if ci < len(gg):
                g = gg[ci]; o = orient[g]
                ax.imshow(disp_patch(grain_avg(z, g), cyx, beam, cr), cmap=cmapI)
                ring(ax, rstar)
                ax.set_title((f"θ={o:.0f}°" if np.isfinite(o) else "grain"), fontsize=7, color="#444")
                for s in ax.spines.values():
                    s.set_edgecolor("#bbb"); s.set_linewidth(0.6)
            else:
                ax.set_axis_off()
            if ci == 0:
                ax.set_ylabel(f"class {c}", fontsize=11, fontweight="bold", color=col,
                              rotation=0, labelpad=30, va="center")
        # class average
        ax = fig.add_subplot(gs[ri, NGRAIN]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(disp_patch(cavg, cyx, beam, cr), cmap=cmapI)
        ring(ax, rstar)
        dsp = 1.0 / (rstar * INV_ANG)        # d-spacing of the principal ring (Å)
        ax.set_title(f"class avg · principal ring α{{102}}/{{112}} d={dsp:.1f}Å (cyan=first α{{022}} 7.4Å)", fontsize=5.6, color=col)
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(1.6)
        # unrolled first-ring strip, one row per grain
        ax = fig.add_subplot(gs[ri, NGRAIN + 1])
        strips = np.stack([unroll(grain_avg(z, g), cyx, rstar) for g in gg], 0)
        ax.imshow(strips, cmap="inferno", aspect="auto", vmin=0, vmax=2.5,
                  extent=[0, 360, len(gg), 0])
        ax.set_yticks([]); ax.tick_params(labelsize=6)
        ax.set_xlabel("azimuth around principal ring (deg)", fontsize=7)
        ax.set_title(f"spottiness={spotmed[c]:.2f}   χ(peak/halo)={np.median([feats[g]['chi'] for g in cand[c]]):.2f}",
                     fontsize=8, color=col)
    finding = {"SI3": "azimuthal spottiness rises 0.2→1.8 across classes, but the radial profile barely separates them (within-class r=0.58 vs between 0.51): texture, not which rings, defines the class",
               "SI4": "BOTH axes sharpen — the radial rings separate classes well (within r=0.72 vs between 0.43) and spottiness rises 0.5→1.9",
               "SI5": "spottiness rises 0.1→0.85 and carries the class; radial profiles separate only weakly (within r=0.70 vs between 0.63, far less than SI4)"}
    fig.suptitle(f"{name}: a DINO class = grains with the same AZIMUTHAL TEXTURE on the principal ring (smooth ring → discrete spots), spanning all orientations.\n"
                 f"Green dashed = principal ring (α {{102}}/{{112}}, ≈4.75Å, strongest peak over the halo, where spottiness is measured); cyan dotted = first α reflection {{022}} (d=7.4Å, innermost). "
                 f"Within a class the texture is shared but spots sit at different θ. {finding[name]}.",
                 fontsize=9.5, color="#222")
    fig.tight_layout(rect=[0.03, 0, 1, 0.93]); FigureCanvasAgg(fig)
    p = os.path.join(OUT, f"grain_signature_{name}.png")
    fig.savefig(p, dpi=160, facecolor="white")
    import shutil; os.makedirs(REVIEW, exist_ok=True)
    shutil.copy(p, os.path.join(REVIEW, f"grain_signature_{name}.png"))
    print(f"wrote grain_signature_{name}.png  classes={order}", flush=True)


if __name__ == "__main__":
    names = [sys.argv[1]] if len(sys.argv) > 1 else ["SI3", "SI4", "SI5"]
    for n in names:
        build(n)
