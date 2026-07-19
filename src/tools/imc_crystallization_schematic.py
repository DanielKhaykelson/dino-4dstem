"""Figure 5: 'what is happening to IMC'. TOP: a structural scheme (drawn, not
HAADF) of a less-ordered matrix BLOB from which highly-ordered needles grow, as a
continuous gain of structural order, with the three imaged fields of view (SI3
overview, SI4 needles, SI5 interface) highlighted as ROI boxes. BOTTOM: three
annotated representative diffraction patterns spanning the same ladder (less
ordered -> highly ordered), with the masked beam, the principal alpha ring, and
the discrete spots labelled, plus the 'unrolled first-ring' strip explained.

  python tools/imc_crystallization_schematic.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import map_coordinates
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtrans
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")
INV = 0.00185; KMAX = 0.35; FOV = {"SI3": 187, "SI4": 160, "SI5": 160}; NTH = 360
NAVY = "#21295C"; SLATE = "#2E5E8C"; GREY = "#9aa7b3"


def grains(name):
    z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
    gsum, gcnt, vac, gscat = z["gsum"], z["gcnt"], z["vac"], z["gscat"]; H = int(z["H"])
    cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV[name])
    out = []
    for g in range(gsum.shape[0]):
        if vac[g] or gcnt[g] < 40:
            continue
        avg = gsum[g] / max(gcnt[g], 1)
        m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
        seg = m[lo:hi]; vseg = v[lo:hi]
        if seg.size < 5 or seg.sum() <= 0:
            continue
        halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); pk = np.clip(seg - halo, 0, None)
        cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None)
        out.append(dict(g=g, avg=avg, spot=float(np.percentile(cv, 90)), rstar=lo + int(np.argmax(pk)),
                        cyx=cyx, beam=beam, n=int(gcnt[g]), gscat=float(gscat[g])))
    return out


def pick(gs, lowest, frac=0.35, min_n=60):
    """On-sample (above-median scatter) so we never pick a thin hole/carbon region."""
    med = np.median([d["gscat"] for d in gs])
    cand = [d for d in gs if d["gscat"] >= med and d["n"] >= min_n] or [d for d in gs if d["gscat"] >= med] or gs
    s = sorted(cand, key=lambda d: d["spot"]); k = max(1, int(len(s) * frac))
    return max(s[:k] if lowest else s[-k:], key=lambda d: d["n"])


def disp(avg, cyx, beam, cr):
    p = avg[cr, cr].astype(np.float32); pc = (p.shape[0] - 1) / 2.0
    yy, xx = np.indices(p.shape); mask = np.sqrt((yy - pc) ** 2 + (xx - pc) ** 2) > beam
    lo, hh = np.percentile(p[mask], 2), np.percentile(p[mask], 99.5)
    return np.log1p(np.clip(p, lo, hh) * mask - lo), pc


def unroll(avg, cyx, rstar, band=2):
    th = np.linspace(0, 2 * np.pi, NTH, endpoint=False); rr = np.arange(rstar - band, rstar + band + 1)
    R, T = np.meshgrid(rr, th); ys = cyx + R * np.sin(T); xs = cyx + R * np.cos(T)
    p = map_coordinates(avg, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(NTH, len(rr)).mean(1)
    return (p / (p.mean() + 1e-9))[None, :]


def draw_scheme(ax):
    ax.set_xlim(0, 980); ax.set_ylim(470, 0); ax.axis("off")
    bcx, bcy, brx, bry = 250, 230, 165, 120
    th = np.linspace(0, 2 * np.pi, 80)
    r = 1 + 0.10 * np.cos(3 * th + 0.6) + 0.06 * np.cos(5 * th + 1.3) + 0.04 * np.cos(2 * th)
    ax.add_patch(Polygon(np.column_stack([bcx + brx * r * np.cos(th), bcy + bry * r * np.sin(th)]),
                         closed=True, fc="#e3e9ef", ec="#b3bdc7", lw=1.6, zorder=1))
    rng = np.random.RandomState(5)
    for _ in range(34):
        a = rng.uniform(0, 2 * np.pi); rr2 = rng.uniform(0, 0.82)
        ax.scatter(bcx + brx * rr2 * np.cos(a), bcy + bry * rr2 * np.sin(a), s=11, color=GREY, alpha=0.45, lw=0, zorder=2)
    nx, ny = 355, 232
    for ang, L, col, al, hw in [(-24, 150, "#5a7d9e", .55, 5), (-14, 300, "#3e6a90", .72, 5.5),
                                (-5, 392, SLATE, .88, 6), (4, 410, "#244e78", .96, 6.5),
                                (13, 360, SLATE, .88, 6), (22, 285, "#3e6a90", .72, 5.5), (30, 150, "#5a7d9e", .55, 5)]:
        rp = Rectangle((nx, ny - hw / 2), L, hw, fc=col, ec="none", alpha=al, zorder=3)
        rp.set_transform(mtrans.Affine2D().rotate_deg_around(nx, ny, ang) + ax.transData); ax.add_patch(rp)
    ax.text(150, 150, "less-ordered\nmatrix", ha="center", va="center", fontsize=11.5, fontweight="600", color="#5b6770", zorder=4)
    ax.text(770, 120, "highly-ordered\nneedles", ha="center", va="center", fontsize=11.5, fontweight="600", color="#21557f", zorder=4)

    def roi(x, y, w, h, col, tag):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fill=False, ec=col, lw=2.2, ls=(0, (6, 4)), zorder=5))
        ax.add_patch(FancyBboxPatch((x + 6, y - 19), 10 * len(tag) + 8, 22, boxstyle="round,pad=0,rounding_size=5", fc=col, ec="none", zorder=6))
        ax.text(x + 11, y - 8, tag, ha="left", va="center", fontsize=10, color="white", fontweight="700", zorder=7)
    roi(96, 120, 470, 235, "#27AE60", "SI3: interface")
    roi(470, 150, 300, 170, "#2E86C1", "SI4: needles")
    roi(300, 168, 150, 150, "#CA6F1E", "SI5: interface (mag.)")
    cmap = LinearSegmentedColormap.from_list("ord", ["#c4ccd4", SLATE])
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto", extent=[160, 780, 432, 420], zorder=2)
    ax.add_patch(Polygon([(780, 414), (812, 426), (780, 438)], closed=True, fc=SLATE, ec="none", zorder=3))
    ax.text(160, 452, "less ordered", ha="left", va="center", fontsize=11, color="#5b6770")
    ax.text(780, 452, "highly ordered", ha="right", va="center", fontsize=11, color="#21557f", fontweight="600")
    ax.text(470, 412, "increasing structural order", ha="center", va="center", fontsize=12, fontweight="600", color=NAVY)


g3 = grains("SI3"); g5 = grains("SI5"); g4 = grains("SI4")
STAGES = [("Less-ordered matrix\n(lightly crystalline, SI3)", pick(g3, True), "smooth halo, faint order"),
          ("Crystallization front (SI5)", pick(g5, False), "ring breaking into spots"),
          ("Highly-ordered needle (SI4)", pick(g4, False), "discrete α Bragg spots")]
print("stage spottiness:", [(s[0].split(chr(10))[0], round(s[1]["spot"], 2), "g%d" % s[1]["g"]) for s in STAGES], flush=True)

fig = Figure(figsize=(12.5, 9.6), facecolor="white")
gs = fig.add_gridspec(3, 3, height_ratios=[1.45, 1.5, 0.32], hspace=0.3, wspace=0.12)
_axsch = fig.add_subplot(gs[0, :]); draw_scheme(_axsch)
_axsch.text(0.008, 0.96, "a", transform=_axsch.transAxes, fontsize=16, fontweight="bold", va="top",
            ha="left", color="black", bbox=dict(boxstyle="round,pad=0.16", fc="white", alpha=0.65, ec="none"), zorder=30)
for ci, (label, mm, plain) in enumerate(STAGES):
    cyx = mm["cyx"]; cr = slice(int(cyx) - 145, int(cyx) + 145)
    d, pc = disp(mm["avg"], cyx, mm["beam"], cr)
    ax = fig.add_subplot(gs[1, ci]); ax.imshow(d, cmap="inferno"); ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.04, 0.95, "bcd"[ci], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top",
            ha="left", color="white", bbox=dict(boxstyle="round,pad=0.16", fc="black", alpha=0.55, ec="none"), zorder=30)
    ax.add_patch(Circle((pc, pc), mm["rstar"], fill=False, ec="#39FF14", lw=1.1, ls=(0, (5, 3))))
    ax.set_title(label, fontsize=10.5, fontweight="bold", color=NAVY)
    ax.set_xlabel(f"{plain}  (spottiness {mm['spot']:.1f})", fontsize=8.5)
    if ci == 0:
        ax.annotate("central beam\n(blocked)", xy=(pc, pc), xytext=(pc, pc - 95), ha="center", fontsize=7,
                    color="white", arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
        ax.annotate(f"α ring d≈{1.0/(mm['rstar']*INV):.1f}Å", xy=(pc + mm["rstar"] * 0.7, pc - mm["rstar"] * 0.7),
                    xytext=(pc + 95, pc - 122), ha="center", fontsize=7, color="#39FF14",
                    arrowprops=dict(arrowstyle="->", color="#39FF14", lw=0.8))
    if ci == 2:
        ax.annotate("discrete\nα spots", xy=(pc + mm["rstar"] * 0.6, pc + mm["rstar"] * 0.55), xytext=(pc + 58, pc + 122),
                    ha="center", fontsize=7, color="#FFD24D", arrowprops=dict(arrowstyle="->", color="#FFD24D", lw=0.9))
    axs = fig.add_subplot(gs[2, ci])
    axs.imshow(unroll(mm["avg"], cyx, mm["rstar"]), cmap="inferno", aspect="auto", vmin=0, vmax=2.5, extent=[0, 360, 0, 1])
    axs.set_yticks([]); axs.tick_params(labelsize=6); axs.set_xlabel("intensity around the α ring (azimuth, deg)", fontsize=7)

fig.suptitle("What is happening to IMC: a continuous gain of structural order\n"
             "Top: less-ordered matrix → highly-ordered needles, with the three fields of view marked. "
             "Bottom: the α ring (green) goes from a smooth halo to discrete Bragg spots; the strip is that ring unrolled vs angle (flat = ring, dots = spots).",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_crystallization_schematic.png"); fig.savefig(p, dpi=170, facecolor="white")
import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "imc_crystallization_schematic.png"))
print("wrote imc_crystallization_schematic.png", flush=True)


# ===== AM-Communication split: scheme alone (-> main Fig 4) and the diffraction
# order-axis example alone (-> Supporting Information) =====
def _save(fig_, name):
    FigureCanvasAgg(fig_)
    fig_.savefig(os.path.join(OUT, name), dpi=170, facecolor="white")
    shutil.copy(os.path.join(OUT, name), os.path.join(REVIEW, name))
    print(f"wrote {name}", flush=True)

# (1) scheme only
figS = Figure(figsize=(10.5, 5.0), facecolor="white")
axS = figS.add_subplot(111); draw_scheme(axS)
figS.tight_layout(rect=[0, 0, 1, 1]); _save(figS, "imc_scheme_only.png")

# (2) diffraction order-axis example only (matrix -> front -> needle), labelled a/b/c
figE = Figure(figsize=(11.5, 4.7), facecolor="white")
gsE = figE.add_gridspec(2, 3, height_ratios=[1.5, 0.34], hspace=0.32, wspace=0.12)
for ci, (label, mm, plain) in enumerate(STAGES):
    cyx = mm["cyx"]; cr = slice(int(cyx) - 145, int(cyx) + 145)
    d, pc = disp(mm["avg"], cyx, mm["beam"], cr)
    ax = figE.add_subplot(gsE[0, ci]); ax.imshow(d, cmap="inferno"); ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.04, 0.95, "abc"[ci], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top",
            ha="left", color="white", bbox=dict(boxstyle="round,pad=0.16", fc="black", alpha=0.55, ec="none"), zorder=30)
    ax.add_patch(Circle((pc, pc), mm["rstar"], fill=False, ec="#39FF14", lw=1.1, ls=(0, (5, 3))))
    ax.set_title(label, fontsize=10.5, fontweight="bold", color=NAVY)
    ax.set_xlabel(f"{plain}  (spottiness {mm['spot']:.1f})", fontsize=8.5)
    if ci == 0:
        ax.annotate("central beam\n(blocked)", xy=(pc, pc), xytext=(pc, pc - 95), ha="center", fontsize=7,
                    color="white", arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
        ax.annotate(f"α ring d≈{1.0/(mm['rstar']*INV):.1f}Å", xy=(pc + mm["rstar"] * 0.7, pc - mm["rstar"] * 0.7),
                    xytext=(pc + 95, pc - 122), ha="center", fontsize=7, color="#39FF14",
                    arrowprops=dict(arrowstyle="->", color="#39FF14", lw=0.8))
    if ci == 2:
        ax.annotate("discrete\nα spots", xy=(pc + mm["rstar"] * 0.6, pc + mm["rstar"] * 0.55), xytext=(pc + 58, pc + 122),
                    ha="center", fontsize=7, color="#FFD24D", arrowprops=dict(arrowstyle="->", color="#FFD24D", lw=0.9))
    axs = figE.add_subplot(gsE[1, ci])
    axs.imshow(unroll(mm["avg"], cyx, mm["rstar"]), cmap="inferno", aspect="auto", vmin=0, vmax=2.5, extent=[0, 360, 0, 1])
    axs.set_yticks([]); axs.tick_params(labelsize=6); axs.set_xlabel("intensity around the α ring (azimuth, deg)", fontsize=7)
figE.tight_layout(rect=[0, 0, 1, 1]); _save(figE, "imc_order_axis_example.png")
