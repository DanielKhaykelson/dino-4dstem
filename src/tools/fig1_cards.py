"""Figure 1 - professional soft-card schematic matching the requested layout:
input diffraction pattern -> two rotated views -> student/teacher networks ->
'Classes' column (real class-average patterns + probabilities) -> domain map, with
the radial fingerprint feeding in and class-probability bars below. Real data,
grayscale diffraction, clean labels, drop-shadowed rounded cards.
  python src/tools/fig1_cards.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import rotate as nd_rotate
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
import matplotlib as mpl
from gui_app.crystallinity_panel import _radial_mean_var

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
W, H = 14.0, 7.7
INK = "#26324a"; TEAL = "#2b7a99"; TEAL2 = "#3a8fb0"; ACC = "#e0912f"; EDGE = "#d3dae3"; GREY = "#7a828e"

z = np.load(os.path.join(FIGS, "grain_acom_v2_SI4.npz")); gsum, gcnt, cls = z["gsum"], z["gcnt"], z["cls"]
dino = np.load(os.path.join(FIGS, "boris_nmf_cache_SI4.npz"))["dino"]


def disp(avg, crop=150):
    h = avg.shape[0]; c = (h - 1) / 2.0; beam = max(8, round(0.11 * h))
    yy, xx = np.indices(avg.shape); m = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) > beam
    p = avg.astype(np.float32) * m; cr = slice(int(c) - crop, int(c) + crop)
    pm = p[m]; lo, hi = np.percentile(pm, 2), np.percentile(pm, 99.6)
    return np.log1p(np.clip(p, lo, hi) - lo)[cr, cr]


def cavg(c):
    s = cls == c; return gsum[s].sum(0) / max(gcnt[s].sum(), 1)


def radial(avg):
    c = (avg.shape[0] - 1) / 2.0; beam = max(8, round(0.11 * avg.shape[0]))
    m, _, _ = _radial_mean_var(avg, (c, c), beam_px=beam); seg = m[beam + 1:beam + 1 + 170]
    return seg / (seg.max() + 1e-9)


patt = disp(gsum[17] / max(gcnt[17], 1))
view1 = nd_rotate(patt, 16, reshape=False, order=1); view2 = nd_rotate(patt, -22, reshape=False, order=1)
cA, cB, cC = disp(cavg(1)), disp(cavg(4)), disp(cavg(2))
uni = sorted(np.unique(dino).tolist())
cmap_cls = ListedColormap([mpl.colormaps.get_cmap("tab20b").resampled(len(uni))(i) for i in range(len(uni))])
dmap = np.vectorize({u: i for i, u in enumerate(uni)}.get)(dino).astype(float)

fig = Figure(figsize=(W, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
SHADOW = [pe.withSimplePatchShadow(offset=(2.5, -2.5), shadow_rgbFace=(0.55, 0.58, 0.63), alpha=0.28)]


def card(x, y, w, h, fc="white", ec=EDGE, lw=1.2, shadow=True, r=0.06):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=3)
    if shadow: p.set_path_effects(SHADOW)
    ax.add_patch(p); return p


def thumb(x, y, w, h, img, cmap="gray"):
    card(x - 0.08, y - 0.08, w + 0.16, h + 0.16, fc="white", r=0.05)
    a = fig.add_axes([x / W, y / H, w / W, h / H]); a.set_xticks([]); a.set_yticks([])
    a.imshow(img, cmap=cmap, interpolation="nearest", aspect="auto")
    for s in a.spines.values(): s.set_color("#c9d0d9"); s.set_linewidth(1.0)
    return a


def arrow(x0, y0, x1, y1, color=INK, lw=2.0, dashed=False, ms=15, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                 ls=((0, (4, 3)) if dashed else "-"), connectionstyle=f"arc3,rad={rad}", zorder=2))


def label(x, y, s, size=11, color=INK, w="bold", ha="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=w, ha=ha, va="center", zorder=6)


def sub(x, y, s, ha="center", color=GREY, fs=9):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va="center", zorder=6)


def netglyph(cx, cy):
    cols = [[-0.32, 0.22, -0.22], [-0.36, 0, 0.36], [-0.22, 0.22]]
    xs = [cx - 0.42, cx, cx + 0.42]
    pts = [[(x, cy + dy) for dy in c] for x, c in zip(xs, cols)]
    for i in range(len(pts) - 1):
        for a in pts[i]:
            for b in pts[i + 1]:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#b9c4d1", lw=0.7, zorder=4)
    for col in pts:
        for (x, y) in col:
            ax.add_patch(Circle((x, y), 0.055, fc=TEAL2, ec="white", lw=0.6, zorder=5))


label(W / 2, 7.35, "Self-supervised classification of 4D-STEM electron diffraction", 15)
sub(W / 2, 7.02, "no labels    ·    no preset number of classes", fs=10.5, color=TEAL)

MY = 4.55
# 1 input
thumb(0.55, 3.75, 1.55, 1.55, patt)
sub(1.32, 3.5, "diffraction pattern"); sub(1.32, 3.26, "one per scan position", fs=8)
# 2 two views
card(2.75, 2.95, 1.75, 3.0)
thumb(2.95, 4.4, 1.35, 1.3, view1); thumb(2.95, 3.05, 1.35, 1.3, view2)
sub(3.62, 6.2, "two rotated views", color=INK, fs=10); sub(3.62, 2.72, "orientation-independent", fs=8)
arrow(2.15, MY, 2.7, MY)
# 3 student / teacher
card(5.0, 4.55, 1.85, 1.15); netglyph(5.7, 5.12); label(5.92, 5.5, "student", 10, TEAL2)
card(5.0, 2.9, 1.85, 1.15); netglyph(5.7, 3.47); label(5.92, 2.72, "teacher", 10, TEAL2)
arrow(4.55, MY + 0.25, 4.98, 5.12, color=TEAL, lw=1.8); arrow(4.55, MY - 0.25, 4.98, 3.47, color=TEAL, lw=1.8)
arrow(5.92, 4.55, 5.92, 4.05, color=GREY, lw=1.6, dashed=True)
sub(6.15, 4.3, "a slow copy", ha="left", fs=8.2)
# 4 Classes
card(7.35, 1.55, 2.25, 4.5, fc="#f4f8fb", ec=TEAL, lw=1.4)
label(8.47, 5.72, "Classes", 12.5, TEAL)
thumb(7.7, 4.55, 1.05, 1.05, cA); label(9.15, 5.08, "P = 0.45", 10)
thumb(7.7, 3.3, 1.05, 1.05, cB); label(9.15, 3.83, "P = 0.22", 10)
thumb(7.7, 2.05, 1.05, 1.05, cC); label(9.15, 2.58, "P = 0.13", 10)
arrow(6.9, 5.05, 7.32, 4.75, color=INK, lw=1.8); arrow(6.9, 3.4, 7.32, 3.7, color=INK, lw=1.8)
sub(7.08, 5.25, "must agree", color=INK, fs=8.4)
# 5 domain map
thumb(10.35, 3.35, 2.4, 2.4, dmap, cmap=cmap_cls)
sub(11.55, 5.95, "class (domain) map", color=INK); sub(11.55, 3.08, "one class per position", fs=8)
sub(11.55, 2.82, "number of classes emerges", fs=8)
arrow(9.65, MY, 10.28, MY, lw=2.1)
# radial fingerprint (feeds classes)
rc = fig.add_axes([0.62 / W, 0.6 / H, 1.55 / W, 1.15 / H]); rc.set_xticks([]); rc.set_yticks([])
rc.plot(radial(cavg(0)), color=TEAL2, lw=2.0); rc.plot(radial(cavg(1)), color=ACC, lw=2.0)
for s in rc.spines.values(): s.set_color("#c9d0d9")
card(0.5, 0.5, 1.8, 1.35, shadow=True); rc.set_zorder(5)
sub(1.4, 0.38, "radial fingerprint of each pattern", fs=8.6)
arrow(1.32, 3.7, 1.32, 1.9, color=INK, lw=1.7)
arrow(2.35, 1.2, 7.33, 2.2, color=INK, lw=1.7, rad=0.05)
# class probabilities
pc = fig.add_axes([7.7 / W, 0.65 / H, 1.5 / W, 0.75 / H]); pc.set_xticks([]); pc.set_yticks([])
pc.bar(range(5), [0.45, 0.22, 0.13, 0.12, 0.08], color=TEAL2, width=0.72)
for s in pc.spines.values(): s.set_visible(False)
card(7.5, 0.5, 1.95, 1.15, shadow=True); pc.set_zorder(5)
sub(8.47, 0.36, "class probabilities", fs=8.6)
arrow(8.47, 1.5, 8.47, 1.68, color=INK, lw=1.5)

FigureCanvasAgg(fig)
for d in (OUT, REVIEW):
    fig.savefig(os.path.join(d, "fig1_cards.png"), dpi=200, facecolor="white")
print("wrote fig1_cards.png", flush=True)
