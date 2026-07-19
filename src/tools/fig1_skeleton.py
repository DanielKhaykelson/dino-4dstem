"""Figure 1 built on the user's skeleton: input pattern -> two augmented views ->
student/teacher networks -> a 'Classes' column of real class-average patterns with
assignment probabilities -> class (domain) map; the 1-D radial profile of each
pattern feeds in as a second signal, and the class-probability bars sit below.
  python src/tools/fig1_skeleton.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import rotate as nd_rotate
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib as mpl
from gui_app.crystallinity_panel import _radial_mean_var

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
REGION = "SI4"
W, H = 13.6, 7.6
INK = "#20304d"; TEAL = "#1c7293"; TEAL2 = "#2e86ab"; TEAL3 = "#6fa6bf"; NAVY = "#21295c"; GOLD = "#e8a33d"; GREY = "#7c828c"

z = np.load(os.path.join(FIGS, f"grain_acom_v2_{REGION}.npz"))
gsum, gcnt, cls = z["gsum"], z["gcnt"], z["cls"]
dino = np.load(os.path.join(FIGS, f"boris_nmf_cache_{REGION}.npz"))["dino"]


def disp(avg, crop=150):
    h = avg.shape[0]; c = (h - 1) / 2.0; beam = max(8, round(0.11 * h))
    yy, xx = np.indices(avg.shape); m = np.sqrt((yy - c) ** 2 + (xx - c) ** 2) > beam
    p = avg.astype(np.float32) * m; cr = slice(int(c) - crop, int(c) + crop)
    pm = p[m]; lo, hi = np.percentile(pm, 2), np.percentile(pm, 99.6)
    return np.log1p(np.clip(p, lo, hi) - lo)[cr, cr]


def class_avg(c):
    sel = cls == c; return gsum[sel].sum(0) / max(gcnt[sel].sum(), 1)


def radial(avg):
    c = (avg.shape[0] - 1) / 2.0; beam = max(8, round(0.11 * avg.shape[0]))
    m, _, _ = _radial_mean_var(avg, (c, c), beam_px=beam); seg = m[beam + 1:beam + 1 + 170]
    return seg / (seg.max() + 1e-9)


gpat = gsum[17] / max(gcnt[17], 1); patt = disp(gpat)
view1 = nd_rotate(patt, 18, reshape=False, order=1); view2 = nd_rotate(patt, -27, reshape=False, order=1)
cavg1 = disp(class_avg(1)); cavg2 = disp(class_avg(2))
uni = sorted(np.unique(dino).tolist())
cmap_cls = ListedColormap([mpl.colormaps.get_cmap("tab20b").resampled(len(uni))(i) for i in range(len(uni))])
dmap = np.vectorize({u: i for i, u in enumerate(uni)}.get)(dino).astype(float)

fig = Figure(figsize=(W, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def img_ax(x, y, w, h, ec=None, lw=1.3):
    a = fig.add_axes([x / W, y / H, w / W, h / H]); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        (s.set_color(ec), s.set_linewidth(lw)) if ec else s.set_visible(False)
    return a


def box(x, y, w, h, text, fc, tc="white", fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1", fc=fc, ec="none", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold", zorder=5)


def arrow(x0, y0, x1, y1, color=INK, lw=2.0, dashed=False, ms=15, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                 ls=((0, (4, 3)) if dashed else "-"), connectionstyle=f"arc3,rad={rad}", zorder=2))


def plain(x, y, s, size=10.5, w="bold", color=INK, ha="center"):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=color, ha=ha, va="center", zorder=6)


def tag(x, y, s, ha="center", color=GREY, fs=8.6):
    ax.text(x, y, s, fontsize=fs, style="italic", color=color, ha=ha, va="center", zorder=6)


# title
plain(W / 2, 7.32, "Self-supervised    ·    no labels    ·    no preset number of classes", 13.5, color=NAVY)

# 1. input
ip = img_ax(0.4, 3.35, 1.7, 1.7, ec=INK); ip.imshow(patt, cmap="inferno")
plain(1.25, 3.12, "diffraction pattern", 9.5); tag(1.25, 2.86, "one per scan position")

# 2. two views
v1 = img_ax(2.9, 4.75, 1.25, 1.25, ec=TEAL); v1.imshow(view1, cmap="inferno")
v2 = img_ax(2.9, 2.9, 1.25, 1.25, ec=TEAL); v2.imshow(view2, cmap="inferno")
arrow(2.1, 4.4, 2.85, 5.35, color=INK, lw=1.9); arrow(2.1, 4.1, 2.85, 3.5, color=INK, lw=1.9)
tag(3.52, 4.55, "two rotated views  (orientation-independent)")

# 3. student / teacher
box(4.75, 5.05, 1.7, 0.9, "neural\nnetwork", TEAL2)
box(4.75, 2.9, 1.7, 0.9, "neural\nnetwork", TEAL3)
plain(5.6, 6.12, "student", 10, color=TEAL2); plain(5.6, 2.72, "teacher", 10, color=TEAL3)
arrow(4.15, 5.37, 4.7, 5.5, color=TEAL, lw=1.9); arrow(4.15, 3.52, 4.7, 3.35, color=TEAL, lw=1.9)
arrow(5.6, 5.05, 5.6, 3.85, color=GREY, lw=1.7, dashed=True)
tag(5.78, 4.45, "a slow copy\n(moving average)", ha="left", color="#5a6470")

# 4. Classes column
CX, CY, CW, CH = 7.0, 1.55, 2.55, 5.25
ax.add_patch(FancyBboxPatch((CX, CY), CW, CH, boxstyle="round,pad=0.04,rounding_size=0.18",
             fc="#eaf4f8", ec=TEAL, lw=1.6, zorder=1))
plain(CX + CW / 2, CY + CH - 0.28, "“Classes”", 13, color=TEAL)
c1 = img_ax(7.52, 4.62, 1.5, 1.25); c1.imshow(cavg1, cmap="inferno")
plain(8.27, 6.08, "P = 0.45", 10, color=INK)
c2 = img_ax(7.52, 3.0, 1.5, 1.25); c2.imshow(cavg2, cmap="inferno")
plain(8.27, 4.46, "P = 0.13", 10, color=INK)
for yy in (2.55, 2.25, 1.95):
    ax.add_patch(Circle((8.27, yy), 0.075, fc=TEAL, ec="none", zorder=4))
# converge arrows student/teacher -> Classes
arrow(6.45, 5.4, 6.95, 4.65, color=INK, lw=1.9); arrow(6.45, 3.45, 6.95, 3.7, color=INK, lw=1.9)
tag(6.72, 4.88, "must agree", color=INK, fs=8.4)

# 5. 1-D profile feeding in
rax = img_ax(0.6, 0.7, 1.5, 1.15)
c0 = radial(class_avg(0)); rr = radial(class_avg(1))
rax.plot(c0, color=TEAL2, lw=2.0); rax.plot(rr, color=GOLD, lw=2.0)
for s in rax.spines.values(): s.set_color(GREY); s.set_linewidth(0.7); s.set_visible(True)
plain(1.35, 0.48, "radial fingerprint", 9)
arrow(1.25, 3.35, 1.25, 1.9, color=INK, lw=1.8)                        # input -> 1D
arrow(2.15, 1.28, 6.95, 2.15, color=INK, lw=1.8, rad=0.04)            # 1D -> Classes

# 6. class-probability bars
bx = img_ax(7.55, 0.45, 1.45, 0.85)
kv = np.array([0.45, 0.13, 0.22, 0.08, 0.12])
bx.bar(range(len(kv)), kv, color=TEAL2, width=0.75)
arrow(8.27, 1.5, 8.27, 1.35, color=INK, lw=1.6)
tag(8.27, 0.26, "class probabilities")

# 7. output class map
cm = img_ax(10.3, 3.05, 2.35, 2.35, ec=INK); cm.imshow(dmap, cmap=cmap_cls, interpolation="nearest")
arrow(9.6, 4.2, 10.25, 4.2, color=INK, lw=2.1)
plain(11.47, 5.62, "class (domain) map", 10.5)
tag(11.47, 2.82, "one class per position;\nthe number of classes emerges")

FigureCanvasAgg(fig)
for d in (OUT, REVIEW):
    fig.savefig(os.path.join(d, "fig1_scheme_v5.png"), dpi=195, facecolor="white")
print("wrote fig1_scheme_v5.png", flush=True)
