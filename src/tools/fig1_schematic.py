"""Figure 1 - DINO4DSTEM method schematic. Detailed but jargon-free, following the
conventions of ML / ML-for-science method figures: two grouped stages (Training vs
Applying the model), symmetric student/teacher branches, solid = data flow, dashed =
no-gradient / EMA, losses summed into one objective, real-data insets, plain labels
with small grey expert tags.
  python src/tools/fig1_schematic.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import rotate as nd_rotate
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import image as mpimg
import matplotlib as mpl
from gui_app.crystallinity_panel import _radial_mean_var

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
REGION = "SI4"
W, H = 14.0, 8.2
INK = "#20304d"; STU = "#1f6f8b"; TEA = "#5b8ea3"; ACC = "#e0912f"; GREY = "#8a8f98"
TRAIN_FC = "#eef3f7"; APPLY_FC = "#fbf4e8"

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
lowc = disp(class_avg(0)); highc = disp(class_avg(1))
uni = sorted(np.unique(dino).tolist())
base = mpl.colormaps.get_cmap("tab20b").resampled(len(uni))
cmap_cls = ListedColormap([base(i) for i in range(len(uni))])
dmap = np.vectorize({u: i for i, u in enumerate(uni)}.get)(dino).astype(float)

fig = Figure(figsize=(W, H), facecolor="white")
ax0 = fig.add_axes([0, 0, 1, 1]); ax0.set_xlim(0, W); ax0.set_ylim(0, H); ax0.axis("off")


def img_ax(x, y, w, h, ec=None, lw=1.2):
    a = fig.add_axes([x / W, y / H, w / W, h / H]); a.set_xticks([]); a.set_yticks([])
    if ec:
        for s in a.spines.values(): s.set_color(ec); s.set_linewidth(lw)
    else:
        for s in a.spines.values(): s.set_visible(False)
    return a


def panel(x, y, w, h, fc):
    ax0.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                  fc=fc, ec="none", zorder=0))


def box(x, y, w, h, fc, ec=INK, lw=1.6, txt="", tc=INK, fs=10):
    ax0.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.09",
                  fc=fc, ec=ec, lw=lw, zorder=3))
    if txt: ax0.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc, zorder=6)


def arrow(x0, y0, x1, y1, color=INK, lw=2.0, dashed=False, rad=0.0, ms=15, style="-|>"):
    ax0.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color,
                  lw=lw, ls=((0, (4, 3)) if dashed else "-"), connectionstyle=f"arc3,rad={rad}", zorder=4))


def plain(x, y, s, size=10.5, w="bold", color=INK, ha="center", va="center"):
    ax0.text(x, y, s, fontsize=size, fontweight=w, color=color, ha=ha, va=va, zorder=6)


def tag(x, y, s, ha="center", color=GREY):
    ax0.text(x, y, s, fontsize=7.6, style="italic", color=color, ha=ha, va="center", zorder=6)


def bars(x, y, w, h, peak, col):
    a = img_ax(x, y, w, h); K = 10
    vals = np.exp(-0.5 * ((np.arange(K) - peak) / 1.05) ** 2) + 0.05 * np.random.default_rng(peak).random(K)
    a.bar(range(K), vals, color=col, width=0.8); a.set_ylim(0, 1.15)


# ===== stage panels =====
panel(0.25, 3.0, 13.5, 4.65, TRAIN_FC)
panel(0.25, 0.30, 13.5, 2.5, APPLY_FC)
plain(0.55, 7.4, "Training", 12.5, ha="left", color=STU); tag(2.0, 7.4, "self-supervised, no labels", ha="left")
plain(0.55, 2.62, "Applying the model", 12, ha="left", color=ACC)
plain(W / 2, 8.0, "DINO4DSTEM: a label-free classifier for 4D-STEM diffraction", 14)

# ===== (a) input + augmentation =====
ap = img_ax(0.55, 5.35, 1.25, 1.25, ec=INK, lw=1.3); ap.imshow(patt, cmap="inferno")
plain(1.17, 6.78, "one diffraction\npattern", 9.5)
plain(0.6, 7.08, "a", 12, ha="left")
v1 = img_ax(2.4, 5.95, 0.88, 0.88, ec=STU); v1.imshow(view1, cmap="inferno")
v2 = img_ax(2.4, 4.85, 0.88, 0.88, ec=STU); v2.imshow(view2, cmap="inferno")
arrow(1.85, 5.98, 2.35, 6.35, color=INK, lw=1.8); arrow(1.85, 5.98, 2.35, 5.25, color=INK, lw=1.8)
ax0.add_patch(FancyArrowPatch((3.31, 6.55), (3.31, 5.45), arrowstyle="<->", mutation_scale=11, color=ACC,
              lw=1.7, connectionstyle="arc3,rad=-0.5", zorder=5))
tag(2.84, 4.6, "two rotations\n(orientation invariance)")

# ===== (b) self-distillation =====
plain(4.05, 7.08, "b", 12, ha="left")
box(3.95, 5.95, 1.45, 0.8, "#ffffff", ec=STU, txt="student", tc=STU)
box(3.95, 4.55, 1.45, 0.8, "#ffffff", ec=TEA, txt="teacher", tc=TEA)
arrow(3.28, 6.39, 3.9, 6.35, color=STU, lw=1.8); arrow(3.28, 5.29, 3.9, 4.95, color=TEA, lw=1.8)
# EMA dashed (no gradient) student -> teacher
arrow(4.68, 5.9, 4.68, 5.4, color=GREY, lw=1.7, dashed=True)
tag(4.4, 4.28, "teacher = moving average\nof student (no gradient)")
bars(5.7, 5.95, 1.15, 0.8, 4, STU); bars(5.7, 4.55, 1.15, 0.8, 4, TEA)
arrow(5.42, 6.35, 5.66, 6.35, color=STU, lw=1.6); arrow(5.42, 4.95, 5.66, 4.95, color=TEA, lw=1.6)
plain(6.28, 6.95, "class scores", 9)
tag(6.62, 4.35, "centred + sharpened")
ax0.add_patch(FancyArrowPatch((6.98, 6.3), (6.98, 4.7), arrowstyle="<->", mutation_scale=12, color=ACC,
              lw=2.2, connectionstyle="arc3,rad=0.22", zorder=5))
plain(7.55, 5.75, "agree", 10, color=ACC); tag(7.55, 5.42, "cross-entropy")

# ===== 1-D radial loss (parallel term, own lane below) =====
box(6.05, 3.15, 3.1, 1.0, "#fff6ea", ec=ACC, lw=1.4)
rax = img_ax(6.22, 3.3, 0.78, 0.72)
rng = np.random.default_rng(0); c0 = radial(class_avg(0)); c1 = radial(class_avg(1))
for bc, col in ((c0, "#3b6ea5"), (c1, ACC)):
    for k in range(3): rax.plot(bc * (1 + 0.06 * rng.standard_normal(bc.size)), color=col, lw=0.7, alpha=0.5)
    rax.plot(bc, color=col, lw=1.9)
plain(8.0, 3.9, "radial fingerprint", 9, color=INK)
tag(8.0, 3.45, "1-D radial clustering loss")

# ===== total objective + gradient feedback =====
box(10.85, 5.0, 2.35, 0.9, "#ffffff", ec=INK, txt="total training\nloss", fs=9.5)
arrow(8.15, 5.65, 10.8, 5.6, color=ACC, lw=1.7, rad=-0.06)     # agree -> loss
arrow(9.2, 3.7, 10.8, 5.1, color=ACC, lw=1.7, rad=-0.18)       # radial -> loss
arrow(12.0, 5.9, 4.7, 6.78, color=GREY, lw=1.7, dashed=True, rad=0.2)  # gradient back (high arc)
tag(8.45, 7.12, "gradient updates the student", color=GREY)

# ===== (c) applying the model =====
plain(0.62, 2.35, "c", 11, ha="left")
box(0.6, 1.15, 1.5, 0.8, "#ffffff", ec=STU, txt="trained\nnetwork", tc=STU, fs=9.5)
st = img_ax(2.5, 1.0, 0.95, 0.95); st.imshow(np.stack([lowc, highc, patt[::-1]]).mean(0), cmap="inferno")
plain(2.98, 2.2, "every pattern\nin the scan", 8.6)
arrow(2.15, 1.55, 2.45, 1.5, color=INK, lw=1.8)
arrow(3.5, 1.45, 4.1, 1.42, color=INK, lw=2.0)
cm = img_ax(4.2, 0.5, 1.7, 1.7, ec=INK, lw=1.3); cm.imshow(dmap, cmap=cmap_cls, interpolation="nearest")
plain(5.05, 2.36, "domain map", 10)
tag(5.08, 0.4, "one class per position")
arrow(6.0, 1.45, 6.5, 1.45, color=INK, lw=1.8)
o1 = img_ax(6.55, 1.55, 0.78, 0.78, ec=GREY, lw=1.0); o1.imshow(highc, cmap="inferno")
o2 = img_ax(6.55, 0.65, 0.78, 0.78, ec=GREY, lw=1.0); o2.imshow(lowc, cmap="inferno")
plain(8.5, 2.0, "each class's average pattern", 9.5, ha="left")
tag(8.5, 1.6, "averaging reveals structure below the noise of one frame", ha="left")
plain(8.5, 1.05, "classes emerge on their own", 9.5, ha="left", color=ACC)
tag(8.5, 0.7, "start from many prototypes; unused ones fade, so K is never set", ha="left")

FigureCanvasAgg(fig)
for d in (OUT, REVIEW):
    fig.savefig(os.path.join(d, "fig1_schematic_v3.png"), dpi=185, facecolor="white")
print("wrote fig1_schematic_v3.png", flush=True)
