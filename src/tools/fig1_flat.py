"""Figure 1 - clean flat flowchart (the style used in the original model_scheme),
made nicer and more detailed but jargon-free: plain box labels, a two-path
student/teacher stage, an EMA feedback arrow, and the two training signals shown
feeding the 'agree' step. Small grey tags carry the plain-language detail.
  python src/tools/fig1_flat.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
DEEP = "#0b5a82"; TEAL = "#1c7293"; TEAL2 = "#2e86ab"; TEAL3 = "#6fa6bf"; NAVY = "#21295c"; GOLD = "#e8a33d"; GREY = "#7c828c"
W, H = 13.6, 5.6

fig = Figure(figsize=(W, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, text, fc, tc="white", fs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.13",
                 fc=fc, ec="none", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold", zorder=4)


def arrow(x0, y0, x1, y1, color=NAVY, lw=2.2, dashed=False, ms=16, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                 ls=((0, (4, 3)) if dashed else "-"), connectionstyle=f"arc3,rad={rad}", zorder=2))


def tag(x, y, s, ha="center", color=GREY, fs=8.6):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, style="italic", color=color, zorder=4)


# --- banner ---
ax.add_patch(FancyBboxPatch((0.5, 4.75, ), 12.6, 0.55, boxstyle="round,pad=0.02,rounding_size=0.12",
             fc="#eef3f9", ec="#c2d3e6", lw=1.1, zorder=1))
ax.text(6.8, 5.02, "Self-supervised    ·    no labels    ·    no preset number of classes",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color=NAVY, zorder=2)

YC = 2.95
# --- 1. input ---
box(0.5, YC - 0.55, 1.9, 1.1, "one diffraction\npattern", DEEP)
tag(1.45, 2.15, "one per scan position")
# --- 2. two views ---
box(2.85, YC - 0.55, 1.9, 1.1, "two rotated\nviews", TEAL)
tag(3.8, 2.05, "same pattern, rotated:\norientation-independent")
arrow(2.4, YC, 2.85, YC)
# --- 3. student / teacher (two copies of one network) ---
box(5.2, 3.55, 1.95, 0.82, "student", TEAL2)
box(5.2, 2.13, 1.95, 0.82, "teacher", TEAL3)
tag(6.17, 4.6, "the same network, two copies", color="#5a6470")
arrow(4.75, YC + 0.15, 5.15, 3.9, color=TEAL, lw=1.9)
arrow(4.75, YC - 0.15, 5.15, 2.5, color=TEAL, lw=1.9)
arrow(6.17, 3.55, 6.17, 2.98, color=GREY, lw=1.7, dashed=True)      # EMA
tag(6.3, 3.25, "a slow copy\nof the student", ha="left", color="#5a6470")
# --- 4. agree / cluster ---
box(7.55, YC - 0.6, 2.2, 1.2, "make the two\nviews agree\n(group into classes)", NAVY, fs=11)
arrow(7.15, 3.9, 7.5, YC + 0.2, color=NAVY, lw=1.9)
arrow(7.15, 2.5, 7.5, YC - 0.2, color=NAVY, lw=1.9)
# --- 5. output ---
box(10.45, YC - 0.55, 2.15, 1.1, "class (domain)\nmap", GOLD, tc="#402c07")
tag(11.52, 2.1, "one class per position;\nthe number of classes emerges")
arrow(9.75, YC, 10.45, YC)

# --- training signals feeding the 'agree' step ---
ax.add_patch(FancyBboxPatch((3.55, 0.55, ), 8.5, 0.62, boxstyle="round,pad=0.02,rounding_size=0.12",
             fc="#fbf3e6", ec=GOLD, lw=1.2, zorder=1))
ax.text(7.8, 0.86, "Training signals:   the two views must agree    +    patterns in one class share the same radial fingerprint",
        ha="center", va="center", fontsize=10.3, color="#7a5a12", fontweight="bold", zorder=2)
arrow(8.65, 1.17, 8.65, YC - 0.62, color=GOLD, lw=1.8, dashed=True)

FigureCanvasAgg(fig)
for d in (OUT, REVIEW):
    fig.savefig(os.path.join(d, "fig1_scheme_v4.png"), dpi=200, facecolor="white")
print("wrote fig1_scheme_v4.png", flush=True)
