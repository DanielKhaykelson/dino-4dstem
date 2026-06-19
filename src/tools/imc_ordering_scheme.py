"""Pretty structural scheme for IMC crystallization: a less-ordered matrix BLOB
from which highly-ordered needle crystals grow, as a continuous gain of structural
order (no 'amorphous' label). The three imaged fields of view (SI3 overview, SI4
needles, SI5 interface) are highlighted as labelled ROI boxes. Saved as PNG.

  python tools/imc_ordering_scheme.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as mtrans

OUT = "docs/explainer/figs"; REVIEW = "docs/paper/draft_v2/figs/latest_review"
NAVY = "#21295C"; SLATE = "#2E5E8C"; GREY = "#9aa7b3"
fig = Figure(figsize=(9.8, 4.6), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 980); ax.set_ylim(470, 0); ax.axis("off")

ax.text(490, 30, "Crystallization of the IMC film: a continuous gain of structural order",
        ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

# ---- less-ordered matrix as a BLOB ----
bcx, bcy, brx, bry = 250, 230, 165, 120
th = np.linspace(0, 2 * np.pi, 80)
r = 1 + 0.10 * np.cos(3 * th + 0.6) + 0.06 * np.cos(5 * th + 1.3) + 0.04 * np.cos(2 * th)
bx = bcx + brx * r * np.cos(th); by = bcy + bry * r * np.sin(th)
ax.add_patch(Polygon(np.column_stack([bx, by]), closed=True, fc="#e3e9ef", ec="#b3bdc7", lw=1.6, zorder=1))
# faint internal speckle = less ordered
rng = np.random.RandomState(5)
for _ in range(34):
    a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0, 0.82)
    ax.scatter(bcx + brx * rr * np.cos(a), bcy + bry * rr * np.sin(a), s=12, color=GREY, alpha=0.45, linewidths=0, zorder=2)

# ---- needles growing from the blob's right flank, longer + sharper to the right ----
nx, ny = 355, 232
specs = [(-24, 150, "#5a7d9e", 0.55, 5), (-14, 300, "#3e6a90", 0.72, 5.5),
         (-5, 392, SLATE, 0.88, 6), (4, 410, "#244e78", 0.96, 6.5),
         (13, 360, SLATE, 0.88, 6), (22, 285, "#3e6a90", 0.72, 5.5),
         (30, 150, "#5a7d9e", 0.55, 5)]
for ang, L, col, al, hw in specs:
    rpatch = Rectangle((nx, ny - hw / 2), L, hw, fc=col, ec="none", alpha=al, zorder=3)
    rpatch.set_transform(mtrans.Affine2D().rotate_deg_around(nx, ny, ang) + ax.transData)
    ax.add_patch(rpatch)

# ---- soft labels ----
ax.text(150, 150, "less-ordered\nmatrix", ha="center", va="center", fontsize=12.5,
        fontweight="600", color="#5b6770", zorder=4)
ax.text(760, 120, "highly-ordered\nneedles", ha="center", va="center", fontsize=12.5,
        fontweight="600", color="#21557f", zorder=4)

# ---- ROI highlight boxes for the three fields of view ----
def roi(x, y, w, h, col, tag):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                fill=False, ec=col, lw=2.2, ls=(0, (6, 4)), zorder=5))
    ax.add_patch(FancyBboxPatch((x + 6, y - 19), 11 * len(tag) + 10, 22,
                                boxstyle="round,pad=0,rounding_size=5", fc=col, ec="none", zorder=6))
    ax.text(x + 11, y - 8, tag, ha="left", va="center", fontsize=10.5, color="white",
            fontweight="700", zorder=7)
roi(96, 120, 470, 235, "#27AE60", "SI3: overview")
roi(470, 150, 300, 170, "#2E86C1", "SI4: needles")
roi(300, 168, 150, 150, "#CA6F1E", "SI5: interface")

# ---- order gradient arrow ----
cmap = LinearSegmentedColormap.from_list("ord", ["#c4ccd4", SLATE])
ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto", extent=[160, 780, 432, 420], zorder=2)
ax.add_patch(Polygon([(780, 414), (812, 426), (780, 438)], closed=True, fc=SLATE, ec="none", zorder=3))
ax.text(160, 452, "less ordered", ha="left", va="center", fontsize=12, color="#5b6770")
ax.text(780, 452, "highly ordered", ha="right", va="center", fontsize=12, color="#21557f", fontweight="600")
ax.text(470, 412, "increasing structural order", ha="center", va="center", fontsize=12.5, fontweight="600", color=NAVY)

FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_ordering_scheme.png"); fig.savefig(p, dpi=200, facecolor="white")
import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "imc_ordering_scheme.png"))
print("wrote imc_ordering_scheme.png", flush=True)
