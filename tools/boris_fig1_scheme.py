"""Figure 1 (Boris edit): self-supervised two-path (student/teacher) scheme.
Generic labels only (no theta-roll / ResNet jargon -- those go in the caption).
Real diffraction pattern as input, real DINO class map as output, 'self-supervised'
shown. Saved to BorisEdits/fig1_scheme.png.
  python tools/boris_fig1_scheme.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import rotate
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap
import matplotlib as mpl

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits")
NAVY = "#21295C"; TEAL = "#1C7293"; AMBER = "#E8A33D"; INK = "#222"
os.makedirs(OUT, exist_ok=True)

# --- real diffraction input (a crystalline SI4 grain) ---
z = np.load(os.path.join(FIGS, "grain_acom_v2_SI4.npz"))
g = 57 if 57 < z["gsum"].shape[0] else int(np.argmax(z["gcnt"]))
avg = z["gsum"][g] / max(z["gcnt"][g], 1); H = int(z["H"]); cyx = (H - 1) / 2.0
beam = max(8, round(0.11 * H)); cr = slice(int(cyx) - 150, int(cyx) + 150)


def disp(a):
    p = a[cr, cr].astype(np.float32); pc = (p.shape[0] - 1) / 2.0
    yy, xx = np.indices(p.shape); mask = np.sqrt((yy - pc) ** 2 + (xx - pc) ** 2) > beam
    lo, hi = np.percentile(p[mask], 2), np.percentile(p[mask], 99.5)
    return np.log1p(np.clip(p, lo, hi) * mask - lo)


D = disp(avg)
Dv1 = rotate(D, 28, reshape=False, order=1, mode="nearest")
Dv2 = rotate(D, -19, reshape=False, order=1, mode="nearest")

# --- real class map (SI4 DINO) ---
asg = np.load("runs/_gui/IMC_SI4_m097_k60/eval/inference.npz")["assigns"].astype(int).reshape(128, 128)
uni = sorted(np.unique(asg).tolist()); lut = {u: i for i, u in enumerate(uni)}
cmap_cls = ListedColormap([mpl.colormaps.get_cmap("tab20").resampled(max(len(uni), 1))(i) for i in range(len(uni))])
clsmap = np.vectorize(lut.get)(asg).astype(float)

fig = Figure(figsize=(13, 5.0), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")

# self-supervised banner
ax.add_patch(FancyBboxPatch((0.3, 4.35), 12.4, 0.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="#EAF1FA", ec="#9bb8d6", lw=1))
ax.text(6.5, 4.6, "Self-supervised learning  ·  no labels  ·  no preset number of classes",
        ha="center", va="center", fontsize=13, fontweight="bold", color=NAVY)


def imbox(x, y, w, h, img, cmap, title=None, ec="#b9b9b9"):
    ax.imshow(img, cmap=cmap, extent=[x, x + w, y, y + h], aspect="auto", zorder=3,
              interpolation="nearest")
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fill=False, ec=ec, lw=1.2, zorder=4))
    if title: ax.text(x + w / 2, y - 0.18, title, ha="center", va="top", fontsize=10, color=INK)

def box(x, y, w, h, text, fc, fs=11, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="none", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold", zorder=4)

def arrow(x0, y0, x1, y1, color=INK, style="-|>", lw=2.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16, lw=lw, color=color, ls=ls, zorder=2))

# 1. input diffraction
imbox(0.4, 1.85, 1.5, 1.5, D, "inferno", "diffraction pattern\n(one per scan position)")
# 2. two augmented views
imbox(2.6, 2.7, 1.1, 1.1, Dv1, "inferno")
imbox(2.6, 1.15, 1.1, 1.1, Dv2, "inferno")
ax.text(3.15, 0.85, "two augmented views", ha="center", va="top", fontsize=10, color=INK)
arrow(1.95, 2.75, 2.55, 3.2); arrow(1.95, 2.45, 2.55, 1.75)
# 3. two-path encoders
box(4.4, 2.85, 2.1, 0.95, "neural-network\nencoder", TEAL); ax.text(5.45, 3.9, "student", ha="center", fontsize=9.5, color=TEAL, style="italic")
box(4.4, 1.2, 2.1, 0.95, "neural-network\nencoder", "#5a86a6"); ax.text(5.45, 1.06, "teacher", ha="center", va="top", fontsize=9.5, color="#5a86a6", style="italic")
arrow(3.75, 3.25, 4.35, 3.32); arrow(3.75, 1.7, 4.35, 1.68)
# EMA feedback student -> teacher
arrow(5.45, 2.83, 5.45, 2.17, color="#888", style="-|>", lw=1.6, ls=(0, (4, 3)))
ax.text(5.62, 2.5, "slow moving\naverage", ha="left", va="center", fontsize=8, color="#777")
# 4. cluster / prototypes
box(7.0, 1.95, 1.9, 1.1, "group by\nsimilarity\n(clusters)", AMBER, fs=10.5, tc="#3b2a08")
arrow(6.5, 3.3, 6.95, 2.75); arrow(6.5, 1.65, 6.95, 2.25)
# 5. agreement objective
ax.text(9.55, 2.5, "make the two\nviews agree", ha="center", va="center", fontsize=10, color=INK)
arrow(8.9, 2.5, 10.3, 2.5)
# 6. output class map
imbox(10.7, 1.7, 1.9, 1.9, clsmap, cmap_cls, "emergent class map")
arrow(10.55, 2.5, 10.65, 2.5)

p = os.path.join(OUT, "fig1_scheme.png"); FigureCanvasAgg(fig); fig.savefig(p, dpi=200, facecolor="white")
print(f"wrote {p}", flush=True)
