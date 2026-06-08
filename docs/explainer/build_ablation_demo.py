"""Build a before/after demo of each input modification (ablation), on a
representative diffraction pattern, for the explainer PPTX + PDF."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from data import LoadPRZ
from gui_app import interpret_core as ic

OUT = os.path.join(os.path.dirname(__file__), "figs", "ablation_demo.png")
CUBE = (r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/"
        r"EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy")   # IMC SI3

ds = LoadPRZ(CUBE, resize=192, vmax=5.0)
# pick a representative, structure-rich crystalline pattern: highest post-beam
# energy among a random-ish sample (use a fixed stride, no RNG).
H = 192; cy = cx = (H - 1) / 2.0
yy, xx = np.indices((H, H))
post = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) >= 21
best_i, best_v = 0, -1.0
for i in range(0, len(ds), 97):
    v = float(ds[i][0].numpy()[post].sum())
    if v > best_v:
        best_v, best_i = v, i
x = ds[best_i]                                  # (1,192,192) torch
print(f"using pattern index {best_i}", flush=True)

beam_r = max(8.0, round(0.11 * 192))
mods = [
    ("scattered_norm", ic._scattered_norm(beam_r),
     "normalise overall scattered intensity to a constant\n"
     "(removes ‘how much it scatters’, keeps the pattern)"),
    ("radial_only", ic._radial_only,
     "replace by its azimuthal average\n(keeps rings, removes all angular detail)"),
    ("blur (sigma=2)", ic._blur(2.0),
     "Gaussian blur\n(washes out sharp Bragg spots, keeps broad rings)"),
    ("qmask_low", ic._qmask(0.0, 0.45),
     "blank the inner / low-q band\n(removes the inner rings & spots)"),
    ("qmask_high", ic._qmask(0.45, 1.01),
     "blank the outer / high-q band\n(removes the outer rings)"),
]


def disp(t):
    a = t[0].numpy().astype(np.float32)
    return np.log1p(np.clip(a, 0, None))


orig_d = disp(x)
vmax = float(np.percentile(orig_d[post], 99.5))

nrow = len(mods)
fig = Figure(figsize=(6.2, 2.45 * nrow), facecolor="white")
for r, (name, fn, desc) in enumerate(mods):
    mod = fn(x)
    for c, (img, ttl) in enumerate([(orig_d, "Original"),
                                    (disp(mod), "Modified")]):
        ax = fig.add_subplot(nrow, 2, r * 2 + c + 1)
        ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax,
                  interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(ttl, fontsize=12, fontweight="bold")
        if c == 0:
            ax.set_ylabel(name, fontsize=11, fontweight="bold")
            ax.text(-0.06, 0.5, desc, transform=ax.transAxes, fontsize=7.5,
                    color="#333", rotation=90, va="center", ha="right")
fig.suptitle("What each input modification (ablation) does to a pattern",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0.06, 0, 1, 0.97])
FigureCanvasAgg(fig)
fig.savefig(OUT, dpi=160, facecolor="white", bbox_inches="tight")
print("wrote", OUT, flush=True)

# landscape variant for the PDF: Original + 5 modified in one row
OUT2 = os.path.join(os.path.dirname(__file__), "figs", "ablation_demo_row.png")
panels = [("Original", orig_d)] + [(n, disp(fn(x))) for n, fn, _ in mods]
figr = Figure(figsize=(2.05 * len(panels), 2.5), facecolor="white")
for c, (ttl, img) in enumerate(panels):
    ax = figr.add_subplot(1, len(panels), c + 1)
    ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax,
              interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(ttl, fontsize=10, fontweight=("bold" if c == 0 else "normal"))
figr.suptitle("The same diffraction pattern, before (Original) and after each "
              "input modification", fontsize=11, fontweight="bold")
figr.tight_layout(rect=[0, 0, 1, 0.93])
FigureCanvasAgg(figr)
figr.savefig(OUT2, dpi=170, facecolor="white", bbox_inches="tight")
print("wrote", OUT2, flush=True)
