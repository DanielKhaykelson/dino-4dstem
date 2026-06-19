"""
generate_schematic.py — Figure S1: DINO4DSTEM architecture schematic.

Hand-drawn-style block diagram showing:
  input pattern -> polar transform -> student/teacher split ->
  student aug vs teacher aug -> CNN encoders -> projectors ->
  (prototype head + contrastive head) -> losses.

Outputs paper_figures/figS1_schematic.png
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figures")
os.makedirs(OUT, exist_ok=True)


def box(ax, xy, wh, label, face="#e7f0fa", edge="#2c72af", fontsize=9,
         fontweight="normal"):
    x, y = xy; w, h = wh
    p = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         linewidth=1.2, edgecolor=edge, facecolor=face)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight)


def arrow(ax, xy_start, xy_end, color="black", lw=1.2, style="->",
           mutation=15, rad=0.0):
    arr = FancyArrowPatch(xy_start, xy_end, arrowstyle=style,
                            color=color, linewidth=lw,
                            connectionstyle=f"arc3,rad={rad}",
                            mutation_scale=mutation)
    ax.add_patch(arr)


fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.set_axis_off()

# Input pattern
box(ax, (0.2, 3.8), (1.5, 0.8),
     "raw diffraction\npattern  x", face="#fff", edge="#666", fontweight="bold")
arrow(ax, (1.7, 4.2), (2.4, 4.2))

# Polar transform
box(ax, (2.4, 3.8), (1.8, 0.8),
     "polar transform\n+ center-beam mask", face="#fbe9e7", edge="#c84b31")
arrow(ax, (4.2, 4.2), (4.9, 4.8))       # up to student
arrow(ax, (4.2, 4.2), (4.9, 3.6))       # down to teacher

# Student aug + encoder
box(ax, (4.9, 5.5), (2.1, 0.7),
     "student aug:\nhflip / vflip / jitter / blur", face="#e3f2fd", edge="#1a6dc3")
arrow(ax, (7.0, 5.85), (7.6, 5.85))
box(ax, (7.6, 5.5), (1.7, 0.7),
     "theta-roll\npm 180 deg", face="#fff3e0", edge="#e6821c")
arrow(ax, (9.3, 5.85), (10.0, 5.85))
box(ax, (10.0, 5.2), (0.9, 1.3),
     "ResNet\nL1\ntrunk\n(circular\ntheta pad)",
     face="#e8f5e9", edge="#2e8b57", fontsize=8)

# Teacher aug + encoder
box(ax, (4.9, 3.3), (2.1, 0.7),
     "teacher aug:\nCenterCrop only", face="#e3f2fd", edge="#1a6dc3")
arrow(ax, (7.0, 3.65), (7.6, 3.65))
box(ax, (7.6, 3.3), (1.7, 0.7),
     "theta-roll\npm 15 deg", face="#fff3e0", edge="#e6821c")
arrow(ax, (9.3, 3.65), (10.0, 3.65))
box(ax, (10.0, 2.4), (0.9, 1.3),
     "ResNet\nL1\n(EMA-\nslow)",
     face="#fff8e1", edge="#b98900", fontsize=8)

# Arrows down from trunks
arrow(ax, (10.45, 5.2), (10.45, 4.3))

# Post-encoder heads
# Projector + prototype head: middle row
box(ax, (4.0, 1.4), (2.2, 0.8),
     "prototype head\nK=10 learnable prototypes", face="#f3e5f5", edge="#6a1b9a")
box(ax, (7.2, 1.4), (2.2, 0.8),
     "contrastive MLP\n64 -> 256 -> 128",
     face="#e0f2f1", edge="#00838f")

# Outputs
box(ax, (4.0, 0.2), (2.2, 0.8),
     "softmax over K\n(class assignment)", face="#fff", edge="#6a1b9a",
     fontweight="bold")
box(ax, (7.2, 0.2), (2.2, 0.8),
     "L2-normed 128-D\nembedding", face="#fff", edge="#00838f",
     fontweight="bold")
arrow(ax, (5.1, 1.4), (5.1, 1.0))
arrow(ax, (8.3, 1.4), (8.3, 1.0))

# Student encoder -> both heads
arrow(ax, (10.45, 5.2), (5.1, 2.2), rad=-0.3)
arrow(ax, (10.45, 5.2), (8.3, 2.2), rad=-0.3)

# Teacher encoder -> prototype head only (via no-grad projection)
arrow(ax, (10.45, 2.4), (5.1, 2.2), color="#b98900", lw=1.0, style="->",
       rad=0.3)

# Losses
box(ax, (0.4, 0.2), (2.8, 0.8),
     "L_DINO = KL(teacher || student)\n+ centering + sharpening",
     face="#f8f9d7", edge="#777", fontsize=8)
arrow(ax, (3.2, 0.6), (4.0, 0.6))

box(ax, (0.4, 1.4), (2.8, 0.8),
     "L_contrastive: cos-sim of teacher\nsoftmax rows matched to student\nembedding rows",
     face="#ffeaea", edge="#777", fontsize=7.5)
arrow(ax, (3.2, 1.8), (7.2, 1.8), rad=0.2)

# Title
ax.text(5.5, 7.5, "DINO4DSTEM — architecture schematic",
         fontsize=13, fontweight="bold", ha="center")
ax.text(5.5, 7.1,
         "Single-step classifier: polar-space patterns -> CNN trunk (circular-theta conv) "
         "-> prototype head (cluster assignment) + contrastive head (embedding structure)",
         fontsize=8, ha="center", style="italic", color="#444")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "figS1_schematic.png"), dpi=300,
             bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}/figS1_schematic.png")
