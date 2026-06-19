"""plot_model_schematic.py -- block-diagram schematic of the DINO + cluster1d
training architecture used in this paper. Produces a paper-ready figure
(PNG + PDF) at runs/_per_family_v5/fig_model_schematic.{png,pdf}.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


def box(ax, x, y, w, h, text, fc="white", ec="black", fontsize=9, lw=1.2):
    rect = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2, text=None, color="black", style="-|>",
           text_offset=(0.0, 0.05), fontsize=8, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle=style, mutation_scale=12,
                         color=color, linewidth=1.0, linestyle=ls)
    ax.add_patch(a)
    if text:
        mx = (x1 + x2) / 2 + text_offset[0]
        my = (y1 + y2) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize,
                 color=color)


def main():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15); ax.set_ylim(0, 10); ax.set_axis_off()

    # --- Input ---
    box(ax, 6.5, 9.0, 2.0, 0.7, "Raw diffraction pattern\n4D-STEM (H × W)",
        fc="#FFE7C2")

    # --- Two augmentations ---
    box(ax, 3.5, 7.7, 2.5, 0.9,
        "Student augmentation\nθ-roll  (full 360°)\n+ Gaussian noise",
        fc="#D9EAFB")
    box(ax, 9.0, 7.7, 2.5, 0.9,
        "Teacher augmentation\nθ-roll  (~30°)\n+ Gaussian noise",
        fc="#D9EAFB")
    arrow(ax, 7.0, 9.0, 4.7, 8.6)
    arrow(ax, 8.0, 9.0, 10.3, 8.6)

    # --- Pre-pipeline ---
    box(ax, 3.5, 6.5, 2.5, 0.9,
        "CenterCrop → COM\n→ Resize 192² → Polar\n→ MaskLeft 45 cols",
        fc="#EFEFEF")
    box(ax, 9.0, 6.5, 2.5, 0.9,
        "CenterCrop → COM\n→ Resize 192² → Polar\n→ MaskLeft 45 cols",
        fc="#EFEFEF")
    arrow(ax, 4.7, 7.7, 4.7, 7.4)
    arrow(ax, 10.3, 7.7, 10.3, 7.4)

    # --- Encoders ---
    box(ax, 3.5, 5.3, 2.5, 0.9,
        "Student encoder\nResNet-18 + θ-circular convs",
        fc="#C9F2C9")
    box(ax, 9.0, 5.3, 2.5, 0.9,
        "Teacher encoder\n(EMA copy of student)",
        fc="#C9F2C9", ec="#666")
    arrow(ax, 4.7, 6.5, 4.7, 6.2)
    arrow(ax, 10.3, 6.5, 10.3, 6.2)

    # --- Projectors ---
    box(ax, 3.5, 4.3, 2.5, 0.7, "Projector  →  z_s (128-d)",
        fc="#C9F2C9")
    box(ax, 9.0, 4.3, 2.5, 0.7, "Projector  →  z_t (128-d)",
        fc="#C9F2C9", ec="#666")
    arrow(ax, 4.7, 5.3, 4.7, 5.0)
    arrow(ax, 10.3, 5.3, 10.3, 5.0)

    # --- Prototypes ---
    box(ax, 3.5, 3.2, 2.5, 0.8, "Prototypes  P (K=8 × 128)\nlogits_s = z_s · Pᵀ",
        fc="#C9F2C9")
    box(ax, 9.0, 3.2, 2.5, 0.8, "Prototypes  (shared)\nlogits_t = z_t · Pᵀ",
        fc="#C9F2C9", ec="#666")
    arrow(ax, 4.7, 4.3, 4.7, 4.0)
    arrow(ax, 10.3, 4.3, 10.3, 4.0)

    # --- Centering / sharpening on teacher ---
    box(ax, 9.0, 2.0, 2.5, 0.8,
        "center (EMA running mean)\n+ sharpen (τ_t schedule)",
        fc="#FFE0B0", ec="#666")
    arrow(ax, 10.3, 3.2, 10.3, 2.8)
    arrow(ax, 9.0, 2.4, 6.0, 2.4, text="  p_t  (target)", color="#A66400",
            text_offset=(0, 0.18), fontsize=9)

    # --- DINO loss ---
    box(ax, 3.5, 2.0, 2.5, 0.8,
        "L_DINO  =  −Σ p_t · log( softmax(logits_s / τ_s) )\n(γ-confidence weighted)",
        fc="#F8E8E8")
    arrow(ax, 4.7, 3.2, 4.7, 2.8)

    # --- 1D radial profiles ---
    box(ax, 0.3, 5.0, 2.6, 1.4,
        "1D radial profile r(i)\n(precomputed)\nSAXS-style: log+poly\nbaseline subtraction",
        fc="#FFF7C2")
    arrow(ax, 2.9, 5.7, 3.5, 5.7, ls=":", text="r_i ", text_offset=(0, 0.12),
            fontsize=8)

    # --- cluster1d block ---
    box(ax, 0.3, 2.6, 2.6, 1.6,
        "L_cluster1d  =  α·L_intra + β·L_inter\n\n"
        "L_intra: pull r_i toward\n  soft centroid r̄_c  (via p_i,c)\n"
        "L_inter: push r̄_c apart  (hinge,\n  margin = 0.4)",
        fc="#FFF7C2")
    # arrows from p_s and r to cluster1d
    arrow(ax, 3.5, 3.6, 2.9, 3.6, text="p_s", text_offset=(0, 0.13),
            fontsize=8)
    # connector from r block down to cluster1d block
    arrow(ax, 1.5, 5.0, 1.5, 4.2, ls=":", text="", fontsize=8)

    # --- Total loss ---
    box(ax, 6.0, 0.6, 3.0, 0.9,
        "Total loss\nL = L_DINO + λ_1d · L_cluster1d",
        fc="#E6CDF8")
    arrow(ax, 4.7, 2.0, 6.4, 1.5)
    arrow(ax, 1.6, 2.6, 6.4, 1.5)

    # --- backprop and EMA ---
    arrow(ax, 7.5, 0.6, 4.7, 0.4, ls="--", color="#1f77b4")
    ax.text(6.1, 0.25, "backprop  (student + projector + prototypes)",
            color="#1f77b4", fontsize=8, ha="center")

    arrow(ax, 4.7, 5.7, 9.0, 5.7, ls=":", color="#888",
           text="EMA: θ_t ← m·θ_t + (1−m)·θ_s",
           text_offset=(0, 0.18), fontsize=8)

    # --- Inference path (dashed, to the right) ---
    box(ax, 12.5, 5.3, 2.3, 1.5,
        "Inference (transfer)\nForward only:\nencoder + projector\n+ argmax over P\n→ class id",
        fc="#FFFFFF", ec="#A0A0A0", lw=1.1)
    arrow(ax, 11.5, 6.0, 12.5, 6.0, ls="--", color="#777",
           text="frozen", text_offset=(0, 0.15), fontsize=8)

    fig.suptitle("DINO + cluster1d:  self-supervised 4D-STEM clustering with "
                  "physics-informed (1D radial) regulariser", fontsize=12)

    out_root = os.path.join("runs", "_per_family_v5")
    os.makedirs(out_root, exist_ok=True)
    out_png = os.path.join(out_root, "fig_model_schematic.png")
    out_pdf = os.path.join(out_root, "fig_model_schematic.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
