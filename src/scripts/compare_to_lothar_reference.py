"""compare_to_lothar_reference.py — side-by-side EuInAs class map: our new
result vs the Lothar reference (D:/DINOSR/Lothar/Chapter3Lothar/classmap_e40.png).

Usage:
    python compare_to_lothar_reference.py <run_dir>
        run_dir is e.g. dino_sr_contrastive/runs/_followup_EuInAs_K10_anticollapse/w_ent_0.1
        (must contain eval/inference.npz)

Output: <run_dir>/eval/fig_compare_lothar.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

REF = r"D:\DINOSR\Lothar\Chapter3Lothar\classmap_e40.png"


def main(run_dir: str):
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    assigns = inf["assigns"]
    K = int(inf["K_original"])
    K_act = int(np.unique(assigns).size)
    Nx, Ny = 66, 396
    class_map = assigns.reshape(Nx, Ny)

    ref_img = plt.imread(REF)

    base = plt.get_cmap("tab10").colors[:K]
    cmap = ListedColormap(base, name=f"K{K}")
    norm = BoundaryNorm(np.arange(K + 1) - 0.5, K)

    fig, axes = plt.subplots(2, 1, figsize=(13, 5),
                              gridspec_kw={"hspace": 0.25})

    axes[0].imshow(ref_img, aspect="equal", interpolation="nearest")
    axes[0].set_title(
        "Reference (Lothar) — K=7 active, clean strata, crisp interfaces",
        fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    im = axes[1].imshow(class_map, cmap=cmap, norm=norm,
                         aspect="equal", interpolation="nearest")
    label = os.path.basename(os.path.normpath(run_dir))
    axes[1].set_title(
        f"Ours: {label}  K={K} (K_active={K_act})",
        fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    cb = fig.colorbar(im, ax=axes[1], fraction=0.022, pad=0.01,
                       ticks=range(K))
    cb.set_label("class id", fontsize=9)

    out = os.path.join(run_dir, "eval", "fig_compare_lothar.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"K_active = {K_act}/{K}")
    print(f"class fractions: "
          f"{(np.bincount(assigns, minlength=K)/assigns.size).round(3).tolist()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
