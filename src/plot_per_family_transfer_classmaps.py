"""plot_per_family_transfer_classmaps.py -- one multi-panel figure per
family (NaPHI, MgNaPHI) showing every test sample's class map under that
family's transfer model, with sample name + manually-labelled line coverage.

Output:
    runs/_per_family_v5/NaPHI_combined_K8_30ep/fig_all_transfer_classmaps.png
    runs/_per_family_v5/MgNaPHI_combined_K8_30ep/fig_all_transfer_classmaps.png
"""
from __future__ import annotations
import os, sys, csv, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from data import SAMPLES

ROOT = os.path.join("runs", "_per_family_v5")
K = 8
# fixed cmap so the same prototype id is the same color across samples
BASE_CMAP = list(plt.get_cmap("tab10").colors[:K])
CMAP = ListedColormap(BASE_CMAP, name=f"K{K}")
NORM = BoundaryNorm(np.arange(K + 1) - 0.5, K)

NAPHI_DIR = os.path.join(ROOT, "NaPHI_combined_K8_30ep")
MGNAPHI_DIR = os.path.join(ROOT, "MgNaPHI_combined_K8_30ep")


def _read_coverage(family_dir):
    """Return dict {sample: coverage} from line_labels_summary.csv."""
    p = os.path.join(family_dir, "line_labels_summary.csv")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("sample"):
                out[r["sample"]] = float(r["coverage"])
    return out


def _short(s):
    return (s.replace("MgNaPHI_remeas_", "")
                .replace("NaPHI_Nadja_", ""))


def _render_family(family_dir, samples, ncols, out_path,
                    title_prefix):
    coverage = _read_coverage(family_dir)
    n = len(samples)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows),
                              squeeze=False)
    for idx, sample in enumerate(samples):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        cfg = SAMPLES[sample]
        Ny, Nx = cfg["scan_shape"]
        inf_path = os.path.join(family_dir, "transfer", sample, "eval",
                                  "inference.npz")
        if not os.path.exists(inf_path):
            ax.text(0.5, 0.5, f"missing\n{sample}", ha="center", va="center")
            ax.set_axis_off()
            continue
        inf = np.load(inf_path)
        assigns = inf["assigns"]
        cm = assigns.reshape(Ny, Nx)
        ax.imshow(cm, cmap=CMAP, norm=NORM, aspect="equal",
                   interpolation="nearest")
        cov = coverage.get(sample)
        title = _short(sample)
        if cov is not None:
            title += f"\nline cov = {cov:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    # turn off any leftover axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_axis_off()

    # shared K=8 colorbar
    cb_ax = fig.add_axes([0.92, 0.18, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cb_ax, ticks=list(range(K)))
    cbar.set_label("prototype id", fontsize=10)

    fig.suptitle(f"{title_prefix} — class maps for every test sample "
                  f"(K=8, vmax=5, transfer-eval only)", fontsize=12)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.05,
                         wspace=0.10, hspace=0.30)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    naphi_test = ["NaPHI_Nadja_SI003", "NaPHI_Nadja_SI004",
                   "NaPHI_Nadja_SI005", "NaPHI_Nadja_SI006",
                   "NaPHI_Nadja_SI007", "NaPHI_Nadja_SI008",
                   "NaPHI_Nadja_SI009", "NaPHI_Nadja_SI010"]
    mgnaphi_test = ["MgNaPHI_remeas_SI001", "MgNaPHI_remeas_SI003",
                     "MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI005",
                     "MgNaPHI_remeas_SI006", "MgNaPHI_remeas_SI007",
                     "MgNaPHI_remeas_SI008", "MgNaPHI_remeas_SI009",
                     "MgNaPHI_remeas_SI010", "MgNaPHI_remeas_SI011"]

    _render_family(NAPHI_DIR, naphi_test, ncols=4,
                    out_path=os.path.join(NAPHI_DIR,
                                            "fig_all_transfer_classmaps.png"),
                    title_prefix="NaPHI (Model N transfer)")
    _render_family(MGNAPHI_DIR, mgnaphi_test, ncols=5,
                    out_path=os.path.join(MGNAPHI_DIR,
                                            "fig_all_transfer_classmaps.png"),
                    title_prefix="MgNaPHI (Model M transfer)")


if __name__ == "__main__":
    main()
