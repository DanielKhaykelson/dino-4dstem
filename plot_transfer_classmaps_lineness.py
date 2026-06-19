"""plot_transfer_classmaps_lineness.py -- recolor every transfer class map
by per-prototype line score (0 = no lines, 1 = lines, partial = pct/100)
using line_labels.json from each family's manual labelling session.

Outputs:
    runs/_per_family_v5/NaPHI_combined_K8_30ep/fig_all_transfer_classmaps_lineness.png
    runs/_per_family_v5/MgNaPHI_combined_K8_30ep/fig_all_transfer_classmaps_lineness.png
    runs/_per_family_v5/fig_all_transfer_classmaps_lineness_combined.png
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from data import SAMPLES

ROOT = os.path.join("runs", "_per_family_v5")
NAPHI_DIR = os.path.join(ROOT, "NaPHI_combined_K8_30ep")
MGNAPHI_DIR = os.path.join(ROOT, "MgNaPHI_combined_K8_30ep")

NAPHI_TEST = ["NaPHI_Nadja_SI003", "NaPHI_Nadja_SI004",
                "NaPHI_Nadja_SI005", "NaPHI_Nadja_SI006",
                "NaPHI_Nadja_SI007", "NaPHI_Nadja_SI008",
                "NaPHI_Nadja_SI009", "NaPHI_Nadja_SI010"]
MGNAPHI_TEST = ["MgNaPHI_remeas_SI001", "MgNaPHI_remeas_SI003",
                  "MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI005",
                  "MgNaPHI_remeas_SI006", "MgNaPHI_remeas_SI007",
                  "MgNaPHI_remeas_SI008", "MgNaPHI_remeas_SI009",
                  "MgNaPHI_remeas_SI010", "MgNaPHI_remeas_SI011"]

CMAP = "viridis"   # 0 = dark, 1 = bright
VMIN, VMAX = 0.0, 1.0


def _short(s):
    return (s.replace("MgNaPHI_remeas_", "")
                .replace("NaPHI_Nadja_", ""))


def _load_labels(family_dir):
    """Return dict {(sample, proto): line_score in [0,1]}."""
    p = os.path.join(family_dir, "line_labels.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    out = {}
    for row in d.get("labels", []):
        sample = row["sample"]; proto = int(row["proto"])
        lab = row.get("label")
        if lab == "lines":     score = 1.0
        elif lab == "nolines": score = 0.0
        elif lab == "partial": score = float(row.get("pct", 50)) / 100.0
        else:                   score = np.nan
        out[(sample, proto)] = score
    return out


def _make_lineness_map(family_dir, sample, label_dict):
    cfg = SAMPLES[sample]
    Ny, Nx = cfg["scan_shape"]
    inf_path = os.path.join(family_dir, "transfer", sample, "eval",
                              "inference.npz")
    if not os.path.exists(inf_path):
        return None
    inf = np.load(inf_path)
    assigns = inf["assigns"]
    K = inf["soft_probs"].shape[1]
    # build proto->score lookup; missing protos become NaN
    proto_score = np.full(K, np.nan, dtype=np.float32)
    for c in range(K):
        proto_score[c] = label_dict.get((sample, c), np.nan)
    # map per-pattern
    pixel_score = proto_score[assigns].reshape(Ny, Nx)
    return pixel_score


def _plot_family(family_dir, samples, ncols, out_path, title):
    labels = _load_labels(family_dir)
    n = len(samples)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows),
                              squeeze=False)
    last_im = None
    for idx, sample in enumerate(samples):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        m = _make_lineness_map(family_dir, sample, labels)
        if m is None:
            ax.text(0.5, 0.5, f"missing\n{sample}",
                     ha="center", va="center")
            ax.set_axis_off()
            continue
        last_im = ax.imshow(m, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                              aspect="equal", interpolation="nearest")
        cov = np.nanmean(m)
        ax.set_title(f"{_short(sample)}\nmean line score = {cov:.3f}",
                      fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_axis_off()
    if last_im is not None:
        cb_ax = fig.add_axes([0.92, 0.18, 0.015, 0.65])
        cbar = fig.colorbar(last_im, cax=cb_ax)
        cbar.set_label("per-pixel line score (0 = no lines, 1 = lines)",
                        fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.05,
                         wspace=0.10, hspace=0.30)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def _plot_combined(out_path):
    """Stack NaPHI (top) and MgNaPHI (bottom) on one figure with a single
    shared colorbar; no other labels except a thin family annotation."""
    naphi_labels = _load_labels(NAPHI_DIR)
    mgnaphi_labels = _load_labels(MGNAPHI_DIR)

    # NaPHI: 8 samples in 2 rows of 4. MgNaPHI: 10 samples in 2 rows of 5.
    # Use a 4-row x 5-col grid; col-fill from left. NaPHI in rows 0-1,
    # MgNaPHI in rows 2-3.
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 5, left=0.03, right=0.90, top=0.95, bottom=0.03,
                           hspace=0.20, wspace=0.10)
    last_im = None

    # NaPHI: 8 samples in 2x4 region, occupying gs[0:2, 0:4]; rightmost col empty
    for idx, sample in enumerate(NAPHI_TEST):
        r, c = divmod(idx, 4)
        ax = fig.add_subplot(gs[r, c])
        m = _make_lineness_map(NAPHI_DIR, sample, naphi_labels)
        if m is None:
            ax.set_axis_off(); continue
        last_im = ax.imshow(m, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                              aspect="equal", interpolation="nearest")
        ax.set_title(_short(sample), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # NaPHI label on the empty right column of row 0
    lab_ax = fig.add_subplot(gs[0:2, 4])
    lab_ax.set_axis_off()
    lab_ax.text(0.5, 0.5, "NaPHI", fontsize=20, fontweight="bold",
                 ha="center", va="center", rotation=270)

    # MgNaPHI: 10 samples in 2x5 region, occupying gs[2:4, 0:5]
    for idx, sample in enumerate(MGNAPHI_TEST):
        r, c = divmod(idx, 5)
        ax = fig.add_subplot(gs[2 + r, c])
        m = _make_lineness_map(MGNAPHI_DIR, sample, mgnaphi_labels)
        if m is None:
            ax.set_axis_off(); continue
        last_im = ax.imshow(m, cmap=CMAP, vmin=VMIN, vmax=VMAX,
                              aspect="equal", interpolation="nearest")
        ax.set_title(_short(sample), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # MgNaPHI side label
    fig.text(0.92, 0.27, "MgNaPHI", fontsize=20, fontweight="bold",
              ha="center", va="center", rotation=270)

    # divider line between families
    fig.add_artist(plt.Line2D([0.04, 0.89], [0.5, 0.5],
                                color="black", lw=1.2,
                                transform=fig.transFigure))

    # shared colorbar on the far right
    if last_im is not None:
        cb_ax = fig.add_axes([0.94, 0.10, 0.012, 0.80])
        cbar = fig.colorbar(last_im, cax=cb_ax)
        cbar.set_label("per-pixel line score (0 = no lines, 1 = lines)",
                        fontsize=11)

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def main():
    _plot_family(NAPHI_DIR, NAPHI_TEST, ncols=4,
                  out_path=os.path.join(NAPHI_DIR,
                                         "fig_all_transfer_classmaps_lineness.png"),
                  title="NaPHI test set — per-pixel line score (0 = no lines, 1 = lines)")
    _plot_family(MGNAPHI_DIR, MGNAPHI_TEST, ncols=5,
                  out_path=os.path.join(MGNAPHI_DIR,
                                         "fig_all_transfer_classmaps_lineness.png"),
                  title="MgNaPHI test set — per-pixel line score (0 = no lines, 1 = lines)")
    _plot_combined(os.path.join(ROOT,
                                  "fig_all_transfer_classmaps_lineness_combined.png"))


if __name__ == "__main__":
    main()
