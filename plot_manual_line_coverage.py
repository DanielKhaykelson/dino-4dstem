"""plot_manual_line_coverage.py -- final coverage plot from the manual
line-labels (line_labels_summary.csv) per family. Two rows (NaPHI top,
MgNaPHI bottom), every sample plotted by name. Labels staggered onto
multiple horizontal "tracks" with leader lines so they never overlap.
SI-007/008/009 trio highlighted as the outlier.
"""
from __future__ import annotations
import os, csv, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.join("runs", "_per_family_v5")
NAPHI_CSV = os.path.join(ROOT, "NaPHI_combined_K8_30ep",
                            "line_labels_summary.csv")
MGNAPHI_CSV = os.path.join(ROOT, "MgNaPHI_combined_K8_30ep",
                              "line_labels_summary.csv")

NAPHI_COLOR = "#2166AC"
NAPHI_TRAIN = "#5292BC"
MGNAPHI_COLOR = "#B2182B"
MGNAPHI_TRAIN = "#D17A8B"
OUTLIER_COLOR = "#F4A582"

NAPHI_TRAIN_SAMPLES = {"NaPHI_Nadja_SI003", "NaPHI_Nadja_SI004"}
MGNAPHI_TRAIN_SAMPLES = {"MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI011"}
OUTLIER_TRIO = {"MgNaPHI_remeas_SI007",
                 "MgNaPHI_remeas_SI008",
                 "MgNaPHI_remeas_SI009"}


def _read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("sample"): continue
            rows.append({"sample": r["sample"],
                          "total": int(r["total"]),
                          "coverage": float(r["coverage"])})
    return rows


def _short(name):
    return (name.replace("MgNaPHI_remeas_", "")
                  .replace("NaPHI_Nadja_", ""))


def _assign_tracks(rows, min_sep=0.05, n_tracks=6):
    """Greedy: sort by x; place each label on the lowest track whose last x
    is at least `min_sep` away. Returns list parallel to rows with each item's
    track index (0 = closest to row baseline)."""
    sorted_idx = sorted(range(len(rows)), key=lambda i: rows[i]["coverage"])
    tracks_last_x = [-np.inf] * n_tracks
    out = [None] * len(rows)
    for i in sorted_idx:
        x = rows[i]["coverage"]
        placed = False
        for t in range(n_tracks):
            if x - tracks_last_x[t] >= min_sep:
                out[i] = t
                tracks_last_x[t] = x
                placed = True
                break
        if not placed:
            # too crowded; reuse track 0 anyway
            out[i] = 0
            tracks_last_x[0] = x
    return out


def main():
    naphi = _read(NAPHI_CSV)
    mgnaphi = _read(MGNAPHI_CSV)

    # tracks (above NaPHI baseline, below MgNaPHI baseline)
    naphi_tracks = _assign_tracks(naphi, min_sep=0.06, n_tracks=5)
    mgnaphi_tracks = _assign_tracks(mgnaphi, min_sep=0.06, n_tracks=5)

    NAPHI_Y = 1.0
    MGNAPHI_Y = 0.0
    TRACK_STEP = 0.16
    BASE_OFFSET = 0.18

    fig, ax = plt.subplots(figsize=(13.5, 6.0))

    # baselines for visual reference
    ax.axhline(NAPHI_Y, color="lightgrey", lw=0.6, zorder=0)
    ax.axhline(MGNAPHI_Y, color="lightgrey", lw=0.6, zorder=0)

    # gap shading (MgNaPHI non-outlier max -> outlier trio min)
    mg_other = [r["coverage"] for r in mgnaphi
                 if r["sample"] not in OUTLIER_TRIO]
    mg_out = [r["coverage"] for r in mgnaphi
                if r["sample"] in OUTLIER_TRIO]
    if mg_other and mg_out:
        gap_lo, gap_hi = max(mg_other), min(mg_out)
        if gap_hi > gap_lo:
            ax.axvspan(gap_lo, gap_hi, color="grey", alpha=0.10, zorder=0)
            ax.text((gap_lo + gap_hi) / 2,
                     MGNAPHI_Y - BASE_OFFSET - 5 * TRACK_STEP - 0.05,
                     f"outlier gap: {gap_lo:.2f} → {gap_hi:.2f}",
                     ha="center", va="top", fontsize=9,
                     color="dimgrey", style="italic")

    def _plot_row(rows, tracks, base_y, direction, default_color, train_color,
                    is_outlier_set=()):
        """direction = +1 for above, -1 for below."""
        for r, t in zip(rows, tracks):
            s = r["sample"]; x = r["coverage"]
            is_train = s in (NAPHI_TRAIN_SAMPLES | MGNAPHI_TRAIN_SAMPLES)
            is_outlier = s in is_outlier_set
            if is_outlier:
                color = OUTLIER_COLOR; marker = "*"; size = 320
            elif is_train:
                color = train_color; marker = "s"; size = 220
            else:
                color = default_color; marker = "o"; size = 220
            # point ON baseline
            ax.scatter([x], [base_y], s=size, color=color, marker=marker,
                        edgecolors="black", linewidths=0.7, zorder=4)
            # label position
            label_y = base_y + direction * (BASE_OFFSET + t * TRACK_STEP)
            # leader line from point to label baseline
            ax.plot([x, x], [base_y, label_y], color="grey", lw=0.5,
                     alpha=0.6, zorder=1)
            ha = "center"
            va = "bottom" if direction > 0 else "top"
            suffix = " (train)" if is_train else ""
            ax.annotate(_short(s) + suffix, (x, label_y),
                         fontsize=9, ha=ha, va=va,
                         bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                    ec="grey", lw=0.4, alpha=0.9))

    _plot_row(naphi, naphi_tracks, NAPHI_Y, +1,
                NAPHI_COLOR, NAPHI_TRAIN)
    _plot_row(mgnaphi, mgnaphi_tracks, MGNAPHI_Y, -1,
                MGNAPHI_COLOR, MGNAPHI_TRAIN, is_outlier_set=OUTLIER_TRIO)

    ax.set_yticks([MGNAPHI_Y, NAPHI_Y])
    ax.set_yticklabels(["MgNaPHI", "NaPHI"], fontsize=12)
    ax.set_xlabel("Manually-labelled line-phase coverage  "
                  "(line frames / total frames)", fontsize=11)
    ax.set_xlim(-0.02, 1.05)
    # Y range needs to fit all tracks both directions
    ax.set_ylim(MGNAPHI_Y - BASE_OFFSET - 5 * TRACK_STEP - 0.2,
                 NAPHI_Y + BASE_OFFSET + 5 * TRACK_STEP + 0.2)
    ax.set_title(
        "Line-phase coverage across NaPHI (Model N) and MgNaPHI (Model M) test sets\n"
        "Per-prototype labels assigned manually from class averages + 200 examples each "
        "(K=8, vmax=5)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NAPHI_COLOR,
                markeredgecolor="black", markersize=11, label="NaPHI test"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=NAPHI_TRAIN,
                markeredgecolor="black", markersize=11,
                label="NaPHI training sample"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MGNAPHI_COLOR,
                markeredgecolor="black", markersize=11, label="MgNaPHI test"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=MGNAPHI_TRAIN,
                markeredgecolor="black", markersize=11,
                label="MgNaPHI training sample"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=OUTLIER_COLOR,
                markeredgecolor="black", markersize=15,
                label="MgNaPHI low-Mg outlier"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left",
               framealpha=0.95)
    fig.tight_layout()

    out_png = os.path.join(ROOT, "fig_manual_line_coverage.png")
    out_pdf = os.path.join(ROOT, "fig_manual_line_coverage.pdf")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
