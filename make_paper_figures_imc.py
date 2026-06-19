"""
make_paper_figures_imc.py — publication-layout multi-panel figures for the
IMC thickness-comparison story.

Fig 1 (main result):
  a) 50nm class map on scan grid, colored by prototype
  b) 150nm class map on scan grid
  c) amorphous-vs-crystalline area fraction (stacked horizontal bar)
  d) per-prototype area fraction, both samples, color-coded by crystallinity

Fig 2 (evidence):
  a) 50nm radial profiles (amorphous faded, crystalline emphasized)
  b) 150nm radial profiles (same coloring rule)
  c) class-average diffraction panels of every crystalline prototype
  d) cross-sample radial-profile cosine similarity matrix with unmatched
     150nm-unique prototypes circled

Both saved under runs/IMC_comparison/ at 300 DPI PNG (drop-in-ready for a
journal at double-column width).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SAMPLES, LoadPRZ
from analyze_imc import (
    _beam_mask, _clip_log1p_aggressive,
    class_means, radial_profile, find_peaks_simple,
    crystallinity_score, cross_match_profiles,
    STEP_NM, PROBE_NM,
)

# Publication style.
mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

XTAL_THRESHOLD = 1.5  # amorphous / crystalline split


def _load(sample: str, config: str):
    base = os.path.dirname(os.path.abspath(__file__))
    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    scan_shape = cfg["scan_shape"]
    run_dir = os.path.join(base, "runs", sample, config)
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]; assigns = inf["assigns"]
    K = int(soft_probs.shape[1])
    means = class_means(dataset, assigns, soft_probs, K, N_top=300)
    profiles = np.stack([radial_profile(m, r_min=15, n_bins=80)[0] for m in means], 0)
    centers = radial_profile(means[0], r_min=15, n_bins=80)[1]
    xtal = np.array([crystallinity_score(p) for p in profiles])
    counts = np.bincount(assigns, minlength=K)
    return dict(
        sample=sample, dataset=dataset, assigns=assigns,
        soft_probs=soft_probs, scan_shape=scan_shape,
        K=K, means=means, profiles=profiles,
        centers=centers, xtal=xtal, counts=counts,
    )


# =========================================================================
# FIGURE 1 — Main result (class maps + fractions)
# =========================================================================

def figure_1(d50, d150, out_path):
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[3.2, 0.9, 1.3],
                    hspace=0.55, wspace=0.30,
                    left=0.08, right=0.97, top=0.94, bottom=0.08)

    cmap_tab10 = plt.get_cmap("tab10")

    # Panel a) 50nm map
    ax_a = fig.add_subplot(gs[0, 0])
    m50 = d50["assigns"].reshape(d50["scan_shape"])
    im_a = ax_a.imshow(m50, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    ax_a.set_title(f"(a) 50 nm film  (K={d50['K']})", loc="left", fontweight="bold")
    ax_a.set_xticks([]); ax_a.set_yticks([])
    # scale bar: 1 µm = 1000nm / STEP_NM = 22.7 px
    scale_nm = 1000
    scale_px = scale_nm / STEP_NM
    ax_a.plot([5, 5 + scale_px], [d50["scan_shape"][0] - 6, d50["scan_shape"][0] - 6],
               color="white", lw=2.5, solid_capstyle="butt")
    ax_a.text(5 + scale_px / 2, d50["scan_shape"][0] - 9, f"{scale_nm/1000:.0f} µm",
                color="white", ha="center", va="bottom", fontsize=8,
                path_effects=[])
    # Panel b) 150nm map
    ax_b = fig.add_subplot(gs[0, 1])
    m150 = d150["assigns"].reshape(d150["scan_shape"])
    im_b = ax_b.imshow(m150, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    ax_b.set_title(f"(b) 150 nm film  (K={d150['K']})", loc="left", fontweight="bold")
    ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.plot([5, 5 + scale_px], [d150["scan_shape"][0] - 6, d150["scan_shape"][0] - 6],
                color="white", lw=2.5, solid_capstyle="butt")
    ax_b.text(5 + scale_px / 2, d150["scan_shape"][0] - 9, f"{scale_nm/1000:.0f} µm",
                color="white", ha="center", va="bottom", fontsize=8)

    # Panel c) amorphous / crystalline fraction
    ax_c = fig.add_subplot(gs[1, :])
    frac = {}
    for name, d in (("50 nm", d50), ("150 nm", d150)):
        total = d["counts"].sum()
        am = d["counts"][d["xtal"] < XTAL_THRESHOLD].sum()
        cr = total - am
        frac[name] = (am / total, cr / total)
    labels = list(frac.keys())
    am_vals = [frac[n][0] for n in labels]
    cr_vals = [frac[n][1] for n in labels]
    y = np.arange(len(labels))
    ax_c.barh(y, am_vals, color="#b0b0b0", edgecolor="black",
                height=0.55, label="amorphous")
    ax_c.barh(y, cr_vals, left=am_vals, color="#2b8cbe", edgecolor="black",
                height=0.55, label="crystalline")
    for i, (a, c) in enumerate(zip(am_vals, cr_vals)):
        ax_c.text(a / 2, i, f"{a*100:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="black")
        ax_c.text(a + c / 2, i, f"{c*100:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels)
    ax_c.set_xlim(0, 1)
    ax_c.set_xlabel("area fraction of scan")
    ax_c.set_title("(c) area fraction: amorphous (halo-only) vs crystalline (Bragg peaks present)",
                     loc="left", fontweight="bold")
    ax_c.legend(loc="lower right", frameon=False)
    ax_c.grid(alpha=0.3, axis="x")

    # Panel d) per-prototype occupancy, both samples
    ax_d = fig.add_subplot(gs[2, :])
    def _add_bars(ax, d, y_base, sample_name):
        total = d["counts"].sum()
        fracs = d["counts"] / total
        xpos = np.arange(len(fracs))
        colors = ["#2b8cbe" if x >= XTAL_THRESHOLD else "#b0b0b0"
                    for x in d["xtal"]]
        bars = ax.bar(xpos + y_base * 0.45, fracs * 100,
                        width=0.38, color=colors, edgecolor="black",
                        linewidth=0.6,
                        label=sample_name if y_base == 0 else None)
        # annotate each bar
        for i, (xp, fr) in enumerate(zip(xpos, fracs)):
            ax.text(xp + y_base * 0.45, fr * 100 + 0.5,
                     f"p{i}\n{fr*100:.1f}%", ha="center", va="bottom",
                     fontsize=6.5)
        return bars
    # Use separate x positions per sample so we can show both 8 and 10 bars.
    off = 0
    x50 = np.arange(d50["K"])
    x150 = np.arange(d50["K"] + 1, d50["K"] + 1 + d150["K"])
    fracs50 = d50["counts"] / d50["counts"].sum()
    fracs150 = d150["counts"] / d150["counts"].sum()
    colors50 = ["#2b8cbe" if x >= XTAL_THRESHOLD else "#b0b0b0" for x in d50["xtal"]]
    colors150 = ["#2b8cbe" if x >= XTAL_THRESHOLD else "#b0b0b0" for x in d150["xtal"]]
    ax_d.bar(x50, fracs50 * 100, color=colors50, edgecolor="black", linewidth=0.6)
    ax_d.bar(x150, fracs150 * 100, color=colors150, edgecolor="black", linewidth=0.6)
    # bar labels
    for i, (xp, fr) in enumerate(zip(x50, fracs50)):
        ax_d.text(xp, fr * 100 + 0.4, f"p{i}", ha="center", va="bottom", fontsize=7)
    for i, (xp, fr) in enumerate(zip(x150, fracs150)):
        ax_d.text(xp, fr * 100 + 0.4, f"p{i}", ha="center", va="bottom", fontsize=7)
    # group labels
    ax_d.axvline(d50["K"] + 0.5, color="black", lw=0.8, alpha=0.3)
    ax_d.text(d50["K"] / 2 - 0.5, ax_d.get_ylim()[1] * 1.02 if ax_d.get_ylim()[1] else 18,
                "50 nm", ha="center", fontsize=9, fontweight="bold")
    ax_d.text(d50["K"] + 0.5 + d150["K"] / 2, ax_d.get_ylim()[1] * 1.02 if ax_d.get_ylim()[1] else 18,
                "150 nm", ha="center", fontsize=9, fontweight="bold")
    ax_d.set_xticks([])
    ax_d.set_ylim(0, max(fracs50.max(), fracs150.max()) * 110)
    ax_d.set_ylabel("% of scan")
    ax_d.set_title("(d) per-prototype area fraction   (blue = crystalline, grey = amorphous halo)",
                     loc="left", fontweight="bold")
    ax_d.grid(alpha=0.3, axis="y")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# FIGURE 2 — Evidence (radial profiles + class averages + cross-match)
# =========================================================================

def figure_2(d50, d150, out_path):
    # Determine crystalline prototypes in each sample (for emphasis + class-avg panels).
    xtal50 = np.where(d50["xtal"] >= XTAL_THRESHOLD)[0]
    xtal150 = np.where(d150["xtal"] >= XTAL_THRESHOLD)[0]
    n_cr50 = len(xtal50); n_cr150 = len(xtal150)
    n_cr_cols = max(n_cr50, n_cr150, 1)
    fig = plt.figure(figsize=(7.5, 9.0))
    gs = GridSpec(4, max(n_cr_cols, 4), figure=fig,
                    height_ratios=[2.2, 1.8, 1.8, 2.4],
                    hspace=0.55, wspace=0.25,
                    left=0.08, right=0.97, top=0.95, bottom=0.06)

    cmap = plt.get_cmap("tab10")
    axp50 = fig.add_subplot(gs[0, :])
    axp150 = fig.add_subplot(gs[1, :])
    # Panel a: 50nm radial profiles
    for c in range(d50["K"]):
        prof = d50["profiles"][c]
        prof_n = prof - prof.min()
        prof_n = prof_n / (prof_n.max() + 1e-12)
        is_cr = d50["xtal"][c] >= XTAL_THRESHOLD
        color = cmap(c % 10)
        lw = 1.7 if is_cr else 0.9
        alpha = 1.0 if is_cr else 0.35
        label = f"p{c} ({'C' if is_cr else 'A'}, {d50['counts'][c]})"
        axp50.plot(d50["centers"], prof_n, color=color, lw=lw, alpha=alpha, label=label)
        if is_cr:
            peaks = find_peaks_simple(prof_n, min_sep=2, min_rel_prominence=0.03)
            for p in peaks:
                axp50.axvline(d50["centers"][p], color=color, alpha=0.2, lw=0.6)
    axp50.set_title("(a) 50 nm radial profiles — crystalline bold, amorphous faded",
                      loc="left", fontweight="bold")
    axp50.set_ylabel("normalised intensity")
    axp50.legend(ncol=4, fontsize=6.5, loc="upper right", frameon=False)
    axp50.grid(alpha=0.3)
    axp50.set_xlim(d50["centers"].min(), d50["centers"].max())

    # Panel b: 150nm radial profiles
    for c in range(d150["K"]):
        prof = d150["profiles"][c]
        prof_n = prof - prof.min()
        prof_n = prof_n / (prof_n.max() + 1e-12)
        is_cr = d150["xtal"][c] >= XTAL_THRESHOLD
        color = cmap(c % 10)
        lw = 1.7 if is_cr else 0.9
        alpha = 1.0 if is_cr else 0.35
        label = f"p{c} ({'C' if is_cr else 'A'}, {d150['counts'][c]})"
        axp150.plot(d150["centers"], prof_n, color=color, lw=lw, alpha=alpha, label=label)
        if is_cr:
            peaks = find_peaks_simple(prof_n, min_sep=2, min_rel_prominence=0.03)
            for p in peaks:
                axp150.axvline(d150["centers"][p], color=color, alpha=0.2, lw=0.6)
    axp150.set_title("(b) 150 nm radial profiles",
                       loc="left", fontweight="bold")
    axp150.set_xlabel("radius in post-resize 192px space  (scan-invariant; ∝ scattering vector q)")
    axp150.set_ylabel("normalised intensity")
    axp150.legend(ncol=4, fontsize=6.5, loc="upper right", frameon=False)
    axp150.grid(alpha=0.3)
    axp150.set_xlim(d150["centers"].min(), d150["centers"].max())

    # Panel c: crystalline class-average diffraction patterns
    H0, W0 = d50["dataset"].H, d50["dataset"].W
    bm = _beam_mask(H0, W0, radius=40)
    for j in range(n_cr_cols):
        ax = fig.add_subplot(gs[2, j])
        if j < n_cr50:
            c = xtal50[j]
            arr = d50["means"][c]
            ax.imshow(_clip_log1p_aggressive(arr, mask=bm, pct_lo=5, pct_hi=95),
                       cmap="inferno")
            ax.set_title(f"50nm p{c}\n{d50['counts'][c]/d50['counts'].sum()*100:.1f}%",
                           fontsize=7.5)
        else:
            ax.set_axis_off()
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("50 nm\ncrystalline\nprototypes",
                           fontsize=8, rotation=0, labelpad=30,
                           ha="right", va="center")
    for j in range(n_cr_cols):
        ax = fig.add_subplot(gs[3, j])
        if j < n_cr150:
            c = xtal150[j]
            arr = d150["means"][c]
            ax.imshow(_clip_log1p_aggressive(arr, mask=bm, pct_lo=5, pct_hi=95),
                       cmap="inferno")
            ax.set_title(f"150nm p{c}\n{d150['counts'][c]/d150['counts'].sum()*100:.1f}%",
                           fontsize=7.5)
        else:
            ax.set_axis_off()
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("150 nm\ncrystalline\nprototypes",
                           fontsize=8, rotation=0, labelpad=30,
                           ha="right", va="center")
    fig.text(0.02, 0.385,
              "(c) class-average diffraction patterns of crystalline prototypes, "
              "percentile-clipped for ring visibility",
              fontsize=9, fontweight="bold", ha="left")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# FIGURE 2b — standalone cross-similarity matrix, so it doesn't compete for
# space with panel (c) of Fig 2.
# =========================================================================

def figure_2b(d50, d150, out_path):
    pairs, sim = cross_match_profiles(d50["profiles"], d150["profiles"])
    matched_150 = set(p[1] for p in pairs)
    unmatched_150 = [c for c in range(d150["K"]) if c not in matched_150]

    fig = plt.figure(figsize=(6.5, 5.0))
    gs = GridSpec(1, 1, left=0.15, right=0.98, top=0.92, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("150 nm prototype"); ax.set_ylabel("50 nm prototype")
    ax.set_xticks(range(d150["K"]))
    ax.set_yticks(range(d50["K"]))
    ax.set_title("Radial-profile cosine similarity "
                   "(Hungarian-best pairs circled; 150-only marked)",
                   fontweight="bold")
    for r, c, s in pairs:
        ec = "red" if s >= 0.90 else "orange"
        ax.scatter(c, r, marker="o", s=160, facecolors="none",
                    edgecolors=ec, linewidths=1.8)
    for c in unmatched_150:
        # draw a downward arrow above the unmatched column
        ax.annotate(f"150-unique\np{c}", xy=(c, -0.4), xytext=(c, -2.0),
                     ha="center", fontsize=7, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                      color="white" if sim[i, j] < 0.7 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="cos-sim")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# Driver
# =========================================================================

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "runs", "IMC_comparison")
    os.makedirs(out_dir, exist_ok=True)

    d50 = _load("IMC_50nm_SI2",  "winner_polar_centroid")
    d150 = _load("IMC_150nm_SI5", "winner_polar_centroid")

    figure_1(d50, d150,
              os.path.join(out_dir, "paper_fig_1_IMC_main_result.png"))
    print(f"[paper_fig] wrote paper_fig_1_IMC_main_result.png")
    figure_2(d50, d150,
              os.path.join(out_dir, "paper_fig_2_IMC_fingerprints.png"))
    print(f"[paper_fig] wrote paper_fig_2_IMC_fingerprints.png")
    figure_2b(d50, d150,
               os.path.join(out_dir, "paper_fig_2b_IMC_crossmatch.png"))
    print(f"[paper_fig] wrote paper_fig_2b_IMC_crossmatch.png")
    print(f"[paper_fig] done. See {out_dir}/")


if __name__ == "__main__":
    main()
