"""plot_radial_profiles.py — visualize the 1D radial profiles that gate the
SupCon loss. Pulls a few representative patterns per class, also shows the
distribution of pairwise cosines.

Inputs (existing):
    <sample>.radial.npy           per-pattern profile (high-pass log-norm-centered)
    <sample>.gate_thresholds.json tau_pos / tau_neg

Output:
    runs/_diagnostics/<sample>_radial_diagnostics.png
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SAMPLES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "runs", "_diagnostics")
os.makedirs(OUT, exist_ok=True)


def main(sample: str):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    rad_path = base + ".radial.npy"
    th_path = base + ".gate_thresholds.json"

    rad = np.load(rad_path)        # (N, 192) high-pass log-normed centered
    th = json.load(open(th_path))
    tau_pos = th["tau_pos"]; tau_neg = th["tau_neg"]
    print(f"[{sample}] N={rad.shape[0]}, n_bins={rad.shape[1]}, "
          f"tau_pos={tau_pos:.4f}, tau_neg={tau_neg:.4f}")

    rng = np.random.default_rng(42)
    # pick a small set of "anchors" (random patterns across the scan)
    n_anchor = 12
    anchor_idx = rng.choice(rad.shape[0], size=n_anchor, replace=False)

    # cosine of all anchors against everyone
    rad_n = rad / (np.linalg.norm(rad, axis=1, keepdims=True) + 1e-12)
    sims = rad_n[anchor_idx] @ rad_n.T   # (n_anchor, N)

    # per-anchor stats
    n_pos = (sims > tau_pos).sum(axis=1) - 1  # exclude self
    n_neg = (sims < tau_neg).sum(axis=1)
    print(f"  per-anchor: avg n_pos={n_pos.mean():.0f}  n_neg={n_neg.mean():.0f}")

    fig, axes = plt.subplots(4, 1, figsize=(12, 14),
                              gridspec_kw={"height_ratios": [3, 3, 3, 2]})

    # Plot 1: 12 random radial profiles overlaid
    for i, ax_i in enumerate(anchor_idx):
        axes[0].plot(rad[ax_i], lw=0.7, alpha=0.8,
                      label=f"i={ax_i}" if i < 6 else None)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_title(f"{sample}: 12 random patterns (mean-centered log-radial, "
                       f"high-pass σ=10)", fontsize=11)
    axes[0].set_xlabel("radial bin (0=center, 191=outer)")
    axes[0].set_ylabel("log(I/sum) − baseline")
    axes[0].legend(fontsize=8, ncol=3, loc="upper right")
    axes[0].grid(alpha=0.3)

    # Plot 2: an anchor + its top-5 positives + top-5 negatives
    a = anchor_idx[0]
    sims_a = sims[0]
    pos_idx = np.argsort(-sims_a)[1:6]   # top 5, exclude self
    neg_idx = np.argsort(sims_a)[:5]     # bottom 5
    axes[1].plot(rad[a], color="black", lw=2, label=f"anchor i={a}")
    for i, ai in enumerate(pos_idx):
        axes[1].plot(rad[ai], color="C2", alpha=0.7, lw=1,
                      label=f"top-pos i={ai} cos={sims_a[ai]:.3f}" if i < 3 else None)
    for i, ai in enumerate(neg_idx):
        axes[1].plot(rad[ai], color="C3", alpha=0.7, lw=1,
                      label=f"top-neg i={ai} cos={sims_a[ai]:.3f}" if i < 3 else None)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_title(f"Anchor i={a}: top-5 positives (green) and top-5 "
                       f"negatives (red) — these are what SupCon pulls/pushes",
                       fontsize=11)
    axes[1].set_xlabel("radial bin")
    axes[1].set_ylabel("log(I/sum) − baseline")
    axes[1].legend(fontsize=8, ncol=2, loc="upper right")
    axes[1].grid(alpha=0.3)

    # Plot 3: pairwise cosine histogram on a random sample of pairs
    n_pairs = 20000
    a_idx = rng.integers(0, rad.shape[0], size=n_pairs)
    b_idx = rng.integers(0, rad.shape[0], size=n_pairs)
    keep = a_idx != b_idx
    cos_pairs = (rad_n[a_idx[keep]] * rad_n[b_idx[keep]]).sum(axis=1)
    axes[2].hist(cos_pairs, bins=80, color="steelblue", edgecolor="black",
                  alpha=0.8)
    axes[2].axvline(tau_pos, color="green", lw=2, label=f"τ_pos = {tau_pos:.4f}")
    axes[2].axvline(tau_neg, color="red", lw=2, label=f"τ_neg = {tau_neg:.4f}")
    axes[2].set_title(f"Distribution of pairwise cosines among radial profiles "
                       f"(n_pairs={n_pairs}). Region above green = positives, "
                       f"below red = negatives, between = abstain.",
                       fontsize=11)
    axes[2].set_xlabel("cosine similarity")
    axes[2].set_ylabel("count")
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3)

    # Plot 4: cumulative fraction (so we see the percentiles directly)
    sorted_cos = np.sort(cos_pairs)
    pcts = np.linspace(0, 1, len(sorted_cos))
    axes[3].plot(sorted_cos, pcts, color="black")
    axes[3].axvline(tau_pos, color="green", lw=2)
    axes[3].axvline(tau_neg, color="red", lw=2)
    axes[3].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[3].axhline(0.85, color="gray", linestyle="--", alpha=0.5)
    axes[3].set_title("CDF of pair cosine. Right of green = top 15% (positives), "
                       "left of red = bottom 50% (negatives).", fontsize=11)
    axes[3].set_xlabel("cosine")
    axes[3].set_ylabel("cumulative fraction")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT, f"{sample}_radial_diagnostics.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    samples = sys.argv[1:] if len(sys.argv) > 1 else ["Na007b", "EuInAs_B100"]
    for s in samples:
        main(s)
