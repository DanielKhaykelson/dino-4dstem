"""compare_full_vs_halfmask_class_averages.py
Side-by-side: EuInAs K=10 vanilla, w_ent=0, full mask vs half mask.
Renders class averages in a single figure with training-mask values annotated.

Outputs: runs/_followup_EuInAs_K10_halfmask/fig_compare_full_vs_half_classavg.png
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SAMPLES, LoadPRZ

ROOT = os.path.dirname(os.path.abspath(__file__))

RUNS = [
    ("FULL mask (mask_r=10, polar_cols=30)",
     "runs/EuInAs_B100/winner_polar_centroid",
     {"mask_r": 10, "polar_mask_cols": 30}),
    ("HALF mask (mask_r=5, polar_cols=15)",
     "runs/_followup_EuInAs_K10_halfmask/w_ent_0.00",
     {"mask_r": 5,  "polar_mask_cols": 15}),
]

SAMPLE = "EuInAs_B100"
N_TOP = 200
DISPLAY_BEAM_RADIUS_RAW = 40   # display only; matches viz_gradcam default


def _build_class_avgs(run_dir: str, dataset, vmax: float, K: int):
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft = inf["soft_probs"]
    assigns = inf["assigns"]
    K = soft.shape[1]
    avgs = []
    counts = []
    for c in range(K):
        mask = (assigns == c)
        if int(mask.sum()) < 5:
            avgs.append(None); counts.append(int(mask.sum()))
            continue
        idx_in_class = np.where(mask)[0]
        scores = soft[idx_in_class, c]
        top = idx_in_class[np.argsort(-scores)[:min(N_TOP, len(idx_in_class))]]
        # confidence-weighted mean
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0).astype(np.float32)
        weights = soft[top, c].astype(np.float32)
        wavg = (patterns * weights[:, None, None]).sum(0) / (weights.sum() + 1e-12)
        avgs.append(wavg)
        counts.append(int(mask.sum()))
    return avgs, counts, K


def _disp(arr, mask_r_raw, vmax_clip):
    """Display the pattern as the model sees it: vmax-clipped + training-mask
    applied. No artificial display-only beam mask."""
    H, W = arr.shape
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H/2.0, W/2.0
    # Training mask (red region in plot): zero out r <= mask_r_raw
    train_mask_keep = ((yy - cy)**2 + (xx - cx)**2) > mask_r_raw**2
    # Use percentile-based clip on ALL pixels including the center, so the
    # bright saturated beam doesn't blow the colormap. Linear scale.
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, vmax_clip)
    out = np.clip(arr, lo, hi)
    # Show the training-mask region as black (zeroed) -- this IS what the
    # model receives at that pixel.
    out = out * train_mask_keep
    return out, train_mask_keep


def main():
    cfg = SAMPLES[SAMPLE]
    print(f"loading dataset...")
    ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    K_max = 10

    all_runs = []
    for label, rel, mask_cfg in RUNS:
        run_dir = os.path.join(ROOT, rel)
        print(f"computing class avgs for {rel}...")
        avgs, counts, K = _build_class_avgs(run_dir, ds, cfg["vmax"], K_max)
        all_runs.append((label, avgs, counts, K, mask_cfg, run_dir))

    # Compute training mask radius in RAW pixel frame for annotation:
    #   mask_r is in the 192-frame after crop+resize
    #   crop = 140 of 256 -> resize to 192
    #   So 1 px in 192 = 140/192 of 1 px in 256
    SCALE_192_TO_256 = 140.0 / 192.0

    # Layout: place N_show rows side-by-side per run. Limit to K_active per
    # column for compactness.
    K_act_per_run = []
    for label, avgs, counts, K, mask_cfg, run_dir in all_runs:
        ks = sum(1 for a in avgs if a is not None)
        K_act_per_run.append(ks)
    n_rows = max(K_act_per_run)
    panel_in = 2.4
    fig, axes = plt.subplots(n_rows, 2,
                              figsize=(2 * panel_in, n_rows * panel_in),
                              squeeze=False)

    for col, (label, avgs, counts, K, mask_cfg, run_dir) in enumerate(all_runs):
        # Convert training mask values to RAW px frame for display
        mask_r_raw_cart = mask_cfg["mask_r"] * SCALE_192_TO_256
        # polar_mask_cols masks first N cols of polar (= r-bins).
        # Polar maps 192-frame inscribed circle (R=96) to 192 cols.
        # So polar_cols=30 covers r <= 30/192*96 = 15 in 192-frame
        # = 15 * 140/192 = 10.94 in 256-frame.
        polar_eff_r_raw = (mask_cfg["polar_mask_cols"] / 192.0) * 96.0 * SCALE_192_TO_256
        eff_mask_r_raw = max(mask_r_raw_cart, polar_eff_r_raw)

        # Prototype usage order (by count desc) for fair comparison
        order = np.argsort(-np.asarray(counts))
        for r in range(n_rows):
            ax = axes[r, col]
            if r >= K:
                ax.set_axis_off(); continue
            c = order[r]
            wavg = avgs[c]
            if wavg is None:
                ax.set_axis_off(); continue
            disp, train_mask_keep = _disp(wavg, eff_mask_r_raw, vmax_clip=99.0)
            ax.imshow(disp, cmap="inferno", aspect="equal")
            cy = wavg.shape[0] / 2.0
            cx = wavg.shape[1] / 2.0
            # Training mask circle in red (the actual hidden region from model)
            ax.add_patch(Circle((cx, cy), eff_mask_r_raw,
                                 fill=False, color="red", lw=1.5, ls="-"))
            ax.set_title(f"p{c}  N={int(counts[c])}", fontsize=10, pad=4)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.text(0.02, 0.97,
                         f"{label}\ntraining mask: r≤{eff_mask_r_raw:.1f}px (raw)",
                         transform=ax.transAxes, fontsize=10,
                         color="red", va="top",
                         bbox=dict(facecolor="white", alpha=0.7,
                                    edgecolor="none", pad=2))

    fig.suptitle("EuInAs K=10 vanilla, w_ent=0  —  class averages "
                  "(what the model actually sees)\n"
                  "Linear scale, 1–99 pct clip. Black region in center = the "
                  "TRAINING mask (zeroed for the model). Red circle = mask edge.",
                  fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(
        ROOT, "runs", "_followup_EuInAs_K10_halfmask",
        "fig_compare_full_vs_half_classavg.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
