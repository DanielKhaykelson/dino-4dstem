"""
compare_maps.py — side-by-side + IoU + disagreement map for two runs on
the same sample.

For every pair of dense classes (i_old, j_new), compute IoU on the scan grid:
    IoU(i, j) = |{assigns_old==i} ∩ {assigns_new==j}| / |{assigns_old==i} ∪ {assigns_new==j}|

Greedy Hungarian matching picks the permutation of new-class IDs that
maximizes total IoU with the old map. Then:

  - per-matched-pair IoU reported (plus unmatched remainders)
  - pixel-level agreement fraction on the scan
  - NMI and Adjusted Rand Index for overall partition similarity
  - disagreement map (red where the new map differs from the old)

Outputs: `<out_dir>/fig_compare_<old>_vs_<new>.png` and
`<out_dir>/compare_<old>_vs_<new>.json` with metrics.

Usage:
    python compare_maps.py --sample EuInAs_B100 \\
        --old winner_polar_centroid --new winner_conf_gated
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SAMPLES


def _iou_matrix(a: np.ndarray, b: np.ndarray, Ka: int, Kb: int) -> np.ndarray:
    """(Ka, Kb) IoU between each a-class and b-class on the same scan."""
    M = np.zeros((Ka, Kb), dtype=np.float64)
    for i in range(Ka):
        ai = (a == i)
        if not ai.any():
            continue
        for j in range(Kb):
            bj = (b == j)
            inter = np.sum(ai & bj)
            union = np.sum(ai | bj)
            M[i, j] = inter / union if union > 0 else 0.0
    return M


def _hungarian_match(iou: np.ndarray):
    """Return (rows, cols) that MAXIMIZES sum of iou[rows, cols]."""
    from scipy.optimize import linear_sum_assignment
    # Pad to square for unambiguous assignment when Ka != Kb.
    Ka, Kb = iou.shape
    K = max(Ka, Kb)
    pad = np.zeros((K, K))
    pad[:Ka, :Kb] = iou
    # linear_sum_assignment minimizes; negate IoU.
    r, c = linear_sum_assignment(-pad)
    keep = (r < Ka) & (c < Kb)
    return r[keep], c[keep]


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized mutual information between two labelings (sklearn formula)."""
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(a, b))


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def compare(sample: str, old_config: str, new_config: str,
             out_dir: str | None = None):
    cfg = SAMPLES[sample]
    scan_shape = cfg["scan_shape"]
    base = os.path.dirname(os.path.abspath(__file__))

    old_eval = os.path.join(base, "runs", sample, old_config, "eval")
    new_eval = os.path.join(base, "runs", sample, new_config, "eval")

    old_assigns = np.load(os.path.join(old_eval, "inference.npz"))["assigns"]
    new_assigns = np.load(os.path.join(new_eval, "inference.npz"))["assigns"]
    Ka = int(old_assigns.max()) + 1
    Kb = int(new_assigns.max()) + 1

    iou = _iou_matrix(old_assigns, new_assigns, Ka, Kb)
    r, c = _hungarian_match(iou)
    matched_pairs = [(int(i), int(j), float(iou[i, j])) for i, j in zip(r, c)]
    # Apply the matching: relabel new_assigns so new-class j -> matched old-class i.
    relabel = np.full(Kb, -1, dtype=np.int64)
    for i, j, _ in matched_pairs:
        relabel[j] = i
    # For any new classes not matched (new had more classes than old), assign
    # them a fresh ID above Ka.
    unmatched = [j for j in range(Kb) if relabel[j] < 0]
    for k_off, j in enumerate(unmatched):
        relabel[j] = Ka + k_off
    new_on_old_ids = relabel[new_assigns]
    agreement = float((old_assigns == new_on_old_ids).mean())

    nmi = _nmi(old_assigns, new_assigns)
    ari = _ari(old_assigns, new_assigns)

    # ---------- Figure ----------
    Ny, Nx = scan_shape
    aspect = Nx / max(Ny, 1)
    if aspect >= 1:
        row_w, row_h = 10, max(2.2, 10 / aspect)
    else:
        row_h, row_w = 10, max(2.8, 10 * aspect)

    fig = plt.figure(figsize=(row_w, 3 * row_h + 3.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[row_h, row_h, row_h, 3.0],
                            hspace=0.35, wspace=0.10)

    old_map = old_assigns.reshape(scan_shape)
    new_map_relabeled = new_on_old_ids.reshape(scan_shape)
    disagree = (old_map != new_map_relabeled).astype(np.uint8)

    K_any = max(int(old_map.max()), int(new_map_relabeled.max())) + 1
    if K_any <= 10:
        cmap = "tab10"; vmax = 9
    elif K_any <= 20:
        cmap = "tab20"; vmax = 19
    else:
        cmap = "turbo"; vmax = K_any - 1

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(old_map, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"OLD: {old_config}  (K_active={Ka})", fontsize=10)
    ax.set_axis_off()

    ax = fig.add_subplot(gs[1, :])
    ax.imshow(new_map_relabeled, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"NEW (relabeled to match old via Hungarian):  "
                  f"{new_config}  (K_active={Kb})", fontsize=10)
    ax.set_axis_off()

    ax = fig.add_subplot(gs[2, :])
    im = ax.imshow(disagree, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"disagreement map (red = assignment changed)  "
                  f"— agreement {agreement:.1%},  NMI={nmi:.3f},  ARI={ari:.3f}",
                  fontsize=10)
    ax.set_axis_off()

    # IoU matrix + matched pairs as a bar chart.
    ax = fig.add_subplot(gs[3, 0])
    im = ax.imshow(iou, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"IoU matrix  (rows=old, cols=new)", fontsize=9)
    ax.set_xlabel("new class id"); ax.set_ylabel("old class id")
    for i, j, v in matched_pairs:
        ax.scatter(j, i, marker="o", s=40, facecolors="none",
                    edgecolors="red", linewidths=1.5)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax = fig.add_subplot(gs[3, 1])
    pairs_sorted = sorted(matched_pairs, key=lambda x: -x[2])
    labels = [f"old{i} vs new{j}" for i, j, _ in pairs_sorted]
    vals = [v for _, _, v in pairs_sorted]
    ax.barh(range(len(vals)), vals, color="tab:blue")
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("per-matched-pair IoU (sorted)", fontsize=9)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle(f"{sample}: compare  {old_config}  vs  {new_config}",
                  fontsize=11, y=0.995)
    if out_dir is None:
        out_dir = new_eval
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir,
                             f"fig_compare_{old_config}_vs_{new_config}.png")
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    summary = dict(
        sample=sample,
        old_config=old_config,
        new_config=new_config,
        K_old=Ka, K_new=Kb,
        agreement_fraction=agreement,
        NMI=nmi,
        ARI=ari,
        matched_pairs=[dict(old=p[0], new=p[1], iou=p[2]) for p in matched_pairs],
        unmatched_new_classes=unmatched,
        mean_matched_iou=float(np.mean([p[2] for p in matched_pairs]))
                          if matched_pairs else 0.0,
    )
    json_path = os.path.join(out_dir,
                              f"compare_{old_config}_vs_{new_config}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[compare] wrote {fig_path}")
    print(f"[compare] wrote {json_path}")
    print(f"[compare] agreement={agreement:.1%}  NMI={nmi:.3f}  ARI={ari:.3f}")
    print(f"[compare] matched IoUs:")
    for p in matched_pairs:
        print(f"            old{p[0]} -> new{p[1]}:  IoU={p[2]:.3f}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--old", required=True, help="old config subdir name")
    ap.add_argument("--new", required=True, help="new config subdir name")
    args = ap.parse_args(argv)
    compare(args.sample, args.old, args.new)


if __name__ == "__main__":
    main()
