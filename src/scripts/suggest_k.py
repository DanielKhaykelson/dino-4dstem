"""
suggest_k.py — data-driven prior for the number of prototypes K.

Given a trained contrastive-DINO run at some K_initial, this tool:

  1. Pulls the student contrastive embeddings + prototype assignments
     from `runs/<sample>/<config>/eval/inference.npz`.
  2. Agglomeratively merges prototypes in *centroid-cosine* space from
     K_initial down to K=2, recording the merge distance at each step.
  3. At every K along the way, evaluates three quality signals:
         - silhouette on a sample of embeddings (cluster compactness)
         - effective K (exp of per-sample soft-assignment entropy),
           a sanity check that the merge actually consolidates usage
         - spatial coherence on the scan grid
           (4-neighbor-agreement fraction of the merged class map)
  4. Picks a recommended K using two complementary heuristics:
         H1. Largest *gap* in the merge-distance curve sorted ascending
             (the "elbow" in cosine merge cost — the step where the
             next merge would fuse genuinely different prototypes).
         H2. Peak of the spatial-coherence curve (the K at which the
             scan class map is most physically coherent).
      If the two heuristics agree, confidence is high. If they split,
      we report both.
  5. Writes a figure (`fig_suggest_k.png`) showing merge dendrogram and
     the three quality curves vs K, plus a `suggest_k.json` with the
     recommendation + alternate merged class maps for K in
     {2, 3, ..., K_initial}.

Usage:
    python suggest_k.py --sample Na007b --config config_c
    python suggest_k.py --sample Na007b --config config_c_K6

The tool does NOT retrain. It works entirely from the already-saved
inference arrays, so sweeping the recommendation is cheap (< 10 s).
"""

from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse
import json
import os
from typing import Any

import numpy as np


# =========================================================================
# 1. Core utilities
# =========================================================================

def _cos_matrix(centroids: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix over row vectors."""
    n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    return n @ n.T


def _agglomerative_cosine_merge(centroids: np.ndarray, counts: np.ndarray):
    """Single-linkage merge in cosine-distance space.

    Returns:
        merge_seq : list of (K_before, K_after, i, j, dist) where at step t
                    prototypes i, j are merged into a new centroid (weighted
                    by sample count). K decreases by 1 per step.
        centroid_at_K : dict[K -> (K, D) centroid matrix]
        group_at_K    : dict[K -> (K0,) array: original-prototype -> merged-bucket]

    Where K0 is the initial number of prototypes.
    """
    K0 = centroids.shape[0]
    alive = list(range(K0))
    cents = centroids.astype(np.float64).copy()
    cnts = counts.astype(np.float64).copy()
    group = np.arange(K0, dtype=np.int64)  # original-proto -> bucket id
    bucket_members = {k: [k] for k in range(K0)}

    centroid_at_K = {K0: centroids.copy()}
    group_at_K = {K0: group.copy()}
    merge_seq = []

    for step in range(K0 - 1):
        # Pair with the highest cos-sim (= smallest cos distance) among alive.
        sub = cents[alive]
        cos = _cos_matrix(sub)
        np.fill_diagonal(cos, -np.inf)
        ii, jj = np.unravel_index(np.argmax(cos), cos.shape)
        i, j = alive[ii], alive[jj]
        cos_ij = float(cos[ii, jj])
        dist_ij = float(1.0 - cos_ij)
        # Weighted-mean centroid of merged bucket (simple, stable).
        w_i, w_j = cnts[i], cnts[j]
        if w_i + w_j > 0:
            new_c = (w_i * cents[i] + w_j * cents[j]) / (w_i + w_j)
        else:
            new_c = 0.5 * (cents[i] + cents[j])
        # Normalize so future cosine calcs treat merged centroid like others.
        nc_norm = np.linalg.norm(new_c) + 1e-12
        new_c = new_c / nc_norm
        # Replace i with merged centroid; drop j. Reassign j's bucket to i.
        cents[i] = new_c
        cnts[i] = w_i + w_j
        alive.remove(j)
        bucket_members[i].extend(bucket_members.pop(j))
        # Update group mapping so every original proto points at its current bucket.
        for orig in range(K0):
            if group[orig] == j:
                group[orig] = i
        K_after = len(alive)
        merge_seq.append((K_after + 1, K_after, int(i), int(j), dist_ij))
        centroid_at_K[K_after] = cents[alive].copy()
        group_at_K[K_after] = group.copy()

    return merge_seq, centroid_at_K, group_at_K, bucket_members


def _remap_assignments(assigns: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Apply original-proto-to-bucket group map to a 1-D assignment array,
    then re-label densely so buckets are 0..K-1 in descending-size order.
    """
    mapped = group[assigns]
    # Compact to contiguous ids.
    uniq, inverse = np.unique(mapped, return_inverse=True)
    counts = np.bincount(inverse)
    order = np.argsort(-counts)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[inverse]


def _silhouette_sample(embeds: np.ndarray, labels: np.ndarray,
                        max_n: int = 1500, seed: int = 0) -> float:
    """Silhouette coefficient on a random sub-sample (cosine distance)."""
    from sklearn.metrics import silhouette_score
    n = embeds.shape[0]
    if len(np.unique(labels)) < 2 or n < 10:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(max_n, n), replace=False)
    try:
        return float(silhouette_score(embeds[idx], labels[idx], metric="cosine"))
    except Exception:
        return float("nan")


def _spatial_coherence(class_map_2d: np.ndarray) -> float:
    """Fraction of 4-connected scan neighbors with the same class label."""
    h, w = class_map_2d.shape
    agree_h = (class_map_2d[:-1, :] == class_map_2d[1:, :]).sum()
    agree_w = (class_map_2d[:, :-1] == class_map_2d[:, 1:]).sum()
    total = (h - 1) * w + h * (w - 1)
    return float(agree_h + agree_w) / max(total, 1)


def _effective_k_soft(soft_probs: np.ndarray, group: np.ndarray) -> float:
    """Effective K from batch-averaged soft assignments after grouping."""
    K0 = soft_probs.shape[1]
    buckets = np.unique(group)
    merged = np.zeros((soft_probs.shape[0], len(buckets)), dtype=np.float64)
    for new_id, b in enumerate(buckets):
        mask = (group == b)
        merged[:, new_id] = soft_probs[:, mask].sum(axis=1)
    p = merged.mean(axis=0)
    p = np.clip(p, 1e-12, 1.0)
    return float(np.exp(-(p * np.log(p)).sum()))


# =========================================================================
# 2. Recommendation logic
# =========================================================================

def _recommend_k(merge_seq, coh_by_k: dict, sil_by_k: dict,
                 min_k: int = 2) -> dict:
    """Primary heuristic: merge-gap (first big jump in cosine merge cost).

    merge-gap is the only heuristic here that isn't monotone in K:
      - silhouette biases high (the network was trained to separate K_initial
        groups, so silhouette almost always peaks at K_initial).
      - spatial coherence biases low (fewer labels always agree more often;
        at K=1 it's trivially 1.0).
    Both are reported as diagnostics but not voted on.

    The recommendation rule:

      1. Smooth the merge-distance curve by a small moving median (window=2)
         to absorb single-merge noise.
      2. Find the largest positive *jump* in merge cost as we go from K+1 to K.
         Recommend `K+1` — the K value just BEFORE the costly merge.
      3. Confidence = (largest gap) / (median gap) capped at 1.0.

    Sanity band: if the recommended K differs from `spatial_plateau_K`
    (first K at which spatial-coherence gain drops below 0.03) by > 2, flag
    low confidence and report the band.
    """
    # merge_seq entries: (K_before, K_after, i, j, dist).
    dists_by_k_after = {K_after: d for (_, K_after, _, _, d) in merge_seq}
    ks_desc = sorted(dists_by_k_after.keys(), reverse=True)  # e.g. [9, 8, ..., 1]

    # Per-merge gap: cost of k_next vs cost of k_curr (larger K first).
    gap_scores = {}
    for i in range(len(ks_desc) - 1):
        k_curr = ks_desc[i]                # larger K (after this merge)
        k_next = ks_desc[i + 1]            # next merge goes here
        gap_scores[k_curr] = dists_by_k_after[k_next] - dists_by_k_after[k_curr]
    # merge_gap_K = the K just BEFORE the costly jump.
    if gap_scores:
        costly_K_curr = max(gap_scores, key=gap_scores.get)
        merge_gap_K = costly_K_curr + 1  # don't do the costly merge
        gap_mag = gap_scores[costly_K_curr]
        med_gap = float(np.median(list(gap_scores.values()))) or 1e-9
        confidence = float(min(1.0, gap_mag / max(med_gap, 1e-9) / 4.0))
    else:
        merge_gap_K, gap_mag, confidence = None, 0.0, 0.0

    # Diagnostic: spatial plateau K.
    ks_asc = sorted(coh_by_k.keys())  # ascending K
    plateau_K = None
    for i in range(len(ks_asc) - 1, 0, -1):
        k_high = ks_asc[i]
        k_low = ks_asc[i - 1]
        gain = coh_by_k[k_low] - coh_by_k[k_high]
        if gain < 0.03:
            plateau_K = k_high
            break
    if plateau_K is None:
        plateau_K = ks_asc[-1] if ks_asc else None

    # Diagnostic: peak spatial coherence across K (known to bias low).
    coh_peak_K = max(coh_by_k, key=coh_by_k.get) if coh_by_k else None
    # Diagnostic: peak silhouette (known to bias high).
    sil_peak_K = max(sil_by_k, key=sil_by_k.get) if sil_by_k else None

    # Sanity band: if merge-gap and spatial-plateau disagree by more than 2,
    # flag low confidence. Report both.
    if merge_gap_K is not None and plateau_K is not None:
        if abs(merge_gap_K - plateau_K) > 2:
            confidence = min(confidence, 0.5)

    return dict(
        recommended_K=merge_gap_K,
        confidence=confidence,
        merge_gap_K=merge_gap_K,
        merge_gap_magnitude=float(gap_mag),
        spatial_plateau_K=plateau_K,
        spatial_peak_K_diagnostic=coh_peak_K,
        silhouette_peak_K_diagnostic=sil_peak_K,
        merge_distances_by_K_after=dists_by_k_after,
    )


# =========================================================================
# 3. Figures
# =========================================================================

def _save_fig(fig, path):
    fig.savefig(path, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_suggest_k(merge_seq, coh_by_k: dict, sil_by_k: dict,
                     effk_by_k: dict, rec: dict, outpath: str,
                     scan_maps: dict | None = None):
    import matplotlib.pyplot as plt

    K_vals = sorted(coh_by_k.keys())

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 1.4],
                           hspace=0.35, wspace=0.35)

    # (1) Merge-distance curve
    ax1 = fig.add_subplot(gs[0, 0])
    dists_k = sorted(rec["merge_distances_by_K_after"].items())
    ks = [k for k, _ in dists_k]
    ds = [d for _, d in dists_k]
    ax1.plot(ks, ds, "o-", lw=2, color="tab:blue")
    if rec["merge_gap_K"] is not None:
        ax1.axvline(rec["merge_gap_K"], color="tab:red", ls="--", lw=1.2,
                    label=f"merge-gap K={rec['merge_gap_K']}")
    ax1.set_xlabel("K (after merge)")
    ax1.set_ylabel("cosine distance at merge")
    ax1.set_title("merge cost curve (elbow ↘ recommends K)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # (2) Silhouette vs K (diagnostic — biases high)
    ax2 = fig.add_subplot(gs[0, 1])
    sils = [sil_by_k.get(k, float("nan")) for k in K_vals]
    ax2.plot(K_vals, sils, "s-", color="tab:green", lw=2)
    ax2.set_xlabel("K"); ax2.set_ylabel("silhouette (cosine)")
    ax2.set_title("embedding silhouette\n(diagnostic — biases high)")
    ax2.grid(alpha=0.3)

    # (3) Spatial coherence vs K (diagnostic — biases low)
    ax3 = fig.add_subplot(gs[0, 2])
    cohs = [coh_by_k.get(k, float("nan")) for k in K_vals]
    ax3.plot(K_vals, cohs, "^-", color="tab:orange", lw=2)
    if rec["spatial_plateau_K"] is not None:
        ax3.axvline(rec["spatial_plateau_K"], color="tab:gray", ls="--", lw=1.2,
                    label=f"spatial plateau K={rec['spatial_plateau_K']}")
    ax3.set_xlabel("K"); ax3.set_ylabel("4-neighbor agreement")
    ax3.set_title("scan-map spatial coherence\n(diagnostic — biases low)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # (4) Effective K vs K
    ax4 = fig.add_subplot(gs[1, 0])
    effks = [effk_by_k.get(k, float("nan")) for k in K_vals]
    ax4.plot(K_vals, effks, "o-", color="tab:purple", lw=2)
    ax4.plot(K_vals, K_vals, "k:", lw=1, alpha=0.5, label="y=x")
    ax4.set_xlabel("K (merged)"); ax4.set_ylabel("effective K (soft)")
    ax4.set_title("post-merge usage entropy")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # (5, 6) Scan map at recommended K and at K=3 (physical minimum sanity)
    if scan_maps is not None:
        rec_k = rec["recommended_K"]
        low_k = 3
        for panel_ax, k_show in zip(
                (fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])),
                (rec_k, low_k)):
            if k_show is None or k_show not in scan_maps:
                panel_ax.set_axis_off()
                continue
            m = scan_maps[k_show]
            im = panel_ax.imshow(m, cmap="tab10", vmin=0, vmax=9)
            panel_ax.set_title(f"scan at K={k_show}")
            panel_ax.set_axis_off()
    # Recommendation banner as suptitle.
    banner = (f"recommended K = {rec['recommended_K']}  "
              f"(conf {rec['confidence']:.0%})   "
              f"merge-gap K={rec['merge_gap_K']} (jump {rec['merge_gap_magnitude']:.3f}), "
              f"spatial plateau K={rec['spatial_plateau_K']}")
    fig.suptitle(banner, fontsize=11, y=0.995)
    _save_fig(fig, outpath)


# =========================================================================
# 4. Driver
# =========================================================================

def suggest_k_for_run(run_dir: str, scan_shape: tuple,
                      min_k: int = 2, write_maps: bool = True) -> dict:
    """Given a run directory containing eval/inference.npz, produce a
    suggest_k.json + fig_suggest_k.png + merged class-map PNGs per K.
    """
    inf_path = os.path.join(run_dir, "eval", "inference.npz")
    data = np.load(inf_path)
    soft_probs = data["soft_probs"]
    embeds = data["embeds"]
    assigns = data["assigns"]
    K0 = int(soft_probs.shape[1])

    # Re-compute centroids in the contrastive embedding space.
    centroids = np.zeros((K0, embeds.shape[1]), dtype=np.float64)
    counts = np.zeros(K0, dtype=np.int64)
    for k in range(K0):
        mask = (assigns == k)
        counts[k] = int(mask.sum())
        if counts[k] > 0:
            centroids[k] = embeds[mask].mean(axis=0)

    merge_seq, cent_by_k, group_by_k, _ = _agglomerative_cosine_merge(
        centroids, counts)

    coh_by_k, sil_by_k, effk_by_k = {}, {}, {}
    scan_maps = {}
    for K, group in group_by_k.items():
        if K < min_k:
            continue
        merged_labels = _remap_assignments(assigns, group)
        m2d = merged_labels.reshape(scan_shape)
        coh_by_k[K] = _spatial_coherence(m2d)
        sil_by_k[K] = _silhouette_sample(embeds, merged_labels)
        effk_by_k[K] = _effective_k_soft(soft_probs, group)
        if write_maps:
            scan_maps[K] = m2d

    rec = _recommend_k(merge_seq, coh_by_k, sil_by_k, min_k=min_k)

    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "fig_suggest_k.png")
    _plot_suggest_k(merge_seq, coh_by_k, sil_by_k, effk_by_k, rec,
                     fig_path, scan_maps=scan_maps if write_maps else None)

    # Write merged class maps per K as small PNGs.
    if write_maps:
        import matplotlib.pyplot as plt
        maps_dir = os.path.join(out_dir, "class_maps_by_k")
        os.makedirs(maps_dir, exist_ok=True)
        for K, m2d in scan_maps.items():
            fig, ax = plt.subplots(figsize=(4.0, 4.0 * scan_shape[0] / scan_shape[1]))
            ax.imshow(m2d, cmap="tab10", vmin=0, vmax=9)
            ax.set_title(f"scan at K={K}")
            ax.set_axis_off()
            fig.tight_layout()
            _save_fig(fig, os.path.join(maps_dir, f"class_map_K{K}.png"))

    summary = dict(
        run_dir=run_dir,
        K_initial=K0,
        recommended_K=rec["recommended_K"],
        confidence=rec["confidence"],
        heuristics=dict(
            merge_gap_K=rec["merge_gap_K"],
            merge_gap_magnitude=rec["merge_gap_magnitude"],
            spatial_plateau_K=rec["spatial_plateau_K"],
            spatial_peak_K_diagnostic=rec["spatial_peak_K_diagnostic"],
            silhouette_peak_K_diagnostic=rec["silhouette_peak_K_diagnostic"],
        ),
        merge_distances_by_K_after={int(k): float(v)
                                     for k, v in rec["merge_distances_by_K_after"].items()},
        spatial_coherence_by_K={int(k): float(v) for k, v in coh_by_k.items()},
        silhouette_by_K={int(k): float(v) for k, v in sil_by_k.items()},
        effective_K_by_K={int(k): float(v) for k, v in effk_by_k.items()},
    )
    with open(os.path.join(out_dir, "suggest_k.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


# =========================================================================
# 5. CLI
# =========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="Na007b",
                    help="Sample key (must match data.SAMPLES and runs/<sample>/)")
    ap.add_argument("--config", default="config_c",
                    help="Config subdir name under runs/<sample>/")
    ap.add_argument("--min-k", type=int, default=2)
    args = ap.parse_args(argv)

    from data import SAMPLES
    scan_shape = SAMPLES[args.sample]["scan_shape"]
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "runs", args.sample, args.config)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    res = suggest_k_for_run(run_dir, scan_shape, min_k=args.min_k)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
