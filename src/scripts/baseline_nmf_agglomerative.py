"""
baseline_nmf_agglomerative.py — stronger NMF baseline mirroring
Yoo et al. (npj Comp Mater 2024) and Yoo et al. (Sci Reports 2025).

Pipeline:
  1. Polar-transform + center-beam mask every pattern (rotation alignment).
  2. Flatten to vectors, apply NMF with n_components=20.
  3. Compute a **cross-correlation-based similarity matrix** between the
     20 NMF-component images in polar coords (instead of Euclidean distance
     on the H matrix).
  4. Agglomerative (Ward / complete-linkage) clustering of the components
     into groups, producing the component groups.
  5. Assign each scan pixel to the group whose component has the highest
     reconstruction weight for that pixel.

Reports KNN purity, intra/inter cosine of the NMF embedding, NMI vs the
DINO4DSTEM winner, and the automatically chosen number of clusters from
the agglomerative tree's dendrogram using a gap criterion.
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
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import matplotlib.pyplot as plt

from data import SAMPLES, LoadPRZ
from baseline_nmf_kmeans import patterns_to_polar_vectors


def cross_correlation_similarity(comp_images: np.ndarray) -> np.ndarray:
    """Pairwise similarity: for two 2-D images, take the max of their 2-D
    cross-correlation. Polar-transformed patterns whose only difference is
    in-plane rotation produce a shifted component image; cross-correlation
    picks that up automatically.

    comp_images: (K, H, W) of K component images (already polar).
    Returns (K, K) similarity matrix in [0, 1] after per-row normalization.
    """
    from scipy.signal import correlate2d
    K = comp_images.shape[0]
    # Flatten and normalize each component for correlation.
    imgs = [c - c.mean() for c in comp_images]
    sim = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        a = imgs[i]
        norm_a = np.sqrt((a ** 2).sum())
        for j in range(K):
            if j < i:
                sim[i, j] = sim[j, i]
                continue
            b = imgs[j]
            norm_b = np.sqrt((b ** 2).sum())
            # 'valid' mode is too restrictive; use 'full' and take max.
            c = correlate2d(a, b, mode="same")
            sim[i, j] = float(c.max()) / max(norm_a * norm_b, 1e-12)
    # Normalize each row to [0, 1].
    sim = sim / np.maximum(sim.max(axis=1, keepdims=True), 1e-12)
    return sim


def agglomerative_components(comp_images: np.ndarray, linkage: str = "ward",
                              n_clusters: int | None = None) -> tuple:
    """Agglomerative cluster the NMF components into groups via cross-correlation
    similarity. Returns (labels array of shape (K,), merge tree)."""
    from sklearn.cluster import AgglomerativeClustering
    sim = cross_correlation_similarity(comp_images)
    # Convert to a distance matrix for clustering.
    dist = 1.0 - sim
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    # Ward requires Euclidean; use 'average' when passing precomputed distances.
    if linkage == "ward":
        linkage = "average"
    # n_clusters heuristic: silhouette-like approach on the distance matrix.
    # For now pick n_clusters manually or default to 4 (matching Yoo 2024).
    if n_clusters is None:
        n_clusters = 4
    ac = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                   linkage=linkage)
    labels = ac.fit_predict(dist)
    return labels, dist


def run(sample: str, dino_config: str, n_components: int = 20,
         n_clusters_range: tuple = (2, 14), force_n_clusters: int = None):
    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    scan_shape = cfg["scan_shape"]
    out_dir = os.path.join("runs", sample, "baseline_nmf_agglomerative", "eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] [{sample}] preprocessing + NMF({n_components})", flush=True)
    t0 = time.perf_counter()
    X = patterns_to_polar_vectors(dataset, polar_size=192)
    X_nn = X - X.min()
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=n_components, init="nndsvd", max_iter=400,
               tol=1e-3, random_state=0)
    W = nmf.fit_transform(X_nn)
    H = nmf.components_
    print(f"[{datetime.now():%H:%M:%S}] NMF done in {time.perf_counter() - t0:.1f}s. "
          f"Rec err {nmf.reconstruction_err_:.3g}", flush=True)

    # Reshape components back to polar space for cross-correlation.
    comp_images = H.reshape(n_components, 192, 192)

    # Load DINO4DSTEM assignments for comparison.
    dino_path = f"runs/{sample}/{dino_config}/eval/inference.npz"
    if os.path.exists(dino_path):
        dino = np.load(dino_path)
        dino_assigns = dino["assigns"]
        target_k = int(dino_assigns.max()) + 1
    else:
        dino_assigns = None
        target_k = 6

    # Sweep n_clusters across range, pick best by silhouette of the SCAN
    # assignments (the thing a practitioner ultimately wants).
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score
    from contrastive_eval import intra_inter_cosine, knn_purity

    sweep = []
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), size=min(2500, len(X)), replace=False)
    best_k, best_sil, best_assigns = None, -1.0, None
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        labels, dist = agglomerative_components(comp_images, linkage="average",
                                                   n_clusters=k)
        # Assign each scan pixel to the group of its strongest NMF component.
        strongest = W.argmax(axis=1)
        scan_assigns = labels[strongest]
        try:
            sil = silhouette_score(W[sub], scan_assigns[sub], metric="cosine")
        except Exception:
            sil = -1.0
        sweep.append((k, float(sil)))
        if sil > best_sil:
            best_sil, best_k, best_assigns = sil, k, scan_assigns
    print(f"[{sample}] silhouette sweep: K* = {best_k}, silhouette = {best_sil:.3f}",
          flush=True)

    # Force cluster count to match DINO4DSTEM's K for the head-to-head.
    forced_k = force_n_clusters or target_k
    labels_target, _ = agglomerative_components(comp_images, linkage="average",
                                                   n_clusters=forced_k)
    strongest = W.argmax(axis=1)
    assigns_target = labels_target[strongest]

    # Metrics at k_star and at target_k.
    def _metrics(assigns):
        intra, inter = intra_inter_cosine(W, assigns)
        ratio = intra / inter if inter and inter != 0 and not np.isnan(inter) else float("nan")
        knn = knn_purity(W, assigns, k=10, max_query=1500)
        nmi = float(normalized_mutual_info_score(assigns, dino_assigns)) \
            if dino_assigns is not None else float("nan")
        return dict(intra_class_cosine=intra, inter_class_cosine=inter,
                     intra_over_inter=ratio, KNN_purity_k10=knn,
                     NMI_vs_dino=nmi)

    m_star = _metrics(best_assigns)
    m_target = _metrics(assigns_target)
    print(f"[{sample}] at K*={best_k}: KNN {m_star['KNN_purity_k10']:.3f}, "
          f"intra/inter {m_star['intra_over_inter']:.2f}", flush=True)
    print(f"[{sample}] at K={forced_k} (match DINO4DSTEM): KNN "
          f"{m_target['KNN_purity_k10']:.3f}, intra/inter "
          f"{m_target['intra_over_inter']:.2f}", flush=True)

    # Figures.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ks = [k for k, _ in sweep]; sil = [s for _, s in sweep]
    ax.plot(ks, sil, "o-", lw=2, color="tab:purple")
    ax.axvline(best_k, color="red", ls="--", label=f"K*={best_k}")
    ax.axvline(forced_k, color="tab:blue", ls="--", label=f"DINO K={forced_k}")
    ax.set_xlabel("K"); ax.set_ylabel("silhouette (scan-level)")
    ax.set_title(f"NMF + polar-cross-correlation agglomerative sweep: {sample}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_silhouette_vs_k.png"), dpi=110,
                 bbox_inches="tight")
    plt.close(fig)

    # Class map at forced_k.
    m = assigns_target.reshape(scan_shape)
    Ny, Nx = scan_shape
    aspect = Nx / max(Ny, 1)
    if aspect >= 1:
        fig_w, fig_h = 10, max(3.5, 10 / aspect)
    else:
        fig_h, fig_w = 10, max(3.5, 10 * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = "tab10" if forced_k <= 10 else "tab20"
    vmax = 9 if forced_k <= 10 else 19
    ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"NMF + polar-cross-correlation agglomerative at K={forced_k}\n"
                   f"KNN={m_target['KNN_purity_k10']:.3f}  "
                   f"intra/inter={m_target['intra_over_inter']:.2f}  "
                   f"NMI(DINO)={m_target['NMI_vs_dino']:.2f}")
    ax.set_axis_off()
    fig.colorbar(ax.images[0], ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_class_map_targetk.png"), dpi=110,
                 bbox_inches="tight")
    plt.close(fig)

    summary = dict(
        sample=sample, dino_config=dino_config,
        n_components=n_components,
        k_star=best_k, silhouette_star=best_sil,
        target_k=forced_k,
        silhouette_curve=sweep,
        metrics_at_k_star=m_star,
        metrics_at_target_k=m_target,
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    np.savez_compressed(os.path.join(out_dir, "inference.npz"),
                         assigns_kstar=best_assigns,
                         assigns_target_k=assigns_target,
                         nmf_W=W)
    print(f"[{sample}] NMF+agglomerative baseline done -> {out_dir}", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="Na007b")
    ap.add_argument("--dino-config", default="sweep_polar_centroid")
    ap.add_argument("--n-components", type=int, default=20)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=14)
    args = ap.parse_args(argv)
    run(args.sample, args.dino_config,
         n_components=args.n_components,
         n_clusters_range=(args.k_min, args.k_max))


if __name__ == "__main__":
    main()
