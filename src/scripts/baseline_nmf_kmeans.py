"""
baseline_nmf_kmeans.py — reference implementation of the standard
NMF-then-k-means workflow, for head-to-head comparison with DINO4DSTEM.

Follows the Yoo et al. (npj Comp Mater 2024) pattern but simpler — PCA or
NMF on vmax-normalized, polar-transformed patterns, then k-means on the
low-dim embeddings. Silhouette sweep picks K automatically; we also
report results at the same K DINO4DSTEM chose.

Writes under `runs/<sample>/baseline_nmf_kmeans/eval/`:

  fig_class_map.png              NMF+kmeans assignment on scan (at K*)
  fig_silhouette_vs_k.png        silhouette score across K in [2, 14]
  fig_nmf_components.png         top-9 NMF components as heatmaps
  metrics.json                   KNN purity, intra/inter cosine, NMI to DINO4DSTEM
                                  winner, k selected
  inference.npz                  assigns, embeddings (NMF-space),
                                  silhouette curve

Runs on CPU; no GPU needed. ~2 min per sample end-to-end.
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
from dino_sr_contrastive_model import PolarTransform, PolarMaskLeft


# =========================================================================
# 1. Pre-process patterns to vectors (shared with DINO4DSTEM eval pipeline)
# =========================================================================

def patterns_to_polar_vectors(dataset, center_crop_size=140, polar_size=192,
                               polar_mask_cols=30, batch_size=256):
    """Apply the eval preprocessing (center crop, resize, polar, mask) to
    every pattern and return an (N, polar_size * polar_size) float array.

    Uses torch on CPU since the transforms are cheap vs the clustering.
    """
    import torch
    from torchvision.transforms import v2 as T
    from torchvision.transforms import InterpolationMode

    eval_tf = T.Compose([
        T.CenterCrop(center_crop_size),
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        PolarTransform(output_size=polar_size),
        PolarMaskLeft(k_cols=polar_mask_cols),
    ])

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    vecs = []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).float()
        x = eval_tf(x)                          # (B, 1, 192, 192)
        vecs.append(x.squeeze(1).reshape(x.size(0), -1).numpy())
    return np.concatenate(vecs, axis=0).astype(np.float32)


# =========================================================================
# 2. NMF + k-means pipeline with silhouette K-sweep
# =========================================================================

def run_nmf_kmeans(X: np.ndarray,
                    n_components: int = 20,
                    k_range: tuple = (2, 14),
                    target_k: int | None = None,
                    random_state: int = 0) -> dict:
    """NMF on X, then k-means across a K-range with silhouette scoring.

    Returns dict with:
      nmf_W          : (N, n_components) low-dim embedding
      k_silhouette   : list of (k, silhouette_score)
      k_star         : k with peak silhouette
      assigns_star   : (N,) cluster assignment at k_star
      assigns_target : (N,) cluster assignment at target_k (if supplied)
    """
    from sklearn.decomposition import NMF
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # NMF requires non-negative. Shift by min if necessary.
    X_nn = X - X.min()
    nmf = NMF(n_components=n_components, init="nndsvd", max_iter=400,
               tol=1e-3, random_state=random_state)
    W = nmf.fit_transform(X_nn)
    H = nmf.components_
    print(f"[nmf] fit done: W shape {W.shape}  reconstruction err {nmf.reconstruction_err_:.3g}")

    k_silhouette = []
    best_k, best_sil = None, -1.0
    best_assigns = None
    # For silhouette use a subsample to stay fast.
    rng = np.random.default_rng(random_state)
    sub_idx = rng.choice(len(W), size=min(3000, len(W)), replace=False)
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        a = km.fit_predict(W)
        sil = silhouette_score(W[sub_idx], a[sub_idx], metric="cosine")
        k_silhouette.append((k, sil))
        if sil > best_sil:
            best_sil, best_k, best_assigns = sil, k, a

    # Also fit at target_k if requested (for head-to-head at DINO4DSTEM's K).
    assigns_target = None
    if target_k is not None:
        km = KMeans(n_clusters=target_k, n_init=10, random_state=random_state)
        assigns_target = km.fit_predict(W)

    return dict(
        nmf_W=W, H=H,
        k_silhouette=k_silhouette,
        k_star=best_k, silhouette_star=best_sil,
        assigns_star=best_assigns,
        assigns_target=assigns_target,
        target_k=target_k,
    )


# =========================================================================
# 3. Compare to DINO4DSTEM winner (NMI, intra/inter)
# =========================================================================

def nmi_and_cosine_metrics(assigns: np.ndarray, embeds: np.ndarray,
                             dino_assigns: np.ndarray | None):
    """Cluster-quality metrics on the NMF embedding + cross-agreement with
    DINO4DSTEM winner assignments (if available)."""
    from sklearn.metrics import normalized_mutual_info_score
    from contrastive_eval import intra_inter_cosine, knn_purity

    intra, inter = intra_inter_cosine(embeds, assigns)
    ratio = intra / inter if inter and inter != 0 and not np.isnan(inter) else float("nan")
    # KNN purity on the NMF embedding is cheating in a sense (we clustered
    # there), but same for DINO4DSTEM — both are fair comparisons if we report
    # the same metric.
    knn = knn_purity(embeds, assigns, k=10, max_query=1500)
    nmi = float(normalized_mutual_info_score(assigns, dino_assigns)) \
        if dino_assigns is not None and len(dino_assigns) == len(assigns) else float("nan")
    return dict(
        intra_class_cosine=intra,
        inter_class_cosine=inter,
        intra_over_inter=ratio,
        KNN_purity_k10=knn,
        NMI_vs_dino=nmi,
    )


# =========================================================================
# 4. Figures
# =========================================================================

def plot_silhouette(k_silhouette, k_star, outpath):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ks = [k for k, _ in k_silhouette]; sil = [s for _, s in k_silhouette]
    ax.plot(ks, sil, "o-", lw=2)
    ax.axvline(k_star, color="red", ls="--", lw=1, label=f"peak K={k_star}")
    ax.set_xlabel("K"); ax.set_ylabel("silhouette (cosine)")
    ax.set_title("NMF + k-means silhouette sweep")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_nmf_components(H: np.ndarray, polar_size: int, outpath: str, n_show: int = 9):
    import math
    n = min(n_show, H.shape[0])
    cols = 3; rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.atleast_2d(axes).ravel()
    for k in range(n):
        h = H[k].reshape(polar_size, polar_size)
        axes[k].imshow(h, cmap="inferno", aspect="auto")
        axes[k].set_title(f"NMF component {k}", fontsize=8)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    for k in range(n, len(axes)):
        axes[k].set_axis_off()
    fig.suptitle("Top NMF components (polar space)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_class_map_nmf(assigns, scan_shape, outpath, k, title_prefix=""):
    m = assigns.reshape(scan_shape)
    Ny, Nx = scan_shape
    aspect = Nx / max(Ny, 1)
    if aspect >= 1:
        fig_w, fig_h = 10, max(3.5, 10 / aspect)
    else:
        fig_h, fig_w = 10, max(3.5, 10 * aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = "tab10" if k <= 10 else ("tab20" if k <= 20 else "turbo")
    vmax = 9 if k <= 10 else (19 if k <= 20 else k - 1)
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{title_prefix}NMF+k-means at K={k}")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=range(k))
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# 5. Driver
# =========================================================================

def run(sample: str, dino_config_for_comparison: str, n_components: int = 20,
         k_range=(2, 14)):
    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    scan_shape = cfg["scan_shape"]

    out_dir = os.path.join("runs", sample, "baseline_nmf_kmeans", "eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] [{sample}] preprocessing patterns -> polar vectors",
          flush=True)
    t0 = time.perf_counter()
    X = patterns_to_polar_vectors(dataset, polar_size=192)
    print(f"[{datetime.now():%H:%M:%S}] [{sample}] preprocessed in {time.perf_counter() - t0:.1f}s"
          f", shape {X.shape}", flush=True)

    # Pull DINO4DSTEM winner assignments for comparison.
    dino_path = os.path.join("runs", sample, dino_config_for_comparison,
                               "eval", "inference.npz")
    dino_assigns = None
    target_k = 10
    if os.path.exists(dino_path):
        dino_inf = np.load(dino_path)
        dino_assigns = dino_inf["assigns"]
        target_k = int(dino_assigns.max()) + 1
        print(f"[{sample}] DINO4DSTEM winner has K_active={target_k}", flush=True)
    else:
        print(f"[{sample}] no DINO4DSTEM reference at {dino_path}; skipping NMI", flush=True)

    print(f"[{datetime.now():%H:%M:%S}] [{sample}] NMF({n_components}) + k-means sweep K={k_range}",
          flush=True)
    t0 = time.perf_counter()
    res = run_nmf_kmeans(X, n_components=n_components, k_range=k_range,
                           target_k=target_k)
    print(f"[{datetime.now():%H:%M:%S}] [{sample}] NMF+kmeans done in "
          f"{time.perf_counter() - t0:.1f}s  peak K*={res['k_star']}  "
          f"silhouette*={res['silhouette_star']:.3f}", flush=True)

    # Figures.
    plot_silhouette(res["k_silhouette"], res["k_star"],
                      os.path.join(out_dir, "fig_silhouette_vs_k.png"))
    plot_nmf_components(res["H"], 192,
                          os.path.join(out_dir, "fig_nmf_components.png"))
    plot_class_map_nmf(res["assigns_star"], scan_shape,
                         os.path.join(out_dir, "fig_class_map_kstar.png"),
                         res["k_star"], title_prefix="")
    if res["assigns_target"] is not None:
        plot_class_map_nmf(res["assigns_target"], scan_shape,
                             os.path.join(out_dir, "fig_class_map_targetk.png"),
                             res["target_k"],
                             title_prefix=f"(match DINO4DSTEM K={res['target_k']})  ")

    # Metrics (at k_star and at target_k).
    metrics_kstar = nmi_and_cosine_metrics(res["assigns_star"], res["nmf_W"],
                                             dino_assigns)
    metrics_target = (nmi_and_cosine_metrics(res["assigns_target"],
                                               res["nmf_W"], dino_assigns)
                       if res["assigns_target"] is not None else None)

    out = dict(
        sample=sample,
        dino_config_for_comparison=dino_config_for_comparison,
        n_components=n_components,
        k_range=list(k_range),
        k_star=res["k_star"],
        silhouette_star=res["silhouette_star"],
        target_k=res["target_k"],
        silhouette_curve=res["k_silhouette"],
        metrics_at_k_star=metrics_kstar,
        metrics_at_target_k=metrics_target,
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    np.savez_compressed(os.path.join(out_dir, "inference.npz"),
                        assigns_kstar=res["assigns_star"],
                        assigns_target_k=(res["assigns_target"]
                                           if res["assigns_target"] is not None
                                           else np.zeros(0)),
                        nmf_W=res["nmf_W"])
    print(f"[{datetime.now():%H:%M:%S}] [{sample}] baseline done -> {out_dir}",
          flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="Na007b")
    ap.add_argument("--dino-config", default="sweep_polar_centroid",
                    help="DINO4DSTEM config folder to fetch reference assignments from")
    ap.add_argument("--n-components", type=int, default=20)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=14)
    args = ap.parse_args(argv)
    run(args.sample, args.dino_config,
         n_components=args.n_components,
         k_range=(args.k_min, args.k_max))


if __name__ == "__main__":
    main()
