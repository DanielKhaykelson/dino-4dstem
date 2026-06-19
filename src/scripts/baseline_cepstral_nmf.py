"""baseline_cepstral_nmf.py -- cepstral-features + NMF + k-means baseline,
matching the npj Comp Mater 2024 (Yoo et al.) workflow.

Pipeline:
    1. Per pattern: compute cepstrum = |IFFT(log(|I(k)| + 1))|^2.
    2. Central-patch crop (default 32x32 = 1024 features) -- low-frequency
        cepstral coefficients carry the lattice / texture info.
    3. Single-fit NMF with K_components on the (N, 1024) feature matrix.
    4. k-means on the NMF weight matrix to get final cluster assignments.

Times each stage and reports total wall-clock so we can fairly compare
against DINO4DSTEM training + transfer cost.

Usage:
    python baseline_cepstral_nmf.py --sample Na007b --K 6
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from data import SAMPLES, LoadPRZ


def compute_cepstrum_patch(img: np.ndarray, patch: int = 32) -> np.ndarray:
    """Power cepstrum: |IFFT(log(|I| + 1))|^2, central patch."""
    F = np.fft.fft2(np.log(np.abs(img.astype(np.float32)) + 1.0))
    C = np.abs(np.fft.ifft2(np.log(np.abs(F) + 1.0))) ** 2
    C = np.fft.fftshift(C)
    H, W = C.shape
    cy, cx = H // 2, W // 2
    half = patch // 2
    return C[cy - half:cy + half, cx - half:cx + half].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--K", type=int, default=None,
                    help="fixed number of NMF components and k-means clusters; "
                         "if omitted, auto-select via silhouette over --k-range")
    ap.add_argument("--k-range", type=int, nargs=2, default=[2, 12],
                    help="K range for auto-select sweep (inclusive), default 2 12")
    ap.add_argument("--patch", type=int, default=32,
                    help="cepstrum central-patch side (default 32 -> 1024 features)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="override sample vmax")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    K_used = args.K  # may be None until auto-select picks K*

    cfg = SAMPLES[args.sample]
    vmax = args.vmax if args.vmax is not None else cfg["vmax"]
    K_label = "Kauto" if K_used is None else f"K{K_used}"
    out_dir = args.out_dir or os.path.join("runs", "_baselines",
                                              f"cepstral_nmf_{args.sample}_{K_label}_v{vmax}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cepstral-nmf] sample={args.sample}  K={'auto' if K_used is None else K_used}  "
          f"patch={args.patch}  vmax={vmax}  out={out_dir}", flush=True)

    t_total = time.perf_counter()

    # 1. load raw cube (mmap)
    t0 = time.perf_counter()
    ds = LoadPRZ(cfg["path"], resize=192, vmax=vmax)
    Ny, Nx = cfg["scan_shape"]
    N = len(ds)
    print(f"  N={N}  scan={Ny}x{Nx}", flush=True)
    t_load = time.perf_counter() - t0

    # 2. cepstral features per pattern
    t0 = time.perf_counter()
    features = np.zeros((N, args.patch * args.patch), dtype=np.float32)
    for i in range(N):
        raw = ds.get_raw(i).astype(np.float32)
        norm = np.clip(raw / vmax, 0.0, 1.0)
        cep_patch = compute_cepstrum_patch(norm, patch=args.patch)
        features[i] = cep_patch.flatten()
        if i % 1000 == 0:
            print(f"    cepstrum {i}/{N}", flush=True)
    t_cepstrum = time.perf_counter() - t0
    print(f"  cepstrum done in {t_cepstrum:.1f}s  features shape={features.shape}",
          flush=True)

    # 3+4. NMF + k-means with optional silhouette auto-K sweep.
    from sklearn.decomposition import NMF
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    silhouette_curve = None
    nmf_recon_err = None
    if K_used is None:
        kmin, kmax = int(args.k_range[0]), int(args.k_range[1])
        ks = list(range(kmin, kmax + 1))
        print(f"  AUTO-K sweep over {ks} via silhouette on NMF(W) + k-means",
              flush=True)
        results = []
        # Sample subset for silhouette to keep cost bounded on large cubes
        rng = np.random.default_rng(0)
        sub = rng.choice(N, size=min(N, 5000), replace=False)
        t_sweep0 = time.perf_counter()
        for k in ks:
            ts0 = time.perf_counter()
            nmf_k = NMF(n_components=k, init="nndsvd", max_iter=400,
                         random_state=42, tol=1e-4)
            W_k = nmf_k.fit_transform(features)
            km_k = KMeans(n_clusters=k, n_init=10, random_state=42)
            assigns_k = km_k.fit_predict(W_k)
            sil = float(silhouette_score(W_k[sub], assigns_k[sub], metric="cosine"))
            t_iter = time.perf_counter() - ts0
            print(f"    K={k:2d}  silhouette={sil:.4f}  recon={nmf_k.reconstruction_err_:.4g}  "
                  f"({t_iter:.1f}s)", flush=True)
            results.append({"K": k, "silhouette": sil,
                              "reconstruction_err": float(nmf_k.reconstruction_err_),
                              "W": W_k, "assigns": assigns_k, "H": nmf_k.components_})
        # Pick K with highest silhouette
        best = max(results, key=lambda r: r["silhouette"])
        K_star = best["K"]
        W = best["W"]; H = best["H"]; assigns = best["assigns"]
        nmf_recon_err = best["reconstruction_err"]
        silhouette_curve = [(r["K"], r["silhouette"], r["reconstruction_err"])
                              for r in results]
        t_nmf = time.perf_counter() - t_sweep0
        t_kmeans = 0.0  # rolled into sweep
        print(f"  AUTO-K selected K*={K_star}  "
              f"(best silhouette={best['silhouette']:.4f})", flush=True)
        # plot silhouette curve
        fig, ax = plt.subplots(figsize=(7, 4))
        sk = [r["K"] for r in results]
        ss = [r["silhouette"] for r in results]
        ax.plot(sk, ss, marker="o")
        ax.axvline(K_star, color="red", linestyle="--", alpha=0.6,
                    label=f"K*={K_star}")
        ax.set_xlabel("K"); ax.set_ylabel("silhouette (cosine)")
        ax.set_title(f"{args.sample} cepstral+NMF+kmeans  silhouette sweep")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_silhouette_vs_k.png"),
                     dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        K_used = K_star
    else:
        t0 = time.perf_counter()
        print(f"  fitting NMF(K={K_used}) on (N={N}, D={features.shape[1]})...",
              flush=True)
        nmf = NMF(n_components=K_used, init="nndsvd", max_iter=400,
                   random_state=42, tol=1e-4)
        W = nmf.fit_transform(features)
        H = nmf.components_
        t_nmf = time.perf_counter() - t0
        print(f"  NMF done in {t_nmf:.1f}s  W={W.shape}  H={H.shape}  "
              f"reconstruction_err={nmf.reconstruction_err_:.4g}", flush=True)
        t0 = time.perf_counter()
        print(f"  k-means(K={K_used}) on W...", flush=True)
        km = KMeans(n_clusters=K_used, n_init=10, random_state=42)
        assigns = km.fit_predict(W)
        t_kmeans = time.perf_counter() - t0
        print(f"  k-means done in {t_kmeans:.1f}s", flush=True)
        nmf_recon_err = float(nmf.reconstruction_err_)

    t_pipeline = time.perf_counter() - t_total

    # -------- outputs --------
    counts = np.bincount(assigns, minlength=K_used).tolist()
    print(f"  cluster sizes: {counts}", flush=True)

    # class map
    from matplotlib.colors import ListedColormap, BoundaryNorm
    K_act = int(np.unique(assigns).size)
    base = plt.get_cmap("tab10").colors[:K_used]
    cmap = ListedColormap(base, name=f"K{K_used}")
    norm = BoundaryNorm(np.arange(K_used + 1) - 0.5, K_used)
    class_map = assigns.reshape(Ny, Nx)
    aspect = Nx / max(Ny, 1)
    if aspect > 1:
        fig_h = 5.0; fig_w = min(15, fig_h * aspect)
    else:
        fig_w = 5.0; fig_h = min(12, fig_w / aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(class_map, cmap=cmap, norm=norm, aspect="equal",
                    interpolation="nearest")
    ax.set_title(
        f"{args.sample} cepstral+NMF+kmeans K={K_used} K_act={K_act}",
        fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                         ticks=list(range(K_used)), shrink=0.9)
    cbar.set_label("cluster", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_class_map.png"), dpi=200,
                 bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # NMF components heatmaps
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for k in range(min(K_used, len(axes))):
        h = H[k].reshape(args.patch, args.patch)
        axes[k].imshow(h, cmap="inferno", aspect="auto",
                        interpolation="nearest")
        axes[k].set_title(f"comp {k}", fontsize=9)
        axes[k].set_xticks([]); axes[k].set_yticks([])
    for j in range(K_used, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle(f"NMF components (cepstral central {args.patch}x{args.patch} patch)",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_nmf_components.png"), dpi=160,
                 bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # save data
    np.savez(os.path.join(out_dir, "inference.npz"),
              assigns=assigns, W=W, H=H,
              features_shape=np.array(features.shape))
    summary = {
        "sample": args.sample,
        "K": int(K_used),
        "patch": int(args.patch),
        "vmax": float(vmax),
        "N_patterns": int(N),
        "scan_shape": list(cfg["scan_shape"]),
        "feature_dim": int(features.shape[1]),
        "cluster_sizes": [int(c) for c in counts],
        "K_active": int(K_act),
        "nmf_reconstruction_err": float(nmf_recon_err) if nmf_recon_err is not None else None,
        "timing_seconds": {
            "load_dataset_mmap": float(t_load),
            "cepstrum_per_pattern": float(t_cepstrum),
            "nmf_fit": float(t_nmf),
            "kmeans": float(t_kmeans),
            "total_pipeline": float(t_pipeline),
        },
        "auto_K": args.K is None,
        "k_range": list(args.k_range) if args.K is None else None,
        "silhouette_curve": [{"K": k, "silhouette": s, "recon": r}
                                for (k, s, r) in (silhouette_curve or [])],
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[cepstral-nmf] {args.sample} done.", flush=True)
    print(f"  total pipeline: {t_pipeline:.1f}s "
          f"(cepstrum={t_cepstrum:.1f}s, NMF={t_nmf:.1f}s, kmeans={t_kmeans:.1f}s)",
          flush=True)
    print(f"  output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
