"""analyze_bottom_layers.py -- separation analysis for the two bottom film
classes of EuInAs (or any layered sample).

Pipeline (mirrors Chapter 3 EuInAs analysis, then adds more metrics):

  1. Auto-detect the "bottom two" film classes from the class map
     (overridable via --classA/--classB).
  2. Compute confidence-weighted class-mean diffraction patterns.
  3. Analytic strain via analyze_strain.strain_between_classes:
        rotation, scale_major/minor, shear, eps_major/minor, n_inliers, ...
  4. Polar comparison (matches Fig. 3.8 of the chapter):
        polar transforms of A, B, and (A-B)/(A+B), plus angular intensity
        traces at fixed radii to show in-plane misorientation.
  5. Cartesian normalized difference (Fig. 3.7) before and after warp.
  6. Additional separation metrics:
        - SSIM between class means
        - L2 / cosine distance between (raw) means and (warped) means
        - Embedding-space centroid cosine + intra-class spread (radius)
        - 1D radial profile cosine
        - Per-class membership counts and spatial mean position

Output: <run_dir>/eval/strain_bottom_layers/
    fig_classes_overview.png
    fig_diff_cartesian.png
    fig_polar_comparison.png
    fig_strain_<A>_vs_<B>.png        (delegated to analyze_strain)
    metrics_summary.json
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys, json, argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SAMPLES, LoadPRZ
from analyze_strain import strain_between_classes


def _normalized_diff(A: np.ndarray, B: np.ndarray, eps: float = 1e-6):
    return (A - B) / (A + B + eps)


def _ssim(A: np.ndarray, B: np.ndarray):
    try:
        from skimage.metrics import structural_similarity
        # use the common range based on both arrays
        rng = float(max(A.max() - A.min(), B.max() - B.min(), 1e-6))
        return float(structural_similarity(A, B, data_range=rng))
    except Exception:
        return None


def _polar_warp(img: np.ndarray, n_theta: int = 360, n_r: "int | None" = None):
    """Cartesian-to-polar via OpenCV warpPolar; returns (n_theta, n_r) array.
    Center is the image center."""
    import cv2
    H, W = img.shape
    cx, cy = W / 2.0, H / 2.0
    if n_r is None:
        n_r = int(min(cx, cy))
    polar = cv2.warpPolar(img.astype(np.float32), (n_r, n_theta), (cx, cy),
                            n_r, cv2.WARP_POLAR_LINEAR)
    # cv2 returns (n_theta, n_r) when dsize=(n_r, n_theta)
    return polar


def _angular_traces(polar: np.ndarray, radii_frac=(0.30, 0.45, 0.60, 0.75)):
    """Return list of (frac, trace) — intensity vs theta at fixed radial fractions."""
    n_theta, n_r = polar.shape
    return [(f, polar[:, int(round(f * (n_r - 1)))]) for f in radii_frac]


def _membership_centroid_xy(assigns: np.ndarray, scan_shape):
    """Mean (y, x) of each class's membership on the scan grid."""
    Ny, Nx = scan_shape
    K = int(assigns.max() + 1)
    yx = np.zeros((K, 2), dtype=np.float64)
    counts = np.bincount(assigns, minlength=K)
    yy, xx = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
    flat = assigns.reshape(Ny, Nx)
    for c in range(K):
        m = (flat == c)
        if m.sum() == 0:
            yx[c] = (np.nan, np.nan)
        else:
            yx[c] = (yy[m].mean(), xx[m].mean())
    return yx, counts


def _pick_bottom_two(yx, counts, scan_shape, min_frac=0.03, exclude=()):
    """Pick the two classes with the largest mean-y (bottom of scan)
    that satisfy a minimum fraction of total pixels and aren't excluded.
    """
    Ny, Nx = scan_shape
    total = int(counts.sum())
    candidates = []
    for c, ((y, x), n) in enumerate(zip(yx, counts)):
        if c in exclude:
            continue
        if n / total < min_frac:
            continue
        if not np.isfinite(y):
            continue
        candidates.append((c, y, n))
    candidates.sort(key=lambda t: -t[1])  # highest y (bottom) first
    if len(candidates) < 2:
        raise ValueError("could not find two bottom-side classes "
                          f"(found {len(candidates)})")
    return candidates[0][0], candidates[1][0]


def _confidence_weighted_mean(dataset, soft_probs, assigns, c, n_top=300):
    idx = np.where(assigns == c)[0]
    if len(idx) == 0:
        raise ValueError(f"class {c} empty")
    scores = soft_probs[idx, c]
    top = idx[np.argsort(-scores)[:min(n_top, len(idx))]]
    patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0).astype(np.float32)
    w = soft_probs[top, c].astype(np.float32)
    return (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12), top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="path to the trained run dir, e.g. "
                         "runs/_winner_followup/EuInAs_K6_50ep")
    ap.add_argument("--sample", default="EuInAs_B100")
    ap.add_argument("--classA", type=int, default=None,
                    help="override: dense class id for A (lower of the two)")
    ap.add_argument("--classB", type=int, default=None,
                    help="override: dense class id for B")
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="class ids to exclude when auto-picking (e.g. vacuum, substrate)")
    ap.add_argument("--n-top", type=int, default=300)
    ap.add_argument("--blob-min-sigma", type=float, default=2.0)
    ap.add_argument("--blob-max-sigma", type=float, default=10.0)
    ap.add_argument("--blob-threshold", type=float, default=0.05)
    ap.add_argument("--match-gate", type=float, default=10.0)
    ap.add_argument("--beam-mask-radius", type=int, default=40)
    ap.add_argument("--out-name", default="strain_bottom_layers",
                    help="subdir under run_dir/eval/ to write outputs to "
                         "(use to keep multiple pair analyses separate)")
    args = ap.parse_args()

    cfg = SAMPLES[args.sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    inf = np.load(os.path.join(args.run_dir, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]
    embeds = inf["embeds"]
    scan_shape = cfg["scan_shape"]
    K = soft_probs.shape[1]

    out_dir = os.path.join(args.run_dir, "eval", args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- pick A, B ----
    yx, counts = _membership_centroid_xy(assigns, scan_shape)
    if args.classA is not None and args.classB is not None:
        cA, cB = int(args.classA), int(args.classB)
    else:
        cA, cB = _pick_bottom_two(yx, counts, scan_shape, min_frac=0.03,
                                   exclude=set(args.exclude))
        # ensure A is the lower (largest mean y)
        if yx[cA][0] < yx[cB][0]:
            cA, cB = cB, cA
    print(f"[bottom-layers] A=class {cA} (mean y={yx[cA][0]:.1f}, "
          f"N={int(counts[cA])})", flush=True)
    print(f"[bottom-layers] B=class {cB} (mean y={yx[cB][0]:.1f}, "
          f"N={int(counts[cB])})", flush=True)

    # ---- class means (raw Cartesian) ----
    mean_A, top_A = _confidence_weighted_mean(dataset, soft_probs, assigns,
                                                cA, args.n_top)
    mean_B, top_B = _confidence_weighted_mean(dataset, soft_probs, assigns,
                                                cB, args.n_top)

    # ---- delegate analytic strain ----
    strain_dir = os.path.join(out_dir, f"strain_p{cA}_vs_p{cB}")
    strain = strain_between_classes(
        mean_A, mean_B, strain_dir,
        label_A=f"p{cA}", label_B=f"p{cB}",
        blob_min_sigma=args.blob_min_sigma,
        blob_max_sigma=args.blob_max_sigma,
        blob_threshold=args.blob_threshold,
        match_gate_px=args.match_gate,
        # --beam-mask-radius now applies to BOTH blob detection and the
        # visualization. Pass 0 to disable masking entirely.
        beam_mask_radius=args.beam_mask_radius,
        display_beam_mask_radius=args.beam_mask_radius,
    )

    # ---- additional separation metrics ----
    # 1) SSIM (structural similarity)
    ssim_AB = _ssim(mean_A, mean_B)
    # 2) Cosine + L2 of flattened class means (raw, on the diffraction patterns)
    a_flat = mean_A.flatten().astype(np.float64)
    b_flat = mean_B.flatten().astype(np.float64)
    cos_AB = float((a_flat @ b_flat) / (np.linalg.norm(a_flat) *
                                          np.linalg.norm(b_flat) + 1e-12))
    l2_AB = float(np.linalg.norm(a_flat - b_flat) / np.sqrt(a_flat.size))
    # 3) Embedding-space centroid cosine + intra-class radius
    emb_A = embeds[assigns == cA]
    emb_B = embeds[assigns == cB]
    cA_emb = emb_A.mean(0); cA_emb /= np.linalg.norm(cA_emb) + 1e-12
    cB_emb = emb_B.mean(0); cB_emb /= np.linalg.norm(cB_emb) + 1e-12
    centroid_cos = float(cA_emb @ cB_emb)
    intra_A = float((emb_A @ cA_emb).mean())
    intra_B = float((emb_B @ cB_emb).mean())
    # 4) 1D radial-profile cosine (load the SAXS-treated profiles if present)
    rad_path_candidates = [
        cfg["path"][:-4] + ".radial.npy",
        cfg["path"] + ".radial.npy",
    ]
    radial_cos = None
    for rp in rad_path_candidates:
        if os.path.exists(rp):
            R = np.load(rp)
            rA = R[top_A].mean(0); rA /= np.linalg.norm(rA) + 1e-12
            rB = R[top_B].mean(0); rB /= np.linalg.norm(rB) + 1e-12
            radial_cos = float(rA @ rB)
            break

    # ---- Cartesian difference image (Fig 3.7) ----
    bm = ((np.indices(mean_A.shape) - np.array(mean_A.shape).reshape(2, 1, 1) / 2.0) ** 2).sum(0) > args.beam_mask_radius ** 2
    mean_A_n = mean_A / max(mean_A.max(), 1e-6)
    mean_B_n = mean_B / max(mean_B.max(), 1e-6)
    diff_cart = _normalized_diff(mean_A_n, mean_B_n) * bm

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(np.log1p(mean_A_n * 50) * bm, cmap="inferno"); axes[0].set_title(f"class A = p{cA} (N={int(counts[cA])})")
    axes[1].imshow(np.log1p(mean_B_n * 50) * bm, cmap="inferno"); axes[1].set_title(f"class B = p{cB} (N={int(counts[cB])})")
    im = axes[2].imshow(diff_cart, cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title(r"$(A - B) / (A + B)$")
    fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_diff_cartesian.png"),
                 dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Polar comparison (Fig 3.8): polar A, polar B, polar diff, angular traces ----
    polar_A = _polar_warp(mean_A_n)
    polar_B = _polar_warp(mean_B_n)
    polar_diff = _normalized_diff(polar_A, polar_B)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                              gridspec_kw={"height_ratios": [3, 2]})
    axes[0, 0].imshow(np.log1p(polar_A * 50), cmap="inferno", aspect="auto")
    axes[0, 0].set_title(f"polar A = p{cA}")
    axes[0, 0].set_xlabel("r"); axes[0, 0].set_ylabel(r"$\theta$ (deg)")
    axes[0, 1].imshow(np.log1p(polar_B * 50), cmap="inferno", aspect="auto")
    axes[0, 1].set_title(f"polar B = p{cB}")
    axes[0, 1].set_xlabel("r")
    im = axes[0, 2].imshow(polar_diff, cmap="bwr", vmin=-1, vmax=1, aspect="auto")
    axes[0, 2].set_title(r"polar $(A-B)/(A+B)$")
    axes[0, 2].set_xlabel("r")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.04, pad=0.02)

    # angular traces — show shifts as evidence of in-plane misorientation
    traces_A = _angular_traces(polar_A)
    traces_B = _angular_traces(polar_B)
    n_theta = polar_A.shape[0]
    theta_deg = np.linspace(0, 360, n_theta, endpoint=False)
    ax_tr = plt.subplot(2, 1, 2)
    for (fA, tA), (fB, tB) in zip(traces_A, traces_B):
        ax_tr.plot(theta_deg, tA / max(tA.max(), 1e-6),
                    label=f"A r={fA:.2f}", alpha=0.85, lw=1.0)
        ax_tr.plot(theta_deg, tB / max(tB.max(), 1e-6),
                    label=f"B r={fB:.2f}", alpha=0.85, lw=1.0, ls="--")
    ax_tr.set_xlabel(r"$\theta$ (deg)")
    ax_tr.set_ylabel("normalized intensity")
    ax_tr.set_title("Angular intensity traces at fixed radii — peak shift = in-plane rotation")
    ax_tr.legend(fontsize=8, ncol=4)
    ax_tr.grid(alpha=0.3)
    # Hide the original axes that subplot reused space from
    for ax in axes[1, :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_polar_comparison.png"),
                 dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Class spatial overview ----
    fig, ax = plt.subplots(figsize=(13, 3))
    cmap_assign = plt.get_cmap("tab10", K)
    flat = assigns.reshape(scan_shape)
    ax.imshow(flat, cmap=cmap_assign, vmin=-0.5, vmax=K - 0.5,
               interpolation="nearest")
    ax.set_title(f"class map  (A={cA} red ring, B={cB} green ring)")
    # mark centroids
    for c, color in [(cA, "red"), (cB, "lime")]:
        y, x = yx[c]
        ax.plot(x, y, marker="o", mfc="none", mec=color, ms=14, mew=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_classes_overview.png"),
                 dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Summary JSON ----
    summary = {
        "run_dir": args.run_dir,
        "sample": args.sample,
        "class_A": int(cA),
        "class_B": int(cB),
        "yx_A": [float(yx[cA][0]), float(yx[cA][1])],
        "yx_B": [float(yx[cB][0]), float(yx[cB][1])],
        "n_A": int(counts[cA]),
        "n_B": int(counts[cB]),
        "strain": strain,
        "ssim_raw_means": ssim_AB,
        "cosine_raw_means": cos_AB,
        "l2_raw_means_per_pixel": l2_AB,
        "embedding_centroid_cosine": centroid_cos,
        "embedding_intra_A_mean_cos": intra_A,
        "embedding_intra_B_mean_cos": intra_B,
        "radial_profile_cosine": radial_cos,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n=== separation summary ===")
    for k, v in summary.items():
        if k == "strain":
            def _f(x, fmt):
                return format(x, fmt) if x is not None else "n/a"
            print(f"  strain.status       = {v.get('status')}")
            print(f"  strain.rotation_deg = {_f(v.get('rotation_deg'), '.4f')}")
            print(f"  strain.eps_major    = {_f(v.get('eps_major'), '.5f')}")
            print(f"  strain.eps_minor    = {_f(v.get('eps_minor'), '.5f')}")
            print(f"  strain.shear_xy     = {_f(v.get('shear_xy'), '.5f')}")
            print(f"  strain.n_inliers    = {v.get('n_inliers')}")
            continue
        print(f"  {k} = {v}")
    print(f"\noutputs in: {out_dir}")


if __name__ == "__main__":
    main()
