"""analyze_class_pair.py -- compare two classes from a trained run via
multiple, independent diffraction-space analyses. Class averages are
computed RAW (no masking) per the user's request.

Outputs to <run_dir>/eval/pair_p{A}_vs_p{B}/:
    fig_classes_overview.png        class map with the pair circled
    fig_diff_cartesian.png          A, B, (A-B)/(A+B) (Cartesian)
    fig_polar_comparison.png        polar A, polar B, polar diff, +
                                       angular intensity traces at fixed r
    fig_radial_profiles.png         1D I(r) overlay + difference + peak hits
    fig_peak_intensities.png        scatter of A vs B Bragg-peak intensity
                                       (matched peaks via mutual NN of
                                       (r, theta) detections)
    fig_strain_pA_vs_pB.png         RANSAC affine + warp/diff (delegated
                                       to analyze_strain)
    metrics_summary.json            all numbers, including line/spot/ring
                                       Gini scores

Usage:
    python analyze_class_pair.py --run-dir runs/_determinism_check/EuInAs_B100 \
        --sample EuInAs_B100 --classA 0 --classB 3
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import argparse, os, sys, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from data import SAMPLES, LoadPRZ
from analyze_strain import strain_between_classes
from analyze_bottom_layers import (
    _normalized_diff, _ssim, _polar_warp, _angular_traces,
    _membership_centroid_xy, _confidence_weighted_mean,
)


def _radial_profile_from_polar(polar: np.ndarray, eps: float = 1e-9):
    """1D I(r) by summing the polar-warped image over theta."""
    return polar.sum(axis=0)


def _detect_radial_peaks(I_r: np.ndarray, prominence_frac: float = 0.05,
                          smooth_sigma: float = 2.0,
                          r_min: int = 20, r_max: "int | None" = None):
    """Find peaks in I(r). Smooth, then SciPy find_peaks with a prominence
    set as a fraction of (max - median).
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    if r_max is None:
        r_max = I_r.size - 1
    seg = gaussian_filter1d(I_r[r_min:r_max], sigma=smooth_sigma)
    base = np.median(seg)
    span = max(seg.max() - base, 1e-12)
    prom = prominence_frac * span
    peaks, props = find_peaks(seg, prominence=prom)
    return peaks + r_min, props["prominences"]


def _detect_2d_peaks(img: np.ndarray, beam_mask_radius: int = 40,
                      min_sigma: float = 2.0, max_sigma: float = 10.0,
                      threshold: float = 0.05):
    """LoG blob detection on raw image (delegated to analyze_strain helper)."""
    from analyze_strain import detect_blobs
    return detect_blobs(img, min_sigma=min_sigma, max_sigma=max_sigma,
                         threshold=threshold,
                         beam_mask_radius=beam_mask_radius)


def _match_blobs_mutual_nn(blobs_A: np.ndarray, blobs_B: np.ndarray,
                            max_dist: float = 6.0):
    """Match blobs (y, x, sigma) between two patterns by mutual nearest
    neighbor with a max-distance gate. Returns idx_A, idx_B (matched).
    """
    if len(blobs_A) == 0 or len(blobs_B) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    Pa = blobs_A[:, :2]; Pb = blobs_B[:, :2]
    d_ab = np.sqrt(((Pa[:, None, :] - Pb[None, :, :]) ** 2).sum(-1))
    nn_ba = d_ab.argmin(axis=0)        # for each B, closest A
    nn_ab = d_ab.argmin(axis=1)        # for each A, closest B
    matched = []
    for i in range(len(Pa)):
        j = nn_ab[i]
        if nn_ba[j] == i and d_ab[i, j] < max_dist:
            matched.append((i, j))
    if not matched:
        return np.array([], dtype=int), np.array([], dtype=int)
    arr = np.array(matched)
    return arr[:, 0], arr[:, 1]


def _peak_intensity_at(img: np.ndarray, blobs: np.ndarray,
                        radius: int = 3) -> np.ndarray:
    """Mean intensity in a `radius`-pixel box around each (y, x) blob center."""
    H, W = img.shape
    vals = []
    for y, x, _ in blobs:
        y, x = int(round(y)), int(round(x))
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        vals.append(float(img[y0:y1, x0:x1].mean()))
    return np.array(vals, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--classA", type=int, required=True)
    ap.add_argument("--classB", type=int, required=True)
    ap.add_argument("--n-top", type=int, default=300)
    ap.add_argument("--blob-min-sigma", type=float, default=2.0)
    ap.add_argument("--blob-max-sigma", type=float, default=10.0)
    ap.add_argument("--blob-threshold", type=float, default=0.05)
    ap.add_argument("--match-gate", type=float, default=10.0)
    ap.add_argument("--beam-mask-radius", type=int, default=40,
                     help="display/blob beam mask, in raw px (NOT applied to "
                          "the class-mean computation)")
    args = ap.parse_args()

    cA, cB = int(args.classA), int(args.classB)
    cfg = SAMPLES[args.sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    inf = np.load(os.path.join(args.run_dir, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]
    embeds = inf["embeds"]
    scan_shape = cfg["scan_shape"]
    K = soft_probs.shape[1]

    out_dir = os.path.join(args.run_dir, "eval", f"pair_p{cA}_vs_p{cB}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[pair] {args.sample} p{cA} vs p{cB}", flush=True)
    yx, counts = _membership_centroid_xy(assigns, scan_shape)
    print(f"  p{cA}: mean(y,x)={yx[cA]}  N={counts[cA]}", flush=True)
    print(f"  p{cB}: mean(y,x)={yx[cB]}  N={counts[cB]}", flush=True)

    # ---- RAW class means (no masking) ----
    mean_A, top_A = _confidence_weighted_mean(dataset, soft_probs, assigns,
                                                cA, args.n_top)
    mean_B, top_B = _confidence_weighted_mean(dataset, soft_probs, assigns,
                                                cB, args.n_top)

    # ---- delegated analytic strain (RANSAC affine) ----
    strain_dir = os.path.join(out_dir, f"strain_p{cA}_vs_p{cB}")
    strain = strain_between_classes(
        mean_A, mean_B, strain_dir,
        label_A=f"p{cA}", label_B=f"p{cB}",
        blob_min_sigma=args.blob_min_sigma,
        blob_max_sigma=args.blob_max_sigma,
        blob_threshold=args.blob_threshold,
        match_gate_px=args.match_gate,
    )

    # ---- separation metrics on raw means ----
    ssim_AB = _ssim(mean_A, mean_B)
    a_flat = mean_A.flatten().astype(np.float64)
    b_flat = mean_B.flatten().astype(np.float64)
    cos_AB = float((a_flat @ b_flat) / (np.linalg.norm(a_flat) *
                                          np.linalg.norm(b_flat) + 1e-12))
    l2_AB = float(np.linalg.norm(a_flat - b_flat) / np.sqrt(a_flat.size))
    emb_A = embeds[assigns == cA]; emb_B = embeds[assigns == cB]
    cAv = emb_A.mean(0); cAv /= np.linalg.norm(cAv) + 1e-12
    cBv = emb_B.mean(0); cBv /= np.linalg.norm(cBv) + 1e-12
    centroid_cos = float(cAv @ cBv)
    intra_A = float((emb_A @ cAv).mean())
    intra_B = float((emb_B @ cBv).mean())

    # ---- Cartesian normalized difference ----
    bm_disp = ((np.indices(mean_A.shape)
                  - np.array(mean_A.shape).reshape(2, 1, 1) / 2.0) ** 2
                ).sum(0) > args.beam_mask_radius ** 2
    A_n = mean_A / max(mean_A.max(), 1e-6)
    B_n = mean_B / max(mean_B.max(), 1e-6)
    diff_cart = _normalized_diff(A_n, B_n) * bm_disp
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(np.log1p(A_n * 50) * bm_disp, cmap="inferno")
    axes[0].set_title(f"class A = p{cA}  (N={int(counts[cA])})")
    axes[1].imshow(np.log1p(B_n * 50) * bm_disp, cmap="inferno")
    axes[1].set_title(f"class B = p{cB}  (N={int(counts[cB])})")
    im = axes[2].imshow(diff_cart, cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title(r"$(A - B) / (A + B)$")
    fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_diff_cartesian.png"),
                 dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Polar comparison + angular traces ----
    polar_A = _polar_warp(A_n)
    polar_B = _polar_warp(B_n)
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
    fig.colorbar(im, ax=axes[0, 2], fraction=0.04, pad=0.02)
    traces_A = _angular_traces(polar_A); traces_B = _angular_traces(polar_B)
    n_theta = polar_A.shape[0]
    theta_deg = np.linspace(0, 360, n_theta, endpoint=False)
    ax_tr = plt.subplot(2, 1, 2)
    for (fA, tA), (fB, tB) in zip(traces_A, traces_B):
        ax_tr.plot(theta_deg, tA / max(tA.max(), 1e-6),
                    label=f"A r={fA:.2f}", alpha=0.85, lw=1.0)
        ax_tr.plot(theta_deg, tB / max(tB.max(), 1e-6),
                    label=f"B r={fB:.2f}", alpha=0.85, lw=1.0, ls="--")
    ax_tr.set_xlabel(r"$\theta$ (deg)"); ax_tr.set_ylabel("normalized intensity")
    ax_tr.set_title("Angular intensity traces (peak shifts = in-plane rotation)")
    ax_tr.legend(fontsize=8, ncol=4); ax_tr.grid(alpha=0.3)
    for ax in axes[1, :]: ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_polar_comparison.png"),
                 dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 1D radial profile overlay + peak detection ----
    I_r_A = _radial_profile_from_polar(polar_A)
    I_r_B = _radial_profile_from_polar(polar_B)
    peaks_A, prom_A = _detect_radial_peaks(I_r_A)
    peaks_B, prom_B = _detect_radial_peaks(I_r_B)
    print(f"  radial peaks: A={list(peaks_A)}  B={list(peaks_B)}", flush=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                              gridspec_kw={"height_ratios": [2, 1]})
    r_axis = np.arange(I_r_A.size)
    axes[0].plot(r_axis, I_r_A, color="C0", label=f"A = p{cA}", lw=1.4)
    axes[0].plot(r_axis, I_r_B, color="C3", label=f"B = p{cB}", lw=1.4)
    axes[0].plot(peaks_A, I_r_A[peaks_A], "v", color="C0", ms=8,
                  label="A peaks")
    axes[0].plot(peaks_B, I_r_B[peaks_B], "v", color="C3", ms=8,
                  label="B peaks")
    axes[0].set_xlabel("r (polar bin)"); axes[0].set_ylabel("I(r) = sum_theta")
    axes[0].set_title("1D radial profiles -- peak shifts evidence "
                       "lattice strain between A and B")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # log-ratio
    ratio = np.log10((I_r_A + 1e-9) / (I_r_B + 1e-9))
    axes[1].plot(r_axis, ratio, color="purple", lw=1.2)
    axes[1].axhline(0, color="black", lw=0.6, alpha=0.5)
    axes[1].set_xlabel("r (polar bin)"); axes[1].set_ylabel("log10 I_A / I_B")
    axes[1].set_title("log-ratio: structure-factor differences (peaks "
                       "where A is brighter; troughs where B is brighter)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_radial_profiles.png"),
                 dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Peak-by-peak intensity comparison (matched 2D blobs) ----
    bA = _detect_2d_peaks(mean_A, beam_mask_radius=args.beam_mask_radius,
                            min_sigma=args.blob_min_sigma,
                            max_sigma=args.blob_max_sigma,
                            threshold=args.blob_threshold)
    bB = _detect_2d_peaks(mean_B, beam_mask_radius=args.beam_mask_radius,
                            min_sigma=args.blob_min_sigma,
                            max_sigma=args.blob_max_sigma,
                            threshold=args.blob_threshold)
    iA, iB = _match_blobs_mutual_nn(bA, bB, max_dist=args.match_gate)
    intens_A = _peak_intensity_at(mean_A, bA[iA]) if len(iA) else np.array([])
    intens_B = _peak_intensity_at(mean_B, bB[iB]) if len(iB) else np.array([])
    if len(iA) >= 2:
        # robust slope (least-squares through origin) + Pearson r
        slope = float((intens_A * intens_B).sum() / max((intens_A ** 2).sum(), 1e-12))
        # intensity-ratio summary stats
        ratio_AB = intens_A / np.clip(intens_B, 1e-6, None)
        ratio_log = np.log(ratio_AB)
        ratio_mean = float(ratio_AB.mean())
        ratio_std_log = float(ratio_log.std())
        # Pearson r of log intensities (multiplicative differences are linear in log)
        lA, lB = np.log(np.clip(intens_A, 1e-6, None)), np.log(np.clip(intens_B, 1e-6, None))
        pearson_r = float(np.corrcoef(lA, lB)[0, 1])
    else:
        slope = ratio_mean = ratio_std_log = pearson_r = float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    if len(iA):
        axes[0].scatter(intens_B, intens_A, s=40, alpha=0.7,
                          edgecolors="black", linewidths=0.4)
        m = max(intens_A.max(), intens_B.max())
        axes[0].plot([0, m], [0, m], "--", color="black", alpha=0.5,
                      label="y = x")
        axes[0].set_xlabel(f"intensity at peak in B (p{cB})")
        axes[0].set_ylabel(f"intensity at peak in A (p{cA})")
        axes[0].set_title(f"Per-peak intensity (matched: {len(iA)} peaks)\n"
                           f"slope={slope:.3f}  Pearson r(log)={pearson_r:.3f}")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        # second panel: bar of log ratio per peak
        order = np.argsort(intens_A + intens_B)[::-1]   # brightest first
        axes[1].bar(np.arange(len(order)), np.log(intens_A[order] / np.clip(intens_B[order], 1e-6, None)),
                     color=["C0" if v >= 0 else "C3" for v in
                             np.log(intens_A[order] / np.clip(intens_B[order], 1e-6, None))])
        axes[1].axhline(0, color="black", lw=0.6, alpha=0.5)
        axes[1].set_xlabel("matched peak rank (brightest first)")
        axes[1].set_ylabel("log(I_A / I_B)")
        axes[1].set_title(f"Per-peak log-intensity ratio "
                           f"(mean={np.log(ratio_mean):.3f}, "
                           f"std={ratio_std_log:.3f})")
        axes[1].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "no matched peaks (try lowering threshold)",
                      ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_peak_intensities.png"),
                 dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Class spatial overview ----
    Ny, Nx = scan_shape
    fig, ax = plt.subplots(figsize=(13, 3))
    cmap_assign = plt.get_cmap("tab10", K)
    flat = assigns.reshape(scan_shape)
    ax.imshow(flat, cmap=cmap_assign, vmin=-0.5, vmax=K - 0.5,
               interpolation="nearest")
    ax.set_title(f"class map  (A=p{cA} red ring, B=p{cB} green ring)")
    for c, color in [(cA, "red"), (cB, "lime")]:
        y, x = yx[c]
        ax.plot(x, y, marker="o", mfc="none", mec=color, ms=14, mew=2.5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_classes_overview.png"),
                 dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "run_dir": args.run_dir, "sample": args.sample,
        "class_A": cA, "class_B": cB,
        "yx_A": [float(yx[cA][0]), float(yx[cA][1])],
        "yx_B": [float(yx[cB][0]), float(yx[cB][1])],
        "n_A": int(counts[cA]), "n_B": int(counts[cB]),
        "strain": strain,
        "ssim_raw_means": ssim_AB, "cosine_raw_means": cos_AB,
        "l2_raw_means_per_pixel": l2_AB,
        "embedding_centroid_cosine": centroid_cos,
        "embedding_intra_A_mean_cos": intra_A,
        "embedding_intra_B_mean_cos": intra_B,
        "n_radial_peaks_A": int(len(peaks_A)),
        "n_radial_peaks_B": int(len(peaks_B)),
        "radial_peaks_A": [int(p) for p in peaks_A],
        "radial_peaks_B": [int(p) for p in peaks_B],
        "n_2d_peaks_A": int(len(bA)), "n_2d_peaks_B": int(len(bB)),
        "n_matched_peaks": int(len(iA)),
        "peak_intensity_slope_AoverB": slope,
        "peak_intensity_pearson_r_log": pearson_r,
        "peak_intensity_mean_ratio_AoverB": ratio_mean,
        "peak_intensity_log_ratio_std": ratio_std_log,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n=== summary ===")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k} = (nested)")
        elif isinstance(v, list) and len(v) > 6:
            print(f"  {k} = [{len(v)} elements]")
        elif isinstance(v, float):
            print(f"  {k} = {v:.6g}")
        else:
            print(f"  {k} = {v}")
    print(f"\nout dir: {out_dir}")


if __name__ == "__main__":
    main()
