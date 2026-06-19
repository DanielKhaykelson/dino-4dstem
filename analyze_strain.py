"""
analyze_strain.py — crystallographic strain extraction between two DINOp classes.

Given two class-average diffraction patterns from a DINOp run, this module:

  1. Detects Bragg-spot centroids via multi-scale Laplacian-of-Gaussian (LoG).
  2. Matches centroids between the two patterns by mutual nearest neighbors
     with a conservative distance gate.
  3. Fits a 2-D affine transform class_B -> class_A via RANSAC, rejecting
     spurious correspondences.
  4. Decomposes the affine into
        rotation theta (rigid in-plane)
        anisotropic scaling lambda_x, lambda_y (reciprocal-space stretch)
        shear
        translation
     and converts the reciprocal-space scaling into real-space lattice
     strain eps_xx = 1/lambda_x - 1, eps_yy = 1/lambda_y - 1
     (opposite sign: reciprocal expansion = real-space compression).
  5. Warps class_B by the affine and computes the normalized difference
     (A - warp(B)) / (A + warp(B) + eps); if the lobes collapse, the two
     classes are crystallographically equivalent modulo the extracted
     deformation.

All outputs are saved to an eval/ subdirectory supplied by the caller.

Usage:
    from analyze_strain import strain_between_classes
    result = strain_between_classes(
        mean_A=classA_pattern, mean_B=classB_pattern,
        out_dir="runs/EuInAs_B100/winner_polar_centroid/eval/strain_C3_vs_C5",
        label_A="C3", label_B="C5",
    )
    # result: dict with rotation_deg, strain_xx, strain_yy, shear, n_inliers, ...
"""
from __future__ import annotations

import math
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import matplotlib.pyplot as plt


# =========================================================================
# 1. Multi-scale LoG blob detection
# =========================================================================

def detect_blobs(img: np.ndarray,
                  min_sigma: float = 2.0,
                  max_sigma: float = 10.0,
                  num_sigma: int = 10,
                  threshold: float = 0.05,
                  beam_mask_radius: int = 40,
                  percentile_norm: bool = True) -> np.ndarray:
    """Multi-scale Laplacian-of-Gaussian blob detector.

    Returns (N, 3) array with columns (y, x, sigma) for each detected blob.
    The direct-beam region (radius < beam_mask_radius) is zeroed before
    detection so the central peak does not flood the response.
    """
    from skimage.feature import blob_log

    arr = img.astype(np.float32).copy()
    H, W = arr.shape
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    bmask = ((yy - cy) ** 2 + (xx - cx) ** 2) > beam_mask_radius ** 2
    arr = arr * bmask

    if percentile_norm:
        lo, hi = np.percentile(arr[bmask], 2), np.percentile(arr[bmask], 99.5)
        arr = np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)

    blobs = blob_log(arr, min_sigma=min_sigma, max_sigma=max_sigma,
                      num_sigma=num_sigma, threshold=threshold)
    # blob_log returns (y, x, sigma); skimage sigma is already in pixel units.
    return blobs  # (N, 3)


# =========================================================================
# 2. Mutual nearest-neighbor matching with distance gate
# =========================================================================

def match_blobs(blobs_A: np.ndarray, blobs_B: np.ndarray,
                 distance_gate: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest neighbor matching between two blob sets.

    Returns (matches_A, matches_B): each (M, 2) of matched coordinates
    in (y, x) order. Only pairs where A's NN in B and B's NN in A are
    the same AND Euclidean distance <= distance_gate are kept.
    """
    pts_A = blobs_A[:, :2]                    # (N_A, 2)
    pts_B = blobs_B[:, :2]
    if len(pts_A) == 0 or len(pts_B) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    # Pairwise distances.
    d = np.linalg.norm(pts_A[:, None, :] - pts_B[None, :, :], axis=-1)  # (N_A, N_B)
    nn_AtoB = d.argmin(axis=1)
    nn_BtoA = d.argmin(axis=0)
    matches = []
    for i in range(len(pts_A)):
        j = nn_AtoB[i]
        if nn_BtoA[j] == i and d[i, j] <= distance_gate:
            matches.append((i, j))
    if not matches:
        return np.empty((0, 2)), np.empty((0, 2))
    mA = np.array([pts_A[i] for i, _ in matches])
    mB = np.array([pts_B[j] for _, j in matches])
    return mA, mB


# =========================================================================
# 3. RANSAC affine fit
# =========================================================================

def ransac_affine(mA: np.ndarray, mB: np.ndarray,
                   residual_threshold: float = 2.0,
                   max_trials: int = 2000) -> tuple:
    """Fit an affine transform that maps B coordinates onto A via RANSAC.

    skimage.measure.ransac works on (x, y) ordering; we passed (y, x).
    Internally, swap columns. The returned matrix operates on (x, y, 1)
    homogeneous vectors.

    Returns (affine_model, inlier_mask) or (None, None) if no fit.
    """
    from skimage.measure import ransac
    from skimage.transform import AffineTransform

    if len(mA) < 3:
        return None, None
    # Swap to (x, y) for skimage convention.
    src = mB[:, ::-1]   # B points we transform FROM
    dst = mA[:, ::-1]   # A points we transform TO
    try:
        # Seed numpy for RANSAC reproducibility. skimage.measure.ransac
        # uses np.random internally and doesn't expose a random_state kwarg
        # in all versions; seeding globally before the call is the portable
        # way to get determinism.
        np.random.seed(0)
        model, inliers = ransac(
            (src, dst), AffineTransform,
            min_samples=3,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )
    except Exception as exc:
        print(f"[ransac] exception: {exc!r}")
        return None, None
    return model, inliers


# =========================================================================
# 4. Decompose affine into rotation + anisotropic scale + shear
# =========================================================================

def decompose_affine(matrix_2x3: np.ndarray) -> dict:
    """Decompose a 2x3 affine matrix into rotation, scale, shear, translation.

    Matrix is [ [a, b, tx], [c, d, ty] ] operating on (x, y, 1).

    The 2x2 linear part A = [[a, b], [c, d]] is decomposed via SVD:
        A = U @ S @ Vt
    where U is an in-plane rotation (plus optional reflection handled
    below), S is a diagonal scaling (eigenvalues of A^T A's sqrt), and
    Vt is a second rotation that we absorb by redefining the axes. The
    net is: effective in-plane rotation + anisotropic scaling along two
    orthogonal axes + shear from the asymmetry of A.

    Returns a dict with rotation_deg, scale_major, scale_minor,
    shear_xy, translation_px, plus lattice_strain_xx, lattice_strain_yy
    (reciprocal-to-real inversion).
    """
    A = matrix_2x3[:2, :2]
    t = matrix_2x3[:2, 2]
    # Polar decomposition: A = R @ S where R is rotation (det=1), S is
    # symmetric positive semidefinite.
    U, Sigma, Vt = np.linalg.svd(A)
    # Handle reflection: if det(U @ Vt) < 0, flip the last singular value
    # sign and the last column of U to keep a proper rotation.
    rot = U @ Vt
    det = np.linalg.det(rot)
    if det < 0:
        U[:, -1] *= -1
        Sigma = Sigma.copy()
        Sigma[-1] *= -1
        rot = U @ Vt
    theta = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    # Anisotropic scales (principal axes in reciprocal space).
    scale_major = float(Sigma[0])
    scale_minor = float(Sigma[1])
    # Shear measured as off-diagonal of symmetric stretch tensor S = Vt.T @ diag(Sigma) @ Vt.
    S = Vt.T @ np.diag(Sigma) @ Vt
    shear_xy = float(S[0, 1])

    # Reciprocal-space expansion <=> real-space contraction.
    # lattice strain eps_ii = 1/scale - 1.
    # Here we label by the MAJOR / MINOR principal axes, NOT x / y cartesian,
    # because SVD diagonalizes anisotropy along its own principal frame.
    eps_major = 1.0 / scale_major - 1.0
    eps_minor = 1.0 / scale_minor - 1.0

    # Principal-axis orientation (in real-space, same angle as the rotation
    # of S's eigenvector frame).
    phi = math.degrees(math.atan2(Vt[0, 1], Vt[0, 0]))

    return dict(
        rotation_deg=theta,
        scale_major=scale_major,
        scale_minor=scale_minor,
        shear_xy=shear_xy,
        translation_px=t.tolist(),
        eps_major=eps_major,
        eps_minor=eps_minor,
        principal_axis_deg=phi,
        affine_2x3=matrix_2x3.tolist(),
    )


# =========================================================================
# 5. Warp-back validation
# =========================================================================

def warp_and_diff(mean_A: np.ndarray, mean_B: np.ndarray,
                   model) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fitted affine to B, returning (warped_B, normalized_diff).

    normalized_diff = (A - warp(B)) / (A + warp(B) + eps)
    Bounded in [-1, +1]; zero means perfect match.
    """
    from skimage.transform import warp
    # skimage.warp uses the INVERSE mapping: for each pixel in output,
    # compute where to sample from. Our `model` maps B -> A, so to warp B
    # into A's frame we need model's inverse.
    inv = model.inverse
    warped_B = warp(mean_B, inv, output_shape=mean_A.shape,
                     preserve_range=True, order=1, mode="constant",
                     cval=0.0)
    eps = 1e-6 * max(mean_A.max(), warped_B.max(), 1.0)
    diff = (mean_A - warped_B) / (mean_A + warped_B + eps)
    return warped_B.astype(mean_A.dtype), diff


# =========================================================================
# 6. Full pipeline + figure
# =========================================================================

def _percentile_display(arr: np.ndarray, pct_lo=2.0, pct_hi=99.5,
                          beam_mask_radius: int = 40) -> np.ndarray:
    """Percentile-clipped, log1p'd display.
    If beam_mask_radius <= 0 the central beam is left intact (clip/log are
    computed on the WHOLE frame and no zeroing is applied)."""
    H, W = arr.shape
    if beam_mask_radius <= 0:
        vals = arr
        lo = np.percentile(vals, pct_lo)
        hi = np.percentile(vals, pct_hi)
        clipped = np.clip(arr, lo, hi)
        return np.log1p(clipped - lo)
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    bm = ((yy - cy) ** 2 + (xx - cx) ** 2) > beam_mask_radius ** 2
    vals = arr[bm]
    if vals.size == 0:
        return arr
    lo = np.percentile(vals, pct_lo)
    hi = np.percentile(vals, pct_hi)
    clipped = np.clip(arr, lo, hi) * bm
    return np.log1p(clipped - lo)


def strain_between_classes(mean_A: np.ndarray, mean_B: np.ndarray,
                              out_dir: str,
                              label_A: str = "A", label_B: str = "B",
                              blob_min_sigma: float = 2.0,
                              blob_max_sigma: float = 10.0,
                              blob_num_sigma: int = 10,
                              blob_threshold: float = 0.05,
                              beam_mask_radius: int = 40,
                              display_beam_mask_radius: "int | None" = None,
                              match_gate_px: float = 10.0,
                              ransac_residual_px: float = 2.0,
                              ransac_max_trials: int = 2000) -> dict:
    """End-to-end strain analysis. Writes figures under out_dir, returns a
    dict with the quantitative results.
    `beam_mask_radius` controls the central-beam mask used for BLOB
    DETECTION ONLY (without it `blob_log` floods on the direct beam).
    `display_beam_mask_radius` controls the mask used in the figure
    visualisation; pass 0 to keep the central beam visible. Default is
    to use beam_mask_radius (backward compatible)."""
    if display_beam_mask_radius is None:
        display_beam_mask_radius = beam_mask_radius
    os.makedirs(out_dir, exist_ok=True)

    # Detect blobs.
    blobs_A = detect_blobs(mean_A, min_sigma=blob_min_sigma,
                             max_sigma=blob_max_sigma,
                             num_sigma=blob_num_sigma,
                             threshold=blob_threshold,
                             beam_mask_radius=beam_mask_radius)
    blobs_B = detect_blobs(mean_B, min_sigma=blob_min_sigma,
                             max_sigma=blob_max_sigma,
                             num_sigma=blob_num_sigma,
                             threshold=blob_threshold,
                             beam_mask_radius=beam_mask_radius)
    result = dict(
        label_A=label_A, label_B=label_B,
        n_blobs_A=int(len(blobs_A)),
        n_blobs_B=int(len(blobs_B)),
    )

    # Match.
    mA, mB = match_blobs(blobs_A, blobs_B, distance_gate=match_gate_px)
    result["n_initial_matches"] = int(len(mA))

    if len(mA) < 3:
        print(f"[strain] too few matches ({len(mA)}) for affine; skipping RANSAC")
        result["status"] = "too_few_matches"
        _save_diagnostic_figure(mean_A, mean_B, blobs_A, blobs_B, mA, mB,
                                  None, None, None, result, out_dir,
                                  beam_mask_radius=display_beam_mask_radius,
                                  label_A=label_A, label_B=label_B)
        return result

    # RANSAC.
    model, inliers = ransac_affine(mA, mB,
                                     residual_threshold=ransac_residual_px,
                                     max_trials=ransac_max_trials)
    if model is None:
        result["status"] = "ransac_failed"
        _save_diagnostic_figure(mean_A, mean_B, blobs_A, blobs_B, mA, mB,
                                  None, None, None, result, out_dir,
                                  beam_mask_radius=display_beam_mask_radius,
                                  label_A=label_A, label_B=label_B)
        return result
    matrix_2x3 = np.hstack([model.params[:2, :2], model.params[:2, 2:3]])
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    result["n_inliers"] = n_inliers
    result["inlier_fraction"] = float(n_inliers / max(len(mA), 1))
    decomp = decompose_affine(matrix_2x3)
    result.update(decomp)
    result["status"] = "ok"

    # Warp-back validation.
    warped_B, diff = warp_and_diff(mean_A, mean_B, model)
    result["diff_mean_abs"] = float(np.abs(diff).mean())
    # Naive difference without affine, for comparison.
    eps = 1e-6 * max(mean_A.max(), mean_B.max(), 1.0)
    diff_raw = (mean_A - mean_B) / (mean_A + mean_B + eps)
    result["diff_raw_mean_abs"] = float(np.abs(diff_raw).mean())

    _save_diagnostic_figure(mean_A, mean_B, blobs_A, blobs_B, mA, mB,
                              model, warped_B, diff, result, out_dir,
                              beam_mask_radius=beam_mask_radius,
                              label_A=label_A, label_B=label_B)

    import json
    with open(os.path.join(out_dir, "strain_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    return result


# =========================================================================
# 7. Figure
# =========================================================================

def _save_diagnostic_figure(mean_A, mean_B, blobs_A, blobs_B, mA, mB,
                              model, warped_B, diff, result, out_dir,
                              beam_mask_radius: int = 40,
                              label_A: str = "A", label_B: str = "B"):
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.20,
                            height_ratios=[1.0, 1.0, 0.15])

    # Row 1: class A with blobs, class B with blobs, raw difference.
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(_percentile_display(mean_A, beam_mask_radius=beam_mask_radius),
               cmap="inferno")
    for y, x, s in blobs_A:
        ax.add_patch(plt.Circle((x, y), s * np.sqrt(2), color="lime",
                                  fill=False, lw=0.8))
    ax.set_title(f"{label_A} mean  ({len(blobs_A)} blobs)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(_percentile_display(mean_B, beam_mask_radius=beam_mask_radius),
               cmap="inferno")
    for y, x, s in blobs_B:
        ax.add_patch(plt.Circle((x, y), s * np.sqrt(2), color="lime",
                                  fill=False, lw=0.8))
    ax.set_title(f"{label_B} mean  ({len(blobs_B)} blobs)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    eps = 1e-6 * max(mean_A.max(), mean_B.max(), 1.0)
    diff_raw = (mean_A - mean_B) / (mean_A + mean_B + eps)
    ax.imshow(diff_raw, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"({label_A} - {label_B}) / ({label_A} + {label_B})\n"
                   "lobes = rotation + strain signature", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Row 2: matches overlay, warped B, warp-back difference.
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(_percentile_display(mean_A, beam_mask_radius=beam_mask_radius),
               cmap="inferno")
    if len(mA) > 0:
        for (ya, xa), (yb, xb) in zip(mA, mB):
            ax.plot([xa, xb], [ya, yb], color="cyan", lw=0.6, alpha=0.8)
        ax.scatter(mA[:, 1], mA[:, 0], color="cyan", s=15, label=f"{label_A} matched")
        ax.scatter(mB[:, 1], mB[:, 0], color="yellow", s=15, label=f"{label_B} matched", marker="x")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Mutual NN matches (gate {result.get('n_initial_matches', 0)})",
                   fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    if warped_B is not None:
        ax.imshow(_percentile_display(warped_B, beam_mask_radius=beam_mask_radius),
                   cmap="inferno")
        ax.set_title(f"warp({label_B}) in {label_A}'s frame", fontsize=9)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no affine fit", ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    if diff is not None:
        ax.imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"({label_A} - warp({label_B})) / ({label_A} + warp({label_B}))\n"
                       "lobes collapsed  =>  same lattice + deformation",
                       fontsize=9)
    else:
        ax.set_axis_off()
    ax.set_xticks([]); ax.set_yticks([])

    # Row 3: text summary.
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.set_axis_off()
    if result.get("status") == "ok":
        txt = (
            f"rotation theta = {result['rotation_deg']:+.3f} deg     "
            f"eps_major = {result['eps_major']:+.4f}     "
            f"eps_minor = {result['eps_minor']:+.4f}     "
            f"shear_xy = {result['shear_xy']:+.4f}     "
            f"principal axis phi = {result['principal_axis_deg']:+.1f} deg\n"
            f"|diff|_raw = {result['diff_raw_mean_abs']:.3f}   ->   "
            f"|diff|_warped = {result['diff_mean_abs']:.3f}   "
            f"(collapse ratio {result['diff_raw_mean_abs']/max(result['diff_mean_abs'],1e-9):.2f}x)     "
            f"inliers = {result['n_inliers']}/{result['n_initial_matches']} "
            f"({100*result['inlier_fraction']:.0f}%)"
        )
    else:
        txt = f"status: {result.get('status', 'unknown')}"
    ax_txt.text(0.01, 0.5, txt, fontsize=9, family="monospace", va="center")

    fig.suptitle(f"Crystallographic strain between {label_A} and {label_B}  "
                   f"(DINOp prototype means)", fontsize=11, y=0.995)
    out_path = os.path.join(out_dir, f"fig_strain_{label_A}_vs_{label_B}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# 8. CLI convenience
# =========================================================================

if __name__ == "__main__":
    import argparse
    import json

    from data import SAMPLES, LoadPRZ
    from contrastive_eval import infer_scan

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--class-A", type=int, required=True,
                    help="Dense class id for prototype A")
    ap.add_argument("--class-B", type=int, required=True,
                    help="Dense class id for prototype B")
    ap.add_argument("--n-top", type=int, default=300,
                    help="Use the top-N highest-confidence samples per "
                         "class for the class mean.")
    ap.add_argument("--blob-min-sigma", type=float, default=2.0)
    ap.add_argument("--blob-max-sigma", type=float, default=10.0)
    ap.add_argument("--blob-threshold", type=float, default=0.05)
    ap.add_argument("--match-gate", type=float, default=10.0)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    cfg = SAMPLES[args.sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    run_dir = os.path.join(base, "runs", args.sample, args.config)
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]

    def class_mean(c: int) -> np.ndarray:
        idx = np.where(assigns == c)[0]
        if len(idx) == 0:
            raise ValueError(f"class {c} is empty")
        scores = soft_probs[idx, c]
        top = idx[np.argsort(-scores)[:min(args.n_top, len(idx))]]
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0).astype(np.float32)
        w = soft_probs[top, c].astype(np.float32)
        return (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)

    mean_A = class_mean(args.class_A)
    mean_B = class_mean(args.class_B)
    out_dir = os.path.join(run_dir, "eval",
                            f"strain_p{args.class_A}_vs_p{args.class_B}")
    result = strain_between_classes(
        mean_A, mean_B, out_dir,
        label_A=f"p{args.class_A}", label_B=f"p{args.class_B}",
        blob_min_sigma=args.blob_min_sigma,
        blob_max_sigma=args.blob_max_sigma,
        blob_threshold=args.blob_threshold,
        match_gate_px=args.match_gate,
    )
    print(json.dumps(result, indent=2, default=float))
