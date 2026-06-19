"""compute_radial_profile.py — pre-compute per-pattern radial profiles AND
auto-calibrate the SupCon gate thresholds for each sample.

Run once per sample. Outputs alongside the cube:
    <sample>.radial.npy            (N, n_bins) float32, log-normalized
    <sample>.gate_thresholds.json  {tau_pos, tau_neg, frac_pos, frac_neg, ...}

The radial profile is computed via the same polar pipeline used in training:
center-crop (140) -> resize (192) -> polar transform -> column-mask (45)
-> sum over theta. So the bins line up with the model's polar input.

Profile transform: I(r) -> log(I(r)/sum(I(r)) + eps), then per-profile
mean-centered (so cosine is sensitive to peak SHAPE, not amplitude).
"""
from __future__ import annotations
import os, sys, json, argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SAMPLES, LoadPRZ
from dino_sr_ablation import PolarTransform, PolarMaskLeft

CENTER_CROP = 140
POLAR_SIZE = 192
POLAR_MASK_COLS = 45
EPS = 1e-6
# SAXS-style treatment: restrict to mid-q (skip mask-edge artifact and
# high-q noise floor), subtract a polynomial decay in log space so the
# residuals are "peaks above smooth background" -- analogous to crystalline-
# vs-amorphous separation in WAXS literature.
Q_LOW = 70          # bins. Below this is mask-edge artifact (spike at
                     # polar_mask_cols boundary) + steepest beam-halo decay.
                     # Tightened from 60 -- diagnostics showed bins 60-69
                     # still carry mask-edge bleed-through.
Q_HIGH = 140        # bins. Above this is high-q noise floor.
                     # Tightened from 150 -- bins 140-149 are quiet/noise.
POLY_ORDER = 3      # polynomial order for log-space background fit
                     # (smooth decay; doesn't fit sharp peaks)


def _build_pre(polar_size, polar_mask_cols, center_crop):
    return T.Compose([
        T.CenterCrop(center_crop),
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        PolarTransform(output_size=polar_size),
        PolarMaskLeft(k_cols=polar_mask_cols),
    ])


def compute_radial(sample: str, batch: int = 256, device=None) -> np.ndarray:
    cfg = SAMPLES[sample]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{sample}] loading dataset...", flush=True)
    ds = LoadPRZ(cfg["path"], resize=POLAR_SIZE, vmax=cfg["vmax"])
    pre = _build_pre(POLAR_SIZE, POLAR_MASK_COLS, CENTER_CROP)
    N = len(ds)
    n_bins = POLAR_SIZE          # polar columns = radial bins
    radials = np.zeros((N, n_bins), dtype=np.float32)
    print(f"[{sample}] N={N}  n_bins={n_bins}  batch={batch}", flush=True)

    with torch.no_grad():
        for i in range(0, N, batch):
            j = min(i + batch, N)
            xs = torch.stack([ds[k] for k in range(i, j)]).to(device).float()
            xs = pre(xs)                                   # (b, 1, theta, r)
            # Sum over channel (it's 1) and theta (axis=-2) to get I(r)
            prof = xs.sum(dim=(1, 2))                      # (b, r)
            radials[i:j] = prof.cpu().numpy()
            if (i // batch) % 20 == 0:
                print(f"  {i}/{N}", flush=True)

    # SAXS-style treatment:
    #   1. Restrict to mid-q (Q_LOW..Q_HIGH) — skip mask-edge artifact + noise
    #   2. log-transform (Porod is power-law in I(q); linear in log I vs log q)
    #   3. Subtract a polynomial-fitted smooth decay per profile
    #   4. Mean-center for cosine shape comparison
    print(f"[{sample}] SAXS-style treatment: keep bins {Q_LOW}..{Q_HIGH}, "
          f"poly_order={POLY_ORDER} background subtraction...", flush=True)
    radials_keep = radials[:, Q_LOW:Q_HIGH]              # (N, n_keep)
    sums = radials_keep.sum(axis=1, keepdims=True)
    sums[sums < EPS] = EPS
    radials_norm = radials_keep / sums                   # per-pattern unit area
    log_I = np.log(radials_norm + EPS)                   # log intensity

    # Per-profile polynomial fit (degree POLY_ORDER) and subtract.
    # We use a robust two-pass fit: first fit, then re-fit excluding pixels
    # more than 2*MAD above the residuals (so peaks don't bias the baseline).
    n_keep = radials_keep.shape[1]
    q = np.arange(n_keep, dtype=np.float64)
    n_pat = log_I.shape[0]
    residuals = np.zeros_like(log_I, dtype=np.float32)
    print(f"  fitting {n_pat} polynomial baselines...", flush=True)
    for i in range(n_pat):
        y = log_I[i].astype(np.float64)
        # First pass: fit all
        coef = np.polyfit(q, y, POLY_ORDER)
        bg = np.polyval(coef, q)
        r = y - bg
        # Down-weight peaks (above-baseline residuals)
        med = np.median(r)
        mad = np.median(np.abs(r - med)) + 1e-12
        peak_mask = (r - med) > 2.0 * mad
        if peak_mask.sum() < n_keep - POLY_ORDER - 1:
            # Second pass: fit only non-peak points
            non_peak = ~peak_mask
            coef = np.polyfit(q[non_peak], y[non_peak], POLY_ORDER)
            bg = np.polyval(coef, q)
        r2 = y - bg
        residuals[i] = (r2 - r2.mean()).astype(np.float32)
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{n_pat}", flush=True)
    return residuals


def calibrate_thresholds(radials_log: np.ndarray, n_pairs: int = 50_000,
                          frac_pos: float = 0.15, frac_neg: float = 0.50,
                          seed: int = 42) -> dict:
    """Sample n_pairs random pairs, compute centered cosine, set thresholds
    such that ~frac_pos of pairs land above tau_pos (positives) and
    ~frac_neg of pairs land below tau_neg (negatives)."""
    N = radials_log.shape[0]
    rng = np.random.default_rng(seed)
    # uniform random pairs (no self-pairs)
    a = rng.integers(0, N, size=n_pairs)
    b = rng.integers(0, N, size=n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]

    # cosine of mean-centered log-radial profiles
    Va = radials_log[a]            # (P, n_bins)
    Vb = radials_log[b]
    na = np.linalg.norm(Va, axis=1) + EPS
    nb = np.linalg.norm(Vb, axis=1) + EPS
    cos = (Va * Vb).sum(axis=1) / (na * nb)            # (P,)

    tau_pos = float(np.quantile(cos, 1.0 - frac_pos))
    tau_neg = float(np.quantile(cos, frac_neg))
    actual_frac_pos = float((cos > tau_pos).mean())
    actual_frac_neg = float((cos < tau_neg).mean())
    actual_frac_abstain = float(((cos >= tau_neg) & (cos <= tau_pos)).mean())

    return {
        "tau_pos": tau_pos,
        "tau_neg": tau_neg,
        "target_frac_pos": frac_pos,
        "target_frac_neg": frac_neg,
        "actual_frac_pos": actual_frac_pos,
        "actual_frac_neg": actual_frac_neg,
        "actual_frac_abstain": actual_frac_abstain,
        "n_pairs_sampled": int(len(cos)),
        "cos_min": float(cos.min()),
        "cos_max": float(cos.max()),
        "cos_p10": float(np.quantile(cos, 0.10)),
        "cos_p50": float(np.quantile(cos, 0.50)),
        "cos_p90": float(np.quantile(cos, 0.90)),
        "n_bins": int(radials_log.shape[1]),
        "n_patterns": int(radials_log.shape[0]),
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True,
                    help="sample key in SAMPLES (e.g. Na007b, EuInAs_B100)")
    ap.add_argument("--n-pairs", type=int, default=50_000)
    ap.add_argument("--frac-pos", type=float, default=0.15)
    ap.add_argument("--frac-neg", type=float, default=0.50)
    args = ap.parse_args()

    cfg = SAMPLES[args.sample]
    cube_path = cfg["path"]
    base = cube_path[:-4] if cube_path.endswith(".prz") else cube_path
    rad_path = base + ".radial.npy"
    thresh_path = base + ".gate_thresholds.json"

    if os.path.exists(rad_path):
        print(f"[{args.sample}] radial already exists at {rad_path}, loading", flush=True)
        radials = np.load(rad_path)
    else:
        radials = compute_radial(args.sample)
        np.save(rad_path, radials)
        print(f"saved {rad_path}  shape={radials.shape}  dtype={radials.dtype}", flush=True)

    th = calibrate_thresholds(radials, n_pairs=args.n_pairs,
                               frac_pos=args.frac_pos, frac_neg=args.frac_neg)
    th["sample"] = args.sample
    with open(thresh_path, "w") as f:
        json.dump(th, f, indent=2)
    print(f"saved {thresh_path}", flush=True)
    print("calibration:")
    for k, v in th.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
