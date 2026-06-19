"""
elliptical_correction.py — per-sample elliptical-distortion correction.

Detector geometry / sample tilt can make diffraction rings elliptical rather
than circular. That breaks rotation symmetry: the same physical phase at two
orientations produces two different-looking patterns. We fit a single ellipse
on the sample-mean pattern (PACBED), then affine-warp every pattern to
circularize the rings.

Pipeline order (after this module is wired in):
    raw pattern → EllipticalCorrection → [aug] → PolarTransform → model

One ellipse per sample (not per-pattern). Fit-once, apply-forever.

Public API:
    compute_pacbed(dataset, n_samples=1000, seed=0)  → (H, W) ndarray
    fit_ellipse_affine(pacbed, mask_radius=20)       → (A_2x2 ndarray, meta dict)
    class EllipticalCorrection(nn.Module)            → torchvision-style transform

Fit is moment-based (2D covariance of intensity with the central beam masked).
This captures the dominant anisotropy of the ring structure. No peak finding,
no per-ring fitting — robust to patterns with sparse or noisy rings.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pacbed(dataset, n_samples: int = 1000, seed: int = 0) -> np.ndarray:
    """Compute the position-averaged CBED by averaging n_samples random
    patterns from the dataset. On the `LoadPRZ` interface this returns the
    resized/rescaled intensity (whatever the dataset's __getitem__ gives).
    """
    N = len(dataset)
    rng = np.random.default_rng(seed)
    k = min(n_samples, N)
    idxs = rng.choice(N, size=k, replace=False)
    acc = None
    for i in idxs:
        x = dataset[int(i)]              # (1, H, W) torch tensor
        a = x.squeeze(0).numpy().astype(np.float64)
        if acc is None:
            acc = np.zeros_like(a)
        acc += a
    return acc / max(k, 1)


def fit_ellipse_affine(pacbed: np.ndarray,
                        mask_radius: int = 20) -> tuple[np.ndarray, dict]:
    """Fit a moment-based ellipse to `pacbed` after masking out the central
    direct-beam disc of radius `mask_radius`, and return the 2x2 affine
    matrix A that maps a unit circle to the fitted ellipse (normalized so
    the major semi-axis is 1).

    Usage: pass A as the 2x2 part of `theta` in `F.affine_grid` (no
    translation, center of rotation = image center). `grid_sample` then
    produces a circularized output from the elliptical input.

    Returns
    -------
    A : ndarray, shape (2, 2), dtype float32
        The forward ellipse-inducing matrix. Columns and rows are in (x, y)
        order matching `affine_grid`'s convention.
    meta : dict
        Diagnostics: axis_ratio (b/a ≤ 1), phi_deg (ellipse major axis angle
        from +x in degrees), eigvals (lambda_minor, lambda_major).
    """
    H, W = pacbed.shape
    yy, xx = np.indices((H, W)).astype(np.float64)
    dy = yy - H / 2.0
    dx = xx - W / 2.0
    r2 = dy * dy + dx * dx
    outside_beam = r2 > mask_radius ** 2

    I = np.clip(pacbed.astype(np.float64), 0.0, None) * outside_beam
    S = I.sum()
    if S < 1e-9:
        return np.eye(2, dtype=np.float32), {
            "axis_ratio": 1.0, "phi_deg": 0.0, "eigvals": (1.0, 1.0),
            "note": "PACBED empty after masking; identity used."}

    # Moment-based covariance (x, y) order
    Cxx = float((dx * dx * I).sum() / S)
    Cyy = float((dy * dy * I).sum() / S)
    Cxy = float((dx * dy * I).sum() / S)
    cov = np.array([[Cxx, Cxy], [Cxy, Cyy]], dtype=np.float64)

    vals, vecs = np.linalg.eigh(cov)  # ascending (lambda_minor, lambda_major)
    lam_min, lam_max = float(vals[0]), float(vals[1])
    if lam_max <= 0:
        return np.eye(2, dtype=np.float32), {
            "axis_ratio": 1.0, "phi_deg": 0.0, "eigvals": (lam_min, lam_max),
            "note": "zero variance; identity used."}

    axis_ratio = math.sqrt(lam_min / lam_max)   # b/a, in (0, 1]
    v_major = vecs[:, 1]                         # eigenvector of lam_max, (vx, vy)
    phi_rad = math.atan2(float(v_major[1]), float(v_major[0]))
    phi_deg = math.degrees(phi_rad)

    # Build A = R(phi) · diag(1, axis_ratio) · R(phi)^T
    # where R(phi) = [[cos, -sin], [sin, cos]] acts on (x, y) columns.
    c, s = math.cos(phi_rad), math.sin(phi_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    D = np.diag([1.0, axis_ratio])
    A = (R @ D @ R.T).astype(np.float32)

    return A, {
        "axis_ratio": axis_ratio,
        "phi_deg": phi_deg,
        "eigvals": (lam_min, lam_max),
    }


class EllipticalCorrection(nn.Module):
    """Apply a fixed 2x2 affine warp to the input tensor via grid_sample.

    Input  : (C, H, W) or (B, C, H, W), any device, any float dtype.
    Output : same shape; same device/dtype.

    The warp is pre-computed (one per sample) and held as a module buffer.
    No gradients flow through the warp parameters — they're fixed.

    The affine matrix A is "circle → ellipse" (what affine_grid expects as
    theta, mapping normalized output → normalized input). For each output
    pixel on a unit circle, `grid_sample` looks up the corresponding point
    on the fitted ellipse in the input.

    For identity (no correction) pass A = identity or use `None` at
    construction.
    """
    def __init__(self, A: np.ndarray | None = None):
        super().__init__()
        if A is None:
            A = np.eye(2, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32).reshape(2, 2)
        # theta = [A | 0] shape (2, 3)
        theta = np.concatenate([A, np.zeros((2, 1), dtype=np.float32)], axis=1)
        self.register_buffer("theta", torch.from_numpy(theta))    # (2, 3)
        self.register_buffer("A", torch.from_numpy(A))             # (2, 2)
        self.is_identity = bool(np.allclose(A, np.eye(2), atol=1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_identity:
            return x
        single = (x.dim() == 3)
        if single:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape
        # Move theta to input's device/dtype — buffers don't travel with the
        # transform when used inside a torchvision T.Compose (not an nn.Module
        # wrapper), so input-device adaptation must be explicit.
        theta_b = self.theta.to(device=x.device, dtype=x.dtype) \
                             .unsqueeze(0).expand(B, -1, -1)
        grid = F.affine_grid(theta_b, size=[B, C, H, W], align_corners=True)
        out = F.grid_sample(x, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)
        return out.squeeze(0) if single else out
