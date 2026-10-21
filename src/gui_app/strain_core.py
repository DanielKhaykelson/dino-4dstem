"""strain_core.py -- lattice strain from Bragg-spot positions.

The reference is a 2-D reciprocal lattice, B_ref = [g1 g2] (two basis
vectors, detector px): from a MEASURED reference (data ROI, grain, class
average -> relative strain, that region = 0) or from a CIF simulated at a
zone axis (absolute strain - only as good as the 1/Å-per-px calibration).

For every pattern:
  1. spot positions are refined to sub-pixel (intensity centroid around
     each detected peak) and taken from the pattern's own beam centre;
  2. the reference lattice, rotated by the in-plane angle found by the
     data-reference match, indexes the spots (h, k = nearest integers);
  3. B is fitted by least squares to every indexed spot: v ≈ h g1 + k g2
     + t (the offset t soaks up a slightly wrong beam centre);
  4. the reciprocal deformation F* = B B_ref^-1 gives the real-space one,
     F = (F*)^-T, whose polar decomposition F = R U gives the strain
     ε = U - I (εxx, εyy, εxy) and the lattice rotation ω (from R).

Axes: x = detector columns (right), y = detector rows UP (so ω > 0 is
counter-clockwise on screen).  Real-space directions equal detector
directions only if the scan and diffraction are not rotated relative to
each other (check the scan rotation of your microscope).
"""
from __future__ import annotations

import numpy as np


def refine_subpixel(pattern, peaks, half: int = 2):
    """Sub-pixel spot centres: intensity centroid (local minimum
    subtracted) in a (2*half+1)^2 window around each detected peak.
    peaks (N, 3) (row, col, I) -> same, refined."""
    p = np.asarray(peaks, float).reshape(-1, 3).copy()
    if not len(p):
        return p
    f = np.asarray(pattern, np.float64)
    H, W = f.shape
    for n in range(len(p)):
        r0, c0 = int(round(p[n, 0])), int(round(p[n, 1]))
        ya, yb = max(r0 - half, 0), min(r0 + half + 1, H)
        xa, xb = max(c0 - half, 0), min(c0 + half + 1, W)
        win = f[ya:yb, xa:xb]
        w = win - win.min()
        s = w.sum()
        if s <= 0:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        p[n, 0] = (w * yy).sum() / s
        p[n, 1] = (w * xx).sum() / s
    return p


def to_xy(vecs):
    """(dy, dx) detector vectors -> (x, y) with y pointing UP."""
    v = np.asarray(vecs, float).reshape(-1, 2)
    return np.stack([v[:, 1], -v[:, 0]], axis=1)


def lattice_basis(xy, tol: float = 3.0, min_angle_deg: float = 20.0):
    """Two shortest non-collinear vectors of a spot set, then refined by a
    least-squares fit to every spot they index.  Returns (B (2x2, columns
    g1 g2), n_indexed)."""
    xy = np.asarray(xy, float)
    r = np.hypot(xy[:, 0], xy[:, 1])
    order = np.argsort(r)
    g1 = xy[order[0]]
    s_min = np.sin(np.radians(min_angle_deg))
    g2 = None
    for i in order[1:]:
        v = xy[i]
        if abs(g1[0] * v[1] - g1[1] * v[0]) > s_min * r[order[0]] * r[i]:
            g2 = v
            break
    if g2 is None:
        raise ValueError("all spots are collinear - need two independent "
                         "lattice directions")
    B = np.stack([g1, g2], axis=1)
    B, n, _, _ = fit_lattice(B, xy, tol)
    return B, n


def fit_lattice(B0, xy, tol: float = 3.0, n_iter: int = 3,
                min_spots: int = 4):
    """Index spots xy with lattice B0 (nearest integer h, k), keep those
    within tol px and != origin, then least-squares fit v = B·hk + t.
    The offset t absorbs a small error of the beam centre (it would
    otherwise look like strain).  Returns (B, n_indexed, rms px, t);
    B is NaN if fewer than min_spots spots in >= 2 directions."""
    xy = np.asarray(xy, float)
    B = np.asarray(B0, float)
    t = np.zeros(2)
    nan = (np.full((2, 2), np.nan), 0, np.nan, np.full(2, np.nan))
    if len(xy) < min_spots:
        return nan
    for _ in range(n_iter):
        try:
            hk = np.rint(np.linalg.solve(B, (xy - t).T).T)
        except np.linalg.LinAlgError:
            return nan
        res = np.hypot(*(hk @ B.T + t - xy).T)
        ok = (res < tol) & (np.abs(hk).sum(1) > 0)
        if ok.sum() < max(min_spots, 3) or np.linalg.matrix_rank(
                hk[ok]) < 2:
            return nan
        A = np.hstack([hk[ok], np.ones((int(ok.sum()), 1))])
        sol, *_ = np.linalg.lstsq(A, xy[ok], rcond=None)
        B, t = sol[:2].T, sol[2]
    pred = hk[ok] @ B.T + t
    rms = float(np.sqrt(np.mean(np.sum((pred - xy[ok]) ** 2, axis=1))))
    return B, int(ok.sum()), rms, t


def rot2(theta_deg):
    t = np.radians(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def strain_from_lattices(B_ref, B):
    """Real-space strain of lattice B relative to B_ref (reciprocal
    bases, columns).  Returns (exx, eyy, exy, rotation_deg)."""
    if not np.all(np.isfinite(B)):
        return (np.nan,) * 4
    Fs = B @ np.linalg.inv(B_ref)                  # reciprocal deformation
    F = np.linalg.inv(Fs).T                        # real-space deformation
    U_, S, Vt = np.linalg.svd(F)
    R = U_ @ Vt
    if np.linalg.det(R) < 0:
        Vt = Vt.copy(); Vt[-1] *= -1; S = S.copy(); S[-1] *= -1
        R = U_ @ Vt
    U = Vt.T @ np.diag(S) @ Vt
    e = U - np.eye(2)
    # a lattice and its inverse (-B) are the same spot set: rotation is
    # defined modulo 180 degrees -> report it in (-90, 90]
    w = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    w = -((-(w + 90.0)) % 180.0) + 90.0
    return (float(e[0, 0]), float(e[1, 1]), float(0.5 * (e[0, 1] + e[1, 0])),
            w)


def strain_one(B_ref, xy, theta_deg=0.0, tol=3.0, min_spots=4):
    """Fit one pattern: B_ref rotated by the matched in-plane angle
    indexes the spots.  Returns dict(exx, eyy, exy, rot, n, rms, B, t)."""
    # theta (data-reference match) is counter-clockwise on screen, the
    # same sense as rot2 in this y-up frame
    B0 = rot2(float(theta_deg)) @ B_ref
    B, n, rms, t = fit_lattice(B0, xy, tol, min_spots=min_spots)
    exx, eyy, exy, rot = strain_from_lattices(B_ref, B)
    return dict(exx=exx, eyy=eyy, exy=exy, rot=rot, n=n, rms=rms, B=B, t=t)
