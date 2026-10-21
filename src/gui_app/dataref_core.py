"""dataref_core.py -- ACOM against MEASURED references instead of a CIF.

A reference is a measured pattern (one frame, a grain, an ROI or a class
average): its detected Bragg spots, as vectors from its own direct beam,
ARE the template.  Every pattern of the scan is compared with it:

  1. spots are taken relative to the pattern's OWN beam centre, so beam
     drift across the scan does not matter;
  2. the in-plane rotation that best lays the reference spots on the
     pattern's spots is found by voting: a reference spot and a pattern
     spot at the same radius (within ``tol`` px) vote for the angle
     between them; the most-voted angle wins and is then refined by a
     least-squares rotation of the matched pairs;
  3. score = F1 of the one-to-one matching (tolerance ``tol`` px):
     2·m / (n_reference + n_pattern), counting only spots inside the
     analysis ring r_min..r_max.  1 = the same spot set, 0 = nothing in
     common.  Positions only -- intensities are not compared.

The rotation is reported in (-P/2, P/2], P = the reference's own rotational symmetry
(detected from the reference itself: 2-fold for a centrosymmetric
pattern, 4- / 6-fold for square / hexagonal zone axes), so equivalent
orientations get the same colour.

A single 2-D reference cannot see out-of-plane tilt: a tilted grain just
scores low.  Add a second reference for it (several references -> a phase
/ zone-axis map: each position takes the best-scoring one).

Pure numpy / scipy; no GUI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
@dataclass
class DataReference:
    name: str
    vecs: np.ndarray                 # (M, 2) spot (dy, dx) from the beam, px
    period_deg: float = 360.0        # rotational self-symmetry
    origin: str = ""                 # where it came from (for the legend)
    pattern: Optional[np.ndarray] = None   # the averaged pattern (display)
    centre: tuple = (0.0, 0.0)       # beam centre in ``pattern``
    colour: tuple = (0.5, 0.5, 0.5)
    extra: dict = field(default_factory=dict)

    @property
    def r(self):
        return np.hypot(self.vecs[:, 0], self.vecs[:, 1])

    @property
    def phi(self):
        return np.arctan2(self.vecs[:, 0], self.vecs[:, 1])


def beam_centre(pattern, guess=None, search=None, win: int = 12):
    """Direct-beam centre (row, col): thresholded centre of mass around
    the brightest pixel (same estimator as the denoiser / flatten).

    With ``guess`` (e.g. the scan-mean beam position) the brightest
    pixel is looked for only within ``search`` px of it (default 8% of
    the detector) -- on a strongly diffracting zone axis a Bragg spot can
    be brighter than the direct beam."""
    from denoise_n2n import beam_centre as _bc
    f = np.asarray(pattern, dtype=np.float32)
    if guess is None:
        return _bc(f, win=win)
    H, W = f.shape
    rad = float(search) if search else 0.08 * min(H, W)
    gy, gx = float(guess[0]), float(guess[1])
    y0, y1 = int(max(0, gy - rad)), int(min(H, gy + rad + 1))
    x0, x1 = int(max(0, gx - rad)), int(min(W, gx + rad + 1))
    sub = f[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    sub = np.where(np.hypot(yy - gy, xx - gx) <= rad, sub, -np.inf)
    by, bx = np.unravel_index(int(np.argmax(sub)), sub.shape)
    by, bx = by + y0, bx + x0
    ya, yb = max(0, by - win), min(H, by + win + 1)
    xa, xb = max(0, bx - win), min(W, bx + win + 1)
    w = np.clip(f[ya:yb, xa:xb] - 0.5 * f[by, bx], 0, None)
    if w.sum() <= 0:
        return float(by), float(bx)
    yy, xx = np.mgrid[ya:yb, xa:xb]
    return float((w * yy).sum() / w.sum()), float((w * xx).sum() / w.sum())


def spot_vectors(peaks, centre, r_min: float = 0.0,
                 r_max: float = np.inf) -> np.ndarray:
    """(N, 3) detected peaks (row, col, I) -> (n, 2) (dy, dx) vectors
    from ``centre`` with r_min <= |v| <= r_max (the direct beam and
    anything outside the analysis ring dropped)."""
    p = np.asarray(peaks, dtype=float).reshape(-1, 3) if len(peaks) else \
        np.zeros((0, 3))
    v = p[:, :2] - np.asarray(centre, dtype=float)[None, :]
    r = np.hypot(v[:, 0], v[:, 1])
    return v[(r >= r_min) & (r <= r_max)]


def _rot(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    # angle phi = atan2(dy, dx): rotate (dx, dy) by theta
    dx = c * v[:, 1] - s * v[:, 0]
    dy = s * v[:, 1] + c * v[:, 0]
    return np.stack([dy, dx], axis=1)


def _pair(a, b, tol):
    """One-to-one matching of point sets a (n,2) and b (m,2) with
    distance < tol (Hungarian on the distance matrix).  Returns index
    arrays (ia, ib)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    close = d < tol
    if not close.any():
        return np.zeros(0, int), np.zeros(0, int)
    from scipy.optimize import linear_sum_assignment
    cost = np.where(close, d, 1e6)
    ia, ib = linear_sum_assignment(cost)
    ok = close[ia, ib]
    return ia[ok], ib[ok]


def match_vectors(ref, pat, tol: float = 3.0, period_deg: float = 360.0,
                  fixed_theta: Optional[float] = None):
    """Best in-plane rotation of reference vectors ``ref`` onto pattern
    vectors ``pat`` (both (n, 2) dy, dx).

    Returns dict(score, theta_deg, n_match, n_ref, n_pat, ia, ib) where
    theta (counter-clockwise on screen = positive) is in (-period/2,
    period/2] and ia / ib index the matched pairs.  Lay the reference on
    the pattern with ``rotate_ccw(ref, theta_deg)``."""
    out = dict(score=0.0, theta_deg=np.nan, n_match=0, n_ref=len(ref),
               n_pat=len(pat), ia=np.zeros(0, int), ib=np.zeros(0, int),
               margin=np.nan)
    if len(ref) == 0 or len(pat) == 0:
        return out
    rr = np.hypot(ref[:, 0], ref[:, 1]); pr = np.hypot(pat[:, 0], pat[:, 1])
    if fixed_theta is None:
        # vote: every same-radius (ref, pattern) pair proposes an angle
        i, j = np.nonzero(np.abs(rr[:, None] - pr[None, :]) < tol)
        if len(i) == 0:
            return out
        pa = np.arctan2(ref[:, 0], ref[:, 1]); pb = np.arctan2(pat[:, 0], pat[:, 1])
        ang = np.degrees(pb[j] - pa[i]) % 360.0
        # each vote is spread over the angle the tolerance allows at
        # that radius (tol / r), on a 1-degree circular histogram
        w = np.clip(np.degrees(tol / np.maximum(rr[i], 1.0)), 1.0, 30.0)
        lo = np.floor(ang - w).astype(int); n_b = np.ceil(ang + w).astype(
            int) - lo + 1
        span = np.arange(int(n_b.max()))
        ok = span[None, :] < n_b[:, None]
        bins = (lo[:, None] + span[None, :]) % 360
        hist = np.bincount(bins[ok], weights=np.broadcast_to(
            (1.0 / n_b)[:, None], bins.shape)[ok], minlength=360)
        # try the best few distinct candidate angles, keep the best match
        cands = []
        for c in np.argsort(-hist):
            if hist[c] <= 0 or len(cands) == 4:
                break
            if all(min(abs(c - d), 360 - abs(c - d)) > 3 for d in cands):
                cands.append(int(c))
        tried = [_refine(ref, pat, np.radians(float(c)), tol)
                 for c in cands]
        tried.sort(key=lambda t: (-t[2], t[3]))
        best = tried[0]
        th, ia, ib = best[0], best[1][0], best[1][1]
        # runner-up at a genuinely different orientation (not the same
        # one plus a symmetry operation): how unambiguous is the angle?
        second = 0
        for t in tried[1:]:
            d = _wrap(np.degrees(t[0] - th), period_deg)
            if abs(d) > 3.0:
                second = t[2]
                break
        out["margin"] = ((best[2] - second) / best[2]) if best[2] else 0.0
    else:
        th = -np.radians(float(fixed_theta))
        ia, ib = _pair(_rot(ref, th), pat, tol)
    m = len(ia)
    out.update(score=2.0 * m / (len(ref) + len(pat)),
               # internal angles are in the y-DOWN detector frame;
               # report counter-clockwise ON SCREEN as positive
               theta_deg=_wrap(-np.degrees(th), period_deg),
               n_match=m, ia=ia, ib=ib)
    return out


def rotate_ccw(vecs, theta_deg):
    """(dy, dx) vectors rotated counter-clockwise ON SCREEN by theta."""
    return _rot(np.asarray(vecs, float), -np.radians(float(theta_deg)))


def _wrap(deg, period):
    """Angle in (-period/2, period/2]: small rotations either way stay
    small numbers instead of wrapping to ~period."""
    p = float(period)
    return float(-((-(float(deg) + p / 2)) % p) + p / 2)


def _refine(ref, pat, th, tol, n_iter: int = 2):
    """Match at angle th, then least-squares rotation of the pairs;
    returns (theta, (ia, ib), n_match, rms)."""
    ia, ib = _pair(_rot(ref, th), pat, tol)
    for _ in range(n_iter):
        if len(ia) < 1:
            break
        a, b = ref[ia], pat[ib]
        # 2-D Procrustes about the origin (rotation only)
        cross = (a[:, 1] * b[:, 0] - a[:, 0] * b[:, 1]).sum()
        dot = (a[:, 1] * b[:, 1] + a[:, 0] * b[:, 0]).sum()
        th = float(np.arctan2(cross, dot))
        ia2, ib2 = _pair(_rot(ref, th), pat, tol)
        if len(ia2) < len(ia):
            break
        ia, ib = ia2, ib2
    rms = (float(np.sqrt(np.mean(np.sum((_rot(ref[ia], th) - pat[ib]) ** 2,
                                        axis=1)))) if len(ia) else np.inf)
    return th, (ia, ib), len(ia), rms


def self_symmetry(vecs, tol: float = 3.0, min_frac: float = 0.7) -> float:
    """Rotational period (deg) of a spot set: the smallest of 60 / 90 /
    120 under which >= min_frac of the spots land on spots, else 180 -
    never more: by Friedel's law a diffraction pattern is (to the
    precision of spot positions) centrosymmetric, so a 180-degree turn
    always fits as well and would otherwise flip the angle at random."""
    v = np.asarray(vecs, float)
    if len(v) < 2:
        return 180.0
    for period in (60.0, 90.0, 120.0):
        ia, _ = _pair(_rot(v, np.radians(period)), v, tol)
        if len(ia) >= min_frac * len(v):
            return period
    return 180.0


def make_reference(name, pattern, peaks, *, r_min=0.0, r_max=np.inf,
                   tol=3.0, centre=None, guess=None, origin="",
                   colour=None):
    """Reference from a measured pattern and its detected peaks."""
    pat = np.asarray(pattern, dtype=np.float32)
    c = (beam_centre(pat, guess) if centre is None else tuple(centre))
    v = spot_vectors(peaks, c, r_min, r_max)
    return DataReference(name=name, vecs=v,
                         period_deg=self_symmetry(v, tol), origin=origin,
                         pattern=pat, centre=c,
                         colour=colour or (0.5, 0.5, 0.5))


def match_dataset(peaks_all: Sequence[np.ndarray], centres_all,
                  refs: Sequence[DataReference], scan_shape, *,
                  tol=3.0, r_min=0.0, r_max=np.inf, min_spots: int = 3,
                  sampled=None, progress=None, stop=None):
    """Match every pattern against every reference.

    peaks_all   : list (Ny*Nx) of (N, 3) detected peaks (row, col, I)
    centres_all : list (Ny*Nx) of (cy, cx) beam centres
    sampled     : optional bool (Ny*Nx) - positions actually analysed
                  (stride); the rest are NaN
    Returns dict of (Ny, Nx) maps: best (ref index, -1 = none / too few
    spots), score (best), theta (deg, of the best ref), margin (how much
    better the best angle fits than the best different angle: 0 = tie,
    1 = no competitor), n_match, n_spots,
    plus per-reference stacks score_k / theta_k (K, Ny, Nx)."""
    Ny, Nx = scan_shape
    K, N = len(refs), Ny * Nx
    score_k = np.full((K, N), np.nan, np.float32)
    theta_k = np.full((K, N), np.nan, np.float32)
    nm_k = np.zeros((K, N), np.int16)
    mg_k = np.full((K, N), np.nan, np.float32)
    n_sp = np.zeros(N, np.int16)
    for n in range(N):
        if stop is not None and stop():
            raise RuntimeError("stopped by user")
        if sampled is not None and not sampled[n]:
            continue
        v = spot_vectors(peaks_all[n], centres_all[n], r_min, r_max)
        n_sp[n] = len(v)
        if len(v) < min_spots:
            continue
        for k, ref in enumerate(refs):
            res = match_vectors(ref.vecs, v, tol, ref.period_deg)
            score_k[k, n] = res["score"]; theta_k[k, n] = res["theta_deg"]
            nm_k[k, n] = res["n_match"]
            mg_k[k, n] = res["margin"]
        if progress is not None and n % 200 == 0:
            progress(n, N)
    valid = np.isfinite(score_k).any(0)
    best = np.where(valid, np.nanargmax(np.where(np.isfinite(score_k),
                                                 score_k, -1), axis=0), -1)
    idx = np.clip(best, 0, max(K - 1, 0))
    take = lambda a: np.where(valid, a[idx, np.arange(N)], np.nan)
    if progress is not None:
        progress(N, N)
    return dict(
        best=best.reshape(Ny, Nx),
        score=take(score_k).reshape(Ny, Nx),
        theta=take(theta_k).reshape(Ny, Nx),
        margin=take(mg_k).reshape(Ny, Nx),
        n_match=np.where(valid, nm_k[idx, np.arange(N)], 0).reshape(Ny, Nx),
        n_spots=n_sp.reshape(Ny, Nx),
        score_k=score_k.reshape(K, Ny, Nx),
        theta_k=theta_k.reshape(K, Ny, Nx),
        periods=np.array([r.period_deg for r in refs]),
        names=[r.name for r in refs])
