"""lattice_fit.py -- fit a 2-D lattice to a diffraction pattern's INTENSITY.

The spot-based route (detect peaks -> pick basis vectors -> count indexed
spots against pixel / percentage thresholds) treats the list of detected
peaks as the truth, so every spurious peak (beam halo, detector seam,
streak), every missed weak spot and every chance coincidence becomes an
error of its own.  Here the lattice is fitted to the image itself:

  1. spot image  S = G_sigma * (A - G_8 * A),  A = 2 sqrt(I + 3/8)
     (Anscombe: Poisson noise -> unit variance; the wide Gaussian removes
     the smooth background, the narrow one matches the spot size);
  2. lattice (B = [g1 g2], offset t) maximises the mean of S over its sites
     in the analysis ring - L-BFGS with the analytic gradient, coarse to
     fine (sigma 3 px, then 1.5 px), the set of sites fixed by the start;
  3. evidence  z = (Q - median Q_null) / (1.4826 MAD Q_null), where Q_null
     is the same score for the same lattice at random in-plane rotations
     (away from its symmetry angles): how much more the pattern looks like
     this lattice than like a random one of the same cell.  No peak counts,
     no pixel tolerance;
  4. explained: the share of the spot intensity on the lattice sites,
     corrected for chance - whether this lattice accounts for the pattern
     (a wrong lattice of similar spacing has a significant z through its
     coinciding sites, but explains only part of the intensity).

Detected peaks are used only for the starting rotation (dataref_core).
Axes: x = detector right, y = detector UP (as strain_core).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize


class SpotImage:
    """Background-free, noise-stabilised, spot-matched image + gradient."""

    def __init__(self, pat, sigma: float = 1.5, bg_sigma: float = 8.0):
        a = 2.0 * np.sqrt(np.clip(np.asarray(pat, np.float64), 0, None)
                          + 0.375)
        s = gaussian_filter(a - gaussian_filter(a, bg_sigma), sigma)
        self.S = s
        self.gy, self.gx = np.gradient(s)
        self.H, self.W = s.shape

    def sample(self, rows, cols):
        c = np.vstack([rows, cols])
        return (map_coordinates(self.S, c, order=1, mode="nearest"),
                map_coordinates(self.gy, c, order=1, mode="nearest"),
                map_coordinates(self.gx, c, order=1, mode="nearest"))


def site_hk(B, r_min, r_max, nmax=40):
    h = np.arange(-nmax, nmax + 1)
    hk = np.stack(np.meshgrid(h, h, indexing="ij"), -1).reshape(-1, 2)
    r = np.hypot(*(hk @ B.T).T)
    return hk[(r >= r_min) & (r <= r_max)].astype(float)


def _score(img, centre, B, t, hk):
    xy = hk @ B.T + t
    rows, cols = centre[0] - xy[:, 1], centre[1] + xy[:, 0]
    ok = (rows >= 1) & (rows < img.H - 2) & (cols >= 1) & (cols < img.W - 2)
    if ok.sum() < 3:
        return -np.inf, None
    s, gy, gx = img.sample(rows[ok], cols[ok])
    return float(s.mean()), (hk[ok], gx, -gy, ok.sum())


def refine(img, centre, B0, t0, hk, max_iter=60):
    """Maximise the mean spot-image value over the lattice sites."""
    def f(p):
        B = p[:4].reshape(2, 2); t = p[4:]
        q, aux = _score(img, centre, B, t, hk)
        if aux is None:
            return 1e6, np.zeros(6)
        h, dx, dy, n = aux
        # d/dB[i, j] of mean S(x, y): x = B[0]·hk + t0, y = B[1]·hk + t1
        g = np.zeros(6)
        g[0] = (dx * h[:, 0]).sum(); g[1] = (dx * h[:, 1]).sum()
        g[2] = (dy * h[:, 0]).sum(); g[3] = (dy * h[:, 1]).sum()
        g[4] = dx.sum(); g[5] = dy.sum()
        return -q, -g / n
    p0 = np.concatenate([np.asarray(B0, float).ravel(), np.asarray(t0, float)])
    r = minimize(f, p0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=max_iter))
    p = r.x
    return p[:4].reshape(2, 2), p[4:], -float(r.fun)


def rot2(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def evidence(img, centre, B, t, hk, period_deg=180.0, n=16, seed=0):
    """z-score of the lattice against itself at random rotations."""
    q, _ = _score(img, centre, B, t, hk)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n * 3):
        if len(nulls) >= n:
            break
        a = rng.uniform(0, 360)
        if min(abs((a + period_deg / 2) % period_deg - period_deg / 2),
               abs((-a + period_deg / 2) % period_deg - period_deg / 2)) < 8:
            continue
        qn, _ = _score(img, centre, rot2(a) @ B, t, hk)
        if np.isfinite(qn):
            nulls.append(qn)
    nulls = np.asarray(nulls)
    if len(nulls) < 4:
        return q, np.nan
    med = np.median(nulls)
    mad = 1.4826 * np.median(np.abs(nulls - med)) + 1e-9
    return q, float((q - med) / mad)


def explained(img, centre, B, t, r_min, r_max, rad=2.5):
    """Share of the pattern's spot intensity (positive part of the spot
    image, inside the ring) that lies within ``rad`` px of the lattice
    sites, corrected for chance (the share of the ring the site disks
    cover):  (E - c) / (1 - c).  1 = the lattice explains every spot,
    0 = no better than a random set of positions."""
    H, W = img.H, img.W
    yy, xx = np.mgrid[0:H, 0:W]
    x, y = xx - centre[1], centre[0] - yy
    r = np.hypot(x, y)
    ring = (r >= r_min) & (r <= min(r_max, min(H, W) / 2))
    hk = site_hk(B, max(r_min - rad, 0), r_max + rad)
    xy = hk @ B.T + t
    mask = np.zeros((H, W), bool)
    R = int(np.ceil(rad))
    dy, dx = np.mgrid[-R:R + 1, -R:R + 1]
    disk = np.hypot(dy, dx) <= rad
    dy, dx = dy[disk], dx[disk]
    rows = np.rint(centre[0] - xy[:, 1]).astype(int)
    cols = np.rint(centre[1] + xy[:, 0]).astype(int)
    rr = (rows[:, None] + dy[None, :]).ravel()
    cc = (cols[:, None] + dx[None, :]).ravel()
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    mask[rr[ok], cc[ok]] = True
    # spot intensity = spot-image values clearly above its noise (robust
    # sigma inside the ring); the positive half of the noise is not "spot"
    v = img.S[ring]
    noise = 1.4826 * np.median(np.abs(v - np.median(v)))
    sp = np.clip(img.S - 2.0 * noise, 0, None)
    tot = sp[ring].sum()
    if tot <= 0 or ring.sum() == 0:
        return np.nan
    E = sp[ring & mask].sum() / tot
    c = (ring & mask).sum() / ring.sum()
    return float((E - c) / max(1.0 - c, 1e-6))


def fit(pat, centre, B0, t0=(0.0, 0.0), r_min=0.0, r_max=np.inf,
        period_deg=180.0, imgs=None):
    """Intensity lattice fit, coarse to fine.  -> (B, t, z, explained).
    ``imgs`` may pass prepared (coarse, fine) SpotImages to reuse across
    lattices."""
    if imgs is None:
        imgs = (SpotImage(pat, 3.0), SpotImage(pat, 1.5))
    B, t = np.asarray(B0, float), np.asarray(t0, float)
    hk = site_hk(B, r_min, r_max)
    if len(hk) < 4:
        return B, t, np.nan, np.nan
    for img in imgs:
        B, t, _q = refine(img, centre, B, t, hk)
    hk = site_hk(B, r_min, r_max)
    _q, z = evidence(imgs[-1], centre, B, t, hk, period_deg)
    return B, t, z, explained(imgs[-1], centre, B, t, r_min, r_max)
