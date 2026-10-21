"""celldenoise_flatten.py -- the cellulose project's annulus-median flattening.

``flatten_cart`` below is copied VERBATIM (extracted byte-for-byte, not
rewritten) from the cellulose low-dose 4D-STEM project,
``denoise/pipeline/step154_cartesian.py::flatten_cart``, so both projects
flatten patterns identically:

    for every 1-px-wide annulus about the beam centre, subtract the median of
    the pattern on that annulus; zero everything outside RMIN <= r < RMAX;
    divide each pattern by its standard deviation inside that annulus.

The original reads three module constants -- ``RD`` (radius of every pixel
from the centre), ``RMIN`` and ``RMAX`` (the analysis annulus).  They are set
here by :func:`set_geometry` for the loaded cube instead of the fixed 208 x
208 crop of that pipeline; the function body itself is untouched.

``flatten_fast`` computes exactly the same numbers (tested equal to
``flatten_cart``) ~3x faster: it gathers each 1-px annulus once per
geometry instead of scanning the whole frame once per radius, which matters
when every pattern of a scan is flattened on the fly.

Like the cellulose loader, every pattern is first integer-shifted so its own
direct beam sits on the reference centre (``recentre``); otherwise a beam
that wanders across the scan leaves a bright crescent of its disk edge in
the annulus.

DINO-4DSTEM uses the POSITIVE part of the residual (``clip=True`` in the
wrappers): pixels brighter than their annulus median, in units of the
pattern's std.  Negative residuals are sub-background noise and would make
the per-pattern min/max normalisation of the model input meaningless.
"""
from __future__ import annotations

import numpy as np

RD = None          # (H, W) radius of each pixel from the beam centre
RMIN = 0           # inner edge of the analysis annulus (px)
RMAX = 0           # outer edge (px, exclusive)


_ORDER = None      # flat indices of annulus pixels, grouped by radius
_SEGS = None       # (start, stop) of every radius RMIN..RMAX-1 in _ORDER
_RIDX = None       # profile index of every annulus pixel (flatten_cart's)
_M = None          # annulus mask


def set_geometry(H: int, W: int, centre, r_min: int, r_max: int) -> None:
    """Fill the constants flatten_cart reads, for an H x W detector with
    the beam at ``centre`` = (row, col)."""
    global RD, RMIN, RMAX, _ORDER, _SEGS, _RIDX, _M
    yy, xx = np.mgrid[0:H, 0:W]
    RD = np.hypot(yy - float(centre[0]), xx - float(centre[1]))
    RMIN, RMAX = int(r_min), int(r_max)
    # lookup tables for flatten_fast (same rings as flatten_cart)
    ri = RD.astype(int)
    _M = (RD >= RMIN) & (RD < RMAX)
    flat_ri = ri.ravel()
    sel = np.flatnonzero((flat_ri >= RMIN) & (flat_ri < RMAX))
    _ORDER = sel[np.argsort(flat_ri[sel], kind="stable")]
    cuts = np.searchsorted(flat_ri[_ORDER], np.arange(RMIN, RMAX + 1))
    _SEGS = list(zip(cuts[:-1], cuts[1:]))
    _RIDX = np.clip(ri[_M] - RMIN, 0, RMAX - RMIN - 1)


def flatten_fast(img):
    """Same result as flatten_cart (same rings, medians, mask and std),
    without re-scanning the frame for every radius."""
    out = np.empty_like(img)
    for i in range(len(img)):
        f = img[i]
        v = f.ravel()[_ORDER]
        prof = np.array([np.median(v[a:b]) if b > a else 0.0
                         for a, b in _SEGS])
        bg = np.zeros_like(f)
        bg[_M] = prof[_RIDX]
        g = (f - bg) * _M
        out[i] = g / max(g[_M].std(), 1e-6)
    return out


# ---- verbatim from step154_cartesian.py ----------------------------------
def flatten_cart(img):
    """The Cartesian analogue of nrm: subtract the radial median profile.

    The profile is evaluated per pixel from a 1-px-wide radial binning, so
    the DATA is never interpolated - only the background model is.
    """
    ri = RD.astype(int)
    out = np.empty_like(img)
    m = (RD >= RMIN) & (RD < RMAX)
    for i in range(len(img)):
        f = img[i]
        prof = np.array([np.median(f[ri == k]) if (ri == k).any() else 0.0
                         for k in range(RMIN, RMAX)])
        bg = np.zeros_like(f)
        bg[m] = prof[np.clip(ri[m] - RMIN, 0, RMAX - RMIN - 1)]
        g = (f - bg) * m
        out[i] = g / max(g[m].std(), 1e-6)
    return out
# ---- end verbatim ---------------------------------------------------------


import threading
_LOCK = threading.Lock()
_GEOM = None       # (H, W, cy, cx, r_min, r_max) currently in RD/RMIN/RMAX


def _ensure_geometry(H, W, centre, r_min, r_max):
    global _GEOM
    key = (H, W, float(centre[0]), float(centre[1]), int(r_min), int(r_max))
    if _GEOM != key:
        set_geometry(H, W, centre, r_min, r_max)
        _GEOM = key


def recentre(frame, centre, max_shift=None) -> np.ndarray:
    """Integer-shift ``frame`` so its own direct beam sits on ``centre``
    (zero fill), as the cellulose loader does before flatten_cart.  A
    shift larger than ``max_shift`` (default 10% of the detector) is
    taken as a failed beam detection and the frame is left alone."""
    from denoise_n2n import beam_centre
    f = np.asarray(frame, dtype=np.float32)
    H, W = f.shape
    cy, cx = beam_centre(f)
    dy, dx = int(round(centre[0] - cy)), int(round(centre[1] - cx))
    lim = 0.1 * min(H, W) if max_shift is None else max_shift
    if (dy == 0 and dx == 0) or max(abs(dy), abs(dx)) > lim:
        return f
    out = np.zeros_like(f)
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] =         f[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out


def flatten_one(frame, centre, r_min: int, r_max: int,
                clip: bool = True, recentre_beam: bool = True) -> np.ndarray:
    """Flatten ONE (H, W) pattern (flatten_fast == flatten_cart); first
    integer-recentre it on its own beam (``recentre_beam``), and with
    ``clip`` keep only the positive residual.  The geometry is rebuilt
    only when it changes, and a lock keeps concurrent callers (preview,
    NMF, training loader) from swapping the tables under each other."""
    f = (recentre(frame, centre) if recentre_beam
         else np.asarray(frame, dtype=np.float32))
    H, W = f.shape
    with _LOCK:
        _ensure_geometry(H, W, centre, r_min, r_max)
        g = flatten_fast(f[None])[0]
    return np.maximum(g, 0.0, out=g) if clip else g


def suggest_vmax(frames, centre, r_min: int, r_max: int) -> float:
    """A display/model vmax for flattened data: the 99.5th percentile of
    the positive residual (std units) over the sample ``frames``, so only
    the brightest Bragg pixels saturate."""
    vals = []
    for fr in frames:
        g = flatten_one(fr, centre, r_min, r_max, clip=True)
        vals.append(g[g > 0])
    v = np.concatenate(vals) if vals else np.zeros(1)
    q = float(np.percentile(v, 99.5)) if v.size else 1.0
    return float(f"{max(q, 0.5):.2g}")


def auto_params(mean_pattern, r_min="auto", r_max="auto") -> dict:
    """Beam centre (from the mean pattern) and the analysis annulus:
    'auto' r_min = 1.3 x the direct-beam disk edge, 'auto' r_max = half the
    detector."""
    from denoise_n2n import beam_centre
    m = np.asarray(mean_pattern, dtype=np.float64)
    H, W = m.shape
    cy, cx = beam_centre(m)
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(yy - cy, xx - cx).astype(int)
    prof = (np.bincount(rr.ravel(), m.ravel())
            / np.maximum(np.bincount(rr.ravel()), 1))
    # direct-beam disk edge = steepest fall of the log radial profile in
    # the inner quarter (the half-maximum radius sits INSIDE a flat-topped
    # disk and lets its edge leak into the annulus)
    lp = np.log(np.maximum(prof[:max(6, min(H, W) // 4)], 1e-12))
    r_beam = int(np.argmin(np.diff(lp))) + 1

    def _px(v, auto):
        v = str(v).strip().lower()
        return int(auto) if v in ("", "auto") else int(float(v))
    lo = _px(r_min, int(np.ceil(1.3 * r_beam)) + 1)
    hi = _px(r_max, min(H, W) // 2)
    if not (0 <= lo < hi):
        raise ValueError(f"need 0 <= r_min < r_max (got {lo}, {hi})")
    return dict(centre=[float(cy), float(cx)], r_min=lo, r_max=hi)


def flatten_cube_into(cube, out, *, centre, r_min: int, r_max: int,
                      progress=None, cancel=None, clip: bool = True,
                      recentre_beam: bool = True) -> None:
    """Flatten every pattern of ``cube`` (Ny, Nx, H, W) -> ``out``, one
    scan row at a time (any size of cube)."""
    Ny, Nx, H, W = cube.shape
    for y in range(Ny):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        try:
            row = np.asarray(cube[y], dtype=np.float32)
        except Exception:
            row = np.stack([np.asarray(cube[y, x], dtype=np.float32)
                            for x in range(Nx)], 0)
        if recentre_beam:
            row = np.stack([recentre(fr, centre) for fr in row], 0)
        with _LOCK:
            _ensure_geometry(H, W, centre, r_min, r_max)
            g = flatten_fast(row)
        out[y] = np.maximum(g, 0.0, out=g) if clip else g
        if progress is not None:
            progress(y + 1, Ny)
