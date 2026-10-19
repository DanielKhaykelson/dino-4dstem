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
"""
from __future__ import annotations

import numpy as np

RD = None          # (H, W) radius of each pixel from the beam centre
RMIN = 0           # inner edge of the analysis annulus (px)
RMAX = 0           # outer edge (px, exclusive)


def set_geometry(H: int, W: int, centre, r_min: int, r_max: int) -> None:
    """Fill the constants flatten_cart reads, for an H x W detector with
    the beam at ``centre`` = (row, col)."""
    global RD, RMIN, RMAX
    yy, xx = np.mgrid[0:H, 0:W]
    RD = np.hypot(yy - float(centre[0]), xx - float(centre[1]))
    RMIN, RMAX = int(r_min), int(r_max)


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


def flatten_one(frame, centre, r_min: int, r_max: int) -> np.ndarray:
    """flatten_cart on ONE (H, W) pattern; the geometry is rebuilt only
    when it changes, and a lock keeps concurrent callers (preview, NMF,
    training loader) from swapping RD / RMIN / RMAX under each other."""
    global _GEOM
    f = np.asarray(frame, dtype=np.float32)
    H, W = f.shape
    key = (H, W, float(centre[0]), float(centre[1]), int(r_min), int(r_max))
    with _LOCK:
        if _GEOM != key:
            set_geometry(H, W, centre, r_min, r_max)
            _GEOM = key
        return flatten_cart(f[None])[0]


def auto_params(mean_pattern, r_min="auto", r_max="auto") -> dict:
    """Beam centre (from the mean pattern) and the analysis annulus:
    'auto' r_min = 1.5 x the direct-beam half-maximum radius, 'auto'
    r_max = half the detector."""
    from denoise_n2n import beam_centre
    m = np.asarray(mean_pattern, dtype=np.float64)
    H, W = m.shape
    cy, cx = beam_centre(m)
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(yy - cy, xx - cx).astype(int)
    prof = (np.bincount(rr.ravel(), m.ravel())
            / np.maximum(np.bincount(rr.ravel()), 1))
    r_beam = int(np.argmax(prof < 0.5 * prof[0])) or 4

    def _px(v, auto):
        v = str(v).strip().lower()
        return int(auto) if v in ("", "auto") else int(float(v))
    lo = _px(r_min, round(1.5 * r_beam))
    hi = _px(r_max, min(H, W) // 2)
    if not (0 <= lo < hi):
        raise ValueError(f"need 0 <= r_min < r_max (got {lo}, {hi})")
    return dict(centre=[float(cy), float(cx)], r_min=lo, r_max=hi)


def flatten_cube_into(cube, out, *, centre, r_min: int, r_max: int,
                      progress=None, cancel=None) -> None:
    """Apply flatten_cart to every pattern of ``cube`` (Ny, Nx, H, W) ->
    ``out``, one scan row at a time (any size of cube)."""
    Ny, Nx, H, W = cube.shape
    set_geometry(H, W, centre, r_min, r_max)
    for y in range(Ny):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        try:
            row = np.asarray(cube[y], dtype=np.float32)
        except Exception:
            row = np.stack([np.asarray(cube[y, x], dtype=np.float32)
                            for x in range(Nx)], 0)
        out[y] = flatten_cart(row)
        if progress is not None:
            progress(y + 1, Ny)
