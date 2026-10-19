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
