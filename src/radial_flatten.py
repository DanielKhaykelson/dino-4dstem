"""radial_flatten.py -- radial-background flattening of diffraction patterns.

Vendored, self-contained.  Removes the smooth, azimuthally-symmetric part of
every diffraction pattern so the anisotropic Bragg / orientation signal stands
out.  This is the "flatten" step from the cellulose low-dose 4D-STEM sandbox
(celldenoise): `models/n2n_model.py::remove_radial_background` (subtract the
azimuthal median at each radius) and `pipeline.py::flatten` (divide by the
azimuthal mean).  Reimplemented here in Cartesian detector space so it applies
to any centred 4D-STEM cube, with no dependency on that project.

Why it matters
--------------
The intensity of a diffraction pattern is dominated by a steep, isotropic
falloff with radius -- the direct-beam skirt and the amorphous halo -- that is
nearly identical in every pattern and carries no orientation information.  It
falls by orders of magnitude over the first tens of pixels, so no single linear
or log stretch shows both it and the weak reflections.  Estimating that profile
at each radius and removing it puts a featureless background at ~0 (subtract) or
~1 (divide) and leaves the reflections as local excess.  In the celldenoise
pipeline this is stated bluntly: without it a network learns the falloff and a
detection score reaches 0.72 even against an unrelated frame.

The azimuthal **median** is used for the profile (not the mean): Bragg peaks are
sparse in angle, so the median at a radius is the background and is not dragged
up by the peaks sitting on that ring.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np


def build_radius_bins(H: int, W: int,
                      center: Optional[Tuple[float, float]] = None
                      ) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """Precompute, once for a fixed geometry, the flat pixel indices at each
    integer radius and the per-pixel radius-bin map.

    Returns ``(bins, ridx, n_bins)`` where ``bins[r]`` is the array of flat
    pixel indices whose rounded radius is ``r`` and ``ridx`` is the (H*W,)
    radius-bin index of every pixel.
    """
    cy, cx = (H / 2.0, W / 2.0) if center is None else center
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(yy - cy, xx - cx)
    n_bins = int(np.ceil(rr.max())) + 1
    ridx = np.clip(rr.astype(np.int64), 0, n_bins - 1).ravel()
    order = np.argsort(ridx, kind="stable")
    sorted_bins = ridx[order]
    edges = np.searchsorted(sorted_bins, np.arange(n_bins + 1))
    bins = [order[edges[r]:edges[r + 1]] for r in range(n_bins)]
    return bins, ridx, n_bins


def flatten_frame(img: np.ndarray, mode: str = "subtract",
                  bins: Optional[List[np.ndarray]] = None,
                  ridx: Optional[np.ndarray] = None,
                  center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Remove the azimuthally-symmetric radial background from one pattern.

    ``mode='subtract'`` -> ``img - median_r`` (signed residual; the N2N flatten).
    ``mode='divide'``   -> ``img / mean_r``  (ratio, background ~1; non-negative).

    Pass ``bins``/``ridx`` from :func:`build_radius_bins` to reuse the geometry
    across a whole cube; otherwise they are computed for this frame.
    """
    a = np.asarray(img, dtype=np.float32)
    H, W = a.shape
    if bins is None or ridx is None:
        bins, ridx, _ = build_radius_bins(H, W, center)
    flat = a.ravel()
    if mode == "divide":
        # azimuthal MEAN per radius, then divide (background -> ~1)
        cnt = np.bincount(ridx, minlength=len(bins))
        tot = np.bincount(ridx, weights=flat, minlength=len(bins))
        prof = tot / np.maximum(cnt, 1)
        out = flat / np.maximum(prof[ridx], 1e-9)
    else:
        # azimuthal MEDIAN per radius, then subtract (robust to Bragg peaks)
        prof = np.empty(len(bins), dtype=np.float32)
        for r, idx in enumerate(bins):
            prof[r] = np.median(flat[idx]) if idx.size else 0.0
        out = flat - prof[ridx]
    return out.reshape(H, W)


def flatten_cube_into(cube, out, *, mode: str = "subtract",
                      center: Optional[Tuple[float, float]] = None,
                      progress: Optional[Callable[[int, int], None]] = None,
                      cancel: Optional[Callable[[], bool]] = None):
    """Flatten every pattern of ``cube`` (Ny,Nx,H,W) into ``out``.

    ``out`` supports ``out[y] = block`` (e.g. a ``np.lib.format.open_memmap``).
    Streams one scan row at a time.  The radius geometry is precomputed once and
    reused for every pattern.  ``progress(row, Ny)`` per row.
    """
    Ny, Nx = cube.shape[:2]
    H, W = cube.shape[-2], cube.shape[-1]
    bins, ridx, _ = build_radius_bins(H, W, center)
    for y in range(Ny):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        try:
            row = np.asarray(cube[y], dtype=np.float32)
        except Exception:
            row = np.stack([np.asarray(cube[y, x], dtype=np.float32)
                            for x in range(Nx)], 0)
        res = np.empty((Nx, H, W), dtype=np.float32)
        for x in range(Nx):
            res[x] = flatten_frame(row[x], mode=mode, bins=bins, ridx=ridx)
        out[y] = res
        if progress is not None:
            progress(y + 1, Ny)
