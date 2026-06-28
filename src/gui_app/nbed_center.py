"""nbed_center.py -- BF-disk centering for NBED / low-mag 4D-STEM.

The default `CenterOnCOM` (intensity-weighted centroid in a constrained
search disk) misbehaves on NBED data because:
  * the BF disk is large and often saturated → the centroid is biased
    toward whichever side of the saturated plateau happens to have more
    un-clipped pixels;
  * weak diffuse haloes (amorphous support, charging streaks) drag the
    mass off the BF spot;
  * off-axis Bragg disks pull the COM if they fall inside the search
    disk.

This module provides a more robust centring that uses the **shape** of
the BF disk rather than the total intensity:

  1. Find the brightest pixel within a coarse search disk (handles
     small misalignments).
  2. Inside a smaller disk centred on that pixel, take the intensity-
     weighted centroid of pixels above `threshold_frac × local_max`.
     Saturated plateaus contribute uniformly, so the centroid sits at
     the geometric centre of the BF disk shape — not at the centroid
     of the intensity field.

The whole cube is rolled (integer pixel) per pattern; sub-pixel shifts
are not applied (the rest of the pipeline does its own resize/COM).
"""
from __future__ import annotations
import os
import numpy as np


def find_bf_center(pattern: np.ndarray, *,
                     search_radius: float,
                     disk_radius: float,
                     threshold_frac: float = 0.5) -> "tuple[float, float]":
    """Return (cy, cx) — the BF-disk centre of a single 2D pattern.

    Coordinates are in the pattern's pixel space (cy is the row
    coordinate, cx the column coordinate).
    """
    H, W = pattern.shape
    cy0 = (H - 1) / 2.0
    cx0 = (W - 1) / 2.0
    p = np.asarray(pattern, dtype=np.float32)
    # NaN / inf would silently poison argmax (NaN compares > -inf in
    # numpy) and the subsequent centroid math; sanitise.
    if not np.isfinite(p).all():
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    yy = np.arange(H, dtype=np.float32).reshape(H, 1)
    xx = np.arange(W, dtype=np.float32).reshape(1, W)
    dy = yy - cy0
    dx = xx - cx0

    # Step 1 — coarse: brightest pixel within `search_radius` of the
    # geometric centre.
    in_search = (dy * dy) + (dx * dx) <= float(search_radius) ** 2
    if not bool(in_search.any()):
        # search radius too small for any pixel — fall back to centre
        return cy0, cx0
    masked = np.where(in_search, p, -np.inf)
    iy, ix = np.unravel_index(np.argmax(masked), masked.shape)
    iy = float(iy); ix = float(ix)

    # Step 2 — refine: threshold-centroid within `disk_radius` of the
    # bright pixel.
    dy2 = yy - iy
    dx2 = xx - ix
    in_disk = (dy2 * dy2) + (dx2 * dx2) <= float(disk_radius) ** 2
    region = p * in_disk
    vmax = float(region.max())
    if vmax <= 0:
        return iy, ix
    above = (region >= float(threshold_frac) * vmax) & in_disk
    if not bool(above.any()):
        return iy, ix
    w = p * above.astype(np.float32)
    s = float(w.sum())
    if s <= 0:
        return iy, ix
    cy = float((yy * w).sum() / s)
    cx = float((xx * w).sum() / s)
    return cy, cx


def recenter_to(pattern: np.ndarray, cy: float, cx: float) -> np.ndarray:
    """Integer-roll `pattern` so (cy, cx) lands at the geometric
    centre.  Sub-pixel residual is left to whatever downstream stage
    (typically: model's polar-transform / resize)."""
    H, W = pattern.shape
    cy0 = (H - 1) / 2.0
    cx0 = (W - 1) / 2.0
    sy = int(round(cy0 - cy))
    sx = int(round(cx0 - cx))
    return np.roll(np.roll(pattern, sy, axis=0), sx, axis=1)


def nbed_center_cube(cube_in,
                      dest_path: str,
                      *,
                      search_radius: float,
                      disk_radius: float,
                      threshold_frac: float = 0.5,
                      progress_cb=None,
                      stop_flag=None) -> np.ndarray:
    """Recenter every pattern in `cube_in` (shape Ny, Nx, H, W) using
    `find_bf_center`, write the result to `dest_path` as a memory-
    mapped numpy array (`.cube.npy` format).

    Parameters
    ----------
    cube_in : array-like, shape (Ny, Nx, H, W)
        Source cube — typically a memory-mapped numpy array.
    dest_path : str
        Path to write the result.  Must end in `.npy`.
    search_radius, disk_radius, threshold_frac : see `find_bf_center`.
    progress_cb : callable(done, total), optional
        Called once per scan row (i.e. `total = Ny`).
    stop_flag : callable() -> bool, optional
        Polled once per row; if it returns True, processing aborts and
        the partially written file is closed (but not deleted — caller
        decides).

    Returns
    -------
    centers : (Ny, Nx, 2) float32 array
        Per-pattern (cy, cx) centres in raw-pattern pixel coordinates.
    """
    # Do NOT np.asanyarray(cube_in) the whole cube — lazy loaders (h5 master,
    # zip-offset memmap) expose .shape + [y, x] indexing but materialize to a
    # 0-d object array under np.asanyarray. Read the shape directly; the
    # per-pattern loop below indexes cube_in[y, x] which all backends support.
    shape = getattr(cube_in, "shape", None)
    if shape is None:
        cube_in = np.asanyarray(cube_in)
        shape = cube_in.shape
    if len(shape) != 4:
        raise ValueError(f"cube_in must be 4D (Ny, Nx, H, W); got "
                          f"shape {tuple(shape)}")
    Ny, Nx, H, W = (int(s) for s in shape)
    if not dest_path.lower().endswith(".npy"):
        raise ValueError("dest_path must end in .npy")

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".",
                 exist_ok=True)
    out = np.lib.format.open_memmap(
        dest_path, mode="w+", dtype=np.float32,
        shape=(Ny, Nx, H, W))
    centers = np.empty((Ny, Nx, 2), dtype=np.float32)

    sr = float(search_radius); dr = float(disk_radius)
    tf = float(threshold_frac)
    for y in range(Ny):
        if stop_flag is not None and stop_flag():
            break
        for x in range(Nx):
            p = np.asarray(cube_in[y, x]).astype(np.float32, copy=False)
            cy, cx = find_bf_center(p, search_radius=sr,
                                       disk_radius=dr,
                                       threshold_frac=tf)
            centers[y, x] = (cy, cx)
            out[y, x] = recenter_to(p, cy, cx)
        if progress_cb is not None:
            try: progress_cb(y + 1, Ny)
            except Exception: pass
    out.flush()
    # Drop the memmap so the file handle is released. On Windows this
    # is required before the caller can shutil.move() the file.
    del out
    import gc as _gc
    _gc.collect()
    return centers
