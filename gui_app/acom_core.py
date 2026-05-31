"""acom_core.py -- single-pattern + batch ACOM helpers.

Backend for any ACOM mode in the GUI: full cube, per-class average,
per-grain average.  None of these helpers depend on a DataCube — they
take 2D patterns directly and produce a BraggVectors object suitable
for `Crystal.match_orientations`.

Pipeline (per pattern):
    1. detect_peaks_2d(pattern, ...)              -> (N, 3) (qx, qy, I)
    2. build_bragg_vectors(peaks_list, qx0, qy0,  -> BraggVectors
                              inv_ang_per_pixel)
    3. crystal.match_orientations(bv)              -> OrientationMap

For class / grain averages we typically have NO vacuum probe template
(it's an averaged pattern, not a single scan position), so detection
defaults to **scikit-image blob_log** on a log-stretched copy of the
pattern.  Pass `probe_kernel=...` to use py4DSTEM template matching
instead.

Calibration note (from the user's microscope):
    raw detector  = 0.0185 nm^-1 / px
                  = 0.00185 1/Å / px       (1 nm^-1 = 0.1 1/Å)
So `inv_ang_per_pixel` defaults to 0.00185 here.
"""
from __future__ import annotations
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# py4DSTEM imports (lazy, so the module loads even when py4DSTEM is absent)
# ---------------------------------------------------------------------------

def _import_py4DSTEM():
    """Return (py4DSTEM module, Crystal class) or (None, None) if missing."""
    try:
        import py4DSTEM
        from py4DSTEM.process.diffraction import Crystal
        return py4DSTEM, Crystal
    except Exception:
        return None, None


def _import_pointlist():
    try:
        from py4DSTEM.io.classes import PointList, PointListArray
        return PointList, PointListArray
    except Exception:
        from emdfile import PointList, PointListArray
        return PointList, PointListArray


def _import_braggvectors():
    try:
        from py4DSTEM.braggvectors import BraggVectors
        return BraggVectors
    except Exception:
        from py4DSTEM import BraggVectors
        return BraggVectors


# ---------------------------------------------------------------------------
# Crystal helpers
# ---------------------------------------------------------------------------

def load_crystal(cif_path: str):
    """Load a CIF as a py4DSTEM Crystal, with the version-robust fallback."""
    _, Crystal = _import_py4DSTEM()
    if Crystal is None:
        raise RuntimeError("py4DSTEM is not importable.")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(cif_path)
    try:
        return Crystal.from_CIF(cif_path)
    except AttributeError:
        return Crystal(filepath=cif_path)


def prepare_crystal(crystal,
                     k_max: float = 2.0,
                     accel_voltage: float = 300e3,
                     plan_mode: str = "fiber",
                     fiber_axis: Sequence[float] = (0.0, 0.0, 1.0),
                     fiber_angles: Sequence[float] = (0.0, 360.0),
                     corners_zone_axes: Optional[np.ndarray] = None,
                     angle_step_zone_axis: float = 2.0,
                     angle_step_in_plane: float = 2.0,
                     corr_kernel_size: float = 0.05):
    """Compute structure factors + orientation plan in one shot.

    Defaults match what `blob_acom_panel.py` uses in the GUI — fiber
    mode along [0,0,1] with 2° resolution — which is a sensible
    starting point for a crystal whose orientation we don't know a
    priori.  Pass `plan_mode="corners"` + `corners_zone_axes` to
    constrain to a triangle.
    """
    # Lower tol_structure_factor (default ~1e-2) so weak low-index
    # rings like γ(100) or α(002) — which are intensity-weak in
    # molecular crystals because the unit-cell atoms destructively
    # interfere along single-axis directions — are kept in the
    # structure factor list and show up on the 1D plot.
    try:
        crystal.calculate_structure_factors(
            float(k_max), tol_structure_factor=1e-4)
    except TypeError:
        crystal.calculate_structure_factors(float(k_max))
    # Build a kwarg dict that is filtered against the installed
    # py4DSTEM version's `orientation_plan` signature so we don't blow
    # up on kwargs that have been renamed across versions.
    import inspect
    sig = inspect.signature(crystal.orientation_plan)
    available = set(sig.parameters.keys())

    common = dict(
        angle_step_zone_axis=float(angle_step_zone_axis),
        angle_step_in_plane=float(angle_step_in_plane),
        accel_voltage=float(accel_voltage),
        corr_kernel_size=float(corr_kernel_size),
        # power knobs — names changed between 0.13 and 0.14
        power_radial=1.0, radial_power=1.0,
        power_intensity=0.25, intensity_power=0.25,
        tol_distance=0.01, tol_intensity=1e-3,
        progress_bar=False,
    )
    kw = {k: v for k, v in common.items() if k in available}

    if plan_mode == "fiber":
        if "fiber_axis" in available:
            kw["fiber_axis"] = list(map(float, fiber_axis))
            kw["fiber_angles"] = list(map(float, fiber_angles))
        if "zone_axis_range" in available:
            # In 0.14 zone_axis_range is an ndarray; the string "fiber"
            # was a 0.13 convention.  Skip it when fiber_axis is set.
            if "fiber_axis" not in available:
                kw["zone_axis_range"] = "fiber"
        crystal.orientation_plan(**kw)
    else:
        rng = (np.asarray(corners_zone_axes, dtype=np.float64)
                  if corners_zone_axes is not None
                  else np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  dtype=np.float64))
        if "zone_axis_range" in available:
            crystal.orientation_plan(zone_axis_range=rng, **kw)
        else:
            crystal.orientation_plan(corners_zone_axes=rng, **kw)
    return crystal


# ---------------------------------------------------------------------------
# Peak detection (per single 2D pattern)
# ---------------------------------------------------------------------------

def detect_peaks_cached(patterns, detect_kw, cache_path,
                            centers=None, progress_cb=None,
                            force=False):
    """Detect peaks on a list of 2D patterns with on-disk caching.

    Saves an .npz keyed by the detection params + number of patterns,
    so re-running matching on the same source skips the (slow) blob
    detection.  Returns a list of (Ni, 3) peak arrays.

    cache_path : .npz file to read/write.
    force      : ignore any existing cache and recompute.
    """
    import hashlib, json
    key = hashlib.md5(
        json.dumps({"n": len(patterns),
                      "kw": {k: float(v) if isinstance(v, (int, float))
                                else v for k, v in detect_kw.items()}},
                     sort_keys=True, default=str).encode()).hexdigest()
    if (not force) and cache_path and os.path.exists(cache_path):
        try:
            d = np.load(cache_path, allow_pickle=True)
            if str(d.get("key")) == key:
                arrs = list(d["peaks"])
                return [np.asarray(a, dtype=float) for a in arrs]
        except Exception:
            pass
    out = []
    n = len(patterns)
    for i, pat in enumerate(patterns):
        out.append(detect_peaks_2d(pat, **detect_kw))
        if progress_cb is not None and (i % 64 == 0):
            try: progress_cb(i, n, "detect")
            except Exception: pass
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(
                cache_path, key=key,
                peaks=np.array(out, dtype=object))
        except Exception as e:
            print(f"[acom_core] peak cache save failed: {e!r}",
                  flush=True)
    return out


def detect_peaks_2d(pattern: np.ndarray,
                       probe_kernel: Optional[np.ndarray] = None,
                       *,
                       min_sigma: float = 1.5,
                       max_sigma: float = 8.0,
                       num_sigma: int = 6,
                       threshold: float = 0.02,
                       overlap: float = 0.4,
                       log_stretch: bool = True,
                       py4dstem_params: Optional[dict] = None
                       ) -> np.ndarray:
    """Detect Bragg peaks on a single 2D pattern.

    Returns an (N, 3) array of (qx, qy, intensity) in **raw detector
    pixels** (no centering applied here — the caller subtracts qx0/qy0
    when building BraggVectors).

    Two modes:
      • `probe_kernel` provided   → py4DSTEM template matching
        (`find_Bragg_disks_single_DP`).
      • otherwise                  → scikit-image `blob_log` on
        log1p(pattern) with sensible defaults.  This is robust on
        class/grain *averages* where no vacuum probe is available.
    """
    pat = np.asarray(pattern, dtype=np.float32)
    if pat.ndim != 2:
        raise ValueError(f"pattern must be 2D, got shape {pat.shape}")

    if probe_kernel is not None:
        # py4DSTEM template matching path.
        py4dstem, _ = _import_py4DSTEM()
        if py4dstem is None:
            raise RuntimeError("py4DSTEM required for template matching.")
        from py4DSTEM.process.diskdetection import (
            find_Bragg_disks_single_DP)
        kw = dict(
            corrPower=1.0, sigma=2.0, edgeBoundary=4,
            minRelativeIntensity=1e-3, minAbsoluteIntensity=1e-6,
            minPeakSpacing=8, subpixel="poly", upsample_factor=16,
            maxNumPeaks=200,
        )
        if py4dstem_params:
            kw.update(py4dstem_params)
        pl = find_Bragg_disks_single_DP(pat, probe_kernel, **kw)
        qx = np.asarray(pl.data["qx"], dtype=float)
        qy = np.asarray(pl.data["qy"], dtype=float)
        if "intensity" in pl.data.dtype.names:
            inten = np.asarray(pl.data["intensity"], dtype=float)
        else:
            inten = np.ones_like(qx)
        return np.stack([qx, qy, inten], axis=1)

    # blob_log fallback (the only mode that works on class/grain avgs).
    try:
        from skimage.feature import blob_log
    except Exception as e:
        raise RuntimeError(
            f"scikit-image required for blob detection: {e!r}")

    # Normalise so the threshold is meaningful regardless of count scale.
    p = pat - np.median(pat)
    p = np.clip(p, 0.0, None)
    if log_stretch:
        p = np.log1p(p)
    p = p / (p.max() + 1e-9)

    blobs = blob_log(p, min_sigma=float(min_sigma),
                          max_sigma=float(max_sigma),
                          num_sigma=int(num_sigma),
                          threshold=float(threshold),
                          overlap=float(overlap))
    if blobs.size == 0:
        return np.zeros((0, 3), dtype=float)
    # blob_log returns (y, x, sigma).  Use blob amplitude (the value of
    # p at the blob centre) as intensity proxy — it's bounded [0,1] and
    # downstream calibration scales it.
    yy = blobs[:, 0]
    xx = blobs[:, 1]
    sig = blobs[:, 2]
    H, W = p.shape
    ii = np.clip(yy.astype(int), 0, H - 1)
    jj = np.clip(xx.astype(int), 0, W - 1)
    amp = p[ii, jj]
    # In py4DSTEM convention, qx = vertical (row), qy = horizontal (col).
    return np.stack([yy, xx, amp * (1.0 + sig)], axis=1)


# ---------------------------------------------------------------------------
# BraggVectors construction (per pattern OR list of patterns)
# ---------------------------------------------------------------------------

def build_bragg_vectors(peak_lists: Sequence[np.ndarray],
                          *,
                          centers: Optional[Sequence[Tuple[float, float]]] = None,
                          inv_ang_per_pixel: float = 0.00185,
                          Rshape: Optional[Tuple[int, int]] = None,
                          ):
    """Wrap a sequence of peak arrays into a calibrated BraggVectors.

    peak_lists : list of (Ni, 3) arrays of (qx, qy, intensity) in raw
                  detector pixels (NOT centered).
    centers    : list of (qx0, qy0) per pattern; if None we use the
                  pattern centre (assumed already centered).  All
                  vectors are stored as (qx - qx0, qy - qy0) so the BF
                  disk sits at the origin.
    inv_ang_per_pixel : 1/Å per raw detector pixel.  For the user's
                  data, 0.0185 nm^-1/px → 0.00185 1/Å/px.
    Rshape     : (Ny, Nx) layout of the PointListArray.  Default = (N, 1).
    """
    PointList, PointListArray = _import_pointlist()
    BraggVectors = _import_braggvectors()

    N = len(peak_lists)
    if N == 0:
        raise ValueError("peak_lists is empty.")
    # Convention: lay out the N patterns along R_x with R_y = 1.
    # `bv.cal[k, 0]` then returns the k-th calibrated pattern.
    if Rshape is None:
        Rshape = (N, 1)
    R_Nx, R_Ny = Rshape
    if R_Nx * R_Ny != N:
        raise ValueError(
            f"Rshape={Rshape} not compatible with N={N} patterns.")
    if centers is None:
        centers = [None] * N

    # py4DSTEM >= 0.14 uses emdfile's PointListArray (dtype + shape +
    # name).  Older versions had a `coordinates=` kw.  Build the array
    # in a version-tolerant way.
    dt = np.dtype([("qx", float), ("qy", float), ("intensity", float)])
    try:
        pla = PointListArray(dtype=dt, shape=(R_Nx, R_Ny))
    except TypeError:
        coords = [("qx", float), ("qy", float), ("intensity", float)]
        pla = PointListArray(coordinates=coords, shape=(R_Nx, R_Ny))

    def _make_point_list(arr):
        try:
            return PointList(arr)
        except TypeError:
            coords = [("qx", float), ("qy", float), ("intensity", float)]
            return PointList(arr, coordinates=coords)

    empty_arr = np.zeros(0, dtype=dt)

    for k, peaks in enumerate(peak_lists):
        rx, ry = divmod(k, R_Ny)
        peaks = np.asarray(peaks, dtype=float)
        if peaks.ndim != 2 or peaks.shape[0] == 0:
            pla[rx, ry] = _make_point_list(empty_arr)
            continue
        qx = peaks[:, 0].copy()
        qy = peaks[:, 1].copy()
        inten = (peaks[:, 2].copy() if peaks.shape[1] > 2
                    else np.ones(peaks.shape[0]))
        if centers[k] is not None:
            cx, cy = centers[k]
            qx -= float(cx)
            qy -= float(cy)
        arr = np.zeros(len(qx), dtype=dt)
        arr["qx"] = qx; arr["qy"] = qy; arr["intensity"] = inten
        pla[rx, ry] = _make_point_list(arr)

    try:
        bv = BraggVectors(Rshape=(R_Nx, R_Ny), Qshape=(0, 0))
    except TypeError:
        bv = BraggVectors(Rshape=(R_Nx, R_Ny))
    # py4DSTEM 0.14+ wires up raw/cal getters via set_raw_vectors.
    # Direct attribute assignment leaves the getters stale.
    if hasattr(bv, "set_raw_vectors"):
        bv.set_raw_vectors(pla)
    else:
        bv._v_uncal = pla

    try:
        bv.calibration.set_Q_pixel_size(float(inv_ang_per_pixel))
        bv.calibration.set_Q_pixel_units("A^-1")
    except Exception:
        pass

    # `match_orientations` hard-codes `center=True` when extracting
    # vectors — meaning the calibration MUST have an origin grid even
    # if our peaks are already centered.  We've already subtracted
    # qx0/qy0 in the loop above, so set the calibration origin to
    # all-zeros: subtraction becomes a no-op and `_transform`'s
    # `assert origin is not None` passes.
    try:
        zero_qx = np.zeros((R_Nx, R_Ny), dtype=float)
        zero_qy = np.zeros((R_Nx, R_Ny), dtype=float)
        bv.calibration.set_origin((zero_qx, zero_qy))
    except Exception:
        # 0.13-ish API took (qx0, qy0) as plain floats; tolerate both.
        try:
            bv.calibration.set_qx0(0.0); bv.calibration.set_qy0(0.0)
        except Exception:
            pass

    if hasattr(bv, "setcal"):
        try:
            # center=True because match_orientations hard-codes it;
            # the no-op zero origin above makes that safe.
            bv.setcal(center=True, ellipse=False,
                       pixel=True, rotate=False)
        except Exception:
            pass
    else:
        try:
            bv.calibrate()
        except Exception:
            pass
    return bv


# ---------------------------------------------------------------------------
# End-to-end single-pattern + batch ACOM
# ---------------------------------------------------------------------------

def acom_single_pattern(crystal,
                            pattern: np.ndarray,
                            *,
                            center: Optional[Tuple[float, float]] = None,
                            inv_ang_per_pixel: float = 0.00185,
                            probe_kernel: Optional[np.ndarray] = None,
                            detect_kw: Optional[dict] = None,
                            ):
    """One-pattern ACOM.  Returns dict with::

        peaks        : (N, 3) raw detector-pixel peaks
        bragg_peaks  : BraggVectors object (1×1)
        orientation  : the crystal.match_single_pattern result
        corr         : best correlation score (float)
        fit_pattern  : crystal.generate_diffraction_pattern(...) (for plot)

    The caller decides what to do with the result (overlay on the
    pattern, summarise in a card, etc.).
    """
    detect_kw = dict(detect_kw or {})
    peaks = detect_peaks_2d(pattern, probe_kernel=probe_kernel, **detect_kw)
    H, W = pattern.shape
    if center is None:
        center = (H / 2.0, W / 2.0)
    bv = build_bragg_vectors([peaks], centers=[center],
                                  inv_ang_per_pixel=inv_ang_per_pixel,
                                  Rshape=(1, 1))
    pl = bv.cal[0, 0]
    orient = crystal.match_single_pattern(pl, verbose=False)
    try:
        fit = crystal.generate_diffraction_pattern(
            orient, ind_orientation=0, sigma_excitation_error=0.03)
    except Exception:
        fit = None
    # Best correlation score (top-1).  Different py4DSTEM versions
    # expose this differently; try the common ones.
    corr = float("nan")
    for attr in ("corr", "correlation"):
        v = getattr(orient, attr, None)
        if v is not None:
            try:
                arr = np.asarray(v).ravel()
                if arr.size:
                    corr = float(arr[0])
                    break
            except Exception:
                pass
    return dict(peaks=peaks, bragg_peaks=bv, orientation=orient,
                  corr=corr, fit_pattern=fit, calibrated_pl=pl,
                  center=center)


def acom_full_dataset(crystal,
                          cube,
                          *,
                          inv_ang_per_pixel: float = 0.00185,
                          probe_kernel: Optional[np.ndarray] = None,
                          detect_kw: Optional[dict] = None,
                          center: Optional[Tuple[float, float]] = None,
                          subsample_stride: int = 1,
                          progress_cb=None,
                          ):
    """Per-scan-position ACOM over the entire cube.

    Returns ``(orientation_map, bragg_peaks, (Ny, Nx))`` so the caller
    can render the result as a 2D orientation map (correlation field,
    matrix → ZA RGB, etc.).  The BraggVectors object has Rshape =
    (Ny, Nx), so ``bv.cal[rx, ry]`` works as expected.

    cube : array-like of shape (Ny, Nx, H, W).  An mmap'd numpy array
            from `np.load(.npy, mmap_mode='r')` is fine.
    subsample_stride : downsample the scan grid by this factor (≥1) so
            development runs don't take 25 minutes.  stride=2 → quarter
            the scan positions; the output is still (Ny, Nx) but
            non-sampled positions contain empty PointLists.
    """
    cube = np.asarray(cube) if not hasattr(cube, "shape") else cube
    if cube.ndim != 4:
        raise ValueError(f"cube must be (Ny, Nx, H, W); got {cube.shape}")
    Ny, Nx, H, W = cube.shape
    if center is None:
        center = (H / 2.0, W / 2.0)
    stride = max(int(subsample_stride), 1)
    detect_kw = dict(detect_kw or {})

    # Flat list of N = Ny*Nx peak arrays in (rx, ry)-major order so the
    # subsequent build_bragg_vectors lays them out (rx, ry) correctly.
    peaks_all: List[np.ndarray] = []
    centers_all: List[Tuple[float, float]] = []
    total_pix = Ny * Nx
    done = 0
    for rx in range(Ny):
        for ry in range(Nx):
            done += 1
            if (rx % stride != 0) or (ry % stride != 0):
                peaks_all.append(np.zeros((0, 3), dtype=float))
                centers_all.append(center)
                continue
            try:
                pat = np.asarray(cube[rx, ry], dtype=np.float32)
            except Exception:
                peaks_all.append(np.zeros((0, 3), dtype=float))
                centers_all.append(center)
                continue
            peaks = detect_peaks_2d(pat, probe_kernel=probe_kernel,
                                       **detect_kw)
            peaks_all.append(peaks)
            centers_all.append(center)
            if progress_cb is not None and (done % 64 == 0):
                try: progress_cb(done, total_pix, "detect")
                except Exception: pass

    if progress_cb is not None:
        try: progress_cb(total_pix, total_pix, "build_vectors")
        except Exception: pass
    bv = build_bragg_vectors(peaks_all, centers=centers_all,
                                  inv_ang_per_pixel=inv_ang_per_pixel,
                                  Rshape=(Ny, Nx))
    if progress_cb is not None:
        try: progress_cb(total_pix, total_pix, "match")
        except Exception: pass
    omap = crystal.match_orientations(bv)
    return omap, bv, (Ny, Nx)


def acom_multiphase_full_dataset(crystals_by_name,
                                       cube,
                                       *,
                                       inv_ang_per_pixel: float = 0.00185,
                                       probe_kernel: Optional[np.ndarray] = None,
                                       detect_kw: Optional[dict] = None,
                                       center: Optional[Tuple[float, float]] = None,
                                       subsample_stride: int = 1,
                                       progress_cb=None,
                                       threshold: float = 0.0,
                                       margin: float = 0.0,
                                       ):
    """Multi-phase ACOM over the full cube.

    crystals_by_name : ``dict[str, Crystal]`` — each entry is a
                  pre-built + planned py4DSTEM Crystal (call
                  :func:`prepare_crystal` first).  Example::

                      {"alpha": cr_alpha, "gamma": cr_gamma}

    threshold : if the winning phase's correlation is below this,
                  the pixel is labelled "neither" (phase_id = -1).
    margin    : if the winning phase's correlation is within `margin`
                  of the runner-up phase's, the pixel is labelled
                  "ambiguous" (phase_id = -2).  Set to 0 to disable.

    Returns dict with::

        phase_names      : list[str]  in input order
        phase_id         : (Ny, Nx) int   (-1 = neither, -2 = ambiguous)
        corr_per_phase   : (N, Ny, Nx) float
        rmat_per_phase   : (N, Ny, Nx, 3, 3) float (best ZA per phase)
        winning_corr     : (Ny, Nx) float
        winning_rmat     : (Ny, Nx, 3, 3) float (NaN where phase < 0)
        bragg_peaks      : the shared BraggVectors (post-calibration)
        scan_shape       : (Ny, Nx)
    """
    names = list(crystals_by_name.keys())
    if not names:
        raise ValueError("crystals_by_name is empty.")
    cube = np.asarray(cube) if not hasattr(cube, "shape") else cube
    if cube.ndim != 4:
        raise ValueError(f"cube must be (Ny, Nx, H, W); got {cube.shape}")
    Ny, Nx, H, W = cube.shape
    if center is None:
        center = (H / 2.0, W / 2.0)
    stride = max(int(subsample_stride), 1)
    detect_kw = dict(detect_kw or {})

    # Detection pass — shared across all phases.
    peaks_all: List[np.ndarray] = []
    centers_all: List[Tuple[float, float]] = []
    total_pix = Ny * Nx
    done = 0
    for rx in range(Ny):
        for ry in range(Nx):
            done += 1
            if (rx % stride != 0) or (ry % stride != 0):
                peaks_all.append(np.zeros((0, 3), dtype=float))
                centers_all.append(center)
                continue
            try:
                pat = np.asarray(cube[rx, ry], dtype=np.float32)
            except Exception:
                peaks_all.append(np.zeros((0, 3), dtype=float))
                centers_all.append(center)
                continue
            peaks = detect_peaks_2d(pat, probe_kernel=probe_kernel,
                                       **detect_kw)
            peaks_all.append(peaks)
            centers_all.append(center)
            if progress_cb is not None and (done % 64 == 0):
                try: progress_cb(done, total_pix, "detect")
                except Exception: pass

    bv = build_bragg_vectors(peaks_all, centers=centers_all,
                                  inv_ang_per_pixel=inv_ang_per_pixel,
                                  Rshape=(Ny, Nx))

    # Per-phase match.
    corr_per_phase = np.full((len(names), Ny, Nx), -1.0, dtype=np.float32)
    rmat_per_phase = np.full((len(names), Ny, Nx, 3, 3), np.nan,
                                dtype=np.float32)
    for pi, name in enumerate(names):
        if progress_cb is not None:
            try: progress_cb(pi, len(names), f"match[{name}]")
            except Exception: pass
        cr = crystals_by_name[name]
        omap = cr.match_orientations(bv)
        cv = None
        for attr in ("corr", "correlation"):
            v = getattr(omap, attr, None)
            if v is not None:
                cv = np.asarray(v); break
        if cv is not None:
            try:
                if cv.ndim == 3:
                    corr_per_phase[pi] = cv[..., 0]
                elif cv.ndim == 2:
                    corr_per_phase[pi] = cv
            except Exception:
                pass
        mv = getattr(omap, "matrix", None)
        if mv is not None:
            mv = np.asarray(mv)
            try:
                # (Ny, Nx, n_matches, 3, 3) → take top-1
                if mv.ndim == 5:
                    rmat_per_phase[pi] = mv[..., 0, :, :]
                elif mv.ndim == 4:
                    rmat_per_phase[pi] = mv
            except Exception:
                pass

    # Combine: winning phase per pixel.
    # corr_per_phase has shape (N, Ny, Nx); argmax over axis 0.
    sorted_idx = np.argsort(-corr_per_phase, axis=0)
    top1_pi = sorted_idx[0]                            # (Ny, Nx)
    top1_corr = np.take_along_axis(corr_per_phase,
                                          top1_pi[None], axis=0)[0]
    if len(names) >= 2:
        top2_pi = sorted_idx[1]
        top2_corr = np.take_along_axis(corr_per_phase,
                                              top2_pi[None], axis=0)[0]
    else:
        top2_corr = np.full_like(top1_corr, -1.0)

    phase_id = top1_pi.astype(np.int32)
    # Below threshold → neither.
    if threshold and threshold > 0:
        phase_id[top1_corr < float(threshold)] = -1
    # Too close to runner-up → ambiguous.
    if margin and margin > 0:
        phase_id[(top1_corr - top2_corr) < float(margin)] = -2

    winning_rmat = np.full((Ny, Nx, 3, 3), np.nan, dtype=np.float32)
    for pi in range(len(names)):
        m = (phase_id == pi)
        if m.any():
            winning_rmat[m] = rmat_per_phase[pi][m]

    return dict(
        phase_names=names,
        phase_id=phase_id,
        corr_per_phase=corr_per_phase,
        rmat_per_phase=rmat_per_phase,
        winning_corr=top1_corr,
        winning_rmat=winning_rmat,
        bragg_peaks=bv,
        scan_shape=(Ny, Nx),
    )


def _match_safe(crystal, bv, N):
    """Run match_orientations, returning (corr[N], rmat[N,3,3]).

    py4DSTEM's vectorised match crashes ('argmin of empty sequence')
    when a pattern has too few peaks to match.  We try the fast
    vectorised path first; on ANY exception we fall back to a
    per-pattern loop with try/except so one degenerate pattern (an
    amorphous class with ~0 Bragg peaks) doesn't kill the whole
    batch — those patterns get corr=-1, rmat=NaN.
    """
    corr = np.full(N, -1.0, dtype=np.float32)
    rmat = np.full((N, 3, 3), np.nan, dtype=np.float32)
    try:
        omap = crystal.match_orientations(bv, progress_bar=False)
        cv = np.asarray(getattr(omap, "corr", None))
        mv = np.asarray(getattr(omap, "matrix", None))
        if cv is not None and cv.size:
            corr = (cv[..., 0].ravel() if cv.ndim == 3
                      else cv.ravel())[:N]
        if mv is not None and mv.size:
            rmat = (mv[..., 0, :, :].reshape(N, 3, 3) if mv.ndim == 5
                      else mv.reshape(N, 3, 3))
        return corr.astype(np.float32), rmat.astype(np.float32)
    except Exception as e:
        print(f"[acom_core] vectorised match failed ({e!r}); "
                f"falling back to per-pattern.", flush=True)
    # Per-pattern fallback.
    for k in range(N):
        try:
            pl = bv.cal[k, 0]
            if len(pl.data) < 3:        # too few peaks to match
                continue
            orient = crystal.match_single_pattern(pl, verbose=False)
            c = np.asarray(getattr(orient, "corr", [np.nan])).ravel()
            m = np.asarray(getattr(orient, "matrix", None))
            corr[k] = float(c[0]) if c.size else -1.0
            if m is not None and m.size:
                rmat[k] = (m[0] if m.ndim == 3 else m)
        except Exception:
            continue
    return corr, rmat


def acom_multiphase_batch(crystals_by_name,
                              patterns: Sequence[np.ndarray],
                              *,
                              centers: Optional[Sequence[Tuple[float, float]]] = None,
                              inv_ang_per_pixel: float = 0.00185,
                              probe_kernel: Optional[np.ndarray] = None,
                              detect_kw: Optional[dict] = None,
                              threshold: float = 0.0,
                              margin: float = 0.0,
                              progress_cb=None,
                              ):
    """Multi-phase ACOM on a list of independent patterns (class avgs,
    grain avgs).  Same return contract as the full-dataset version but
    with scan_shape = (N, 1) and the maps reshaped to (N,)."""
    names = list(crystals_by_name.keys())
    if not names:
        raise ValueError("crystals_by_name is empty.")
    detect_kw = dict(detect_kw or {})
    N = len(patterns)
    if centers is None:
        centers = [None] * N
    peaks_all: List[np.ndarray] = []
    centers_all: List[Tuple[float, float]] = []
    for k, pat in enumerate(patterns):
        H, W = pat.shape
        c = (centers[k] if centers[k] is not None
              else (H / 2.0, W / 2.0))
        peaks = detect_peaks_2d(pat, probe_kernel=probe_kernel,
                                   **detect_kw)
        peaks_all.append(peaks); centers_all.append(c)
        if progress_cb is not None:
            try: progress_cb(k + 1, N, "detect")
            except Exception: pass
    if progress_cb is not None:
        try: progress_cb(N, N, "match")
        except Exception: pass
    bv = build_bragg_vectors(peaks_all, centers=centers_all,
                                  inv_ang_per_pixel=inv_ang_per_pixel,
                                  Rshape=(N, 1))
    corr_per_phase = np.full((len(names), N), -1.0, dtype=np.float32)
    rmat_per_phase = np.full((len(names), N, 3, 3), np.nan,
                                dtype=np.float32)
    for pi, name in enumerate(names):
        cr = crystals_by_name[name]
        cvec, mvec = _match_safe(cr, bv, N)
        corr_per_phase[pi] = cvec
        rmat_per_phase[pi] = mvec

    sorted_idx = np.argsort(-corr_per_phase, axis=0)
    top1_pi = sorted_idx[0]
    top1_corr = np.take_along_axis(corr_per_phase,
                                          top1_pi[None], axis=0)[0]
    if len(names) >= 2:
        top2_pi = sorted_idx[1]
        top2_corr = np.take_along_axis(corr_per_phase,
                                              top2_pi[None], axis=0)[0]
    else:
        top2_corr = np.full_like(top1_corr, -1.0)
    phase_id = top1_pi.astype(np.int32)
    if threshold and threshold > 0:
        phase_id[top1_corr < float(threshold)] = -1
    if margin and margin > 0:
        phase_id[(top1_corr - top2_corr) < float(margin)] = -2

    return dict(
        phase_names=names,
        phase_id=phase_id,                # (N,) int
        corr_per_phase=corr_per_phase,    # (P, N) float
        rmat_per_phase=rmat_per_phase,    # (P, N, 3, 3)
        winning_corr=top1_corr,           # (N,)
        bragg_peaks=bv,
    )


def acom_batch(crystal,
                  patterns: Sequence[np.ndarray],
                  *,
                  centers: Optional[Sequence[Tuple[float, float]]] = None,
                  inv_ang_per_pixel: float = 0.00185,
                  probe_kernel: Optional[np.ndarray] = None,
                  detect_kw: Optional[dict] = None,
                  progress_cb=None,
                  ):
    """Batch ACOM over N independent patterns (e.g. K class avgs or N
    grain avgs).  Builds an (N, 1) BraggVectors, calls
    `match_orientations` once, then unpacks per-pattern results.

    Returns a list of N dicts mirroring `acom_single_pattern`'s output
    (peaks, corr, orientation_index_0, ...).
    """
    detect_kw = dict(detect_kw or {})
    peaks_all: List[np.ndarray] = []
    centers_all: List[Tuple[float, float]] = []
    for k, pat in enumerate(patterns):
        if progress_cb is not None:
            try: progress_cb(k, len(patterns), "detect")
            except Exception: pass
        H, W = pat.shape
        c = (centers[k] if (centers is not None and centers[k] is not None)
              else (H / 2.0, W / 2.0))
        peaks = detect_peaks_2d(pat, probe_kernel=probe_kernel, **detect_kw)
        peaks_all.append(peaks)
        centers_all.append(c)

    bv = build_bragg_vectors(peaks_all, centers=centers_all,
                                  inv_ang_per_pixel=inv_ang_per_pixel,
                                  Rshape=(len(patterns), 1))
    if progress_cb is not None:
        try: progress_cb(len(patterns), len(patterns), "match")
        except Exception: pass
    # Crash-safe match (per-pattern fallback for degenerate patterns).
    N = len(patterns)
    corr_vec, rmat_vec = _match_safe(crystal, bv, N)
    omap = None  # vectorised omap not retained in the safe path

    results = []
    for k in range(N):
        results.append(dict(
            peaks=peaks_all[k],
            center=centers_all[k],
            corr=float(corr_vec[k]),
            rotation_matrix=(None
                               if not np.isfinite(rmat_vec[k]).any()
                               else rmat_vec[k]),
            bragg_peaks=bv,
            orientation_map=omap,
            slot=k,
        ))
    return results, omap, bv


# ---------------------------------------------------------------------------
# Convenience: pull a zone-axis hkl out of a rotation matrix
# ---------------------------------------------------------------------------

def zone_axis_from_matrix(rmat: np.ndarray,
                              max_index: int = 4
                              ) -> Tuple[Tuple[int, int, int], float]:
    """Return the integer zone axis [u v w] closest to the third column
    of `rmat` (the beam direction in crystal frame), plus the
    misorientation (deg) between the integer ZA and the continuous one.
    """
    if rmat is None:
        return ((0, 0, 0), float("nan"))
    rmat = np.asarray(rmat, dtype=float)
    # Guard: NaN matrix (pattern had too few peaks → no match) or
    # wrong shape → return null ZA instead of crashing downstream.
    if rmat.size < 3 or not np.isfinite(rmat).any():
        return ((0, 0, 0), float("nan"))
    # py4DSTEM stores R such that the third column is the beam direction
    # in the crystal frame.
    v = rmat[:, 2] if rmat.shape == (3, 3) else rmat.ravel()[:3]
    nrm = np.linalg.norm(v)
    if not np.isfinite(nrm) or nrm < 1e-9:
        return ((0, 0, 0), float("nan"))
    v = v / (nrm + 1e-12)
    # Search small-integer (u, v, w) up to max_index.  When two
    # candidates tie on alignment (e.g. (0,0,1) vs (0,0,2)), prefer the
    # one with smaller |u|+|v|+|w| so we return canonical Miller indices.
    best, best_cos, best_norm = (1, 0, 0), -1.0, 999
    rng = range(-max_index, max_index + 1)
    for u in rng:
        for vv in rng:
            for w in rng:
                if u == 0 and vv == 0 and w == 0:
                    continue
                n = np.array([u, vv, w], dtype=float)
                n /= np.linalg.norm(n)
                c = float(abs(n @ v))
                norm = abs(u) + abs(vv) + abs(w)
                if (c > best_cos + 1e-6
                        or (abs(c - best_cos) < 1e-6
                              and norm < best_norm)):
                    best_cos = c
                    best = (u, vv, w)
                    best_norm = norm
    # Canonicalise sign: first nonzero component positive.
    z = list(best)
    for i in range(3):
        if z[i] != 0:
            if z[i] < 0:
                z = [-x for x in z]
            break
    best = tuple(z)
    mis_deg = float(np.degrees(np.arccos(np.clip(best_cos, -1.0, 1.0))))
    return (best, mis_deg)
