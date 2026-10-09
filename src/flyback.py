"""flyback.py -- find the scan raster (and the flyback frames) in a stream.

Vendored from the cellulose low-dose 4D-STEM work
(``celldenoise/preprocess.py::find_raster``), which is the reference
implementation; the comments below keep the reasons its earlier versions were
wrong, because each one is a trap worth not re-entering.

Why this is needed
------------------
A continuously-streamed scan keeps recording while the beam flies back to the
start of the next row.  Those frames are not probe positions.  Counted as if
they were, the map shears and -- worse -- the descanned beam lays a bright
streak across the detector that reads as a spurious reflection, which in that
study was the strongest apparent reflection in the dataset.

How it is found
---------------
The descan makes this unambiguous and specimen-independent: within a row the
pattern centre ramps steadily along the fast axis, and the flyback returns it
in one or two frames with a large step of the opposite sign.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def frame_centres(get_frame: Callable[[int], np.ndarray], n: int,
                  progress: Optional[Callable[[int, int], None]] = None,
                  cancel: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """Per-frame intensity centroid along the detector x axis.

    A centroid is a good enough stand-in for a fitted centre here: we only
    need the *ramp and its jumps*, not sub-pixel accuracy.
    """
    out = np.full(int(n), np.nan, dtype=np.float64)
    every = max(1, int(n) // 100)
    for i in range(int(n)):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        a = np.asarray(get_frame(i), dtype=np.float32)
        if a.ndim == 3:
            a = a.reshape(-1, a.shape[-1])
        col = a.sum(axis=0)                      # collapse the slow axis
        tot = float(col.sum())
        if tot > 0:
            out[i] = float((col * np.arange(col.size)).sum() / tot)
        if progress is not None and ((i + 1) % every == 0 or i + 1 == n):
            progress(i + 1, int(n))
    return out


def find_raster(centre_x, period: Optional[int] = None,
                n_flyback: Optional[int] = None,
                jump_px: float = 10.0) -> dict:
    """Locate the row period and the flyback block from the descan.

    Returns ``period``, ``n_scan``, ``n_flyback``, ``fly_phase``,
    ``f0_phase``, ``keep`` and ``confidence`` (the share of detected jumps
    falling inside the flyback block -- below ~0.5 the raster was not found
    cleanly and the answer should not be trusted).
    """
    x = np.asarray(centre_x, dtype=np.float64)
    n = int(np.sum(np.isfinite(x)))
    if n < 50:
        raise ValueError("too few usable frame centres to find a raster")
    x = x[:n]
    x = np.where(np.isfinite(x), x, np.nanmedian(x))

    d = np.diff(x, prepend=x[0])
    ramp = float(np.median(d))
    jumps = np.flatnonzero(d * np.sign(ramp) < -abs(jump_px))
    if len(jumps) < 4:
        raise ValueError(
            f"only {len(jumps)} flyback jumps found; the raster could not be "
            "located from the descan. If the descan is compensated the beam "
            "centre does not ramp, and this method cannot work -- enter the "
            "grid by hand.")

    if period is None:
        # Autocorrelation alone picks harmonics (it chose 20 instead of 57 on
        # scan_01), so score each candidate by how tightly the jumps cluster
        # in phase -- only the true period can make them line up.
        best_p, best_s = None, -1.0
        for pp in range(25, min(160, max(26, len(x) // 8))):
            c = np.bincount(jumps % pp, minlength=pp)
            w = max(sum(c[(k + j) % pp] for j in range(3)) for k in range(pp))
            sc = (w / len(jumps)) - 3.0 / pp        # penalise short periods
            if sc > best_s:
                best_p, best_s = pp, sc
        period = int(best_p)

    cnt = np.bincount(jumps % period, minlength=period)

    if n_flyback is None:
        # Preferred estimate: fold every row onto one period and measure the
        # ramp itself.  The scan part of a row is the longest stretch where
        # the folded median centre climbs steadily; whatever is left is the
        # flyback plus its settling.  This beats counting jump phases, which
        # only sees the single frame where the beam snaps back and therefore
        # under-reports the block whenever the settling frames sit still.
        rows = n // period
        if rows >= 4:
            fold = np.nanmedian(
                x[:rows * period].reshape(rows, period), axis=0)
            step = np.diff(fold, append=fold[0])
            fwd = step * np.sign(ramp) > 0          # moving along the ramp
            best_len, best_end = 0, 0
            run = 0
            for k in range(2 * period):             # wrap once
                if fwd[k % period]:
                    run += 1
                    if run > best_len:
                        best_len, best_end = run, k
                else:
                    run = 0
            # a run of `best_len` forward STEPS spans best_len+1 frames
            n_scan_est = int(min(best_len + 1, period - 1))
            if n_scan_est >= 2:
                n_flyback = period - n_scan_est

    if n_flyback is None:
        # Fallback: the jump-phase cluster.  "smallest window holding 90% of
        # jumps" is too greedy -- stray jumps at random phases drag the width
        # from a true 7 up to 19.  Require each phase to carry a real share
        # of the peak instead.
        peak = int(cnt.max())
        hot = cnt >= max(0.25 * peak, 2)
        k0 = int(np.argmax(cnt))
        lo = hi = k0
        while hot[(lo - 1) % period] and (k0 - lo) < period // 2:
            lo -= 1
        while hot[(hi + 1) % period] and (hi - k0) < period // 2:
            hi += 1
        n_flyback = (hi - lo + 1) + 2            # the descan needs to settle
    n_flyback = int(min(max(int(n_flyback), 1), period - 1))

    start, best = 0, -1
    for k in range(period):
        w = sum(cnt[(k + j) % period] for j in range(n_flyback))
        if w > best:
            best, start = w, k
    # Report the ABSOLUTE phase. Returning only an f0 and assuming "flyback is
    # the last n_flyback phases" silently breaks whenever something else picks
    # a different f0 -- that mistake marked phases 48-56 as flyback when the
    # jumps were at 39-46, keeping the real flyback and discarding good frames.
    fly_phase = int(start % period)
    f0_phase = int((start + n_flyback) % period)
    idx = np.arange(n)
    keep = ((idx - f0_phase) % period) < (period - n_flyback)
    return dict(period=int(period), n_scan=int(period - n_flyback),
                n_flyback=int(n_flyback), fly_phase=fly_phase,
                f0_phase=f0_phase, keep=keep,
                confidence=float(best / len(jumps)), ramp_px=ramp)


def suggest_grid(n_frames: int, period: int, n_flyback: int) -> dict:
    """Turn a detected raster into a scan grid the loader can use."""
    period = int(period)
    n_scan = int(period - n_flyback)
    rows = int(n_frames // period)
    return dict(Ny=rows, Nx=n_scan, period=period,
                used=rows * period, dropped_flyback=rows * int(n_flyback),
                dropped_tail=int(n_frames) - rows * period)
