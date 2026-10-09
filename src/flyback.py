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
        # The jump-phase cluster.  "smallest window holding 90% of jumps" is
        # too greedy -- stray jumps at random phases drag the width from a
        # true 7 up to 19.  Require each phase to carry a real share of the
        # peak instead.  (A fold-the-ramp estimator was tried here and is
        # worse on real data: it reported 52+5 on all six beamDamage scans
        # against a recorded 48+9 / 49+8, because real settling frames do
        # carry jumps and this cluster is what captures them.)
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
                confidence=float(best / len(jumps)), ramp_px=ramp,
                n_jumps=int(len(jumps)))


def _bin2(a: np.ndarray, b: int) -> np.ndarray:
    if b <= 1:
        return a
    H, W = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:H, :W].reshape(H // b, b, W // b, b).sum(axis=(1, 3))


def _disc_centre(p: np.ndarray, r: float) -> tuple:
    """py4DSTEM-style origin: smooth by the probe radius, take the argmax,
    then the centre of mass inside 1.2 r of it.  A plain argmax lands on a
    hot pixel or a reflection; a whole-pattern CoM is dragged by diffuse
    scattering (cellulose find_centre)."""
    from scipy.ndimage import gaussian_filter
    a = np.asarray(p, dtype=np.float64)
    H, W = a.shape
    qy, qx = np.unravel_index(
        np.argmax(gaussian_filter(a, max(r, 1.0), mode="nearest")), (H, W))
    y0, y1 = max(0, int(qy - 1.2 * r) - 1), min(H, int(qy + 1.2 * r) + 2)
    x0, x1 = max(0, int(qx - 1.2 * r) - 1), min(W, int(qx + 1.2 * r) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    w = a[y0:y1, x0:x1] * (np.hypot(yy - qy, xx - qx) < 1.2 * r)
    w = np.maximum(w, 0)
    s = w.sum()
    if s <= 0:
        return float(qy), float(qx)
    return float((w * yy).sum() / s), float((w * xx).sum() / s)


def probe_radius(mean_pattern: np.ndarray, cy: float, cx: float) -> int:
    """Half-maximum radius of the direct beam from the radial profile of a
    large average (stable, unlike per-frame estimates)."""
    m = np.asarray(mean_pattern, dtype=np.float64)
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    rmax = int(np.hypot(*m.shape)) + 1
    ri = np.clip(np.hypot(yy - cy, xx - cx).astype(int), 0, rmax)
    prof = (np.bincount(ri.ravel(), m.ravel(), rmax + 1)
            / np.maximum(np.bincount(ri.ravel(), minlength=rmax + 1), 1))
    core = prof[:4].mean()
    return max(2, int(np.argmax(prof < core / 2)))


def frame_signals(get_frame: Callable[[int], np.ndarray], n: int,
                  binning: int = 2,
                  progress: Optional[Callable[[int, int], None]] = None,
                  cancel: Optional[Callable[[], bool]] = None) -> dict:
    """One pass over the series: per-frame disc centre (x, y) in FULL-
    resolution detector pixels, bright-field counts and total counts.

    Frames are binned for speed; centres are converted back so the 10-px
    flyback-jump threshold keeps its meaning.
    """
    n = int(n)
    b = max(1, int(binning))
    step = max(1, n // 200)
    acc = None
    for i in range(0, n, step):
        f = _bin2(np.asarray(get_frame(i), dtype=np.float64), b)
        acc = f if acc is None else acc + f
    mean = acc / len(range(0, n, step))
    cy0, cx0 = _disc_centre(mean, 20.0 / b)
    rb = probe_radius(mean, cy0, cx0)
    H, W = mean.shape
    yy, xx = np.mgrid[0:H, 0:W]
    bfmask = np.hypot(yy - cy0, xx - cx0) < (rb + 2.0 / b)
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    bf = np.zeros(n)
    tot = np.zeros(n)
    every = max(1, n // 100)
    for i in range(n):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        f = _bin2(np.asarray(get_frame(i), dtype=np.float64), b)
        py, px = _disc_centre(f, rb)
        cy[i] = py * b + (b - 1) / 2.0
        cx[i] = px * b + (b - 1) / 2.0
        bf[i] = f[bfmask].sum()
        tot[i] = f.sum()
        if progress is not None and ((i + 1) % every == 0 or i + 1 == n):
            progress(i + 1, n)
    return dict(cx=cx, cy=cy, bf=bf, tot=tot, probe_r=rb * b)


def find_scan_start(signal, period: int, n_scan: int, n_rows: int,
                    phase: Optional[int] = None) -> dict:
    """First frame of the first row, by image coherence (cellulose
    find_scan_start).  The descan repeats every row and cannot tell a lead-in
    row from a real one; a correctly registered image has correlated
    neighbouring rows and a mis-registered one does not.  Only starts
    consistent with the raster phase are legal -- searching freely lets the
    best start land where flyback is inside the kept frames."""
    s = np.asarray(signal, dtype=np.float64)
    n = len(s)
    span = period * n_rows
    if span > n:
        raise ValueError(f"{n} frames cannot hold {n_rows} rows of "
                         f"{period} frames")
    hi = n - span + 1
    scores = np.full(hi, -np.inf)
    cands = (range(hi) if phase is None
             else range(int(phase) % period, hi, period))
    for f0 in cands:
        g = s[f0:f0 + span].reshape(n_rows, period)[:, :n_scan]
        if not np.all(np.isfinite(g)):
            continue
        c = [np.corrcoef(g[i], g[i + 1])[0, 1] for i in range(n_rows - 1)]
        c = [v for v in c if np.isfinite(v)]
        if c:
            scores[f0] = float(np.mean(c))
    f0 = int(np.argmax(scores))
    good = scores[np.isfinite(scores)]
    margin = float(scores[f0] - np.median(good)) if good.size else 0.0
    return dict(f0=f0, row_corr=float(scores[f0]), margin=margin)


def row_quality(centre_x, f0: int, period: int, n_scan: int,
                n_rows: int, rms_px: float = 2.0) -> np.ndarray:
    """Per-row slope check: fit a line to the descan over each row's scan
    frames; a row whose residual RMS exceeds rms_px does not ramp cleanly
    (typically the row straddling the lead-in) and is rejected."""
    x = np.asarray(centre_x, dtype=np.float64)
    t = np.arange(n_scan)
    ok = np.zeros(n_rows, dtype=bool)
    for r in range(n_rows):
        seg = x[f0 + r * period: f0 + r * period + n_scan]
        if len(seg) < n_scan or not np.all(np.isfinite(seg)):
            continue
        a = np.polyfit(t, seg, 1)
        ok[r] = float((seg - np.polyval(a, t)).std()) < rms_px
    return ok


def detect_raster(sig: dict, n_rows: Optional[int] = None) -> dict:
    """Full cellulose recipe on the signals from frame_signals.

    raster (period / flyback / phase) -> scan start (coherence) -> per-row
    slope check -> explicit frame_index (rows x n_scan).  If n_rows is not
    known, every row that fits is examined and the longest run of rows that
    pass the slope check is kept.
    """
    cx, bf, tot = sig["cx"], sig["bf"], sig["tot"]
    N = len(cx)
    ras = find_raster(cx)
    P, NS = ras["period"], ras["n_scan"]
    frac = bf / np.maximum(tot, 1)
    rows_fit = (N - ras["f0_phase"]) // P
    want = int(n_rows) if n_rows else rows_fit
    st = find_scan_start(frac, P, NS, want, phase=ras["f0_phase"])
    f0 = st["f0"]
    total = (N - f0) // P
    ok = row_quality(cx, f0, P, NS, total)
    if n_rows:
        # keep n_rows, skipping bad rows at the START (lead-in)
        start = 0
        while start < total and not ok[start]:
            start += 1
        if start + int(n_rows) > total:
            start = max(0, total - int(n_rows))
        rows = np.arange(start, start + int(n_rows))
    else:
        best, cur, best_end = 0, 0, -1
        for r in range(total):
            cur = cur + 1 if ok[r] else 0
            if cur > best:
                best, best_end = cur, r
        rows = np.arange(best_end - best + 1, best_end + 1)
    fi = (f0 + rows[:, None] * P + np.arange(NS)[None, :]).astype(np.int64)
    # a confidence a handful of coincident jumps cannot fake: require about
    # one flyback jump per row before trusting the phase clustering
    n_jumps = int(ras.get("n_jumps", 0))
    enough = n_jumps >= 0.5 * max(1, rows_fit)
    return dict(period=P, n_scan=NS, n_flyback=ras["n_flyback"],
                fly_phase=ras["fly_phase"], f0=int(fi[0, 0]),
                n_rows=int(len(rows)), frame_index=fi,
                raster_conf=float(ras["confidence"]) if enough else 0.0,
                n_jumps=n_jumps, row_corr=st["row_corr"],
                start_margin=st["margin"],
                rows_rejected=int(total - ok[:total].sum()))
