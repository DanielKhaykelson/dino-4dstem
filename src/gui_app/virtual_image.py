"""Virtual bright-field / annular dark-field images of a 4D-STEM cube.

A real-space map that exists for every dataset (no NMF / DINO needed), for
picking scan positions and regions.  Detectors are centred on the direct
beam (found from the mean of up to 64 patterns): BF = the beam disk,
ADF = an annulus outside it (same geometry as the Viewing tab)."""
import numpy as np


def _radial_mask(H, W, r0, r1, centre):
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - centre[0], xx - centre[1])
    return (r >= r0) & (r < r1)


def bf_adf(cube, progress=None, stop=None):
    """cube: (R, C, H, W) array-like (lazy is fine) -> (BF, ADF) float32
    (R, C) images.  progress(rows_done, R) is called every 16 rows; stop()
    returning True aborts (returns None)."""
    from denoise_n2n import beam_centre
    R, C, H, W = cube.shape
    idx = np.linspace(0, R * C - 1, min(64, R * C)).astype(int)
    acc = np.zeros((H, W), np.float64)
    for i in idx:
        acc += np.asarray(cube[i // C, i % C], np.float64)
    centre = beam_centre(acc / len(idx))
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(yy - centre[0], xx - centre[1]).astype(int)
    prof = (np.bincount(rr.ravel(), acc.ravel())
            / np.maximum(np.bincount(rr.ravel()), 1))
    lp = np.log(np.maximum(prof[:max(6, min(H, W) // 4)], 1e-12))
    r_edge = int(np.argmin(np.diff(lp))) + 1
    r_bf = max(0.06 * H, 1.2 * r_edge)
    bm = _radial_mask(H, W, 0.0, r_bf, centre).ravel()
    am = _radial_mask(H, W, max(0.18 * H, 2 * r_bf), 0.45 * H, centre).ravel()
    BF = np.zeros((R, C), np.float32)
    ADF = np.zeros((R, C), np.float32)
    for y in range(R):
        if stop is not None and stop():
            return None
        flat = np.asarray(cube[y]).astype(np.float32).reshape(C, H * W)
        BF[y] = flat[:, bm].sum(1)
        ADF[y] = flat[:, am].sum(1)
        if progress is not None and (y % 16 == 0 or y == R - 1):
            progress(y + 1, R)
    return BF, ADF
