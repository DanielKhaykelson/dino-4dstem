"""Calibration utilities — shared scale-bar + bin-to-q helpers.

The user's `recip_res` is in nm⁻¹/raw-detector-px. Display scaling
depends on which pipeline the image went through:

    Pattern popup (right-click), grain-average popup, raw frames:
        no resize, 1 disp px = 1 raw-detector-px
        → q_per_disp_px = recip_res

    Triptych (avg/cam/IG), class-avg grid, polar→cart class avgs:
        F.interpolate(raw → 192) → CenterCrop(140) → Resize(192)
        → 1 disp px = (140/192)·(raw_size/192) raw-px
        → q_per_disp_px = recip_res · 140/192 · raw_size/192

Polar 1D radial bin x-axis uses the same factor as the cart triptych.

`real_res` is nm/scan-pixel (the scan grid spacing) — independent of
the diffraction pipeline.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


CENTER_CROP = 140
POLAR_SIZE = 192


def get_raw_detector_size(sample_key: str) -> int | None:
    """Detector size (last axis of cube) for a sample, or None if it can't
    be determined cheaply (no full .prz/.npz read)."""
    try:
        from data import SAMPLES
        cfg = SAMPLES.get(sample_key)
        if not cfg:
            return None
        path = cfg.get("path")
        if not path:
            return None
        base, _ = os.path.splitext(path)
        sidecar = base + ".cube.npy"
        if os.path.exists(sidecar):
            arr = np.load(sidecar, mmap_mode="r", allow_pickle=True)
            return int(arr.shape[-1])
        if path.endswith(".npy"):
            arr = np.load(path, mmap_mode="r", allow_pickle=True)
            return int(arr.shape[-1])
        return None
    except Exception:
        return None


def q_per_polar_bin(recip_per_raw_px: float,
                     raw_detector_size: int | None,
                     center_crop: int = CENTER_CROP,
                     polar_size: int = POLAR_SIZE) -> float:
    """nm⁻¹ per polar bin (or per cart-cropped 192-display pixel)."""
    if not recip_per_raw_px or recip_per_raw_px <= 0:
        return 0.0
    rs = raw_detector_size if raw_detector_size else polar_size
    return float(recip_per_raw_px * (center_crop / polar_size)
                  * (rs / polar_size))


def add_recip_scalebar(ax, q_per_disp_px: float, length_q: float = 0.2,
                        color: str = "white", loc: str = "lower right") -> bool:
    """Add a `length_q` nm⁻¹ scale bar. Returns True if added."""
    if not q_per_disp_px or q_per_disp_px <= 0:
        return False
    pix_len = length_q / q_per_disp_px
    fp = fm.FontProperties(size=9, weight="bold")
    bar = AnchoredSizeBar(ax.transData, pix_len,
                            f"{length_q:g} nm⁻¹",
                            loc, pad=0.4, color=color,
                            frameon=False, size_vertical=2,
                            fontproperties=fp)
    ax.add_artist(bar)
    return True


def add_real_scalebar(ax, nm_per_scan_px: float, length_nm: float = 100,
                       color: str = "white",
                       loc: str = "lower right") -> bool:
    """Add a `length_nm` nm scale bar to a real-space (scan-grid) image."""
    if not nm_per_scan_px or nm_per_scan_px <= 0:
        return False
    pix_len = length_nm / nm_per_scan_px
    fp = fm.FontProperties(size=9, weight="bold")
    bar = AnchoredSizeBar(ax.transData, pix_len,
                            f"{length_nm:g} nm",
                            loc, pad=0.4, color=color,
                            frameon=False, size_vertical=2,
                            fontproperties=fp)
    ax.add_artist(bar)
    return True


def set_q_axis(ax, n_bins: int, q_per_bin: float, axis: str = "x"):
    """Re-tick a 1D axis whose data coordinate is bin-index, so labels
    show q in nm⁻¹ (when q_per_bin > 0). No-op otherwise."""
    if q_per_bin <= 0:
        return False
    # Pick ~6 nice ticks across the data range.
    q_max = (n_bins - 1) * q_per_bin
    # Round q-step to a 1/2/5×10^k value for cleanliness.
    if q_max <= 0:
        return False
    raw_step = q_max / 6
    exp = np.floor(np.log10(raw_step))
    base = 10 ** exp
    for mult in (1, 2, 5, 10):
        if mult * base >= raw_step:
            step_q = mult * base
            break
    qs = np.arange(0.0, q_max + 1e-9, step_q)
    bins = qs / q_per_bin
    if axis == "x":
        ax.set_xticks(bins)
        ax.set_xticklabels([f"{q:g}" for q in qs])
        ax.set_xlabel("q  (nm⁻¹)")
    else:
        ax.set_yticks(bins)
        ax.set_yticklabels([f"{q:g}" for q in qs])
        ax.set_ylabel("q  (nm⁻¹)")
    return True
