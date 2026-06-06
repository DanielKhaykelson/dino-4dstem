"""crystallinity_panel.py -- per-position 1D crystallinity maps.

Crystallinity is measured in 1D (no 2D peak finding, so weak peaks
boosted by azimuthal integration aren't lost).  For each pattern, within
a user-chosen radial window [r_min, r_max] (1/Å, truncated to start
*after* the central beam/mask):

  * azimuthal MEAN  I(q) -> a SNIP baseline estimates the amorphous
    halo; the area above it ÷ total is the **peak/halo ratio**
    (degree-of-crystallinity, lin-log treated like the SAXS gate).
  * azimuthal VARIANCE around each ring -> **spottiness** (normalized
    by mean²): high for sharp Bragg spots, ~0 for smooth amorphous
    rings.  Catches single-grain/spotty signal the mean dilutes.

Two complementary scalars per position -> two virtual maps (Ny × Nx).
**Run on full dataset** computes both; the map panel toggles between
them.  **Per-cluster** aggregates both per DINO class (median ± IQR).
"""
from __future__ import annotations
import os, sys, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg", force=True)
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                  NavigationToolbar2Tk)
from matplotlib.colors import ListedColormap

from data import SAMPLES, LoadPRZ


def _section_header(parent, text):
    ctk.CTkLabel(parent, text=text,
                  font=("Segoe UI", 12, "bold"),
                  anchor="w").pack(anchor="w", padx=10, pady=(10, 0))
    ctk.CTkFrame(parent, height=2,
                  fg_color=("#cccccc", "#444444")
                  ).pack(fill="x", padx=10, pady=(0, 4))


def _hint(parent, text):
    ctk.CTkLabel(parent, text=text, font=("Segoe UI", 9),
                  text_color=("#666", "#aaa"),
                  wraplength=270, justify="left",
                  anchor="w").pack(anchor="w", padx=10, pady=(0, 2))


# ---------------------------------------------------------------------------
def cart_to_polar(pat: np.ndarray, n_theta: int = 192,
                     n_r: int | None = None,
                     center: tuple | None = None) -> np.ndarray:
    """Quick numpy polar warp (display-only).  Returns (n_theta, n_r)."""
    from scipy.ndimage import map_coordinates
    H, W = pat.shape
    if n_r is None: n_r = min(H, W) // 2
    cy, cx = center if center is not None else (H / 2.0, W / 2.0)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    r = np.linspace(0.0, n_r - 1, n_r)
    rr, tt = np.meshgrid(r, theta, indexing="xy")
    xx = cx + rr * np.cos(tt)
    yy = cy + rr * np.sin(tt)
    out = map_coordinates(pat.astype(np.float32),
                              [yy.ravel(), xx.ravel()],
                              order=1, mode="constant",
                              cval=0.0)
    return out.reshape(n_theta, n_r)


def compute_crystallinity_ratio(pat: np.ndarray,
                                    r_min_px: float, r_max_px: float,
                                    detect_kw: dict,
                                    center: tuple | None = None
                                    ) -> tuple:
    """For a single 2D pattern, return
        (ratio, peaks_in_window, peak_sum, bg_sum, annulus_mask)

    ratio = peak_sum / (bg_sum + ε)
    where:
        peak_sum = sum of pattern intensity AT detected peak pixels
                    in the annulus
        bg_sum   = total intensity in annulus minus peak_sum
    """
    from gui_app.acom_core import detect_peaks_2d
    H, W = pat.shape
    cy, cx = center if center is not None else (H / 2.0, W / 2.0)
    yy, xx = np.indices((H, W))
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    annulus = (r2 >= r_min_px ** 2) & (r2 < r_max_px ** 2)

    peaks = detect_peaks_2d(pat, **detect_kw)
    if peaks.size:
        py = peaks[:, 0]; px = peaks[:, 1]
        rad2 = (py - cy) ** 2 + (px - cx) ** 2
        in_win = (rad2 >= r_min_px ** 2) & (rad2 < r_max_px ** 2)
        peaks_w = peaks[in_win]
        if peaks_w.size:
            pi = np.clip(peaks_w[:, 0].astype(int), 0, H - 1)
            pj = np.clip(peaks_w[:, 1].astype(int), 0, W - 1)
            peak_sum = float(pat[pi, pj].sum())
        else:
            peak_sum = 0.0
    else:
        peaks_w = np.zeros((0, 3))
        peak_sum = 0.0
    total_ann = float(pat[annulus].sum())
    bg_sum = max(total_ann - peak_sum, 1e-6)
    ratio = peak_sum / bg_sum
    return ratio, peaks_w, peak_sum, bg_sum, annulus


# ---------------------------------------------------------------------------
# 1D ("SAXS-like") crystallinity — azimuthal mean + variance per radial bin,
# SNIP halo baseline, continuous peak/halo ratio.  NO 2D peak finding, so
# weak peaks (boosted by azimuthal integration) aren't lost.
# ---------------------------------------------------------------------------
def _radial_mean_var(pat: np.ndarray, center: tuple, beam_px: float = 0.0):
    """Azimuthal mean m(r) and variance v(r) per integer radial bin.

    v(r) is the variance of intensity AROUND the ring at radius r — high
    for spotty (crystalline) rings, ~0 for smooth amorphous rings.

    ``beam_px`` masks the direct beam: bins inside that radius are zeroed
    (mean/var/count → 0) so the (huge) central beam can't dominate the
    radial profile, the SNIP baseline, or the scattered-intensity sum.
    """
    H, W = pat.shape
    cy, cx = center
    yy, xx = np.indices((H, W))
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    n_bins = int(min(H, W) // 2)
    ind = np.clip(rad.astype(int), 0, n_bins - 1).ravel()
    flat = pat.astype(np.float64).ravel()
    cnt = np.bincount(ind, minlength=n_bins)[:n_bins].astype(np.float64)
    s1 = np.bincount(ind, weights=flat, minlength=n_bins)[:n_bins]
    s2 = np.bincount(ind, weights=flat * flat, minlength=n_bins)[:n_bins]
    cntc = np.maximum(cnt, 1.0)
    m = s1 / cntc
    v = np.maximum(s2 / cntc - m * m, 0.0)
    nb = int(max(0, min(round(beam_px), n_bins)))
    if nb > 0:
        m[:nb] = 0.0
        v[:nb] = 0.0
        cnt[:nb] = 0.0
    return m, v, cnt


def _snip_baseline(y: np.ndarray, n_iter: int = 14) -> np.ndarray:
    """SNIP (iterative peak-clipping) baseline on a 1D signal.

    Operate on log-intensity (caller passes log I): each pass with growing
    half-window m replaces y[i] with min(y[i], (y[i-m]+y[i+m])/2), so
    features NARROWER than the window (Bragg peaks) get clipped while the
    BROAD halo survives.  One intuitive knob: max window ≈ widest peak.
    """
    z = np.asarray(y, dtype=np.float64).copy()
    n = z.size
    for m in range(1, int(max(n_iter, 1)) + 1):
        if 2 * m >= n:
            break
        cand = 0.5 * (z[:n - 2 * m] + z[2 * m:])
        np.minimum(z[m:n - m], cand, out=z[m:n - m])
    return z


def compute_crystallinity_1d(pat: np.ndarray, r_lo_px: float, r_hi_px: float,
                              *, center: tuple | None = None,
                              snip_window: int = 14,
                              beam_px: float = 0.0) -> dict | None:
    """1D crystallinity from a single 2D pattern.

    Returns a dict with:
        ratio      : ∫max(I-halo,0) / ∫I over the q-window  (peak/halo)
        var_index  : mean azimuthal variance / mean²  (spottiness)
        m, v       : azimuthal mean / variance over the window
        halo, peak : SNIP halo and the residual peak signal (linear)
        lo, hi, r_px
    """
    H, W = pat.shape
    cy, cx = center if center is not None else (H / 2.0, W / 2.0)
    m_all, v_all, _cnt = _radial_mean_var(pat, (cy, cx), beam_px)
    return _crystallinity_window(m_all, v_all, round(r_lo_px),
                                  round(r_hi_px), snip_window)


def _crystallinity_window(m_all: np.ndarray, v_all: np.ndarray,
                           lo: int, hi: int, snip_window: int = 14):
    """Evaluate peak/halo + azimuthal-variance crystallinity over the
    radial-bin window [lo, hi) from PRECOMPUTED azimuthal mean/variance
    profiles — so Report-all can sweep many windows with ONE radial pass
    per pattern."""
    n = m_all.size
    lo = int(max(0, min(n - 1, lo)))
    hi = int(min(n, hi))
    if hi - lo < 8:
        return None
    m = m_all[lo:hi]
    v = v_all[lo:hi]
    # peak/halo via SNIP baseline in log space (handles the steep,
    # power-law-ish amorphous halo); ratio in LINEAR intensity.
    log_I = np.log(np.clip(m, 1e-6, None))
    base_log = _snip_baseline(log_I, snip_window)
    halo = np.exp(base_log)
    peak = np.clip(m - halo, 0.0, None)
    tot = float(m.sum()) + 1e-12
    ratio = float(peak.sum() / tot)
    # normalized azimuthal variance (dose-robust spottiness)
    var_index = float(np.mean(v / np.maximum(m * m, 1e-12)))
    return dict(ratio=ratio, var_index=var_index, m=m, v=v,
                 halo=halo, peak=peak, lo=lo, hi=hi,
                 r_px=np.arange(lo, hi, dtype=float))


# ---------------------------------------------------------------------------
# Report-all overlays (combine the per-dr maps into one colored image) +
# intrinsic quality scoring.
# ---------------------------------------------------------------------------
def _stack_zscore(maps):
    """(nW, Ny, Nx) z-scored per window so windows are comparable for
    argmax (one window's larger absolute scale shouldn't always win)."""
    st = np.stack(maps, axis=0).astype(np.float32)
    out = np.full_like(st, np.nan)
    for w in range(st.shape[0]):
        a = st[w]
        if np.isfinite(a).sum() < 2:
            continue
        mu = np.nanmean(a); sd = np.nanstd(a)
        out[w] = (a - mu) / (sd if sd > 1e-12 else 1.0)
    return out


def _overlay_argmax_rgb(maps, colors):
    """Dominant-dr map: each pixel coloured by the window with the
    strongest z-scored signal.  Returns (rgb, idx, valid)."""
    z = _stack_zscore(maps)
    valid = np.isfinite(z).any(axis=0)
    zf = np.where(np.isfinite(z), z, -np.inf)
    idx = np.argmax(zf, axis=0)
    Ny, Nx = idx.shape
    rgb = np.zeros((Ny, Nx, 3), np.float32)
    for w, c in enumerate(colors):
        rgb[(idx == w) & valid] = c[:3]
    rgb[~valid] = 0.0
    return rgb, idx, valid


def _overlay_additive_rgb(maps, colors):
    """Additive false-colour: each window a hue, summed weighted by its
    robust-normalized value."""
    Ny, Nx = maps[0].shape
    rgb = np.zeros((Ny, Nx, 3), np.float32)
    for mp, c in zip(maps, colors):
        a = mp.astype(np.float32)
        f = np.isfinite(a)
        if f.sum() == 0:
            continue
        lo = np.nanpercentile(a, 2); hi = np.nanpercentile(a, 98)
        if hi - lo < 1e-12:
            hi = lo + 1.0
        n = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        n[~f] = 0.0
        for k in range(3):
            rgb[..., k] += n * c[k]
    mx = float(rgb.max())
    if mx > 0:
        rgb /= mx
    return rgb


def _spatial_autocorr(mp):
    """Lag-1 (right + down) spatial autocorrelation of a 2D map.  High =
    spatially structured (a real map); ~0 = salt-and-pepper noise."""
    a = np.asarray(mp, dtype=np.float64)
    finite = np.isfinite(a)
    if finite.sum() < 8:
        return 0.0
    mu = np.nanmean(a); sd = np.nanstd(a)
    if sd < 1e-12:
        return 0.0
    z = (a - mu) / sd
    vals = []
    for sh, ax_ in ((1, 0), (1, 1)):
        b = np.roll(z, sh, axis=ax_)
        m = finite & np.roll(finite, sh, axis=ax_)
        if m.sum() > 0:
            vals.append(float(np.nanmean(z[m] * b[m])))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
class CrystallinityPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        # State.
        self._test_pattern = None
        self._test_origin = "(none)"
        self._test_center = None
        self._test_polar = None
        self._test_result = None
        self._cryst_map = None
        self._var_map = None
        self._scatter_map = None
        self._busy = False
        self._build()

    # ------------------------------------------------------------------
    def on_runtime_sample_added(self, key):
        pass

    def _recip_per_px(self):
        try:
            return float(self.app.recip_res.get()) if self.app else 0.0
        except Exception:
            return 0.0

    def _beam_radius_px(self) -> int:
        try:
            return max(int(float(self._beam_px.get())), 0)
        except Exception:
            return 0

    def _ensure_beam_mask(self, sample: str, H: int):
        """Resolve the direct-beam mask radius (detector px) for `sample`.

        Use the dataset's center_mask_radius (scaled from the 192-px model
        space to this raw detector) when known; otherwise pop a dialog
        asking the user.  Runs ONCE per sample (main thread only)."""
        if (self._beam_radius_px() > 0
                and getattr(self, "_beam_sample", None) == sample):
            return
        cfg = SAMPLES.get(sample, {})
        cmr = cfg.get("center_mask_radius")
        if cmr:
            guess = int(round(float(cmr) * (H / 192.0)))
        else:
            guess = max(6, int(H // 24))
        val = None
        if not cmr:
            # No mask recorded for this dataset → ask the user.
            from tkinter import simpledialog
            val = simpledialog.askinteger(
                "Direct-beam mask",
                f"No center mask recorded for this dataset.\n\n"
                f"Direct-beam mask radius in DETECTOR px "
                f"(pattern is {H}px):",
                initialvalue=guess, minvalue=0, parent=self)
        self._beam_px.set(int(val if val is not None else guess))
        self._beam_sample = sample

    def _inv_ang_per_px(self):
        try:
            return float(self._inv_ang.get())
        except Exception:
            return self._recip_per_px() * 0.1 or 0.00185

    def _posthoc(self):
        return getattr(self.app, "posthoc", None)

    # ------------------------------------------------------------------
    def _build(self):
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)
        from gui_app._ui import StatusDot
        self._dot_data = StatusDot(top, label="dataset")
        self._dot_run  = StatusDot(top, label="run")
        for d in (self._dot_data, self._dot_run):
            d.pack(side="left", padx=10)
        sess = getattr(self.app, "session", None)
        if sess is not None:
            sess.subscribe(self._on_session_change)
            self._on_session_change(sess)

        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True,
                   padx=6, pady=4)

        # LEFT: scrollable controls.
        sidebar = ctk.CTkScrollableFrame(body, width=320)
        sidebar.pack(side="left", fill="y")

        # 1. SOURCE picker
        _section_header(sidebar, "1.  Pick a test pattern")
        _hint(sidebar,
              "Tune r-window + detection params on one pattern. "
              "Same params get applied to every scan position when "
              "you click 'Run on full dataset'.")
        self._source_var = ctk.StringVar(value="dp_max")
        for label, val in (
                ("dp_max (per-pixel max over whole scan)", "dp_max"),
                ("dp_mean (mean over whole scan)", "dp_mean"),
                ("class avg #k", "class_avg"),
                ("scan pos (y, x)", "scan_pos")):
            ctk.CTkRadioButton(sidebar, text=label,
                                variable=self._source_var,
                                value=val).pack(anchor="w", padx=10, pady=1)
        cls_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        cls_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(cls_row, text="class:", width=44,
                       anchor="w").pack(side="left")
        self._src_class = ctk.StringVar(value="0")
        ctk.CTkEntry(cls_row, textvariable=self._src_class,
                       width=44).pack(side="left", padx=2)
        ctk.CTkLabel(cls_row, text="(y, x):", width=58,
                       anchor="w").pack(side="left", padx=(8, 2))
        self._src_y = ctk.StringVar(value="64")
        self._src_x = ctk.StringVar(value="64")
        ctk.CTkEntry(cls_row, textvariable=self._src_y,
                       width=44).pack(side="left", padx=2)
        ctk.CTkEntry(cls_row, textvariable=self._src_x,
                       width=44).pack(side="left", padx=2)
        ctk.CTkButton(sidebar, text="Load source →",
                       width=240,
                       fg_color=("#4D6FB0", "#3A5380"),
                       command=self._load_source
                       ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Pick point from BF / HAADF…",
                       width=240,
                       command=self._open_point_picker
                       ).pack(anchor="w", padx=10, pady=2)

        # 2. CALIBRATION
        _section_header(sidebar, "2.  Calibration")
        cal_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        cal_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(cal_row, text="1/Å per px:",
                       width=92, anchor="w").pack(side="left")
        rp = self._recip_per_px() or 0.0185
        self._inv_ang = ctk.DoubleVar(value=round(rp * 0.1, 6))
        ctk.CTkEntry(cal_row, textvariable=self._inv_ang,
                       width=80).pack(side="left", padx=2)
        # Direct-beam mask (detector px).  Defaults to the dataset's
        # center_mask_radius on load (else a dialog asks).  Bins inside
        # this radius are zeroed everywhere so the beam can't dominate
        # the radial profile / SNIP baseline / scattered-intensity.
        beam_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        beam_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(beam_row, text="beam mask (px):",
                       width=110, anchor="w").pack(side="left")
        self._beam_px = ctk.IntVar(value=0)
        eb = ctk.CTkEntry(beam_row, textvariable=self._beam_px, width=56)
        eb.pack(side="left", padx=2)
        eb.bind("<Return>", lambda _e: self._recompute_test())
        self._beam_sample = None

        # 3. R WINDOW
        _section_header(sidebar, "3.  Radial window  (1/Å)")
        _hint(sidebar,
              "Sets the q-range where peak/background ratio is "
              "computed.  Sliders re-tune in real-time.  Default = "
              "first-3rd ring band.")
        # Slider range is updated dynamically from the loaded pattern's
        # half-size × calibration (see _update_r_slider_range).  Start
        # from a 256-px-radius guess (512 detector) so it isn't tiny.
        max_q_default = 256 * (self._recip_per_px() * 0.1 or 0.00185)
        self._r_min = ctk.DoubleVar(value=max(max_q_default * 0.10,
                                                     0.02))
        self._r_max = ctk.DoubleVar(value=max(max_q_default * 0.60,
                                                     0.2))
        self._r_sliders = []
        for label, var in (("r_min", self._r_min),
                              ("r_max", self._r_max)):
            row = ctk.CTkFrame(sidebar, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=1)
            ctk.CTkLabel(row, text=label, width=60,
                          anchor="w").pack(side="left")
            e = ctk.CTkEntry(row, textvariable=var, width=70)
            e.pack(side="left", padx=2)
            e.bind("<Return>", lambda _e: self._recompute_test())
            sl = ctk.CTkSlider(row, from_=0.0, to=max_q_default * 1.05,
                                  variable=var, width=120,
                                  command=lambda _v: self._recompute_test())
            sl.pack(side="left", padx=4)
            self._r_sliders.append(sl)

        # 4. HALO BASELINE (1D) — peak/halo + azimuthal variance, no 2D
        # peak finding (weak peaks survive azimuthal integration).
        _section_header(sidebar, "4.  Halo baseline (1D)")
        _hint(sidebar,
              "Crystallinity is computed in 1D from the azimuthal mean "
              "I(q): a SNIP baseline estimates the amorphous halo, and the "
              "area above it (÷ total) is the peak/halo ratio.  Azimuthal "
              "variance gives spottiness.  No 2D peak finding — weak peaks "
              "aren't lost.")
        snip_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        snip_row.pack(fill="x", padx=10, pady=1)
        ctk.CTkLabel(snip_row, text="SNIP window (bins):", width=140,
                       anchor="w").pack(side="left")
        self._snip_win = ctk.IntVar(value=14)
        e = ctk.CTkEntry(snip_row, textvariable=self._snip_win, width=56)
        e.pack(side="left", padx=2)
        e.bind("<Return>", lambda _e: self._recompute_test())
        sl = ctk.CTkSlider(sidebar, from_=2, to=60,
                            variable=self._snip_win, width=240,
                            command=lambda _v: self._recompute_test())
        sl.pack(anchor="w", padx=10, pady=1)
        metric_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        metric_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(metric_row, text="map metric:", width=78,
                       anchor="w").pack(side="left")
        self._map_metric = ctk.StringVar(value="ratio")
        ctk.CTkSegmentedButton(
            metric_row, values=["ratio", "variance"],
            variable=self._map_metric,
            command=lambda _v: self._render_test_panels()
            ).pack(side="left", padx=2)
        ctk.CTkButton(sidebar, text="Recompute ▶",
                       width=240,
                       command=self._recompute_test
                       ).pack(anchor="w", padx=10, pady=2)
        self._test_status = ctk.CTkLabel(sidebar,
            text="(no test pattern loaded)",
            font=("Consolas", 9),
            text_color=("#666", "#aaa"),
            wraplength=290, justify="left", anchor="w")
        self._test_status.pack(anchor="w", padx=10, pady=(2, 4))

        # 5. RUN FULL
        _section_header(sidebar, "5.  Run on full dataset")
        full_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        full_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(full_row, text="stride:",
                       width=60, anchor="w").pack(side="left")
        self._full_stride = ctk.IntVar(value=1)
        ctk.CTkEntry(full_row, textvariable=self._full_stride,
                       width=44).pack(side="left", padx=2)
        # Vacuum/noise mask: positions whose post-beam scattered intensity
        # falls in the bottom percentile are NaN'd out (no scattering =
        # empty/vacuum, or pure noise).  0 = off.  Applied non-
        # destructively so you can tune it without re-running.
        vac_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        vac_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(vac_row, text="mask vacuum <pctile:", width=140,
                       anchor="w").pack(side="left")
        self._vac_pct = ctk.DoubleVar(value=0.0)
        ev = ctk.CTkEntry(vac_row, textvariable=self._vac_pct, width=50)
        ev.pack(side="left", padx=2)
        ev.bind("<Return>", lambda _e: self._render_test_panels())
        ctk.CTkButton(sidebar, text="Run on full dataset ▶",
                       width=240,
                       fg_color=("#2D7A2D", "#1F7A1F"),
                       command=self._run_full_dataset
                       ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Per-cluster crystallinity  (bars)",
                       width=240,
                       command=self._render_per_cluster
                       ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Save map  (npy + png)",
                       width=240,
                       command=self._save_map
                       ).pack(anchor="w", padx=10, pady=2)

        # 6. REPORT-ALL: sweep sliding windows [r, r+dr] from the
        # center-beam truncation outward; both maps for every window.
        _section_header(sidebar, "6.  Report all  (r-sweep)")
        _hint(sidebar,
              "Sweeps non-overlapping windows of width dr from r_min "
              "(your center-beam truncation) to the detector edge.  For "
              "EACH window it computes both maps (peak/halo + variance) "
              "and saves every .npy + .png plus montages to one folder.")
        dr_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        dr_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(dr_row, text="dr (1/Å):", width=70,
                       anchor="w").pack(side="left")
        self._dr_var = ctk.DoubleVar(value=0.05)
        ctk.CTkEntry(dr_row, textvariable=self._dr_var,
                       width=70).pack(side="left", padx=2)
        ctk.CTkButton(sidebar, text="Report all ▶",
                       width=240,
                       fg_color=("#7A4DA0", "#5A3A80"),
                       command=self._report_all
                       ).pack(anchor="w", padx=10, pady=2)
        self._full_status = ctk.CTkLabel(sidebar,
            text="(no full-dataset run yet)",
            font=("Consolas", 9),
            text_color=("#666", "#aaa"),
            wraplength=290, justify="left", anchor="w")
        self._full_status.pack(anchor="w", padx=10, pady=(2, 8))

        # RIGHT: canvas with 2×2 panels.
        canvas_holder = ctk.CTkFrame(body)
        canvas_holder.pack(side="left", fill="both", expand=True,
                            padx=(8, 0))
        tb_frame = tk.Frame(canvas_holder, bg="#f4f4f4")
        tb_frame.pack(side="top", fill="x")
        self._fig = Figure(figsize=(11, 8), dpi=95,
                              facecolor="#f4f4f4")
        self._canvas = FigureCanvasTkAgg(self._fig, master=canvas_holder)
        self._canvas.get_tk_widget().pack(side="top", fill="both",
                                            expand=True)
        self._toolbar = NavigationToolbar2Tk(self._canvas, tb_frame,
                                               pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(side="left")
        self._build_axes()
        self._refresh_from_posthoc()

    def _reset_axes(self):
        """Rebuild the 2×2 axes from scratch.  Rebuilding (rather than
        ax.clear()) is essential: each render adds a twinx() axis and a
        map colorbar, which otherwise STACK up every time the source
        changes."""
        self._fig.clf()
        gs = self._fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)
        self._ax_cart  = self._fig.add_subplot(gs[0, 0])
        self._ax_polar = self._fig.add_subplot(gs[0, 1])
        self._ax_1d    = self._fig.add_subplot(gs[1, 0])
        self._ax_map   = self._fig.add_subplot(gs[1, 1])

    def _build_axes(self):
        self._reset_axes()
        for ax, txt in ((self._ax_cart,  "(load source — step 1)"),
                          (self._ax_polar, "(polar view appears here)"),
                          (self._ax_1d,    "(1D radial appears here)"),
                          (self._ax_map,   "(map appears after step 5)")):
            ax.text(0.5, 0.5, txt, ha="center", va="center",
                     fontsize=10, color="#888", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    def _refresh_from_posthoc(self):
        """Back-compat shim — session subscription updates dots."""
        self._on_session_change(getattr(self.app, "session", None))

    def _on_session_change(self, sess):
        try:
            self._dot_data.set(
                "ok" if (sess and sess.has_dataset()) else "idle",
                (sess.sample if sess else "") or "no dataset")
            self._dot_run.set(
                "ok" if (sess and sess.has_run()) else "idle",
                (os.path.basename(sess.run_dir or "")
                  if sess and sess.run_dir else "no run loaded"))
        except Exception: pass

    # ------------------------------------------------------------------
    def _load_source(self):
        ph = self._posthoc()
        if ph is None or ph.sample is None:
            messagebox.showinfo("Source",
                "Load a run in the Post-hoc tab first."); return
        src = self._source_var.get()
        cfg = SAMPLES[ph.sample]
        try:
            if src in ("dp_max", "dp_mean"):
                self._test_status.configure(
                    text=f"computing {src} … (~5–30 s)")
                threading.Thread(target=lambda:
                    self._load_dp_async(ph, cfg, src),
                    daemon=True).start()
                return
            if src == "class_avg":
                cid = int(self._src_class.get())
                avgs = ph._compute_class_averages(top_n=256)
                if cid < 0 or cid >= len(avgs) or avgs[cid] is None:
                    messagebox.showerror("Source",
                        f"class p{cid} unavailable."); return
                vm = float(cfg.get("vmax", 5.0))
                self._set_test_pattern(
                    (avgs[cid] * vm).astype(np.float32),
                    f"class p{cid} avg")
                return
            if src == "scan_pos":
                y = int(self._src_y.get()); x = int(self._src_x.get())
                Ny, Nx = ph._scan_shape
                if not (0 <= y < Ny and 0 <= x < Nx):
                    messagebox.showerror("Source",
                        f"(y, x) out of range"); return
                ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
                idx = y * Nx + x
                self._set_test_pattern(
                    ds.get_raw(idx).astype(np.float32),
                    f"scan pos ({y}, {x})")
                return
        except Exception as e:
            messagebox.showerror("Source", repr(e))

    # ------------------------------------------------------------------
    # Point picker — click a virtual BF / HAADF image to choose (y, x).
    # ------------------------------------------------------------------
    def _open_point_picker(self):
        ph = self._posthoc()
        if ph is None or ph.sample is None:
            messagebox.showinfo("Pick point",
                "Load a run / dataset in the Post-hoc tab first."); return
        # Reuse cached virtual images if the sample hasn't changed.
        if (getattr(self, "_vimg_sample", None) == ph.sample
                and getattr(self, "_bf_img", None) is not None):
            self._show_picker_window(ph)
            return
        if self._busy:
            return
        self._busy = True
        self._test_status.configure(
            text="computing virtual BF/HAADF for the picker …")
        threading.Thread(target=self._compute_virtual_images,
                          args=(ph,), daemon=True).start()

    def _compute_virtual_images(self, ph):
        try:
            from gui_app.posthoc_panel import _open_lazy
            cfg = SAMPLES[ph.sample]
            cube = _open_lazy(cfg["path"], scan_shape=ph._scan_shape)
            Ny, Nx, H, W = cube.shape
            cy, cx = H / 2.0, W / 2.0
            R = min(H, W) / 2.0
            yy, xx = np.indices((H, W))
            rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            bf_mask = rr <= (0.12 * R)                 # central beam (BF)
            ha_mask = (rr >= 0.40 * R) & (rr <= 0.98 * R)   # outer (HAADF)
            bf = np.zeros((Ny, Nx), np.float32)
            ha = np.zeros((Ny, Nx), np.float32)
            t0 = time.time()
            for rx in range(Ny):
                for ry in range(Nx):
                    try:
                        pat = np.asarray(cube[rx, ry], dtype=np.float32)
                    except Exception:
                        continue
                    bf[rx, ry] = float(pat[bf_mask].sum())
                    ha[rx, ry] = float(pat[ha_mask].sum())
                if (rx & 7) == 0:
                    dt = time.time() - t0
                    eta = (dt / max(rx + 1, 1)) * (Ny - rx - 1)
                    self.after(0, lambda r=rx, dt=dt, eta=eta:
                        self._test_status.configure(
                            text=f"virtual BF/HAADF: row {r+1}/{Ny}  "
                                  f"({dt:.0f}s, ETA {eta:.0f}s)"))
            self._bf_img = bf
            self._haadf_img = ha
            self._vimg_sample = ph.sample
            self.after(0, lambda: self._show_picker_window(ph))
            self.after(0, lambda: self._test_status.configure(
                text="picker ready — click a point."))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "Pick point", err))
        finally:
            self._busy = False

    def _show_picker_window(self, ph):
        win = tk.Toplevel(self)
        win.title(f"point picker — {ph.sample}")
        win.geometry("640x680")
        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.pack(side="top", fill="x", padx=6, pady=4)
        mode = ctk.StringVar(value="HAADF")
        ctk.CTkLabel(bar, text="image:").pack(side="left", padx=(4, 2))
        ctk.CTkSegmentedButton(bar, values=["BF", "HAADF"],
                                 variable=mode,
                                 command=lambda _v: _draw()
                                 ).pack(side="left", padx=2)
        picked = ctk.CTkLabel(bar, text="click to pick (y, x)")
        picked.pack(side="left", padx=12)
        fig = Figure(figsize=(6.0, 6.0), dpi=110, facecolor="white")
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def _draw():
            ax.clear()
            img = (self._bf_img if mode.get() == "BF" else self._haadf_img)
            finite = img[np.isfinite(img)]
            vmin = float(np.percentile(finite, 2)) if finite.size else 0
            vmax = float(np.percentile(finite, 98)) if finite.size else 1
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="equal")
            ax.set_title(f"virtual {mode.get()} — click a scan position",
                          fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            # marker for the current selection
            try:
                yy = int(self._src_y.get()); xx = int(self._src_x.get())
                ax.scatter([xx], [yy], s=80, facecolors="none",
                            edgecolors="#e0144c", linewidths=1.6)
            except Exception:
                pass
            canvas.draw_idle()

        def _on_click(event):
            if event.inaxes is not ax or event.xdata is None:
                return
            Ny, Nx = self._bf_img.shape
            x = int(max(0, min(Nx - 1, round(event.xdata))))
            y = int(max(0, min(Ny - 1, round(event.ydata))))
            self._src_y.set(str(y)); self._src_x.set(str(x))
            self._source_var.set("scan_pos")
            picked.configure(text=f"picked (y={y}, x={x})")
            _draw()
            # Load that frame as the test pattern + recompute.
            self._load_source()

        canvas.mpl_connect("button_press_event", _on_click)
        _draw()

    def _load_dp_async(self, ph, cfg, src):
        try:
            from gui_app.posthoc_panel import _open_lazy
            cube = _open_lazy(cfg["path"], scan_shape=ph._scan_shape)
            Ny, Nx, H, W = cube.shape
            dp_max = np.zeros((H, W), dtype=np.float32)
            dp_sum = np.zeros((H, W), dtype=np.float64)
            for y in range(Ny):
                block = np.asarray(cube[y], dtype=np.float32)
                dp_max = np.maximum(dp_max, block.max(axis=0))
                dp_sum += block.sum(axis=0)
            pat = (dp_max if src == "dp_max"
                    else (dp_sum / max(Ny * Nx, 1)).astype(np.float32))
            self.after(0, lambda: self._set_test_pattern(
                pat, f"{src}  ({ph.sample})"))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror("dp_max", err))

    def _set_test_pattern(self, pat, origin):
        self._test_pattern = np.asarray(pat, dtype=np.float32)
        self._test_origin = origin
        H, W = self._test_pattern.shape
        self._test_center = (H / 2.0, W / 2.0)
        # Resolve the direct-beam mask for this dataset (uses the data's
        # center mask if known, else asks) before computing anything.
        ph = self._posthoc()
        if ph is not None and ph.sample is not None:
            try:
                self._ensure_beam_mask(ph.sample, H)
            except Exception:
                pass
        self._test_polar = cart_to_polar(self._test_pattern,
                                              n_theta=192,
                                              center=self._test_center)
        self._update_r_slider_range()
        self._test_status.configure(
            text=f"loaded: {origin}  ({H}×{W})")
        self._recompute_test()

    def _update_r_slider_range(self):
        """Set r-slider max to the loaded pattern's max radius × calib
        (1/Å).  Fixes the old hardcoded 0.2 cap which assumed a 192-px
        pattern; raw patterns are often 512 so q extends ~2.5× further."""
        if self._test_pattern is None:
            return
        H, W = self._test_pattern.shape
        r_max_px = float(min(H, W)) / 2.0          # corner-safe radius
        q_max = r_max_px * self._inv_ang_per_px()
        if q_max <= 0:
            return
        for sl in getattr(self, "_r_sliders", []):
            try: sl.configure(to=q_max)
            except Exception: pass
        # If current r_max is below ~half the new range, open it up so
        # the user sees the full q-span by default.
        try:
            if float(self._r_max.get()) < q_max * 0.3:
                self._r_max.set(round(q_max * 0.6, 4))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _recompute_test(self):
        if self._test_pattern is None:
            return
        inv_a = self._inv_ang_per_px()
        r_min_px = float(self._r_min.get()) / max(inv_a, 1e-12)
        r_max_px = float(self._r_max.get()) / max(inv_a, 1e-12)
        if r_min_px >= r_max_px:
            self._test_status.configure(
                text="r_min must be < r_max"); return
        try:
            snipw = int(float(self._snip_win.get()))
        except Exception:
            snipw = 14
        try:
            res = compute_crystallinity_1d(
                self._test_pattern, r_min_px, r_max_px,
                center=self._test_center, snip_window=snipw,
                beam_px=self._beam_radius_px())
        except Exception as e:
            self._test_status.configure(text=f"err: {e!r}"); return
        if res is None:
            self._test_status.configure(
                text="window too narrow (need ≥ 8 radial bins)"); return
        res["r_min_px"] = r_min_px
        res["r_max_px"] = r_max_px
        res["inv_a"] = inv_a
        self._test_result = res
        self._test_status.configure(
            text=f"peak/halo = {res['ratio']:.4f}     "
                  f"azim-var = {res['var_index']:.4f}")
        self._render_test_panels()

    def _render_test_panels(self):
        if self._test_pattern is None or self._test_result is None:
            return
        # Rebuild axes from scratch so twin axes / colorbars from the
        # previous source don't stack up.
        self._reset_axes()
        r = self._test_result
        cy, cx = self._test_center
        # ---- cart pattern + annulus (no 2D peak finding) ----
        ax = self._ax_cart
        bpx = self._beam_radius_px()
        H0, W0 = self._test_pattern.shape
        yy0, xx0 = np.indices((H0, W0))
        beam_mask2d = (np.sqrt((yy0 - cy) ** 2 + (xx0 - cx) ** 2) < bpx) \
            if bpx > 0 else np.zeros((H0, W0), bool)
        disp = self._test_pattern.copy()
        disp[beam_mask2d] = 0.0          # blank the beam so it can't
        img = np.log1p(np.clip(disp, 0, None))   # dominate the contrast
        # autoscale to the diffraction (exclude the masked beam region)
        out = img[~beam_mask2d]
        vmax = float(np.percentile(out, 99.5)) if out.size else float(img.max())
        ax.imshow(img, cmap="inferno", aspect="equal", vmin=0.0,
                    vmax=max(vmax, 1e-6), interpolation="nearest")
        # annulus rings mark the q-window used for the 1D analysis
        for rad, color in ((r["r_min_px"], "#33ddff"),
                              (r["r_max_px"], "#33ddff")):
            ax.add_patch(Circle((cx, cy), rad, color=color,
                                  fill=False, lw=1.5, linestyle="--"))
        # direct-beam mask outline (region blanked above)
        if bpx > 0:
            ax.add_patch(Circle((cx, cy), bpx, color="#ff3b3b",
                                  fill=False, lw=1.2))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{self._test_origin}    "
                       f"peak/halo = {r['ratio']:.4f}", fontsize=10)
        # Hover-q on the cart pattern (raw-detector resolution).
        try:
            from gui_app._ui import attach_hover_q
            rp = self._recip_per_px()
            if rp > 0 and self._test_center is not None:
                if getattr(self, "_hover_cid", None) is not None:
                    self._canvas.mpl_disconnect(self._hover_cid)
                self._hover_cid = attach_hover_q(
                    self._canvas, ax,
                    center=self._test_center,
                    q_per_disp_px=rp, units="nm⁻¹")
        except Exception: pass

        # ---- polar view + r-window band ----
        ax = self._ax_polar; ax.clear()
        polar = np.array(self._test_polar, dtype=np.float32, copy=True)
        n_theta, n_r = polar.shape
        inv_a = r["inv_a"]
        scale_x = n_r / max(self._test_pattern.shape[0] / 2, 1)
        beam_cols = int(round(bpx * scale_x)) if bpx > 0 else 0
        if beam_cols > 0:
            polar[:, :min(beam_cols, n_r)] = 0.0   # blank the beam columns
        pimg = np.log1p(np.clip(polar, 0, None))
        pout = pimg[:, beam_cols:] if beam_cols < n_r else pimg
        pvmax = float(np.percentile(pout, 99.5)) if pout.size \
            else float(pimg.max())
        ax.imshow(pimg, cmap="inferno", aspect="auto", vmin=0.0,
                    vmax=max(pvmax, 1e-6), interpolation="nearest")
        # window x-pixels in polar frame: r-axis is 0..n_r-1 px
        x_min = r["r_min_px"] * scale_x
        x_max = r["r_max_px"] * scale_x
        ax.axvspan(x_min, x_max, color="#33ddff", alpha=0.18)
        ax.axvline(x_min, color="#33ddff", lw=1.2, linestyle="--")
        ax.axvline(x_max, color="#33ddff", lw=1.2, linestyle="--")
        ax.set_xlabel("r  (px in polar frame)")
        ax.set_ylabel("θ  (0..2π)")
        ax.set_yticks([])
        ax.set_title(
            f"polar view  (window q = "
            f"{self._r_min.get():.3g}–{self._r_max.get():.3g} 1/Å)",
            fontsize=10)

        # ---- 1D radial profile (lin-log, truncated): I(q) + SNIP halo +
        #      shaded peak area, with azimuthal variance on a twin axis ----
        ax = self._ax_1d; ax.clear()
        q = r["r_px"] * inv_a                       # 1/Å (truncated window)
        m = np.clip(r["m"], 1e-3, None)
        halo = np.clip(r["halo"], 1e-3, None)
        ax.semilogy(q, m, color="#1f77b4", lw=1.4, label="I(q) azim-mean")
        ax.semilogy(q, halo, color="#888", lw=1.2, ls="--",
                       label="halo (SNIP)")
        ax.fill_between(q, halo, m, where=(m > halo),
                          color="#e0144c", alpha=0.25, label="peak")
        ax.set_xlabel("q  (1/Å)")
        ax.set_ylabel("I(q)   (log)")
        ax.set_title(f"1D radial — peak/halo={r['ratio']:.3f}  "
                       f"azim-var={r['var_index']:.3f}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        try:
            ax2 = ax.twinx()
            ax2.plot(q, r["v"] / np.maximum(r["m"] ** 2, 1e-12),
                      color="#2ca02c", lw=1.0, alpha=0.85)
            ax2.set_ylabel("azim var / mean²", color="#2ca02c", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="#2ca02c", labelsize=7)
        except Exception:
            pass

        # ---- map (placeholder until Run) — honours the metric toggle
        #      and the vacuum/noise mask ----
        ax = self._ax_map; ax.clear()
        metric = getattr(self, "_map_metric", None)
        is_var = bool(metric is not None and metric.get() == "variance")
        the_map = (getattr(self, "_var_map", None) if is_var
                    else self._cryst_map)
        the_map = self._apply_vac(the_map)
        if the_map is not None:
            cut = self._vac_cutoff()
            im = ax.imshow(the_map, cmap="viridis",
                              interpolation="nearest", aspect="equal")
            tag = (f"  (vac<{self._vac_pct.get():g}pct masked)"
                    if cut is not None else "")
            ax.set_title(("azim-variance map" if is_var
                          else "peak/halo map") + tag, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            try:
                self._fig.colorbar(im, ax=ax, fraction=0.045,
                                      pad=0.02)
            except Exception:
                pass
        else:
            ax.text(0.5, 0.5,
                       "(click 'Run on full dataset' below)",
                       ha="center", va="center", fontsize=10,
                       color="#888", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    def _run_full_dataset(self):
        ph = self._posthoc()
        if ph is None or ph.sample is None:
            messagebox.showinfo("Run",
                "Load a run in the Post-hoc tab first."); return
        if self._test_result is None:
            messagebox.showinfo("Run",
                "Tune the test pattern first (steps 1–4)."); return
        if self._busy:
            return
        self._busy = True
        threading.Thread(target=self._run_full_worker,
                          daemon=True).start()

    def _run_full_worker(self):
        try:
            ph = self._posthoc()
            from gui_app.posthoc_panel import _open_lazy
            cfg = SAMPLES[ph.sample]
            cube = _open_lazy(cfg["path"], scan_shape=ph._scan_shape)
            Ny, Nx, H, W = cube.shape
            center = (H / 2.0, W / 2.0)
            inv_a = float(self._inv_ang.get())
            r_min_px = float(self._r_min.get()) / max(inv_a, 1e-12)
            r_max_px = float(self._r_max.get()) / max(inv_a, 1e-12)
            stride = max(int(self._full_stride.get()), 1)
            try:
                snipw = int(float(self._snip_win.get()))
            except Exception:
                snipw = 14
            lo0 = max(0, int(round(r_min_px)))   # post-beam truncation
            beam = self._beam_radius_px()
            ratio_map = np.full((Ny, Nx), np.nan, dtype=np.float32)
            var_map = np.full((Ny, Nx), np.nan, dtype=np.float32)
            scatter_map = np.full((Ny, Nx), np.nan, dtype=np.float32)
            t0 = time.time()
            total = (Ny // stride) * (Nx // stride)
            done = 0
            for rx in range(0, Ny, stride):
                for ry in range(0, Nx, stride):
                    try:
                        pat = np.asarray(cube[rx, ry],
                                            dtype=np.float32)
                    except Exception:
                        continue
                    try:
                        m_all, v_all, cnt = _radial_mean_var(
                            pat, center, beam)
                        # post-beam scattered intensity (vacuum/noise test)
                        scatter_map[rx, ry] = float(
                            (m_all[lo0:] * cnt[lo0:]).sum())
                        res = _crystallinity_window(
                            m_all, v_all, int(round(r_min_px)),
                            int(round(r_max_px)), snipw)
                        if res is not None:
                            ratio_map[rx, ry] = res["ratio"]
                            var_map[rx, ry] = res["var_index"]
                    except Exception:
                        pass
                    done += 1
                if (rx & 7) == 0:
                    dt = time.time() - t0
                    eta = (dt / max(done, 1)) * (total - done)
                    self.after(0, lambda d=done, t=total, dt=dt,
                                  eta=eta: self._full_status.configure(
                        text=f"{d}/{t}  ({dt:.0f}s elapsed, "
                              f"ETA {eta:.0f}s)"))

            def _fill_stride(arr):
                # Stride > 1: nearest-neighbour fill of skipped positions.
                cur = arr.copy()
                mask = ~np.isnan(cur)
                for _ in range(stride):
                    next_ = cur.copy()
                    for shift in (1, -1):
                        for axis in (0, 1):
                            rolled = np.roll(cur, shift, axis=axis)
                            new = ~mask & ~np.isnan(rolled)
                            next_[new] = rolled[new]
                            mask = mask | new
                    cur = next_
                return cur
            if stride > 1:
                ratio_map = _fill_stride(ratio_map)
                var_map = _fill_stride(var_map)
                scatter_map = _fill_stride(scatter_map)
            self._cryst_map = ratio_map
            self._var_map = var_map
            self._scatter_map = scatter_map
            dt = time.time() - t0
            self.after(0, lambda: self._full_status.configure(
                text=f"done ({dt:.0f}s)  stride={stride}  "
                      f"peak/halo med={np.nanmedian(ratio_map):.3f}  "
                      f"azim-var med={np.nanmedian(var_map):.4f}"))
            self.after(0, self._render_test_panels)
            # If a trained run is linked, also aggregate per DINO cluster.
            self.after(0, self._render_per_cluster)
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "Run on full dataset", err))
        finally:
            self._busy = False

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Vacuum / noise mask (scattered-intensity percentile cutoff).
    # ------------------------------------------------------------------
    def _vac_cutoff(self, scatter_map=None):
        """Absolute scattered-intensity cutoff for the current percentile,
        or None if masking is off / unavailable."""
        sm = (scatter_map if scatter_map is not None
              else getattr(self, "_scatter_map", None))
        try:
            pct = float(self._vac_pct.get())
        except Exception:
            pct = 0.0
        if sm is None or pct <= 0:
            return None
        finite = sm[np.isfinite(sm)]
        if finite.size == 0:
            return None
        return float(np.percentile(finite, pct))

    def _apply_vac(self, arr, scatter_map=None):
        """Return a copy of `arr` with vacuum/noise positions (scattered
        intensity below the percentile cutoff) set to NaN."""
        sm = (scatter_map if scatter_map is not None
              else getattr(self, "_scatter_map", None))
        cut = self._vac_cutoff(sm)
        if arr is None or sm is None or cut is None:
            return arr
        out = np.array(arr, dtype=np.float32, copy=True)
        out[~(sm >= cut)] = np.nan
        return out

    def _render_per_cluster(self):
        """Aggregate the crystallinity ratio per DINO cluster and show
        a bar chart (mean ± IQR) + a table.  Needs the ratio map + the
        posthoc inference (class assignments)."""
        if self._cryst_map is None:
            messagebox.showinfo("Per-cluster",
                "Run the full dataset first."); return
        ph = self._posthoc()
        if ph is None or getattr(ph, "_inf", None) is None:
            messagebox.showinfo("Per-cluster",
                "Per-cluster needs a trained run (DINO class map). "
                "Load one via the topbar 'run' badge."); return
        Ny, Nx = ph._scan_shape
        assigns = np.asarray(ph._inf["assigns"]).reshape(Ny, Nx)
        K = int(ph._inf["soft_probs"].shape[1])
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
        modN = 10 if K <= 10 else 20

        def _per_class(metric_map):
            ks, meds, q1s, q3s, ns = [], [], [], [], []
            if metric_map is None:
                return ks, meds, q1s, q3s, ns
            for c in range(K):
                vals = metric_map[(assigns == c) & np.isfinite(metric_map)]
                if vals.size == 0:
                    continue
                ks.append(c); ns.append(int(vals.size))
                meds.append(float(np.median(vals)))
                q1s.append(float(np.percentile(vals, 25)))
                q3s.append(float(np.percentile(vals, 75)))
            return ks, meds, q1s, q3s, ns

        win = tk.Toplevel(self)
        win.title("crystallinity per DINO cluster")
        win.geometry("900x640")
        fig = Figure(figsize=(8.8, 6.2), dpi=110, facecolor="white")
        # Vacuum/noise positions are NaN'd out so they don't bias the
        # per-class medians.
        panels = [("peak/halo ratio", self._apply_vac(self._cryst_map)),
                   ("azimuthal variance (spottiness)",
                    self._apply_vac(getattr(self, "_var_map", None)))]
        any_ok = False
        for pi, (name, mp) in enumerate(panels):
            ax = fig.add_subplot(2, 1, pi + 1)
            ks, meds, q1s, q3s, ns = _per_class(mp)
            if not ks:
                ax.text(0.5, 0.5, f"({name}: no data)", ha="center",
                         va="center", transform=ax.transAxes,
                         color="#888")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            any_ok = True
            order = np.argsort(-np.asarray(meds))   # most crystalline first
            ks = [ks[i] for i in order]; meds = [meds[i] for i in order]
            ns = [ns[i] for i in order]
            q1s = [q1s[i] for i in order]; q3s = [q3s[i] for i in order]
            x = np.arange(len(ks))
            yerr = [np.array(meds) - np.array(q1s),
                    np.array(q3s) - np.array(meds)]
            cols = [cmap(c % modN) for c in ks]
            ax.bar(x, meds, yerr=yerr, capsize=3, color=cols,
                      edgecolor="#333", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([f"p{c}\n{n}px" for c, n in zip(ks, ns)],
                                  fontsize=8)
            ax.set_ylabel(f"{name}\n(median ± IQR)", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if pi == 0:
                ax.set_title(
                    f"per-cluster crystallinity  "
                    f"(q = {self._r_min.get():.3g}–"
                    f"{self._r_max.get():.3g} 1/Å)  ·  higher = more "
                    f"crystalline / spotty", fontsize=10)
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=win)
        c.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(c, win, pack_toolbar=False)
        tb.update(); tb.pack(side="bottom", fill="x")
        if not any_ok:
            self._full_status.configure(
                text="per-cluster: no overlap between maps and class map")
        else:
            self._full_status.configure(
                text="per-cluster crystallinity (peak/halo + variance) "
                      "computed")

    def _save_map(self):
        if self._cryst_map is None:
            messagebox.showinfo("Save",
                "No map yet — run the full dataset first."); return
        ph = self._posthoc()
        outdir = (ph.outdir if ph and ph.outdir
                   else os.path.join("runs", "_crystallinity"))
        out = os.path.join(outdir, "crystallinity")
        os.makedirs(out, exist_ok=True)
        from datetime import datetime
        stamp = datetime.now().strftime("%H%M%S")
        qtag = (f"q = {self._r_min.get():.3g}–"
                 f"{self._r_max.get():.3g} 1/Å")
        saved = []
        for name, mp in (("peakhalo", self._apply_vac(self._cryst_map)),
                          ("azimvar",
                           self._apply_vac(getattr(self, "_var_map", None)))):
            if mp is None:
                continue
            np.save(os.path.join(out, f"{name}_{stamp}.npy"), mp)
            fig, ax = matplotlib.pyplot.subplots(figsize=(7, 6))
            im = ax.imshow(mp, cmap="viridis",
                              interpolation="nearest", aspect="equal")
            title = ("peak/halo crystallinity map" if name == "peakhalo"
                      else "azimuthal-variance (spottiness) map")
            ax.set_title(f"{title}  ({qtag})", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            fig.savefig(os.path.join(out, f"{name}_{stamp}.png"),
                          dpi=180, bbox_inches="tight", facecolor="white")
            matplotlib.pyplot.close(fig)
            saved.append(name)
        self._full_status.configure(
            text=f"saved {', '.join(saved)} (.npy + .png) → "
                  f"{os.path.basename(out)}/")

    # ------------------------------------------------------------------
    def _report_all(self):
        ph = self._posthoc()
        if ph is None or ph.sample is None:
            messagebox.showinfo("Report all",
                "Load a run / dataset in the Post-hoc tab first."); return
        try:
            dr = float(self._dr_var.get())
        except Exception:
            dr = 0.0
        if dr <= 0:
            messagebox.showinfo("Report all", "dr must be > 0."); return
        if self._busy:
            return
        # Resolve the beam mask on the main thread (may pop a dialog).
        try:
            from gui_app.posthoc_panel import _open_lazy
            shp = _open_lazy(SAMPLES[ph.sample]["path"],
                              scan_shape=ph._scan_shape).shape
            self._ensure_beam_mask(ph.sample, int(shp[2]))
        except Exception:
            pass
        self._busy = True
        threading.Thread(target=self._report_all_worker,
                          daemon=True).start()

    def _report_all_worker(self):
        try:
            import json
            from datetime import datetime
            from gui_app.posthoc_panel import _open_lazy
            ph = self._posthoc()
            cfg = SAMPLES[ph.sample]
            cube = _open_lazy(cfg["path"], scan_shape=ph._scan_shape)
            Ny, Nx, H, W = cube.shape
            center = (H / 2.0, W / 2.0)
            n_bins = int(min(H, W) // 2)
            inv_a = float(self._inv_ang.get())
            dr = float(self._dr_var.get())
            stride = max(int(self._full_stride.get()), 1)
            try:
                snipw = int(float(self._snip_win.get()))
            except Exception:
                snipw = 14
            # Windows in radial-bin px: start at r_min (center-beam
            # truncation), step by dr, non-overlapping, to detector edge.
            lo0 = int(round(float(self._r_min.get()) / max(inv_a, 1e-12)))
            step = max(int(round(dr / max(inv_a, 1e-12))), 1)
            lo0 = max(lo0, 0)
            beam = self._beam_radius_px()
            windows = []
            lo = lo0
            while lo + step <= n_bins:
                windows.append((lo, lo + step))
                lo += step
            if not windows:
                self.after(0, lambda: messagebox.showinfo(
                    "Report all",
                    "No windows fit — dr too large or r_min too high."))
                return
            nW = len(windows)
            ratio_maps = [np.full((Ny, Nx), np.nan, np.float32)
                           for _ in range(nW)]
            var_maps = [np.full((Ny, Nx), np.nan, np.float32)
                         for _ in range(nW)]
            scatter_map = np.full((Ny, Nx), np.nan, np.float32)
            sum_m = np.zeros(n_bins, np.float64)   # dataset-mean profile
            n_acc = 0
            t0 = time.time()
            total = (Ny // stride) * (Nx // stride)
            done = 0
            for rx in range(0, Ny, stride):
                for ry in range(0, Nx, stride):
                    try:
                        pat = np.asarray(cube[rx, ry], dtype=np.float32)
                    except Exception:
                        continue
                    # ONE radial pass per pattern; evaluate every window.
                    m_all, v_all, cnt = _radial_mean_var(
                        pat, center, beam)
                    sum_m += m_all
                    n_acc += 1
                    # post-beam scattered intensity (window-independent)
                    scatter_map[rx, ry] = float(
                        (m_all[lo0:] * cnt[lo0:]).sum())
                    for w, (a, b) in enumerate(windows):
                        res = _crystallinity_window(m_all, v_all, a, b, snipw)
                        if res is not None:
                            ratio_maps[w][rx, ry] = res["ratio"]
                            var_maps[w][rx, ry] = res["var_index"]
                    done += 1
                if (rx & 3) == 0:
                    dt = time.time() - t0
                    eta = (dt / max(done, 1)) * (total - done)
                    self.after(0, lambda d=done, t=total, dt=dt, eta=eta,
                                  nw=nW: self._full_status.configure(
                        text=f"Report-all: pos {d}/{t} × {nw} windows  "
                              f"({dt:.0f}s, ETA {eta:.0f}s)"))

            def _fill(arr):
                cur = arr.copy(); mask = ~np.isnan(cur)
                for _ in range(stride):
                    nxt = cur.copy()
                    for sh in (1, -1):
                        for ax_ in (0, 1):
                            rolled = np.roll(cur, sh, axis=ax_)
                            new = ~mask & ~np.isnan(rolled)
                            nxt[new] = rolled[new]; mask = mask | new
                    cur = nxt
                return cur
            if stride > 1:
                ratio_maps = [_fill(a) for a in ratio_maps]
                var_maps = [_fill(a) for a in var_maps]
                scatter_map = _fill(scatter_map)

            # Vacuum/noise mask: NaN out low-scattered-intensity positions.
            vac_cut = self._vac_cutoff(scatter_map)
            if vac_cut is not None:
                bad = ~(scatter_map >= vac_cut)
                for w in range(nW):
                    ratio_maps[w][bad] = np.nan
                    var_maps[w][bad] = np.nan
            n_masked = int(np.isfinite(scatter_map).sum()
                            - (np.isfinite(scatter_map)
                               & (scatter_map >= vac_cut)).sum()) \
                if vac_cut is not None else 0

            # ---- save everything to one folder ----
            outdir = (ph.outdir if ph and ph.outdir
                       else os.path.join("runs", "_crystallinity"))
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(outdir, "crystallinity",
                                   f"report_all_{stamp}")
            os.makedirs(folder, exist_ok=True)
            np.save(os.path.join(folder, "_scattered_intensity.npy"),
                      scatter_map)
            try:
                vac_pct_val = float(self._vac_pct.get())
            except Exception:
                vac_pct_val = 0.0
            manifest = dict(sample=ph.sample, inv_ang_per_px=inv_a,
                             dr=dr, dr_bins=step, r_min=float(self._r_min.get()),
                             beam_mask_px=beam,
                             stride=stride, snip_window=snipw,
                             vac_pctile=vac_pct_val,
                             vac_cutoff=(vac_cut if vac_cut is not None
                                          else None),
                             n_masked=n_masked,
                             n_bins=n_bins, n_windows=nW, windows=[])
            for w, (a, b) in enumerate(windows):
                qa, qb = a * inv_a, b * inv_a
                tag = f"w{w:02d}_q{qa:.3f}-{qb:.3f}"
                np.save(os.path.join(folder, f"peakhalo_{tag}.npy"),
                          ratio_maps[w])
                np.save(os.path.join(folder, f"azimvar_{tag}.npy"),
                          var_maps[w])
                for name, mp in (("peakhalo", ratio_maps[w]),
                                  ("azimvar", var_maps[w])):
                    fig = Figure(figsize=(6, 5.4), facecolor="white")
                    ax = fig.add_subplot(111)
                    im = ax.imshow(mp, cmap="viridis",
                                     interpolation="nearest", aspect="equal")
                    ax.set_title(f"{name}  q={qa:.3f}–{qb:.3f} 1/Å",
                                  fontsize=11)
                    ax.set_xticks([]); ax.set_yticks([])
                    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
                    fig.savefig(os.path.join(folder, f"{name}_{tag}.png"),
                                  dpi=160, bbox_inches="tight",
                                  facecolor="white")
                manifest["windows"].append(
                    dict(idx=w, lo_px=a, hi_px=b, q_lo=qa, q_hi=qb))

            # ---- montage grids (one per metric) ----
            ncol = int(np.ceil(np.sqrt(nW)))
            nrow = int(np.ceil(nW / ncol))
            for name, maps in (("peakhalo", ratio_maps),
                                ("azimvar", var_maps)):
                fig = Figure(figsize=(2.6 * ncol, 2.6 * nrow),
                              facecolor="white")
                for w, (a, b) in enumerate(windows):
                    ax = fig.add_subplot(nrow, ncol, w + 1)
                    mp = maps[w]
                    finite = mp[np.isfinite(mp)]
                    vmin = float(np.percentile(finite, 2)) if finite.size else 0
                    vmax = float(np.percentile(finite, 98)) if finite.size else 1
                    ax.imshow(mp, cmap="viridis", vmin=vmin, vmax=vmax,
                                interpolation="nearest", aspect="equal")
                    ax.set_title(f"{a*inv_a:.3f}–{b*inv_a:.3f}", fontsize=7)
                    ax.set_xticks([]); ax.set_yticks([])
                fig.suptitle(f"{name} — r-sweep (dr={dr:g} 1/Å)  {ph.sample}",
                              fontsize=11)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(os.path.join(folder, f"_montage_{name}.png"),
                              dpi=150, bbox_inches="tight", facecolor="white")

            # ---- overlays per analysis type: dominant-dr + additive ----
            from matplotlib.patches import Patch
            try:
                from matplotlib import colormaps as _cmaps
                _base = _cmaps["turbo"]
            except Exception:
                from matplotlib import cm as _cm
                _base = _cm.get_cmap("turbo")
            colors = [_base(i / max(nW - 1, 1)) for i in range(nW)]
            legend_handles = [
                Patch(facecolor=colors[w],
                       label=f"{windows[w][0]*inv_a:.3f}–"
                             f"{windows[w][1]*inv_a:.3f}")
                for w in range(nW)]
            for name, maps in (("peakhalo", ratio_maps),
                                ("azimvar", var_maps)):
                rgb, idx, valid = _overlay_argmax_rgb(maps, colors)
                np.save(os.path.join(folder,
                          f"overlay_{name}_dominant_idx.npy"), idx)
                for style, rgbimg in (("dominant", rgb),
                                       ("additive",
                                        _overlay_additive_rgb(maps, colors))):
                    fig = Figure(figsize=(7.6, 6.0), facecolor="white")
                    ax = fig.add_subplot(111)
                    ax.imshow(rgbimg, interpolation="nearest", aspect="equal")
                    ax.set_title(f"{name}: {style} dr overlay  "
                                  f"({ph.sample})", fontsize=11)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.legend(handles=legend_handles, title="q (1/Å)",
                               fontsize=6, title_fontsize=7,
                               loc="center left", bbox_to_anchor=(1.01, 0.5))
                    fig.savefig(os.path.join(folder,
                                  f"overlay_{name}_{style}.png"),
                                  dpi=150, bbox_inches="tight",
                                  facecolor="white")

            # ---- per-dr 1D fit (subfolder) + intrinsic quality scores ----
            import csv as _csv
            fitdir = os.path.join(folder, "fits_1d")
            os.makedirs(fitdir, exist_ok=True)
            mean_prof = (sum_m / max(n_acc, 1)).astype(np.float64)
            q_full = np.arange(n_bins) * inv_a
            scores_rows = []
            for w, (a, b) in enumerate(windows):
                qa, qb = a * inv_a, b * inv_a
                mw = mean_prof[a:b]
                halo = np.exp(_snip_baseline(
                    np.log(np.clip(mw, 1e-6, None)), snipw))
                peak = np.clip(mw - halo, 0.0, None)
                resid = mw - halo
                noise = 1.4826 * np.median(
                    np.abs(resid - np.median(resid))) + 1e-12
                peak_snr = float(peak.max() / noise) if peak.size else 0.0
                sp_ph = _spatial_autocorr(ratio_maps[w])
                sp_var = _spatial_autocorr(var_maps[w])
                # combined: needs a SPATIALLY-STRUCTURED map AND prominent
                # peaks above noise.  Both in [0,1]-ish; product in [0,1].
                quality = float(max(sp_ph, 0.0) * (peak_snr / (peak_snr + 1.0)))
                scores_rows.append(dict(idx=w, q_lo=round(qa, 5),
                                         q_hi=round(qb, 5),
                                         peak_snr=round(peak_snr, 4),
                                         sp_autocorr_peakhalo=round(sp_ph, 4),
                                         sp_autocorr_var=round(sp_var, 4),
                                         quality=round(quality, 4)))
                fig = Figure(figsize=(7.0, 4.4), facecolor="white")
                ax = fig.add_subplot(111)
                qq = q_full[a:b]
                ax.semilogy(qq, np.clip(mw, 1e-6, None), color="#1f77b4",
                              lw=1.3, label="dataset-mean I(q)")
                ax.semilogy(qq, np.clip(halo, 1e-6, None), color="#888",
                              lw=1.2, ls="--", label="halo (SNIP)")
                ax.fill_between(qq, np.clip(halo, 1e-6, None),
                                  np.clip(mw, 1e-6, None), where=(mw > halo),
                                  color="#e0144c", alpha=0.25, label="peak")
                ax.set_xlabel("q (1/Å)"); ax.set_ylabel("I(q) (log)")
                ax.set_title(f"dr w{w}  q={qa:.3f}–{qb:.3f}   "
                              f"peak_snr={peak_snr:.2f}  quality={quality:.3f}",
                              fontsize=10)
                ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(os.path.join(
                    fitdir, f"fit_w{w:02d}_q{qa:.3f}-{qb:.3f}.png"),
                    dpi=140, bbox_inches="tight", facecolor="white")

            with open(os.path.join(folder, "scores.csv"), "w",
                       newline="") as cf:
                wr = _csv.DictWriter(cf, fieldnames=list(
                    scores_rows[0].keys()))
                wr.writeheader(); wr.writerows(scores_rows)
            # quality-vs-q summary plot
            qmid = [0.5 * (r["q_lo"] + r["q_hi"]) for r in scores_rows]
            max_snr = max((r["peak_snr"] for r in scores_rows), default=1.0)
            fig = Figure(figsize=(7.0, 4.0), facecolor="white")
            ax = fig.add_subplot(111)
            ax.plot(qmid, [r["quality"] for r in scores_rows], "-o",
                     color="#7A4DA0", label="quality (0–1)")
            ax.plot(qmid, [r["peak_snr"] / (max_snr + 1e-9)
                            for r in scores_rows], "--", alpha=0.6,
                     color="#1f77b4", label="peak_snr (norm)")
            ax.plot(qmid, [r["sp_autocorr_peakhalo"] for r in scores_rows],
                     ":", alpha=0.6, color="#2ca02c", label="spatial autocorr")
            ax.set_xlabel("q (1/Å)"); ax.set_ylabel("score")
            ax.set_title("intrinsic quality vs q  (higher = clearer + more "
                          "spatially structured)", fontsize=10)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(folder, "_quality_vs_q.png"),
                          dpi=150, bbox_inches="tight", facecolor="white")
            best = max(scores_rows, key=lambda r: r["quality"],
                        default=None)
            manifest["best_dr"] = (best["idx"] if best else None)
            for w, row in enumerate(scores_rows):
                manifest["windows"][w].update(
                    peak_snr=row["peak_snr"],
                    sp_autocorr_peakhalo=row["sp_autocorr_peakhalo"],
                    sp_autocorr_var=row["sp_autocorr_var"],
                    quality=row["quality"])

            with open(os.path.join(folder, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

            dt = time.time() - t0
            best_q = (f"{best['q_lo']:.3f}–{best['q_hi']:.3f}"
                       if best else "n/a")
            self.after(0, lambda: self._full_status.configure(
                text=f"Report-all done ({dt:.0f}s): {nW} windows × 2 maps "
                      f"+ overlays + fits_1d + scores.  best dr q≈{best_q} "
                      f"→ {os.path.basename(folder)}/"))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "Report all", err))
        finally:
            self._busy = False
