"""Viewing sub-tab — interactive virtual image + live windowed diffraction.

A quick, mouse-driven inspector that sits next to Pre-processing in the Data
section.  It shows a virtual image (BF or HAADF, user's choice) of the cube
currently loaded in the Pre-processing tab, with a draggable n x m window
drawn on it.  The mean diffraction pattern of the probe positions inside that
window is rendered live in the right panel — including while the window is
being dragged (click + hold to grab, move to slide, release to drop).

Design notes
------------
* Window  = a rectangular ROI over the SCAN grid (n rows x m cols).
* Diff    = the MEAN diffraction pattern of the frames inside the ROI.
* Two exposure controls (a "gain" slider each) tune the virtual image and the
  diffraction image independently; the diffraction panel also has a log toggle.
* Efficiency: the virtual image (a full-cube pass) is computed once in a worker
  thread and cached.  Dragging only reads the small n x m window (one contiguous
  read per scan row for HDF5 via ``read_block``, a memmap slice otherwise) and
  the redraws are coalesced with a short debounce so motion events never queue.
"""

from __future__ import annotations

import threading

import numpy as np
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

from gui_app import display_prefs


def _radial_mask(H, W, r_in, r_out):
    """Boolean annulus mask (r_in <= r < r_out) about the frame centre —
    same convention as the Analysis tab's BF/HAADF detectors."""
    yy, xx = np.ogrid[:H, :W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (rr >= r_in) & (rr < r_out)


def _radial_profile(image):
    """Azimuthally-averaged intensity vs radius (px). Rings show up as
    peaks.  Same convention as eval_all._radial_profile."""
    H, W = image.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel().astype(int)
    vals = image.ravel().astype(np.float64)
    r_max = int(r.max())
    radial = np.zeros(r_max + 1)
    counts = np.zeros(r_max + 1)
    np.add.at(radial, r, vals)
    np.add.at(counts, r, 1)
    counts = np.maximum(counts, 1)
    return np.arange(r_max + 1), radial / counts


def _ring_azimuthal(image, r0, dr, n_bins=120):
    """Intensity vs azimuthal angle (deg) around the ring [r0-dr/2, r0+dr/2).
    Reveals the texture/spottiness of that ring.  Returns (angles, profile,
    n_pixels_in_ring)."""
    H, W = image.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    th = (np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0)
    sel = (rr >= r0 - dr / 2.0) & (rr < r0 + dr / 2.0)
    if not sel.any():
        return np.array([]), np.array([]), 0
    bins = np.floor(th[sel] / (360.0 / n_bins)).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    vals = image[sel].astype(np.float64)
    az = np.zeros(n_bins); cnt = np.zeros(n_bins)
    np.add.at(az, bins, vals)
    np.add.at(cnt, bins, 1)
    cnt = np.maximum(cnt, 1)
    angles = np.arange(n_bins) * (360.0 / n_bins)
    return angles, az / cnt, int(sel.sum())


class ViewingPanel(ctk.CTkFrame):
    """Interactive virtual-image + live-window diffraction viewer."""

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app

        # --- state ---
        self.cube = None
        self._loaded_path = None
        self._BF = None
        self._HA = None
        self._COMx = None               # per-scan COMx (cols) — for iCOM/DPC
        self._COMy = None               # per-scan COMy (rows)
        self._dp = None                 # current mean diffraction (H, W)
        self._drag = False
        self._pending = None            # pending (y0, y1, x0, x1)
        self._diff_pending = False
        self._center = None             # last window centre (row, col)
        # measurement (distance between peaks) on the diffraction panel
        self._measure_on = False
        self._measure_pts = []
        self._measure_artists = []
        # open profile popups that track the moving window live
        self._live_profiles = []

        # --- vars ---
        self.kind = ctk.StringVar(value="BF")
        self.n_var = ctk.StringVar(value="8")       # window rows
        self.m_var = ctk.StringVar(value="8")       # window cols
        self.vi_gain = ctk.DoubleVar(value=1.0)     # virtual-image exposure
        self.dp_gain = ctk.DoubleVar(value=1.0)     # diffraction exposure
        self.dp_log = ctk.BooleanVar(value=True)
        self.calib_var = ctk.StringVar(value="")    # reciprocal Å⁻¹ / px
        self.ring_r = ctk.StringVar(value="")       # ring radius (px)
        self.ring_dr = ctk.StringVar(value="4")     # ring width (px)
        self.diff_cmap_var = ctk.StringVar(
            value=display_prefs.get_diff_cmap_name())

        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self):
        bar = ctk.CTkFrame(self)
        bar.pack(side="top", fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(bar, text="Virtual image:").pack(side="left",
                                                      padx=(8, 4))
        ctk.CTkSegmentedButton(bar, values=["BF", "HAADF"],
                               variable=self.kind,
                               command=lambda _v: self._render_virtual()
                               ).pack(side="left")

        ctk.CTkLabel(bar, text="   window  n×m:").pack(side="left",
                                                       padx=(12, 2))
        ctk.CTkEntry(bar, textvariable=self.n_var, width=48).pack(side="left")
        ctk.CTkLabel(bar, text="×").pack(side="left", padx=2)
        ctk.CTkEntry(bar, textvariable=self.m_var, width=48).pack(side="left")
        ctk.CTkButton(bar, text="apply", width=54,
                      command=self._on_size_change).pack(side="left",
                                                         padx=(4, 0))

        ctk.CTkButton(bar, text="Refresh from dataset", width=160,
                      command=lambda: self.on_show(force=True)
                      ).pack(side="right", padx=8)

        # exposure row
        exp = ctk.CTkFrame(self)
        exp.pack(side="top", fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(exp, text="Virtual exposure:").pack(side="left",
                                                         padx=(8, 4))
        ctk.CTkSlider(exp, from_=0.1, to=4.0, variable=self.vi_gain,
                      width=180,
                      command=lambda _v: self._render_virtual_image_only()
                      ).pack(side="left")
        ctk.CTkLabel(exp, text="     Diffraction exposure:").pack(
            side="left", padx=(16, 4))
        ctk.CTkSlider(exp, from_=0.1, to=4.0, variable=self.dp_gain,
                      width=180,
                      command=lambda _v: self._render_diff()
                      ).pack(side="left")
        ctk.CTkCheckBox(exp, text="log", variable=self.dp_log,
                        command=self._render_diff).pack(side="left",
                                                        padx=(16, 8))
        ctk.CTkLabel(exp, text="colormap:").pack(side="left", padx=(12, 2))
        ctk.CTkOptionMenu(
            exp, variable=self.diff_cmap_var, values=display_prefs.DIFF_CMAPS,
            width=110,
            command=lambda v: display_prefs.set_diff_cmap_name(v)
            ).pack(side="left", padx=2)
        display_prefs.subscribe(self._on_diff_cmap_change)

        # --- measurement / analysis tools (operate on the window's mean DP) ---
        tools = ctk.CTkFrame(self)
        tools.pack(side="top", fill="x", padx=8, pady=(0, 4))
        self._measure_btn = ctk.CTkButton(
            tools, text="Measure distance: OFF", width=170,
            command=self._toggle_measure)
        self._measure_btn.pack(side="left", padx=(8, 4))
        ctk.CTkLabel(tools, text="Å⁻¹/px:").pack(side="left", padx=(8, 2))
        ctk.CTkEntry(tools, textvariable=self.calib_var, width=76).pack(
            side="left")
        ctk.CTkButton(tools, text="Radial profile", width=110,
                      command=self._radial_popup).pack(side="left",
                                                       padx=(14, 4))
        ctk.CTkLabel(tools, text="ring r:").pack(side="left", padx=(8, 2))
        ctk.CTkEntry(tools, textvariable=self.ring_r, width=52).pack(
            side="left")
        ctk.CTkLabel(tools, text="dr:").pack(side="left", padx=(4, 2))
        ctk.CTkEntry(tools, textvariable=self.ring_dr, width=44).pack(
            side="left")
        ctk.CTkButton(tools, text="Azimuthal @ring", width=120,
                      command=self._azimuthal_popup).pack(side="left",
                                                         padx=(4, 4))
        ctk.CTkButton(tools, text="iCOM (DPC)", width=100,
                      command=self._icom_popup).pack(side="right", padx=8)

        # figures — two independent canvases so dragging the window only
        # redraws the (cheap) virtual-image canvas and the diffraction
        # canvas updates on its own debounced schedule.
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=8, pady=(0, 8))

        left = ctk.CTkFrame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self._fig_v = Figure(figsize=(4.6, 4.6), dpi=100)
        self._ax_v = self._fig_v.add_subplot(111)
        self._ax_v.set_xticks([]); self._ax_v.set_yticks([])
        self._im_v = None
        self._rect = None
        self._canvas_v = FigureCanvasTkAgg(self._fig_v, master=left)
        self._canvas_v.get_tk_widget().pack(fill="both", expand=True)

        right = ctk.CTkFrame(body)
        right.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self._fig_d = Figure(figsize=(4.6, 4.6), dpi=100)
        self._ax_d = self._fig_d.add_subplot(111)
        self._ax_d.set_xticks([]); self._ax_d.set_yticks([])
        self._im_d = None
        self._canvas_d = FigureCanvasTkAgg(self._fig_d, master=right)
        self._canvas_d.get_tk_widget().pack(fill="both", expand=True)

        # status line
        self._status = ctk.CTkLabel(self, text="", anchor="w")
        self._status.pack(side="bottom", fill="x", padx=10, pady=(0, 6))

        # mouse interaction (press-hold-drag on the virtual image)
        self._canvas_v.mpl_connect("button_press_event", self._on_press)
        self._canvas_v.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas_v.mpl_connect("button_release_event", self._on_release)
        # click-to-measure on the diffraction panel
        self._canvas_d.mpl_connect("button_press_event", self._on_diff_press)

        self._set_status("Load a cube in the Pre-processing tab, then click "
                         "“Refresh from dataset”.")

    def _set_status(self, msg):
        try:
            self._status.configure(text=msg)
        except Exception:
            pass

    # -------------------------------------------------------------- cube I/O
    def _grab_cube(self):
        pre = getattr(self.app, "pre", None)
        cube = getattr(pre, "cube", None) if pre is not None else None
        path = None
        try:
            path = pre.get_loaded_path() if pre is not None else None
        except Exception:
            path = None
        return cube, path

    def on_show(self, force=False):
        """Called when the Viewing tab becomes visible (and on Refresh).
        Loads the virtual image for the currently-loaded cube if needed."""
        cube, path = self._grab_cube()
        if cube is None:
            self._set_status("No cube loaded — load one in the "
                             "Pre-processing tab first.")
            return
        if (not force) and path == self._loaded_path and self._BF is not None:
            return
        self.cube = cube
        self._loaded_path = path
        self._BF = None; self._HA = None
        self._compute_virtual_async()

    def _compute_virtual_async(self):
        cube = self.cube
        if cube is None:
            return
        R, C, H, W = cube.shape
        self._set_status(f"Computing virtual images  ({R}×{C} scan, "
                         f"{H}×{W} frames) — one-time …")

        def work():
            try:
                bm = _radial_mask(H, W, 0.0, 0.06 * H)
                ham = _radial_mask(H, W, 0.18 * H, 0.45 * H)
                BF = np.zeros((R, C), dtype=np.float32)
                HA = np.zeros((R, C), dtype=np.float32)
                CX = np.zeros((R, C), dtype=np.float32)
                CY = np.zeros((R, C), dtype=np.float32)
                yc = (H - 1) / 2.0; xc = (W - 1) / 2.0
                yr = np.arange(H, dtype=np.float32)
                xr = np.arange(W, dtype=np.float32)
                for y in range(R):
                    block = np.asarray(cube[y]).astype(np.float32)
                    BF[y] = (block * bm).sum(axis=(1, 2))
                    HA[y] = (block * ham).sum(axis=(1, 2))
                    # centre-of-mass per frame (vectorised over the row)
                    tot = block.sum(axis=(1, 2))
                    safe = np.maximum(tot, 1e-12)
                    py = (block.sum(axis=2) * yr[None, :]).sum(axis=1)
                    px = (block.sum(axis=1) * xr[None, :]).sum(axis=1)
                    CY[y] = py / safe - yc
                    CX[y] = px / safe - xc
                    if (y % 16 == 0) or (y == R - 1):
                        self.after(0, self._set_status,
                                   f"Computing virtual images … "
                                   f"{y + 1}/{R} rows")
            except Exception as e:            # noqa: BLE001
                self.after(0, self._set_status, f"Virtual image failed: {e}")
                return
            self._BF = BF; self._HA = HA
            self._COMx = CX; self._COMy = CY
            self.after(0, self._on_virtual_ready)

        threading.Thread(target=work, daemon=True).start()

    def _on_virtual_ready(self):
        self._render_virtual()
        # default the window to the scan centre and render its diffraction
        R, C = self._BF.shape
        self._center = (R // 2, C // 2)
        self._flush_diff(immediate=True)
        self._set_status("Ready — click + hold on the virtual image to move "
                         "the window.")

    # --------------------------------------------------------- rendering
    def _current_virtual(self):
        return self._HA if self.kind.get() == "HAADF" else self._BF

    @staticmethod
    def _stretch(img, gain, log=False):
        x = np.asarray(img, dtype=np.float32)
        if log:
            x = np.log1p(np.clip(x, 0, None))
        base = np.percentile(x, 99.5)
        if not np.isfinite(base) or base <= 0:
            base = float(x.max()) or 1.0
        return np.clip(x / base * float(gain), 0.0, 1.0)

    def _render_virtual(self):
        img = self._current_virtual()
        if img is None:
            return
        disp = self._stretch(img, self.vi_gain.get())
        if self._im_v is None:
            self._im_v = self._ax_v.imshow(disp, cmap="gray", vmin=0, vmax=1,
                                           aspect="equal",
                                           interpolation="nearest")
        else:
            self._im_v.set_data(disp)
        self._ax_v.set_title(
            f"Virtual {self.kind.get()}", fontsize=10)
        self._ensure_rect()
        self._canvas_v.draw_idle()

    def _render_virtual_image_only(self):
        """Re-stretch the virtual image (exposure slider) without touching
        the window rectangle."""
        img = self._current_virtual()
        if img is None or self._im_v is None:
            self._render_virtual(); return
        self._im_v.set_data(self._stretch(img, self.vi_gain.get()))
        self._canvas_v.draw_idle()

    def _ensure_rect(self):
        if self._center is None:
            return
        n, m = self._win_size()
        r0, c0 = self._clamp_topleft(self._center[0], self._center[1], n, m)
        if self._rect is None:
            self._rect = Rectangle((c0 - 0.5, r0 - 0.5), m, n,
                                   fill=False, edgecolor="#ff3b3b", lw=1.6)
            self._ax_v.add_patch(self._rect)
        else:
            self._rect.set_xy((c0 - 0.5, r0 - 0.5))
            self._rect.set_width(m); self._rect.set_height(n)

    def _render_diff(self):
        if self._dp is None:
            return
        disp = self._stretch(self._dp, self.dp_gain.get(),
                             log=bool(self.dp_log.get()))
        cmap = display_prefs.get_diff_cmap_name()
        if self._im_d is None:
            self._im_d = self._ax_d.imshow(disp, cmap=cmap, vmin=0,
                                           vmax=1, aspect="equal",
                                           interpolation="nearest")
        else:
            self._im_d.set_data(disp)
            self._im_d.set_cmap(cmap)
        self._ax_d.set_title("Mean diffraction — window", fontsize=10)
        self._canvas_d.draw_idle()

    def _on_diff_cmap_change(self):
        """Diffraction colormap changed (here or in another view) — sync the
        dropdown and re-render."""
        try:
            self.diff_cmap_var.set(display_prefs.get_diff_cmap_name())
        except Exception:
            pass
        self._render_diff()

    # ---------------------------------------------------------- geometry
    def _win_size(self):
        def _read(var, default):
            try:
                v = int(float(var.get()))
                return max(1, v)
            except Exception:
                return default
        R = C = 10 ** 9
        if self._BF is not None:
            R, C = self._BF.shape
        n = min(_read(self.n_var, 8), R)
        m = min(_read(self.m_var, 8), C)
        return n, m

    def _clamp_topleft(self, cy, cx, n, m):
        R, C = self._BF.shape
        r0 = int(round(cy)) - n // 2
        c0 = int(round(cx)) - m // 2
        r0 = max(0, min(r0, R - n))
        c0 = max(0, min(c0, C - m))
        return r0, c0

    def _on_size_change(self):
        self._ensure_rect()
        self._canvas_v.draw_idle()
        self._flush_diff(immediate=True)

    # ---------------------------------------------------------- mouse
    def _on_press(self, event):
        if event.inaxes is not self._ax_v or event.xdata is None:
            return
        self._drag = True
        self._move_to(event.xdata, event.ydata)

    def _on_motion(self, event):
        if not self._drag or event.inaxes is not self._ax_v:
            return
        if event.xdata is None:
            return
        self._move_to(event.xdata, event.ydata)

    def _on_release(self, event):
        if self._drag:
            self._drag = False
            self._flush_diff(immediate=True)   # final full-quality render

    def _move_to(self, xdata, ydata):
        if self._BF is None:
            return
        self._center = (ydata, xdata)
        # move the rectangle immediately (cheap) …
        self._ensure_rect()
        self._canvas_v.draw_idle()
        # … and schedule a debounced diffraction update.
        self._flush_diff(immediate=False)

    # ---------------------------------------------------------- diff update
    def _flush_diff(self, immediate=False):
        if self.cube is None or self._center is None:
            return
        n, m = self._win_size()
        r0, c0 = self._clamp_topleft(self._center[0], self._center[1], n, m)
        self._pending = (r0, r0 + n, c0, c0 + m)
        if immediate:
            self._do_diff()
            return
        if not self._diff_pending:
            self._diff_pending = True
            self.after(15, self._do_diff)

    def _do_diff(self):
        self._diff_pending = False
        win = self._pending
        if win is None or self.cube is None:
            return
        try:
            self._dp = self._read_window_mean(*win)
        except Exception as e:               # noqa: BLE001
            self._set_status(f"window read failed: {e}")
            return
        # a distance measured on the previous window is now stale
        if self._measure_artists or self._measure_pts:
            self._clear_measure()
        self._render_diff()
        self._update_live_profiles()

    def _read_window_mean(self, r0, r1, c0, c1):
        cube = self.cube
        reader = getattr(cube, "read_block", None)
        if reader is not None:
            blk = np.asarray(reader(r0, r1 - r0, c0, c1 - c0),
                             dtype=np.float32)
        else:
            blk = np.asarray(cube[r0:r1, c0:c1], dtype=np.float32)
        # blk: (nrows, ncols, H, W) → mean over the scan window
        return blk.reshape(-1, blk.shape[-2], blk.shape[-1]).mean(axis=0)

    # ================================================================
    #  Measurement / analysis tools (on the window's mean diffraction)
    # ================================================================
    def _calib(self):
        """Reciprocal calibration Å⁻¹/px, or None if not set."""
        try:
            v = float(self.calib_var.get())
            return v if v > 0 else None
        except Exception:
            return None

    # ---- distance between peaks (click two points on the DP) ----
    def _toggle_measure(self):
        self._measure_on = not self._measure_on
        self._measure_btn.configure(
            text=f"Measure distance: {'ON' if self._measure_on else 'OFF'}")
        if not self._measure_on:
            self._clear_measure()
        else:
            self._set_status("Click two peaks on the diffraction image.")

    def _clear_measure(self):
        for a in self._measure_artists:
            try: a.remove()
            except Exception: pass
        self._measure_artists = []
        self._measure_pts = []
        try: self._canvas_d.draw_idle()
        except Exception: pass

    def _on_diff_press(self, event):
        if (not self._measure_on or event.inaxes is not self._ax_d
                or event.xdata is None):
            return
        if len(self._measure_pts) >= 2:
            self._clear_measure()
        x, y = float(event.xdata), float(event.ydata)
        self._measure_pts.append((x, y))
        mk, = self._ax_d.plot([x], [y], marker="+", ms=11, mew=1.6,
                              color="#39ff14")
        self._measure_artists.append(mk)
        if len(self._measure_pts) == 2:
            (x0, y0), (x1, y1) = self._measure_pts
            dpx = float(np.hypot(x1 - x0, y1 - y0))
            ln, = self._ax_d.plot([x0, x1], [y0, y1], "-", lw=1.3,
                                  color="#39ff14")
            self._measure_artists.append(ln)
            lab = self._ax_d.annotate(f"{dpx:.1f} px",
                                      ((x0 + x1) / 2, (y0 + y1) / 2),
                                      color="#39ff14", fontsize=8,
                                      ha="center", va="bottom")
            self._measure_artists.append(lab)
            cal = self._calib()
            msg = f"peak distance = {dpx:.2f} px"
            if cal:
                q = dpx * cal
                msg += f"  =  {q:.4f} Å⁻¹"
                if q > 0:
                    msg += f"   (d = {1.0 / q:.3f} Å)"
            self._set_status(msg)
        self._canvas_d.draw_idle()

    # ---- popups ----
    def _make_popup(self, title, figsize):
        win = ctk.CTkToplevel(self)
        win.title(title)
        try:
            win.after(200, lambda: (win.lift(), win.focus_force()))
        except Exception:
            pass
        fig = Figure(figsize=figsize, dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return win, fig, canvas

    def _register_live(self, win, update):
        """Track a profile popup so it refreshes as the window is dragged.
        Auto-unregisters when the popup is closed."""
        entry = {"win": win, "update": update}

        def _on_close():
            try: self._live_profiles.remove(entry)
            except Exception: pass
            try: win.destroy()
            except Exception: pass
        try:
            win.protocol("WM_DELETE_WINDOW", _on_close)
        except Exception:
            pass
        self._live_profiles.append(entry)

    def _update_live_profiles(self):
        """Refresh every open profile popup from the current window DP.
        Called whenever the diffraction window changes."""
        if not self._live_profiles:
            return
        keep = []
        for entry in self._live_profiles:
            try:
                if not entry["win"].winfo_exists():
                    continue
            except Exception:
                continue
            try:
                entry["update"]()
            except Exception:
                pass
            keep.append(entry)
        self._live_profiles = keep

    def _radial_popup(self):
        if self._dp is None:
            self._set_status("Move the window first — no diffraction yet.")
            return
        win, fig, canvas = self._make_popup(
            "Radial profile — window mean DP  (live)", (6.4, 4.2))
        ax = fig.add_subplot(111)
        (line,) = ax.plot([], [], color="#1f77b4")
        ax.set_yscale("log")                 # log-y, linear-x (house style)
        ax.set_xlabel("radius (px)")
        ax.set_ylabel("mean intensity (log)")
        ax.set_title("Radial profile — rings are peaks (live)")
        cal = self._calib()
        if cal:
            ax.secondary_xaxis(
                "top", functions=(lambda x: x * cal, lambda q: q / cal)
            ).set_xlabel("q (Å⁻¹)")
        fig.tight_layout()

        def update():
            if self._dp is None:
                return
            r, prof = _radial_profile(self._dp)
            line.set_data(r, np.maximum(prof, 1e-9))
            ax.relim(); ax.autoscale_view()
            canvas.draw_idle()

        update()
        self._register_live(win, update)

    def _azimuthal_popup(self):
        if self._dp is None:
            self._set_status("Move the window first — no diffraction yet.")
            return
        win, fig, canvas = self._make_popup(
            "Azimuthal profile — ring  (live)", (6.4, 4.2))
        ax = fig.add_subplot(111)
        (line,) = ax.plot([], [], color="#d62728")
        ax.set_xlabel("azimuthal angle (deg)")
        ax.set_ylabel("mean intensity")
        ax.set_xlim(0, 360)
        fig.tight_layout()

        def update():
            if self._dp is None:
                return
            try:
                r0 = float(self.ring_r.get())
            except Exception:
                self._set_status("Enter a ring radius (px) in 'ring r'.")
                return
            try:
                dr = max(1.0, float(self.ring_dr.get()))
            except Exception:
                dr = 4.0
            angles, prof, npix = _ring_azimuthal(self._dp, r0, dr)
            if npix == 0:
                ax.set_title(f"ring r={r0:.0f}±{dr/2:.0f} px is empty")
                line.set_data([], [])
                canvas.draw_idle()
                return
            line.set_data(angles, prof)
            ax.relim(); ax.autoscale_view(scalex=False)
            cal = self._calib()
            rtxt = f"r = {r0:.0f} px"
            if cal:
                rtxt += f"  (q = {r0 * cal:.4f} Å⁻¹)"
            ax.set_title(f"Intensity around the ring   [{rtxt}, {npix} px] "
                         f"(live)")
            canvas.draw_idle()

        update()
        self._register_live(win, update)

    # ---- iCOM / DPC (integrated centre of mass over the whole scan) ----
    @staticmethod
    def _integrate_icom(comx, comy):
        """Integrate the COM vector field to a phase-like iCOM image via a
        Fourier-space least-squares solve (same idea as py4DSTEM's DPC /
        Dectris' iCOM)."""
        comx = np.nan_to_num(comx - np.nanmean(comx))
        comy = np.nan_to_num(comy - np.nanmean(comy))
        Ny, Nx = comx.shape
        kx = (2 * np.pi * np.fft.fftfreq(Nx))[None, :]
        ky = (2 * np.pi * np.fft.fftfreq(Ny))[:, None]
        k2 = kx ** 2 + ky ** 2
        k2[0, 0] = 1.0
        gx = np.fft.fft2(comx)
        gy = np.fft.fft2(comy)
        phi = np.fft.ifft2(-1j * (kx * gx + ky * gy) / k2).real
        return phi

    def _icom_popup(self):
        if self._COMx is None or self._COMy is None:
            self._set_status("Virtual image not ready — click Refresh first.")
            return
        cx, cy = self._COMx, self._COMy
        cmag = np.sqrt(cx ** 2 + cy ** 2)
        icom = self._integrate_icom(cx, cy)
        _, fig, canvas = self._make_popup(
            "iCOM / DPC reconstruction", (9.0, 8.0))
        panels = [
            (cx,   "COM$_x$ (DPC x)", "RdBu_r", True),
            (cy,   "COM$_y$ (DPC y)", "RdBu_r", True),
            (cmag, "COM magnitude",   "inferno", False),
            (icom, "iCOM (integrated)", "gray",  False),
        ]
        for i, (arr, title, cmap, sym) in enumerate(panels, start=1):
            ax = fig.add_subplot(2, 2, i)
            if sym:
                v = np.percentile(np.abs(arr), 99) or 1.0
                im = ax.imshow(arr, cmap=cmap, vmin=-v, vmax=v,
                               interpolation="nearest", aspect="equal")
            else:
                lo, hi = np.percentile(arr, [1, 99])
                im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi,
                               interpolation="nearest", aspect="equal")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle("Centre-of-mass / iCOM  (whole scan)", fontsize=12)
        fig.tight_layout()
        canvas.draw()
