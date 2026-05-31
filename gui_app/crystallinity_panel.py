"""crystallinity_panel.py -- per-position peak/background ratio map.

For each scan position, computes the ratio of detected peak intensity
to non-peak (background / ring) intensity inside a user-chosen radial
window [r_min, r_max] (1/Å).  Output is a 2D virtual map (Ny × Nx)
that behaves like a virtual HAADF but specifically measures
*crystallinity in a chosen q-range*.

Two-part interactive parameter-tuning UI (the user sees both for a
single test pattern before committing):
  - polar view + r-window highlight (which q-range is being sampled)
  - cart pattern with the detected peaks overlaid (red rings inside
    the window, grey outside)

Plus a 1-D radial profile with the window shaded.

Then **Run on full dataset** applies the same detection params +
r-window to every scan position and renders the ratio map in the
bottom-right.  Higher ratio = more Bragg-like signal vs diffuse ring
background in that q-window = more crystalline.
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

        # 4. DETECTION params
        _section_header(sidebar, "4.  Peak detection")
        self._det_thr = ctk.DoubleVar(value=0.02)
        self._det_min = ctk.DoubleVar(value=1.0)
        self._det_max = ctk.DoubleVar(value=6.0)
        self._det_num = ctk.IntVar(value=6)
        self._det_log = ctk.BooleanVar(value=True)
        for label, var, w in (("threshold", self._det_thr, 60),
                                  ("min sigma", self._det_min, 60),
                                  ("max sigma", self._det_max, 60),
                                  ("num sigma", self._det_num, 60)):
            row = ctk.CTkFrame(sidebar, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=1)
            ctk.CTkLabel(row, text=label, width=80,
                          anchor="w").pack(side="left")
            e = ctk.CTkEntry(row, textvariable=var, width=w)
            e.pack(side="left", padx=2)
            e.bind("<Return>", lambda _e: self._recompute_test())
        ctk.CTkCheckBox(sidebar, text="log stretch (detection)",
                          variable=self._det_log,
                          command=self._recompute_test
                          ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Re-detect ▶",
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
        ctk.CTkButton(sidebar, text="Run on full dataset ▶",
                       width=240,
                       fg_color=("#2D7A2D", "#1F7A1F"),
                       command=self._run_full_dataset
                       ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Per-cluster ratios  (bar chart)",
                       width=240,
                       command=self._render_per_cluster
                       ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Save map  (npy + png)",
                       width=240,
                       command=self._save_map
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

    def _build_axes(self):
        self._fig.clf()
        gs = self._fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)
        self._ax_cart  = self._fig.add_subplot(gs[0, 0])
        self._ax_polar = self._fig.add_subplot(gs[0, 1])
        self._ax_1d    = self._fig.add_subplot(gs[1, 0])
        self._ax_map   = self._fig.add_subplot(gs[1, 1])
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

    def _detect_kw(self):
        return dict(
            min_sigma=float(self._det_min.get()),
            max_sigma=float(self._det_max.get()),
            num_sigma=int(self._det_num.get()),
            threshold=float(self._det_thr.get()),
            log_stretch=bool(self._det_log.get()),
        )

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
            ratio, peaks_w, peak_sum, bg_sum, annulus = (
                compute_crystallinity_ratio(
                    self._test_pattern, r_min_px, r_max_px,
                    self._detect_kw(), self._test_center))
        except Exception as e:
            self._test_status.configure(text=f"err: {e!r}"); return
        self._test_result = dict(
            ratio=ratio, peaks_w=peaks_w,
            peak_sum=peak_sum, bg_sum=bg_sum,
            annulus=annulus,
            r_min_px=r_min_px, r_max_px=r_max_px,
            inv_a=inv_a)
        self._test_status.configure(
            text=f"ratio = {ratio:.4f}    peak={peak_sum:.3g}  "
                  f"bg={bg_sum:.3g}    "
                  f"{len(peaks_w)} peaks in window")
        self._render_test_panels()

    def _render_test_panels(self):
        if self._test_pattern is None or self._test_result is None:
            return
        r = self._test_result
        cy, cx = self._test_center
        # ---- cart pattern + annulus + peaks ----
        ax = self._ax_cart; ax.clear()
        img = np.log1p(np.clip(self._test_pattern, 0, None))
        ax.imshow(img, cmap="inferno", aspect="equal",
                    interpolation="nearest")
        # annulus rings
        for rad, color in ((r["r_min_px"], "#33ddff"),
                              (r["r_max_px"], "#33ddff")):
            ax.add_patch(Circle((cx, cy), rad, color=color,
                                  fill=False, lw=1.5, linestyle="--"))
        # peaks: in-window cyan filled, out-of-window grey ring only
        try:
            from gui_app.acom_core import detect_peaks_2d
            all_peaks = detect_peaks_2d(self._test_pattern,
                                              **self._detect_kw())
        except Exception:
            all_peaks = np.zeros((0, 3))
        if all_peaks.size:
            radii = np.sqrt(
                (all_peaks[:, 0] - cy) ** 2
                + (all_peaks[:, 1] - cx) ** 2)
            in_win = (radii >= r["r_min_px"]) & (radii < r["r_max_px"])
            if (~in_win).any():
                p = all_peaks[~in_win]
                ax.scatter(p[:, 1], p[:, 0], s=40,
                            facecolors="none", edgecolors="#888",
                            linewidths=0.7)
            if in_win.any():
                p = all_peaks[in_win]
                ax.scatter(p[:, 1], p[:, 0], s=60,
                            facecolors="none", edgecolors="cyan",
                            linewidths=1.4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{self._test_origin}    "
                       f"ratio = {r['ratio']:.4f}", fontsize=10)
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
        polar = self._test_polar
        ax.imshow(np.log1p(np.clip(polar, 0, None)),
                    cmap="inferno", aspect="auto",
                    interpolation="nearest")
        n_theta, n_r = polar.shape
        # window x-pixels in polar frame: r-axis is 0..n_r-1 px
        inv_a = r["inv_a"]
        scale_x = n_r / max(self._test_pattern.shape[0] / 2, 1)
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

        # ---- 1D radial profile + window shaded ----
        ax = self._ax_1d; ax.clear()
        H, W = self._test_pattern.shape
        yy, xx = np.indices((H, W))
        rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        n_bins = int(min(H, W) // 2)
        bins = np.linspace(0.0, n_bins, n_bins + 1)
        ind = np.digitize(rad.ravel(), bins) - 1
        flat = self._test_pattern.ravel().astype(np.float64)
        sums = np.bincount(ind, weights=flat,
                              minlength=n_bins + 2)[:n_bins + 1]
        cnts = np.bincount(ind, minlength=n_bins + 2)[:n_bins + 1]
        means = sums / np.maximum(cnts, 1)
        rc = 0.5 * (bins[:-1] + bins[1:])
        rc_inva = rc * inv_a
        ax.semilogy(rc_inva, np.clip(means[:-1] -
                                        np.median(means), 0, None) + 1e-3,
                       color="#1f77b4", lw=1.3)
        ax.axvspan(self._r_min.get(), self._r_max.get(),
                     color="#33ddff", alpha=0.20)
        # detected peaks shown as red dots at their q-values
        if r["peaks_w"].size:
            pq = np.sqrt(
                (r["peaks_w"][:, 0] - cy) ** 2
                + (r["peaks_w"][:, 1] - cx) ** 2) * inv_a
            pa = np.zeros_like(pq) + (means.max() * 0.6)
            ax.scatter(pq, pa, s=22, color="#e0144c",
                          edgecolors="white", linewidths=0.6)
        ax.set_xlabel("q  (1/Å)")
        ax.set_yticks([])
        ax.set_title("1D radial  (blue) + detected-peak q's (red)",
                       fontsize=10)

        # ---- map (placeholder until Run) ----
        ax = self._ax_map; ax.clear()
        if self._cryst_map is not None:
            im = ax.imshow(self._cryst_map, cmap="viridis",
                              interpolation="nearest", aspect="equal")
            ax.set_title("crystallinity map", fontsize=10)
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
            detect_kw = self._detect_kw()
            cmap = np.full((Ny, Nx), np.nan, dtype=np.float32)
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
                        ratio, _, _, _, _ = (
                            compute_crystallinity_ratio(
                                pat, r_min_px, r_max_px,
                                detect_kw, center))
                        cmap[rx, ry] = ratio
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
            # Stride > 1: fill skipped positions with nearest sampled
            # value (cheap visual interpolation).
            if stride > 1:
                from scipy.ndimage import maximum_filter
                # propagate by nearest-neighbour fill via a small
                # iterative dilation
                cur = cmap.copy()
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
                cmap = cur
            self._cryst_map = cmap
            dt = time.time() - t0
            self.after(0, lambda: self._full_status.configure(
                text=f"done ({dt:.0f}s)  stride={stride}  "
                      f"min/median/max = "
                      f"{np.nanmin(cmap):.3f} / "
                      f"{np.nanmedian(cmap):.3f} / "
                      f"{np.nanmax(cmap):.3f}"))
            self.after(0, self._render_test_panels)
            # If a trained run is linked, also aggregate the ratio
            # per DINO cluster and show a bar chart.
            self.after(0, self._render_per_cluster)
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "Run on full dataset", err))
        finally:
            self._busy = False

    # ------------------------------------------------------------------
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
        ratio = self._cryst_map
        rows, means, meds, q1s, q3s, ns, ks = [], [], [], [], [], [], []
        for c in range(K):
            vals = ratio[(assigns == c) & np.isfinite(ratio)]
            if vals.size == 0:
                continue
            ks.append(c); ns.append(int(vals.size))
            means.append(float(np.mean(vals)))
            meds.append(float(np.median(vals)))
            q1s.append(float(np.percentile(vals, 25)))
            q3s.append(float(np.percentile(vals, 75)))
        if not ks:
            messagebox.showinfo("Per-cluster",
                "No overlap between ratio map and class map."); return
        # Sort by median ratio (most crystalline first).
        order = np.argsort(-np.asarray(meds))
        ks = [ks[i] for i in order]; means = [means[i] for i in order]
        meds = [meds[i] for i in order]; ns = [ns[i] for i in order]
        q1s = [q1s[i] for i in order]; q3s = [q3s[i] for i in order]
        # Bar chart in its own popup window.
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        win = tk.Toplevel(self)
        win.title("crystallinity ratio per DINO cluster")
        win.geometry("780x520")
        fig = Figure(figsize=(7.6, 5.0), dpi=110, facecolor="white")
        ax = fig.add_subplot(111)
        x = np.arange(len(ks))
        yerr = [np.array(meds) - np.array(q1s),
                np.array(q3s) - np.array(meds)]
        cmap = plt.get_cmap("tab10" if K <= 10 else "tab20")
        cols = [cmap(c % (10 if K <= 10 else 20)) for c in ks]
        ax.bar(x, meds, yerr=yerr, capsize=3, color=cols,
                  edgecolor="#333", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"p{c}\n{n}px" for c, n in zip(ks, ns)],
                              fontsize=8)
        ax.set_ylabel("peak/background ratio  (median ± IQR)")
        ax.set_title(
            f"crystallinity per cluster  "
            f"(q = {self._r_min.get():.3g}–{self._r_max.get():.3g} 1/Å)  "
            f"·  higher = more crystalline", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=win)
        c.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(c, win, pack_toolbar=False)
        tb.update(); tb.pack(side="bottom", fill="x")
        self._full_status.configure(
            text=f"per-cluster ratios computed for {len(ks)} classes "
                  f"(median range {min(meds):.3f}–{max(meds):.3f})")

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
        np.save(os.path.join(out, f"map_{stamp}.npy"),
                  self._cryst_map)
        # PNG
        fig, ax = matplotlib.pyplot.subplots(figsize=(7, 6))
        im = ax.imshow(self._cryst_map, cmap="viridis",
                          interpolation="nearest", aspect="equal")
        ax.set_title(f"crystallinity ratio map  "
                       f"(q = {self._r_min.get():.3g}–"
                       f"{self._r_max.get():.3g} 1/Å)",
                       fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        png = os.path.join(out, f"map_{stamp}.png")
        fig.savefig(png, dpi=180, bbox_inches="tight",
                       facecolor="white")
        matplotlib.pyplot.close(fig)
        self._full_status.configure(
            text=f"saved → {os.path.basename(out)}/map_{stamp}.npy + .png")
