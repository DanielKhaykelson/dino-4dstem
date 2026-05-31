"""acom_tab.py -- dedicated ACOM tab with step-by-step validation.

Replaces the in-line ACOM section that lived in the post-hoc sidebar.
The motivation: ACOM cannot be a black box.  Three things must be
visually validated BEFORE any batch run kicks off:

    1. SPOT DETECTION   — are the right Bragg peaks being found?
                           Tune blob_log params on one pattern and
                           confirm the overlay looks reasonable.
    2. PIXEL CALIBRATION — does the CIF's predicted ring set sit on
                           top of the experimental 1D radial?
                           Adjust 1/Å/px (or auto-fit) until they
                           align.
    3. GOODNESS OF FIT  — does match_orientations actually pick a
                           sensible orientation, with corr above
                           the floor and the fitted pattern visually
                           matching the experimental peaks?

Only after the three pass does the batch (class avgs / grains /
full dataset / multi-phase) run.

UI layout:
    LEFT  : scrollable steps (source, detection, CIF, match, batch).
    RIGHT : 2×2 panel —
              [pattern + detected peaks]   [1D radial + CIF rings]
              [single-pattern match fit]   [class map (source pick)]

The tab pulls live state (sample, scan_shape, inference, class avgs,
grain extraction) from `self.app.posthoc` so the user only has to
load a run once.
"""
from __future__ import annotations
import os, sys, json, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import matplotlib
matplotlib.use("TkAgg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                  NavigationToolbar2Tk)
from matplotlib.colors import ListedColormap

from data import SAMPLES, LoadPRZ


def _section_header(parent, text):
    """Underlined step header."""
    lbl = ctk.CTkLabel(parent, text=text,
                         font=("Segoe UI", 12, "bold"),
                         anchor="w")
    lbl.pack(anchor="w", padx=10, pady=(10, 0))
    sep = ctk.CTkFrame(parent, height=2, fg_color=("#cccccc", "#444444"))
    sep.pack(fill="x", padx=10, pady=(0, 4))


def _hint(parent, text):
    """Small grey explanatory line."""
    ctk.CTkLabel(parent, text=text,
                  font=("Segoe UI", 9),
                  text_color=("#666", "#aaa"),
                  wraplength=270, justify="left",
                  anchor="w").pack(anchor="w", padx=10, pady=(0, 2))


# ---------------------------------------------------------------------------

class ACOMTabPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        # current test pattern under validation
        self._test_pattern: "np.ndarray | None" = None
        self._test_origin: str = "(none)"           # human-readable
        self._test_center: "tuple[float, float] | None" = None
        # detected peaks on _test_pattern (Nx3: qx, qy, intensity)
        self._test_peaks: "np.ndarray | None" = None
        # phases (multi-CIF), keyed by name
        self._phases: list = []           # [(name, cif_path), ...]
        self._phase_crystals: dict = {}
        self._phase_keys: dict = {}
        # cached classmap for picking sources
        self._classmap_axes_xy = None      # last drawn (Ny, Nx)
        # cached dp_max / dp_mean (keyed by sample so they survive
        # source-switching without recomputing).
        self._cached_dp_max = {}    # sample → (H, W) np.float32
        self._cached_dp_mean = {}   # sample → (H, W) np.float32
        # CIF-ring index for hover tooltip: list of
        #   (q_invA, phase_name, (h,k,l), color)
        self._cif_ring_db: list = []
        # Matplotlib annotation handle (lazy).
        self._hover_annot = None
        # Stop-flag for the batch worker (set when user hits Stop).
        import threading as _t
        self._stop_event = _t.Event()
        self._busy = False
        self._build()

    # ------------------------------------------------------------------
    # API for parent app
    # ------------------------------------------------------------------
    def on_runtime_sample_added(self, key):
        pass

    def refresh_from_posthoc(self):
        """Back-compat shim — session subscription handles updates."""
        self._on_session_change(getattr(self.app, "session", None))

    def _on_session_change(self, sess):
        """Re-draw the classmap + status dots when the global session
        changes (sample / run_dir / inference)."""
        if sess is None: return
        try:
            self._dot_data.set(
                "ok" if sess.has_dataset() else "idle",
                sess.sample or "no dataset")
            self._dot_inf.set(
                "ok" if sess.has_inference() else "idle",
                "inference cached" if sess.has_inference()
                else "no inference (load a run)")
            self._dot_cif.set(
                "ok" if self._phase_crystals else "idle",
                f"{len(self._phase_crystals)} crystal(s) built"
                if self._phase_crystals else "no CIF built yet")
            self._draw_classmap_if_possible()
        except Exception: pass

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build(self):
        # Top status bar — three StatusDots reflecting the global
        # Session.  Replaces the old `_link_lbl` + manual refresh
        # button (session subscription auto-updates).
        from gui_app._ui import StatusDot
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)
        self._dot_data = StatusDot(top, label="dataset")
        self._dot_inf  = StatusDot(top, label="inference")
        self._dot_cif  = StatusDot(top, label="crystal")
        for d in (self._dot_data, self._dot_inf, self._dot_cif):
            d.pack(side="left", padx=10)
        ctk.CTkLabel(top,
            text="(uses the topbar session badges; no per-tab Load button)",
            font=("Segoe UI", 9), text_color=("#666", "#aaa")
            ).pack(side="right", padx=8)
        # Subscribe to global session.
        sess = getattr(self.app, "session", None)
        if sess is not None:
            sess.subscribe(self._on_session_change)

        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        # ---- LEFT column: scrollable controls ----
        sidebar = ctk.CTkScrollableFrame(body, width=320)
        sidebar.pack(side="left", fill="y")

        # === STEP 1: source ===========================================
        _section_header(sidebar, "1.  Pick a source")
        _hint(sidebar,
            "Choose ONE pattern to validate detection + calibration "
            "+ fit BEFORE batching.  Class avgs are computed from "
            "the post-hoc inference; grains/positions come from "
            "clicking the class map (bottom right).")
        self._source_var = ctk.StringVar(value="dp_max")
        src_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        src_row.pack(fill="x", padx=10, pady=2)
        for label, val in (
                ("dp_max  (per-pixel max — DATASET only)", "dp_max"),
                ("dp_mean (mean — DATASET only)", "dp_mean"),
                ("scan pos (y, x) — DATASET only", "scan_pos"),
                ("class avg  (needs trained run)", "class_avg"),
                ("grain @ click  (needs trained run)", "grain")):
            ctk.CTkRadioButton(src_row, text=label,
                                variable=self._source_var,
                                value=val
                                ).pack(anchor="w", padx=2)
        cls_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        cls_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(cls_row, text="class:", width=44,
                       anchor="w").pack(side="left")
        self._src_class = ctk.StringVar(value="0")
        ctk.CTkEntry(cls_row, textvariable=self._src_class,
                       width=40).pack(side="left", padx=2)
        ctk.CTkLabel(cls_row, text="(y,x):", width=44,
                       anchor="w").pack(side="left", padx=(8, 2))
        self._src_y = ctk.StringVar(value="64")
        self._src_x = ctk.StringVar(value="64")
        ctk.CTkEntry(cls_row, textvariable=self._src_y,
                       width=40).pack(side="left", padx=2)
        ctk.CTkEntry(cls_row, textvariable=self._src_x,
                       width=40).pack(side="left", padx=2)
        ctk.CTkButton(sidebar, text="Load source →",
                       width=240,
                       fg_color=("#4D6FB0", "#3A5380"),
                       command=self._load_source
                       ).pack(anchor="w", padx=10, pady=2)
        self._source_status = ctk.CTkLabel(sidebar,
            text="(no source loaded)",
            font=("Consolas", 9),
            text_color=("#666", "#aaa"),
            wraplength=290, justify="left", anchor="w")
        self._source_status.pack(anchor="w", padx=10, pady=(2, 4))

        # === STEP 2: detection ========================================
        _section_header(sidebar, "2.  Peak detection (live)")
        _hint(sidebar,
            "Adjust until the cyan rings sit on the diffraction "
            "spots and ignore noise.  The same params will be used "
            "in the batch run.")
        self._det_thr  = ctk.DoubleVar(value=0.02)
        self._det_min  = ctk.DoubleVar(value=1.5)
        self._det_max  = ctk.DoubleVar(value=8.0)
        self._det_num  = ctk.IntVar(value=6)
        self._det_log  = ctk.BooleanVar(value=True)
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
            e.bind("<Return>", lambda _e: self._detect_and_redraw())
        ctk.CTkCheckBox(sidebar, text="log stretch before detection",
                          variable=self._det_log,
                          command=self._detect_and_redraw
                          ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="Detect peaks ▶",
                       width=240,
                       command=self._detect_and_redraw
                       ).pack(anchor="w", padx=10, pady=2)
        self._det_status = ctk.CTkLabel(sidebar,
            text="(no peaks detected yet)",
            font=("Consolas", 9),
            text_color=("#666", "#aaa"),
            wraplength=290, justify="left", anchor="w")
        self._det_status.pack(anchor="w", padx=10, pady=(2, 4))

        # === STEP 3: CIF + calibration ================================
        _section_header(sidebar, "3.  CIF + pixel-size calibration")
        _hint(sidebar,
            "Add one or more CIFs (e.g. α + γ).  Build the crystal "
            "to compute structure factors.  Then nudge '1/Å per px' "
            "until the CIF rings (vertical lines) overlay the "
            "experimental 1D peaks on the top-right panel.")
        # phase list
        self._phase_list = tk.Listbox(sidebar, height=4,
                                          exportselection=False,
                                          font=("Consolas", 9))
        self._phase_list.pack(fill="x", padx=10, pady=2)
        phbtn = ctk.CTkFrame(sidebar, fg_color="transparent")
        phbtn.pack(fill="x", padx=10, pady=1)
        ctk.CTkButton(phbtn, text="+ CIF…", width=66,
                       command=self._add_phase).pack(side="left", padx=2)
        ctk.CTkButton(phbtn, text="− remove", width=74,
                       command=self._remove_phase).pack(side="left", padx=2)
        ctk.CTkButton(phbtn, text="Build all", width=88,
                       command=self._build_all_phases
                       ).pack(side="left", padx=2)
        ctk.CTkLabel(phbtn, text="show on 1D:",
                      width=88, anchor="e").pack(side="left", padx=2)
        kmax_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        kmax_row.pack(fill="x", padx=10, pady=1)
        ctk.CTkLabel(kmax_row, text="k_max (1/Å):",
                       width=92, anchor="w").pack(side="left")
        self._kmax = ctk.DoubleVar(value=2.0)
        ctk.CTkEntry(kmax_row, textvariable=self._kmax,
                       width=70).pack(side="left", padx=2)
        ctk.CTkLabel(kmax_row, text="plan:",
                       width=44, anchor="w").pack(side="left", padx=(8, 2))
        self._plan_mode = ctk.StringVar(value="corners")
        ctk.CTkOptionMenu(kmax_row, variable=self._plan_mode,
                            values=["corners", "fiber"], width=82
                            ).pack(side="left", padx=2)
        inv_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        inv_row.pack(fill="x", padx=10, pady=1)
        ctk.CTkLabel(inv_row, text="1/Å per px:",
                       width=92, anchor="w").pack(side="left")
        # default from topbar 0.0185 nm⁻¹/px → 0.00185 1/Å/px
        rp = self._recip_per_px() or 0.0185
        self._inv_ang = ctk.DoubleVar(value=round(rp * 0.1, 6))
        inv_e = ctk.CTkEntry(inv_row, textvariable=self._inv_ang,
                                width=80)
        inv_e.pack(side="left", padx=2)
        inv_e.bind("<Return>", lambda _e: self._redraw_1d_with_rings())
        ctk.CTkButton(inv_row, text="↺", width=28,
                       command=self._sync_calib_from_topbar
                       ).pack(side="left", padx=2)
        ctk.CTkButton(sidebar, text="Auto-fit 1/Å/px ▶",
                       width=240,
                       fg_color=("#4D6FB0", "#3A5380"),
                       command=self._auto_calibrate
                       ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sidebar, text="⤢ Expand 1D + CIF rings",
                       width=240,
                       command=self._open_1d_popup
                       ).pack(anchor="w", padx=10, pady=2)
        self._cif_status = ctk.CTkLabel(sidebar,
            text="(no phases built)",
            font=("Consolas", 9),
            text_color=("#666", "#aaa"),
            wraplength=290, justify="left", anchor="w")
        self._cif_status.pack(anchor="w", padx=10, pady=(2, 4))

        # === STEP 4: single-pattern match =============================
        _section_header(sidebar, "4.  Goodness of fit (single pattern)")
        _hint(sidebar,
            "Two ways to fit:  (a) Match single = per-phase "
            "match_orientations (returns corr + ZA per phase).  "
            "(b) NNLS multi-phase fit = py4DSTEM's "
            "CrystalPhase.quantify_single_pattern (returns phase "
            "weights + fit residual + reliability — the same "
            "output as the multi-phase Colab notebook).  Use (b) "
            "for the mixture question.")
        match_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        match_row.pack(fill="x", padx=10, pady=1)
        ctk.CTkButton(match_row, text="Match single ▶",
                       width=140,
                       command=self._run_single_match
                       ).pack(side="left", padx=2)
        ctk.CTkButton(match_row, text="NNLS multi-phase fit ▶",
                       width=160,
                       fg_color=("#4D6FB0", "#3A5380"),
                       command=self._run_nnls_phase_fit
                       ).pack(side="left", padx=2)
        ctk.CTkButton(match_row, text="Top-N",
                       width=70,
                       command=self._show_topN_matches
                       ).pack(side="left", padx=2)
        self._match_status = ctk.CTkLabel(sidebar,
            text="(no match yet)",
            font=("Consolas", 9),
            text_color=("#666", "#aaa"),
            wraplength=290, justify="left", anchor="w")
        self._match_status.pack(anchor="w", padx=10, pady=(2, 4))

        # === STEP 5: batch run ========================================
        _section_header(sidebar, "5.  Batch run  (only after 1–4)")
        _hint(sidebar,
            "Apply the validated detection params + calibration + "
            "phase set to every class / grain / scan position.  "
            "Multi-phase variants assign each item to the phase "
            "with max corr above the threshold (else 'neither').")
        thr_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        thr_row.pack(fill="x", padx=10, pady=1)
        ctk.CTkLabel(thr_row, text="corr threshold:",
                       width=110, anchor="w").pack(side="left")
        self._mp_threshold = ctk.DoubleVar(value=0.02)
        ctk.CTkEntry(thr_row, textvariable=self._mp_threshold,
                       width=60).pack(side="left", padx=2)
        mar_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        mar_row.pack(fill="x", padx=10, pady=1)
        ctk.CTkLabel(mar_row, text="winner margin:",
                       width=110, anchor="w").pack(side="left")
        self._mp_margin = ctk.DoubleVar(value=0.0)
        ctk.CTkEntry(mar_row, textvariable=self._mp_margin,
                       width=60).pack(side="left", padx=2)
        for label, mode in (("Class averages", "classes"),
                              ("Largest grain per class", "grains"),
                              ("Multi-phase on class avgs", "mp_classes"),
                              ("Multi-phase on grains", "mp_grains")):
            ctk.CTkButton(sidebar, text=label, width=240,
                           command=lambda m=mode: self._run_batch(m)
                           ).pack(anchor="w", padx=10, pady=2)
        full_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        full_row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(full_row, text="full-dataset stride:",
                       width=130, anchor="w").pack(side="left")
        self._full_stride = ctk.IntVar(value=4)
        ctk.CTkEntry(full_row, textvariable=self._full_stride,
                       width=40).pack(side="left", padx=2)
        full_btn = ctk.CTkFrame(sidebar, fg_color="transparent")
        full_btn.pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(full_btn, text="Single-phase full ▶",
                       width=140,
                       command=lambda: self._run_batch("full")
                       ).pack(side="left", padx=2)
        ctk.CTkButton(full_btn, text="Multi-phase full ▶",
                       width=140,
                       fg_color=("#A23BB0", "#7A2680"),
                       command=lambda: self._run_batch("mp_full")
                       ).pack(side="left", padx=2)
        # Stop / kill the in-progress batch run.  Sets the
        # threading.Event the worker polls between major steps.
        ctk.CTkButton(sidebar, text="■  STOP batch run",
                       width=240,
                       fg_color=("#a33", "#722"),
                       hover_color=("#c33", "#822"),
                       command=self._stop_batch_run
                       ).pack(anchor="w", padx=10, pady=2)

        self._status = ctk.CTkLabel(sidebar, text="",
            font=("Consolas", 9), text_color=("#444", "#aaa"),
            wraplength=290, justify="left", anchor="w")
        self._status.pack(anchor="w", padx=10, pady=(4, 8))

        # ---- RIGHT column: canvas with 2×2 panels ----
        canvas_holder = ctk.CTkFrame(body)
        canvas_holder.pack(side="left", fill="both", expand=True,
                            padx=(8, 0))
        tb_frame = tk.Frame(canvas_holder, bg="#f4f4f4")
        tb_frame.pack(side="top", fill="x")
        self._fig = Figure(figsize=(11, 8), dpi=95, facecolor="#f4f4f4")
        self._canvas = FigureCanvasTkAgg(self._fig, master=canvas_holder)
        self._canvas.get_tk_widget().pack(side="top", fill="both",
                                            expand=True)
        self._toolbar = NavigationToolbar2Tk(self._canvas, tb_frame,
                                               pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.pack(side="left")
        # Save-panel button
        ctk.CTkButton(tb_frame, text="Save panel (PDF+PNG)",
                       width=170, height=28,
                       command=self._save_panel).pack(side="left", padx=8)
        # Initial layout
        self._build_axes()
        self._redraw_all()

        # Classmap click handler
        self._click_cid = self._canvas.mpl_connect(
            "button_press_event", self._on_classmap_click)
        # Hover handler for CIF-ring (h,k,l) tooltip on the 1D panel.
        self._hover_cid = self._canvas.mpl_connect(
            "motion_notify_event", self._on_1d_hover)

    def _build_axes(self):
        self._fig.clf()
        gs = self._fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)
        self._ax_pat   = self._fig.add_subplot(gs[0, 0])
        self._ax_1d    = self._fig.add_subplot(gs[0, 1])
        self._ax_fit   = self._fig.add_subplot(gs[1, 0])
        self._ax_cmap  = self._fig.add_subplot(gs[1, 1])
        for ax, txt in ((self._ax_pat,  "(load a source — step 1)"),
                          (self._ax_1d,   "(needs source + CIF — steps 1+3)"),
                          (self._ax_fit,  "(needs single match — step 4)"),
                          (self._ax_cmap, "(load posthoc inference)")):
            ax.text(0.5, 0.5, txt, ha="center", va="center",
                     fontsize=10, color="#888", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])

    def _ensure_quad_layout(self):
        """A batch render does self._fig.clf(), destroying the 4 step
        axes.  Before any step-1..4 redraw, rebuild the 2×2 layout if
        the step axes are no longer attached to the figure (otherwise
        the step views draw onto dead axes and the canvas stays stuck
        on the batch output)."""
        ax = getattr(self, "_ax_pat", None)
        if ax is None or ax not in self._fig.axes:
            self._build_axes()
            return True
        return False

    def _redraw_all(self):
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # status / utility
    # ------------------------------------------------------------------
    def _set_status(self, msg):
        try:
            self._status.configure(text=msg)
            self.update_idletasks()
        except Exception:
            pass

    def _recip_per_px(self):
        try:
            return float(self.app.recip_res.get()) if self.app else 0.0
        except Exception:
            return 0.0

    def _sync_calib_from_topbar(self):
        rp = self._recip_per_px()
        if rp > 0:
            self._inv_ang.set(round(rp * 0.1, 6))
            self._set_status(
                f"calibration synced: {rp:.5g} nm⁻¹/px → "
                f"{rp * 0.1:.5g} 1/Å/px")
            self._redraw_1d_with_rings()

    def _save_panel(self):
        ph = getattr(self.app, "posthoc", None)
        outdir = getattr(ph, "outdir", None) if ph else None
        if not outdir:
            messagebox.showinfo("save",
                "No run linked — link a post-hoc run first."); return
        from datetime import datetime
        out = os.path.join(outdir, "eval", "acom_tab")
        os.makedirs(out, exist_ok=True)
        stamp = datetime.now().strftime("%H%M%S")
        png = os.path.join(out, f"acom_{stamp}.png")
        pdf = os.path.join(out, f"acom_{stamp}.pdf")
        self._fig.savefig(png, dpi=200, bbox_inches="tight",
                            facecolor="white")
        self._fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        self._set_status(f"saved → {png}")

    # ------------------------------------------------------------------
    # source loading
    # ------------------------------------------------------------------
    def _posthoc(self):
        return getattr(self.app, "posthoc", None)

    # --- dataset (no model needed) vs inference (needs trained run) ---
    def _active_sample(self):
        """Sample key from the global session OR a linked posthoc run.
        Requires only a DATASET — works before any training."""
        sess = getattr(self.app, "session", None)
        if sess is not None and sess.sample and sess.sample in SAMPLES:
            return sess.sample
        ph = self._posthoc()
        if ph is not None and ph.sample in SAMPLES:
            return ph.sample
        return None

    def _active_scan_shape(self):
        s = self._active_sample()
        return SAMPLES[s]["scan_shape"] if s else None

    def _need_dataset(self):
        """Gate for sources that need only the cube (dp_max/dp_mean/
        scan pos / full-dataset ACOM) — no trained model required."""
        s = self._active_sample()
        if s is None:
            messagebox.showinfo("ACOM",
                "Load a DATASET first — pick one with the topbar "
                "'dataset' badge (no trained run needed for dp_max / "
                "single-pattern / full-dataset ACOM).")
        return s

    def _need_posthoc_inference(self):
        """Gate for sources that need DINO inference (class avg / grain
        modes).  Falls back to the legacy posthoc panel."""
        sess = getattr(self.app, "session", None)
        ph = self._posthoc()
        if (sess is None or not sess.has_dataset()
                or not sess.has_inference()):
            if ph is None or ph.sample is None or ph._inf is None:
                messagebox.showinfo("ACOM",
                    "This mode (grain / class average) needs DINO "
                    "inference — load a trained run via the topbar "
                    "'run' badge.  dp_max / single-pattern / "
                    "full-dataset ACOM work with just a dataset.")
                return None
        return ph

    def _compute_dp_max_mean(self, sample):
        """Stream over the cube to build (dp_max, dp_mean) without
        loading the whole thing into RAM.  Cached per sample.
        Needs only the dataset (no model)."""
        if (sample in self._cached_dp_max
                and sample in self._cached_dp_mean):
            return (self._cached_dp_max[sample],
                    self._cached_dp_mean[sample])
        cfg = SAMPLES[sample]
        from gui_app.posthoc_panel import _open_lazy
        cube = _open_lazy(cfg["path"], scan_shape=cfg["scan_shape"])
        Ny, Nx, H, W = cube.shape
        dp_max = np.zeros((H, W), dtype=np.float32)
        dp_sum = np.zeros((H, W), dtype=np.float64)
        t0 = time.time()
        for y in range(Ny):
            try:
                block = np.asarray(cube[y], dtype=np.float32)
            except Exception:
                continue
            # block is (Nx, H, W)
            dp_max = np.maximum(dp_max, block.max(axis=0))
            dp_sum += block.sum(axis=0)
            if (y & 7) == 0:
                self._set_status(
                    f"computing dp_max + dp_mean … row {y+1}/{Ny}  "
                    f"({time.time()-t0:.0f}s)")
        dp_mean = (dp_sum / max(Ny * Nx, 1)).astype(np.float32)
        self._cached_dp_max[sample] = dp_max
        self._cached_dp_mean[sample] = dp_mean
        self._set_status(
            f"dp_max + dp_mean cached  ({time.time()-t0:.0f}s)")
        return dp_max, dp_mean

    def _load_source(self):
        src = self._source_var.get()
        # dp_max / dp_mean / scan_pos need only the DATASET.
        if src in ("dp_max", "dp_mean"):
            sample = self._need_dataset()
            if sample is None: return
            def _w():
                try:
                    dpm, dpu = self._compute_dp_max_mean(sample)
                except Exception as e:
                    err = repr(e)
                    self.after(0, lambda: messagebox.showerror(
                        "dp_max/dp_mean", err))
                    return
                pat = dpm if src == "dp_max" else dpu
                self._test_pattern = pat.astype(np.float32)
                H, W = pat.shape
                self._test_center = (H / 2.0, W / 2.0)
                self._test_origin = (f"{src}  ({pat.shape[0]}×{pat.shape[1]})  "
                                      f"sample={sample}")
                def _done():
                    self._source_status.configure(
                        text=f"loaded: {self._test_origin}")
                    self._detect_and_redraw()
                self.after(0, _done)
            threading.Thread(target=_w, daemon=True).start()
            self._source_status.configure(
                text=f"computing {src}… (streams the cube; ~5–30 s)")
            return
        if src == "scan_pos":
            # single position by (y,x) index — dataset only.
            sample = self._need_dataset()
            if sample is None: return
            cfg = SAMPLES[sample]
            Ny, Nx = cfg["scan_shape"]
            try:
                y = int(self._src_y.get()); x = int(self._src_x.get())
            except Exception:
                self._source_status.configure(
                    text="enter scan (y, x), or click the class map "
                         "(needs a trained run)"); return
            if not (0 <= y < Ny and 0 <= x < Nx):
                messagebox.showerror("scan_pos",
                    f"(y,x) out of range [0..{Ny-1}],[0..{Nx-1}]"); return
            ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
            self._test_pattern = ds.get_raw(
                y * Nx + x).astype(np.float32)
            H, W = self._test_pattern.shape
            self._test_center = (H / 2.0, W / 2.0)
            self._test_origin = f"scan pos ({y}, {x})  sample={sample}"
            self._source_status.configure(
                text=f"loaded: {self._test_origin}")
            self._detect_and_redraw()
            return
        # class_avg / grain need DINO inference.
        ph = self._need_posthoc_inference()
        if ph is None: return
        cfg = SAMPLES[ph.sample]
        if src == "class_avg":
            try:
                cid = int(self._src_class.get())
            except Exception:
                messagebox.showerror("source",
                    "class id must be an integer."); return
            K = int(ph._inf["soft_probs"].shape[1])
            if not (0 <= cid < K):
                messagebox.showerror("source",
                    f"class id {cid} out of range [0..{K-1}]."); return
            # Compute class avg via post-hoc helper.
            try:
                avgs = ph._compute_class_averages(top_n=256)
            except Exception as e:
                messagebox.showerror("class avg", repr(e)); return
            vm = float(cfg.get("vmax", 5.0))
            pat = (avgs[cid] * vm).astype(np.float32)
            self._test_pattern = pat
            self._test_origin = f"class p{cid} avg"
            H, W = pat.shape
            self._test_center = (H / 2.0, W / 2.0)
            self._source_status.configure(
                text=f"loaded: class p{cid} avg  "
                      f"({pat.shape[0]}×{pat.shape[1]})")
        else:
            # grain / scan_pos require a click on the classmap
            self._source_status.configure(
                text="click on the class map (bottom-right) to pick "
                      "the position.")
        self._detect_and_redraw()

    def _on_classmap_click(self, event):
        # Only respond to clicks on the classmap axes.
        if event.inaxes is not getattr(self, "_ax_cmap", None):
            return
        if event.xdata is None or event.ydata is None:
            return
        src = self._source_var.get()
        ph = self._need_posthoc_inference()
        if ph is None: return
        Ny, Nx = ph._scan_shape
        x = max(0, min(Nx - 1, int(round(event.xdata))))
        y = max(0, min(Ny - 1, int(round(event.ydata))))
        if src == "scan_pos":
            cfg = SAMPLES[ph.sample]
            ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
            idx = y * Nx + x
            try:
                raw = ds.get_raw(int(idx)).astype(np.float32)
            except Exception as e:
                messagebox.showerror("scan_pos", repr(e)); return
            self._test_pattern = raw
            self._test_origin = f"scan position (y={y}, x={x})"
            H, W = raw.shape
            self._test_center = (H / 2.0, W / 2.0)
            self._source_status.configure(
                text=f"loaded: scan ({y}, {x})  "
                      f"({raw.shape[0]}×{raw.shape[1]})")
        elif src == "grain":
            gi = ph._compute_grain_average(y, x)
            if gi is None:
                messagebox.showerror("grain",
                    "pixel not in any grain."); return
            self._test_pattern = gi["grain_avg"].astype(np.float32)
            self._test_origin = (f"grain @ ({y}, {x})  class p{gi['cls']}  "
                                  f"{gi['n_pix']}px")
            H, W = self._test_pattern.shape
            self._test_center = (H / 2.0, W / 2.0)
            self._source_status.configure(
                text=f"loaded: grain @ ({y}, {x})  class p{gi['cls']}  "
                      f"{gi['n_pix']}px")
        elif src == "class_avg":
            try:
                self._src_class.set(
                    str(int(ph._inf["assigns"].reshape(Ny, Nx)[y, x])))
            except Exception:
                pass
            self._load_source()
            return
        self._detect_and_redraw()

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def _detect_and_redraw(self):
        if self._test_pattern is None:
            return
        from gui_app.acom_core import detect_peaks_2d
        try:
            peaks = detect_peaks_2d(
                self._test_pattern,
                min_sigma=float(self._det_min.get()),
                max_sigma=float(self._det_max.get()),
                num_sigma=int(self._det_num.get()),
                threshold=float(self._det_thr.get()),
                log_stretch=bool(self._det_log.get()),
            )
            self._test_peaks = peaks
            self._det_status.configure(
                text=f"{len(peaks)} peak(s) detected.")
        except Exception as e:
            messagebox.showerror("detection", repr(e))
            return
        self._draw_pattern_panel()
        self._redraw_1d_with_rings()

    def _draw_pattern_panel(self):
        self._ensure_quad_layout()
        ax = self._ax_pat
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        if self._test_pattern is None:
            ax.text(0.5, 0.5, "(no source loaded)",
                     ha="center", va="center", fontsize=10,
                     color="#888", transform=ax.transAxes)
            self._redraw_all(); return
        img = np.log1p(np.clip(self._test_pattern, 0, None))
        ax.imshow(img, cmap="inferno", aspect="equal",
                    interpolation="nearest")
        if self._test_peaks is not None and len(self._test_peaks):
            ax.scatter(self._test_peaks[:, 1], self._test_peaks[:, 0],
                        s=70, facecolors="none", edgecolors="cyan",
                        linewidths=1.3)
        np_peaks = (0 if self._test_peaks is None
                     else len(self._test_peaks))
        ax.set_title(
            f"step 1+2  ·  {np_peaks} peaks\n"
            f"{self._test_origin}",
            fontsize=9)
        # Hover-q readout — pattern is shown at raw-detector
        # resolution (no resize/crop), so 1 display px = 1 raw px →
        # q_per_disp_px = recip_res (nm⁻¹/px).
        try:
            from gui_app._ui import attach_hover_q
            rp = self._recip_per_px()
            if rp > 0 and self._test_center is not None:
                if getattr(self, "_pat_hover_cid", None) is not None:
                    self._canvas.mpl_disconnect(self._pat_hover_cid)
                self._pat_hover_cid = attach_hover_q(
                    self._canvas, ax,
                    center=self._test_center,
                    q_per_disp_px=rp, units="nm⁻¹")
        except Exception: pass
        self._redraw_all()

    # ------------------------------------------------------------------
    # CIF + phases
    # ------------------------------------------------------------------
    def _refresh_phase_listbox(self):
        try:
            self._phase_list.delete(0, "end")
            for (n, p) in self._phases:
                built = " ✓" if n in self._phase_crystals else ""
                self._phase_list.insert("end",
                    f"{n}  ({os.path.basename(p)}){built}")
        except Exception:
            pass

    def _add_phase(self):
        p = filedialog.askopenfilename(
            filetypes=[("CIF", "*.cif"), ("All", "*.*")])
        if not p:
            return
        default = os.path.splitext(os.path.basename(p))[0]
        name = simpledialog.askstring(
            "Phase name",
            "Short name for this phase (e.g. 'alpha', 'gamma'):",
            initialvalue=default, parent=self)
        if not name:
            return
        self._phases = [(n, q) for (n, q) in self._phases if n != name]
        self._phases.append((name, p))
        self._refresh_phase_listbox()
        self._cif_status.configure(
            text=f"{len(self._phases)} phase(s) listed. "
                  f"Click 'Build all'.")

    def _remove_phase(self):
        try:
            sel = self._phase_list.curselection()
            if not sel:
                return
            i = int(sel[0])
            n, _ = self._phases.pop(i)
            self._phase_crystals.pop(n, None)
            self._phase_keys.pop(n, None)
            self._refresh_phase_listbox()
        except Exception:
            pass

    def _build_all_phases(self):
        if not self._phases:
            messagebox.showinfo("CIF",
                "Add at least one CIF first."); return
        def _w():
            from gui_app.acom_core import load_crystal, prepare_crystal
            kmax = float(self._kmax.get())
            plan = self._plan_mode.get()
            t0 = time.time()
            for i, (name, cif) in enumerate(self._phases):
                key = (cif, kmax, plan)
                if (self._phase_keys.get(name) == key
                        and name in self._phase_crystals):
                    continue
                try:
                    self.after(0, lambda n=name, i=i:
                        self._cif_status.configure(
                            text=f"building [{i+1}/{len(self._phases)}] "
                                  f"{n}…"))
                    cr = load_crystal(cif)
                    prepare_crystal(cr, k_max=kmax, plan_mode=plan)
                    self._phase_crystals[name] = cr
                    self._phase_keys[name] = key
                except Exception as e:
                    err = repr(e)
                    self.after(0, lambda n=name, e=err:
                        messagebox.showerror("CIF build",
                            f"{n}: {e}"))
            self.after(0, self._refresh_phase_listbox)
            dt = time.time() - t0
            self.after(0, lambda:
                self._cif_status.configure(
                    text=f"{len(self._phase_crystals)} crystal(s) ready "
                          f"({dt:.0f}s).  Refresh 1D radial."))
            self.after(0, self._redraw_1d_with_rings)
            # Crystal-build doesn't fire a session change but the
            # 'crystal' StatusDot reads from session_change.  Push a
            # manual refresh so the dot flips to ✓.
            self.after(0, lambda:
                self._on_session_change(
                    getattr(self.app, "session", None)))
        threading.Thread(target=_w, daemon=True).start()

    # ------------------------------------------------------------------
    # 1D radial + CIF rings
    # ------------------------------------------------------------------
    def _radial_1d_of_pattern(self, pat, center, n_bins=200):
        """Polar-averaged 1D radial intensity vs pixel radius."""
        H, W = pat.shape
        cy, cx = center
        yy, xx = np.indices((H, W))
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_max = float(min(H, W) // 2 - 1)
        bins = np.linspace(0.0, r_max, n_bins + 1)
        idx = np.digitize(r.ravel(), bins) - 1
        flat = pat.ravel().astype(np.float64)
        sums = np.bincount(idx, weights=flat,
                              minlength=n_bins + 2)[:n_bins + 1]
        counts = np.bincount(idx, minlength=n_bins + 2)[:n_bins + 1]
        means = sums / np.maximum(counts, 1)
        # bin centers in px
        rc = 0.5 * (bins[:-1] + bins[1:])
        return rc, means[:-1]

    def _detected_peak_q_values(self):
        """Convert the currently detected (qx, qy, intensity) peaks
        into (q_inva, intensity) pairs using the current 1/Å/px and
        the test_center."""
        if (self._test_peaks is None or len(self._test_peaks) == 0
                or self._test_center is None):
            return np.zeros(0), np.zeros(0)
        cy, cx = self._test_center
        inv_a = float(self._inv_ang.get())
        qx = self._test_peaks[:, 0]
        qy = self._test_peaks[:, 1]
        amp = self._test_peaks[:, 2] if self._test_peaks.shape[1] > 2 \
                  else np.ones(len(qx))
        rpx = np.sqrt((qx - cy) ** 2 + (qy - cx) ** 2)
        return rpx * inv_a, amp

    def _redraw_1d_with_rings(self):
        # Skip the layout guard when rendering into the standalone 1D
        # popup (it swaps self._ax_1d to a foreign axes temporarily).
        if not getattr(self, "_in_1d_popup", False):
            self._ensure_quad_layout()
        ax = self._ax_1d
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        if self._test_pattern is None:
            ax.text(0.5, 0.5, "(no source loaded)",
                     ha="center", va="center", fontsize=10,
                     color="#888", transform=ax.transAxes)
            self._redraw_all(); return
        rc_px, prof = self._radial_1d_of_pattern(
            self._test_pattern, self._test_center)
        inv_a = float(self._inv_ang.get())
        rc_inva = rc_px * inv_a       # 1/Å
        # background subtract for visibility
        p = np.clip(prof - np.median(prof), 0, None)
        ax.semilogy(rc_inva, p + 1e-3, color="#1f77b4", lw=1.3,
                       alpha=0.65, label="experimental 1D")

        # *** Always overlay DETECTED peak q-values as red markers ***
        # This is the user's KEY validation signal — even on grain /
        # single-pattern sources where the smooth 1D is noisy, the
        # detected peaks pin the exact q-positions to align with CIF.
        peak_q, peak_amp = self._detected_peak_q_values()
        if peak_q.size:
            # Marker height: value of the smooth 1D profile at each
            # peak's bin, so the dot sits on the curve.  Fall back to
            # the peak's intensity if the bin lookup fails.
            mh = np.zeros_like(peak_q)
            for i, q in enumerate(peak_q):
                j = int(np.clip(round(q / max(inv_a, 1e-12)),
                                  0, len(prof) - 1))
                mh[i] = max(p[j], peak_amp[i], 1e-3)
            ax.vlines(peak_q, 1e-3, mh, color="#e0144c",
                        alpha=0.85, lw=1.0, label="detected peaks")
            ax.scatter(peak_q, mh, s=24, color="#e0144c",
                          zorder=6, edgecolors="white", linewidths=0.6)

        # CIF rings per phase
        leg_handles = []
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        palette = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                     "#8c564b", "#e377c2"]
        any_phase = False
        # Rebuild the (q, phase, hkl, color) database for hover lookup.
        self._cif_ring_db = []
        for pi, (name, _) in enumerate(self._phases):
            cr = self._phase_crystals.get(name)
            if cr is None:
                continue
            gleng = getattr(cr, "g_vec_leng", None)
            sint  = getattr(cr, "struct_factors_int", None)
            hkl   = getattr(cr, "hkl", None)
            if gleng is None or sint is None:
                continue
            color = palette[pi % len(palette)]
            sint_a = np.asarray(sint)
            # Adaptive ring display: keep the TOP_REF strongest
            # rings as orientation anchors, PLUS any other ring whose
            # q sits within `match_tol` of a detected experimental
            # peak.  This avoids overcrowding the plot with weak
            # reflections while still surfacing weak matches (γ100,
            # α002 …) when the experimental peak supports them.
            TOP_REF = 8
            in_range = (np.asarray(gleng) <= rc_inva.max())
            sint_vis = sint_a[in_range]
            gleng_vis = np.asarray(gleng)[in_range]
            hkl_arr = np.asarray(hkl) if hkl is not None else None
            hkl_vis = (hkl_arr[:, in_range]
                          if hkl_arr is not None else None)
            i_max = float(sint_vis.max()) if sint_vis.size else 1.0
            # Top-N indices (always shown).
            topN = set(np.argsort(-sint_vis)[:TOP_REF].tolist())
            # Match tolerance = 1.5 % of visible q-range.
            match_tol = max((rc_inva.max() - rc_inva.min()) * 0.015,
                                 1e-3)
            for k_idx in range(gleng_vis.size):
                q = float(gleng_vis[k_idx])
                I = float(sint_vis[k_idx])
                is_top = k_idx in topN
                is_matched = (peak_q.size > 0
                                and np.min(np.abs(peak_q - q)) <= match_tol)
                if not (is_top or is_matched):
                    continue
                # Opacity / lw scale with intensity (sqrt).  Matched
                # weak rings get a small alpha boost so the user can
                # see they were kept on purpose.
                base = np.sqrt(I / max(i_max, 1e-12))
                alpha_ring = 0.20 + 0.70 * base
                if is_matched and not is_top:
                    alpha_ring = max(alpha_ring, 0.55)
                lw_ring = 0.8 + 0.6 * base
                ax.axvline(q, color=color, alpha=float(alpha_ring),
                            lw=float(lw_ring), linestyle="--")
                if hkl_vis is not None and hkl_vis.shape[1] > int(k_idx):
                    h, kk, l = (int(round(float(v)))
                                  for v in hkl_vis[:, int(k_idx)])
                    self._cif_ring_db.append(
                        (q, name, (h, kk, l), color, I))
            any_phase = True
            leg_handles.append(Line2D([0], [0], color=color, lw=1.4,
                                          linestyle="--",
                                          label=f"{name} (CIF)"))
        if not any_phase:
            ax.text(0.5, 0.92,
                     "(build a CIF in step 3 to see predicted rings)",
                     ha="center", va="top", fontsize=9, color="#a55",
                     transform=ax.transAxes)
        leg_handles.insert(0, Line2D([0], [0], color="#1f77b4", lw=1.4,
                                          label="experimental 1D"))
        if peak_q.size:
            leg_handles.insert(1,
                Line2D([0], [0], marker="o", color="#e0144c",
                          markeredgecolor="white", markersize=6,
                          lw=0, label=f"detected peaks ({peak_q.size})"))
        ax.legend(handles=leg_handles, loc="upper right",
                    fontsize=8, framealpha=0.6)
        ax.set_xlabel("q  (1/Å)")
        ax.set_ylabel("⟨I⟩ - median  (log)")
        ax.set_title(
            f"step 3  ·  calib = {inv_a:.5g} 1/Å/px\n"
            f"align RED markers (peaks) with DASHED lines (CIF) "
            f"·  hover CIF lines for hkl",
            fontsize=9)
        ax.set_xticks(np.arange(0, rc_inva.max() + 0.05, 0.1))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks([])
        self._redraw_all()

    # ---- hover-hkl tooltip on the 1D panel -----------------------
    def _on_1d_hover(self, event):
        """When mouse is near a CIF ring on ax_1d, annotate phase +
        (h k l).  Tolerance is a fraction of the visible q-range."""
        if event.inaxes is not getattr(self, "_ax_1d", None):
            if self._hover_annot is not None:
                self._hover_annot.set_visible(False)
                self._canvas.draw_idle()
            return
        if not self._cif_ring_db or event.xdata is None:
            return
        # tolerance = 1.5 % of the visible q-range
        xlo, xhi = self._ax_1d.get_xlim()
        tol = max((xhi - xlo) * 0.015, 1e-3)
        best = None
        best_d = tol
        for (q, name, hkl, color, inten) in self._cif_ring_db:
            d = abs(event.xdata - q)
            if d < best_d:
                best_d = d; best = (q, name, hkl, color, inten)
        if best is None:
            if self._hover_annot is not None:
                self._hover_annot.set_visible(False)
                self._canvas.draw_idle()
            return
        q, name, (h, k, l), color, inten = best
        text = (f"{name}  ({h} {k} {l})\n"
                  f"q = {q:.4f} 1/Å\n"
                  f"I = {inten:.3g}")
        if self._hover_annot is None:
            self._hover_annot = self._ax_1d.annotate(
                text, xy=(q, event.ydata),
                xytext=(14, 14), textcoords="offset points",
                fontsize=8, color="#111",
                bbox=dict(boxstyle="round,pad=0.35",
                            fc="#fff7c2", ec=color, lw=1.2,
                            alpha=0.96),
                arrowprops=dict(arrowstyle="->", color=color,
                                  lw=1.0),
                annotation_clip=False)
            self._hover_annot.set_zorder(20)
        else:
            self._hover_annot.xy = (q, event.ydata)
            self._hover_annot.set_text(text)
            box = self._hover_annot.get_bbox_patch()
            if box is not None:
                box.set_edgecolor(color)
            arr = self._hover_annot.arrow_patch
            if arr is not None:
                arr.set_color(color)
            self._hover_annot.set_visible(True)
        self._canvas.draw_idle()

    # ---- Expand 1D + CIF rings as a standalone popup -------------
    def _open_1d_popup(self):
        """Open a Toplevel with just the 1D radial + CIF rings at
        full size.  Wires the hover-hkl on the popup's own axes."""
        if self._test_pattern is None:
            messagebox.showinfo("Expand 1D",
                "Load a source first (step 1)."); return
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        win = tk.Toplevel(self)
        win.title("1D radial + CIF rings  (hover for hkl)")
        win.geometry("1180x620")
        fig = Figure(figsize=(11.5, 5.5), dpi=110, facecolor="white")
        ax = fig.add_subplot(111)
        # Redraw the same content onto this fresh axis.
        keep_ax = self._ax_1d
        self._ax_1d = ax
        self._in_1d_popup = True
        try:
            self._redraw_1d_with_rings()
        finally:
            self._ax_1d = keep_ax
            self._in_1d_popup = False
        fig.tight_layout()
        canv = FigureCanvasTkAgg(fig, master=win)
        canv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(canv, win, pack_toolbar=False)
        tb.update(); tb.pack(side="bottom", fill="x")
        # Local hover wired to the popup's axes.
        annot = {"a": None}
        def _hover(event):
            if event.inaxes is not ax or not self._cif_ring_db: return
            xlo, xhi = ax.get_xlim()
            tol = max((xhi - xlo) * 0.015, 1e-3)
            best, best_d = None, tol
            for (q, name, hkl, color, inten) in self._cif_ring_db:
                d = abs(event.xdata - q)
                if d < best_d: best_d = d; best = (q, name, hkl,
                                                       color, inten)
            if best is None:
                if annot["a"] is not None:
                    annot["a"].set_visible(False); canv.draw_idle()
                return
            q, name, (h, k, l), color, inten = best
            txt = (f"{name}  ({h} {k} {l})\n"
                     f"q = {q:.4f} 1/Å\n"
                     f"I = {inten:.3g}")
            if annot["a"] is None:
                annot["a"] = ax.annotate(
                    txt, xy=(q, event.ydata),
                    xytext=(18, 18), textcoords="offset points",
                    fontsize=10, color="#111",
                    bbox=dict(boxstyle="round,pad=0.4",
                                fc="#fff7c2", ec=color, lw=1.4,
                                alpha=0.97),
                    arrowprops=dict(arrowstyle="->", color=color,
                                      lw=1.2),
                    annotation_clip=False)
                annot["a"].set_zorder(20)
            else:
                annot["a"].xy = (q, event.ydata)
                annot["a"].set_text(txt)
                bx = annot["a"].get_bbox_patch()
                if bx is not None: bx.set_edgecolor(color)
                ar = annot["a"].arrow_patch
                if ar is not None: ar.set_color(color)
                annot["a"].set_visible(True)
            canv.draw_idle()
        canv.mpl_connect("motion_notify_event", _hover)

    def _auto_calibrate(self):
        """Auto-fit 1/Å/px by aligning *detected peaks* (not the
        smooth 1D curve) with the strongest CIF rings.

        Strategy: for each candidate scale s, place every detected
        peak at q = r_px * s.  Score = sum over (CIF ring, detected
        peak) of  CIF_intensity × peak_intensity × exp(-Δ²/2σ²).
        Argmax over s.  Aligning *peaks* with *rings* is far more
        decisive than aligning a smooth curve with rings, especially
        on noisy grain / single-pattern sources.
        """
        if self._test_pattern is None:
            messagebox.showinfo("Auto-fit",
                "Load a source first (step 1)."); return
        if self._test_peaks is None or len(self._test_peaks) == 0:
            messagebox.showinfo("Auto-fit",
                "No peaks detected yet — run step 2 first."); return
        cr = next((self._phase_crystals[n]
                    for (n, _) in self._phases
                    if n in self._phase_crystals), None)
        if cr is None:
            messagebox.showinfo("Auto-fit",
                "Build a crystal in step 3 first."); return
        gleng = np.asarray(cr.g_vec_leng)
        sint  = np.asarray(cr.struct_factors_int)
        order = np.argsort(-sint)
        ring_q = gleng[order[:40]]              # (M,)
        ring_w = sint[order[:40]]
        # Normalise ring weights so the score is in a stable range.
        ring_w = ring_w / (ring_w.max() + 1e-12)

        # Detected peaks: radius in raw px + their intensities.
        cy, cx = self._test_center
        peak_rpx = np.sqrt(
            (self._test_peaks[:, 0] - cy) ** 2 +
            (self._test_peaks[:, 1] - cx) ** 2)
        if self._test_peaks.shape[1] > 2:
            peak_amp = self._test_peaks[:, 2].astype(float)
        else:
            peak_amp = np.ones(len(peak_rpx))
        peak_amp = peak_amp / (peak_amp.max() + 1e-12)
        # Drop the BF disk: peaks too close to center contribute
        # nothing useful (they're not Bragg rings).  Keep |r| > 2 px.
        keep = peak_rpx > 2.0
        peak_rpx = peak_rpx[keep]; peak_amp = peak_amp[keep]
        if peak_rpx.size == 0:
            messagebox.showinfo("Auto-fit",
                "All detected peaks are at the BF disk."); return

        cur = float(self._inv_ang.get())
        scales = cur * np.linspace(0.5, 1.8, 261)
        sigma = 0.01     # 1/Å — about one ring half-width
        best_s, best_score = scales[0], -np.inf
        for s in scales:
            q_peaks = peak_rpx * s                # (N,)
            # Pairwise distance (M, N): CIF rings × peaks
            dq = ring_q[:, None] - q_peaks[None, :]
            kern = np.exp(-(dq ** 2) / (2 * sigma ** 2))
            score = float((ring_w[:, None] * peak_amp[None, :]
                              * kern).sum())
            if score > best_score:
                best_score = score
                best_s = s
        self._inv_ang.set(round(float(best_s), 6))
        self._set_status(
            f"auto-fit (peaks ↔ CIF rings): 1/Å/px = {best_s:.5g}  "
            f"score = {best_score:.3g}   "
            f"(using {peak_rpx.size} peaks vs {ring_q.size} rings)")
        self._redraw_1d_with_rings()

    # ------------------------------------------------------------------
    # single-pattern match (step 4)
    # ------------------------------------------------------------------
    # ---- NNLS multi-phase fit (mirrors the py4DSTEM colab) ---------
    def _build_phase_for_patterns(self, patterns_list,
                                       centers_list, Rshape):
        """Build a CrystalPhase + calibrated BV for one OR many test
        patterns.  Runs match_orientations on every phase to populate
        crystal.orientation_map (required by quantify_*).  Returns
        (cp, bv)."""
        from gui_app.acom_core import (detect_peaks_2d,
                                            build_bragg_vectors)
        from py4DSTEM.process.diffraction.crystal_phase import (
            CrystalPhase)
        detect_kw = dict(
            min_sigma=float(self._det_min.get()),
            max_sigma=float(self._det_max.get()),
            num_sigma=int(self._det_num.get()),
            threshold=float(self._det_thr.get()),
            log_stretch=bool(self._det_log.get()),
        )
        peak_arrays = [detect_peaks_2d(p, **detect_kw)
                          for p in patterns_list]
        bv = build_bragg_vectors(
            peak_arrays, centers=centers_list,
            inv_ang_per_pixel=float(self._inv_ang.get()),
            Rshape=Rshape)
        names = list(self._phase_crystals.keys())
        crystals = list(self._phase_crystals.values())
        for cr in crystals:
            try:
                cr.match_orientations(bv, progress_bar=False)
            except Exception as e:
                # Degenerate pattern → match_orientations argmin
                # crash.  Fall back to per-pattern; leaves an empty
                # orientation_map slot that quantify handles.
                print(f"[NNLS] match_orientations fallback: {e!r}",
                      flush=True)
                from gui_app.acom_core import _match_safe
                _match_safe(cr, bv, Rshape[0] * Rshape[1])
        cp = CrystalPhase(crystals, crystal_names=names)
        return cp, bv

    def _run_nnls_phase_fit(self):
        """Single-pattern CrystalPhase.quantify_single_pattern fit.
        Shows the gray-circle (experimental) + colored-triangle (per
        phase) plot from the colab notebook plus the per-phase
        weights, fit total, residual, and reliability."""
        if self._test_pattern is None:
            messagebox.showinfo("NNLS fit",
                "Load a source first (step 1)."); return
        if not self._phase_crystals:
            messagebox.showinfo("NNLS fit",
                "Build at least one CIF (step 3)."); return
        self._match_status.configure(
            text="NNLS phase fit running…")
        def _w():
            try:
                cp, bv = self._build_phase_for_patterns(
                    [self._test_pattern],
                    [self._test_center],
                    Rshape=(1, 1))
                kmax = float(self._kmax.get())
                # Per the notebook: experiment markers are radius
                # scaled by peak intensity ∝ 0.02; calculated marker
                # base ∝ 100.
                res = cp.quantify_single_pattern(
                    bv, xy_position=(0, 0), k_max=kmax,
                    corr_kernel_size=0.04,
                    sigma_excitation_error=0.02,
                    power_intensity=0.25,
                    power_intensity_experiment=0.25,
                    max_number_patterns=1,
                    allow_strain=False,
                    plot_result=True, returnfig=True,
                    verbose=False,
                    plot_unmatched_peaks=True,
                    scale_markers_experiment=0.02,
                    scale_markers_calculated=100,
                    figsize=(7.5, 5.5))
                # Return is
                # (phase_weights, residual, reliability, int_total,
                #  fig, ax)
                pw, pr, prel, _it, fig_pt, _ax_pt = res
                names = list(self._phase_crystals.keys())
                per_phase = np.zeros(len(names))
                for fi, (pi, _) in enumerate(cp.crystal_identity):
                    per_phase[pi] += float(pw[fi])
                self.after(0,
                    lambda: self._draw_nnls_result(
                        names, per_phase, float(pr),
                        float(prel), fig_pt))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[NNLS fit] FAILED:\n{tb}", flush=True)
                err = repr(e)
                self.after(0, lambda: messagebox.showerror(
                    "NNLS fit",
                    f"{err}\n\n(full traceback in console)"))
                self.after(0, lambda: self._match_status.configure(
                    text=f"NNLS fit failed: {err[:120]}"))
        threading.Thread(target=_w, daemon=True).start()

    def _draw_nnls_result(self, phase_names, per_phase_weights,
                              residual, reliability, fig_pt):
        """Render the quantify_single_pattern figure on ax_fit by
        rasterising it (the py4DSTEM call already built its own fig;
        we re-host it as an image on our axes)."""
        import io
        import matplotlib.pyplot as _plt
        try:
            buf = io.BytesIO()
            fig_pt.savefig(buf, format="png", dpi=140,
                              bbox_inches="tight", facecolor="white")
            _plt.close(fig_pt)
            buf.seek(0)
            from matplotlib.image import imread
            arr = imread(buf)
        except Exception as e:
            err = repr(e)
            self._match_status.configure(text=f"raster err: {err}")
            return
        ax = self._ax_fit
        ax.clear()
        ax.imshow(arr, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(
            "step 4 — NNLS multi-phase fit  "
            "(gray = experimental peaks; coloured triangles = "
            "predicted per phase)",
            fontsize=10)
        # Status line: weights + residual + reliability
        lines = ["NNLS  "]
        total = float(per_phase_weights.sum())
        for n, w in zip(phase_names, per_phase_weights):
            frac = (w / total * 100) if total > 0 else 0.0
            lines.append(f"{n}={w:.3f} ({frac:.1f}%)")
        lines.append(f"resid={residual:.3f}")
        lines.append(f"reliab={reliability:.3f}")
        self._match_status.configure(text="    ".join(lines))
        self._redraw_all()

    def _run_single_match(self):
        if self._test_pattern is None:
            messagebox.showinfo("Match",
                "Load a source first (step 1)."); return
        if not self._phase_crystals:
            messagebox.showinfo("Match",
                "Build at least one CIF (step 3)."); return
        def _w():
            from gui_app.acom_core import (acom_single_pattern,
                                                zone_axis_from_matrix)
            results = {}
            for name, cr in self._phase_crystals.items():
                try:
                    res = acom_single_pattern(
                        cr, self._test_pattern,
                        center=self._test_center,
                        inv_ang_per_pixel=float(self._inv_ang.get()),
                        detect_kw=dict(
                            min_sigma=float(self._det_min.get()),
                            max_sigma=float(self._det_max.get()),
                            num_sigma=int(self._det_num.get()),
                            threshold=float(self._det_thr.get()),
                            log_stretch=bool(self._det_log.get()),
                        ))
                except Exception as e:
                    res = {"error": repr(e)}
                results[name] = res
            self.after(0, lambda r=results: self._draw_fit_panel(r))
        threading.Thread(target=_w, daemon=True).start()
        self._match_status.configure(text="running match…")

    def _draw_fit_panel(self, results):
        from gui_app.acom_core import zone_axis_from_matrix
        self._ensure_quad_layout()
        ax = self._ax_fit
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        if not results:
            ax.text(0.5, 0.5, "(no match yet)",
                     ha="center", va="center", fontsize=10,
                     color="#888", transform=ax.transAxes)
            self._redraw_all(); return
        # Pick the best phase by corr.
        best_name, best_res = None, None
        for name, res in results.items():
            if "error" in res:
                continue
            if best_res is None or res["corr"] > best_res["corr"]:
                best_name, best_res = name, res
        status_lines = []
        for name, res in results.items():
            if "error" in res:
                status_lines.append(f"{name}: error {res['error'][:50]}")
                continue
            R = None
            ort = res["orientation"]
            for attr in ("matrix",):
                v = getattr(ort, attr, None)
                if v is not None:
                    arr = np.asarray(v)
                    R = arr[0] if arr.ndim == 3 else arr
                    break
            za, mis = zone_axis_from_matrix(R)
            tag = "★" if name == best_name else " "
            status_lines.append(
                f"{tag} {name}: corr={res['corr']:.4f}  "
                f"ZA=[{za[0]} {za[1]} {za[2]}]  miso={mis:.1f}°")
        self._match_status.configure(text="\n".join(status_lines))

        if best_res is None:
            ax.text(0.5, 0.5,
                     "all matches errored — see status",
                     ha="center", va="center", fontsize=10,
                     color="#a33", transform=ax.transAxes)
            self._redraw_all(); return

        # Overlay experimental peaks vs CIF predicted.  py4DSTEM's
        # plot_diffraction_pattern sizes each marker as
        # `scale * intensity`, so a fixed scale produces giant blobs
        # when intensities are large and dots when they're small.
        # Auto-scale BOTH so the median marker is a sane ~80 px², and
        # draw experimental as hollow rings (not filled) so they don't
        # bury the predicted hkl markers.
        try:
            fit = best_res["fit_pattern"]
            pl  = best_res["calibrated_pl"]
            def _scale_for(obj, target=80.0):
                try:
                    inten = np.asarray(obj.data["intensity"],
                                          dtype=float)
                    med = np.median(inten[inten > 0]) if (
                        inten.size and (inten > 0).any()) else 1.0
                    return float(target / max(med, 1e-9))
                except Exception:
                    return 1.0
            s_fit = _scale_for(fit)
            s_exp = _scale_for(pl)
            from py4DSTEM.process.diffraction import (
                plot_diffraction_pattern)
            plot_diffraction_pattern(
                fit, bragg_peaks_compare=pl,
                scale_markers=s_fit,
                scale_markers_compare=s_exp,
                min_marker_size=2, figsize=(5, 5),
                input_fig_handle=(self._fig, [ax]))
        except Exception as e:
            ax.text(0.5, 0.5, f"plot err:\n{e!r}",
                     ha="center", va="center", fontsize=9,
                     color="#a33", transform=ax.transAxes)
        ax.set_title(
            f"step 4 — best fit: {best_name}  "
            f"corr={best_res['corr']:.4f}\n"
            f"(red = CIF hkl, cyan = experimental peaks)",
            fontsize=9)
        self._redraw_all()

    def _show_topN_matches(self):
        """Pop a small window with the top-N (ZA, corr) candidates for
        the current test pattern + best phase."""
        if self._test_pattern is None or not self._phase_crystals:
            messagebox.showinfo("Top-N",
                "Need a source + at least one built CIF."); return
        from gui_app.acom_core import zone_axis_from_matrix
        # Just run single-pattern match against the first phase and
        # interrogate orient.matrix / orient.corr (which py4DSTEM
        # already returns as top-num_matches arrays).
        name = next(iter(self._phase_crystals))
        cr = self._phase_crystals[name]
        def _w():
            from gui_app.acom_core import acom_single_pattern
            try:
                res = acom_single_pattern(
                    cr, self._test_pattern,
                    center=self._test_center,
                    inv_ang_per_pixel=float(self._inv_ang.get()),
                    detect_kw=dict(
                        min_sigma=float(self._det_min.get()),
                        max_sigma=float(self._det_max.get()),
                        num_sigma=int(self._det_num.get()),
                        threshold=float(self._det_thr.get()),
                        log_stretch=bool(self._det_log.get()),
                    ))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "Top-N match", repr(e)))
                return
            self.after(0, lambda r=res, n=name: self._show_topN_window(n, r))
        threading.Thread(target=_w, daemon=True).start()

    def _show_topN_window(self, name, res):
        from gui_app.acom_core import zone_axis_from_matrix
        ort = res["orientation"]
        mat = np.asarray(getattr(ort, "matrix", np.zeros((0, 3, 3))))
        corr = np.asarray(getattr(ort, "corr", np.zeros(0))).ravel()
        n = min(len(corr), 5)
        win = tk.Toplevel(self)
        win.title(f"top-{n} matches — phase '{name}'")
        win.geometry("500x340")
        txt = tk.Text(win, font=("Consolas", 10), wrap="word")
        txt.pack(fill="both", expand=True, padx=8, pady=6)
        lines = [f"phase: {name}",
                  f"# peaks fed: {len(res['peaks'])}",
                  f"calibration: {float(self._inv_ang.get()):.5g} 1/Å/px",
                  ""]
        for i in range(n):
            R = mat[i] if mat.ndim == 3 else mat
            za, mis = zone_axis_from_matrix(R)
            lines.append(
                f"  #{i+1}  corr={corr[i]:.4f}   "
                f"ZA=[{za[0]:>2} {za[1]:>2} {za[2]:>2}]   "
                f"miso={mis:5.2f}°")
        if n == 0:
            lines.append("  (no matches returned)")
        txt.insert("1.0", "\n".join(lines))
        txt.configure(state="disabled")

    # ------------------------------------------------------------------
    # class map (for picking grain / scan_pos)
    # ------------------------------------------------------------------
    def _draw_classmap_if_possible(self):
        self._ensure_quad_layout()
        ax = self._ax_cmap
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        ph = self._posthoc()
        if ph is None or ph._inf is None or ph._scan_shape is None:
            ax.text(0.5, 0.5, "(load posthoc inference)",
                     ha="center", va="center", fontsize=10,
                     color="#888", transform=ax.transAxes)
            self._redraw_all(); return
        Ny, Nx = ph._scan_shape
        K = int(ph._inf["soft_probs"].shape[1])
        amap = ph._inf["assigns"].reshape(Ny, Nx)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("tab10")
        palette = ListedColormap([cmap(i) for i in range(K)])
        ax.imshow(amap, cmap=palette, vmin=-0.5, vmax=K - 0.5,
                    interpolation="nearest")
        ax.set_title(
            f"class map  ({ph.sample}, K={K})  — "
            f"click to pick source (steps 1)",
            fontsize=10)
        self._classmap_axes_xy = (Ny, Nx)
        self._redraw_all()

    # ------------------------------------------------------------------
    # batch run (step 5) — reuses acom_core multi-phase backend
    # ------------------------------------------------------------------
    def _stop_batch_run(self):
        """Signal the batch worker to stop at the next safe checkpoint.
        match_orientations / quantify_phase are single py4DSTEM C-calls
        we can't interrupt mid-call; the stop fires between major
        steps (detection loop, between phases, before quantify_phase)."""
        if not self._stop_event.is_set():
            self._stop_event.set()
            self._set_status("STOP requested — finishing current "
                                "py4DSTEM call, then exiting…")

    def _run_batch(self, mode):
        # full / mp_full need only the DATASET; classes/grains need
        # DINO inference (they operate on class avgs / grains).
        if mode in ("full", "mp_full"):
            if self._need_dataset() is None: return
        else:
            if self._need_posthoc_inference() is None: return
        if not self._phase_crystals:
            messagebox.showinfo("Batch",
                "Build at least one CIF (step 3)."); return
        # Sanity gate: make sure user has done steps 1-4 at least once.
        if self._test_pattern is None:
            if not messagebox.askyesno("Batch",
                "You haven't validated a source yet (step 1).  "
                "Continue anyway?"):
                return
        # Cache prompt (full-dataset modes only): if a peak cache for
        # the current (stride, detection params) already exists, ask
        # whether to reuse it (skip detection) or re-detect.  Decided
        # on the MAIN thread here; the worker honours the flag.
        self._use_cached_peaks = True
        if mode in ("full", "mp_full"):
            try:
                stride = max(int(self._full_stride.get()), 1)
                detect_kw = self._detect_kw_now()
                cp = self._peak_cache_path(stride, detect_kw)
                if cp and os.path.exists(cp):
                    info = self._peak_cache_info(cp)
                    use = messagebox.askyesno(
                        "Cached peaks found",
                        f"A peak cache matching the current stride + "
                        f"detection params exists:\n\n"
                        f"  {os.path.basename(cp)}\n  {info}\n\n"
                        f"USE the cached peaks (Yes) and skip "
                        f"detection — only re-run the orientation "
                        f"matching?\n\n"
                        f"Choose No to re-detect peaks from scratch.")
                    self._use_cached_peaks = bool(use)
            except Exception as e:
                print(f"[acom] cache prompt skipped: {e!r}",
                      flush=True)
        self._stop_event.clear()        # arm a fresh run
        threading.Thread(
            target=lambda: self._batch_worker(mode),
            daemon=True).start()

    def _detect_kw_now(self):
        return dict(
            min_sigma=float(self._det_min.get()),
            max_sigma=float(self._det_max.get()),
            num_sigma=int(self._det_num.get()),
            threshold=float(self._det_thr.get()),
            log_stretch=bool(self._det_log.get()))

    def _peak_cache_path(self, stride, detect_kw):
        """Path of the peak cache for (run, stride, detect params).
        Independent of CIF/phase so any phase run reuses it.
        Stored under <run>/acom/ when a trained run is linked, else
        next to the cube in <cube_dir>/_acom_peakcache/ (dataset-only
        mode)."""
        import json, hashlib
        sample = self._active_sample()
        if sample is None:
            return None
        Ny, Nx = SAMPLES[sample]["scan_shape"]
        kh = hashlib.md5(json.dumps(
            {"sample": sample, "stride": stride, "kw": detect_kw,
               "shape": [Ny, Nx]},
            sort_keys=True, default=str).encode()).hexdigest()[:10]
        ph = self._posthoc()
        run = getattr(ph, "outdir", None) if ph else None
        if run and os.path.isdir(run):
            base = os.path.join(run, "acom")
        else:
            base = os.path.join(
                os.path.dirname(SAMPLES[sample]["path"]),
                "_acom_peakcache")
        return os.path.join(base, f"peaks_full_{kh}.npz")

    def _peak_cache_info(self, path):
        try:
            import datetime
            mt = datetime.datetime.fromtimestamp(
                os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
            mb = os.path.getsize(path) / 1e6
            return f"saved {mt}  ·  {mb:.1f} MB"
        except Exception:
            return ""

    def _batch_worker(self, mode):
        from scipy.ndimage import label
        from gui_app.acom_core import (acom_batch, acom_full_dataset,
                                            acom_multiphase_batch,
                                            acom_multiphase_full_dataset,
                                            zone_axis_from_matrix)
        ph = self._posthoc()
        try:
            sample = self._active_sample()
            cfg = SAMPLES[sample]
            Ny, Nx = cfg["scan_shape"]
            inv_a = float(self._inv_ang.get())
            detect_kw = dict(
                min_sigma=float(self._det_min.get()),
                max_sigma=float(self._det_max.get()),
                num_sigma=int(self._det_num.get()),
                threshold=float(self._det_thr.get()),
                log_stretch=bool(self._det_log.get()),
            )
            thr = float(self._mp_threshold.get())
            mar = float(self._mp_margin.get())
            crystals = dict(self._phase_crystals)
            t0 = time.time()
            # Inference-derived arrays are only needed for class/grain
            # modes — leave them None for full / mp_full (dataset-only).
            assigns = soft = assigns_grid = None
            K = 0
            if mode not in ("full", "mp_full"):
                assigns = ph._inf["assigns"]
                soft = ph._inf["soft_probs"]
                K = int(soft.shape[1])
                assigns_grid = assigns.reshape(Ny, Nx)

            if mode in ("full", "mp_full"):
                from gui_app.posthoc_panel import _open_lazy
                cube = _open_lazy(cfg["path"], scan_shape=(Ny, Nx))
                stride = max(int(self._full_stride.get()), 1)
                def _prog(done, total, stage):
                    if stage == "detect" and done % 256 == 0:
                        dt = time.time() - t0
                        eta = (dt / max(done, 1)) * (total - done)
                        self.after(0, lambda: self._set_status(
                            f"{mode} detect {done}/{total}  "
                            f"({dt:.0f}s, ETA {eta:.0f}s)"))
                    elif stage.startswith("match"):
                        self.after(0, lambda s=stage:
                            self._set_status(f"{mode} {s}…"))

                if mode == "full":
                    from gui_app.acom_core import build_bragg_vectors
                    cr = next(iter(crystals.values()))
                    Ny, Nx = cube.shape[:2]
                    # Cached detection (skips on rerun) → match only.
                    peaks_all, centers_all = self._detect_full_cached(
                        cube, stride, detect_kw, _prog)
                    bv = build_bragg_vectors(
                        peaks_all, centers=centers_all,
                        inv_ang_per_pixel=inv_a, Rshape=(Ny, Nx))
                    self._set_status("matching orientations…")
                    cr.match_orientations(bv, progress_bar=False)
                    omap = cr.orientation_map
                    scan_shape = (Ny, Nx)
                    dt = time.time() - t0
                    self.after(0, lambda:
                        self._render_classical_orientation_strain(
                            cr, omap, bv, scan_shape, stride, dt))
                else:
                    # Mirror the notebook: build PointListArray over
                    # the full cube → match_orientations per phase →
                    # CrystalPhase.quantify_phase (NNLS fit at every
                    # position) → plot_dominant_phase.
                    self._set_status(
                        f"NNLS full-dataset: detecting peaks "
                        f"(stride={stride})…")
                    cp = self._run_nnls_full_dataset(
                        cube, stride, detect_kw, _prog)
                    dt = time.time() - t0
                    self.after(0, lambda: self._render_nnls_full_map(
                        cp, stride, dt))
                self.after(0, lambda: self._set_status(
                    f"{mode} done ({dt:.0f}s)."))
                return

            # ---- per-class / per-grain modes ----
            patterns, labels, classes = [], [], []
            # masks: for mp_classes one mask per class; for mp_grains
            # one mask per connected component.  Used downstream to
            # paint the class/grain map by phase.
            region_masks = []
            if mode in ("classes", "mp_classes"):
                avgs = ph._compute_class_averages(top_n=256)
                vm = float(cfg.get("vmax", 5.0))
                for k in range(K):
                    p = (avgs[k] * vm).astype(np.float32)
                    patterns.append(p)
                    n = int((assigns == k).sum())
                    labels.append(f"class p{k}  N={n}")
                    classes.append(k)
                    region_masks.append(assigns_grid == k)
            elif mode in ("grains", "mp_grains"):
                # Collect EVERY grain ≥ min_grain_size so the resulting
                # phase map covers most of the field, not just the
                # 1 largest per class.
                min_grain_size = 20
                for k in range(K):
                    mask = (assigns_grid == k)
                    if not mask.any(): continue
                    lab, _n = label(mask)
                    if _n == 0: continue
                    sizes = np.bincount(lab.ravel()); sizes[0] = 0
                    for gid in range(1, len(sizes)):
                        if sizes[gid] < min_grain_size: continue
                        gmask = (lab == gid)
                        ys, xs = np.where(gmask)
                        yi = int(ys[len(ys)//2])
                        xi = int(xs[len(xs)//2])
                        gi = ph._compute_grain_average(yi, xi)
                        if gi is None: continue
                        patterns.append(
                            gi["grain_avg"].astype(np.float32))
                        labels.append(
                            f"p{k} g{gid}  {int(sizes[gid])}px  "
                            f"⟨p⟩={gi['mean_conf']:.2f}")
                        classes.append(k)
                        region_masks.append(gmask)
            else:
                raise ValueError(f"unknown mode: {mode}")
            if not patterns:
                self.after(0, lambda: messagebox.showinfo(
                    "Batch", "no patterns")); return

            # Progress callback shared by both batch paths.
            def _bprog(done, total, stage):
                self.after(0, lambda: self._set_status(
                    f"{mode}: {stage} {done}/{total}  "
                    f"({time.time()-t0:.0f}s)"))
            self.after(0, lambda: self._set_status(
                f"{mode}: detecting peaks on {len(patterns)} "
                f"patterns…"))

            if mode in ("classes", "grains"):
                phase_name = next(iter(crystals.keys()))
                cr = crystals[phase_name]
                results, omap, bv = acom_batch(
                    cr, patterns, inv_ang_per_pixel=inv_a,
                    detect_kw=detect_kw, progress_cb=_bprog)
                zas = []
                for r in results:
                    za, mis = zone_axis_from_matrix(r["rotation_matrix"])
                    zas.append((za, mis))
                self.after(0, lambda: self._render_batch_cards_singlephase(
                    patterns, labels, classes, results, zas, mode,
                    phase_name=phase_name))
            else:
                mp = acom_multiphase_batch(
                    crystals, patterns, inv_ang_per_pixel=inv_a,
                    detect_kw=detect_kw, threshold=thr, margin=mar,
                    progress_cb=_bprog)
                # Phase-colored grain/class MAP (the spatial view the
                # user asked for): same shape as the class map but
                # painted by phase.
                self.after(0, lambda: self._render_phase_region_map(
                    mp, region_masks, classes, labels, ph._scan_shape,
                    mode))
            dt = time.time() - t0
            self.after(0, lambda: self._set_status(
                f"{mode} done ({dt:.0f}s)."))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[acom batch] FAILED:\n{tb}", flush=True)
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "Batch ACOM",
                f"{err}\n\n(full traceback printed to console)"))
            self.after(0, lambda: self._set_status(
                f"batch failed: {err[:120]}"))

    # ------------------------------------------------------------------
    # batch result renderers — paint to the bottom-left panel +
    # repaint the canvas full-bleed when results are big.
    # ------------------------------------------------------------------
    def _phase_palette(self, n):
        base = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
        return [base[i % len(base)] for i in range(n)]

    # -- NNLS full-dataset (mirrors notebook's quantify_phase) -------
    def _detect_full_cached(self, cube, stride, detect_kw,
                                 progress_cb):
        """Full-dataset peak detection with on-disk caching.

        Cache key = run + stride + detect params (NOT the CIF), so
        changing CIF / k_max / phase set and re-running reuses the
        peaks and skips straight to matching — 'rerun only the fit'.
        Returns (peaks_all, centers_all).
        """
        from gui_app.acom_core import detect_peaks_2d
        cube = np.asarray(cube) if not hasattr(cube, "shape") else cube
        Ny, Nx, H, W = cube.shape
        center = (H / 2.0, W / 2.0)
        cache_path = self._peak_cache_path(stride, detect_kw)
        # Honour the user's cache decision made in _run_batch
        # (defaults to True for any direct caller).
        use_cache = getattr(self, "_use_cached_peaks", True)
        if use_cache and cache_path and os.path.exists(cache_path):
            try:
                d = np.load(cache_path, allow_pickle=True)
                peaks_all = [np.asarray(a, dtype=float)
                                for a in d["peaks"]]
                if progress_cb is not None:
                    progress_cb(Ny * Nx, Ny * Nx, "detect (cached)")
                self._set_status(
                    f"✓ loaded cached peaks "
                    f"({os.path.basename(cache_path)}) — skipping "
                    f"detection, matching only")
                self._warn_if_sparse(peaks_all, stride)
                return peaks_all, [center] * (Ny * Nx)
            except Exception:
                pass
        peaks_all, centers_all = [], []
        total = Ny * Nx; done = 0
        for rx in range(Ny):
            if self._stop_event.is_set():
                raise RuntimeError("stopped by user (during detection)")
            for ry in range(Nx):
                done += 1
                if (rx % stride) or (ry % stride):
                    peaks_all.append(np.zeros((0, 3), dtype=float))
                    centers_all.append(center); continue
                try:
                    pat = np.asarray(cube[rx, ry], dtype=np.float32)
                except Exception:
                    peaks_all.append(np.zeros((0, 3), dtype=float))
                    centers_all.append(center); continue
                peaks_all.append(detect_peaks_2d(pat, **detect_kw))
                centers_all.append(center)
                if progress_cb is not None and (done % 256 == 0):
                    progress_cb(done, total, "detect")
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.savez_compressed(
                    cache_path, peaks=np.array(peaks_all, dtype=object))
                self._set_status(
                    f"peaks cached → {os.path.basename(cache_path)}")
            except Exception as e:
                print(f"[acom] peak cache save: {e!r}", flush=True)
        self._warn_if_sparse(peaks_all, stride)
        return peaks_all, centers_all

    def _warn_if_sparse(self, peaks_all, stride):
        """Detection sanity check: matching needs ≥3 peaks/pattern.
        If the median sampled pattern has <3, warn the user that
        their detection params are too strict (the match will be
        mostly empty / black map)."""
        cnt = []
        for p in peaks_all:
            a = np.asarray(p)
            n = a.shape[0] if a.ndim == 2 else 0
            if n > 0: cnt.append(n)
        if not cnt:
            med = 0
        else:
            med = int(np.median(cnt))
        msg = (f"detected peaks/pattern: median={med} over "
                 f"{len(cnt)} non-empty patterns")
        self._set_status(msg)
        if med < 3:
            self.after(0, lambda: messagebox.showwarning(
                "Sparse peaks — match will be poor",
                f"Detection found a median of only {med} Bragg "
                f"peak(s) per pattern (need ≥3, ideally ≥10 for a "
                f"good orientation match).\n\n"
                f"The orientation map will be mostly black.\n\n"
                f"Fix: in Step 2 LOWER the detection threshold and/or "
                f"widen the sigma range, verify on a single pattern "
                f"that you get 10+ peaks, then re-run and choose "
                f"'No' (re-detect) at the cache prompt."))

    def _run_nnls_full_dataset(self, cube, stride, detect_kw,
                                    progress_cb):
        """Detect → build BV → match_orientations per phase →
        CrystalPhase.quantify_phase NNLS over all (Ny, Nx).  Returns
        the fitted CrystalPhase object."""
        from gui_app.acom_core import build_bragg_vectors
        from py4DSTEM.process.diffraction.crystal_phase import (
            CrystalPhase)
        cube = np.asarray(cube) if not hasattr(cube, "shape") else cube
        Ny, Nx, H, W = cube.shape
        peaks_all, centers_all = self._detect_full_cached(
            cube, stride, detect_kw, progress_cb)
        bv = build_bragg_vectors(
            peaks_all, centers=centers_all,
            inv_ang_per_pixel=float(self._inv_ang.get()),
            Rshape=(Ny, Nx))
        names = list(self._phase_crystals.keys())
        crystals = list(self._phase_crystals.values())
        for n, cr in zip(names, crystals):
            if self._stop_event.is_set():
                raise RuntimeError(
                    f"stopped by user (before match[{n}])")
            if progress_cb is not None:
                progress_cb(0, 1, f"match_orientations[{n}]")
            cr.match_orientations(bv, progress_bar=False)
        if self._stop_event.is_set():
            raise RuntimeError("stopped by user (before NNLS)")
        cp = CrystalPhase(crystals, crystal_names=names)
        if progress_cb is not None:
            progress_cb(0, 1, "quantify_phase NNLS")
        cp.quantify_phase(
            bv, k_max=float(self._kmax.get()),
            corr_kernel_size=0.04,
            sigma_excitation_error=0.02,
            power_intensity=0.25,
            power_intensity_experiment=0.25,
            max_number_patterns=1,
            allow_strain=False,
            progress_bar=False)
        return cp

    def _render_nnls_full_map(self, cp, stride, elapsed_s):
        """Render py4DSTEM CrystalPhase outputs as a 2×2:
            dominant phase / per-phase weight maps / residual."""
        self._fig.clf()
        try:
            fig_dom, _ = cp.plot_dominant_phase(
                figsize=(6, 6), returnfig=True,
                print_fractions=False, ticks=False,
                legend_add=True)
        except Exception as e:
            fig_dom = None
            self._set_status(f"plot_dominant_phase err: {e!r}")
        try:
            fig_pw, _ = cp.plot_phase_weights(
                figsize=(8, 4), returnfig=True)
        except Exception:
            fig_pw = None

        gs = self._fig.add_gridspec(2, 2, hspace=0.25, wspace=0.12)
        import io
        from matplotlib.image import imread
        def _embed(ax, fig_in, title):
            ax.clear(); ax.set_xticks([]); ax.set_yticks([])
            if fig_in is None:
                ax.text(0.5, 0.5, "(plot unavailable)",
                          ha="center", va="center", fontsize=10,
                          color="#888", transform=ax.transAxes)
                return
            buf = io.BytesIO()
            fig_in.savefig(buf, format="png", dpi=140,
                              bbox_inches="tight", facecolor="white")
            import matplotlib.pyplot as _plt
            _plt.close(fig_in)
            buf.seek(0)
            ax.imshow(imread(buf), aspect="equal")
            ax.set_title(title, fontsize=10)
            for s in ax.spines.values(): s.set_visible(False)

        ax_dom  = self._fig.add_subplot(gs[0, 0])
        ax_pw   = self._fig.add_subplot(gs[0, 1])
        ax_res  = self._fig.add_subplot(gs[1, 0])
        ax_rel  = self._fig.add_subplot(gs[1, 1])
        _embed(ax_dom, fig_dom,
                  "dominant phase (NNLS)")
        _embed(ax_pw, fig_pw,
                  "per-phase weight maps")
        # Residual + reliability directly via imshow.
        try:
            im = ax_res.imshow(cp.phase_residuals, cmap="magma",
                                  interpolation="nearest", aspect="equal")
            ax_res.set_title("phase fit residual (lower = better)",
                                fontsize=10)
            ax_res.set_xticks([]); ax_res.set_yticks([])
            self._fig.colorbar(im, ax=ax_res, fraction=0.045,
                                  pad=0.02)
        except Exception as e:
            ax_res.text(0.5, 0.5, repr(e), ha="center", va="center",
                          transform=ax_res.transAxes)
        try:
            im2 = ax_rel.imshow(cp.phase_reliability, cmap="viridis",
                                    interpolation="nearest",
                                    aspect="equal")
            ax_rel.set_title(
                "reliability (top-1 − top-2 phase weight)",
                fontsize=10)
            ax_rel.set_xticks([]); ax_rel.set_yticks([])
            self._fig.colorbar(im2, ax=ax_rel, fraction=0.045,
                                  pad=0.02)
        except Exception as e:
            ax_rel.text(0.5, 0.5, repr(e), ha="center", va="center",
                          transform=ax_rel.transAxes)
        self._fig.suptitle(
            f"NNLS full-dataset multi-phase fit  "
            f"phases={list(self._phase_crystals.keys())}  "
            f"stride={stride}  ({elapsed_s:.0f}s)",
            fontsize=11)
        self._fig.tight_layout(rect=[0, 0, 1, 0.96])
        # Stash on the panel so the user can save later.
        self._last_full_cp = cp
        self._canvas.draw_idle()

    def _render_classical_orientation_strain(self, crystal, omap, bv,
                                                    scan_shape, stride,
                                                    elapsed_s):
        """Classical py4DSTEM ACOM output: IPF orientation map +
        relative-rotation (strain) map, styled like the colab/PANDA
        notebooks — correlation-masked, diverging colormap centred on
        the median, robust range, nm scalebar.

        Renders into the main canvas as a 2-up.  Falls back gracefully
        if calculate_strain / plot_orientation_maps aren't available.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.anchored_artists import (
            AnchoredSizeBar)
        Ny, Nx = scan_shape

        # Compute the strain tensor FIRST so we know whether to lay
        # out the 2×2 strain grid (success) or a single rotation
        # panel (SVD fallback).
        strain_ok = False; strain = None
        try:
            strain = crystal.calculate_strain(
                bv, omap, min_num_peaks=3, rotation_range=np.pi / 2)
            strain = np.asarray(strain, dtype=float)
            strain_ok = strain.shape[0] >= 5
        except Exception as e:
            print(f"[acom] calculate_strain failed ({e!r}); "
                    f"falling back to in-plane rotation panel.",
                    flush=True)

        self._fig.clf()
        if strain_ok:
            # IPF spans the left column; strain 2×2 on the right.
            gs = self._fig.add_gridspec(2, 3, wspace=0.20, hspace=0.22,
                                              width_ratios=[1.5, 1, 1])
            ax_ipf = self._fig.add_subplot(gs[:, 0])
            ax_exx = self._fig.add_subplot(gs[0, 1])
            ax_eyy = self._fig.add_subplot(gs[0, 2])
            ax_exy = self._fig.add_subplot(gs[1, 1])
            ax_rot = self._fig.add_subplot(gs[1, 2])
        else:
            gs = self._fig.add_gridspec(1, 2, wspace=0.15)
            ax_ipf = self._fig.add_subplot(gs[0, 0])
            ax_rot = self._fig.add_subplot(gs[0, 1])
            ax_exx = ax_eyy = ax_exy = None

        # --- (1) IPF orientation map via py4DSTEM ---
        import io
        from matplotlib.image import imread
        try:
            # py4DSTEM returns (images_orientation, fig, ax) when
            # returnfig=True — the fig is element [1], not [0]
            # ([0] is the RGB image array).  Default corr_range is
            # [0,5]; keep it permissive.
            res = crystal.plot_orientation_maps(
                orientation_map=omap, orientation_ind=0,
                figsize=(12, 4), returnfig=True, progress_bar=False)
            ofig = None
            if isinstance(res, (tuple, list)):
                for el in res:
                    if hasattr(el, "savefig"):
                        ofig = el; break
            elif hasattr(res, "savefig"):
                ofig = res
            if ofig is None:
                raise RuntimeError("no figure handle in return")
            buf = io.BytesIO()
            ofig.savefig(buf, format="png", dpi=130,
                            bbox_inches="tight", facecolor="white")
            plt.close(ofig); buf.seek(0)
            ax_ipf.imshow(imread(buf), aspect="equal")
            ax_ipf.set_title("orientation (IPF + legend)", fontsize=11)
        except Exception as e:
            ax_ipf.text(0.5, 0.5, f"orientation map\nunavailable:\n{e!r}",
                          ha="center", va="center", fontsize=9,
                          color="#a33", transform=ax_ipf.transAxes)
        ax_ipf.set_xticks([]); ax_ipf.set_yticks([])

        # --- (2) strain tensor (or rotation fallback) ---
        def _scalebar(ax):
            try:
                nm = float(self.app.real_res.get()) if self.app else 0
                if nm > 0:
                    ax.add_artist(AnchoredSizeBar(
                        ax.transData, 50.0 / nm, "50 nm",
                        "lower right", pad=0.3, color="black",
                        frameon=False,
                        size_vertical=max(Ny * 0.01, 1)))
            except Exception:
                pass
        def _masked_panel(ax, data, mask, title, label, pct=False):
            """Diverging map, corr-masked, robust symmetric range."""
            d = np.full_like(data, np.nan, dtype=float)
            d[mask] = data[mask] - np.nanmedian(data[mask])
            if pct: d = d * 100.0      # strain → %
            fin = d[np.isfinite(d)]
            vmax = (float(np.percentile(np.abs(fin), 98))
                      if fin.size else 1.0)
            vmax = max(vmax, 1e-3)
            im = ax.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                              interpolation="nearest", aspect="equal")
            cb = self._fig.colorbar(im, ax=ax, fraction=0.046,
                                       pad=0.02)
            cb.set_label(label, fontsize=8)
            cb.ax.tick_params(labelsize=7)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        if strain_ok:
            # py4DSTEM strain: [e_xx, e_yy, e_xy, theta(rad), mask].
            exx = strain[0]; eyy = strain[1]; exy = strain[2]
            rot = np.degrees(strain[3])
            cmask = strain[4]
            mask = np.isfinite(rot) & (cmask >= 0.3)
            if not mask.any():
                mask = np.isfinite(rot)
            _masked_panel(ax_exx, exx, mask, "ε_xx", "strain (%)",
                            pct=True)
            _masked_panel(ax_eyy, eyy, mask, "ε_yy", "strain (%)",
                            pct=True)
            _masked_panel(ax_exy, exy, mask, "ε_xy", "shear (%)",
                            pct=True)
            # rotation: wrap to ±180, cap 90
            rotc = (rot + 180.0) % 360.0 - 180.0
            _masked_panel(ax_rot, rotc, mask,
                            "rotation θ", "Δθ (deg)")
            for a in (ax_exx, ax_eyy, ax_exy, ax_rot):
                _scalebar(a)
            sub = "strain tensor (ε_xx, ε_yy, ε_xy, θ)"
        else:
            # SVD failed → in-plane rotation from the ZA fit only.
            delta = None
            try:
                ang = np.asarray(getattr(omap, "angles", None),
                                    dtype=float)
                cv = np.asarray(getattr(omap, "corr", None), dtype=float)
                if ang.ndim == 4:
                    rot_deg = np.degrees(ang[:, :, 0, 2])
                    cm = (cv[..., 0] if cv.ndim == 3 else cv)
                else:
                    rot_deg = np.degrees(ang.reshape(Ny, Nx))
                    cm = np.ones_like(rot_deg)
                mask = np.isfinite(rot_deg)
                delta = np.full_like(rot_deg, np.nan)
                if mask.any():
                    delta[mask] = (rot_deg[mask]
                                     - np.nanmedian(rot_deg[mask]))
                    delta = (delta + 180.0) % 360.0 - 180.0
            except Exception as e2:
                ax_rot.text(0.5, 0.5, f"rotation unavailable:\n{e2!r}",
                              ha="center", va="center", fontsize=9,
                              color="#a33", transform=ax_rot.transAxes)
            if delta is not None:
                fin = delta[np.isfinite(delta)]
                vmax = float(min(max(
                    np.percentile(np.abs(fin), 98) if fin.size else 30,
                    2.0), 90.0))
                im = ax_rot.imshow(delta, cmap="RdBu_r",
                                      vmin=-vmax, vmax=vmax,
                                      interpolation="nearest",
                                      aspect="equal")
                cb = self._fig.colorbar(im, ax=ax_rot, fraction=0.045,
                                           pad=0.02)
                cb.set_label("Δθ (deg, rel. median)", fontsize=9)
                ax_rot.set_title(
                    "relative in-plane rotation (ZA fit; "
                    "strain SVD failed)", fontsize=10)
                _scalebar(ax_rot)
            ax_rot.set_xticks([]); ax_rot.set_yticks([])
            sub = "relative rotation (strain SVD failed)"

        self._fig.suptitle(
            f"Classical ACOM — orientation + {sub}  "
            f"(stride={stride}, {elapsed_s:.0f}s)",
            fontsize=11)
        self._fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._canvas.draw_idle()

    def _render_full_singlephase(self, omap, scan_shape, stride,
                                       elapsed_s):
        """Repaint the full 2×2 figure as the full-dataset render."""
        from gui_app.acom_core import zone_axis_from_matrix
        Ny, Nx = scan_shape
        corr = np.full((Ny, Nx), np.nan, dtype=np.float32)
        cv = getattr(omap, "corr", None)
        if cv is not None:
            v = np.asarray(cv)
            corr[:] = (v[..., 0] if v.ndim == 3 else v)
        mv = getattr(omap, "matrix", None)
        za_rgb = np.zeros((Ny, Nx, 3), dtype=np.float32)
        if mv is not None and np.asarray(mv).ndim >= 4:
            arr = np.asarray(mv)
            top = arr[..., 0, :, :] if arr.ndim == 5 else arr
            import matplotlib.pyplot as plt
            za_map = {}
            for rx in range(Ny):
                for ry in range(Nx):
                    R = top[rx, ry]
                    if not np.isfinite(R).all(): continue
                    za, _ = zone_axis_from_matrix(R)
                    if za not in za_map:
                        idx = len(za_map) % 10
                        za_map[za] = plt.get_cmap("tab10")(idx)[:3]
                    za_rgb[rx, ry] = za_map[za]
        self._fig.clf()
        gs = self._fig.add_gridspec(1, 2, wspace=0.12)
        ax1 = self._fig.add_subplot(gs[0, 0])
        ax2 = self._fig.add_subplot(gs[0, 1])
        im = ax1.imshow(corr, cmap="viridis", interpolation="nearest")
        ax1.set_title("top-1 correlation", fontsize=11)
        ax1.set_xticks([]); ax1.set_yticks([])
        self._fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.02)
        ax2.imshow(za_rgb, interpolation="nearest")
        ax2.set_title("dominant ZA (RGB hash)", fontsize=11)
        ax2.set_xticks([]); ax2.set_yticks([])
        self._fig.suptitle(
            f"ACOM full dataset  stride={stride}  "
            f"calib={float(self._inv_ang.get()):.5g} 1/Å/px  "
            f"({elapsed_s:.0f}s)", fontsize=11)
        self._fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._canvas.draw_idle()

    def _render_multiphase_map(self, mp, elapsed_s, stride):
        from gui_app.acom_core import zone_axis_from_matrix
        from matplotlib.colors import to_rgb
        Ny, Nx = mp["scan_shape"]
        names = mp["phase_names"]
        n_ph = len(names)
        phase_id = mp["phase_id"]
        corr_win = mp["winning_corr"]
        rmat_win = mp["winning_rmat"]
        palette = self._phase_palette(n_ph)
        phase_rgb = np.zeros((Ny, Nx, 3), dtype=np.float32)
        phase_rgb[phase_id == -2] = (0.45, 0.45, 0.45)
        for pi in range(n_ph):
            m = (phase_id == pi)
            if m.any(): phase_rgb[m] = to_rgb(palette[pi])
        za_rgb = np.zeros((Ny, Nx, 3), dtype=np.float32)
        import matplotlib.pyplot as plt
        za_map = {}
        for rx in range(Ny):
            for ry in range(Nx):
                pi = int(phase_id[rx, ry])
                if pi < 0: continue
                R = rmat_win[rx, ry]
                if not np.isfinite(R).all(): continue
                za, _ = zone_axis_from_matrix(R)
                key = (pi, za)
                if key not in za_map:
                    za_map[key] = plt.get_cmap("tab20")(len(za_map) % 20)[:3]
                za_rgb[rx, ry] = za_map[key]
        if n_ph >= 2:
            cp_sorted = np.sort(-mp["corr_per_phase"], axis=0)
            corr_diff = (-cp_sorted[0] - (-cp_sorted[1])).astype(np.float32)
        else:
            corr_diff = np.zeros((Ny, Nx), dtype=np.float32)
        self._fig.clf()
        gs = self._fig.add_gridspec(2, 2, hspace=0.20, wspace=0.10)
        a_mask = self._fig.add_subplot(gs[0, 0])
        a_corr = self._fig.add_subplot(gs[0, 1])
        a_za   = self._fig.add_subplot(gs[1, 0])
        a_diff = self._fig.add_subplot(gs[1, 1])
        a_mask.imshow(phase_rgb, interpolation="nearest")
        from matplotlib.patches import Patch
        lh = []
        for pi, n in enumerate(names):
            frac = float((phase_id == pi).sum()) / max(Ny * Nx, 1)
            lh.append(Patch(color=palette[pi],
                              label=f"{n} {frac*100:.1f}%"))
        f_n = float((phase_id == -1).sum()) / max(Ny * Nx, 1)
        if f_n > 0: lh.append(Patch(color="black",
                                       label=f"neither {f_n*100:.1f}%"))
        a_mask.legend(handles=lh, loc="lower right", fontsize=8,
                          framealpha=0.75)
        a_mask.set_title("phase mask", fontsize=11)
        a_mask.set_xticks([]); a_mask.set_yticks([])
        im = a_corr.imshow(corr_win, cmap="viridis",
                              interpolation="nearest")
        a_corr.set_title("winning correlation", fontsize=11)
        a_corr.set_xticks([]); a_corr.set_yticks([])
        self._fig.colorbar(im, ax=a_corr, fraction=0.045, pad=0.02)
        a_za.imshow(za_rgb, interpolation="nearest")
        a_za.set_title("winning ZA (phase × ZA hash)", fontsize=11)
        a_za.set_xticks([]); a_za.set_yticks([])
        im2 = a_diff.imshow(corr_diff, cmap="magma",
                                interpolation="nearest")
        a_diff.set_title("corr(top1) − corr(top2)", fontsize=11)
        a_diff.set_xticks([]); a_diff.set_yticks([])
        self._fig.colorbar(im2, ax=a_diff, fraction=0.045, pad=0.02)
        self._fig.suptitle(
            f"Multi-phase ACOM  phases={names}  "
            f"thr={float(self._mp_threshold.get()):.3f}  "
            f"margin={float(self._mp_margin.get()):.3f}  "
            f"stride={stride}  ({elapsed_s:.0f}s)",
            fontsize=11)
        self._fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._canvas.draw_idle()

    def _render_batch_cards_singlephase(self, patterns, labels,
                                              classes, results, zas, mode,
                                              phase_name="?"):
        N = len(patterns)
        cols = 4 if N > 6 else max(N, 1)
        rows = (N + cols - 1) // cols
        self._fig.clf()
        # Generous per-card footprint so titles don't collide.
        self._fig.set_size_inches(3.0 * cols, 3.4 * rows + 0.6)
        gs = self._fig.add_gridspec(rows, cols, hspace=0.45,
                                          wspace=0.12)
        # corr range for a green→grey quality cue on the border.
        corrs = [float(r["corr"]) for r in results
                    if np.isfinite(r["corr"])]
        cmax = max(corrs) if corrs else 1.0
        for k in range(N):
            ax = self._fig.add_subplot(gs[k // cols, k % cols])
            ax.imshow(np.log1p(np.clip(patterns[k], 0, None)),
                       cmap="inferno", interpolation="nearest",
                       aspect="equal")
            r = results[k]
            if r["peaks"].size:
                ax.scatter(r["peaks"][:, 1], r["peaks"][:, 0],
                            s=18, facecolors="none", edgecolors="cyan",
                            linewidths=0.8)
            za, mis = zas[k]
            cls = classes[k]
            corr = float(r["corr"])
            ax.set_xticks([]); ax.set_yticks([])
            # Border + title color: green = strong fit, grey = weak.
            q = (corr / cmax) if cmax > 0 and np.isfinite(corr) else 0
            if not np.isfinite(corr) or corr <= 0:
                col = "#999"; za_s = "no match"
            else:
                col = ("#2D7A2D" if q > 0.6
                          else "#c97c20" if q > 0.3 else "#999")
                za_s = f"[{za[0]} {za[1]} {za[2]}]"
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.2)
            # One-line title (label is below in smaller grey).
            ax.set_title(
                f"p{cls}  ·  {phase_name} {za_s}  ·  corr {corr:.2f}",
                fontsize=9, color=col, pad=4)
            ax.set_xlabel(labels[k], fontsize=7.5,
                             color="#666", labelpad=2)
        for k in range(N, rows * cols):
            ax = self._fig.add_subplot(gs[k // cols, k % cols])
            ax.set_axis_off()
        self._fig.suptitle(
            f"ACOM {mode}  ·  phase = {phase_name}  ·  N={N}  ·  "
            f"calib {float(self._inv_ang.get()):.5g} 1/Å/px  ·  "
            f"green=strong fit, grey=weak/none",
            fontsize=11, y=0.99)
        self._fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._canvas.draw_idle()

    def _render_phase_region_map(self, mp, region_masks, classes,
                                       labels, scan_shape, mode):
        """Two side-by-side region maps:

          LEFT  — PHASE map: each region (class or grain) painted by
                  its NNLS-winning phase (α/γ/…), neither=grey.
                  Clickable: L-click → single pattern, R-click →
                  grain-average diffraction.
          RIGHT — ZONE-AXIS map: same regions painted by their
                  integer zone axis [u v w], with an EXPLICIT legend
                  (colour ↔ 'phase [u v w]') and the [u v w] text
                  printed on the larger grains — far more intuitive
                  than py4DSTEM's continuous IPF colouring.
        """
        from matplotlib.colors import to_rgb
        from matplotlib.patches import Patch
        from gui_app.acom_core import zone_axis_from_matrix
        import matplotlib.pyplot as plt
        Ny, Nx = scan_shape
        names = list(mp["phase_names"])
        n_ph = len(names)
        palette = self._phase_palette(n_ph)
        NA = (0.55, 0.55, 0.55); AMB = (0.30, 0.30, 0.30)
        phase_id = mp["phase_id"]
        rmat_per = mp.get("rmat_per_phase")

        phase_rgb = np.zeros((Ny, Nx, 3), dtype=np.float32)
        za_rgb    = np.zeros((Ny, Nx, 3), dtype=np.float32)
        per_phase_pix  = [0] * n_ph
        per_phase_regs = [0] * n_ph
        na_regs = na_pix = amb_regs = amb_pix = 0
        za_color = {}          # (phase_idx, (u,v,w)) -> rgb
        za_count = {}          # same key -> #regions
        region_za = [None] * len(region_masks)   # per-region ZA text
        cmap20 = plt.get_cmap("tab20")
        for k, gm in enumerate(region_masks):
            pi = int(phase_id[k]); n_px = int(gm.sum())
            if pi == -1:
                phase_rgb[gm] = NA; za_rgb[gm] = NA
                na_regs += 1; na_pix += n_px; continue
            if pi == -2:
                phase_rgb[gm] = AMB; za_rgb[gm] = AMB
                amb_regs += 1; amb_pix += n_px; continue
            phase_rgb[gm] = to_rgb(palette[pi])
            per_phase_pix[pi] += n_px; per_phase_regs[pi] += 1
            # Zone axis for this region's winning phase.
            za = (0, 0, 0)
            try:
                R = rmat_per[pi, k]
                za, _ = zone_axis_from_matrix(R)
            except Exception:
                pass
            key = (pi, tuple(za))
            if key not in za_color:
                za_color[key] = cmap20(len(za_color) % 20)[:3]
                za_count[key] = 0
            za_count[key] += 1
            za_rgb[gm] = za_color[key]
            region_za[k] = (pi, za, gm)

        self._fig.clf()
        gs = self._fig.add_gridspec(1, 2, wspace=0.10)
        ax_ph = self._fig.add_subplot(gs[0, 0])
        ax_za = self._fig.add_subplot(gs[0, 1])
        what = ("class" if mode == "mp_classes" else "grain")

        # ---- LEFT: phase map ----
        ax_ph.imshow(phase_rgb, interpolation="nearest", aspect="equal")
        ax_ph.set_xticks([]); ax_ph.set_yticks([])
        ax_ph.set_title(f"PHASE  ({what}-level)\n"
                          f"L-click=single · R-click=grain avg",
                          fontsize=10)
        tot = Ny * Nx
        ph_handles = []
        for pi, nm in enumerate(names):
            ph_handles.append(Patch(
                color=palette[pi],
                label=f"{nm}  ·  {per_phase_regs[pi]} {what}s  ·  "
                      f"{per_phase_pix[pi]/tot*100:.0f}%"))
        if na_regs:
            ph_handles.append(Patch(color=NA,
                label=f"neither  ·  {na_regs}"))
        if amb_regs:
            ph_handles.append(Patch(color=AMB,
                label=f"ambiguous  ·  {amb_regs}"))
        ax_ph.legend(handles=ph_handles, loc="upper right",
                        fontsize=8, framealpha=0.85)

        # ---- RIGHT: zone-axis map + explicit legend + labels ----
        ax_za.imshow(za_rgb, interpolation="nearest", aspect="equal")
        ax_za.set_xticks([]); ax_za.set_yticks([])
        ax_za.set_title("ZONE AXIS per region  (colour ↔ [u v w])",
                          fontsize=10)
        # Print [u v w] text on the larger grains (top by area).
        sized = sorted(
            [(k, int(region_za[k][2].sum()), region_za[k])
               for k in range(len(region_masks))
               if region_za[k] is not None],
            key=lambda t: -t[1])
        for (k, npx, (pi, za, gm)) in sized[:25]:
            if npx < 25: break
            ys, xs = np.where(gm)
            cy, cx = float(ys.mean()), float(xs.mean())
            ax_za.text(cx, cy, f"{za[0]}{za[1]}{za[2]}",
                          ha="center", va="center", fontsize=6.5,
                          color="white", weight="bold",
                          path_effects=[])
        # Legend: colour ↔ 'phase [u v w]  (Nregions)', top 14 by count.
        za_items = sorted(za_color.items(),
                             key=lambda kv: -za_count[kv[0]])[:14]
        za_handles = [
            Patch(color=col,
                    label=f"{names[pi]} [{za[0]} {za[1]} {za[2]}]"
                          f"  ·  {za_count[(pi, za)]}")
            for (pi, za), col in za_items]
        ax_za.legend(handles=za_handles, loc="center left",
                        bbox_to_anchor=(1.01, 0.5), fontsize=8,
                        framealpha=1.0, title="phase  [zone axis]")

        # Scalebars.
        try:
            nm_per_px = float(self.app.real_res.get()) if self.app else 0
            if nm_per_px > 0:
                from gui_app._calib_utils import add_real_scalebar
                for a in (ax_ph, ax_za):
                    add_real_scalebar(a, nm_per_px, length_nm=100,
                                           color="white")
        except Exception:
            pass

        # Click handler on BOTH maps → grain/single diffraction.
        self._pmap_axes = (ax_ph, ax_za)
        self._pmap_scan = (Ny, Nx)
        self._pmap_phase_id = phase_id
        self._pmap_region_masks = region_masks
        self._pmap_names = list(names)
        if getattr(self, "_pmap_click_cid", None) is not None:
            try: self._canvas.mpl_disconnect(self._pmap_click_cid)
            except Exception: pass
        self._pmap_click_cid = self._canvas.mpl_connect(
            "button_press_event", self._on_phase_map_click)

        self._fig.suptitle(
            f"Multi-phase ACOM {mode} — phase + zone-axis  "
            f"(calib {float(self._inv_ang.get()):.5g} 1/Å/px)",
            fontsize=11)
        self._fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._canvas.draw_idle()

    def _on_phase_map_click(self, event):
        """L-click → single scan-position pattern; R-click → grain
        average.  Pops a diffraction window with hover-q + the phase
        the clicked region was assigned to."""
        if event.inaxes not in getattr(self, "_pmap_axes", ()):
            return
        if event.xdata is None or event.ydata is None:
            return
        Ny, Nx = self._pmap_scan
        x = max(0, min(Nx - 1, int(round(event.xdata))))
        y = max(0, min(Ny - 1, int(round(event.ydata))))
        # Which region/phase was this pixel assigned to?
        phase_txt = "—"
        try:
            for k, gm in enumerate(self._pmap_region_masks):
                if gm[y, x]:
                    pi = int(self._pmap_phase_id[k])
                    if pi == -1:   phase_txt = "neither"
                    elif pi == -2: phase_txt = "ambiguous"
                    else:          phase_txt = self._pmap_names[pi]
                    break
        except Exception:
            pass
        ph = self._posthoc()
        if ph is None or ph.sample is None:
            return
        right = (event.button == 3)
        try:
            if right:
                gi = ph._compute_grain_average(y, x)
                if gi is None:
                    self._set_status("no grain at that pixel"); return
                pat = gi["grain_avg"].astype(np.float32)
                title = (f"grain @ ({y},{x})  p{gi['cls']}  "
                           f"{gi['n_pix']}px  →  phase: {phase_txt}")
            else:
                from data import SAMPLES, LoadPRZ
                cfg = SAMPLES[ph.sample]
                ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
                pat = ds.get_raw(y * Nx + x).astype(np.float32)
                title = (f"single ({y},{x})  →  phase: {phase_txt}")
        except Exception as e:
            messagebox.showerror("inspect", repr(e)); return
        self._popup_diffraction(pat, title)

    def _popup_diffraction(self, pat, title):
        """Small diffraction popup with log toggle + hover-q.

        Uses PLAIN tkinter widgets only — mixing customtkinter widgets
        as siblings of a matplotlib FigureCanvasTkAgg triggers a CTk
        resize crash ('FigureCanvasTkAgg has no attribute
        winfo_exists').
        """
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from gui_app._ui import attach_hover_q
        win = tk.Toplevel(self)
        win.title(title); win.geometry("620x660")
        ctrl = tk.Frame(win, bg="#f4f4f4")
        ctrl.pack(side="top", fill="x")
        log_var = tk.BooleanVar(value=True)
        tk.Label(ctrl, text=title, font=("Consolas", 9),
                   bg="#f4f4f4", anchor="w").pack(side="left", padx=6,
                                                     pady=4)
        fig = Figure(figsize=(6, 6), dpi=110, facecolor="white")
        ax = fig.add_subplot(111)
        rp = self._recip_per_px()
        canvas_holder = tk.Frame(win)
        canvas_holder.pack(side="top", fill="both", expand=True)
        canv = FigureCanvasTkAgg(fig, master=canvas_holder)
        canv.get_tk_widget().pack(fill="both", expand=True)
        def _draw():
            ax.clear()
            img = (np.log1p(np.clip(pat, 0, None)) if log_var.get()
                     else np.clip(pat, 0, None))
            ax.imshow(img, cmap="inferno", aspect="equal",
                        interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=9)
            canv.draw_idle()
        tk.Checkbutton(ctrl, text="log", variable=log_var,
                         command=_draw, bg="#f4f4f4").pack(side="right",
                                                              padx=8)
        tb_frame = tk.Frame(win)
        tb_frame.pack(side="bottom", fill="x")
        tb = NavigationToolbar2Tk(canv, tb_frame, pack_toolbar=False)
        tb.update(); tb.pack(side="left", fill="x")
        _draw()
        if rp > 0:
            H, W = pat.shape
            attach_hover_q(canv, ax, center=(H / 2.0, W / 2.0),
                              q_per_disp_px=rp, units="nm⁻¹")

    def _render_multiphase_cards(self, mp, patterns, labels, classes,
                                       mode):
        from gui_app.acom_core import zone_axis_from_matrix
        names = mp["phase_names"]
        phase_id = mp["phase_id"]
        corr_per = mp["corr_per_phase"]
        rmat_per = mp["rmat_per_phase"]
        palette = self._phase_palette(len(names))
        N = len(patterns); cols = 4
        rows = (N + cols - 1) // cols
        self._fig.clf()
        self._fig.set_size_inches(2.7 * cols, 3.0 * rows + 0.5)
        gs = self._fig.add_gridspec(rows, cols)
        for k in range(N):
            ax = self._fig.add_subplot(gs[k // cols, k % cols])
            ax.imshow(np.log1p(np.clip(patterns[k], 0, None)),
                       cmap="inferno", interpolation="nearest",
                       aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            pi = int(phase_id[k])
            if pi < 0:
                tag = "neither" if pi == -1 else "ambiguous"
                cs = "  ".join(f"{names[p]}={corr_per[p, k]:.3f}"
                                  for p in range(len(names)))
                ax.set_title(f"{tag}  ({cs})\n{labels[k]}",
                              fontsize=8, color="#888")
            else:
                R = rmat_per[pi, k]
                za, _ = zone_axis_from_matrix(R)
                ax.set_title(
                    f"{names[pi]}  ZA=[{za[0]} {za[1]} {za[2]}]  "
                    f"corr={corr_per[pi, k]:.3f}\n{labels[k]}",
                    fontsize=8, color=palette[pi])
                for spine in ax.spines.values():
                    spine.set_edgecolor(palette[pi])
                    spine.set_linewidth(2.0)
        for k in range(N, rows * cols):
            ax = self._fig.add_subplot(gs[k // cols, k % cols])
            ax.set_axis_off()
        self._fig.suptitle(
            f"Multi-phase ACOM {mode}  phases={names}  "
            f"thr={float(self._mp_threshold.get()):.3f}",
            fontsize=11)
        self._fig.tight_layout(rect=[0, 0, 1, 0.97])
        self._canvas.draw_idle()
