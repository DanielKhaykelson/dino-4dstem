"""acom_panel.py -- Tab 5 (ACOM via py4DSTEM).

GUI port of the WS2 ACOM/strain notebook
(orient_strain_01_WS2.ipynb).  The pipeline is:

    1. Wrap the loaded .prz cube as a py4DSTEM DataCube
       (we share PrePanel.cube directly when the path matches, so the
       file is not re-opened or re-mmapped)
    2. Virtual imaging  →  dp_max, dp_mean, get_probe_size, virtual ADF
    3. Vacuum probe template
       probe = DataCube.get_vacuum_probe(ROI=(rx0,rx1,ry0,ry1))
       probe.get_kernel(mode='sigmoid', radii=(0, 4*semiangle))
    4. Bragg disk detection (template matching)
       dataset.find_Bragg_disks(template=probe.kernel, **detect_params)
       — preview with 10 DRAGGABLE markers on the HAADF;
         drop the marker, the corresponding diffraction tile + detected
         disks redraw.
    5. Centering + calibration
       bragg_peaks.measure_origin() / fit_origin()
       bragg_peaks.calibration.set_Q_pixel_size(inv_Ang_per_pixel)
    6. Crystal + structure factors  (CIF → calculate_structure_factors)
    7. Orientation plan  (corners or 'fiber' mode)
    8. Match orientations  (full scan)
    9. Strain map  (calculate_strain + show_strain)

All heavy work runs in worker threads.  Intermediates cache under
<run-outdir>/acom/<sample>/ so re-opening the tab doesn't redo them.
"""
from __future__ import annotations
import os, sys, json, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg", force=True)
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                  NavigationToolbar2Tk)

from data import SAMPLES, LoadPRZ
from gui_app.tooltip import add_help_button


# ---------------------------------------------------------------------------
def _try_import_py4DSTEM():
    try:
        import py4DSTEM
        from py4DSTEM.process.diffraction import Crystal
        return py4DSTEM, Crystal
    except Exception:
        return None, None


def _open_lazy(path, scan_shape=None):
    """Lazy 4D-cube loader (.prz/.npz/.npy/.h5 + Dectris masters).
    Delegates to data.open_lazy_cube so every panel shares one loader."""
    from data import open_lazy_cube
    return open_lazy_cube(path, scan_shape=scan_shape)


_TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------

class ACOMPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app

        py4dstem, Crystal = _try_import_py4DSTEM()
        self._py4dstem = py4dstem
        self._Crystal = Crystal

        # ── pipeline state ──────────────────────────────────────────────
        # data
        self.sample: "str | None" = None
        self._cube_path: "str | None" = None
        self._cube_mmap = None              # shared with PrePanel if possible
        self.datacube = None                # py4DSTEM DataCube wrapper
        # virtual imaging
        self.dp_max = None
        self.dp_mean = None
        self.virtual_adf = None             # (Ny, Nx) np array
        self.probe_semiangle = None         # px
        self.probe_qx0 = None               # px
        self.probe_qy0 = None               # px
        # probe template
        self._probe = None                  # py4DSTEM Probe object
        self._probe_kernel = None           # 2D np array
        # bragg detection
        self.bragg_peaks = None             # py4DSTEM BraggVectors
        self._bvm = None
        self._origin_fit = None             # (qx0_fit, qy0_fit)
        self._calibrated = False
        # crystal + ACOM
        self.cif_path: "str | None" = None
        self.crystal = None
        self.orientation_map = None
        self.strain_map = None
        # 10-frame preview drag state
        self._preview_positions = []        # list[ (rx, ry) ] length 10
        self._preview_artists = []          # marker artists on HAADF
        self._preview_axes = []             # diffraction tile axes
        self._preview_disks = []            # one Line2D per tile
        self._dragging = None               # idx being dragged, or None
        self._preview_cids = []             # mpl callback ids
        self._haadf_ax = None
        # threading
        self._busy = False

        self._build()

    # ------------------------------------------------------------------
    # cube discovery (share with PrePanel)
    # ------------------------------------------------------------------
    def _resolve_cube_for_sample(self, sample):
        """Pick the right cube source for `sample`. Returns the mmap'd
        numpy array of shape (Ny, Nx, H, W). Reuses PrePanel.cube when
        the sample's path matches what's already loaded in Tab 1."""
        if sample not in SAMPLES:
            raise RuntimeError(f"unknown sample: {sample}")
        cfg = SAMPLES[sample]
        path = cfg.get("path") or (cfg.get("paths") or [None])[0]
        if path is None or not os.path.exists(path):
            raise RuntimeError(f"no .prz path for sample {sample}: {path!r}")
        # PrePanel may already have this exact file mmapped — reuse.
        try:
            pp = getattr(self.app, "pre", None) if self.app else None
            if pp is not None and getattr(pp, "cube", None) is not None:
                if os.path.abspath(getattr(pp, "path", "")) == os.path.abspath(path):
                    return pp.cube, path
        except Exception:
            pass
        return _open_lazy(path), path

    # ------------------------------------------------------------------
    # UI scaffolding
    # ------------------------------------------------------------------
    def _build(self):
        # Top: CIF + sample + crystal info
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)
        ctk.CTkLabel(top, text="CIF:").pack(side="left", padx=(6, 4))
        self._cif_var = ctk.StringVar()
        ctk.CTkEntry(top, textvariable=self._cif_var, width=400).pack(
            side="left", padx=2)
        ctk.CTkButton(top, text="Browse", width=70,
                       command=self._browse_cif).pack(side="left", padx=2)
        ctk.CTkButton(top, text="Build crystal", width=110,
                       command=self._build_crystal).pack(side="left",
                                                          padx=(2, 12))
        ctk.CTkLabel(top, text="Sample:").pack(side="left", padx=(8, 4))
        self._sample_var = ctk.StringVar(value="Na007b")
        self._sample_menu = ctk.CTkOptionMenu(top, variable=self._sample_var,
                            values=sorted(SAMPLES.keys()), width=180,
                            command=lambda _v: self._on_sample_change())
        self._sample_menu.pack(side="left", padx=2)

        # Crystal info display
        info_frame = ctk.CTkFrame(self)
        info_frame.pack(side="top", fill="x", padx=6, pady=(0, 4))
        self._crystal_info = ctk.CTkLabel(info_frame,
            text="(no crystal built)", font=("Consolas", 10), anchor="w",
            justify="left")
        self._crystal_info.pack(side="left", padx=10, pady=4)

        # Body: scrollable sidebar + figure
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        sidebar = ctk.CTkScrollableFrame(body, width=340)
        sidebar.pack(side="left", fill="y")

        self._build_sidebar(sidebar)

        # canvas
        canvas_holder = ctk.CTkFrame(body)
        canvas_holder.pack(side="left", fill="both", expand=True,
                            padx=(8, 0))
        import tkinter as tk
        toolbar_frame = tk.Frame(canvas_holder, bg="#f4f4f4")
        toolbar_frame.pack(side="top", fill="x")
        self._fig = Figure(figsize=(11, 7), dpi=95, facecolor="#f4f4f4")
        if self._py4dstem is None:
            self._fig.text(0.5, 0.5,
                "py4DSTEM is not installed in this environment.\n"
                "pip install py4DSTEM",
                ha="center", va="center", fontsize=12, color="#a00")
        else:
            self._fig.text(0.5, 0.5,
                "Step 1 → Step 9, top-down on the left.\n"
                "Build crystal · run virtual imaging · pick vacuum ROI · "
                "build kernel · detect disks · fit origin · calibrate · "
                "orientation plan · match · strain.",
                ha="center", va="center", fontsize=11, color="#666")
        self._canvas = FigureCanvasTkAgg(self._fig, master=canvas_holder)
        self._canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame,
                                               pack_toolbar=False)
        self._toolbar.update(); self._toolbar.pack(side="left")

        # status (sidebar bottom would clip in scrollable frame; put it
        # in the canvas footer instead)
        self._sb = ctk.CTkLabel(canvas_holder, text="",
                                  font=("Consolas", 9), anchor="w",
                                  justify="left", wraplength=900)
        self._sb.pack(side="bottom", fill="x", padx=4, pady=2)

    def _build_sidebar(self, sb):
        # ── 1. DataCube ────────────────────────────────────────────────
        self._heading(sb, "1. DataCube")
        ctk.CTkButton(sb, text="Wrap cube as py4DSTEM DataCube",
                       width=300, command=self._step_build_datacube
                       ).pack(anchor="w", padx=10, pady=2)

        # ── 2. Virtual imaging ─────────────────────────────────────────
        self._heading(sb, "2. Virtual imaging")
        self._adf_inner = ctk.IntVar(value=13)
        self._adf_outer = ctk.IntVar(value=24)
        self._row(sb, "ADF inner radius (px)",
            "Inner radius of the virtual annular detector (in detector "
            "pixels). Default 13 — the WS2 notebook value; adjust to "
            "match your first diffraction ring.",
            self._adf_inner)
        self._row(sb, "ADF outer radius (px)",
            "Outer radius of the virtual annular detector.",
            self._adf_outer)
        ctk.CTkButton(sb, text="Run virtual imaging  (dp_max + ADF)",
                       width=300, command=self._step_virtual_imaging
                       ).pack(anchor="w", padx=10, pady=2)

        # ── 3. Vacuum probe template ───────────────────────────────────
        self._heading(sb, "3. Vacuum probe template")
        self._roi_rx0 = ctk.IntVar(value=10)
        self._roi_rx1 = ctk.IntVar(value=30)
        self._roi_ry0 = ctk.IntVar(value=10)
        self._roi_ry1 = ctk.IntVar(value=30)
        self._row(sb, "vacuum ROI rx0",
            "Real-space ROI used to average a vacuum probe template. "
            "Pick a region with no specimen.", self._roi_rx0)
        self._row(sb, "vacuum ROI rx1", "(exclusive)", self._roi_rx1)
        self._row(sb, "vacuum ROI ry0", "", self._roi_ry0)
        self._row(sb, "vacuum ROI ry1", "(exclusive)", self._roi_ry1)
        ctk.CTkButton(sb, text="Build vacuum probe + sigmoid kernel",
                       width=300, command=self._step_build_probe
                       ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sb, text="Show probe kernel",
                       width=300, command=self._render_probe_kernel
                       ).pack(anchor="w", padx=10, pady=2)

        # ── 4. Bragg disk detection ────────────────────────────────────
        self._heading(sb, "4. Bragg disk detection (template matching)")
        # WS2-notebook defaults
        self._dp_corrPower            = ctk.DoubleVar(value=1.0)
        self._dp_sigma                = ctk.DoubleVar(value=0.0)
        self._dp_edgeBoundary         = ctk.IntVar(value=4)
        self._dp_minRelIntensity      = ctk.DoubleVar(value=0.0)
        self._dp_minAbsIntensity      = ctk.DoubleVar(value=1e-7)
        self._dp_minPeakSpacing       = ctk.IntVar(value=12)
        self._dp_subpixel             = ctk.StringVar(value="poly")
        self._dp_upsample_factor      = ctk.IntVar(value=8)
        self._dp_maxNumPeaks          = ctk.IntVar(value=1000)
        self._row(sb, "corrPower",
            "γ-power applied to the cross-correlation map before peak "
            "finding. 1.0 = standard.", self._dp_corrPower)
        self._row(sb, "sigma (px)",
            "Pre-Gaussian smoothing σ on each pattern before correlation.",
            self._dp_sigma)
        self._row(sb, "edgeBoundary",
            "Pixels to ignore around the pattern edge.",
            self._dp_edgeBoundary)
        self._row(sb, "minRelativeIntensity",
            "Minimum disk intensity relative to the brightest disk on the "
            "same pattern. 0 = keep all.", self._dp_minRelIntensity)
        self._row(sb, "minAbsoluteIntensity",
            "Minimum disk intensity (absolute units).",
            self._dp_minAbsIntensity)
        self._row(sb, "minPeakSpacing (px)",
            "Minimum centre-to-centre spacing between disks.",
            self._dp_minPeakSpacing)
        self._row(sb, "subpixel  ('poly' or 'multicorr')",
            "Sub-pixel refinement method.", self._dp_subpixel, width=110)
        self._row(sb, "upsample_factor",
            "Subpixel upsample factor (multicorr only).",
            self._dp_upsample_factor)
        self._row(sb, "maxNumPeaks",
            "Maximum number of disks to keep per pattern.",
            self._dp_maxNumPeaks)
        ctk.CTkButton(sb,
            text="Preview on 10 frames  (drag markers!)",
            width=300, command=self._step_detect_preview
            ).pack(anchor="w", padx=10, pady=(6, 2))
        ctk.CTkButton(sb, text="Detect on full cube",
                       width=300, command=self._step_detect_full
                       ).pack(anchor="w", padx=10, pady=2)

        # ── 5. Centering + calibration ─────────────────────────────────
        self._heading(sb, "5. Centering + calibration")
        ctk.CTkButton(sb, text="Measure + fit origin",
                       width=300, command=self._step_fit_origin
                       ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sb, text="Show BVM (Bragg vector map)",
                       width=300, command=self._step_show_bvm
                       ).pack(anchor="w", padx=10, pady=2)
        # default 1/Å/px from the topbar's recip_res (nm⁻¹/px → /10)
        default_inv_ang = 0.01
        try:
            if self.app and float(self.app.recip_res.get()) > 0:
                default_inv_ang = float(self.app.recip_res.get()) / 10.0
        except Exception:
            pass
        self._inv_ang_per_pixel = ctk.DoubleVar(value=default_inv_ang)
        self._row(sb, "inv_Å per pixel",
            "Reciprocal-space pixel size in 1/Å (for matching). Auto-"
            "filled from the topbar's nm⁻¹/px (÷10). Override per dataset.",
            self._inv_ang_per_pixel)
        ctk.CTkButton(sb, text="Apply pixel-size calibration",
                       width=300, command=self._step_apply_calibration
                       ).pack(anchor="w", padx=10, pady=2)

        # ── 6. Crystal & structure factors ─────────────────────────────
        self._heading(sb, "6. Crystal + structure factors")
        self._k_max = ctk.DoubleVar(value=1.4)
        self._row(sb, "k_max (1/Å)",
            "Maximum reciprocal-space radius for structure factors.",
            self._k_max)
        ctk.CTkButton(sb, text="Calculate structure factors",
                       width=300, command=self._step_struct_factors
                       ).pack(anchor="w", padx=10, pady=2)

        # ── 7. Orientation plan ────────────────────────────────────────
        self._heading(sb, "7. Orientation plan")
        self._plan_mode = ctk.StringVar(value="corners")
        ctk.CTkRadioButton(sb, text="corners  (3-vector range)",
            variable=self._plan_mode, value="corners"
            ).pack(anchor="w", padx=10, pady=1)
        ctk.CTkRadioButton(sb, text="fiber  (textured 2D, e.g. WS2)",
            variable=self._plan_mode, value="fiber"
            ).pack(anchor="w", padx=10, pady=1)
        self._zone_axis_range = ctk.StringVar(
            value="[0,0,1] [1,1,1] [0,1,1]")
        self._fiber_axis = ctk.StringVar(value="[0,0,1]")
        self._fiber_angles = ctk.StringVar(value="[0,0]")
        self._angle_step_za = ctk.DoubleVar(value=2.0)
        self._angle_step_ip = ctk.DoubleVar(value=2.0)
        self._cor_kernel = ctk.DoubleVar(value=0.04)
        self._row(sb, "zone_axis_range  (corners mode)",
            "Three corner zone axes for the stereographic triangle.",
            self._zone_axis_range, width=180)
        self._row(sb, "fiber_axis  (fiber mode)",
            "Texture axis for fiber-mode plan, e.g. [0,0,1] for c-axis.",
            self._fiber_axis, width=110)
        self._row(sb, "fiber_angles  (fiber mode)",
            "[α, β] tilt angles in degrees. [0,0] = perfect c-axis fibre.",
            self._fiber_angles, width=110)
        self._row(sb, "angle_step_zone_axis (°)",
            "Angular step over zone axes.", self._angle_step_za)
        self._row(sb, "angle_step_in_plane (°)",
            "Angular step for in-plane rotation.", self._angle_step_ip)
        self._row(sb, "corr_kernel_size (1/Å)",
            "Gaussian σ used by orientation matching.",
            self._cor_kernel)
        ctk.CTkButton(sb, text="Generate orientation plan",
                       width=300, command=self._step_generate_plan
                       ).pack(anchor="w", padx=10, pady=2)

        # ── 8. Match orientations ──────────────────────────────────────
        self._heading(sb, "8. Match orientations")
        self._test_rx = ctk.IntVar(value=32)
        self._test_ry = ctk.IntVar(value=32)
        self._row(sb, "test rx (single-pattern)",
            "Real-space x of the test pattern for match_single_pattern.",
            self._test_rx)
        self._row(sb, "test ry (single-pattern)", "",
            self._test_ry)
        ctk.CTkButton(sb, text="Match single (test rx, ry)",
                       width=300, command=self._step_match_single
                       ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sb, text="Match all  (full scan, slow)",
                       width=300, command=self._step_match_all
                       ).pack(anchor="w", padx=10, pady=2)

        # ── 9. Strain ──────────────────────────────────────────────────
        self._heading(sb, "9. Strain map")
        self._rotation_range = ctk.DoubleVar(value=60.0)
        self._strain_vrange  = ctk.DoubleVar(value=3.0)
        self._theta_vrange   = ctk.DoubleVar(value=60.0)
        self._row(sb, "rotation_range (°)",
            "Search range for the in-plane rotation when fitting strain. "
            "60° fits hexagonal symmetry.", self._rotation_range)
        self._row(sb, "vrange_exx (%)",
            "Symmetric ± range for the εxx / εyy / εxy display.",
            self._strain_vrange)
        self._row(sb, "vrange_theta (°)",
            "Symmetric ± range for the in-plane rotation display.",
            self._theta_vrange)
        ctk.CTkButton(sb, text="Calculate strain",
                       width=300, command=self._step_calculate_strain
                       ).pack(anchor="w", padx=10, pady=2)
        ctk.CTkButton(sb, text="Show strain (4-panel)",
                       width=300, command=self._render_strain
                       ).pack(anchor="w", padx=10, pady=2)

        # ── Renders ────────────────────────────────────────────────────
        self._heading(sb, "Other renders")
        for label, cmd in [
            ("DP max + probe circle", self._render_dp_max),
            ("Virtual ADF",           self._render_adf),
            ("Orientation map (RGB)", self._render_orientation),
            ("Correlation max",       self._render_correlation),
            ("Plan zone-axis preview",self._render_plan_preview),
        ]:
            ctk.CTkButton(sb, text=label, width=300, command=cmd
                           ).pack(anchor="w", padx=10, pady=1)

    # ------------------------------------------------------------------
    # small UI helpers
    # ------------------------------------------------------------------
    def _heading(self, parent, text):
        ctk.CTkLabel(parent, text=text,
                      font=("Segoe UI", 12, "bold")
                      ).pack(anchor="w", padx=10, pady=(10, 2))

    def _row(self, parent, label, help_text, var, width=110):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=2)
        ctk.CTkLabel(row, text=label, width=200, anchor="w").pack(side="left")
        ctk.CTkEntry(row, textvariable=var, width=width).pack(side="left")
        if help_text:
            h = add_help_button(row, help_text); h.pack(side="left",
                                                          padx=(6, 0))

    def _set_status(self, msg):
        self._sb.configure(text=msg)
        self.update_idletasks()

    def _new_fig(self):
        self._disconnect_preview()
        self._fig.clear()
        return self._fig

    def _redraw(self, analysis=None):
        self._fig.tight_layout()
        self._canvas.draw_idle()
        # Auto-save key ACOM results next to the loaded data.
        if analysis:
            try:
                import assistant_io
                assistant_io.gui_autosave(
                    self, "acom", self._fig, name=analysis,
                    summary=f"ACOM {analysis} — sample "
                            f"{getattr(self, '_sample_var', None) and self._sample_var.get()}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # runtime-sample notifications
    # ------------------------------------------------------------------
    def on_runtime_sample_added(self, key):
        try:
            keys = sorted(SAMPLES.keys())
            if hasattr(self, "_sample_menu"):
                self._sample_menu.configure(values=keys)
            if key in keys:
                self._sample_var.set(key)
                self._on_sample_change()
        except Exception:
            pass

    def _on_sample_change(self):
        # invalidate per-sample state so the user re-runs the pipeline
        self.datacube = None
        self.dp_max = None; self.dp_mean = None
        self.virtual_adf = None
        self._probe = None; self._probe_kernel = None
        self.bragg_peaks = None; self._bvm = None
        self._origin_fit = None; self._calibrated = False
        self.orientation_map = None; self.strain_map = None
        self._set_status(f"sample = {self._sample_var.get()}; pipeline "
                          f"state cleared.")

    # ------------------------------------------------------------------
    # CIF + crystal
    # ------------------------------------------------------------------
    def _browse_cif(self):
        p = filedialog.askopenfilename(
            filetypes=[("CIF", "*.cif"), ("All", "*.*")])
        if p:
            self._cif_var.set(p)

    def _build_crystal(self):
        if self._py4dstem is None:
            messagebox.showerror("py4DSTEM",
                "py4DSTEM is not available."); return
        p = self._cif_var.get().strip()
        if not p or not os.path.exists(p):
            messagebox.showerror("CIF", f"Not found: {p}"); return
        try:
            try:
                self.crystal = self._Crystal.from_CIF(p)
            except AttributeError:
                self.crystal = self._Crystal(filepath=p)
        except Exception as e:
            messagebox.showerror("Crystal build failed", str(e))
            return
        self.cif_path = p
        info_lines = [f"CIF: {os.path.basename(p)}"]
        for attr in ("a", "b", "c", "alpha", "beta", "gamma"):
            v = getattr(self.crystal, attr, None)
            if v is not None:
                try: info_lines.append(f"  {attr} = {float(v):.4f}")
                except Exception: pass
        for attr in ("numbers", "positions", "spaceGroupNumber"):
            v = getattr(self.crystal, attr, None)
            if v is not None:
                if hasattr(v, "shape"):
                    info_lines.append(f"  {attr}: shape={v.shape}")
                else:
                    info_lines.append(f"  {attr}: {v}")
        self._crystal_info.configure(text="    ".join(info_lines))
        self._set_status("crystal built. Step 6 to compute structure factors.")

    # ------------------------------------------------------------------
    # Step 1: build DataCube
    # ------------------------------------------------------------------
    def _step_build_datacube(self):
        if self._py4dstem is None:
            messagebox.showerror("py4DSTEM", "py4DSTEM is not installed.")
            return
        sample = self._sample_var.get()
        try:
            cube, path = self._resolve_cube_for_sample(sample)
        except Exception as e:
            messagebox.showerror("cube", str(e)); return
        # py4DSTEM.DataCube wants a real numpy array; mmap views are fine.
        try:
            DC = getattr(self._py4dstem, "DataCube", None)
            if DC is None:
                from py4DSTEM.datacube import DataCube as DC
            self.datacube = DC(data=np.asarray(cube))
        except Exception as e:
            messagebox.showerror("DataCube build failed", repr(e)); return
        self.sample = sample
        self._cube_mmap = cube; self._cube_path = path
        Ny, Nx, H, W = cube.shape
        self._set_status(
            f"DataCube ready: {sample}  shape={cube.shape}  path={path}\n"
            f"(reused PrePanel cube)" if (
                self.app and getattr(self.app, "pre", None) is not None
                and getattr(self.app.pre, "cube", None) is cube
            ) else
            f"DataCube ready: {sample}  shape={cube.shape}  "
            f"path={os.path.basename(path)}")

    def _ensure_datacube(self):
        if self.datacube is None:
            self._step_build_datacube()
        return self.datacube is not None

    # ------------------------------------------------------------------
    # Step 2: virtual imaging
    # ------------------------------------------------------------------
    def _step_virtual_imaging(self):
        if not self._ensure_datacube(): return
        if self._busy: return
        self._busy = True
        self._set_status("computing dp_max, dp_mean, probe size, and ADF…")
        threading.Thread(target=self._virtual_worker, daemon=True).start()

    def _virtual_worker(self):
        try:
            ds = self.datacube
            ds.get_dp_max(); ds.get_dp_mean()
            try:
                self.dp_max  = np.asarray(ds.tree("dp_max").data
                                            if hasattr(ds.tree("dp_max"),
                                                        "data")
                                            else ds.tree("dp_max"))
            except Exception:
                self.dp_max  = np.asarray(ds.tree("dp_max"))
            try:
                self.dp_mean = np.asarray(ds.tree("dp_mean").data
                                            if hasattr(ds.tree("dp_mean"),
                                                        "data")
                                            else ds.tree("dp_mean"))
            except Exception:
                self.dp_mean = np.asarray(ds.tree("dp_mean"))

            # probe size
            ps = ds.get_probe_size(self.dp_mean)
            self.probe_semiangle = float(ps[0])
            self.probe_qx0 = float(ps[1])
            self.probe_qy0 = float(ps[2])

            # virtual ADF
            r_in  = int(self._adf_inner.get())
            r_out = int(self._adf_outer.get())
            try:
                ds.get_virtual_image(
                    mode="annulus",
                    geometry=((self.probe_qx0, self.probe_qy0),
                                (r_in, r_out)),
                    name="virtual_adf",
                )
                self.virtual_adf = np.asarray(
                    ds.tree("virtual_adf").data
                    if hasattr(ds.tree("virtual_adf"), "data")
                    else ds.tree("virtual_adf"))
            except Exception as e:
                # fallback: compute it ourselves on the mmap'd cube
                Ny, Nx, H, W = self._cube_mmap.shape
                yy, xx = np.ogrid[:H, :W]
                r2 = (yy - self.probe_qx0) ** 2 + (xx - self.probe_qy0) ** 2
                mask = (r2 >= r_in ** 2) & (r2 < r_out ** 2)
                adf = np.zeros((Ny, Nx), dtype=np.float64)
                for y in range(Ny):
                    block = np.asarray(self._cube_mmap[y]).astype(np.float32)
                    adf[y] = (block * mask).sum(axis=(1, 2))
                self.virtual_adf = adf
            self.after(0, lambda: self._set_status(
                f"virtual imaging done.  probe_semiangle = "
                f"{self.probe_semiangle:.2f} px,  centre = "
                f"({self.probe_qx0:.2f}, {self.probe_qy0:.2f})"))
            self.after(0, self._render_dp_max)
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "virtual imaging failed", err))
            self.after(0, lambda: self._set_status(
                "virtual imaging failed."))
        finally:
            self._busy = False

    def _render_dp_max(self):
        if self.dp_max is None:
            messagebox.showinfo("virtual",
                "Run Step 2 (Virtual imaging) first."); return
        fig = self._new_fig()
        ax = fig.add_subplot(111)
        img = self.dp_max
        # log1p stretch for visibility
        img_d = np.log1p(np.clip(img, 0, None))
        ax.imshow(img_d, cmap="inferno", aspect="equal",
                   interpolation="nearest")
        if self.probe_semiangle is not None:
            ax.add_patch(Circle((self.probe_qy0, self.probe_qx0),
                                  self.probe_semiangle, color="white",
                                  fill=False, lw=1.5))
        ax.set_title(
            f"{self.sample} — dp_max  (probe r = "
            f"{self.probe_semiangle:.2f} px)" if self.probe_semiangle
            else "dp_max")
        ax.set_xticks([]); ax.set_yticks([])
        self._redraw()

    def _render_adf(self):
        if self.virtual_adf is None:
            messagebox.showinfo("virtual",
                "Run Step 2 first."); return
        fig = self._new_fig()
        ax = fig.add_subplot(111)
        img = self.virtual_adf
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        im = ax.imshow(np.clip(img, lo, hi), cmap="gray",
                        aspect="equal", interpolation="nearest")
        ax.set_title(
            f"{self.sample} — virtual ADF  "
            f"(annulus {self._adf_inner.get()}–{self._adf_outer.get()} px)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self._redraw()

    # ------------------------------------------------------------------
    # Step 3: vacuum probe + kernel
    # ------------------------------------------------------------------
    def _step_build_probe(self):
        if not self._ensure_datacube(): return
        if self.probe_semiangle is None:
            messagebox.showinfo("virtual",
                "Run Step 2 first to estimate the probe radius.")
            return
        if self._busy: return
        self._busy = True
        self._set_status("building vacuum probe + sigmoid kernel …")
        threading.Thread(target=self._probe_worker, daemon=True).start()

    def _probe_worker(self):
        try:
            rx0 = int(self._roi_rx0.get()); rx1 = int(self._roi_rx1.get())
            ry0 = int(self._roi_ry0.get()); ry1 = int(self._roi_ry1.get())
            ds = self.datacube
            self._probe = ds.get_vacuum_probe(ROI=(rx0, rx1, ry0, ry1))
            self._probe.get_kernel(
                mode="sigmoid",
                radii=(self.probe_semiangle * 0.0,
                        self.probe_semiangle * 4.0),
                bilinear=True,
            )
            self._probe_kernel = np.asarray(self._probe.kernel)
            self.after(0, lambda: self._set_status(
                f"vacuum probe built  ROI=({rx0}:{rx1},{ry0}:{ry1}),  "
                f"kernel shape = {self._probe_kernel.shape}"))
            self.after(0, self._render_probe_kernel)
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "probe build failed", err))
            self.after(0, lambda: self._set_status("probe build failed."))
        finally:
            self._busy = False

    def _render_probe_kernel(self):
        if self._probe_kernel is None:
            messagebox.showinfo("probe", "Build the probe first."); return
        fig = self._new_fig()
        ax = fig.add_subplot(111)
        k = self._probe_kernel
        ax.imshow(k, cmap="RdBu_r", aspect="equal",
                   interpolation="nearest",
                   vmin=-np.abs(k).max(), vmax=np.abs(k).max())
        ax.set_title(f"sigmoid probe kernel  shape={k.shape}")
        ax.set_xticks([]); ax.set_yticks([])
        self._redraw()

    # ------------------------------------------------------------------
    # Step 4: Bragg disk detection
    # ------------------------------------------------------------------
    def _detect_params(self):
        sub = self._dp_subpixel.get().strip().lower() or "poly"
        return dict(
            corrPower            = float(self._dp_corrPower.get()),
            sigma                = float(self._dp_sigma.get()),
            edgeBoundary         = int(self._dp_edgeBoundary.get()),
            minRelativeIntensity = float(self._dp_minRelIntensity.get()),
            minAbsoluteIntensity = float(self._dp_minAbsIntensity.get()),
            minPeakSpacing       = int(self._dp_minPeakSpacing.get()),
            subpixel             = sub,
            upsample_factor      = int(self._dp_upsample_factor.get()),
            maxNumPeaks          = int(self._dp_maxNumPeaks.get()),
        )

    def _seed_preview_positions(self):
        if not self._preview_positions:
            if self._cube_mmap is None: return
            Ny, Nx, _, _ = self._cube_mmap.shape
            # 10 positions on a 2×5 grid spread over the scan
            ys = np.linspace(0.15, 0.85, 2) * (Ny - 1)
            xs = np.linspace(0.10, 0.90, 5) * (Nx - 1)
            self._preview_positions = [
                (int(round(y)), int(round(x))) for y in ys for x in xs]

    def _step_detect_preview(self):
        if not self._ensure_datacube(): return
        if self._probe_kernel is None:
            messagebox.showinfo("probe", "Build the probe first (Step 3)."); return
        if self.virtual_adf is None:
            messagebox.showinfo("virtual",
                "Run Step 2 first so we have a HAADF for the markers."); return
        self._seed_preview_positions()
        self._draw_preview_layout()
        self._refresh_preview_disks()

    def _draw_preview_layout(self):
        """Lay out: HAADF on the LEFT, 2×5 grid of diffraction tiles on the
        RIGHT. Each marker on the HAADF has a matching colour border on
        its tile."""
        fig = self._new_fig()
        gs = GridSpec(2, 6, figure=fig,
                       width_ratios=[2.2, 1, 1, 1, 1, 1],
                       hspace=0.30, wspace=0.10)
        # HAADF spans both rows of the leftmost column
        self._haadf_ax = fig.add_subplot(gs[:, 0])
        adf = self.virtual_adf
        lo, hi = np.percentile(adf, 1), np.percentile(adf, 99)
        self._haadf_ax.imshow(np.clip(adf, lo, hi), cmap="gray",
                                aspect="equal", interpolation="nearest")
        self._haadf_ax.set_title(
            f"HAADF — drag any marker  "
            f"(release → re-run Bragg detect on that frame)",
            fontsize=10)
        self._haadf_ax.set_xticks([]); self._haadf_ax.set_yticks([])

        # markers
        self._preview_artists = []
        for i, (rx, ry) in enumerate(self._preview_positions):
            color = _TAB10[i % 10]
            # NOTE: imshow x = col = ry, y = row = rx
            artist, = self._haadf_ax.plot(
                [ry], [rx], marker="o", markersize=11,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=1.4, picker=8, zorder=5)
            self._haadf_ax.annotate(str(i), (ry, rx),
                xytext=(6, 6), textcoords="offset points",
                color="white", fontsize=9, weight="bold",
                path_effects=[])
            self._preview_artists.append(artist)

        # 10 diffraction tiles in the right 5 columns of both rows
        self._preview_axes = []
        self._preview_disks = []
        for i in range(10):
            r = i // 5; c = (i % 5) + 1
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([]); ax.set_yticks([])
            color = _TAB10[i % 10]
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(2.0)
            ax.set_title(f"#{i}", fontsize=9, color=color, weight="bold")
            self._preview_axes.append(ax)
            self._preview_disks.append(None)

        self._connect_preview_drag()
        self._redraw()
        self._set_status(
            "preview layout drawn. drag any marker on the HAADF; the "
            "matching tile re-renders on release.")

    def _refresh_preview_disks(self, marker_idx=None):
        """Run find_Bragg_disks on the 10 (or just one) preview position(s)
        and redraw the corresponding tiles + detected disks. Heavy-ish
        (~50–200 ms per pattern), so when a single marker is dragged
        we only update its tile."""
        if self.datacube is None or self._probe_kernel is None:
            return
        idxs = [marker_idx] if marker_idx is not None else range(10)
        for i in idxs:
            rx, ry = self._preview_positions[i]
            ax = self._preview_axes[i]
            try:
                pat = np.asarray(self._cube_mmap[rx, ry]).astype(np.float32)
            except Exception:
                ax.cla(); ax.text(0.5, 0.5, "(out of range)",
                                    ha="center", va="center",
                                    transform=ax.transAxes)
                continue
            ax.cla()
            ax.set_xticks([]); ax.set_yticks([])
            color = _TAB10[i % 10]
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(2.0)
            # log1p display
            ax.imshow(np.log1p(np.clip(pat, 0, None)), cmap="inferno",
                       aspect="equal", interpolation="nearest")
            try:
                pl_list = self.datacube.find_Bragg_disks(
                    data=([rx], [ry]),
                    template=self._probe_kernel,
                    **self._detect_params(),
                )
                # py4DSTEM returns a list-like; first entry is a PointList
                pl = pl_list[0] if hasattr(pl_list, "__getitem__") else pl_list
                qx = np.asarray(pl.data["qx"])
                qy = np.asarray(pl.data["qy"])
                ax.scatter(qy, qx, s=80, facecolors="none",
                            edgecolors=color, linewidths=1.4)
                ax.set_title(f"#{i}  ({rx},{ry})  N={len(qx)}",
                              fontsize=9, color=color, weight="bold")
            except Exception as e:
                ax.set_title(f"#{i}  ({rx},{ry})  detect err",
                              fontsize=8, color="red")
        self._canvas.draw_idle()

    # drag callbacks
    def _connect_preview_drag(self):
        self._disconnect_preview()
        cids = []
        cids.append(self._canvas.mpl_connect(
            "pick_event", self._on_preview_pick))
        cids.append(self._canvas.mpl_connect(
            "motion_notify_event", self._on_preview_motion))
        cids.append(self._canvas.mpl_connect(
            "button_release_event", self._on_preview_release))
        self._preview_cids = cids

    def _disconnect_preview(self):
        for cid in self._preview_cids:
            try: self._canvas.mpl_disconnect(cid)
            except Exception: pass
        self._preview_cids = []
        self._dragging = None

    def _on_preview_pick(self, event):
        for i, art in enumerate(self._preview_artists):
            if event.artist is art:
                self._dragging = i
                return

    def _on_preview_motion(self, event):
        if self._dragging is None: return
        if event.inaxes is not self._haadf_ax: return
        if event.xdata is None or event.ydata is None: return
        # imshow coords: x = col = ry, y = row = rx
        ry = float(event.xdata); rx = float(event.ydata)
        # clamp to scan extent
        Ny, Nx, _, _ = self._cube_mmap.shape
        rx = max(0, min(Ny - 1, int(round(rx))))
        ry = max(0, min(Nx - 1, int(round(ry))))
        self._preview_positions[self._dragging] = (rx, ry)
        art = self._preview_artists[self._dragging]
        art.set_data([ry], [rx])
        self._canvas.draw_idle()

    def _on_preview_release(self, event):
        if self._dragging is None: return
        i = self._dragging
        self._dragging = None
        self._set_status(
            f"running Bragg detect on marker #{i} at "
            f"({self._preview_positions[i][0]}, "
            f"{self._preview_positions[i][1]}) …")
        # heavy-ish; drop to a worker so the UI stays alive
        threading.Thread(
            target=lambda: self.after(0,
                lambda: self._refresh_preview_disks(marker_idx=i)),
            daemon=True).start()

    # full-cube detection
    def _step_detect_full(self):
        if not self._ensure_datacube(): return
        if self._probe_kernel is None:
            messagebox.showinfo("probe", "Build the probe first."); return
        if self._busy: return
        self._busy = True
        self._set_status("Bragg disk detection on full cube  (slow) …")
        threading.Thread(target=self._detect_full_worker,
                          daemon=True).start()

    def _detect_full_worker(self):
        try:
            t0 = time.time()
            self.bragg_peaks = self.datacube.find_Bragg_disks(
                template=self._probe_kernel, **self._detect_params())
            dt = time.time() - t0
            self.after(0, lambda: self._set_status(
                f"find_Bragg_disks done in {dt:.1f}s. "
                f"Step 5: Measure + fit origin."))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "find_Bragg_disks failed", err))
            self.after(0, lambda: self._set_status("detect failed."))
        finally:
            self._busy = False

    # ------------------------------------------------------------------
    # Step 5: centering + calibration
    # ------------------------------------------------------------------
    def _step_fit_origin(self):
        if self.bragg_peaks is None:
            messagebox.showinfo("bragg",
                "Run 'Detect on full cube' first (Step 4)."); return
        if self._busy: return
        self._busy = True
        self._set_status("measuring + fitting origin …")
        threading.Thread(target=self._origin_worker, daemon=True).start()

    def _origin_worker(self):
        try:
            self.bragg_peaks.measure_origin()
            qx0_fit, qy0_fit, qx_res, qy_res = (
                self.bragg_peaks.fit_origin())
            self._origin_fit = (np.asarray(qx0_fit), np.asarray(qy0_fit))
            self.after(0, lambda: self._set_status(
                "origin measured + fit. center now applied to bragg_peaks."))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "fit_origin failed", err))
            self.after(0, lambda: self._set_status("fit_origin failed."))
        finally:
            self._busy = False

    def _step_show_bvm(self):
        if self.bragg_peaks is None:
            messagebox.showinfo("bragg", "Run Step 4 first."); return
        try:
            self._bvm = np.asarray(self.bragg_peaks.get_bvm())
        except Exception as e:
            messagebox.showerror("BVM failed", repr(e)); return
        fig = self._new_fig()
        ax = fig.add_subplot(111)
        ax.imshow(np.log1p(np.clip(self._bvm, 0, None)),
                   cmap="inferno", aspect="equal", interpolation="nearest")
        ax.set_title(f"{self.sample} — BVM (centered Bragg vector map)")
        ax.set_xticks([]); ax.set_yticks([])
        self._redraw()

    def _step_apply_calibration(self):
        if self.bragg_peaks is None:
            messagebox.showinfo("bragg", "Run Step 4 first."); return
        try:
            inv_a = float(self._inv_ang_per_pixel.get())
            self.bragg_peaks.calibration.set_Q_pixel_size(inv_a)
            self.bragg_peaks.calibration.set_Q_pixel_units("A^-1")
            self._calibrated = True
            self._set_status(
                f"applied Q pixel size = {inv_a:.6f} 1/Å.  "
                f"calstate = {getattr(self.bragg_peaks, 'calstate', '?')}")
        except Exception as e:
            messagebox.showerror("calibration failed", repr(e))

    # ------------------------------------------------------------------
    # Step 6: structure factors
    # ------------------------------------------------------------------
    def _step_struct_factors(self):
        if self.crystal is None:
            messagebox.showinfo("crystal",
                "Build the crystal first (Browse CIF + Build crystal)."); return
        try:
            self.crystal.calculate_structure_factors(
                float(self._k_max.get()))
            self._set_status(
                f"structure factors computed for k_max = "
                f"{self._k_max.get()} 1/Å.")
        except Exception as e:
            messagebox.showerror("structure factors failed", repr(e))

    # ------------------------------------------------------------------
    # Step 7: orientation plan
    # ------------------------------------------------------------------
    def _parse_3vec(self, s):
        import re
        m = re.findall(r"-?\d+\.?\d*", s)
        if len(m) < 3: raise ValueError(f"need 3 numbers in {s!r}")
        return [float(m[0]), float(m[1]), float(m[2])]

    def _parse_2vec(self, s):
        import re
        m = re.findall(r"-?\d+\.?\d*", s)
        if len(m) < 2: raise ValueError(f"need 2 numbers in {s!r}")
        return [float(m[0]), float(m[1])]

    def _step_generate_plan(self):
        if self.crystal is None:
            messagebox.showinfo("crystal", "Build a crystal first.")
            return
        if self._busy: return
        self._busy = True
        self._set_status("generating orientation plan (~30 s) …")
        threading.Thread(target=self._plan_worker, daemon=True).start()

    def _plan_worker(self):
        try:
            mode = self._plan_mode.get()
            kw = dict(
                angle_step_zone_axis=float(self._angle_step_za.get()),
                angle_step_in_plane=float(self._angle_step_ip.get()),
                accel_voltage=300e3,
                corr_kernel_size=float(self._cor_kernel.get()),
                radial_power=1.0, intensity_power=0.25,
                tol_distance=0.01, tol_intensity=1e-3,
            )
            if mode == "fiber":
                kw["zone_axis_range"] = "fiber"
                kw["fiber_axis"] = self._parse_3vec(
                    self._fiber_axis.get())
                kw["fiber_angles"] = self._parse_2vec(
                    self._fiber_angles.get())
                self.crystal.orientation_plan(**kw)
            else:
                # corners
                import re
                s = self._zone_axis_range.get()
                triples = re.findall(r"\[([^\]]+)\]", s)
                rows = []
                for t in triples[:3]:
                    parts = [float(x.strip()) for x in t.split(",")
                             if x.strip()]
                    if len(parts) == 3:
                        rows.append(parts)
                rng = np.array(rows, dtype=np.float64)
                try:
                    self.crystal.orientation_plan(
                        zone_axis_range=rng, **kw)
                except TypeError:
                    self.crystal.orientation_plan(
                        corners_zone_axes=rng, **kw)
            self.after(0, lambda: self._set_status(
                f"orientation plan ready  ({mode} mode). "
                "Step 8: Match."))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "plan failed", err))
            self.after(0, lambda: self._set_status("plan failed."))
        finally:
            self._busy = False

    # ------------------------------------------------------------------
    # Step 8: matching
    # ------------------------------------------------------------------
    def _step_match_single(self):
        if self.crystal is None or self.bragg_peaks is None:
            messagebox.showinfo("match",
                "Need crystal + plan + bragg_peaks first.")
            return
        rx = int(self._test_rx.get()); ry = int(self._test_ry.get())
        try:
            pl = self.bragg_peaks.cal[rx, ry]
            orient = self.crystal.match_single_pattern(pl, verbose=False)
            fit = self.crystal.generate_diffraction_pattern(
                orient, ind_orientation=0,
                sigma_excitation_error=0.03)
            fig = self._new_fig()
            ax = fig.add_subplot(111)
            from py4DSTEM.process.diffraction import (
                plot_diffraction_pattern)
            plot_diffraction_pattern(
                fit, bragg_peaks_compare=pl,
                scale_markers=1000, scale_markers_compare=4e4,
                min_marker_size=1, figsize=(5, 5),
                input_fig_handle=(fig, ax))
            ax.set_title(f"single-pattern match @ ({rx}, {ry})")
            self._redraw()
            self._set_status(
                f"single-pattern match done @ ({rx}, {ry}).")
        except Exception as e:
            messagebox.showerror("match_single_pattern failed", repr(e))

    def _step_match_all(self):
        if self.crystal is None or self.bragg_peaks is None:
            messagebox.showinfo("match",
                "Need crystal + plan + bragg_peaks first."); return
        if self._busy: return
        self._busy = True
        self._set_status("matching orientations on full scan (slow) …")
        threading.Thread(target=self._match_worker, daemon=True).start()

    def _match_worker(self):
        try:
            t0 = time.time()
            self.orientation_map = self.crystal.match_orientations(
                self.bragg_peaks)
            dt = time.time() - t0
            self.after(0, lambda: self._set_status(
                f"match_orientations done in {dt:.1f}s. "
                f"Step 9 (strain) or render orientation map."))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "match_orientations failed", err))
            self.after(0, lambda: self._set_status("match failed."))
        finally:
            self._busy = False

    # ------------------------------------------------------------------
    # Step 9: strain
    # ------------------------------------------------------------------
    def _step_calculate_strain(self):
        if self.crystal is None or self.bragg_peaks is None or \
                self.orientation_map is None:
            messagebox.showinfo("strain",
                "Need bragg_peaks + orientation_map (Step 8) first.")
            return
        if self._busy: return
        self._busy = True
        self._set_status("calculating strain map …")
        threading.Thread(target=self._strain_worker, daemon=True).start()

    def _strain_worker(self):
        try:
            rot_rng = float(self._rotation_range.get()) * np.pi / 180.0
            self.strain_map = self.crystal.calculate_strain(
                self.bragg_peaks, self.orientation_map,
                rotation_range=rot_rng)
            self.after(0, lambda: self._set_status(
                "strain map computed."))
            self.after(0, self._render_strain)
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "calculate_strain failed", err))
            self.after(0, lambda: self._set_status("strain failed."))
        finally:
            self._busy = False

    def _render_strain(self):
        if self.strain_map is None:
            messagebox.showinfo("strain",
                "Run Step 9 first."); return
        try:
            from py4DSTEM.visualize import show_strain
            fig = self._new_fig()
            v = float(self._strain_vrange.get())
            t = float(self._theta_vrange.get())
            try:
                show_strain(self.strain_map,
                              vrange_exx=[-v, v],
                              vrange_theta=[-t, t],
                              ticknumber=3,
                              axes_plots=(),
                              bkgrd=False,
                              figsize=(7, 7),
                              returnfig=False,
                              input_fig_handle=fig)
            except TypeError:
                # older signatures: let it draw on its own figure, then we
                # at least render a placeholder
                show_strain(self.strain_map,
                              vrange_exx=[-v, v],
                              vrange_theta=[-t, t])
            self._redraw("strain_map")
            self._set_status("strain rendered.")
        except Exception as e:
            messagebox.showerror("show_strain failed", repr(e))

    # ------------------------------------------------------------------
    # orientation render fallbacks
    # ------------------------------------------------------------------
    def _orientation_arrays(self):
        om = self.orientation_map
        if om is None: return None
        out = {"corr": None, "matrix": None, "angles": None}
        for attr in ("corr", "correlation", "corr_max"):
            v = getattr(om, attr, None)
            if v is not None and hasattr(v, "shape"):
                out["corr"] = np.asarray(v); break
        for attr in ("matrix", "orientation_matrices"):
            v = getattr(om, attr, None)
            if v is not None and hasattr(v, "shape"):
                out["matrix"] = np.asarray(v); break
        for attr in ("angles", "orientation_angles", "euler_angles"):
            v = getattr(om, attr, None)
            if v is not None and hasattr(v, "shape"):
                out["angles"] = np.asarray(v); break
        return out

    def _render_orientation(self):
        if self.orientation_map is None:
            messagebox.showinfo("ACOM", "Run Step 8 first."); return
        # py4DSTEM has plot_fiber_orientation_maps for fiber plans;
        # try it for fiber, fall back to RGB-from-Eulers
        if self._plan_mode.get() == "fiber":
            try:
                fig = self._new_fig()
                self.crystal.plot_fiber_orientation_maps(
                    self.orientation_map, symmetry_order=6,
                    corr_range=[0.9, 1.0], figsize=(6, 6),
                    input_fig_handle=fig)
                self._redraw("orientation_map"); return
            except Exception:
                pass
        # generic plotter
        for method in ("plot_orientation_map", "plot"):
            fn = getattr(self.orientation_map, method, None)
            if callable(fn):
                try:
                    fig = self._new_fig()
                    fn(fig=fig); self._redraw("orientation_map"); return
                except TypeError:
                    try: fn(); return
                    except Exception: continue
                except Exception: continue
        # RGB fallback
        arr = self._orientation_arrays()
        fig = self._new_fig()
        ax = fig.add_subplot(111)
        if arr and arr["angles"] is not None:
            a = arr["angles"]
            if a.ndim == 4:
                a = a[..., 0, :]
            if a.ndim == 3 and a.shape[-1] >= 3:
                rgb = np.zeros(a.shape[:2] + (3,), dtype=np.float32)
                for k in range(3):
                    ch = a[..., k]
                    rgb[..., k] = ((ch - ch.min()) /
                                    (ch.ptp() + 1e-9))
                ax.imshow(rgb)
                ax.set_title("orientation map (Euler→RGB)")
            else:
                ax.text(0.5, 0.5, "angles array shape unsupported.",
                          ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "no angles/matrix on orientation_map.",
                      ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        self._redraw("orientation_map")

    def _render_correlation(self):
        if self.orientation_map is None:
            messagebox.showinfo("ACOM", "Run Step 8 first."); return
        arr = self._orientation_arrays()
        fig = self._new_fig()
        ax = fig.add_subplot(111)
        if arr and arr["corr"] is not None:
            c = arr["corr"]
            if c.ndim == 3:
                c = c[..., 0]
            im = ax.imshow(c, cmap="magma")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("correlation max")
        else:
            ax.text(0.5, 0.5,
                "Could not find a correlation array on the "
                "orientation_map.", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        self._redraw()

    def _render_plan_preview(self):
        if self.crystal is None:
            messagebox.showinfo("crystal", "Build a crystal first."); return
        fig = self._new_fig()
        ax = fig.add_subplot(111)
        za = getattr(self.crystal, "orientation_zone_axis_range", None)
        if za is not None:
            try:
                arr = np.asarray(za)
                if arr.shape[0] == 3 and arr.shape[1] >= 1:
                    arr = arr / (np.linalg.norm(arr, axis=0,
                                                  keepdims=True) + 1e-12)
                    x = arr[0] / (1 + arr[2])
                    y = arr[1] / (1 + arr[2])
                    ax.scatter(x, y, s=8, alpha=0.6)
            except Exception:
                pass
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title("orientation plan zone-axis preview "
                      "(stereographic)")
        self._redraw()
