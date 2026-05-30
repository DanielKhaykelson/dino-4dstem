"""blob_acom_panel.py -- ACOM using blobs detected in the Detect sub-tab.

Skips py4DSTEM's preprocessing (probe + disk-detection + origin fit +
ellipse correction). The saved per-pattern blob coordinates from the
Detect sub-tab are wrapped into a `BraggVectors` object directly,
calibrated, then fed through the standard structure-factors →
orientation-plan → match_orientations pipeline.

Workflow
--------
    1. Load CIF + build Crystal.
    2. Calibration: pick Q-pixel-size in 1/Å, either manually or by
       fitting the experimental 1D radial against the CIF-predicted
       g-list (we surface the predicted line locations so you can see
       where they should fall).
    3. Build BraggVectors from <run>/blob/{method}/p<c>/blob_coords.npy
       (one PointList per scan position, with intensity = sigma).
    4. Structure factors  →  orientation plan  →  match_orientations
       (same as acom_panel.py from this point on).
    5. Render the orientation map.

If saved blob outputs cover only some classes (or none), the panel
warns the user and refuses the relevant steps.
"""
from __future__ import annotations
import os, json, time, threading, re

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                 NavigationToolbar2Tk)


def _try_import_py4DSTEM():
    try:
        import py4DSTEM
        from py4DSTEM.process.diffraction import Crystal
        return py4DSTEM, Crystal
    except Exception:
        return None, None


def _section(parent, title):
    f = ctk.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x", padx=2, pady=(8, 0))
    ctk.CTkLabel(f, text=title, font=("Segoe UI", 11, "bold")).pack(
        anchor="w")
    sep = ctk.CTkFrame(parent, height=2, fg_color=("#cccccc", "#444444"))
    sep.pack(fill="x", padx=2, pady=(0, 4))
    return parent


def _entry_row(parent, label, var, width=80):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=4, pady=1)
    ctk.CTkLabel(row, text=label, width=170, anchor="w").pack(side="left")
    ctk.CTkEntry(row, textvariable=var, width=width).pack(side="left")
    return row


# ---------------------------------------------------------------------------
class BlobACOMPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.outdir = None
        self.sample = None
        self._scan_shape = None
        self._py4dstem, self._Crystal = _try_import_py4DSTEM()
        self.crystal = None
        self.bragg_peaks = None
        self.orientation_map = None
        self._busy = False
        self._build()

    # ----- run linkage -------------------------------------------------
    def link_run(self, outdir, sample):
        self.outdir = outdir
        self.sample = sample
        self._info_lbl.configure(
            text=f"linked: {os.path.basename(outdir)}  (sample={sample})")
        self.bragg_peaks = None
        self.orientation_map = None
        self._render_idle()
        self._refresh_blob_status()

    def on_runtime_sample_added(self, key):
        pass

    # ----- UI ----------------------------------------------------------
    def _build(self):
        self._vars = {
            "cif_path":   ctk.StringVar(value=""),
            "k_max":      ctk.DoubleVar(value=2.0),     # 1/Å
            "inv_ang":    ctk.DoubleVar(value=0.005),   # 1/Å per pixel
            "angle_step_za": ctk.DoubleVar(value=2.0),
            "angle_step_ip": ctk.DoubleVar(value=2.0),
            "cor_kernel":    ctk.DoubleVar(value=0.05),
            "plan_mode":     ctk.StringVar(value="fiber"),
            "fiber_axis":    ctk.StringVar(value="0,0,1"),
            "fiber_angles":  ctk.StringVar(value="0, 360"),
            "test_rx":       ctk.IntVar(value=0),
            "test_ry":       ctk.IntVar(value=0),
        }

        # top bar
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)
        self._info_lbl = ctk.CTkLabel(top, text="(no run linked)",
                                        font=("Consolas", 10))
        self._info_lbl.pack(side="left", padx=8)
        if self._py4dstem is None:
            ctk.CTkLabel(top, text="[py4DSTEM not importable]",
                          text_color=("#aa3333", "#e2a05f"),
                          font=("Consolas", 10, "bold")
                          ).pack(side="right", padx=6)

        # body
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        sb = ctk.CTkScrollableFrame(body, width=320)
        sb.pack(side="left", fill="y")

        # ---- 1. CIF + crystal ----
        _section(sb, "1. Crystal (from CIF)")
        cif_row = ctk.CTkFrame(sb, fg_color="transparent")
        cif_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkEntry(cif_row, textvariable=self._vars["cif_path"],
                       width=200).pack(side="left", padx=2)
        ctk.CTkButton(cif_row, text="Browse…", width=70,
                       command=self._browse_cif).pack(side="left",
                                                          padx=2)
        ctk.CTkButton(sb, text="Build crystal",
                       command=self._build_crystal
                       ).pack(fill="x", padx=8, pady=2)
        self._crystal_info = ctk.CTkLabel(sb, text="(no crystal yet)",
                                             font=("Consolas", 9),
                                             justify="left",
                                             wraplength=300)
        self._crystal_info.pack(anchor="w", padx=8, pady=2)

        # ---- 2. Calibration ----
        _section(sb, "2. Q-pixel calibration")
        _entry_row(sb, "1/Å per pixel", self._vars["inv_ang"])
        ctk.CTkButton(sb, text="Show CIF g-list  →  pick scale",
                       command=self._show_cif_glist
                       ).pack(fill="x", padx=8, pady=2)

        # ---- 3. Bragg vectors from blobs ----
        _section(sb, "3. Bragg vectors (from blobs)")
        self._blob_status_lbl = ctk.CTkLabel(sb,
            text="(loading…)", font=("Consolas", 9), justify="left",
            wraplength=300, text_color=("#444", "#aaa"))
        self._blob_status_lbl.pack(anchor="w", padx=8, pady=2)
        ctk.CTkButton(sb, text="Build BraggVectors",
                       command=self._build_bragg
                       ).pack(fill="x", padx=8, pady=2)

        # ---- 4. Structure factors + plan + match ----
        _section(sb, "4. Structure factors")
        _entry_row(sb, "k_max  (1/Å)", self._vars["k_max"])
        ctk.CTkButton(sb, text="Compute structure factors",
                       command=self._step_struct_factors
                       ).pack(fill="x", padx=8, pady=2)

        _section(sb, "5. Orientation plan")
        ctk.CTkOptionMenu(sb, variable=self._vars["plan_mode"],
                            values=["fiber", "corners"], width=160
                            ).pack(anchor="w", padx=8, pady=2)
        _entry_row(sb, "Δ zone-axis (deg)", self._vars["angle_step_za"])
        _entry_row(sb, "Δ in-plane (deg)",  self._vars["angle_step_ip"])
        _entry_row(sb, "corr kernel (1/Å)", self._vars["cor_kernel"])
        _entry_row(sb, "fiber axis  [u,v,w]", self._vars["fiber_axis"])
        _entry_row(sb, "fiber angles [a,b]",  self._vars["fiber_angles"])
        ctk.CTkButton(sb, text="Generate plan",
                       command=self._step_generate_plan
                       ).pack(fill="x", padx=8, pady=2)

        _section(sb, "6. Match")
        rxry = ctk.CTkFrame(sb, fg_color="transparent")
        rxry.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(rxry, text="rx,ry:", width=60).pack(side="left")
        ctk.CTkEntry(rxry, textvariable=self._vars["test_rx"],
                       width=60).pack(side="left", padx=2)
        ctk.CTkEntry(rxry, textvariable=self._vars["test_ry"],
                       width=60).pack(side="left", padx=2)
        ctk.CTkButton(sb, text="Match single",
                       command=self._step_match_single
                       ).pack(fill="x", padx=8, pady=2)
        ctk.CTkButton(sb, text="Match all  →  orientation map",
                       command=self._step_match_all,
                       fg_color=("#2D7A2D", "#1F7A1F")
                       ).pack(fill="x", padx=8, pady=2)
        ctk.CTkButton(sb, text="Render orientation map",
                       command=self._render_orientation_map
                       ).pack(fill="x", padx=8, pady=2)

        self._status_lbl = ctk.CTkLabel(sb,
            text="", font=("Consolas", 9), justify="left",
            text_color=("#444", "#aaa"), wraplength=300)
        self._status_lbl.pack(anchor="w", padx=8, pady=(8, 4))

        # canvas
        canv = ctk.CTkFrame(body)
        canv.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._fig = Figure(figsize=(11, 7))
        self._canvas = FigureCanvasTkAgg(self._fig, master=canv)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, canv)
        self._render_idle()

    # ----- helpers ------------------------------------------------------
    def _set_status(self, msg):
        try: self._status_lbl.configure(text=msg)
        except Exception: pass

    def _new_fig(self):
        self._fig.clear()
        return self._fig

    def _redraw(self):
        try: self._fig.tight_layout()
        except Exception: pass
        self._canvas.draw_idle()

    def _render_idle(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "1. Build crystal from CIF.\n"
                "2. Set Q-pixel scale (1/Å per px).\n"
                "3. Build BraggVectors from saved blobs "
                "(needs whole-class blob run in Detect tab).\n"
                "4. Structure factors → orientation plan → Match.",
                ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        self._canvas.draw_idle()

    # ----- per-pattern blob status (warning) ---------------------------
    def _blob_method_dir(self) -> str | None:
        if not self.outdir:
            return None
        cfg_path = os.path.join(self.outdir, "blob", "blob_config.json")
        if not os.path.exists(cfg_path):
            return None
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            method = cfg.get("method", "DoG")
        except Exception:
            method = "DoG"
        return os.path.join(self.outdir, "blob", method)

    def _refresh_blob_status(self):
        if not self.outdir:
            self._blob_status_lbl.configure(text=""); return
        md = self._blob_method_dir()
        if not md or not os.path.isdir(md):
            self._blob_status_lbl.configure(text=(
                "no saved blob output. Run blob detection in the "
                "Detect sub-tab (Apply to ALL classes)."))
            return
        present = []; missing = []
        for entry in sorted(os.listdir(md)):
            if not entry.startswith("p"):
                continue
            if os.path.exists(
                    os.path.join(md, entry, "blob_coords.npy")):
                present.append(entry)
            else:
                missing.append(entry)
        if not present:
            self._blob_status_lbl.configure(text=(
                "blob detection has not been saved per-pattern.  "
                "ACOM needs 'Apply to ALL classes' in Detect."))
        elif missing:
            self._blob_status_lbl.configure(text=(
                f"per-pattern blobs found for {len(present)} classes, "
                f"missing for {len(missing)}: "
                f"{', '.join(missing[:5])}…  ACOM needs all classes."))
        else:
            self._blob_status_lbl.configure(text=(
                f"per-pattern blobs cover {len(present)} classes "
                f"({md})."))

    # ----- 1. crystal --------------------------------------------------
    def _browse_cif(self):
        p = filedialog.askopenfilename(
            filetypes=[("CIF", "*.cif"), ("All", "*.*")])
        if p: self._vars["cif_path"].set(p)

    def _build_crystal(self):
        if self._py4dstem is None:
            messagebox.showerror("py4DSTEM",
                "py4DSTEM is not available."); return
        p = self._vars["cif_path"].get().strip()
        if not p or not os.path.exists(p):
            messagebox.showerror("CIF", f"Not found: {p!r}"); return
        try:
            try:
                self.crystal = self._Crystal.from_CIF(p)
            except AttributeError:
                self.crystal = self._Crystal(filepath=p)
        except Exception as e:
            messagebox.showerror("Crystal", str(e)); return
        info = [f"CIF: {os.path.basename(p)}"]
        for attr in ("a", "b", "c", "alpha", "beta", "gamma"):
            v = getattr(self.crystal, attr, None)
            try:
                if v is not None: info.append(f"{attr}={float(v):.4f}")
            except Exception: pass
        self._crystal_info.configure(text="  ".join(info))
        self._set_status("crystal built.")

    # ----- 2. calibration overlay --------------------------------------
    def _show_cif_glist(self):
        """Show experimental 1D radial alongside vertical CIF g-lines, so
        user can see whether the current Q-pixel scale matches."""
        if self.crystal is None:
            messagebox.showinfo("calib", "Build a crystal first."); return
        try:
            self.crystal.calculate_structure_factors(
                float(self._vars["k_max"].get()))
        except Exception as e:
            messagebox.showerror("structure factors", repr(e)); return
        # CIF g-magnitudes (1/Å) and intensities.
        g_mag = None
        sf_int = None
        try:
            g_vec = getattr(self.crystal, "g_vec_all", None)
            if g_vec is not None:
                g_mag = np.linalg.norm(np.asarray(g_vec), axis=0)
            sf_int = getattr(self.crystal, "struct_factors_int", None)
            if sf_int is None:
                sf_int = np.abs(getattr(self.crystal,
                                          "struct_factors", 1.0)) ** 2
        except Exception:
            pass
        # Experimental 1D radial — load sidecar.
        rad_path = self._sample_radial_path()
        if rad_path is None or not os.path.exists(rad_path):
            messagebox.showinfo(
                "calib", f"radial sidecar not found: {rad_path!r}\n"
                f"Compute it first via compute_radial_profile.py.")
            return
        rad = np.load(rad_path, allow_pickle=True).astype(np.float32)
        mean_radial = rad.mean(axis=0)
        # Bin → 1/Å using current inv_ang × pixel-scale-factor.
        from gui_app._calib_utils import (q_per_polar_bin,
                                            get_raw_detector_size)
        rp_inv_nm = 0.0
        try:
            rp_inv_nm = float(self.app.recip_res.get()) if self.app else 0.0
        except Exception:
            rp_inv_nm = 0.0
        # If user has set inv_ang directly (per raw-px), use that.  Else
        # fall back to the top-bar reciprocal calibration.
        inv_ang = float(self._vars["inv_ang"].get())
        if inv_ang <= 0 and rp_inv_nm > 0:
            inv_ang = rp_inv_nm * 0.1     # 1/nm → 1/Å
            self._vars["inv_ang"].set(inv_ang)
        if inv_ang <= 0:
            messagebox.showinfo("calib",
                "Set 1/Å per pixel above (or set top-bar reciprocal)."); return
        qpb = q_per_polar_bin(inv_ang,
                                get_raw_detector_size(self.sample))
        n_bins = mean_radial.shape[0]
        q_vals = np.arange(n_bins) * qpb

        fig = self._new_fig()
        ax = fig.add_subplot(111)
        ax.plot(q_vals, mean_radial, color="black", lw=1.2,
                 label="experimental ⟨I(q)⟩")
        if g_mag is not None and sf_int is not None:
            sf_int = np.asarray(sf_int).flatten()
            mx = float(sf_int.max()) if sf_int.size else 1.0
            for gm, si in zip(g_mag, sf_int):
                if gm <= q_vals.max() and si > mx * 0.005:
                    ax.axvline(gm, color="orange",
                                 alpha=0.3 + 0.7 * (si / mx),
                                 lw=0.8)
            ax.plot([], [], color="orange", lw=0.8,
                     label="CIF g-lines")
        ax.set_xlabel("q  (1/Å)")
        ax.set_ylabel("⟨I⟩ (a.u.)")
        ax.set_title(f"calibration overlay   (1/Å per px = {inv_ang:.5f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._redraw()

    def _sample_radial_path(self):
        try:
            from data import SAMPLES
            cfg = SAMPLES.get(self.sample)
            if not cfg: return None
            return cfg["path"][:-4] + ".radial.npy"
        except Exception:
            return None

    # ----- 3. BraggVectors from blobs ---------------------------------
    def _build_bragg(self):
        if self._py4dstem is None:
            messagebox.showerror("py4DSTEM",
                "py4DSTEM is not available."); return
        if not self.outdir:
            messagebox.showinfo("ACOM", "Load a run first."); return
        md = self._blob_method_dir()
        if not md or not os.path.isdir(md):
            messagebox.showerror("blobs",
                "No saved blob output found. Run blob detection's "
                "'Apply to ALL classes' in the Detect sub-tab."); return
        # We need scan_shape and the per-pattern class assignments.
        try:
            ph = getattr(self.app, "posthoc", None)
            if ph is not None and getattr(ph, "outdir", None) != self.outdir:
                ph.link_run(self.outdir, self.sample)
            if ph is None or not ph._ensure_inference():
                raise RuntimeError("inference unavailable from Post-hoc")
            assigns = ph._inf["assigns"]
            self._scan_shape = ph._scan_shape
        except Exception as e:
            messagebox.showerror("ACOM",
                f"can't get assignments:\n{e!r}"); return

        Ny, Nx = self._scan_shape
        N_total = Ny * Nx
        # Assemble per-scan-position blob lists from per-class saves.
        # For each class p<c>, blob_indices[i] is a flat scan-index and
        # the corresponding coords slice [offsets[i]:offsets[i+1]] gives
        # that pattern's (y, x, sigma).
        # We collect into arrays-of-arrays indexed by flat scan position.
        per_pos_coords: list[np.ndarray | None] = [None] * N_total
        for entry in sorted(os.listdir(md)):
            if not entry.startswith("p"):
                continue
            d = os.path.join(md, entry)
            try:
                idx_arr = np.load(os.path.join(d, "blob_indices.npy"),
                                      allow_pickle=True)
                coords  = np.load(os.path.join(d, "blob_coords.npy"),
                                      allow_pickle=True)
                offs    = np.load(os.path.join(d, "blob_offsets.npy"),
                                      allow_pickle=True)
            except Exception as e:
                print(f"[blob-acom] skip {entry}: {e!r}", flush=True)
                continue
            for i, scan_idx in enumerate(idx_arr):
                a = int(offs[i]); b = int(offs[i + 1])
                if 0 <= int(scan_idx) < N_total:
                    per_pos_coords[int(scan_idx)] = coords[a:b]

        # Build a BraggVectors-like dataset with PointList per (rx, ry).
        py4 = self._py4dstem
        try:
            from py4DSTEM.io.classes import PointList, PointListArray
        except Exception:
            try:
                from emdfile import PointList, PointListArray
            except Exception as e:
                messagebox.showerror("py4DSTEM",
                    f"PointList API unavailable: {e!r}"); return

        coordinates = [("qx", float), ("qy", float), ("intensity", float)]
        pla = PointListArray(coordinates=coordinates, shape=(Ny, Nx))
        det_center = self._detector_center()
        n_pts = 0
        for ry in range(Ny):
            for rx in range(Nx):
                pos = ry * Nx + rx
                blobs = per_pos_coords[pos]
                if blobs is None or blobs.shape[0] == 0:
                    pl = PointList(coordinates=coordinates)
                else:
                    yy = blobs[:, 0].astype(float) - det_center[0]
                    xx = blobs[:, 1].astype(float) - det_center[1]
                    sg = blobs[:, 2].astype(float) if blobs.shape[1] > 2 \
                          else np.ones(blobs.shape[0], dtype=float)
                    arr = np.zeros(len(yy), dtype=[
                        ("qx", float), ("qy", float),
                        ("intensity", float)])
                    arr["qx"] = xx; arr["qy"] = yy; arr["intensity"] = sg
                    pl = PointList(arr, coordinates=coordinates)
                    n_pts += len(yy)
                pla[rx, ry] = pl
        try:
            from py4DSTEM.braggvectors import BraggVectors
        except Exception:
            try:
                from py4DSTEM import BraggVectors
            except Exception as e:
                messagebox.showerror("py4DSTEM",
                    f"BraggVectors import failed: {e!r}"); return
        # BraggVectors construction varies between py4DSTEM versions; try
        # the most common signatures.
        bv = None
        try:
            bv = BraggVectors(Rshape=(Ny, Nx), Qshape=(0, 0))
            bv._v_uncal = pla
        except Exception:
            try:
                bv = BraggVectors(Rshape=(Ny, Nx))
                bv._v_uncal = pla
            except Exception as e:
                messagebox.showerror("py4DSTEM",
                    f"BraggVectors failed:\n{e!r}"); return
        # Apply calibration.
        try:
            inv_a = float(self._vars["inv_ang"].get())
            bv.calibration.set_Q_pixel_size(inv_a)
            bv.calibration.set_Q_pixel_units("A^-1")
        except Exception as e:
            print(f"[blob-acom] calibration set failed: {e!r}", flush=True)
        # Run calibrate to populate bv.cal.
        try:
            bv.calibrate()
        except Exception:
            pass
        self.bragg_peaks = bv
        self._set_status(
            f"BraggVectors built: shape={Ny}×{Nx}, total blobs={n_pts}, "
            f"1/Å per px = {self._vars['inv_ang'].get():.5f}")
        self._refresh_blob_status()

    def _detector_center(self) -> tuple[float, float]:
        """Center of the detector image, in pixels (cy, cx).  We use raw
        detector size if known, else 192 (the polar-input size)."""
        from gui_app._calib_utils import get_raw_detector_size
        rs = get_raw_detector_size(self.sample) or 192
        return (rs / 2.0, rs / 2.0)

    # ----- 4–5. structure factors + plan + match -----------------------
    def _step_struct_factors(self):
        if self.crystal is None:
            messagebox.showinfo("crystal", "Build a crystal first."); return
        try:
            self.crystal.calculate_structure_factors(
                float(self._vars["k_max"].get()))
            self._set_status(
                f"structure factors @ k_max = "
                f"{self._vars['k_max'].get()} 1/Å.")
        except Exception as e:
            messagebox.showerror("structure factors", repr(e))

    def _parse_3vec(self, s):
        m = re.findall(r"-?\d+\.?\d*", s)
        if len(m) < 3: raise ValueError(f"need 3 numbers in {s!r}")
        return [float(m[0]), float(m[1]), float(m[2])]

    def _parse_2vec(self, s):
        m = re.findall(r"-?\d+\.?\d*", s)
        if len(m) < 2: raise ValueError(f"need 2 numbers in {s!r}")
        return [float(m[0]), float(m[1])]

    def _step_generate_plan(self):
        if self.crystal is None:
            messagebox.showinfo("plan", "Build a crystal first."); return
        if self._busy: return
        self._busy = True
        self._set_status("generating orientation plan …")
        threading.Thread(target=self._plan_worker, daemon=True).start()

    def _plan_worker(self):
        try:
            mode = self._vars["plan_mode"].get()
            kw = dict(
                angle_step_zone_axis=float(
                    self._vars["angle_step_za"].get()),
                angle_step_in_plane=float(
                    self._vars["angle_step_ip"].get()),
                accel_voltage=300e3,
                corr_kernel_size=float(self._vars["cor_kernel"].get()),
                radial_power=1.0, intensity_power=0.25,
                tol_distance=0.01, tol_intensity=1e-3,
            )
            if mode == "fiber":
                kw["zone_axis_range"] = "fiber"
                kw["fiber_axis"] = self._parse_3vec(
                    self._vars["fiber_axis"].get())
                kw["fiber_angles"] = self._parse_2vec(
                    self._vars["fiber_angles"].get())
                self.crystal.orientation_plan(**kw)
            else:
                # corners: parse [u,v,w] triples
                triples = re.findall(r"\[([^\]]+)\]",
                                       self._vars["fiber_axis"].get())
                rows = []
                for t in triples[:3]:
                    parts = [float(x) for x in re.findall(
                        r"-?\d+\.?\d*", t)]
                    if len(parts) >= 3:
                        rows.append(parts[:3])
                rng = np.array(rows, dtype=np.float64)
                try:
                    self.crystal.orientation_plan(
                        zone_axis_range=rng, **kw)
                except TypeError:
                    self.crystal.orientation_plan(
                        corners_zone_axes=rng, **kw)
            self.after(0, lambda: self._set_status(
                f"orientation plan ready ({mode})."))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror("plan failed", err))
            self.after(0, lambda: self._set_status("plan failed."))
        finally:
            self._busy = False

    def _step_match_single(self):
        if self.crystal is None or self.bragg_peaks is None:
            messagebox.showinfo("match",
                "Need crystal + plan + bragg peaks."); return
        try:
            rx = int(self._vars["test_rx"].get())
            ry = int(self._vars["test_ry"].get())
            pl = self.bragg_peaks.cal[rx, ry]
            orient = self.crystal.match_single_pattern(pl, verbose=False)
            fit = self.crystal.generate_diffraction_pattern(
                orient, ind_orientation=0,
                sigma_excitation_error=0.03)
            fig = self._new_fig()
            ax = fig.add_subplot(111)
            from py4DSTEM.process.diffraction import plot_diffraction_pattern
            plot_diffraction_pattern(
                fit, bragg_peaks_compare=pl,
                scale_markers=1000, scale_markers_compare=4e4,
                min_marker_size=1, figsize=(5, 5),
                input_fig_handle=(fig, ax))
            ax.set_title(f"single match @ ({rx}, {ry})")
            self._redraw()
            self._set_status(f"single-match done @ ({rx}, {ry}).")
        except Exception as e:
            messagebox.showerror("match_single_pattern", repr(e))

    def _step_match_all(self):
        if self.crystal is None or self.bragg_peaks is None:
            messagebox.showinfo("match",
                "Need crystal + plan + bragg peaks."); return
        if self._busy: return
        self._busy = True
        self._set_status("match_orientations on full scan …")
        threading.Thread(target=self._match_worker, daemon=True).start()

    def _match_worker(self):
        try:
            t0 = time.time()
            self.orientation_map = self.crystal.match_orientations(
                self.bragg_peaks)
            dt = time.time() - t0
            self.after(0, lambda: self._set_status(
                f"match_orientations: {dt:.1f}s. Render orientation map."))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "match_orientations", err))
        finally:
            self._busy = False

    def _render_orientation_map(self):
        if self.orientation_map is None:
            messagebox.showinfo("orient", "Run 'Match all' first."); return
        try:
            fig = self._new_fig()
            ax = fig.add_subplot(111)
            from py4DSTEM.process.diffraction.utils import (
                plot_orientation_maps)
            try:
                plot_orientation_maps(
                    self.orientation_map, self.crystal,
                    input_fig_handle=(fig, ax))
            except Exception:
                # Fallback: simple correlation map
                corr = getattr(self.orientation_map, "corr",
                                 None)
                if corr is not None:
                    arr = np.asarray(corr)
                    if arr.ndim == 3:
                        arr = arr[..., 0]
                    ax.imshow(arr, cmap="viridis")
            ax.set_title("orientation map")
            self._redraw()
        except Exception as e:
            messagebox.showerror("render orientation map", repr(e))
