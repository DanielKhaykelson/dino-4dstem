"""blob_panel.py -- Blob-detection tab for the GUI.

Mirrors the SAM tab layout (left controls, right canvas with raw vs.
processed views), but runs scikit-image blob detectors (DoG / LoG /
Determinant-of-Hessian) instead of SAM.

Workflow
--------
1. User links a finished training run (or one finishes automatically).
   The panel borrows PostHocPanel's `_compute_class_averages` to get
   the K class-averaged diffraction patterns.
2. Frame mode toggles between "Class average" (one 2D image per class)
   and "Single frame" (one specific scan-index pattern). Both feed the
   same detector.
3. Sidebar exposes:
       - Method picker (DoG / LoG / DoH) — switches the visible knobs.
       - Common preprocessing: blur σ, rescale lo / hi.
       - Method-specific knobs (min/max σ, threshold, etc.)
       - Output options.
       - Action buttons: Run-on-this-image, Apply-to-class, Apply-to-all.
4. Per-class tunings + outputs persist under
       <run>/blob/{method}/p<c>/blobs.npy
   and the cross-class config under <run>/blob/blob_config.json
   so re-opening a run restores the user's tuning.
"""
from __future__ import annotations
import os, sys, json, time, threading

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
from matplotlib.patches import Circle


# ------------------------------------------------------------------
# Defaults per method.  Picked to be sensible for log-stretched
# polar / cartesian diffraction patterns at H=W~192.
# ------------------------------------------------------------------
COMMON_DEFAULTS = dict(
    blur_sigma=1.0,
    rescale_lo=2.0,        # percentile
    rescale_hi=99.5,       # percentile
    log_stretch=True,
)

METHOD_DEFAULTS = {
    "DoG": dict(
        min_sigma=1.0,
        max_sigma=10.0,
        sigma_ratio=1.6,
        threshold=0.05,
        overlap=0.5,
    ),
    "LoG": dict(
        min_sigma=1.0,
        max_sigma=10.0,
        num_sigma=10,
        threshold=0.05,
        overlap=0.5,
        log_scale=False,    # skimage's `log_scale` for blob_log
    ),
    "DoH": dict(
        min_sigma=1.0,
        max_sigma=10.0,
        num_sigma=10,
        threshold=0.005,
        overlap=0.5,
        log_scale=False,
    ),
}

METHODS = ["DoG", "LoG", "DoH"]


# ------------------------------------------------------------------
# Helpers (mirror sam_panel.py style for visual consistency).
# ------------------------------------------------------------------
def _entry_row(parent, label, var, width=80):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=4, pady=1)
    ctk.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
    ctk.CTkEntry(row, textvariable=var, width=width).pack(side="left")
    return row


def _section(parent, title):
    f = ctk.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x", padx=2, pady=(8, 0))
    ctk.CTkLabel(f, text=title, font=("Segoe UI", 11, "bold")).pack(
        anchor="w")
    sep = ctk.CTkFrame(parent, height=2, fg_color=("#cccccc", "#444444"))
    sep.pack(fill="x", padx=2, pady=(0, 4))
    return parent


def _preprocess(img: np.ndarray, blur_sigma: float,
                rescale_lo: float, rescale_hi: float,
                log_stretch: bool) -> np.ndarray:
    """Same preprocessing pipeline used for SAM: percentile-rescale,
    optional log stretch, optional Gaussian blur. Returns a 2D float32
    array in [0, 1]."""
    from scipy.ndimage import gaussian_filter
    x = img.astype(np.float32)
    if log_stretch:
        x = np.log1p(np.clip(x, 0.0, None))
    lo = float(np.percentile(x, rescale_lo))
    hi = float(np.percentile(x, rescale_hi))
    x = np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    if blur_sigma and blur_sigma > 0:
        x = gaussian_filter(x, sigma=float(blur_sigma))
    return x.astype(np.float32)


def _detect(image: np.ndarray, method: str, kw: dict) -> np.ndarray:
    """Returns blob coords as (N, 3) array of [y, x, sigma]."""
    from skimage.feature import blob_dog, blob_log, blob_doh
    if method == "DoG":
        b = blob_dog(image,
                      min_sigma=float(kw["min_sigma"]),
                      max_sigma=float(kw["max_sigma"]),
                      sigma_ratio=float(kw["sigma_ratio"]),
                      threshold=float(kw["threshold"]),
                      overlap=float(kw["overlap"]))
    elif method == "LoG":
        b = blob_log(image,
                      min_sigma=float(kw["min_sigma"]),
                      max_sigma=float(kw["max_sigma"]),
                      num_sigma=int(kw["num_sigma"]),
                      threshold=float(kw["threshold"]),
                      overlap=float(kw["overlap"]),
                      log_scale=bool(kw.get("log_scale", False)))
        # blob_log returns sigma; effective radius is sigma * sqrt(2)
        if b.size:
            b = np.column_stack([b[:, 0], b[:, 1], b[:, 2] * np.sqrt(2)])
    elif method == "DoH":
        b = blob_doh(image,
                      min_sigma=float(kw["min_sigma"]),
                      max_sigma=float(kw["max_sigma"]),
                      num_sigma=int(kw["num_sigma"]),
                      threshold=float(kw["threshold"]),
                      overlap=float(kw["overlap"]),
                      log_scale=bool(kw.get("log_scale", False)))
    else:
        raise ValueError(f"unknown method: {method}")
    return np.asarray(b, dtype=np.float32) if len(b) else np.zeros((0, 3), np.float32)


# ------------------------------------------------------------------
# Panel
# ------------------------------------------------------------------
class BlobPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.outdir = None
        self.sample = None
        self._cube_path = None
        self._scan_shape = None
        self._class_avgs = None       # (K, H, W) numpy
        self._assigns = None
        self._K = 0
        self._per_class_cfg = {}
        self._last_run = None         # dict: image, blobs, raw, cls/idx
        self._poll_after = None
        self._worker_thread = None
        self._worker_running = False
        self._worker_progress = ""
        self._worker_lock = threading.Lock()
        self._raw_cache = {}          # frame_idx -> raw 2D
        self._build()
        # Subscribe to the global session so loading a run anywhere
        # (topbar badge, post-hoc, …) auto-links Blob.
        sess = getattr(self.app, "session", None) if self.app else None
        if sess is not None:
            sess.subscribe(self._on_session_change)
            self._on_session_change(sess)

    # ------------- run linkage ------------------------------------
    def link_run(self, outdir, sample):
        self.outdir = outdir
        self.sample = sample
        self._info_lbl.configure(
            text=f"linked: {os.path.basename(outdir)}  (sample={sample})")
        self._load_run_state()

    def _on_session_change(self, sess):
        """React to global session updates so Blob picks up the
        dataset/run loaded by any other tab or the topbar badge."""
        if sess is None: return
        if not (sess.has_dataset() and sess.has_run()):
            return
        if (self.outdir == sess.run_dir
                and self.sample == sess.sample):
            return
        try:
            self.link_run(sess.run_dir, sess.sample)
        except Exception as e:
            print(f"[blob-panel] session re-link failed: {e!r}",
                  flush=True)

    def on_runtime_sample_added(self, key):
        pass

    def _load_run_state(self):
        if not self.outdir or not os.path.isdir(self.outdir):
            return
        self._class_avgs = None
        self._assigns = None
        self._K = 0
        self._raw_cache = {}
        self._read_per_class_cfg()
        try:
            from data import SAMPLES
            cfg = SAMPLES.get(self.sample) or {}
            cube = cfg.get("path") or (cfg.get("paths") or [None])[0]
            self._cube_path = cube
            self._scan_shape = (cfg.get("scan_shape")
                                  or cfg.get("scan_size") or None)
        except Exception:
            self._cube_path = None
        self._refresh_class_menu()
        self._render_canvas_idle()

    # ------------- per-class config persistence -------------------
    def _config_path(self) -> str:
        return os.path.join(self.outdir, "blob", "blob_config.json")

    def _read_per_class_cfg(self):
        self._per_class_cfg = {}
        if not self.outdir:
            return
        cp = self._config_path()
        if not os.path.exists(cp):
            return
        try:
            with open(cp) as f:
                blob = json.load(f)
            if "method" in blob and "method" in self._vars:
                self._vars["method"].set(blob["method"])
            self._per_class_cfg = blob.get("per_class", {}) or {}
        except Exception as e:
            print(f"[blob-panel] config load failed: {e!r}", flush=True)

    def _write_per_class_cfg(self):
        if not self.outdir:
            return
        cp = self._config_path()
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        blob = {
            "method": self._vars["method"].get(),
            "per_class": self._per_class_cfg,
        }
        with open(cp, "w") as f:
            json.dump(blob, f, indent=2)

    def _gather_current_knobs(self) -> dict:
        d = {"method": self._vars["method"].get()}
        for k in ("blur_sigma", "rescale_lo", "rescale_hi", "log_stretch"):
            try: d[k] = self._vars[k].get()
            except Exception: d[k] = COMMON_DEFAULTS[k]
        method = d["method"]
        for k in METHOD_DEFAULTS[method]:
            var = self._vars["method_knobs"][method].get(k)
            if var is None:
                d[k] = METHOD_DEFAULTS[method][k]
            else:
                try: d[k] = var.get()
                except Exception: d[k] = METHOD_DEFAULTS[method][k]
        return d

    def _apply_knobs(self, d: dict):
        if "method" in d and "method" in self._vars:
            try: self._vars["method"].set(d["method"])
            except Exception: pass
            self._on_method_change()
        for k in ("blur_sigma", "rescale_lo", "rescale_hi", "log_stretch"):
            if k in d and k in self._vars:
                try: self._vars[k].set(d[k])
                except Exception: pass
        method = d.get("method", self._vars["method"].get())
        for k, v in d.items():
            var = self._vars["method_knobs"].get(method, {}).get(k)
            if var is not None:
                try: var.set(v)
                except Exception: pass

    # ------------- UI ---------------------------------------------
    def _build(self):
        # ----- vars (must exist before _read_per_class_cfg below) -----
        self._vars: dict = {
            "method":       ctk.StringVar(value="DoG"),
            "blur_sigma":   ctk.DoubleVar(value=COMMON_DEFAULTS["blur_sigma"]),
            "rescale_lo":   ctk.DoubleVar(value=COMMON_DEFAULTS["rescale_lo"]),
            "rescale_hi":   ctk.DoubleVar(value=COMMON_DEFAULTS["rescale_hi"]),
            "log_stretch":  ctk.BooleanVar(value=COMMON_DEFAULTS["log_stretch"]),
            "current_class": ctk.StringVar(value="0"),
            "frame_mode":   ctk.StringVar(value="Class average"),
            # New: extra grain-source state.  When frame_mode ==
            # "Grain @ (y, x)", read these.
            "grain_y":      ctk.StringVar(value="64"),
            "grain_x":      ctk.StringVar(value="64"),
            # Class-average member selection: "top200" (sharp, default)
            # vs "all" (honest, unfiltered mean over every member).
            "classavg_members": ctk.StringVar(value="top 200 (sharp)"),
            "frame_index":  ctk.IntVar(value=0),
        }
        # Per-method knob vars, kept distinct so switching method preserves
        # previously-tuned values for the other methods.
        self._vars["method_knobs"] = {
            m: {k: (ctk.IntVar(value=v) if isinstance(v, int)
                     and not isinstance(v, bool)
                   else (ctk.BooleanVar(value=v) if isinstance(v, bool)
                          else ctk.DoubleVar(value=v)))
                 for k, v in METHOD_DEFAULTS[m].items()}
            for m in METHODS
        }

        # ----- top bar -----
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)
        ctk.CTkButton(top, text="Load run dir…", width=140,
                       command=self._load_dir_dialog).pack(side="left",
                                                              padx=4)
        self._info_lbl = ctk.CTkLabel(top, text="(no run linked)",
                                        font=("Consolas", 10))
        self._info_lbl.pack(side="left", padx=8)

        method_box = ctk.CTkFrame(top, fg_color="transparent")
        method_box.pack(side="right", padx=4)
        ctk.CTkLabel(method_box, text="Method:").grid(row=0, column=0,
                                                        padx=(0, 4))
        self._method_menu = ctk.CTkOptionMenu(method_box,
            variable=self._vars["method"], values=METHODS, width=110,
            command=lambda _v: self._on_method_change())
        self._method_menu.grid(row=0, column=1, padx=2)

        # ----- body: sidebar + canvas -----
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        sidebar = ctk.CTkScrollableFrame(body, width=300)
        sidebar.pack(side="left", fill="y")

        # Class + frame selector
        _section(sidebar, "Class / frame")
        cls_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        cls_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(cls_row, text="class:", width=60).pack(side="left")
        self._class_menu = ctk.CTkOptionMenu(cls_row,
            variable=self._vars["current_class"], values=["0"], width=100,
            command=lambda _v: self._on_class_change())
        self._class_menu.pack(side="left", padx=2)
        ctk.CTkButton(cls_row, text="Reload class avgs", width=140,
                       command=self._compute_class_avgs).pack(side="left",
                                                                 padx=4)
        # Class-average member selection.
        ca_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        ca_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(ca_row, text="members:", width=70,
                       anchor="w").pack(side="left")
        ctk.CTkOptionMenu(ca_row,
            variable=self._vars["classavg_members"],
            values=["top 200 (sharp)", "top 500", "all (honest mean)"],
            width=170,
            command=lambda _v: self._compute_class_avgs()
            ).pack(side="left", padx=2)

        mode_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        mode_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(mode_row, text="frame:", width=60).pack(side="left")
        ctk.CTkOptionMenu(mode_row, variable=self._vars["frame_mode"],
            values=["Class average", "Single frame", "Grain @ (y, x)",
                       "dp_max", "dp_mean"],
            width=160,
            command=lambda _v: self._render_canvas_idle()
            ).pack(side="left", padx=2)

        idx_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        idx_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(idx_row, text="frame idx:", width=80,
                       anchor="w").pack(side="left")
        ctk.CTkEntry(idx_row, textvariable=self._vars["frame_index"],
                       width=80).pack(side="left")
        ctk.CTkLabel(idx_row, text="(flat scan idx)",
                       font=("Segoe UI", 9),
                       text_color=("#666", "#aaa")).pack(side="left", padx=4)
        # Grain @ (y, x) source: scan coords used when frame_mode is
        # "Grain @ (y, x)".  Uses the same connected-component grain
        # extraction as the post-hoc grain popup.
        g_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        g_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(g_row, text="grain (y, x):",
                      width=80, anchor="w").pack(side="left")
        ctk.CTkEntry(g_row, textvariable=self._vars["grain_y"],
                       width=50).pack(side="left", padx=2)
        ctk.CTkEntry(g_row, textvariable=self._vars["grain_x"],
                       width=50).pack(side="left", padx=2)

        # Common preprocessing
        _section(sidebar, "Preprocessing")
        ctk.CTkCheckBox(sidebar, text="log1p stretch",
                          variable=self._vars["log_stretch"]
                          ).pack(anchor="w", padx=8, pady=2)
        _entry_row(sidebar, "blur σ", self._vars["blur_sigma"])
        _entry_row(sidebar, "rescale lo %", self._vars["rescale_lo"])
        _entry_row(sidebar, "rescale hi %", self._vars["rescale_hi"])

        # Method-specific knobs (one frame per method, parented to a
        # placeholder container so layout stays correct after the user
        # switches method — a previous version repacked at the end of the
        # sidebar (after Actions), shoving the knobs below the buttons).
        _section(sidebar, "Detector parameters")
        self._method_container = ctk.CTkFrame(sidebar, fg_color="transparent")
        self._method_container.pack(fill="x")
        self._method_frames = {}
        for m in METHODS:
            f = ctk.CTkFrame(self._method_container, fg_color="transparent")
            for k, var in self._vars["method_knobs"][m].items():
                # Booleans get a checkbox, not an entry.
                if isinstance(var, ctk.BooleanVar):
                    ctk.CTkCheckBox(f, text=k, variable=var
                                      ).pack(anchor="w", padx=8, pady=2)
                else:
                    _entry_row(f, k, var)
            self._method_frames[m] = f
        # Show the active one
        self._method_frames[self._vars["method"].get()].pack(fill="x")

        # Action buttons
        _section(sidebar, "Actions")
        ctk.CTkButton(sidebar, text="Run on current image  ▶",
                       fg_color=("#2D7A2D", "#1F7A1F"),
                       command=self._run_on_current
                       ).pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(sidebar, text="Apply to whole class  ▶▶",
                       command=self._apply_to_class
                       ).pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(sidebar, text="Apply to ALL classes  ⏵⏵⏵",
                       command=self._apply_to_all
                       ).pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(sidebar, text="Save tuning to per-class config",
                       command=self._save_class_tuning
                       ).pack(fill="x", padx=8, pady=4)

        self._status_lbl = ctk.CTkLabel(sidebar,
            text="(no class loaded)", font=("Consolas", 9),
            justify="left", text_color=("#444", "#aaa"), wraplength=260)
        self._status_lbl.pack(anchor="w", padx=8, pady=(8, 4))

        # ----- canvas -----
        canv = ctk.CTkFrame(body)
        canv.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._fig = Figure(figsize=(9, 6))
        self._canvas = FigureCanvasTkAgg(self._fig, master=canv)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, canv)

        # Re-load saved global config now that vars + UI exist.
        self._read_per_class_cfg()

    # ------------- top-bar callbacks ------------------------------
    def _load_dir_dialog(self):
        p = filedialog.askdirectory(title="Pick a run dir")
        if not p:
            return
        # Route through the global Session so SAMPLE_LOCK.json /
        # run_summary.json / _train_kwargs.json get resolved
        # consistently with every other tab.
        sess = getattr(self.app, "session", None) if self.app else None
        sample = None
        if sess is not None:
            try:
                sample = sess.load_run_dir(p)
            except Exception as e:
                print(f"[blob-panel] session.load_run_dir: {e!r}",
                      flush=True)
        if sample is None:
            messagebox.showinfo("Blob",
                "Couldn't auto-resolve a sample for this run dir "
                "(no SAMPLE_LOCK.json / _train_kwargs.json / "
                "matching SAMPLES entry).  Pick a sample from the "
                "topbar dataset badge.")
            sample = "?"
        self.link_run(p, sample)

    def _on_method_change(self):
        m = self._vars["method"].get()
        for k, f in self._method_frames.items():
            f.pack_forget()
        self._method_frames[m].pack(fill="x")

    # ------------- class data loading -----------------------------
    def _compute_class_avgs(self):
        if not self.outdir:
            messagebox.showinfo("Blob", "Load a run dir first."); return
        if not self._cube_path:
            messagebox.showinfo("Blob",
                "No cube path resolved for this sample."); return
        self._status_lbl.configure(text="computing class averages …")
        self.update_idletasks()
        try:
            ph = getattr(self.app, "posthoc", None) if self.app else None
            if ph is not None and getattr(ph, "outdir", None) != self.outdir:
                ph.link_run(self.outdir, self.sample)
            if ph is None or not hasattr(ph, "_compute_class_averages"):
                messagebox.showinfo("Blob",
                    "Need PostHocPanel to compute class averages.")
                return
            if not ph._ensure_inference():
                self._status_lbl.configure(
                    text="inference failed — see Post-hoc tab")
                return
            sel = self._vars["classavg_members"].get()
            top_n = (None if sel.startswith("all")
                       else 500 if "500" in sel else 200)
            self._class_avgs = np.asarray(
                ph._compute_class_averages(top_n=top_n))
            self._assigns = ph._inf["assigns"]
            self._K = int(ph._inf["soft_probs"].shape[1])
            self._scan_shape = ph._scan_shape
            self._refresh_class_menu()
            self._status_lbl.configure(
                text=f"class averages ready (K={self._K}). "
                      f"Scan: {self._scan_shape}")
            self._render_canvas_idle()
        except Exception as e:
            messagebox.showerror("Blob", f"compute class avgs failed:\n{e!r}")

    def _refresh_class_menu(self):
        K = max(1, self._K)
        self._class_menu.configure(values=[str(c) for c in range(K)])
        if self._vars["current_class"].get() not in [str(c) for c in range(K)]:
            self._vars["current_class"].set("0")

    def _on_class_change(self):
        c = self._vars["current_class"].get()
        if c in self._per_class_cfg:
            self._apply_knobs(self._per_class_cfg[c])
            self._status_lbl.configure(
                text=f"loaded saved tuning for class p{c}")
        self._render_canvas_idle()

    # ------------- pulling raw frames -----------------------------
    def _get_raw_frame(self, idx: int) -> np.ndarray | None:
        """Pull a single 2D pattern from disk, with a small LRU cache."""
        if idx in self._raw_cache:
            return self._raw_cache[idx]
        try:
            from data import LoadPRZ, SAMPLES
            cfg = SAMPLES[self.sample]
            ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
            raw = ds.get_raw(int(idx)).astype(np.float32)
        except Exception as e:
            print(f"[blob-panel] frame load failed: {e!r}", flush=True)
            return None
        if len(self._raw_cache) > 16:
            self._raw_cache.pop(next(iter(self._raw_cache)))
        self._raw_cache[idx] = raw
        return raw

    def _current_image(self) -> tuple[np.ndarray | None, str]:
        """Return (image, label) according to frame_mode."""
        mode = self._vars["frame_mode"].get()
        if mode == "Class average":
            if self._class_avgs is None:
                return None, ""
            c = int(self._vars["current_class"].get())
            return self._class_avgs[c], f"p{c} class average"
        if mode == "Single frame":
            i = int(self._vars["frame_index"].get())
            raw = self._get_raw_frame(i)
            return raw, f"frame [{i}]"
        if mode == "Grain @ (y, x)":
            # Re-use the posthoc grain extractor (connected-component
            # of same-class pixels containing the click).
            ph = (getattr(self.app, "posthoc", None)
                    if self.app else None)
            if ph is None or ph._inf is None:
                return None, "(need posthoc inference for grain mode)"
            try:
                y = int(self._vars["grain_y"].get())
                x = int(self._vars["grain_x"].get())
                gi = ph._compute_grain_average(y, x)
            except Exception:
                return None, "(grain extract failed)"
            if gi is None:
                return None, "(pixel not in any grain)"
            return (gi["grain_avg"].astype(np.float32),
                      f"grain @ ({y},{x}) p{gi['cls']} {gi['n_pix']}px")
        if mode in ("dp_max", "dp_mean"):
            try:
                pat = self._compute_dp_image(mode)
            except Exception as e:
                return None, f"({mode}: {e!r})"
            if pat is None:
                return None, f"(no cube for {mode})"
            return pat.astype(np.float32), f"{mode} ({self.sample})"
        return None, f"(unknown mode {mode!r})"

    def _compute_dp_image(self, kind: str):
        """Stream the cube row-by-row to build dp_max or dp_mean,
        cached per-sample so toggling doesn't recompute."""
        if not hasattr(self, "_dp_cache"):
            self._dp_cache = {}
        key = (self.sample, kind)
        if key in self._dp_cache:
            return self._dp_cache[key]
        from data import SAMPLES
        from gui_app.posthoc_panel import _open_lazy
        if self.sample not in SAMPLES: return None
        cfg = SAMPLES[self.sample]
        cube = _open_lazy(cfg["path"], scan_shape=self._scan_shape)
        Ny, Nx, H, W = cube.shape
        dp_max = np.zeros((H, W), dtype=np.float32)
        dp_sum = np.zeros((H, W), dtype=np.float64)
        for y in range(Ny):
            blk = np.asarray(cube[y], dtype=np.float32)
            dp_max = np.maximum(dp_max, blk.max(axis=0))
            dp_sum += blk.sum(axis=0)
        dp_mean = (dp_sum / max(Ny * Nx, 1)).astype(np.float32)
        self._dp_cache[(self.sample, "dp_max")]  = dp_max
        self._dp_cache[(self.sample, "dp_mean")] = dp_mean
        return self._dp_cache[key]

    # ------------- single-image run --------------------------------
    def _run_on_current(self):
        if self._vars["frame_mode"].get() == "Class average" \
                and self._class_avgs is None:
            self._compute_class_avgs()
        img, label = self._current_image()
        if img is None:
            messagebox.showinfo("Blob",
                "No image to run on. Load a run, then either reload "
                "class averages or pick a valid frame index.")
            return
        try:
            kw = self._gather_current_knobs()
            t0 = time.perf_counter()
            prep = _preprocess(img, kw["blur_sigma"], kw["rescale_lo"],
                                kw["rescale_hi"], kw["log_stretch"])
            blobs = _detect(prep, kw["method"], kw)
            el = time.perf_counter() - t0
        except Exception as e:
            messagebox.showerror("Blob", f"detection failed:\n{e!r}")
            self._status_lbl.configure(text=f"error: {e!r}")
            return
        self._last_run = dict(raw=img, prep=prep, blobs=blobs, label=label,
                                method=kw["method"])
        self._status_lbl.configure(
            text=f"{label}  [{kw['method']}]: {len(blobs)} blobs  ({el*1000:.0f}ms)")
        self._render_canvas()

    # ------------- whole-class run (threaded) ---------------------
    def _apply_to_class(self):
        if self._worker_running:
            messagebox.showinfo("Blob",
                "A blob run is already in progress. Wait for it to finish.")
            return
        if self._assigns is None:
            self._compute_class_avgs()
            if self._assigns is None:
                return
        c = int(self._vars["current_class"].get())
        idx = np.where(self._assigns == c)[0]
        n = int(idx.size)
        if n == 0:
            messagebox.showinfo("Blob", f"class p{c} has no patterns."); return
        ok = messagebox.askyesno("Confirm blob run",
            f"Run {self._vars['method'].get()} on class p{c}: "
            f"{n} patterns.\n\nEstimated time: ~{n*0.05:.1f}s "
            f"(~50ms/pattern).\n\nContinue?")
        if not ok:
            return
        self._save_class_tuning()
        kw = self._gather_current_knobs()
        outdir = os.path.join(self.outdir, "blob", kw["method"], f"p{c}")
        os.makedirs(outdir, exist_ok=True)
        # Spawn a worker thread (blob detection releases the GIL via
        # numpy/scipy, and the GUI thread keeps polling).
        self._worker_running = True
        self._worker_progress = "starting…"
        self._worker_thread = threading.Thread(
            target=self._whole_class_worker,
            args=(idx, c, kw, outdir), daemon=True)
        self._worker_thread.start()
        self._status_lbl.configure(
            text=f"class p{c}: blob detection running  ({n} patterns).  "
                  f"outputs → {outdir}")
        self._poll()

    def _apply_to_all(self):
        if self._worker_running:
            messagebox.showinfo("Blob",
                "A blob run is already in progress."); return
        if self._assigns is None:
            self._compute_class_avgs()
            if self._assigns is None:
                return
        K = self._K
        ok = messagebox.askyesno("Confirm blob run on ALL classes",
            f"Run {self._vars['method'].get()} on every class (K={K}).\n\n"
            f"Each class uses its saved per-class tuning if present, else "
            f"the current sidebar knobs.\n\nContinue?")
        if not ok:
            return
        kw_global = self._gather_current_knobs()
        method = kw_global["method"]
        # Build per-class kwargs dicts (saved tuning takes precedence).
        per_class_kw = []
        for c in range(K):
            ck = dict(kw_global)
            saved = self._per_class_cfg.get(str(c))
            if saved:
                ck.update(saved)
                ck["method"] = method   # method is a global toggle
            per_class_kw.append(ck)
        self._worker_running = True
        self._worker_progress = "starting…"
        self._worker_thread = threading.Thread(
            target=self._all_classes_worker,
            args=(per_class_kw, method), daemon=True)
        self._worker_thread.start()
        self._status_lbl.configure(
            text=f"running {method} on ALL {K} classes …")
        self._poll()

    def _whole_class_worker(self, idx_arr, class_idx, kw, outdir):
        """Runs in a thread — must not touch tk widgets directly."""
        try:
            from data import LoadPRZ, SAMPLES
            cfg = SAMPLES[self.sample]
            ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
            n = len(idx_arr)
            blobs_out = []
            t0 = time.perf_counter()
            for i, scan_idx in enumerate(idx_arr):
                raw = ds.get_raw(int(scan_idx)).astype(np.float32)
                prep = _preprocess(raw, kw["blur_sigma"], kw["rescale_lo"],
                                    kw["rescale_hi"], kw["log_stretch"])
                b = _detect(prep, kw["method"], kw)
                blobs_out.append(b)
                if (i + 1) % max(1, n // 50) == 0:
                    el = time.perf_counter() - t0
                    rate = (i + 1) / max(el, 1e-3)
                    eta = (n - i - 1) / max(rate, 1e-3)
                    with self._worker_lock:
                        self._worker_progress = (
                            f"{i+1}/{n}  ({rate:.1f}/s, ETA {eta:.0f}s)")
            # Persist
            np.save(os.path.join(outdir, "blob_indices.npy"),
                    np.asarray(idx_arr, dtype=np.int64))
            counts = np.array([len(b) for b in blobs_out], dtype=np.int32)
            np.save(os.path.join(outdir, "blob_counts.npy"), counts)
            # Coords as ragged-friendly format: concatenate + offsets.
            if any(len(b) for b in blobs_out):
                concat = np.concatenate(blobs_out, axis=0).astype(np.float32)
            else:
                concat = np.zeros((0, 3), np.float32)
            offsets = np.zeros(len(blobs_out) + 1, dtype=np.int64)
            for j, b in enumerate(blobs_out):
                offsets[j + 1] = offsets[j] + len(b)
            np.save(os.path.join(outdir, "blob_coords.npy"), concat)
            np.save(os.path.join(outdir, "blob_offsets.npy"), offsets)
            # Render a count map over the scan grid.
            self._render_count_map(outdir, class_idx, idx_arr, counts)
            with open(os.path.join(outdir, "_blob_run_summary.json"), "w") as fh:
                json.dump({
                    "method": kw["method"],
                    "class": int(class_idx),
                    "n_patterns": int(n),
                    "total_blobs": int(counts.sum()),
                    "mean_blobs_per_pattern": float(counts.mean()),
                    "median_blobs_per_pattern": float(np.median(counts)),
                    "knobs": {k: v for k, v in kw.items()
                                if k not in ("method_knobs",)},
                    "elapsed_s": float(time.perf_counter() - t0),
                }, fh, indent=2)
            with self._worker_lock:
                self._worker_progress = (
                    f"done. {counts.sum()} blobs across {n} patterns "
                    f"({counts.mean():.1f}/pattern).")
        except Exception as e:
            with self._worker_lock:
                self._worker_progress = f"failed: {e!r}"
            print(f"[blob-panel] worker failed: {e!r}", flush=True)
        finally:
            self._worker_running = False

    def _all_classes_worker(self, per_class_kw, method):
        try:
            for c, kw in enumerate(per_class_kw):
                idx_arr = np.where(self._assigns == c)[0]
                if idx_arr.size == 0:
                    continue
                outdir = os.path.join(self.outdir, "blob", method, f"p{c}")
                os.makedirs(outdir, exist_ok=True)
                with self._worker_lock:
                    self._worker_progress = (
                        f"class p{c}: starting  ({idx_arr.size} patterns)")
                # Reuse the single-class worker body (inline to share state).
                self._worker_running = True
                # Run synchronously inside this thread so we serialize classes.
                self._whole_class_worker(idx_arr, c, kw, outdir)
                # _whole_class_worker sets _worker_running=False on exit;
                # restore for the next class.
                self._worker_running = True
            with self._worker_lock:
                self._worker_progress = "all classes done."
        finally:
            self._worker_running = False

    def _render_count_map(self, outdir: str, class_idx: int,
                           idx_arr: np.ndarray, counts: np.ndarray):
        """Render and save a 2D map of blob counts on the scan grid."""
        try:
            if not self._scan_shape:
                return
            Ny, Nx = self._scan_shape
            full = np.full((Ny * Nx,), np.nan, dtype=np.float32)
            full[idx_arr] = counts.astype(np.float32)
            grid = full.reshape(Ny, Nx)
            fig = Figure(figsize=(6, 5))
            ax = fig.add_subplot(111)
            im = ax.imshow(grid, cmap="viridis")
            ax.set_title(f"class p{class_idx}: blob count per pattern")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "blob_count_map.png"), dpi=120)
        except Exception as e:
            print(f"[blob-panel] count-map render failed: {e!r}", flush=True)

    def _save_class_tuning(self):
        if not self.outdir:
            return
        c = self._vars["current_class"].get()
        d = self._gather_current_knobs()
        # Drop matplotlib-unfriendly types
        d = {k: (bool(v) if isinstance(v, (bool,)) else
                 (float(v) if isinstance(v, float) else
                  (int(v) if isinstance(v, int) else v)))
             for k, v in d.items() if k != "method_knobs"}
        self._per_class_cfg[c] = d
        try:
            self._write_per_class_cfg()
            self._status_lbl.configure(
                text=f"saved tuning for class p{c} to blob_config.json")
        except Exception as e:
            messagebox.showerror("Blob", f"could not save tuning:\n{e!r}")

    # ------------- worker polling ---------------------------------
    def _poll(self):
        with self._worker_lock:
            prog = self._worker_progress
        running = self._worker_running
        self._status_lbl.configure(text=prog or "(running…)")
        if running:
            self._poll_after = self.after(500, self._poll)
        else:
            self._status_lbl.configure(text=prog or "(idle)")

    # ------------- canvas rendering -------------------------------
    def _render_canvas_idle(self):
        self._fig.clear()
        img, label = self._current_image()
        if img is None:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Load a run dir, then 'Reload class avgs', then\n"
                    "'Run on current image' to detect blobs.",
                    ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            self._canvas.draw_idle(); return
        ax = self._fig.add_subplot(111)
        ax.imshow(img, cmap="inferno")
        ax.set_title(f"{label}  (no detection yet)")
        ax.set_axis_off()
        self._canvas.draw_idle()

    def _render_canvas(self):
        if self._last_run is None:
            self._render_canvas_idle(); return
        self._fig.clear()
        raw = self._last_run["raw"]
        prep = self._last_run["prep"]
        blobs = self._last_run["blobs"]
        label = self._last_run["label"]
        method = self._last_run["method"]

        ax1 = self._fig.add_subplot(1, 2, 1)
        ax1.imshow(raw, cmap="inferno")
        ax1.set_title(f"{label}  (raw)")
        ax1.set_axis_off()

        ax2 = self._fig.add_subplot(1, 2, 2)
        ax2.imshow(prep, cmap="inferno")
        cmap = matplotlib.colormaps.get_cmap("tab10")
        for i, (y, x, sigma) in enumerate(blobs):
            colour = cmap(i % 10)
            r = float(sigma) * (np.sqrt(2) if method == "DoH" else 1.0)
            ax2.add_patch(Circle((x, y), r, edgecolor=colour,
                                    facecolor="none", lw=1.2))
        ax2.set_title(f"{label}  [{method}]: {len(blobs)} blobs")
        ax2.set_axis_off()
        self._fig.tight_layout()
        self._canvas.draw_idle()
