"""sam_panel.py -- SAM (Segment Anything Model) tab for the GUI.

Workflow
--------
1. User links a finished training run (or one finishes automatically
   via the Train/Eval pipeline).  The panel loads the K class-averaged
   diffraction patterns from disk.
2. User picks a class, tunes preprocessing + geometric filter knobs in
   the sidebar.  "Run on this average" runs SAM on a single image
   (~0.5-1 s on RTX 4080) so tuning is interactive.
3. Once happy, "Apply to whole class" spawns a subprocess that runs the
   tuned SAM pipeline over every pattern in that class.  The job runs
   in the background; the GUI close-handler kills it cleanly if the
   user quits with a job still running.
4. Per-class tuning + outputs are persisted under
   <run>/sam/p<c>/{angle.npy, masks_rle.npz, angle_map.png,
                    _sam_run_summary.json}
   and the cross-class config under <run>/sam/sam_config.json
   (auto-loaded when re-opening the run).
"""
from __future__ import annotations
import os, sys, json, time, subprocess

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
from matplotlib.patches import Rectangle


# Default knobs (mirror the original SAM/sam_image_utils.py)
SAM_DEFAULTS = dict(
    blur_sigma=4.0,
    rescale_lo=0.0,
    rescale_hi=0.6,
    downsample=0.5,
    area_min=50,
    area_max=1000,
    aspect_min=1.2,
    min_length=20,
    min_dist=30,
    max_dist=130,
    min_r2=0.5,
    skip_largest=True,
    candidate_max=10,
    sam_model_type="vit_h",
    sam_checkpoint="",
    save_masks=True,
)


def _entry_row(parent, label, var, width=80, help_text=""):
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


# ---------------------------------------------------------------------------
# SamRunJob: thin subprocess wrapper for the per-class SAM run.
# Same interface as TrainingJob (.is_running(), .stop()) so the GUI
# close-confirmation handler can manage it uniformly.
# ---------------------------------------------------------------------------

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_sam_worker.py")


class SamRunJob:
    def __init__(self, outdir: str, kwargs: dict):
        self.outdir = outdir
        self.kwargs = kwargs
        self._proc = None
        self._t_start = 0.0
        self._t_end = 0.0
        self._stopped = False

    def start(self):
        if self.is_running():
            return
        os.makedirs(self.outdir, exist_ok=True)
        spec = dict(self.kwargs)
        spec["outdir"] = self.outdir
        spec_path = os.path.join(self.outdir, "_sam_kwargs.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2, default=str)
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        self._t_start = time.perf_counter()
        self._proc = subprocess.Popen(
            [sys.executable, "-u", WORKER, spec_path],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            stdout=open(os.path.join(self.outdir, "_stdout.log"), "w",
                          encoding="utf-8"),
            stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                            if os.name == "nt" else 0),
        )

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, kill_after_s: float = 4.0):
        if not self.is_running():
            return
        self._stopped = True
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=kill_after_s)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
        except Exception:
            pass
        self._t_end = time.perf_counter()

    def status(self) -> str:
        if self._proc is None: return "idle"
        rc = self._proc.poll()
        if rc is None: return "running"
        if not self._t_end: self._t_end = time.perf_counter()
        if self._stopped: return "stopped"
        if os.path.exists(os.path.join(self.outdir, "_done.flag")): return "done"
        if os.path.exists(os.path.join(self.outdir, "_error.txt")) or rc != 0:
            return "failed"
        return "done"

    def progress(self) -> str:
        p = os.path.join(self.outdir, "_progress.txt")
        if os.path.exists(p):
            try: return open(p).read().strip()
            except Exception: return ""
        return ""


# ---------------------------------------------------------------------------
# SAMPanel
# ---------------------------------------------------------------------------

class SAMPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.outdir = None       # linked run directory
        self.sample = None       # sample key
        self._cube_path = None   # path to .prz cube
        self._scan_shape = None  # (Ny, Nx)
        self._class_avgs = None  # (K, H, W) numpy
        self._assigns = None     # per-pattern argmax
        self._K = 0
        self._per_class_cfg = {}  # str(c) -> dict of knobs
        self.job = None
        self._poll_after = None
        # Last single-image SAM result (for the canvas)
        self._last_run = None    # dict with prep, all_masks, filtered

        self._build()

    # ----- run linkage ----------------------------------------------
    def link_run(self, outdir, sample):
        """Hook called by gui_dino4dstem when a run finishes."""
        self.outdir = outdir
        self.sample = sample
        self._info_lbl.configure(
            text=f"linked: {os.path.basename(outdir)}  "
                  f"(sample={sample})")
        self._load_run_state()

    def on_runtime_sample_added(self, key):
        # Nothing to do — we don't keep a sample dropdown of our own.
        pass

    def _load_run_state(self):
        if not self.outdir or not os.path.isdir(self.outdir):
            return
        # Look for class averages on disk (Post-hoc saves these or we
        # compute them here from the latest checkpoint).
        # For MVP, compute on demand the first time the user picks a class.
        self._class_avgs = None
        self._assigns = None
        self._K = 0
        self._read_per_class_cfg()
        # Resolve cube path
        try:
            from data import SAMPLES
            cfg = SAMPLES.get(self.sample) or {}
            cube = cfg.get("path") or (cfg.get("paths") or [None])[0]
            self._cube_path = cube
            self._scan_shape = (cfg.get("scan_shape")
                                  or cfg.get("scan_size")
                                  or None)
        except Exception:
            self._cube_path = None
        # Reset class dropdown — populated lazily after inference.
        self._refresh_class_menu()

    # ----- per-class config persistence -----------------------------
    def _config_path(self) -> str:
        return os.path.join(self.outdir, "sam", "sam_config.json")

    def _read_per_class_cfg(self):
        self._per_class_cfg = {}
        cp = self._config_path()
        if os.path.exists(cp):
            try:
                with open(cp) as f:
                    blob = json.load(f)
                # Top-level globals (ckpt path, model type) override defaults
                for k in ("sam_checkpoint", "sam_model_type"):
                    if k in blob and k in self._vars:
                        self._vars[k].set(blob[k])
                self._per_class_cfg = blob.get("per_class", {}) or {}
            except Exception as e:
                print(f"[sam-panel] sam_config.json load failed: {e!r}",
                      flush=True)

    def _write_per_class_cfg(self):
        if not self.outdir:
            return
        cp = self._config_path()
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        blob = {
            "sam_checkpoint": self._vars["sam_checkpoint"].get(),
            "sam_model_type": self._vars["sam_model_type"].get(),
            "per_class": self._per_class_cfg,
        }
        with open(cp, "w") as f:
            json.dump(blob, f, indent=2)

    def _gather_current_knobs(self) -> dict:
        d = {}
        for k in ("blur_sigma", "rescale_lo", "rescale_hi", "downsample",
                  "area_min", "area_max", "aspect_min", "min_length",
                  "min_dist", "max_dist", "min_r2", "skip_largest",
                  "candidate_max", "save_masks"):
            try:
                d[k] = self._vars[k].get()
            except Exception:
                d[k] = SAM_DEFAULTS.get(k)
        return d

    def _apply_knobs(self, d: dict):
        for k, v in d.items():
            if k in self._vars:
                try: self._vars[k].set(v)
                except Exception: pass

    # ----- UI -------------------------------------------------------
    def _build(self):
        # --- top bar ---
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)
        ctk.CTkButton(top, text="Load run dir…", width=140,
                       command=self._load_dir_dialog).pack(side="left",
                                                              padx=4)
        self._info_lbl = ctk.CTkLabel(top, text="(no run linked)",
                                        font=("Consolas", 10))
        self._info_lbl.pack(side="left", padx=8)

        # SAM model section
        modsep = ctk.CTkFrame(top, fg_color="transparent")
        modsep.pack(side="right", padx=4)
        ctk.CTkLabel(modsep, text="SAM ckpt:").grid(row=0, column=0,
                                                       padx=(0, 2))
        self._vars = {
            "sam_checkpoint": ctk.StringVar(
                value=SAM_DEFAULTS["sam_checkpoint"]),
            "sam_model_type": ctk.StringVar(
                value=SAM_DEFAULTS["sam_model_type"]),
            "blur_sigma":  ctk.DoubleVar(value=SAM_DEFAULTS["blur_sigma"]),
            "rescale_lo":  ctk.DoubleVar(value=SAM_DEFAULTS["rescale_lo"]),
            "rescale_hi":  ctk.DoubleVar(value=SAM_DEFAULTS["rescale_hi"]),
            "downsample":  ctk.DoubleVar(value=SAM_DEFAULTS["downsample"]),
            "area_min":    ctk.IntVar   (value=SAM_DEFAULTS["area_min"]),
            "area_max":    ctk.IntVar   (value=SAM_DEFAULTS["area_max"]),
            "aspect_min":  ctk.DoubleVar(value=SAM_DEFAULTS["aspect_min"]),
            "min_length":  ctk.IntVar   (value=SAM_DEFAULTS["min_length"]),
            "min_dist":    ctk.DoubleVar(value=SAM_DEFAULTS["min_dist"]),
            "max_dist":    ctk.DoubleVar(value=SAM_DEFAULTS["max_dist"]),
            "min_r2":      ctk.DoubleVar(value=SAM_DEFAULTS["min_r2"]),
            "skip_largest": ctk.BooleanVar(value=SAM_DEFAULTS["skip_largest"]),
            "candidate_max": ctk.IntVar(value=SAM_DEFAULTS["candidate_max"]),
            "save_masks":  ctk.BooleanVar(value=SAM_DEFAULTS["save_masks"]),
            "current_class": ctk.StringVar(value="0"),
        }
        ctk.CTkEntry(modsep, textvariable=self._vars["sam_checkpoint"],
                       width=320).grid(row=0, column=1, padx=2)
        ctk.CTkButton(modsep, text="Browse…", width=70,
                       command=self._pick_ckpt).grid(row=0, column=2,
                                                        padx=2)
        ctk.CTkOptionMenu(modsep, variable=self._vars["sam_model_type"],
                            values=["vit_h", "vit_l", "vit_b"], width=80
                            ).grid(row=0, column=3, padx=4)

        # --- body: sidebar + canvas ---
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        sidebar = ctk.CTkScrollableFrame(body, width=300)
        sidebar.pack(side="left", fill="y")

        # Class selector
        _section(sidebar, "Class")
        cls_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        cls_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(cls_row, text="class:", width=60).pack(side="left")
        self._class_menu = ctk.CTkOptionMenu(cls_row,
            variable=self._vars["current_class"],
            values=["0"], width=100,
            command=lambda _v: self._on_class_change())
        self._class_menu.pack(side="left", padx=2)
        ctk.CTkButton(cls_row, text="Reload class avgs", width=140,
                       command=self._compute_class_avgs).pack(side="left",
                                                                 padx=4)

        # Preprocessing
        _section(sidebar, "Preprocessing")
        _entry_row(sidebar, "blur σ", self._vars["blur_sigma"])
        _entry_row(sidebar, "rescale lo", self._vars["rescale_lo"])
        _entry_row(sidebar, "rescale hi", self._vars["rescale_hi"])
        _entry_row(sidebar, "downsample", self._vars["downsample"])

        # Geometric filter
        _section(sidebar, "Geometric filter")
        _entry_row(sidebar, "area min", self._vars["area_min"])
        _entry_row(sidebar, "area max", self._vars["area_max"])
        _entry_row(sidebar, "aspect min", self._vars["aspect_min"])
        _entry_row(sidebar, "min length", self._vars["min_length"])
        _entry_row(sidebar, "min dist", self._vars["min_dist"])
        _entry_row(sidebar, "max dist", self._vars["max_dist"])
        _entry_row(sidebar, "min r²", self._vars["min_r2"])
        ctk.CTkCheckBox(sidebar, text="skip largest mask",
                          variable=self._vars["skip_largest"]
                          ).pack(anchor="w", padx=8, pady=2)
        _entry_row(sidebar, "candidate max", self._vars["candidate_max"])

        # Output options
        _section(sidebar, "Output")
        ctk.CTkCheckBox(sidebar, text="save filtered mask RLE  (~tens of MB)",
                          variable=self._vars["save_masks"]
                          ).pack(anchor="w", padx=8, pady=2)

        # Action buttons
        _section(sidebar, "Actions")
        ctk.CTkButton(sidebar, text="Run on this average  ▶",
                       fg_color=("#2D7A2D", "#1F7A1F"),
                       command=self._run_on_avg
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
            text="(no class loaded)",
            font=("Consolas", 9), justify="left",
            text_color=("#444", "#aaa"), wraplength=260)
        self._status_lbl.pack(anchor="w", padx=8, pady=(8, 4))

        # --- canvas ---
        canv = ctk.CTkFrame(body)
        canv.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._fig = Figure(figsize=(9, 6))
        self._canvas = FigureCanvasTkAgg(self._fig, master=canv)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, canv)

    # ----- top-bar callbacks ----------------------------------------
    def _pick_ckpt(self):
        p = filedialog.askopenfilename(
            title="SAM checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth"),
                       ("All files", "*.*")])
        if p:
            self._vars["sam_checkpoint"].set(p)

    def _load_dir_dialog(self):
        p = filedialog.askdirectory(title="Pick a run dir")
        if not p:
            return
        # Reuse the post-hoc panel's auto-register-from-train-kwargs trick.
        sample = "?"
        try:
            tk_path = os.path.join(p, "_train_kwargs.json")
            if os.path.exists(tk_path):
                with open(tk_path) as f:
                    blob = json.load(f)
                sample = blob.get("sample", "?")
                cfg = blob.get("_sample_config")
                if cfg and sample.startswith("loaded__"):
                    try:
                        from data import register_runtime_sample
                        register_runtime_sample(key=sample, **cfg)
                    except Exception as e:
                        print(f"[sam-panel] runtime-sample register: "
                              f"{e!r}", flush=True)
        except Exception as e:
            print(f"[sam-panel] _train_kwargs load failed: {e!r}",
                  flush=True)
        self.link_run(p, sample)

    # ----- class data loading ---------------------------------------
    def _compute_class_avgs(self):
        """Run inference on the linked run's best/latest checkpoint and
        compute per-class average diffraction patterns.  Also populate
        the class dropdown."""
        if not self.outdir:
            messagebox.showinfo("SAM", "Load a run dir first."); return
        if not self._cube_path:
            messagebox.showinfo("SAM",
                "No cube path resolved for this sample."); return
        self._status_lbl.configure(text="computing class averages …")
        self.update_idletasks()
        try:
            # Inference is heavy — delegate to PostHocPanel's helper if
            # we have a reference, else do a minimal copy here.
            ph = getattr(self.app, "posthoc", None) if self.app else None
            if ph is not None and (
                    not getattr(ph, "outdir", None) == self.outdir):
                # link the post-hoc panel to the same run so its
                # _ensure_inference + _compute_class_averages work
                ph.link_run(self.outdir, self.sample)
            if ph is None or not hasattr(ph, "_compute_class_averages"):
                messagebox.showinfo("SAM",
                    "Need PostHocPanel to compute class averages.")
                return
            if not ph._ensure_inference():
                self._status_lbl.configure(
                    text="inference failed — see Post-hoc tab")
                return
            self._class_avgs = np.asarray(
                ph._compute_class_averages())
            self._assigns = ph._inf["assigns"]
            self._K = int(ph._inf["soft_probs"].shape[1])
            self._scan_shape = ph._scan_shape
            self._refresh_class_menu()
            self._status_lbl.configure(
                text=f"class averages ready (K={self._K}). "
                      f"Scan: {self._scan_shape}")
            self._render_canvas_idle()
        except Exception as e:
            messagebox.showerror("SAM", f"compute class avgs failed:\n{e!r}")

    def _refresh_class_menu(self):
        K = max(1, self._K)
        self._class_menu.configure(values=[str(c) for c in range(K)])
        if self._vars["current_class"].get() not in [str(c) for c in range(K)]:
            self._vars["current_class"].set("0")

    def _on_class_change(self):
        # Load this class's saved tuning if present.
        c = self._vars["current_class"].get()
        if c in self._per_class_cfg:
            self._apply_knobs(self._per_class_cfg[c])
            self._status_lbl.configure(
                text=f"loaded saved tuning for class p{c}")
        self._render_canvas_idle()

    # ----- single-image SAM run -------------------------------------
    def _run_on_avg(self):
        if self._class_avgs is None:
            self._compute_class_avgs()
            if self._class_avgs is None:
                return
        c = int(self._vars["current_class"].get())
        avg2d = self._class_avgs[c]
        ckpt = self._vars["sam_checkpoint"].get().strip()
        if not ckpt or not os.path.exists(ckpt):
            messagebox.showerror(
                "SAM",
                f"SAM checkpoint not found:\n{ckpt}\n\n"
                f"Pick the .pth file via the Browse button.")
            return
        self._status_lbl.configure(text=f"running SAM on p{c} average …")
        self.update_idletasks()
        try:
            from sam_utils import SamMaskProcessor, filter_masks
        except ImportError as e:
            messagebox.showerror("SAM", str(e)); return
        try:
            proc = SamMaskProcessor(
                checkpoint_path=ckpt,
                model_type=self._vars["sam_model_type"].get(),
                device="cuda",
            )
            knobs = self._gather_current_knobs()
            t0 = time.perf_counter()
            prep, all_m, flt = proc.run_one(
                avg2d.astype(np.float32),
                blur_sigma=knobs["blur_sigma"],
                rescale_lo=knobs["rescale_lo"],
                rescale_hi=knobs["rescale_hi"],
                downsample=knobs["downsample"],
                filter_kwargs=dict(
                    area_range=(knobs["area_min"], knobs["area_max"]),
                    aspect_ratio_threshold=knobs["aspect_min"],
                    min_length=knobs["min_length"],
                    min_distance=knobs["min_dist"],
                    max_distance=knobs["max_dist"],
                    min_r2=knobs["min_r2"],
                    skip_largest=knobs["skip_largest"],
                    candidate_slice=(1, knobs["candidate_max"]),
                ),
            )
            self._last_run = dict(prep=prep, all_masks=all_m, filtered=flt,
                                    cls=c)
            el = time.perf_counter() - t0
            self._status_lbl.configure(
                text=f"p{c}: {len(all_m)} raw masks → "
                      f"{len(flt)} filtered  ({el:.1f}s)")
            self._render_canvas()
        except Exception as e:
            messagebox.showerror("SAM", f"single-image run failed:\n{e!r}")
            self._status_lbl.configure(text=f"error: {e!r}")

    # ----- whole-class SAM run --------------------------------------
    def _build_class_kwargs(self, class_idx: int) -> dict:
        knobs = self._gather_current_knobs()
        # Pattern indices for THIS class (flat scan-index)
        if self._assigns is None:
            raise RuntimeError("no assignments — Reload class avgs first")
        idx = np.where(self._assigns == class_idx)[0].tolist()
        return dict(
            cube_path=self._cube_path,
            pattern_indices=idx,
            scan_shape=list(self._scan_shape) if self._scan_shape else None,
            sam_checkpoint=self._vars["sam_checkpoint"].get(),
            sam_model_type=self._vars["sam_model_type"].get(),
            device="cuda",
            preprocess=dict(
                blur_sigma=knobs["blur_sigma"],
                rescale_lo=knobs["rescale_lo"],
                rescale_hi=knobs["rescale_hi"],
                downsample=knobs["downsample"],
            ),
            filter=dict(
                area_range=[knobs["area_min"], knobs["area_max"]],
                aspect_ratio_threshold=knobs["aspect_min"],
                min_length=knobs["min_length"],
                min_distance=knobs["min_dist"],
                max_distance=knobs["max_dist"],
                min_r2=knobs["min_r2"],
                skip_largest=knobs["skip_largest"],
                candidate_slice=[1, knobs["candidate_max"]],
            ),
            amg={},                  # surface in advanced expander later
            save_masks=knobs["save_masks"],
        )

    def _apply_to_class(self):
        if self.job is not None and self.job.is_running():
            messagebox.showinfo("SAM",
                "A SAM run is already in progress. Wait for it to finish "
                "(or stop the GUI to kill it)."); return
        if self._assigns is None:
            self._compute_class_avgs()
            if self._assigns is None:
                return
        c = int(self._vars["current_class"].get())
        n = int(np.sum(self._assigns == c))
        ckpt = self._vars["sam_checkpoint"].get()
        if not ckpt or not os.path.exists(ckpt):
            messagebox.showerror("SAM",
                f"SAM checkpoint not found:\n{ckpt}"); return
        # Performance estimate: ~1 s/pattern on RTX 4080 for vit_h
        per_s = {"vit_h": 1.0, "vit_l": 0.5, "vit_b": 0.2}.get(
            self._vars["sam_model_type"].get(), 1.0)
        eta_min = n * per_s / 60.0
        ok = messagebox.askyesno("Confirm SAM run",
            f"Run SAM on class p{c}: {n} patterns.\n\n"
            f"Estimated time: ~{eta_min:.1f} min "
            f"(model={self._vars['sam_model_type'].get()}, ~{per_s:.1f} s/pattern).\n\n"
            f"Continue?")
        if not ok:
            return
        # Save the class's tuning into per-class config first.
        self._save_class_tuning()
        outdir = os.path.join(self.outdir, "sam", f"p{c}")
        kw = self._build_class_kwargs(c)
        self.job = SamRunJob(outdir, kw)
        self.job.start()
        self._status_lbl.configure(
            text=f"class p{c}: SAM running in background  ({n} patterns).  "
                  f"outputs → {outdir}")
        self._poll()

    def _apply_to_all(self):
        if self._assigns is None:
            self._compute_class_avgs()
            if self._assigns is None:
                return
        K = self._K
        ok = messagebox.askyesno("Confirm SAM run on ALL classes",
            f"Run SAM on every class (K={K}).\n\n"
            f"Each class uses its saved per-class tuning if present, "
            f"else the current sidebar knobs.\n\n"
            f"This is the long one — could be hours.  Continue?")
        if not ok:
            return
        # MVP: serial — queue them and run one at a time as job slots
        # free up.  For now we just kick off the first; the rest can
        # be done by clicking Apply per class after each completes.
        # (A proper queue is a v2 feature.)
        messagebox.showinfo("SAM",
            "MVP: starts class p0 only. After each finishes, click "
            "Apply on the next class.  A proper queue will come in v2.")
        self._vars["current_class"].set("0")
        self._on_class_change()
        self._apply_to_class()

    def _save_class_tuning(self):
        if not self.outdir:
            return
        c = self._vars["current_class"].get()
        self._per_class_cfg[c] = self._gather_current_knobs()
        try:
            self._write_per_class_cfg()
            self._status_lbl.configure(
                text=f"saved tuning for class p{c} to sam_config.json")
        except Exception as e:
            messagebox.showerror("SAM",
                f"could not save tuning:\n{e!r}")

    # ----- subprocess polling ---------------------------------------
    def _poll(self):
        if self.job is None:
            return
        if self.job.is_running():
            prog = self.job.progress()
            elapsed = time.perf_counter() - self.job._t_start
            self._status_lbl.configure(
                text=f"SAM running… {prog}   "
                      f"elapsed {elapsed:.0f}s")
            self._poll_after = self.after(2000, self._poll)
        else:
            st = self.job.status()
            self._status_lbl.configure(
                text=f"SAM job {st}.  outdir = {self.job.outdir}")
            if st == "done":
                self._render_class_angle_map()
            self.job = None  # ready for next run

    # ----- canvas rendering -----------------------------------------
    def _render_canvas_idle(self):
        # Default canvas content when no SAM result yet.
        self._fig.clear()
        if self._class_avgs is None:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Load a run dir, then click 'Reload class avgs',\n"
                    "then 'Run on this average' to test SAM tuning.",
                    ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            self._canvas.draw_idle()
            return
        c = int(self._vars["current_class"].get())
        ax = self._fig.add_subplot(111)
        ax.imshow(self._class_avgs[c], cmap="inferno")
        ax.set_title(f"class p{c} average  (no SAM run yet)")
        ax.set_axis_off()
        self._canvas.draw_idle()

    def _render_canvas(self):
        """Show class avg + SAM masks overlay + angle map (if available)."""
        if self._last_run is None:
            self._render_canvas_idle()
            return
        self._fig.clear()
        c = self._last_run["cls"]
        prep = self._last_run["prep"]
        flt = self._last_run["filtered"]
        all_m = self._last_run["all_masks"]
        # Two side-by-side panels: preprocessed input + masks overlay.
        ax1 = self._fig.add_subplot(1, 2, 1)
        ax1.imshow(self._class_avgs[c], cmap="inferno")
        ax1.set_title(f"p{c} class average  (raw)")
        ax1.set_axis_off()

        ax2 = self._fig.add_subplot(1, 2, 2)
        # Show preprocessed image as backdrop
        ax2.imshow(prep)
        # Overlay every filtered mask in a distinct semi-transparent colour
        cmap = matplotlib.colormaps.get_cmap("tab10")
        for i, m in enumerate(flt):
            seg = m['segmentation']
            # Build an RGBA overlay
            colour = cmap(i % 10)
            mask_rgba = np.zeros((*seg.shape, 4))
            mask_rgba[seg, 0] = colour[0]
            mask_rgba[seg, 1] = colour[1]
            mask_rgba[seg, 2] = colour[2]
            mask_rgba[seg, 3] = 0.55
            ax2.imshow(mask_rgba)
            # Bounding box
            x, y, w, h = m['bbox']
            ax2.add_patch(Rectangle((x, y), w, h, edgecolor=colour,
                                       facecolor="none", lw=1.0))
            # Angle annotation
            mid_x, mid_y = m.get('midpoint', (x + w/2, y + h/2))
            ax2.text(mid_x, mid_y,
                      f"{m.get('angle', 0):.0f}°",
                      color="white", fontsize=8,
                      bbox=dict(facecolor=colour, alpha=0.7,
                                  edgecolor="none", pad=1))
        ax2.set_title(f"p{c} preprocessed + filtered masks  "
                       f"({len(all_m)} raw → {len(flt)} kept)")
        ax2.set_axis_off()
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _render_class_angle_map(self):
        """Replace the canvas with the per-class angle map after a
        whole-class SAM job finishes."""
        if not self.outdir:
            return
        c = int(self._vars["current_class"].get())
        png = os.path.join(self.outdir, "sam", f"p{c}", "angle_map.png")
        if not os.path.exists(png):
            return
        try:
            img = plt.imread(png)
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.imshow(img)
            ax.set_title(f"p{c} angle map  (loaded from {png})")
            ax.set_axis_off()
            self._fig.tight_layout()
            self._canvas.draw_idle()
        except Exception as e:
            print(f"[sam-panel] angle-map render failed: {e!r}", flush=True)
