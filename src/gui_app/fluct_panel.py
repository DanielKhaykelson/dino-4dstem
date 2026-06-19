"""fluct_panel.py -- Fluctuation/variance-map tab.

Computes per-pattern angular variance in polar space within a
user-chosen annulus [r - dr/2, r + dr/2]. This is the FEM
(fluctuation electron microscopy) signal:
    var(I(theta, r)) over theta, averaged over the annulus radii.
Crystalline regions show large angular variance (peaks at specific
azimuths) at the right q; amorphous regions show low variance.

The expensive bit is one polar transform per pattern. We do it once
on link (in a worker thread, with progress) and cache a (N, n_bins)
variance profile in memory. After that, dragging the r/dr sliders
just slices and re-maps -- instantaneous update.

Three panels:
    - Class map (loaded from the run's inference cache)
    - Variance map at the chosen (r, dr)
    - 1D radial plots: per-class mean radial *and* mean variance profile,
      with the [r - dr/2, r + dr/2] band shaded.
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
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Polar pipeline (same as compute_radial_profile.py — keeps bins comparable
# to the model's polar input and to the per-sample .radial.npy sidecars).
# ---------------------------------------------------------------------------
CENTER_CROP = 140
POLAR_SIZE = 192
POLAR_MASK_COLS = 45    # left-edge mask (matches winner config)


def _build_polar_pre():
    import torch
    from torchvision.transforms import v2 as T
    from torchvision.transforms import InterpolationMode
    from dino_sr_ablation import PolarTransform, PolarMaskLeft
    return T.Compose([
        T.CenterCrop(CENTER_CROP),
        T.Resize(POLAR_SIZE,
                  interpolation=InterpolationMode.BILINEAR, antialias=True),
        PolarTransform(output_size=POLAR_SIZE),
        PolarMaskLeft(k_cols=POLAR_MASK_COLS),
    ])


def compute_var_profile(sample_key: str, progress_cb=None,
                          batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pattern (var-over-theta, mean-over-theta) in polar space.

    Returns:
        var_prof  : (N, POLAR_SIZE) float32
        mean_prof : (N, POLAR_SIZE) float32
    """
    import torch
    from data import SAMPLES, LoadPRZ
    cfg = SAMPLES[sample_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LoadPRZ(cfg["path"], resize=POLAR_SIZE, vmax=cfg["vmax"])
    pre = _build_polar_pre()
    N = len(ds)
    var_prof = np.zeros((N, POLAR_SIZE), dtype=np.float32)
    mean_prof = np.zeros((N, POLAR_SIZE), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, batch):
            j = min(i + batch, N)
            xs = torch.stack([ds[k] for k in range(i, j)]).to(device).float()
            xs = pre(xs)                              # (b, 1, theta=H, r=W)
            # Variance and mean over theta (axis=-2), squeeze channel.
            v = xs.var(dim=-2, unbiased=False).squeeze(1)
            m = xs.mean(dim=-2).squeeze(1)
            var_prof[i:j] = v.cpu().numpy()
            mean_prof[i:j] = m.cpu().numpy()
            if progress_cb is not None and (i // batch) % 4 == 0:
                progress_cb(i + (j - i), N)
    if progress_cb is not None:
        progress_cb(N, N)
    return var_prof, mean_prof


# ---------------------------------------------------------------------------
# UI helpers (consistent with sam_panel / blob_panel)
# ---------------------------------------------------------------------------
def _section(parent, title):
    f = ctk.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x", padx=2, pady=(8, 0))
    ctk.CTkLabel(f, text=title, font=("Segoe UI", 11, "bold")).pack(anchor="w")
    sep = ctk.CTkFrame(parent, height=2, fg_color=("#cccccc", "#444444"))
    sep.pack(fill="x", padx=2, pady=(0, 4))
    return parent


# ---------------------------------------------------------------------------
class FluctPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.outdir = None
        self.sample = None
        self._scan_shape = None
        self._assigns = None              # per-pattern argmax (from posthoc)
        self._K = 0
        self._var_prof = None             # (N, POLAR_SIZE)
        self._mean_prof = None            # (N, POLAR_SIZE)
        self._compute_thread = None
        self._compute_running = False
        self._compute_progress = ""
        self._lock = threading.Lock()
        self._build()

    # ----- run linkage -------------------------------------------------
    def link_run(self, outdir, sample):
        self.outdir = outdir
        self.sample = sample
        self._info_lbl.configure(
            text=f"linked: {os.path.basename(outdir)}  (sample={sample})")
        self._var_prof = None; self._mean_prof = None
        self._assigns = None; self._K = 0
        self._render_idle()

    def on_runtime_sample_added(self, key):
        pass

    # ----- UI ----------------------------------------------------------
    def _build(self):
        # vars
        self._vars = {
            "r":  ctk.IntVar(value=POLAR_SIZE // 4),
            "dr": ctk.IntVar(value=10),
            "use_log": ctk.BooleanVar(value=True),
        }

        # top bar
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)
        ctk.CTkButton(top, text="Load run dir…", width=140,
                       command=self._load_dir_dialog).pack(side="left", padx=4)
        self._info_lbl = ctk.CTkLabel(top, text="(no run linked)",
                                        font=("Consolas", 10))
        self._info_lbl.pack(side="left", padx=8)

        right_box = ctk.CTkFrame(top, fg_color="transparent")
        right_box.pack(side="right", padx=4)
        self._compute_btn = ctk.CTkButton(right_box,
            text="Compute fluct-profile",
            width=180, command=self._kickoff_compute)
        self._compute_btn.pack(side="right", padx=4)

        # body: sidebar + canvas
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        sidebar = ctk.CTkScrollableFrame(body, width=300)
        sidebar.pack(side="left", fill="y")

        _section(sidebar, "Annulus")
        # r slider
        r_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        r_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(r_row, text="r  (radial bin):", width=120, anchor="w"
                       ).pack(side="left")
        self._r_label = ctk.CTkLabel(r_row,
            text=str(self._vars["r"].get()), width=40, anchor="e")
        self._r_label.pack(side="right")
        r_slider = ctk.CTkSlider(sidebar, from_=0, to=POLAR_SIZE - 1,
                                   number_of_steps=POLAR_SIZE - 1,
                                   variable=self._vars["r"],
                                   command=self._on_slider_r)
        r_slider.pack(fill="x", padx=8)
        self._r_slider = r_slider

        # dr slider
        dr_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        dr_row.pack(fill="x", padx=4, pady=(8, 2))
        ctk.CTkLabel(dr_row, text="dr  (annulus thickness):", width=180,
                       anchor="w").pack(side="left")
        self._dr_label = ctk.CTkLabel(dr_row,
            text=str(self._vars["dr"].get()), width=40, anchor="e")
        self._dr_label.pack(side="right")
        dr_max = POLAR_SIZE // 2
        dr_slider = ctk.CTkSlider(sidebar, from_=1, to=dr_max,
                                    number_of_steps=dr_max - 1,
                                    variable=self._vars["dr"],
                                    command=self._on_slider_dr)
        dr_slider.pack(fill="x", padx=8)
        self._dr_slider = dr_slider

        # Direct entries (for typed precision)
        e_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        e_row.pack(fill="x", padx=4, pady=(8, 2))
        ctk.CTkLabel(e_row, text="r =", width=30, anchor="e").pack(side="left")
        ctk.CTkEntry(e_row, textvariable=self._vars["r"], width=60).pack(
            side="left", padx=4)
        ctk.CTkLabel(e_row, text="dr =", width=40, anchor="e").pack(
            side="left", padx=(8, 0))
        ctk.CTkEntry(e_row, textvariable=self._vars["dr"], width=60).pack(
            side="left", padx=4)
        ctk.CTkButton(e_row, text="apply", width=60,
                       command=self._refresh).pack(side="left", padx=4)

        _section(sidebar, "Display")
        ctk.CTkCheckBox(sidebar, text="log-stretch variance map",
                          variable=self._vars["use_log"],
                          command=self._refresh
                          ).pack(anchor="w", padx=8, pady=2)
        ctk.CTkButton(sidebar, text="Save current snapshot",
                       command=self._save_snapshot
                       ).pack(fill="x", padx=8, pady=(8, 4))

        self._status_lbl = ctk.CTkLabel(sidebar,
            text="Load a run, then click 'Compute fluct-profile'.\n"
                  "After it finishes, drag r/dr sliders to inspect.",
            font=("Consolas", 9), justify="left",
            text_color=("#444", "#aaa"), wraplength=270)
        self._status_lbl.pack(anchor="w", padx=8, pady=(8, 4))

        # canvas
        canv = ctk.CTkFrame(body)
        canv.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._fig = Figure(figsize=(13, 5))
        self._canvas = FigureCanvasTkAgg(self._fig, master=canv)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, canv)

        self._render_idle()

    # ----- top-bar callbacks -------------------------------------------
    def _load_dir_dialog(self):
        p = filedialog.askdirectory(title="Pick a run dir")
        if not p:
            return
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
                        print(f"[fluct-panel] register: {e!r}", flush=True)
        except Exception as e:
            print(f"[fluct-panel] _train_kwargs load: {e!r}", flush=True)
        self.link_run(p, sample)

    # ----- compute trigger ---------------------------------------------
    def _kickoff_compute(self):
        if not self.outdir or not self.sample:
            messagebox.showinfo("Fluct-map", "Load a run dir first."); return
        if self._compute_running:
            messagebox.showinfo("Fluct-map",
                "Compute already in progress."); return
        # First grab class assignments via the post-hoc panel (cheap if
        # cached, expensive if not — the post-hoc panel handles its own
        # progress if it has to run inference).
        self._status_lbl.configure(text="getting class assignments …")
        self.update_idletasks()
        try:
            ph = getattr(self.app, "posthoc", None) if self.app else None
            if ph is not None and getattr(ph, "outdir", None) != self.outdir:
                ph.link_run(self.outdir, self.sample)
            if ph is None or not hasattr(ph, "_ensure_inference"):
                messagebox.showinfo("Fluct-map",
                    "Need PostHocPanel to compute class assignments.")
                return
            if not ph._ensure_inference():
                self._status_lbl.configure(
                    text="inference failed — see Post-hoc tab")
                return
            self._assigns = ph._inf["assigns"]
            self._K = int(ph._inf["soft_probs"].shape[1])
            self._scan_shape = ph._scan_shape
        except Exception as e:
            messagebox.showerror("Fluct-map",
                f"class-assigns lookup failed:\n{e!r}")
            return

        self._compute_running = True
        self._compute_btn.configure(state="disabled")
        self._compute_thread = threading.Thread(
            target=self._compute_worker, daemon=True)
        self._compute_thread.start()
        self._poll()

    def _compute_worker(self):
        try:
            def cb(done, total):
                with self._lock:
                    self._compute_progress = (
                        f"polar transform … {done}/{total} "
                        f"({100*done/max(total,1):.0f}%)")
            t0 = time.perf_counter()
            var_prof, mean_prof = compute_var_profile(
                self.sample, progress_cb=cb)
            self._var_prof = var_prof
            self._mean_prof = mean_prof
            with self._lock:
                self._compute_progress = (
                    f"done. ({time.perf_counter() - t0:.1f}s)  "
                    f"shape={var_prof.shape}")
        except Exception as e:
            with self._lock:
                self._compute_progress = f"failed: {e!r}"
            print(f"[fluct-panel] worker failed: {e!r}", flush=True)
        finally:
            self._compute_running = False

    def _poll(self):
        with self._lock:
            prog = self._compute_progress
        self._status_lbl.configure(text=prog or "(running…)")
        if self._compute_running:
            self.after(500, self._poll)
        else:
            self._compute_btn.configure(state="normal")
            if self._var_prof is not None:
                self._refresh()

    # ----- slider callbacks --------------------------------------------
    def _on_slider_r(self, val):
        try:
            self._vars["r"].set(int(round(float(val))))
        except Exception:
            return
        self._r_label.configure(text=str(self._vars["r"].get()))
        self._refresh()

    def _on_slider_dr(self, val):
        try:
            self._vars["dr"].set(int(round(float(val))))
        except Exception:
            return
        self._dr_label.configure(text=str(self._vars["dr"].get()))
        self._refresh()

    # ----- core: build the fluct map and re-render ---------------------
    def _annulus_window(self) -> tuple[int, int]:
        try:
            r = int(self._vars["r"].get())
            dr = max(1, int(self._vars["dr"].get()))
        except Exception:
            return 0, 1
        lo = max(0, r - dr // 2)
        hi = min(POLAR_SIZE, r + dr - dr // 2)
        if hi <= lo:
            hi = min(POLAR_SIZE, lo + 1)
        return lo, hi

    def _refresh(self):
        if self._var_prof is None:
            self._render_idle(); return
        if self._scan_shape is None or self._assigns is None:
            self._render_idle(); return
        lo, hi = self._annulus_window()
        # Mean variance over the annulus radii, per pattern.
        fluct = self._var_prof[:, lo:hi].mean(axis=1)        # (N,)
        Ny, Nx = self._scan_shape
        # Sometimes scan_shape*1 != N (subsampled training); in that case
        # we still try to display the scan grid we have.
        if fluct.size != Ny * Nx:
            # fall back to a linear strip
            print(f"[fluct-panel] N={fluct.size} != Ny*Nx={Ny*Nx}; "
                  f"showing as 1×N strip", flush=True)
            grid = fluct.reshape(1, -1)
        else:
            grid = fluct.reshape(Ny, Nx)

        # ----- plots -----
        self._fig.clear()
        K = self._K
        cmap = plt.get_cmap("tab10")
        palette = ListedColormap([cmap(i) for i in range(K)])

        # Class map (left)
        ax1 = self._fig.add_subplot(1, 3, 1)
        if self._assigns.size == Ny * Nx:
            cls_grid = self._assigns.reshape(Ny, Nx)
        else:
            cls_grid = self._assigns.reshape(1, -1)
        im1 = ax1.imshow(cls_grid, cmap=palette, vmin=-0.5, vmax=K - 0.5,
                          interpolation="nearest")
        ax1.set_title(f"class map  (K={K})")
        ax1.set_axis_off()

        # Variance map (middle)
        ax2 = self._fig.add_subplot(1, 3, 2)
        if self._vars["use_log"].get():
            disp = np.log1p(grid - grid.min())
            cb_label = "log(1 + var − min)"
        else:
            disp = grid
            cb_label = "var(I) over θ, avg over annulus"
        im2 = ax2.imshow(disp, cmap="magma", interpolation="nearest")
        ax2.set_title(f"fluct map  (r={(lo+hi)//2}, dr={hi-lo})")
        ax2.set_axis_off()
        self._fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04,
                            label=cb_label)
        # Real-space scale bar on the scan-grid map.
        try:
            nm_per_px = float(self.app.real_res.get()) if self.app else 0.0
        except Exception:
            nm_per_px = 0.0
        if nm_per_px > 0:
            from gui_app._calib_utils import add_real_scalebar
            add_real_scalebar(ax2, nm_per_px, length_nm=100, color="white")

        # 1D radial plots (right)
        ax3 = self._fig.add_subplot(1, 3, 3)
        # Per-class mean of mean_prof and var_prof -> 2 traces per class
        # would clutter; show var_prof (the primary signal) per class +
        # the global mean_prof for context (gray, dashed).
        bins = np.arange(POLAR_SIZE)
        # Global mean profile (gray dashed) — context for where peaks live.
        global_mean = self._mean_prof.mean(axis=0)
        ax3b = ax3.twinx()
        ax3b.plot(bins, global_mean, ls="--", color="0.6", lw=1.0,
                   label="⟨I⟩ over all patterns (right axis)")
        ax3b.set_ylabel("mean intensity (a.u.)", color="0.4")
        ax3b.tick_params(axis="y", labelcolor="0.4")

        # Per-class mean of var_prof.
        for c in range(K):
            mask = (self._assigns == c)
            if mask.sum() == 0:
                continue
            v = self._var_prof[mask].mean(axis=0)
            ax3.plot(bins, v, color=cmap(c), lw=1.4,
                      label=f"p{c} (n={int(mask.sum())})")
        ax3.axvspan(lo, hi, color="orange", alpha=0.25,
                     label=f"annulus [{lo}, {hi})")
        ax3.set_xlabel("radial bin (q)")
        ax3.set_ylabel("⟨var(I) over θ⟩  per class")
        ax3.set_title("per-class angular variance vs q")
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, alpha=0.3)
        # Re-tick the x-axis to nm⁻¹ when reciprocal calibration is set.
        try:
            rp = float(self.app.recip_res.get()) if self.app else 0.0
        except Exception:
            rp = 0.0
        if rp > 0:
            from gui_app._calib_utils import (q_per_polar_bin,
                                                 get_raw_detector_size,
                                                 set_q_axis)
            qpb = q_per_polar_bin(rp,
                                    get_raw_detector_size(self.sample))
            set_q_axis(ax3, POLAR_SIZE, qpb, axis="x")

        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ----- idle rendering ----------------------------------------------
    def _render_idle(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        if self._var_prof is None:
            ax.text(0.5, 0.5,
                    "1. Load a run dir.\n"
                    "2. Click 'Compute fluct-profile' (~30s).\n"
                    "3. Drag r / dr sliders to inspect crystallinity.",
                    ha="center", va="center", fontsize=11)
        else:
            ax.text(0.5, 0.5,
                    "Profile cached. Drag the sliders or hit 'apply'.",
                    ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        self._canvas.draw_idle()

    def _save_snapshot(self):
        if self._var_prof is None or self.outdir is None:
            messagebox.showinfo("Fluct-map", "Nothing to save yet."); return
        lo, hi = self._annulus_window()
        out_dir = os.path.join(self.outdir, "fluct")
        os.makedirs(out_dir, exist_ok=True)
        # Save the current (r, dr) variance map as PNG + numpy.
        tag = f"r{(lo+hi)//2}_dr{hi-lo}"
        png = os.path.join(out_dir, f"fluct_map_{tag}.png")
        try:
            self._fig.savefig(png, dpi=140)
        except Exception as e:
            messagebox.showerror("Fluct-map", f"save PNG failed:\n{e!r}"); return
        try:
            np.save(os.path.join(out_dir, f"fluct_map_{tag}.npy"),
                    self._var_prof[:, lo:hi].mean(axis=1).astype(np.float32))
            np.save(os.path.join(out_dir, "var_profile.npy"),
                    self._var_prof.astype(np.float32))
            np.save(os.path.join(out_dir, "mean_profile.npy"),
                    self._mean_prof.astype(np.float32))
        except Exception as e:
            messagebox.showerror("Fluct-map", f"save NPY failed:\n{e!r}"); return
        self._status_lbl.configure(
            text=f"saved snapshot to {out_dir} (tag={tag})")
