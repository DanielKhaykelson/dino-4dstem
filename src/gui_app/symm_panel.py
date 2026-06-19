"""symm_panel.py -- Rotational-symmetry map tab.

For each pattern, compute |rFFT_theta(I(theta, r))| — the angular Fourier
spectrum at each radial bin. The k-th component's amplitude measures
how strongly the pattern has k-fold rotational structure at that radius.

In a chosen annulus [r - dr/2, r + dr/2], we average the per-radius
amplitude spectrum to get one number per harmonic per pattern. Two
display modes:

    "argmax k"   per-pattern dominant non-DC harmonic. Coloring by argmax
                 gives a "this pattern looks most n-fold" map.
    "n-fold strength"  amplitude at user-chosen n, divided by the sum
                       over k>=1. Bright = strong n-fold; dim = weak.

The polar transform pipeline matches Fluct-map / compute_radial_profile.py
exactly so radial bins line up across tabs.
"""
from __future__ import annotations
import os, json, time, threading

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


CENTER_CROP = 140
POLAR_SIZE = 192
POLAR_MASK_COLS = 45
K_MAX = 16          # cap angular harmonics we cache (k=0..K_MAX)
                     # (n=2..12 covers physical symmetries; cap at 16 for slack)


def _build_polar_pre():
    import torch  # noqa
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


def compute_angular_fft_amp(sample_key: str,
                              progress_cb=None,
                              batch: int = 128,
                              k_max: int = K_MAX) -> np.ndarray:
    """Compute |rFFT_theta(I)| per pattern, capped at k_max harmonics.

    Returns:
        amp : (N, k_max+1, POLAR_SIZE) float32
              amp[i, k, r] = magnitude of k-th angular Fourier component of
              pattern i at radial bin r.
    """
    import torch
    from data import SAMPLES, LoadPRZ
    cfg = SAMPLES[sample_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LoadPRZ(cfg["path"], resize=POLAR_SIZE, vmax=cfg["vmax"])
    pre = _build_polar_pre()
    N = len(ds)
    nk = k_max + 1
    amp = np.zeros((N, nk, POLAR_SIZE), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, batch):
            j = min(i + batch, N)
            xs = torch.stack([ds[k] for k in range(i, j)]).to(device).float()
            xs = pre(xs)                              # (b, 1, theta, r)
            xs = xs.squeeze(1)                        # (b, theta, r)
            f = torch.fft.rfft(xs, dim=1)             # (b, theta//2+1, r)
            mag = f.abs().float()[:, :nk, :]
            amp[i:j] = mag.cpu().numpy()
            if progress_cb is not None and (i // batch) % 4 == 0:
                progress_cb(j, N)
    if progress_cb is not None:
        progress_cb(N, N)
    return amp


def _section(parent, title):
    f = ctk.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x", padx=2, pady=(8, 0))
    ctk.CTkLabel(f, text=title, font=("Segoe UI", 11, "bold")).pack(anchor="w")
    sep = ctk.CTkFrame(parent, height=2, fg_color=("#cccccc", "#444444"))
    sep.pack(fill="x", padx=2, pady=(0, 4))
    return parent


# ---------------------------------------------------------------------------
class SymmPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.outdir = None
        self.sample = None
        self._scan_shape = None
        self._assigns = None
        self._K = 0
        self._amp = None                 # (N, K_MAX+1, POLAR_SIZE) float32
        self._compute_thread = None
        self._compute_running = False
        self._compute_progress = ""
        self._lock = threading.Lock()
        self._build()

    # ----- run linkage --------------------------------------------------
    def link_run(self, outdir, sample):
        self.outdir = outdir
        self.sample = sample
        self._info_lbl.configure(
            text=f"linked: {os.path.basename(outdir)}  (sample={sample})")
        self._amp = None
        self._assigns = None; self._K = 0
        self._render_idle()

    def on_runtime_sample_added(self, key):
        pass

    # ----- UI ----------------------------------------------------------
    def _build(self):
        self._vars = {
            "r":  ctk.IntVar(value=POLAR_SIZE // 4),
            "dr": ctk.IntVar(value=10),
            "mode": ctk.StringVar(value="argmax k"),
            "n_fold": ctk.IntVar(value=6),
            "k_min": ctk.IntVar(value=2),
            "k_max": ctk.IntVar(value=K_MAX),
            "use_log": ctk.BooleanVar(value=False),
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
            text="Compute symmetry-FFT",
            width=180, command=self._kickoff_compute)
        self._compute_btn.pack(side="right", padx=4)

        # body
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
        ctk.CTkSlider(sidebar, from_=0, to=POLAR_SIZE - 1,
                       number_of_steps=POLAR_SIZE - 1,
                       variable=self._vars["r"],
                       command=self._on_slider_r).pack(fill="x", padx=8)

        # dr slider
        dr_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        dr_row.pack(fill="x", padx=4, pady=(8, 2))
        ctk.CTkLabel(dr_row, text="dr  (annulus thickness):", width=180,
                       anchor="w").pack(side="left")
        self._dr_label = ctk.CTkLabel(dr_row,
            text=str(self._vars["dr"].get()), width=40, anchor="e")
        self._dr_label.pack(side="right")
        ctk.CTkSlider(sidebar, from_=1, to=POLAR_SIZE // 2,
                       number_of_steps=POLAR_SIZE // 2 - 1,
                       variable=self._vars["dr"],
                       command=self._on_slider_dr).pack(fill="x", padx=8)

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

        _section(sidebar, "Map mode")
        mode_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        mode_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkOptionMenu(mode_row, variable=self._vars["mode"],
                            values=["argmax k", "n-fold strength"],
                            width=180,
                            command=lambda _v: self._refresh()
                            ).pack(side="left", padx=2)

        n_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        n_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(n_row, text="n  (for n-fold strength):", width=180,
                       anchor="w").pack(side="left")
        ctk.CTkEntry(n_row, textvariable=self._vars["n_fold"], width=60
                       ).pack(side="left", padx=4)
        ctk.CTkButton(n_row, text="apply", width=60,
                       command=self._refresh).pack(side="left", padx=4)

        k_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        k_row.pack(fill="x", padx=4, pady=(8, 2))
        ctk.CTkLabel(k_row, text="argmax k range:", width=140, anchor="w"
                       ).pack(side="left")
        ctk.CTkEntry(k_row, textvariable=self._vars["k_min"], width=50
                       ).pack(side="left", padx=2)
        ctk.CTkLabel(k_row, text="..", width=20).pack(side="left")
        ctk.CTkEntry(k_row, textvariable=self._vars["k_max"], width=50
                       ).pack(side="left", padx=2)
        ctk.CTkButton(k_row, text="apply", width=60,
                       command=self._refresh).pack(side="left", padx=4)

        _section(sidebar, "Display")
        ctk.CTkCheckBox(sidebar, text="log-stretch strength map",
                          variable=self._vars["use_log"],
                          command=self._refresh
                          ).pack(anchor="w", padx=8, pady=2)
        ctk.CTkButton(sidebar, text="Save current snapshot",
                       command=self._save_snapshot
                       ).pack(fill="x", padx=8, pady=(8, 4))

        self._status_lbl = ctk.CTkLabel(sidebar,
            text="Load a run, then click 'Compute symmetry-FFT'.\n"
                  "Then drag r/dr and pick a mode.",
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

    # ----- callbacks ---------------------------------------------------
    def _on_slider_r(self, val):
        try: self._vars["r"].set(int(round(float(val))))
        except Exception: return
        self._r_label.configure(text=str(self._vars["r"].get()))
        self._refresh()

    def _on_slider_dr(self, val):
        try: self._vars["dr"].set(int(round(float(val))))
        except Exception: return
        self._dr_label.configure(text=str(self._vars["dr"].get()))
        self._refresh()

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
                        print(f"[symm-panel] register: {e!r}", flush=True)
        except Exception as e:
            print(f"[symm-panel] _train_kwargs load: {e!r}", flush=True)
        self.link_run(p, sample)

    # ----- compute trigger ---------------------------------------------
    def _kickoff_compute(self):
        if not self.outdir or not self.sample:
            messagebox.showinfo("Symm-map", "Load a run dir first."); return
        if self._compute_running:
            messagebox.showinfo("Symm-map",
                "Compute already in progress."); return
        self._status_lbl.configure(text="getting class assignments …")
        self.update_idletasks()
        try:
            ph = getattr(self.app, "posthoc", None) if self.app else None
            if ph is not None and getattr(ph, "outdir", None) != self.outdir:
                ph.link_run(self.outdir, self.sample)
            if ph is None or not hasattr(ph, "_ensure_inference"):
                messagebox.showinfo("Symm-map",
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
            messagebox.showerror("Symm-map",
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
                        f"polar+rFFT … {done}/{total} "
                        f"({100*done/max(total,1):.0f}%)")
            t0 = time.perf_counter()
            self._amp = compute_angular_fft_amp(self.sample, progress_cb=cb)
            with self._lock:
                self._compute_progress = (
                    f"done. ({time.perf_counter() - t0:.1f}s)  "
                    f"shape={self._amp.shape}")
        except Exception as e:
            with self._lock:
                self._compute_progress = f"failed: {e!r}"
            print(f"[symm-panel] worker failed: {e!r}", flush=True)
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
            if self._amp is not None:
                self._refresh()

    # ----- map computation ---------------------------------------------
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
        if self._amp is None:
            self._render_idle(); return
        if self._scan_shape is None or self._assigns is None:
            self._render_idle(); return
        lo, hi = self._annulus_window()
        amp_k = self._amp[:, :, lo:hi].mean(axis=2)            # (N, nk)
        nk = amp_k.shape[1]

        try:
            k_min = max(1, int(self._vars["k_min"].get()))
        except Exception:
            k_min = 2
        try:
            k_hi = min(nk - 1, int(self._vars["k_max"].get()))
        except Exception:
            k_hi = nk - 1
        if k_hi < k_min:
            k_hi = k_min

        mode = self._vars["mode"].get()
        if mode == "argmax k":
            slab = amp_k[:, k_min:k_hi + 1]
            argk = slab.argmax(axis=1) + k_min                  # (N,)
            map_data = argk.astype(np.int32)
            cmap_use = "tab20"
            v_lo, v_hi = k_min - 0.5, k_hi + 0.5
            cb_label = f"argmax k  (k={k_min}…{k_hi})"
            symm_int = True
        else:
            try:
                n = max(1, int(self._vars["n_fold"].get()))
            except Exception:
                n = 6
            n = min(n, nk - 1)
            denom = amp_k[:, k_min:].sum(axis=1).clip(min=1e-12)
            strength = amp_k[:, n] / denom                      # (N,)
            if self._vars["use_log"].get():
                strength = np.log1p(strength)
            map_data = strength.astype(np.float32)
            cmap_use = "magma"
            v_lo, v_hi = float(map_data.min()), float(map_data.max())
            cb_label = (f"|F_k=n| / Σ_{{k≥{k_min}}}|F_k|, "
                          f"n={n}"
                          + ("  (log)" if self._vars["use_log"].get() else ""))
            symm_int = False

        Ny, Nx = self._scan_shape
        if map_data.size != Ny * Nx:
            grid = map_data.reshape(1, -1)
        else:
            grid = map_data.reshape(Ny, Nx)

        # ---- plot ----
        self._fig.clear()
        K = self._K
        cmap = plt.get_cmap("tab10")
        palette = ListedColormap([cmap(i) for i in range(K)])

        # Class map
        ax1 = self._fig.add_subplot(1, 3, 1)
        if self._assigns.size == Ny * Nx:
            cls_grid = self._assigns.reshape(Ny, Nx)
        else:
            cls_grid = self._assigns.reshape(1, -1)
        ax1.imshow(cls_grid, cmap=palette, vmin=-0.5, vmax=K - 0.5,
                    interpolation="nearest")
        ax1.set_title(f"class map  (K={K})")
        ax1.set_axis_off()

        # Symmetry map
        ax2 = self._fig.add_subplot(1, 3, 2)
        im2 = ax2.imshow(grid, cmap=cmap_use, vmin=v_lo, vmax=v_hi,
                          interpolation="nearest")
        ax2.set_title(f"symmetry map  ({mode})\n"
                       f"r={(lo+hi)//2}, dr={hi-lo}")
        ax2.set_axis_off()
        try:
            nm_per_px = float(self.app.real_res.get()) if self.app else 0.0
        except Exception:
            nm_per_px = 0.0
        if nm_per_px > 0:
            from gui_app._calib_utils import add_real_scalebar
            add_real_scalebar(ax2, nm_per_px, length_nm=100, color="white")
        if symm_int:
            ks = np.arange(k_min, k_hi + 1)
            cbar = self._fig.colorbar(im2, ax=ax2, ticks=ks,
                                         fraction=0.046, pad=0.04)
            cbar.set_label(cb_label)
        else:
            self._fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04,
                                label=cb_label)

        # 1D power spectra per class
        ax3 = self._fig.add_subplot(1, 3, 3)
        ks_full = np.arange(nk)
        for c in range(K):
            mask = (self._assigns == c)
            if mask.sum() == 0:
                continue
            spec = amp_k[mask].mean(axis=0)
            ax3.plot(ks_full[1:], spec[1:], color=cmap(c), lw=1.4,
                      label=f"p{c} (n={int(mask.sum())})")
        if mode == "n-fold strength":
            try:
                n_show = int(self._vars["n_fold"].get())
                if 1 <= n_show < nk:
                    ax3.axvline(n_show, color="orange", ls="--", lw=1.0,
                                  label=f"n={n_show}")
            except Exception:
                pass
        ax3.axvspan(k_min - 0.4, k_hi + 0.4, color="orange", alpha=0.10,
                     label=f"argmax range [{k_min},{k_hi}]")
        ax3.set_xlabel("angular Fourier component k")
        ax3.set_ylabel("⟨|F_k|⟩  per class")
        ax3.set_title(f"per-class angular FFT spectrum\n"
                       f"(annulus r={(lo+hi)//2}, dr={hi-lo})")
        ax3.set_xticks(np.arange(0, nk, max(1, nk // 8)))
        ax3.legend(loc="upper right", fontsize=7)
        ax3.grid(True, alpha=0.3)

        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ----- idle / save -------------------------------------------------
    def _render_idle(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        if self._amp is None:
            ax.text(0.5, 0.5,
                    "1. Load a run dir.\n"
                    "2. Click 'Compute symmetry-FFT' (~30s).\n"
                    "3. Drag r/dr; pick mode 'argmax k' or 'n-fold strength'.",
                    ha="center", va="center", fontsize=11)
        else:
            ax.text(0.5, 0.5,
                    "Profile cached. Drag the sliders or apply.",
                    ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        self._canvas.draw_idle()

    def _save_snapshot(self):
        if self._amp is None or self.outdir is None:
            messagebox.showinfo("Symm-map", "Nothing to save yet."); return
        lo, hi = self._annulus_window()
        out_dir = os.path.join(self.outdir, "symm")
        os.makedirs(out_dir, exist_ok=True)
        mode = self._vars["mode"].get().replace(" ", "_")
        n_fold = int(self._vars["n_fold"].get())
        tag = f"{mode}_r{(lo+hi)//2}_dr{hi-lo}"
        if mode != "argmax_k":
            tag += f"_n{n_fold}"
        png = os.path.join(out_dir, f"symm_{tag}.png")
        try:
            self._fig.savefig(png, dpi=140)
        except Exception as e:
            messagebox.showerror("Symm-map", f"save PNG failed:\n{e!r}"); return
        try:
            np.save(os.path.join(out_dir, "angular_fft_amp.npy"),
                    self._amp.astype(np.float32))
        except Exception as e:
            messagebox.showerror("Symm-map", f"save NPY failed:\n{e!r}"); return
        self._status_lbl.configure(
            text=f"saved snapshot to {out_dir} (tag={tag})")
