"""ordering_panel.py — Order-parameter diagnostic.

Tests whether the K classes lie on a 1D ordering axis (vs. being
discrete phases) by computing five orthogonal "ordering proxies"
per class and asking whether they all agree on the same class
ranking via Spearman rank correlation.

Metrics (per class):
    angvar     mean angular variance var_θ(I) in user-chosen q-window.
    PBC        peak-to-background contrast: peak(mean profile in window)
               / median(mean profile outside the window).
    FWHM_neg   negative full-width-half-max of the radial peak in the
               window  (sign-flipped so higher=more ordered).
    grain      mean spatial grain size on the scan grid (4-connectivity
               connected components of same-class pixels).
    embPC1     class-centroid projection on PC1 of the centroid covariance
               in embedding space  (the model's intrinsic axis of
               variation across classes).

Key plot: |Spearman ρ| heatmap.  All-pairs |ρ|>~0.85 → independent
metrics agree on a single 1D coordinate → 1D ordering. Lower / mixed
correlations → discrete phases.
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

POLAR_SIZE = 192
METRIC_NAMES = ["angvar", "PBC", "FWHM_neg", "grain", "embPC1"]


# ---------------------------------------------------------------------------
def _section(parent, title):
    f = ctk.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x", padx=2, pady=(8, 0))
    ctk.CTkLabel(f, text=title, font=("Segoe UI", 11, "bold")
                  ).pack(anchor="w")
    sep = ctk.CTkFrame(parent, height=2, fg_color=("#cccccc", "#444444"))
    sep.pack(fill="x", padx=2, pady=(0, 4))
    return parent


# ---------------------------------------------------------------------------
def compute_metrics(sample_key: str, assigns: np.ndarray,
                     embeds: np.ndarray, scan_shape: tuple,
                     r_lo: int, r_hi: int,
                     progress_cb=None) -> dict:
    """Compute the five per-class ordering metrics. Returns dict of arrays
    (length K) plus auxiliary cls_mean / cls_var / counts."""
    from gui_app.fluct_panel import compute_var_profile
    from scipy.ndimage import label

    var_prof, mean_prof = compute_var_profile(
        sample_key, progress_cb=progress_cb)
    K = int(assigns.max()) + 1
    n_bins = var_prof.shape[1]

    cls_mean = np.zeros((K, n_bins), dtype=np.float32)
    cls_var  = np.zeros((K, n_bins), dtype=np.float32)
    counts   = np.zeros(K, dtype=np.int64)
    for k in range(K):
        m = (assigns == k)
        if m.sum() == 0:
            continue
        cls_mean[k] = mean_prof[m].mean(axis=0)
        cls_var[k]  = var_prof[m].mean(axis=0)
        counts[k]   = int(m.sum())

    # 1) angular variance in q-window, per class
    angvar = cls_var[:, r_lo:r_hi].mean(axis=1)

    # 2) PBC: peak in window / median outside
    PBC = np.zeros(K, dtype=np.float32)
    for k in range(K):
        prof = cls_mean[k]
        peak = prof[r_lo:r_hi].max() if r_hi > r_lo else 0.0
        # Background = median in a sensible side region (skip beam halo at
        # very low q and any masked left edge).
        side_lo = max(45, r_lo - 30)   # 45 = polar-mask cols (left edge)
        side_hi = min(n_bins, r_hi + 30)
        bg_left  = prof[side_lo:r_lo]
        bg_right = prof[r_hi:side_hi]
        bg = np.concatenate([bg_left, bg_right])
        bg_val = float(np.median(bg)) if bg.size else 1e-12
        PBC[k] = peak / max(bg_val, 1e-12)

    # 3) FWHM of the in-window peak  (negate so higher = more ordered)
    FWHM = np.zeros(K, dtype=np.float32)
    for k in range(K):
        prof = cls_mean[k]
        if r_hi <= r_lo:
            FWHM[k] = 0.0; continue
        seg = prof[r_lo:r_hi]
        if seg.size == 0 or seg.max() <= 0:
            FWHM[k] = float(r_hi - r_lo); continue
        peak_q = r_lo + int(np.argmax(seg))
        half = prof[peak_q] / 2.0
        # Walk left/right while prof above half, allowing extension
        # outside the window if the peak is broad.
        left = peak_q
        while left > 0 and prof[left] > half:
            left -= 1
        right = peak_q
        while right < n_bins - 1 and prof[right] > half:
            right += 1
        FWHM[k] = float(right - left)
    FWHM_neg = -FWHM   # convention: higher = more ordered

    # 4) Spatial grain size per class (mean CC area, 4-connectivity)
    Ny, Nx = scan_shape
    grid = assigns.reshape(Ny, Nx)
    grain = np.zeros(K, dtype=np.float32)
    for k in range(K):
        m = (grid == k)
        if m.sum() == 0:
            continue
        lab, n = label(m)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())[1:]   # skip bg=0
        grain[k] = float(sizes.mean()) if sizes.size else 0.0

    # 5) Embedding centroid projection on PC1
    if embeds is not None and embeds.size:
        cen = np.zeros((K, embeds.shape[1]), dtype=np.float32)
        for k in range(K):
            m = (assigns == k)
            if m.sum() > 0:
                cen[k] = embeds[m].mean(axis=0)
        cen_c = cen - cen.mean(axis=0, keepdims=True)
        # SVD: (K, D)= U (K, D') S (D',) Vt (D', D)
        U, S, Vt = np.linalg.svd(cen_c, full_matrices=False)
        embPC1 = U[:, 0] * S[0]
        # Variance fraction explained by PC1 across class centroids
        ev = (S ** 2)
        pc1_R2 = float(ev[0] / (ev.sum() + 1e-12))
    else:
        embPC1 = np.zeros(K, dtype=np.float32)
        pc1_R2 = 0.0

    return dict(
        angvar=angvar, PBC=PBC, FWHM_neg=FWHM_neg, grain=grain,
        embPC1=embPC1.astype(np.float32),
        counts=counts, cls_mean=cls_mean, cls_var=cls_var,
        pc1_R2=pc1_R2,
    )


def spearman_matrix(metrics: dict) -> np.ndarray:
    """Return |ρ| matrix (5×5) over METRIC_NAMES."""
    from scipy.stats import spearmanr
    arr = np.stack([metrics[n] for n in METRIC_NAMES], axis=0)
    M = len(METRIC_NAMES)
    R = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            r, _ = spearmanr(arr[i], arr[j])
            if np.isnan(r):
                r = 0.0
            R[i, j] = abs(r)
    return R


# ---------------------------------------------------------------------------
class OrderingPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.outdir = None
        self.sample = None
        self._scan_shape = None
        self._assigns = None
        self._embeds = None
        self._K = 0
        self._metrics = None
        self._class_avgs = None
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
        self._metrics = None
        self._assigns = None
        self._embeds = None
        self._class_avgs = None
        self._K = 0
        self._render_idle()

    def on_runtime_sample_added(self, key):
        pass

    # ----- UI ----------------------------------------------------------
    def _build(self):
        self._vars = {
            "r":  ctk.IntVar(value=POLAR_SIZE // 4),
            "dr": ctk.IntVar(value=20),
            "sort_by": ctk.StringVar(value="embPC1"),
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
            text="Compute ordering metrics",
            width=200, command=self._kickoff_compute)
        self._compute_btn.pack(side="right", padx=4)

        # body
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        sidebar = ctk.CTkScrollableFrame(body, width=280)
        sidebar.pack(side="left", fill="y")

        _section(sidebar, "Ring annulus (q window)")
        # r slider
        r_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        r_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(r_row, text="r (radial bin):", width=120, anchor="w"
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
        ctk.CTkLabel(dr_row, text="dr thickness:", width=120, anchor="w"
                       ).pack(side="left")
        self._dr_label = ctk.CTkLabel(dr_row,
            text=str(self._vars["dr"].get()), width=40, anchor="e")
        self._dr_label.pack(side="right")
        ctk.CTkSlider(sidebar, from_=2, to=POLAR_SIZE // 2,
                       number_of_steps=POLAR_SIZE // 2 - 2,
                       variable=self._vars["dr"],
                       command=self._on_slider_dr).pack(fill="x", padx=8)
        e_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        e_row.pack(fill="x", padx=4, pady=(8, 2))
        ctk.CTkLabel(e_row, text="r =", width=30, anchor="e").pack(side="left")
        ctk.CTkEntry(e_row, textvariable=self._vars["r"], width=60).pack(
            side="left", padx=4)
        ctk.CTkLabel(e_row, text="dr =", width=40, anchor="e"
                       ).pack(side="left", padx=(8, 0))
        ctk.CTkEntry(e_row, textvariable=self._vars["dr"], width=60).pack(
            side="left", padx=4)
        ctk.CTkButton(e_row, text="recompute", width=80,
                       command=self._kickoff_compute).pack(side="left",
                                                              padx=4)

        _section(sidebar, "Class-avg strip")
        srt_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        srt_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(srt_row, text="sort by:", width=80,
                       anchor="w").pack(side="left")
        ctk.CTkOptionMenu(srt_row, variable=self._vars["sort_by"],
            values=METRIC_NAMES, width=140,
            command=lambda _v: self._refresh()).pack(side="left", padx=2)
        ctk.CTkButton(sidebar, text="Save snapshot",
                       command=self._save_snapshot
                       ).pack(fill="x", padx=8, pady=(8, 4))

        self._status_lbl = ctk.CTkLabel(sidebar,
            text="Load a run, choose r/dr, click 'Compute'.",
            font=("Consolas", 9), justify="left",
            text_color=("#444", "#aaa"), wraplength=270)
        self._status_lbl.pack(anchor="w", padx=8, pady=(8, 4))

        # canvas
        canv = ctk.CTkFrame(body)
        canv.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._fig = Figure(figsize=(13, 8))
        self._canvas = FigureCanvasTkAgg(self._fig, master=canv)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, canv)
        self._render_idle()

    # ----- UI callbacks ------------------------------------------------
    def _on_slider_r(self, val):
        try: self._vars["r"].set(int(round(float(val))))
        except Exception: return
        self._r_label.configure(text=str(self._vars["r"].get()))

    def _on_slider_dr(self, val):
        try: self._vars["dr"].set(int(round(float(val))))
        except Exception: return
        self._dr_label.configure(text=str(self._vars["dr"].get()))

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
                        print(f"[ordering] register: {e!r}", flush=True)
        except Exception as e:
            print(f"[ordering] _train_kwargs load: {e!r}", flush=True)
        self.link_run(p, sample)

    def _annulus(self) -> tuple[int, int]:
        try:
            r = int(self._vars["r"].get())
            dr = max(2, int(self._vars["dr"].get()))
        except Exception:
            return 0, 1
        lo = max(0, r - dr // 2)
        hi = min(POLAR_SIZE, r + dr - dr // 2)
        if hi <= lo:
            hi = min(POLAR_SIZE, lo + 1)
        return lo, hi

    # ----- compute trigger ---------------------------------------------
    def _kickoff_compute(self):
        if not self.outdir or not self.sample:
            messagebox.showinfo("Ordering", "Load a run dir first."); return
        if self._compute_running:
            messagebox.showinfo("Ordering",
                "Compute already in progress."); return
        self._status_lbl.configure(text="getting class assigns + embeds …")
        self.update_idletasks()
        try:
            ph = getattr(self.app, "posthoc", None) if self.app else None
            if ph is not None and getattr(ph, "outdir", None) != self.outdir:
                ph.link_run(self.outdir, self.sample)
            if ph is None or not hasattr(ph, "_ensure_inference"):
                messagebox.showinfo("Ordering",
                    "Need PostHocPanel to compute class assignments.")
                return
            if not ph._ensure_inference():
                self._status_lbl.configure(
                    text="inference failed — see Analysis tab")
                return
            self._assigns = ph._inf["assigns"]
            self._embeds = ph._inf.get("embeds")
            self._K = int(ph._inf["soft_probs"].shape[1])
            self._scan_shape = ph._scan_shape
        except Exception as e:
            messagebox.showerror("Ordering",
                f"class lookup failed:\n{e!r}")
            return

        self._compute_running = True
        self._compute_btn.configure(state="disabled")
        self._compute_thread = threading.Thread(
            target=self._compute_worker, daemon=True)
        self._compute_thread.start()
        self._poll()

    def _compute_worker(self):
        try:
            r_lo, r_hi = self._annulus()
            def cb(done, total):
                with self._lock:
                    self._compute_progress = (
                        f"polar+stats … {done}/{total} "
                        f"({100*done/max(total,1):.0f}%)")
            t0 = time.perf_counter()
            self._metrics = compute_metrics(
                self.sample, self._assigns, self._embeds,
                self._scan_shape, r_lo, r_hi, progress_cb=cb)
            # Pull class averages from the post-hoc panel (re-uses its
            # cached LoadPRZ + polar pipeline). Done in this thread to
            # avoid blocking the UI; safe because only torch ops touch
            # CUDA, no Tk widgets are touched here.
            with self._lock:
                self._compute_progress = "computing class averages …"
            try:
                ph = getattr(self.app, "posthoc", None)
                if ph is not None:
                    self._class_avgs = ph._compute_class_averages()
            except Exception as e:
                print(f"[ordering] class-avg compute failed: {e!r}",
                      flush=True)
                self._class_avgs = None
            with self._lock:
                self._compute_progress = (
                    f"done. ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:
            with self._lock:
                self._compute_progress = f"failed: {e!r}"
            print(f"[ordering] worker failed: {e!r}", flush=True)
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
            if self._metrics is not None:
                self._refresh()

    # ----- rendering ---------------------------------------------------
    def _refresh(self):
        if self._metrics is None:
            self._render_idle(); return
        m = self._metrics
        K = self._K
        r_lo, r_hi = self._annulus()
        rho = spearman_matrix(m)

        # Sort key
        sort_by = self._vars["sort_by"].get()
        order = np.argsort(m[sort_by])     # ascending: less → more ordered

        self._fig.clear()
        # Layout: 3 rows.
        #   Row 0: [Spearman matrix]  [per-class metric bars]
        #   Row 1: per-class 1D mean radial profiles, annulus shaded
        #          (so you can see *which ring* the metrics use)
        #   Row 2: sorted class-average strip
        gs = self._fig.add_gridspec(3, 2,
                                       height_ratios=[1.2, 0.7, 1.0])
        ax_rho = self._fig.add_subplot(gs[0, 0])
        ax_bar = self._fig.add_subplot(gs[0, 1])
        ax_prof  = self._fig.add_subplot(gs[1, :])
        ax_strip = self._fig.add_subplot(gs[2, :])

        # ---- |Spearman ρ| heatmap ----
        im = ax_rho.imshow(rho, cmap="viridis", vmin=0, vmax=1,
                            interpolation="nearest")
        ax_rho.set_xticks(range(len(METRIC_NAMES)))
        ax_rho.set_yticks(range(len(METRIC_NAMES)))
        ax_rho.set_xticklabels(METRIC_NAMES, rotation=30, ha="right")
        ax_rho.set_yticklabels(METRIC_NAMES)
        for i in range(len(METRIC_NAMES)):
            for j in range(len(METRIC_NAMES)):
                txt = f"{rho[i,j]:.2f}"
                col = "white" if rho[i,j] < 0.6 else "black"
                ax_rho.text(j, i, txt, ha="center", va="center",
                              color=col, fontsize=8)
        # Off-diagonal mean as a quick "1D ordering" headline number.
        off = rho[~np.eye(len(METRIC_NAMES), dtype=bool)]
        head = float(off.mean())
        ax_rho.set_title(f"|Spearman ρ|   ⟨off-diag⟩ = {head:.2f}\n"
                          f"(>0.85 ≈ 1D ordering;   "
                          f"PC1 R²={m['pc1_R2']:.2f})", fontsize=10)
        self._fig.colorbar(im, ax=ax_rho, fraction=0.046, pad=0.04)

        # ---- Per-class metric bars (z-scored so all 5 share the y-axis) ----
        x = np.arange(K)
        width = 0.16
        cmap_q = plt.get_cmap("tab10")
        for i, name in enumerate(METRIC_NAMES):
            v = m[name].astype(np.float64)
            sd = v.std()
            z = (v - v.mean()) / sd if sd > 1e-12 else v - v.mean()
            ax_bar.bar(x + (i - 2) * width, z, width=width,
                        label=name, color=cmap_q(i))
        ax_bar.axhline(0, color="0.5", lw=0.7)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([f"p{c}" for c in range(K)])
        ax_bar.set_ylabel("z-score (per metric)")
        ax_bar.set_title(f"per-class metric values   "
                          f"q-window=[{r_lo},{r_hi})", fontsize=10)
        ax_bar.legend(fontsize=7, ncol=5, loc="upper right")
        ax_bar.grid(True, axis="y", alpha=0.3)

        # ---- Per-class mean radial profile, with annulus shaded ----
        cls_mean = m["cls_mean"]   # (K, POLAR_SIZE)
        n_bins = cls_mean.shape[1]
        bins = np.arange(n_bins)
        for c in range(K):
            ax_prof.plot(bins, cls_mean[c], color=cmap_q(c), lw=1.2,
                          label=f"p{c}")
        ax_prof.axvspan(r_lo, r_hi, color="orange", alpha=0.25,
                          label=f"q-window [{r_lo}, {r_hi})")
        ax_prof.set_xlabel("radial bin")
        ax_prof.set_ylabel("⟨I⟩  per class")
        ax_prof.set_title(
            f"per-class mean radial profile  "
            f"(annulus = the orange band — this is the ring "
            f"used for angvar / PBC / FWHM)", fontsize=10)
        ax_prof.legend(fontsize=7, ncol=min(K + 1, 8),
                         loc="upper right")
        ax_prof.grid(True, alpha=0.3)
        ax_prof.set_xlim(0, n_bins - 1)
        # Re-tick to nm⁻¹ when reciprocal calibration is set.
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
            set_q_axis(ax_prof, n_bins, qpb, axis="x")

        # ---- Sorted class-avg strip ----
        ax_strip.set_axis_off()
        if self._class_avgs is not None and len(self._class_avgs) == K:
            # Concatenate K HxH images side by side, separated by 4-px black
            avgs = self._class_avgs
            sep = 4
            H = avgs[0].shape[0]
            strip = np.zeros((H, K * H + (K - 1) * sep), dtype=np.float32)
            for slot, c in enumerate(order):
                a = avgs[int(c)]
                # Display-stretch each panel on its own (per-class) so a
                # weak class is still visible next to a bright one.
                ref = a.flatten()
                if ref.size and ref.max() > 0:
                    lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
                    a_disp = np.log1p(np.clip(a, lo, hi) - lo)
                else:
                    a_disp = a
                a_disp = a_disp / max(a_disp.max(), 1e-9)
                x0 = slot * (H + sep)
                strip[:, x0:x0 + H] = a_disp
            ax_strip.imshow(strip, cmap="inferno", aspect="auto",
                              interpolation="nearest")
            # Caption each slot with its class id and metric value
            for slot, c in enumerate(order):
                vmet = m[sort_by][int(c)]
                ax_strip.text(slot * (H + sep) + H / 2, -10,
                                f"p{int(c)}\n{sort_by}={vmet:.3g}",
                                ha="center", va="bottom", fontsize=9)
            ax_strip.set_title(
                f"class averages sorted by '{sort_by}' (low → high)",
                fontsize=10, pad=22)
        else:
            ax_strip.text(0.5, 0.5,
                "class averages not available — recompute", ha="center",
                va="center", fontsize=11)

        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _render_idle(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        if self._metrics is None:
            ax.text(0.5, 0.5,
                    "1. Load a run.\n"
                    "2. Pick r / dr (the q-window of the ring you want\n"
                    "   to use for angvar / PBC / FWHM).\n"
                    "3. Click 'Compute ordering metrics'.\n",
                    ha="center", va="center", fontsize=11)
        else:
            ax.text(0.5, 0.5, "Metrics cached. Adjust controls.",
                    ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        self._canvas.draw_idle()

    # ----- save --------------------------------------------------------
    def _save_snapshot(self):
        if self._metrics is None or self.outdir is None:
            messagebox.showinfo("Ordering", "Nothing to save yet."); return
        out_dir = os.path.join(self.outdir, "ordering")
        os.makedirs(out_dir, exist_ok=True)
        try:
            self._fig.savefig(os.path.join(out_dir, "ordering_panel.png"),
                                dpi=140)
        except Exception as e:
            messagebox.showerror("Ordering", f"PNG save failed:\n{e!r}"); return
        # Save metric arrays + spearman + window
        try:
            r_lo, r_hi = self._annulus()
            blob = {
                "r_lo": int(r_lo), "r_hi": int(r_hi),
                "metrics": {n: self._metrics[n].astype(float).tolist()
                              for n in METRIC_NAMES},
                "counts": self._metrics["counts"].astype(int).tolist(),
                "pc1_R2": float(self._metrics["pc1_R2"]),
                "spearman_abs": spearman_matrix(self._metrics).astype(
                    float).tolist(),
                "spearman_off_diag_mean": float(spearman_matrix(
                    self._metrics)[~np.eye(len(METRIC_NAMES),
                                             dtype=bool)].mean()),
            }
            with open(os.path.join(out_dir, "ordering_summary.json"),
                       "w") as fh:
                json.dump(blob, fh, indent=2)
        except Exception as e:
            messagebox.showerror("Ordering",
                f"JSON save failed:\n{e!r}"); return
        self._status_lbl.configure(text=f"saved to {out_dir}")
