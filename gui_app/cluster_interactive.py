"""cluster_interactive.py -- shared interactive cluster-map viewer.

Gives the NMF and DINO+cluster class maps the SAME click interactivity
as the post-hoc DINO class map, in a large dedicated window (not the
small maps cramped into the summary figure):

    left-click        -> single diffraction pattern popup
    right-click       -> cluster connected-component ("grain") average
    shift+right-click -> add that grain to a stacked-comparison window

No trained model is required: everything is computed from the raw
diffraction cube + a label map, so NMF (model-free) and DINO+cluster
both use the identical viewer.

Public entry point:

    open_interactive_clustermap(parent, sample=..., scan_shape=(Ny,Nx),
                                labels={method: 1d-or-2d int array},
                                recip_per_px=0.0, title="...")
"""
from __future__ import annotations
import numpy as np
import customtkinter as ctk
import tkinter as tk

import matplotlib
matplotlib.use("TkAgg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from data import SAMPLES, LoadPRZ


def open_interactive_clustermap(parent, *, sample, scan_shape, labels,
                                 recip_per_px=0.0, title="cluster map"):
    """Open a big interactive cluster-map window.  Returns the viewer."""
    return _ClusterMapViewer(parent, sample, tuple(scan_shape),
                              labels or {}, float(recip_per_px or 0.0),
                              title)


class _ClusterMapViewer:
    def __init__(self, parent, sample, scan_shape, labels, recip_per_px,
                  title):
        self.parent = parent
        self.sample = sample
        self.scan_shape = scan_shape
        self.rp = recip_per_px
        Ny, Nx = scan_shape
        # Normalise every label array to a 2-D (Ny, Nx) int grid.
        self.labels = {}
        for m, lbl in labels.items():
            a = np.asarray(lbl)
            if a.size == Ny * Nx:
                self.labels[m] = a.reshape(Ny, Nx).astype(int)
        if not self.labels:
            raise ValueError(
                f"no clustering label map matches scan_shape {scan_shape}")
        self._ds = None                # lazy LoadPRZ
        self._grain_stack = []
        self._grain_stack_win = None
        self._build(title)

    # ------------------------------------------------------------------
    def _build(self, title):
        win = tk.Toplevel(self.parent)
        win.title(title)
        win.geometry("1080x920")
        self.win = win
        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.pack(side="top", fill="x", padx=6, pady=4)
        self.method_var = ctk.StringVar(value=next(iter(self.labels)))
        if len(self.labels) > 1:
            ctk.CTkLabel(bar, text="method:").pack(side="left", padx=(6, 2))
            ctk.CTkOptionMenu(bar, variable=self.method_var,
                               values=list(self.labels), width=170,
                               command=lambda _v: self._draw()
                               ).pack(side="left", padx=2)
        ctk.CTkLabel(
            bar,
            text=("left-click → pattern    |    right-click → "
                  "cluster-grain avg    |    shift+right-click → stack"),
            font=("Segoe UI", 10), text_color=("#444", "#bbb")
            ).pack(side="left", padx=14)
        body = ctk.CTkFrame(win)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        self.fig = Figure(figsize=(8.6, 8.2), dpi=110, facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self._draw()

    def _draw(self):
        m = self.method_var.get()
        grid = self.labels[m]
        K = int(grid.max()) + 1
        base = "tab20" if K > 10 else "tab10"
        cmap = plt.get_cmap(base)
        pal = ListedColormap([cmap(i % cmap.N) for i in range(max(K, 1))])
        self.ax.clear()
        im = self.ax.imshow(grid, cmap=pal, vmin=-0.5, vmax=K - 0.5,
                             interpolation="nearest")
        self.ax.set_title(
            f"{self.sample}   {m}   K={K}\n"
            f"left=pattern   right=cluster-grain   shift+right=stack",
            fontsize=11)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        if not getattr(self, "_cb", None):
            self._cb = self.fig.colorbar(im, ax=self.ax, fraction=0.046,
                                          pad=0.04)
            self._cb.set_label("cluster id")
        else:
            self._cb.update_normal(im)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        Ny, Nx = self.scan_shape
        x = int(round(event.xdata)); y = int(round(event.ydata))
        x = max(0, min(Nx - 1, x)); y = max(0, min(Ny - 1, y))
        shift = bool(event.key) and ("shift" in str(event.key).lower())
        if event.button == 3 and shift:
            self._add_grain_to_stack(y, x)
        elif event.button == 1:
            self._show_pattern(y, x)
        elif event.button == 3:
            self._show_grain(y, x)

    # ---- raw access --------------------------------------------------
    def _dataset(self):
        if self._ds is None:
            cfg = SAMPLES[self.sample]
            self._ds = LoadPRZ(cfg["path"], resize=192,
                                vmax=cfg.get("vmax", 2.0))
        return self._ds

    def _raw(self, y, x):
        Ny, Nx = self.scan_shape
        return self._dataset().get_raw(int(y * Nx + x)).astype(np.float32)

    def _grain(self, y, x):
        """Connected-component (4-conn) of same-cluster pixels at (y,x).
        Returns (mask2d, avg2d, cls, n_pix) or None."""
        from scipy.ndimage import label
        grid = self.labels[self.method_var.get()]
        Ny, Nx = self.scan_shape
        cls = int(grid[y, x])
        lab, _ = label(grid == cls)
        gid = int(lab[y, x])
        if gid == 0:
            return None
        mask = (lab == gid)
        pix = np.where(mask.flatten())[0]
        avg = np.mean([self._dataset().get_raw(int(i)) for i in pix],
                      axis=0).astype(np.float32)
        return mask, avg, cls, int(pix.size)

    # ---- popups ------------------------------------------------------
    def _diff_popup(self, title, raw2d):
        cfg = SAMPLES[self.sample]
        train_vmax = float(cfg.get("vmax", 2.0))
        win = tk.Toplevel(self.win)
        win.title(title)
        win.geometry("640x680")
        ctrl = ctk.CTkFrame(win, fg_color="transparent")
        ctrl.pack(side="top", fill="x", padx=6, pady=4)
        vmax_var = ctk.DoubleVar(value=train_vmax)
        log_var = ctk.BooleanVar(value=False)
        ctk.CTkLabel(ctrl, text="vmax:").pack(side="left", padx=(4, 2))
        ent = ctk.CTkEntry(ctrl, textvariable=vmax_var, width=70)
        ent.pack(side="left", padx=2)
        ctk.CTkButton(ctrl, text="reset", width=56,
                       command=lambda: (vmax_var.set(train_vmax), _redraw())
                       ).pack(side="left", padx=2)
        ctk.CTkCheckBox(ctrl, text="log stretch", variable=log_var,
                         command=lambda: _redraw()).pack(side="left", padx=8)
        fig = Figure(figsize=(6.0, 6.0), dpi=110, facecolor="white")
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def _redraw():
            ax.clear()
            try:
                vm = max(float(vmax_var.get()), 1e-6)
            except Exception:
                vm = train_vmax
            img = np.clip(raw2d / vm, 0.0, 1.0)
            if log_var.get():
                img = np.log1p(img * 50)
            ax.imshow(img, cmap="inferno", interpolation="nearest")
            stag = "  log1p×50" if log_var.get() else ""
            ax.set_title(f"{title}  [vmax={vm:.3g}{stag}]", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            canvas.draw_idle()
        ent.bind("<Return>", lambda _e: _redraw())
        _redraw()
        if self.rp > 0:
            try:
                from gui_app._ui import attach_hover_q
                H, W = raw2d.shape
                attach_hover_q(canvas, ax, center=(H / 2.0, W / 2.0),
                                q_per_disp_px=self.rp, units="nm⁻¹")
            except Exception:
                pass

    def _show_pattern(self, y, x):
        self._diff_popup(f"pattern @ (y={y}, x={x})", self._raw(y, x))

    def _show_grain(self, y, x):
        g = self._grain(y, x)
        if g is None:
            return
        mask, avg, cls, n = g
        self._diff_popup(
            f"cluster-grain avg  c{cls} @ (y={y}, x={x})  ({n}px)", avg)

    # ---- stacking (shift+right-click) --------------------------------
    def _add_grain_to_stack(self, y, x):
        g = self._grain(y, x)
        if g is None:
            return
        mask, avg, cls, n = g
        for rec in self._grain_stack:
            if rec["cls"] == cls and rec["mask"][y, x]:
                return
        self._grain_stack.append(dict(y=y, x=x, mask=mask, avg=avg,
                                       cls=cls, n=n))
        self._ensure_stack_win()
        self._redraw_stack()

    def _ensure_stack_win(self):
        win = getattr(self, "_grain_stack_win", None)
        if win is not None and bool(win.winfo_exists()):
            return
        win = tk.Toplevel(self.win)
        win.title("stacked cluster-grains")
        win.geometry("1000x900")
        bar = ctk.CTkFrame(win, fg_color="transparent")
        bar.pack(side="top", fill="x", padx=6, pady=4)
        self._gs_count = ctk.CTkLabel(bar, text="")
        self._gs_count.pack(side="left", padx=6)
        cfg = SAMPLES[self.sample]
        self._gs_vmax = ctk.DoubleVar(value=float(cfg.get("vmax", 2.0)))
        self._gs_log = ctk.BooleanVar(value=False)
        ctk.CTkLabel(bar, text="vmax:").pack(side="left", padx=(12, 2))
        ent = ctk.CTkEntry(bar, textvariable=self._gs_vmax, width=70)
        ent.pack(side="left", padx=2)
        ent.bind("<Return>", lambda _e: self._redraw_stack())
        ctk.CTkCheckBox(bar, text="log stretch", variable=self._gs_log,
                         command=self._redraw_stack).pack(side="left",
                                                          padx=8)
        ctk.CTkButton(bar, text="Clear", width=64,
                       command=self._clear_stack).pack(side="right", padx=4)
        holder = ctk.CTkScrollableFrame(win, fg_color="transparent")
        holder.pack(side="top", fill="both", expand=True)
        self._grain_stack_win = win
        self._gs_holder = holder

    def _redraw_stack(self):
        win = getattr(self, "_grain_stack_win", None)
        if win is None or not bool(win.winfo_exists()):
            return
        stack = self._grain_stack
        n = len(stack)
        for w in self._gs_holder.winfo_children():
            w.destroy()
        try:
            self._gs_count.configure(text=f"{n} grain(s) stacked")
        except Exception:
            pass
        if n == 0:
            return
        Ny, Nx = self.scan_shape
        grid = self.labels[self.method_var.get()]
        K = int(grid.max()) + 1
        cmap = plt.get_cmap("tab20" if K > 10 else "tab10")
        pal = ListedColormap([cmap(i % cmap.N) for i in range(max(K, 1))])
        try:
            vm = max(float(self._gs_vmax.get()), 1e-6)
        except Exception:
            vm = float(SAMPLES[self.sample].get("vmax", 2.0))
        log_on = bool(self._gs_log.get())
        fig = Figure(figsize=(8.0, 3.6 * n), dpi=110, facecolor="white")
        pat_axes = []
        for r, rec in enumerate(stack):
            ax_m = fig.add_subplot(n, 2, 2 * r + 1)
            ax_p = fig.add_subplot(n, 2, 2 * r + 2)
            ax_m.imshow(grid, cmap=pal, vmin=-0.5, vmax=K - 0.5,
                        interpolation="nearest")
            ax_m.imshow(np.where(rec["mask"], 1.0, np.nan),
                        cmap=ListedColormap(["black"]), alpha=0.9,
                        interpolation="nearest")
            ax_m.set_title(f"c{rec['cls']} @ (y={rec['y']}, x={rec['x']})  "
                           f"{rec['n']}px", fontsize=9)
            ax_m.set_xticks([]); ax_m.set_yticks([])
            img = np.clip(rec["avg"] / vm, 0.0, 1.0)
            if log_on:
                img = np.log1p(img * 50)
            ax_p.imshow(img, cmap="inferno", interpolation="nearest")
            stag = "  log1p×50" if log_on else ""
            ax_p.set_title(f"grain-avg diffraction [vmax={vm:.3g}{stag}]",
                           fontsize=9)
            ax_p.set_xticks([]); ax_p.set_yticks([])
            H, W = rec["avg"].shape
            pat_axes.append((ax_p, H, W))
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self._gs_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        if self.rp > 0:
            try:
                from gui_app._ui import attach_hover_q
                for ax_p, H, W in pat_axes:
                    attach_hover_q(canvas, ax_p, center=(H / 2.0, W / 2.0),
                                    q_per_disp_px=self.rp, units="nm⁻¹")
            except Exception:
                pass
        try:
            win.lift()
        except Exception:
            pass

    def _clear_stack(self):
        self._grain_stack = []
        win = getattr(self, "_grain_stack_win", None)
        if win is not None and bool(win.winfo_exists()):
            win.destroy()
        self._grain_stack_win = None
