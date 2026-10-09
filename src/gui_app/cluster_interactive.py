"""cluster_interactive.py -- shared interactive cluster-map viewer.

Gives the NMF and DINO+cluster class maps the SAME click interactivity
as the post-hoc DINO class map, in a large dedicated window (not the
small maps cramped into the summary figure):

    left-click        -> single diffraction pattern popup
    right-click       -> cluster connected-component ("grain") average
    shift+right-click -> add that grain to a stacked-comparison window
    "Live" checkbox   -> the pattern under the mouse, updated as you move
    "Compare classes" -> class A average, class B average and A - B

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
from gui_app import display_prefs


def open_interactive_clustermap(parent, *, sample, scan_shape, labels,
                                 recip_per_px=0.0, title="cluster map"):
    """Open a big interactive cluster-map window.  Returns the viewer."""
    return _ClusterMapViewer(parent, sample, tuple(scan_shape),
                              labels or {}, float(recip_per_px or 0.0),
                              title)


def open_class_difference(parent, *, title, class_ids, counts, get_avg,
                          recip_per_px=0.0):
    """Popup: pick classes A and B -> their average patterns and A - B.

    get_avg(c) returns the (H, W) average pattern of class c (the caller
    caches); counts[c] is the number of probe positions in class c.
    A and B share one intensity scale so they can be compared by eye; the
    difference uses a diverging map centred on zero (red: brighter in A,
    blue: brighter in B).
    """
    ids = [int(c) for c in class_ids]
    if len(ids) < 2:
        raise ValueError("need at least two classes to compare")
    names = [f"c{c}  (N={int(counts.get(c, 0))})" for c in ids]
    win = tk.Toplevel(parent)
    win.title(f"{title} - class difference")
    win.geometry("1320x600")
    try:
        win.lift(); win.focus_force()
        win.attributes("-topmost", True)
        win.after(500, lambda: (win.winfo_exists()
                                and win.attributes("-topmost", False)))
    except Exception:
        pass
    bar = ctk.CTkFrame(win, fg_color="transparent")
    bar.pack(side="top", fill="x", padx=6, pady=4)
    a_var = ctk.StringVar(value=names[0])
    b_var = ctk.StringVar(value=names[1])
    log_var = ctk.BooleanVar(value=True)
    norm_var = ctk.BooleanVar(value=False)
    ctk.CTkLabel(bar, text="class A:").pack(side="left", padx=(6, 2))
    ctk.CTkOptionMenu(bar, variable=a_var, values=names, width=150,
                      command=lambda _v: _redraw()).pack(side="left", padx=2)
    ctk.CTkLabel(bar, text="class B:").pack(side="left", padx=(10, 2))
    ctk.CTkOptionMenu(bar, variable=b_var, values=names, width=150,
                      command=lambda _v: _redraw()).pack(side="left", padx=2)
    ctk.CTkCheckBox(bar, text="log stretch (A, B)", variable=log_var,
                    command=lambda: _redraw()).pack(side="left", padx=10)
    ctk.CTkCheckBox(bar, text="normalise to equal total counts",
                    variable=norm_var,
                    command=lambda: _redraw()).pack(side="left", padx=4)
    ctk.CTkButton(bar, text="Save PNG", width=90,
                  command=lambda: _save()).pack(side="right", padx=4)
    status = ctk.CTkLabel(win, text="", font=("Consolas", 9), anchor="w")
    status.pack(side="top", fill="x", padx=8)
    fig = Figure(figsize=(12.6, 4.8), dpi=100, facecolor="white")
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def _redraw():
        ca = ids[names.index(a_var.get())]
        cb = ids[names.index(b_var.get())]
        status.configure(text=f"averaging c{ca} and c{cb} ...")
        try:
            win.update_idletasks()
        except Exception:
            pass
        A = np.asarray(get_avg(ca), dtype=np.float64)
        B = np.asarray(get_avg(cb), dtype=np.float64)
        if norm_var.get():
            A = A / max(A.sum(), 1e-12) * 1e4
            B = B / max(B.sum(), 1e-12) * 1e4
        D = A - B
        vm = float(np.percentile(np.concatenate([A.ravel(), B.ravel()]),
                                 99.5)) or 1.0
        dm = float(np.percentile(np.abs(D), 99.5)) or 1.0
        fig.clear()
        cmap = display_prefs.get_diff_cmap_name()
        for k, (img, c) in enumerate(((A, ca), (B, cb))):
            ax = fig.add_subplot(1, 3, k + 1)
            disp = np.clip(img / vm, 0.0, 1.0)
            if log_var.get():
                disp = np.log1p(disp * 50)
            ax.imshow(disp, cmap=cmap, interpolation="nearest")
            ax.set_title(f"{'AB'[k]} = c{c}   (N={int(counts.get(c, 0))})",
                         fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(1, 3, 3)
        im = ax.imshow(D, cmap="RdBu_r", vmin=-dm, vmax=dm,
                       interpolation="nearest")
        ax.set_title("A - B   (red: stronger in A, blue: in B)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if recip_per_px and recip_per_px > 0:
            try:
                from gui_app._ui import attach_hover_q
                H, W = D.shape
                for a in fig.axes[:3]:
                    attach_hover_q(canvas, a, center=(H / 2.0, W / 2.0),
                                   q_per_disp_px=recip_per_px, units="nm⁻¹")
            except Exception:
                pass
        fig.suptitle(f"{title}:  c{ca} vs c{cb}", fontsize=11)
        fig.tight_layout()
        canvas.draw_idle()
        status.configure(text=(
            f"c{ca} vs c{cb}   |A-B| 99.5th pct = {dm:.3g}   "
            f"(A, B share vmax {vm:.3g}"
            + ("; each scaled to equal total" if norm_var.get() else "")
            + ")"))

    def _save():
        from tkinter import filedialog
        p = filedialog.asksaveasfilename(
            parent=win, defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=f"class_diff_{a_var.get().split()[0]}_"
                        f"{b_var.get().split()[0]}.png")
        if p:
            fig.savefig(p, dpi=150, bbox_inches="tight")
            status.configure(text=f"saved -> {p}")

    _redraw()
    return win


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
        self._avg_cache = {}           # (method, class) -> mean pattern
        self._live_pending = None      # (y, x) waiting to be drawn
        self._live_last = None
        self._build(title)

    # ------------------------------------------------------------------
    def _build(self, title):
        win = tk.Toplevel(self.parent)
        win.title(title)
        # Position centred over the main window + lift to front, so it
        # never opens off-screen / behind a maximised window.
        W_, Hh_ = 1080, 920
        try:
            root = self.parent.winfo_toplevel()
            root.update_idletasks()
            mx, my = root.winfo_rootx(), root.winfo_rooty()
            mw, mh = root.winfo_width(), root.winfo_height()
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            x = min(max(mx + (mw - W_) // 2, 0), max(0, sw - 300))
            y = min(max(my + (mh - Hh_) // 3, 0), max(0, sh - 300))
            win.geometry(f"{W_}x{Hh_}+{x}+{y}")
        except Exception:
            win.geometry(f"{W_}x{Hh_}")
        self.win = win
        try:
            win.lift(); win.focus_force()
            win.attributes("-topmost", True)
            win.after(500, lambda: (win.winfo_exists()
                                    and win.attributes("-topmost", False)))
        except Exception:
            pass
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
        # Colour-scheme pickers (app-wide, live).
        self._class_cmap_var = ctk.StringVar(
            value=display_prefs.get_class_cmap_name())
        ctk.CTkLabel(bar, text="colors:").pack(side="left", padx=(10, 2))
        ctk.CTkOptionMenu(
            bar, variable=self._class_cmap_var,
            values=display_prefs.CLASS_CMAPS, width=150,
            command=lambda v: display_prefs.set_class_cmap_name(v)
            ).pack(side="left", padx=2)
        self._diff_cmap_var = ctk.StringVar(
            value=display_prefs.get_diff_cmap_name())
        ctk.CTkLabel(bar, text="DP cmap:").pack(side="left", padx=(10, 2))
        ctk.CTkOptionMenu(
            bar, variable=self._diff_cmap_var,
            values=display_prefs.DIFF_CMAPS, width=110,
            command=lambda v: display_prefs.set_diff_cmap_name(v)
            ).pack(side="left", padx=2)
        # Re-draw the map live when the palette changes anywhere in the app.
        display_prefs.subscribe(self._on_cmap_change)
        win.protocol("WM_DELETE_WINDOW", self._on_close)
        bar2 = ctk.CTkFrame(win, fg_color="transparent")
        bar2.pack(side="top", fill="x", padx=6, pady=(0, 2))
        self.live_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(bar2, text="Live pattern (follows the mouse)",
                        variable=self.live_var,
                        command=self._toggle_live).pack(side="left", padx=6)
        self.live_log = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(bar2, text="log", variable=self.live_log,
                        command=self._redraw_live, width=60
                        ).pack(side="left", padx=4)
        ctk.CTkButton(bar2, text="Compare classes (A, B, A-B)...",
                      width=200, command=self._open_compare
                      ).pack(side="left", padx=14)
        body = ctk.CTkFrame(win)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        self.fig = Figure(figsize=(8.6, 8.2), dpi=110, facecolor="white")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.ax_live = None
        self._draw()

    def _draw(self):
        m = self.method_var.get()
        grid = self.labels[m]
        K = int(grid.max()) + 1
        pal = display_prefs.class_palette(K)
        live = bool(getattr(self, "live_var", None) and self.live_var.get())
        if live != (self.ax_live is not None):
            # switching layout: rebuild the axes (and the colorbar with them)
            self.fig.clear()
            self._cb = None
            if live:
                self.ax = self.fig.add_subplot(1, 2, 1)
                self.ax_live = self.fig.add_subplot(1, 2, 2)
            else:
                self.ax = self.fig.add_subplot(111)
                self.ax_live = None
        self.ax.clear()
        self._live_mark = None
        self._live_im = None
        if self.ax_live is not None:
            self.ax_live.clear()
        im = self.ax.imshow(grid, cmap=pal, vmin=-0.5, vmax=K - 0.5,
                             interpolation="nearest", aspect="equal")
        self.ax.set_title(
            f"{self.sample}   {m}   K={K}\n"
            f"left=pattern   right=cluster-grain   shift+right=stack",
            fontsize=11)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self._last_im = im
        if not getattr(self, "_cb", None):
            self._cb = self.fig.colorbar(im, ax=self.ax, fraction=0.046,
                                          pad=0.04)
            self._cb.set_label("cluster id")
        else:
            self._cb.update_normal(im)
        if self.ax_live is not None:
            self._redraw_live(draw=False)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---- live pattern ------------------------------------------------
    def _toggle_live(self):
        Ny, Nx = self.scan_shape
        self._live_last = (Ny // 2, Nx // 2) if self.live_var.get() else None
        self._draw()

    def _on_motion(self, event):
        if (self.ax_live is None or event.inaxes is not self.ax
                or event.xdata is None):
            return
        Ny, Nx = self.scan_shape
        x = max(0, min(Nx - 1, int(round(event.xdata))))
        y = max(0, min(Ny - 1, int(round(event.ydata))))
        if (y, x) == self._live_last:
            return
        # coalesce: only the latest position is drawn, after the event burst
        first = self._live_pending is None
        self._live_pending = (y, x)
        if first:
            self.win.after(15, self._flush_live)

    def _flush_live(self):
        yx, self._live_pending = self._live_pending, None
        if yx is None:
            return
        self._live_last = yx
        try:
            self._redraw_live()
        except Exception as e:
            print(f"[cluster-interactive] live view failed: {e!r}",
                  flush=True)

    def _redraw_live(self, draw=True):
        if self.ax_live is None or self._live_last is None:
            return
        y, x = self._live_last
        raw = self._raw(y, x)
        vm = float(np.percentile(raw, 99.5)) or 1.0
        img = np.clip(raw / vm, 0.0, 1.0)
        if self.live_log.get():
            img = np.log1p(img * 50)
        cmap = display_prefs.get_diff_cmap_name()
        if self._live_im is None:
            self.ax_live.clear()
            self._live_im = self.ax_live.imshow(
                img, cmap=cmap, interpolation="nearest")
            self.ax_live.set_xticks([]); self.ax_live.set_yticks([])
        else:
            self._live_im.set_data(img)
            self._live_im.set_cmap(cmap)
        self._live_im.set_clim(float(img.min()), float(img.max()) or 1.0)
        cls = int(self.labels[self.method_var.get()][y, x])
        self.ax_live.set_title(
            f"live: (y={y}, x={x})  class c{cls}  [vmax={vm:.3g}"
            + ("  log" if self.live_log.get() else "") + "]", fontsize=10)
        if self._live_mark is None:
            (self._live_mark,) = self.ax.plot(
                [x], [y], marker="+", ms=14, mew=2, color="white")
        else:
            self._live_mark.set_data([x], [y])
        if draw:
            self.canvas.draw_idle()

    # ---- class comparison --------------------------------------------
    def _class_avg(self, c, cap=300):
        """Mean pattern of class c (random cap positions, fixed seed --
        the same sampling as the NMF class-average window)."""
        m = self.method_var.get()
        key = (m, int(c))
        if key not in self._avg_cache:
            idx = np.flatnonzero(self.labels[m].ravel() == int(c))
            if idx.size > cap:
                idx = np.random.default_rng(42).choice(idx, cap,
                                                       replace=False)
            ds = self._dataset()
            acc = None
            for i in idx:
                f = ds.get_raw(int(i)).astype(np.float64)
                acc = f if acc is None else acc + f
            self._avg_cache[key] = (acc / max(len(idx), 1)).astype(
                np.float32)
        return self._avg_cache[key]

    def _open_compare(self):
        grid = self.labels[self.method_var.get()]
        ids, cnt = np.unique(grid, return_counts=True)
        try:
            open_class_difference(
                self.win, title=f"{self.sample}  {self.method_var.get()}",
                class_ids=ids.tolist(),
                counts={int(i): int(n) for i, n in zip(ids, cnt)},
                get_avg=self._class_avg, recip_per_px=self.rp)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Compare classes", str(e), parent=self.win)

    def _on_cmap_change(self):
        """Colour scheme changed (here or elsewhere) → recolour the map and
        sync the dropdowns."""
        try:
            if not self.win.winfo_exists():
                display_prefs.unsubscribe(self._on_cmap_change)
                return
        except Exception:
            return
        try:
            self._class_cmap_var.set(display_prefs.get_class_cmap_name())
            self._diff_cmap_var.set(display_prefs.get_diff_cmap_name())
        except Exception:
            pass
        self._draw()

    def _on_close(self):
        display_prefs.unsubscribe(self._on_cmap_change)
        try:
            self.win.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    @staticmethod
    def _shift_held(event) -> bool:
        # event.key needs canvas keyboard focus (often missed); the raw
        # Tk event carries a reliable modifier bitmask (Shift = 0x0001).
        ge = getattr(event, "guiEvent", None)
        if ge is not None:
            try:
                return bool(int(ge.state) & 0x0001)
            except Exception:
                pass
        return bool(event.key) and ("shift" in str(event.key).lower())

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        Ny, Nx = self.scan_shape
        x = int(round(event.xdata)); y = int(round(event.ydata))
        x = max(0, min(Nx - 1, x)); y = max(0, min(Ny - 1, y))
        shift = self._shift_held(event)
        try:
            if event.button == 3 and shift:
                self._add_grain_to_stack(y, x)
            elif event.button == 1:
                self._show_pattern(y, x)
            elif event.button == 3:
                self._show_grain(y, x)
        except Exception as e:
            print(f"[cluster-interactive] click handler failed: {e!r}",
                  flush=True)

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
        try:
            win.lift(); win.focus_force()
            win.attributes("-topmost", True)
            win.after(500, lambda: (win.winfo_exists()
                                    and win.attributes("-topmost", False)))
        except Exception:
            pass
        ctrl = ctk.CTkFrame(win, fg_color="transparent")
        ctrl.pack(side="top", fill="x", padx=6, pady=4)
        # Auto-scale to a high percentile and log-stretch by default so the
        # Bragg disks are visible instead of just the saturating direct
        # beam.  "reset" still returns to the model's training vmax.
        try:
            auto_vmax = float(np.percentile(np.asarray(raw2d), 99.5))
        except Exception:
            auto_vmax = train_vmax
        if not (auto_vmax > 0):
            auto_vmax = train_vmax
        vmax_var = ctk.DoubleVar(value=round(auto_vmax, 4))
        log_var = ctk.BooleanVar(value=True)
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
            ax.imshow(img, cmap=display_prefs.get_diff_cmap_name(),
                      interpolation="nearest")
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
        # Log-stretch on by default so the Bragg disks show, not just the
        # saturating direct beam.
        self._gs_log = ctk.BooleanVar(value=True)
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
        pal = display_prefs.class_palette(K)
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
                        interpolation="nearest", aspect="equal")
            ax_m.imshow(np.where(rec["mask"], 1.0, np.nan),
                        cmap=ListedColormap(["black"]), alpha=0.9,
                        interpolation="nearest", aspect="equal")
            ax_m.set_title(f"c{rec['cls']} @ (y={rec['y']}, x={rec['x']})  "
                           f"{rec['n']}px", fontsize=9)
            ax_m.set_xticks([]); ax_m.set_yticks([])
            img = np.clip(rec["avg"] / vm, 0.0, 1.0)
            if log_on:
                img = np.log1p(img * 50)
            ax_p.imshow(img, cmap=display_prefs.get_diff_cmap_name(),
                        interpolation="nearest")
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
            win.deiconify(); win.lift(); win.update_idletasks()
        except Exception:
            pass

    def _clear_stack(self):
        self._grain_stack = []
        win = getattr(self, "_grain_stack_win", None)
        if win is not None and bool(win.winfo_exists()):
            win.destroy()
        self._grain_stack_win = None
