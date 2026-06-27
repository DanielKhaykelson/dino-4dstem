"""gui_vmax_picker.py -- Tk GUI to inspect .prz patterns at different vmax.

No CLI args needed. Just run it; click Browse to pick a .prz; type a
vmax; navigate via Prev/Next/Random/Jump. Lazy-loads only the
requested frame from disk so it stays snappy on huge cubes.

Inspired by prz_labeler.py.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def rescale_like_vmax(x, vmax, vmin=None, out_range=(0.0, 1.0),
                       dtype=np.float32):
    x = np.asarray(x); lo, hi = map(float, out_range)
    if x.ndim == 2:
        vm = x.min() if vmin is None else float(vmin)
        d = (float(vmax) - vm) or 1e-12
        y = (x - vm) / d
    else:
        raise ValueError("Expected 2D")
    return (np.clip(y, 0, 1) * (hi - lo) + lo).astype(dtype, copy=False)


class LazyPRZLoader:
    def __init__(self, path):
        self.path = path
        # prefer .cube.npy sidecar (true mmap)
        base, _ = os.path.splitext(path)
        cand = base + ".cube.npy"
        if os.path.exists(cand):
            self._cube = np.load(cand, mmap_mode="r")
        else:
            arr = np.load(path, allow_pickle=True, mmap_mode="r")
            self._cube = arr["data"]
        self.Ny, self.Nx, self.H, self.W = self._cube.shape
        self.N = self.Ny * self.Nx

    def get_raw(self, idx):
        y, x = divmod(idx, self.Nx)
        return np.asarray(self._cube[y, x]).astype(np.float32)


class VmaxPickerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("vmax picker")
        self.geometry("780x780")
        self.loader = None
        self._idx = 0
        self._build_ui()

    def _build_ui(self):
        tb = tk.Frame(self, bd=1, relief=tk.RIDGE, pady=3)
        tb.pack(fill=tk.X, padx=4, pady=2)

        tk.Label(tb, text="PRZ:").pack(side=tk.LEFT)
        self._prz_var = tk.StringVar()
        tk.Entry(tb, textvariable=self._prz_var, width=44).pack(side=tk.LEFT, padx=2)
        tk.Button(tb, text="Browse", command=self._browse).pack(side=tk.LEFT)
        tk.Button(tb, text="Load", command=self._load_prz).pack(side=tk.LEFT, padx=(0, 6))

        # ---- second toolbar row: vmax + log + colormap ----
        tb2 = tk.Frame(self, bd=1, relief=tk.RIDGE, pady=3)
        tb2.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(tb2, text="vmax:").pack(side=tk.LEFT)
        self._vmax_var = tk.StringVar(value="2")
        e = tk.Entry(tb2, textvariable=self._vmax_var, width=8)
        e.pack(side=tk.LEFT, padx=2)
        e.bind('<Return>', lambda _e: self._redraw())
        tk.Button(tb2, text="Apply", command=self._redraw).pack(side=tk.LEFT, padx=(0, 6))

        for v in (1, 2, 3, 5, 10, 30, 100):
            tk.Button(tb2, text=str(v), width=4,
                       command=lambda vv=v: self._set_vmax(vv)).pack(side=tk.LEFT, padx=1)

        self._log_var = tk.IntVar(value=0)
        tk.Checkbutton(tb2, text="log1p", variable=self._log_var,
                        command=self._redraw).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(tb2, text="cmap:").pack(side=tk.LEFT, padx=(8, 0))
        self._cmap_var = tk.StringVar(value="inferno")
        cmap_menu = tk.OptionMenu(tb2, self._cmap_var,
                                    "inferno", "viridis", "magma", "gray",
                                    "plasma", "cividis",
                                    command=lambda _v: self._redraw())
        cmap_menu.config(width=8)
        cmap_menu.pack(side=tk.LEFT, padx=2)

        # ---- main: image + stats ----
        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        fig = Figure(figsize=(6, 6), dpi=95)
        self._ax = fig.add_subplot(111)
        self._ax.axis('off')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=body)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close('all')

        self._stats_label = tk.Label(self, text="(no pattern)",
                                       font=("Consolas", 9), anchor='w')
        self._stats_label.pack(fill=tk.X, padx=8)

        # ---- nav ----
        nav = tk.Frame(self, bd=1, relief=tk.RIDGE, pady=3)
        nav.pack(fill=tk.X, padx=4, pady=2)

        tk.Button(nav, text="<< Prev (10)",
                   command=lambda: self._step(-10)).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="< Prev",
                   command=lambda: self._step(-1)).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="Next >",
                   command=lambda: self._step(+1)).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="Next (10) >>",
                   command=lambda: self._step(+10)).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="Random",
                   command=self._random).pack(side=tk.LEFT, padx=8)

        tk.Label(nav, text="  Jump:").pack(side=tk.LEFT)
        self._jump_var = tk.StringVar(value="0")
        je = tk.Entry(nav, textvariable=self._jump_var, width=8)
        je.pack(side=tk.LEFT, padx=2)
        je.bind('<Return>', lambda _e: self._jump())
        tk.Button(nav, text="Go", command=self._jump).pack(side=tk.LEFT)

        # keyboard shortcuts
        self.bind_all('<Left>',  lambda _e: self._step(-1))
        self.bind_all('<Right>', lambda _e: self._step(+1))
        self.bind_all('<Up>',    lambda _e: self._step(+10))
        self.bind_all('<Down>',  lambda _e: self._step(-10))
        self.bind_all('r',       lambda _e: self._random())

    # ---- handlers ----
    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("PRZ/NPZ", "*.prz *.npz"),
                        ("Numpy", "*.npy"),
                        ("All", "*.*")])
        if path:
            self._prz_var.set(path)
            self._load_prz()

    def _load_prz(self):
        path = self._prz_var.get().strip()
        if not path:
            return
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Not found: {path}")
            return
        try:
            self.loader = LazyPRZLoader(path)
        except Exception as e:
            messagebox.showerror("Error", f"Load failed:\n{e}")
            return
        self._idx = self.loader.N // 2
        self._jump_var.set(str(self._idx))
        # Auto-suggest vmax from this frame's 99.9th pct
        try:
            seed = self.loader.get_raw(self._idx)
            p999 = float(np.percentile(seed, 99.9))
            self._vmax_var.set(f"{p999:.3g}")
        except Exception:
            pass
        self.title(f"vmax picker - {os.path.basename(path)} "
                    f"({self.loader.Ny}x{self.loader.Nx} scan, "
                    f"{self.loader.H}x{self.loader.W} pattern)")
        self._redraw()

    def _set_vmax(self, v):
        self._vmax_var.set(str(v))
        self._redraw()

    def _step(self, delta):
        if self.loader is None: return
        self._idx = max(0, min(self.loader.N - 1, self._idx + delta))
        self._jump_var.set(str(self._idx))
        self._redraw()

    def _random(self):
        if self.loader is None: return
        self._idx = random.randrange(self.loader.N)
        self._jump_var.set(str(self._idx))
        self._redraw()

    def _jump(self):
        if self.loader is None: return
        try:
            i = int(self._jump_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Jump must be int")
            return
        if not (0 <= i < self.loader.N):
            messagebox.showerror("Error",
                                  f"Index in [0, {self.loader.N - 1}]")
            return
        self._idx = i
        self._redraw()

    def _redraw(self):
        if self.loader is None:
            return
        try:
            vmax = float(self._vmax_var.get())
            if vmax <= 0:
                vmax = 1e-3
        except ValueError:
            return
        frame = self.loader.get_raw(self._idx)
        norm = rescale_like_vmax(frame, vmax=vmax)
        if self._log_var.get():
            disp = np.log1p(norm * 50.0)
        else:
            disp = norm
        self._ax.clear()
        self._ax.imshow(disp, cmap=self._cmap_var.get(), aspect='equal',
                         interpolation='nearest')
        self._ax.axis('off')
        y, x = divmod(self._idx, self.loader.Nx)
        self._ax.set_title(
            f"idx={self._idx}  (y={y}, x={x})    vmax={vmax:.4g}    "
            f"{'log1p' if self._log_var.get() else 'linear'}",
            fontsize=10)
        self._canvas.draw()

        # stats line below
        self._stats_label.config(
            text=f"frame  min={frame.min():.3g}  max={frame.max():.3g}  "
                  f"mean={frame.mean():.3g}  median={np.median(frame):.3g}    "
                  f"saturated@vmax = {(frame >= vmax).mean()*100:.2f}% of pixels")


if __name__ == "__main__":
    VmaxPickerApp().mainloop()
