"""Small modal dialogs shared across panels."""
from __future__ import annotations
import os
import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox


def ask_load_options(parent, *, shape, dtype, total_bytes, avail_bytes,
                     lazy):
    """On loading a cube, ask whether to real-space bin it first.

    Shows the dataset size vs free RAM and warns if a non-lazy file won't
    fit.  Returns a dict {action: 'bin'|'full'|'cancel', n: int,
    remember: bool} (never None)."""
    Ny, Nx, H, W = shape
    total_gb = total_bytes / 1e9
    avail_gb = avail_bytes / 1e9
    low = (not lazy) and (total_bytes > avail_bytes * 0.85)

    dlg = tk.Toplevel(parent)
    dlg.title("Load data — bin first?")
    dlg.geometry("580x430")
    try:
        dlg.transient(parent.winfo_toplevel())
    except Exception:
        pass
    dlg.grab_set()

    kind = ("lazy — frames read from disk on demand (low RAM)" if lazy
            else "FULL into RAM — needs the memory shown below")
    ctk.CTkLabel(dlg, justify="left", wraplength=556, font=("Segoe UI", 11),
                 text=(f"Dataset:  scan {Ny}×{Nx}   ·   pattern {H}×{W}   ·  "
                       f" {np.dtype(dtype)}\n"
                       f"Full size ≈ {total_gb:.2f} GB     free RAM ≈ "
                       f"{avail_gb:.1f} GB\n"
                       f"Loading:  {kind}")
                 ).pack(padx=12, pady=(10, 4), anchor="w")
    if low:
        ctk.CTkLabel(dlg, wraplength=556, justify="left",
                     text="⚠ Insufficient memory to load this fully — "
                          "binning is strongly recommended.",
                     text_color=("#c0392b", "#e67e22"),
                     font=("Segoe UI", 11, "bold")
                     ).pack(padx=12, pady=2, anchor="w")

    ctk.CTkLabel(dlg, justify="left", wraplength=556,
                 text="Bin in REAL SPACE — average n×n neighbouring scan "
                      "positions (full diffraction detail kept):"
                 ).pack(padx=12, pady=(8, 2), anchor="w")
    n_var = ctk.IntVar(value=2)
    row = ctk.CTkFrame(dlg, fg_color="transparent")
    row.pack(padx=12, pady=2, fill="x")
    ctk.CTkLabel(row, text="bin factor n =").pack(side="left")
    ctk.CTkEntry(row, textvariable=n_var, width=64).pack(side="left", padx=6)
    prev = ctk.CTkLabel(dlg, text="", font=("Consolas", 10))

    def _preview(*_a):
        try:
            n = max(1, int(n_var.get()))
        except Exception:
            n = 1
        oy, ox = Ny // n, Nx // n
        gb = oy * ox * H * W * np.dtype("uint16").itemsize / 1e9
        prev.configure(text=f"→ binned: {oy}×{ox} scan positions  ≈ "
                            f"{gb:.2f} GB on disk")
    for nn in (2, 3, 4, 6, 8):
        ctk.CTkButton(row, text=str(nn), width=34,
                      command=(lambda v=nn: (n_var.set(v), _preview()))
                      ).pack(side="left", padx=2)
    prev.pack(padx=12, pady=2, anchor="w")
    try:
        n_var.trace_add("write", _preview)
    except Exception:
        pass
    _preview()

    remember = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(dlg, variable=remember,
                    text="Remember my choice for this session "
                         "(don't ask again)").pack(padx=12, pady=(8, 6),
                                                    anchor="w")

    result = {"d": {"action": "cancel", "n": 1, "remember": False}}

    def _finish(action):
        try:
            n = max(1, int(n_var.get()))
        except Exception:
            n = 2
        result["d"] = {"action": action, "n": n,
                       "remember": bool(remember.get())}
        dlg.destroy()

    btns = ctk.CTkFrame(dlg, fg_color="transparent")
    btns.pack(side="bottom", fill="x", padx=12, pady=10)
    ctk.CTkButton(btns, text="Cancel", width=84, fg_color=("#a33", "#722"),
                  command=lambda: _finish("cancel")).pack(side="right",
                                                          padx=4)
    full_btn = ctk.CTkButton(btns, text="Load full", width=110,
                             command=lambda: _finish("full"))
    full_btn.pack(side="right", padx=4)
    if low:
        full_btn.configure(fg_color=("#888", "#666"))
    ctk.CTkButton(btns, text="Bin n×n & load", width=150,
                  fg_color=("#2D7A2D", "#1F7A1F"),
                  command=lambda: _finish("bin")).pack(side="right", padx=4)
    dlg.protocol("WM_DELETE_WINDOW", lambda: _finish("cancel"))
    parent.wait_window(dlg)
    return result["d"]


def _closest_factor_pair(n: int) -> tuple[int, int]:
    """Return (a, b) with a·b = n and a as close to √n as possible."""
    if n <= 0:
        return (1, max(1, n))
    s = int(round(n ** 0.5))
    for a in range(s, 0, -1):
        if n % a == 0:
            return (a, n // a)
    return (1, n)


def _all_factor_pairs(n: int, limit: int = 10) -> list[tuple[int, int]]:
    """All (a, b) with a·b = n, a ≤ b, a > 1, sorted closest-to-√n first."""
    if n <= 1:
        return []
    out = []
    for a in range(2, int(n ** 0.5) + 1):
        if n % a == 0:
            out.append((a, n // a))
    out.sort(key=lambda ab: abs(ab[0] - int(round(n ** 0.5))))
    return out[:limit]


def add_cubes_dialog(parent) -> list[str] | None:
    """Incremental multi-file picker.  Opens a modal where the user
    repeatedly clicks 'Add file(s)…' to add cubes from different
    sub-folders, one (or several) at a time.  The running list is
    shown with a count, items can be removed, and 'Done' returns the
    accumulated list.  Returns None on cancel.

    Solves the standard `askopenfilenames` limitation of picking from
    only one directory per dialog.
    """
    win = tk.Toplevel(parent)
    win.title("Multi-load: add cubes one by one")
    win.geometry("780x520")
    try:
        win.transient(parent.winfo_toplevel())
    except Exception:
        pass
    win.grab_set()

    state = {"paths": []}

    head = ctk.CTkLabel(win,
        text=(
            "Click 'Add file(s)…' as many times as you need — each "
            "click opens a fresh file dialog so you can navigate to "
            "a different sub-folder.  Multi-select within one dialog "
            "is supported too.  When done, hit 'Done & register' to "
            "build the multi-sample."),
        justify="left", wraplength=740, font=("Segoe UI", 10))
    head.pack(side="top", padx=10, pady=(8, 4), anchor="w")

    status = ctk.CTkLabel(win, text="0 files added",
                             font=("Segoe UI", 12, "bold"))
    status.pack(side="top", padx=10, pady=4)

    list_frame = ctk.CTkFrame(win)
    list_frame.pack(side="top", fill="both", expand=True,
                      padx=10, pady=4)
    listbox = tk.Listbox(list_frame, font=("Consolas", 9),
                            activestyle="dotbox",
                            selectmode=tk.EXTENDED)
    listbox.pack(side="left", fill="both", expand=True,
                   padx=(4, 0), pady=4)
    sb = tk.Scrollbar(list_frame, command=listbox.yview)
    sb.pack(side="right", fill="y", padx=(0, 4), pady=4)
    listbox.config(yscrollcommand=sb.set)

    def _refresh_list():
        listbox.delete(0, tk.END)
        for i, p in enumerate(state["paths"]):
            listbox.insert(tk.END, f"[{i+1}]  {p}")
        n = len(state["paths"])
        status.configure(
            text=f"{n} file{'s' if n != 1 else ''} added"
                  + (f"  (last: {os.path.basename(state['paths'][-1])})"
                       if state["paths"] else ""))

    def _add_files():
        # askopenfilenames remembers the last folder, so successive
        # adds start where the previous one left off — but you can
        # navigate anywhere.
        ps = filedialog.askopenfilenames(
            parent=win,
            title=f"Add cube(s)  (currently {len(state['paths'])})",
            filetypes=[("Cube files",
                          "*.prz *.npz *.npy *.h5 *.hdf5"),
                          ("PRZ / NPZ", "*.prz *.npz"),
                          ("NPY", "*.npy"),
                          ("HDF5", "*.h5 *.hdf5"),
                          ("All files", "*.*")])
        for p in ps:
            ap = os.path.abspath(p)
            if ap and ap not in state["paths"]:
                state["paths"].append(ap)
        _refresh_list()

    def _remove_selected():
        sel = list(listbox.curselection())
        for idx in reversed(sel):
            if 0 <= idx < len(state["paths"]):
                del state["paths"][idx]
        _refresh_list()

    def _clear_all():
        if not state["paths"]:
            return
        if messagebox.askyesno("Clear", "Remove all files from the list?"):
            state["paths"].clear()
            _refresh_list()

    result = {"paths": None}

    def _done():
        if not state["paths"]:
            messagebox.showinfo(
                "Multi-load",
                "No files added. Hit 'Add file(s)…' first, or "
                "'Cancel' to back out.", parent=win)
            return
        result["paths"] = list(state["paths"])
        win.destroy()

    def _cancel():
        win.destroy()

    # Buttons
    btn_row = ctk.CTkFrame(win, fg_color="transparent")
    btn_row.pack(side="top", fill="x", padx=10, pady=8)
    ctk.CTkButton(btn_row, text="Add file(s)…",
                    fg_color=("#2D7A2D", "#1F7A1F"), width=140,
                    command=_add_files).pack(side="left", padx=4)
    ctk.CTkButton(btn_row, text="Remove selected", width=140,
                    command=_remove_selected).pack(side="left", padx=4)
    ctk.CTkButton(btn_row, text="Clear all", width=100,
                    command=_clear_all).pack(side="left", padx=4)
    ctk.CTkButton(btn_row, text="Cancel", width=80,
                    command=_cancel).pack(side="right", padx=4)
    ctk.CTkButton(btn_row, text="Done & register",
                    fg_color=("#2D7A2D", "#1F7A1F"), width=160,
                    command=_done).pack(side="right", padx=4)

    _refresh_list()
    parent.wait_window(win)
    return result["paths"]


def ask_scan_shape(parent, N: int, H: int, W: int, get_frame=None):
    """Modal popup: pick (Ny, Nx) for a 3D HDF5 master where the cube
    is stored as (N_frames, H, W). Returns (Ny, Nx) or None on cancel.

    Pre-fills with the closest-to-square factor pair of N (for 4D-STEM
    that's almost always the right answer — most scans are square or
    near-square).  Also lists a few plausible factor pairs so the user
    can pick rather than type.
    """
    dlg = tk.Toplevel(parent)
    dlg.title("HDF5: scan shape required")
    dlg.geometry("520x430")
    try:
        dlg.transient(parent.winfo_toplevel())
    except Exception:
        pass
    dlg.grab_set()
    ctk.CTkLabel(dlg, justify="left", wraplength=500, text=(
        f"This HDF5 file (master + data) stores frames as a 3D array:\n"
        f"   N = {N},  H = {H},  W = {W}\n\n"
        f"Provide the scan shape (Ny, Nx) so flat-i → (rx, ry) works.\n"
        f"Ny · Nx may be SMALLER than {N}: continuously-streamed scans "
        f"store extra frames per row while the beam flies back, and those "
        f"are not probe positions. Enter the REAL grid and they are skipped."),
        font=("Segoe UI", 10)).pack(padx=10, pady=(8, 4))
    ny_init, nx_init = _closest_factor_pair(N)
    ny_var = ctk.IntVar(value=ny_init)
    nx_var = ctk.IntVar(value=nx_init)
    row = ctk.CTkFrame(dlg, fg_color="transparent")
    row.pack(padx=10, pady=4, fill="x")
    ctk.CTkLabel(row, text="Ny =").pack(side="left", padx=(2, 2))
    ctk.CTkEntry(row, textvariable=ny_var, width=80).pack(side="left",
                                                              padx=4)
    ctk.CTkLabel(row, text="Nx =").pack(side="left", padx=(8, 2))
    ctk.CTkEntry(row, textvariable=nx_var, width=80).pack(side="left",
                                                              padx=4)
    # Suggestion row: clickable factor pairs.
    sug_pairs = _all_factor_pairs(N, limit=8)
    if sug_pairs:
        ctk.CTkLabel(dlg, text="Suggestions (click to fill):",
                       font=("Segoe UI", 9, "bold")
                       ).pack(padx=10, pady=(8, 0), anchor="w")
        sug_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        sug_frame.pack(padx=10, pady=2, fill="x")
        def _make_setter(a, b):
            def _set():
                ny_var.set(a); nx_var.set(b)
            return _set
        for i, (a, b) in enumerate(sug_pairs):
            ctk.CTkButton(sug_frame, text=f"{a} × {b}", width=90,
                            command=_make_setter(a, b)
                            ).grid(row=i // 4, column=i % 4,
                                       padx=2, pady=2)

    # ---- flyback -----------------------------------------------------
    # A continuously-streamed scan keeps recording while the beam flies back
    # to the start of the next row.  Those frames are not probe positions.
    fly_box = ctk.CTkFrame(dlg, fg_color="transparent")
    fly_box.pack(padx=10, pady=(10, 0), fill="x")
    ctk.CTkLabel(fly_box, text="Is there flyback? Enter your Ny, then",
                 font=("Segoe UI", 10, "bold")).pack(side="left")
    detect_btn = ctk.CTkButton(
        fly_box, text="Yes - find it for me", width=170,
        command=lambda: _detect())
    detect_btn.pack(side="left", padx=8)
    if get_frame is None:
        detect_btn.configure(state="disabled")
    fly_info = ctk.CTkLabel(dlg, text=(
        "" if get_frame is not None else
        "(automatic detection is not available for this file type)"),
        font=("Consolas", 9), justify="left", wraplength=500,
        text_color=("#555", "#aaa"))
    fly_info.pack(padx=10, pady=(2, 0), anchor="w")

    status = ctk.CTkLabel(dlg, text="", font=("Consolas", 9),
                          justify="left", wraplength=500)
    status.pack(padx=10, pady=2)
    result = {"shape": None}

    detected = {"sig": None, "raster": None, "grid": None}

    def _detect():
        """Full cellulose recipe: raster from the descan, scan start from
        row coherence, per-row slope check.  Uses the Ny you entered as the
        number of scan rows (acquisition knowledge, like N_ROWS = 50 in the
        cellulose pipeline); the frame pass is cached so changing Ny and
        pressing again is instant."""
        try:
            from flyback import frame_signals, detect_raster
        except Exception as e:
            fly_info.configure(text=f"detector unavailable: {e}"); return
        try:
            ny_in = int(ny_var.get())
        except Exception:
            ny_in = 0
        detect_btn.configure(state="disabled", text="reading frames...")
        try:
            if detected["sig"] is None:
                def prog(i, n):
                    fly_info.configure(
                        text=f"measuring beam position in every frame... "
                             f"{i}/{n} ({100*i//max(n,1)}%)")
                    try: dlg.update()
                    except Exception: pass
                detected["sig"] = frame_signals(get_frame, N, progress=prog)
            r = detect_raster(detected["sig"],
                              n_rows=ny_in if ny_in > 1 else None)
            ny, nx = r["n_rows"], r["n_scan"]
            detected["raster"] = dict(f0=r["f0"], period=r["period"],
                                      n_scan=nx, n_rows=ny)
            detected["grid"] = (ny, nx)
            ny_var.set(ny); nx_var.set(nx)
            notes = []
            if r["raster_conf"] < 0.5:
                notes.append("LOW CONFIDENCE in the raster - check it.")
            if r["start_margin"] < 0.02:
                notes.append("the scan START is poorly determined.")
            if ny_in <= 1:
                notes.append("Ny was not given, so the row count is a guess "
                             "- enter your real Ny and press again.")
            fly_info.configure(text=(
                f"Raster: {r['period']} frames per row = {nx} scan + "
                f"{r['n_flyback']} flyback.  Scan starts at frame "
                f"{r['f0']}; {ny} rows kept"
                + (f" ({r['rows_rejected']} row(s) failed the slope check)"
                   if r["rows_rejected"] else "")
                + f".\nConfidence {r['raster_conf']:.2f}, "
                f"row-to-row correlation {r['row_corr']:.3f}."
                + ("  " + "  ".join(notes) if notes else "")))
        except Exception as e:
            fly_info.configure(text=f"no raster found: {e}")
        finally:
            detect_btn.configure(state="normal", text="Yes - find it for me")

    def _plan(ny, nx):
        """How (ny, nx) maps onto the N stored frames.

        Exact fit -> every frame is a probe position.  Otherwise treat the
        file as rows of `period` stored frames and keep the first nx of
        each: the tail of a row is flyback, which is not a probe position
        and, left in, shears the map and reads as a bright streak.
        """
        if ny <= 0 or nx <= 0:
            return None, "Ny and Nx must be positive."
        if ny * nx == N:
            return nx, f"exact fit: {ny}x{nx} = {N} frames, none skipped."
        if ny * nx > N:
            return None, (f"Ny x Nx = {ny*nx} needs more than the {N} "
                          f"frames in the file.")
        period = N // ny
        if period < nx:
            return None, (f"{ny} rows of at least {nx} frames needs "
                          f"{ny*nx}, but only {N} frames exist.")
        per_row = period - nx
        tail = N - ny * period
        return period, (f"{ny} rows x {period} stored frames; keeping the "
                        f"first {nx} of each.\n"
                        f"Skipping {per_row}/row (flyback) = {ny*per_row}"
                        + (f", plus {tail} trailing." if tail else "."))

    def _preview(*_a):
        try:
            ny = int(ny_var.get()); nx = int(nx_var.get())
        except Exception:
            status.configure(text=""); return
        period, msg = _plan(ny, nx)
        status.configure(text=msg,
                         text_color=(("#2D7A2D", "#7AC07A") if period
                                     else ("#B00020", "#FF6B6B")))

    def _ok():
        try:
            ny = int(ny_var.get()); nx = int(nx_var.get())
        except Exception:
            status.configure(text="bad integer"); return
        period, msg = _plan(ny, nx)
        if period is None:
            status.configure(text=msg); return
        if detected["raster"] is not None and detected["grid"] == (ny, nx):
            raster = dict(detected["raster"])
        elif ny * nx == N:
            raster = None
        else:
            raster = dict(f0=0, period=int(period), n_scan=nx, n_rows=ny)
        result["shape"] = (ny, nx, raster)
        dlg.destroy()

    def _cancel():
        dlg.destroy()

    btns = ctk.CTkFrame(dlg, fg_color="transparent")
    btns.pack(pady=8)
    ctk.CTkButton(btns, text="OK", width=80,
                   command=_ok).pack(side="left", padx=4)
    ctk.CTkButton(btns, text="Cancel", width=80,
                   command=_cancel).pack(side="left", padx=4)
    for _v in (ny_var, nx_var):
        try: _v.trace_add("write", _preview)
        except Exception: pass
    _preview()
    parent.wait_window(dlg)
    return result["shape"]


def ask_h5_dataset(parent, filename: str, datasets: list):
    """Modal popup: the HDF5 file holds several scans -- which one?
    `datasets` is [(path, shape), ...] largest first.  Returns the chosen
    path or None on cancel."""
    dlg = tk.Toplevel(parent)
    dlg.title("HDF5: several scans in this file")
    dlg.geometry("520x200")
    try:
        dlg.transient(parent.winfo_toplevel())
    except Exception:
        pass
    dlg.grab_set()
    ctk.CTkLabel(dlg, justify="left", wraplength=500, text=(
        f"{os.path.basename(filename)} contains {len(datasets)} data "
        f"arrays.  Pick the one to load (each loads as its own sample):"),
        font=("Segoe UI", 10)).pack(padx=10, pady=(10, 6))
    labels = [f"{p}   {' x '.join(str(x) for x in sh)}"
              for p, sh in datasets]
    var = ctk.StringVar(value=labels[0])
    ctk.CTkOptionMenu(dlg, values=labels, variable=var,
                      width=480).pack(padx=10, pady=6)
    result = {"path": None}

    def _ok():
        result["path"] = datasets[labels.index(var.get())][0]
        dlg.destroy()

    btns = ctk.CTkFrame(dlg, fg_color="transparent")
    btns.pack(pady=(10, 8))
    ctk.CTkButton(btns, text="OK", width=80, command=_ok).pack(
        side="left", padx=6)
    ctk.CTkButton(btns, text="Cancel", width=80,
                  command=dlg.destroy).pack(side="left", padx=6)
    parent.wait_window(dlg)
    return result["path"]
