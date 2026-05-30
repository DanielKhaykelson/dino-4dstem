"""Small modal dialogs shared across panels."""
from __future__ import annotations
import os
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox


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


def ask_scan_shape(parent, N: int, H: int, W: int):
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
        f"Constraint: Ny · Nx = {N}."),
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

    status = ctk.CTkLabel(dlg, text="", font=("Consolas", 9))
    status.pack(padx=10, pady=2)
    result = {"shape": None}

    def _ok():
        try:
            ny = int(ny_var.get()); nx = int(nx_var.get())
        except Exception:
            status.configure(text="bad integer"); return
        if ny * nx != N:
            status.configure(
                text=f"Ny·Nx = {ny*nx} ≠ N = {N}.  Try again.")
            return
        result["shape"] = (ny, nx)
        dlg.destroy()

    def _cancel():
        dlg.destroy()

    btns = ctk.CTkFrame(dlg, fg_color="transparent")
    btns.pack(pady=8)
    ctk.CTkButton(btns, text="OK", width=80,
                   command=_ok).pack(side="left", padx=4)
    ctk.CTkButton(btns, text="Cancel", width=80,
                   command=_cancel).pack(side="left", padx=4)
    parent.wait_window(dlg)
    return result["shape"]
