"""assistant_io.py -- shared output-dir + figure-saving for the assistant.

Every analysis (GUI or headless) saves its figures next to the DATA, in a
sequential per-analysis subfolder so reruns never overwrite each other:

    <data_dir>/<data_name>_assistant/<analysis>_001/
    <data_dir>/<data_name>_assistant/<analysis>_002/   (next run)

Use:
    d = new_output_dir(data_path, "nmf")
    save_fig(fig, d, "class_maps")
    save_array_png(label_map, d, "kmeans_map", cmap="tab10")
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")           # safe even when a Tk GUI is also running


def _data_base(data_path: str) -> str:
    base = os.path.basename(os.path.abspath(data_path))
    for ext in (".cube.npy", ".prz", ".npz", ".npy", ".h5", ".hdf5"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def assistant_root(data_path: str) -> str:
    d = os.path.dirname(os.path.abspath(data_path))
    return os.path.join(d, _data_base(data_path) + "_assistant")


def new_output_dir(data_path: str, analysis: str) -> str:
    """Create and return a fresh, zero-padded, non-overwriting dir
    `<data>_assistant/<analysis>_NNN/`."""
    root = assistant_root(data_path)
    os.makedirs(root, exist_ok=True)
    i = 1
    while True:
        cand = os.path.join(root, f"{analysis}_{i:03d}")
        if not os.path.exists(cand):
            os.makedirs(cand)
            return cand
        i += 1


def save_fig(fig, outdir: str, name: str, dpi: int = 150) -> str:
    if not name.lower().endswith((".png", ".pdf")):
        name += ".png"
    p = os.path.join(outdir, name)
    try:
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    except Exception:
        fig.savefig(p, dpi=dpi)
    return p


def save_array_png(arr, outdir: str, name: str, cmap: str = "inferno",
                   vmax: float = None) -> str:
    import numpy as np
    import matplotlib.pyplot as plt
    if not name.lower().endswith(".png"):
        name += ".png"
    p = os.path.join(outdir, name)
    a = np.asarray(arr, dtype=float)
    if vmax:
        a = np.clip(a / float(vmax), 0.0, 1.0)
    plt.imsave(p, a, cmap=cmap)
    return p


def write_summary(outdir: str, text: str, name: str = "summary.txt") -> str:
    p = os.path.join(outdir, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def gui_autosave(app, analysis: str, fig, name: str = "figure",
                 summary: str = None):
    """Best-effort: save a GUI panel's matplotlib figure next to the loaded
    data, in a sequential `<data>_assistant/<analysis>_NNN/` folder.

    Silent no-op if no data path is known or saving fails — never disturbs
    the GUI.  Returns the output dir (or None).
    """
    try:
        # `app` may be the root App, a panel, or any Tk widget — find the
        # object that owns a `pre` panel with get_loaded_path().
        path = None
        candidates = []
        if app is not None:
            candidates.append(app)
            tl = getattr(app, "winfo_toplevel", None)
            if callable(tl):
                try:
                    candidates.append(tl())
                except Exception:
                    pass
            candidates.append(getattr(app, "app", None))
        for obj in candidates:
            pre = getattr(obj, "pre", None)
            if pre is not None and hasattr(pre, "get_loaded_path"):
                try:
                    path = pre.get_loaded_path()
                except Exception:
                    path = None
                if path:
                    break
        if not path:
            return None
        d = new_output_dir(path, analysis)
        save_fig(fig, d, name)
        if summary:
            write_summary(d, summary)
        return d
    except Exception:
        return None
