"""watchdog_midrun_determinism.py -- watchdog for runs/_determinism_check/.
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys, time, glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from midrun_class_map import infer, REF

SWEEP_ROOT = os.path.join("runs", "_determinism_check")
POLL_SEC = 30
STOP_FLAG = os.path.join(SWEEP_ROOT, "_done.flag")

# label -> sample (label is the sample name in this layout)
LABEL_TO_SAMPLE = {"Na007b": "Na007b", "EuInAs_B100": "EuInAs_B100"}


def render_one(ckpt_path: str, sample: str, out_png: str, device):
    print(f"[{datetime.now():%H:%M:%S}] rendering {out_png}", flush=True)
    assigns, K, _, cfg = infer(ckpt_path, sample, device)
    Nx, Ny = cfg["scan_shape"]
    class_map = assigns.reshape(Nx, Ny)
    K_act = int(np.unique(assigns).size)
    base = plt.get_cmap("tab10").colors[:K]
    cmap = ListedColormap(base, name=f"K{K}")
    norm = BoundaryNorm(np.arange(K + 1) - 0.5, K)
    fig, axes = plt.subplots(2, 1, figsize=(13, 5),
                              gridspec_kw={"hspace": 0.25})
    if os.path.exists(REF):
        try:
            ref_img = plt.imread(REF)
            axes[0].imshow(ref_img, aspect="equal", interpolation="nearest")
        except Exception:
            axes[0].text(0.5, 0.5, "reference unavailable",
                          ha="center", va="center")
    axes[0].set_title(
        "Reference (Lothar) -- K=7 active, clean strata, crisp interfaces",
        fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    im = axes[1].imshow(class_map, cmap=cmap, norm=norm,
                         aspect="equal", interpolation="nearest")
    label = f"midrun  ckpt={os.path.basename(ckpt_path)}  K={K} K_act={K_act}"
    axes[1].set_title(f"Ours: {label}  [{sample}]  (deterministic)",
                       fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    cb = fig.colorbar(im, ax=axes[1], fraction=0.022, pad=0.01,
                       ticks=range(K))
    cb.set_label("class id", fontsize=9)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def discover_pending():
    pending = []
    pattern = os.path.join(SWEEP_ROOT, "*", "ckpt_ep*.pth")
    for ckpt_path in sorted(glob.glob(pattern)):
        parts = os.path.normpath(ckpt_path).split(os.sep)
        label = parts[-2]
        sample = LABEL_TO_SAMPLE.get(label)
        if sample is None:
            continue
        ep_num = os.path.basename(ckpt_path).replace("ckpt_ep", "").replace(".pth", "")
        out_png = os.path.join(os.path.dirname(ckpt_path),
                                f"midrun_ep{ep_num}.png")
        if not os.path.exists(out_png):
            pending.append((ckpt_path, sample, out_png))
    return pending


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[watchdog determinism] device={device}", flush=True)
    rendered = 0
    while True:
        if os.path.exists(STOP_FLAG):
            print(f"[watchdog determinism] stop flag found "
                  f"(rendered {rendered})", flush=True)
            return
        try:
            pending = discover_pending()
        except Exception as e:
            print(f"[watchdog determinism] discover error: {e!r}", flush=True)
            pending = []
        for ckpt_path, sample, out_png in pending:
            try:
                render_one(ckpt_path, sample, out_png, device)
                rendered += 1
            except Exception as e:
                print(f"[watchdog determinism] render error: {e!r}",
                      flush=True)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
