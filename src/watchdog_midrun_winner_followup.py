"""watchdog_midrun_winner_followup.py -- watchdog for runs/_winner_followup/.
Same logic as the cluster1d watchdog but a different root + sentinel.
"""
from __future__ import annotations
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

SWEEP_ROOT = os.path.join("runs", "_winner_followup")
POLL_SEC = 30
STOP_FLAG = os.path.join(SWEEP_ROOT, "_done.flag")

# label -> sample mapping (must match run_winner_followup.py CONFIGS)
LABEL_TO_SAMPLE = {
    "Na007b_K6_50ep":  "Na007b",
    "EuInAs_K6_50ep":  "EuInAs_B100",
    "Na007b_K10_30ep": "Na007b",
}


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
    axes[1].set_title(f"Ours: {label}  [{os.path.basename(os.path.dirname(ckpt_path))}]",
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
    print(f"[watchdog winner-followup] device={device}  "
          f"sweep_root={SWEEP_ROOT}  poll={POLL_SEC}s", flush=True)
    rendered_total = 0
    while True:
        if os.path.exists(STOP_FLAG):
            print(f"[watchdog winner-followup] stop flag found, exiting "
                  f"(rendered {rendered_total})", flush=True)
            return
        try:
            pending = discover_pending()
        except Exception as e:
            print(f"[watchdog winner-followup] discover error: {e!r}", flush=True)
            pending = []
        for ckpt_path, sample, out_png in pending:
            try:
                render_one(ckpt_path, sample, out_png, device)
                rendered_total += 1
            except Exception as e:
                print(f"[watchdog winner-followup] render error {ckpt_path}: {e!r}",
                      flush=True)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
