"""midrun_class_map.py — quick inference from a mid-training checkpoint to
preview the class map without full eval.

Usage:
    python midrun_class_map.py <ckpt_path> <sample> <out_png>

Loads the checkpoint, runs inference at half-mask (matches the half-mask
sweep), renders class map alongside the Lothar reference for comparison.
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SAMPLES, LoadPRZ
from dino_sr_contrastive_model import (
    load_contrastive_checkpoint, PolarTransform, PolarMaskLeft,
)

REF = r"D:\DINOSR\Lothar\Chapter3Lothar\classmap_e40.png"


def infer(ckpt_path: str, sample: str, device, polar_size=192,
           polar_mask_cols=45, center_crop_size=140,
           com_centering: bool = True, mask_r_for_com: int = 15,
           com_search_radius_factor: float = 2.0):
    cfg = SAMPLES[sample]
    ds = LoadPRZ(cfg["path"], resize=polar_size, vmax=cfg["vmax"])
    model, eval_temp, _, _ = load_contrastive_checkpoint(ckpt_path, device=device)
    model.eval()
    ops = [T.CenterCrop(center_crop_size)]
    if com_centering:
        from dino_sr_ablation import CenterOnCOM
        ops.append(CenterOnCOM(
            search_radius=int(com_search_radius_factor * mask_r_for_com)))
    ops += [
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        PolarTransform(output_size=polar_size),
        PolarMaskLeft(k_cols=polar_mask_cols),
    ]
    polar_pre = T.Compose(ops)
    K = model.prototypes.prototypes.shape[0]
    N = len(ds)
    assigns = np.zeros(N, dtype=np.int64)
    bs = 256
    print(f"  inferring {N} patterns at batch={bs}, K={K}")
    with torch.no_grad():
        for i in range(0, N, bs):
            j = min(i + bs, N)
            batch = torch.stack([ds[k] for k in range(i, j)]).to(device).float()
            x_polar = polar_pre(batch)
            f = model.teacher_encoder(x_polar)
            proj = model.teacher_projector(f)
            logits = model.prototypes(proj)
            assigns[i:j] = logits.argmax(dim=-1).cpu().numpy()
    return assigns, K, ds, cfg


def main(ckpt_path: str, sample: str, out_png: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ckpt {ckpt_path}")
    assigns, K, _, cfg = infer(ckpt_path, sample, device)
    Nx, Ny = cfg["scan_shape"]
    class_map = assigns.reshape(Nx, Ny)
    K_act = int(np.unique(assigns).size)
    counts = np.bincount(assigns, minlength=K)
    print(f"K_active = {K_act}/{K}")
    print(f"class fractions: {(counts / counts.sum()).round(3).tolist()}")

    base = plt.get_cmap("tab10").colors[:K]
    cmap = ListedColormap(base, name=f"K{K}")
    norm = BoundaryNorm(np.arange(K + 1) - 0.5, K)

    fig, axes = plt.subplots(2, 1, figsize=(13, 5),
                              gridspec_kw={"hspace": 0.25})

    ref_img = plt.imread(REF)
    axes[0].imshow(ref_img, aspect="equal", interpolation="nearest")
    axes[0].set_title(
        "Reference (Lothar) — K=7 active, clean strata, crisp interfaces",
        fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    im = axes[1].imshow(class_map, cmap=cmap, norm=norm,
                         aspect="equal", interpolation="nearest")
    label = f"midrun  ckpt={os.path.basename(ckpt_path)}  K={K} K_act={K_act}"
    axes[1].set_title(f"Ours: {label}", fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    cb = fig.colorbar(im, ax=axes[1], fraction=0.022, pad=0.01,
                       ticks=range(K))
    cb.set_label("class id", fontsize=9)

    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
