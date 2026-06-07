"""interpret_imc_ablations.py — tests 2 & 4: re-infer the DINO model on
ABLATED patterns and measure how much the class map survives (ARI/NMI vs
the original).  Pins down which physical signal the clustering needs.

Ablations (applied to each cart pattern before the model's polar preproc):
  * radial_only   — replace by its azimuthal average (rings only, no
                    angular/spot info).  survives → radial-driven.
  * blur          — Gaussian blur (kills sharp Bragg spots, keeps rings).
  * perframe_norm — per-frame min-max to [0,1] instead of the global vmax,
                    removing the 'overall brightness' (thickness/scattered-
                    intensity) signal.  collapses → intensity-driven.
  * qmask_low / qmask_high — zero an inner / outer q band.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import (adjusted_rand_score,
                              normalized_mutual_info_score)

RUN = sys.argv[1] if len(sys.argv) > 1 else (
    r"runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60")
CUBE = r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"
VMAX = 5.0
OUT = os.path.join(RUN, "_interpretability")
os.makedirs(OUT, exist_ok=True)
NY = NX = 128
# preprocessing from run_summary.json cfg
MASK_R, MASK_COLS, CCROP, COM = 20, 40, 120, False


class _Wrap:
    """Wrap LoadPRZ; apply `fn` to each (1,H,W) cart tensor before model."""
    def __init__(self, base, fn):
        self.base = base; self.fn = fn
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        return self.fn(self.base[i])


def _radial_only(x):
    a = x[0].numpy()
    H, W = a.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rb = np.clip(r.astype(int), 0, min(H, W) // 2 - 1)
    nb = rb.max() + 1
    prof = np.bincount(rb.ravel(), a.ravel(), nb) / np.maximum(
        np.bincount(rb.ravel(), None, nb), 1)
    out = prof[rb]
    return torch.from_numpy(out.astype(np.float32)).unsqueeze(0)


def _blur(sig):
    from scipy.ndimage import gaussian_filter
    def f(x):
        return torch.from_numpy(
            gaussian_filter(x[0].numpy(), sigma=sig)).unsqueeze(0)
    return f


def _perframe_norm(x):
    a = x[0].numpy()
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) if hi > lo else a * 0
    return torch.from_numpy(a.astype(np.float32)).unsqueeze(0)


def _qmask(inner, outer):
    def f(x):
        a = x[0].numpy().copy()
        H, W = a.shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        yy, xx = np.indices((H, W))
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(H, W) / 2.0)
        a[(r >= inner) & (r < outer)] = 0.0
        return torch.from_numpy(a.astype(np.float32)).unsqueeze(0)
    return f


def main():
    from data import LoadPRZ
    from dino_sr_contrastive_model import load_contrastive_checkpoint
    from contrastive_eval import infer_scan
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(RUN, "best.pth")
    model, _, _, _ = load_contrastive_checkpoint(ckpt, device=dev)
    model.eval()
    base = LoadPRZ(CUBE, resize=192, vmax=VMAX)
    orig = np.load(os.path.join(RUN, "eval", "inference.npz"),
                   allow_pickle=True)["assigns"].astype(int)

    def run(ds):
        inf = infer_scan(model, ds, dev, dense_remap=True, polar_size=192,
                         polar_mask_cols=MASK_COLS, center_crop_size=CCROP,
                         com_centering=COM, center_mask_radius=MASK_R,
                         eval_temp=0.06, batch_size=128)
        return np.asarray(inf["assigns"]).astype(int)

    ablations = {
        "baseline":      lambda x: x,
        "radial_only":   _radial_only,
        "blur(s=2)":     _blur(2.0),
        "perframe_norm": _perframe_norm,
        "qmask_low":     _qmask(0.0, 0.45),
        "qmask_high":    _qmask(0.45, 1.01),
    }
    results = {}; maps = {}
    for name, fn in ablations.items():
        print(f"[ablation] {name} …", flush=True)
        a = run(_Wrap(base, fn))
        ari = float(adjusted_rand_score(orig, a))
        nmi = float(normalized_mutual_info_score(orig, a))
        kact = int(len(np.unique(a)))
        results[name] = dict(ARI_vs_orig=round(ari, 4),
                              NMI_vs_orig=round(nmi, 4), K_active=kact)
        maps[name] = a.reshape(NY, NX)
        print(f"    ARI={ari:.3f}  NMI={nmi:.3f}  K={kact}", flush=True)

    with open(os.path.join(OUT, "test2_4_ablations.json"), "w") as f:
        json.dump(results, f, indent=2)

    # montage of class maps
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4))
    K = orig.max() + 1
    cmap = ListedColormap(plt.get_cmap("tab20")(np.linspace(0, 1, max(K, 2))))
    for ax, (name, m) in zip(axes, maps.items()):
        ax.imshow(m, cmap=cmap, interpolation="nearest", aspect="equal")
        ax.set_title(f"{name}\nARI={results[name]['ARI_vs_orig']}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Ablation re-inference vs original DINO class map "
                 "(high ARI = that info wasn't needed)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "test2_4_ablation_maps.png"), dpi=150)
    plt.close(fig)
    print(f"[ablation] done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
