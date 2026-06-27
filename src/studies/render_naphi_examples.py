"""render_naphi_examples.py -- 200 example patterns per class for a
per-family model. Cartesian display, log1p, percentile-clipped,
beam-mask r<15px.

CLI:
    python render_naphi_examples.py --family NaPHI
    python render_naphi_examples.py --family MgNaPHI

Output: runs/_per_family/<family>_combined_K6_30ep/eval/class_examples_200/p{c}/
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

from data import SAMPLES, LoadPRZMulti

FAMILY_TO_TRAIN_SAMPLES = {
    "NaPHI":   ["NaPHI_Nadja_SI003", "NaPHI_Nadja_SI004"],
    "MgNaPHI": ["MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI011"],
}
N_PER_CLASS = 200
H = 192
MASK_R = 15
CENTER_CROP = 140


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAMILY_TO_TRAIN_SAMPLES))
    args = ap.parse_args()
    SRC_DIR = os.path.join("runs", "_per_family",
                            f"{args.family}_combined_K6_30ep")
    TRAIN_SAMPLES = FAMILY_TO_TRAIN_SAMPLES[args.family]
    print(f"[examples] family={args.family}  src={SRC_DIR}", flush=True)
    inf = np.load(os.path.join(SRC_DIR, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]
    K = soft_probs.shape[1]
    counts = np.bincount(assigns, minlength=K)
    print(f"[examples] K={K}  counts={counts.tolist()}", flush=True)

    paths = [SAMPLES[s]["path"] for s in TRAIN_SAMPLES]
    ds = LoadPRZMulti(paths, resize=H, vmax=2)
    print(f"[examples] dataset N={len(ds)}", flush=True)

    cart_pre = T.Compose([
        T.CenterCrop(CENTER_CROP),
        T.Resize(H, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    cy = cx = H / 2.0
    yy, xx = np.ogrid[:H, :H]
    bm = ((yy - cy) ** 2 + (xx - cx) ** 2) > MASK_R ** 2

    out_root = os.path.join(SRC_DIR, "eval", "class_examples_200")
    os.makedirs(out_root, exist_ok=True)

    for c in range(K):
        idx = np.where(assigns == c)[0]
        if idx.size == 0:
            print(f"  p{c}: empty", flush=True); continue
        scores = soft_probs[idx, c]
        n_ex = min(N_PER_CLASS, len(idx))
        top = idx[np.argsort(-scores)[:n_ex]]
        d = os.path.join(out_root, f"p{c}")
        os.makedirs(d, exist_ok=True)
        for old in os.listdir(d):
            try: os.remove(os.path.join(d, old))
            except Exception: pass
        for rank, gi in enumerate(top):
            gi = int(gi)
            raw = ds.get_raw(gi).astype(np.float32)
            raw_norm = np.clip(raw / 2.0, 0.0, 1.0)
            x = torch.from_numpy(raw_norm).unsqueeze(0).unsqueeze(0).float()
            x = F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)
            x_cart = cart_pre(x)[0, 0].cpu().numpy()
            ref = (x_cart * bm).flatten()
            ref = ref[ref > 0]
            if ref.size:
                lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
                disp = np.log1p(np.clip(x_cart, lo, hi) - lo) * bm
            else:
                disp = x_cart * bm
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
            ax.set_title(f"p{c}  i={gi}  p={float(soft_probs[gi, c]):.2f}",
                          fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            out = os.path.join(d, f"rank{rank:03d}_i{gi:06d}_p{float(soft_probs[gi, c]):.2f}.png")
            fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        print(f"  p{c}: wrote {n_ex} examples to {d}", flush=True)

    print(f"[done] {out_root}", flush=True)


if __name__ == "__main__":
    main()
