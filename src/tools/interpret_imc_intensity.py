"""Corrected intensity ablation: normalise the POST-BEAM scattered energy
to a constant per frame (so 'how much it scatters' is removed while the
spot arrangement is kept), then re-infer.  ARI~1 -> not intensity-driven;
ARI drops -> intensity/thickness matters.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RUN = r"runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60"
CUBE = r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"
VMAX = 5.0
OUT = os.path.join(RUN, "_interpretability")
MASK_R, MASK_COLS, CCROP, COM = 20, 40, 120, False

# beam-fraction mask for the 192-px cart pattern
_H = 192
_cy = (_H - 1) / 2.0
_yy, _xx = np.indices((_H, _H))
_r = np.sqrt((_yy - _cy) ** 2 + (_xx - _cy) ** 2)
_postbeam = _r >= 22.0          # outside the direct beam


class _Wrap:
    def __init__(self, base, fn):
        self.base = base; self.fn = fn
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        return self.fn(self.base[i])


def _scattered_norm(x):
    """Scale frame so its post-beam total = 1 (removes overall scattered
    amount; keeps the relative spot pattern)."""
    a = x[0].numpy().astype(np.float32)
    s = float(a[_postbeam].sum())
    if s > 1e-6:
        a = a / s * 100.0
    return torch.from_numpy(a).unsqueeze(0)


def main():
    from data import LoadPRZ
    from dino_sr_contrastive_model import load_contrastive_checkpoint
    from contrastive_eval import infer_scan
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = load_contrastive_checkpoint(
        os.path.join(RUN, "best.pth"), device=dev)
    model.eval()
    base = LoadPRZ(CUBE, resize=192, vmax=VMAX)
    orig = np.load(os.path.join(RUN, "eval", "inference.npz"),
                   allow_pickle=True)["assigns"].astype(int)
    inf = infer_scan(model, _Wrap(base, _scattered_norm), dev,
                     dense_remap=True, polar_size=192,
                     polar_mask_cols=MASK_COLS, center_crop_size=CCROP,
                     com_centering=COM, center_mask_radius=MASK_R,
                     eval_temp=0.06, batch_size=128)
    a = np.asarray(inf["assigns"]).astype(int)
    ari = float(adjusted_rand_score(orig, a))
    nmi = float(normalized_mutual_info_score(orig, a))
    print(f"[scattered_norm] ARI={ari:.3f} NMI={nmi:.3f} "
          f"K={len(np.unique(a))}", flush=True)
    p = os.path.join(OUT, "test4_scattered_norm.json")
    json.dump(dict(scattered_norm=dict(ARI_vs_orig=round(ari, 4),
              NMI_vs_orig=round(nmi, 4), K_active=int(len(np.unique(a))))),
              open(p, "w"), indent=2)


if __name__ == "__main__":
    main()
