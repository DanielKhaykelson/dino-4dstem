"""render_per_family_averages.py -- run inference of each per-family model
on its OWN training data, save inference.npz + class averages so the user
can pick which prototype index is the line phase.

Outputs (per family):
    runs/_per_family/train_<F>/eval/inference.npz
    runs/_per_family/train_<F>/eval/class_averages/p{c}.png
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from data import SAMPLES, LoadPRZMulti
from dino_sr_contrastive_model import load_contrastive_checkpoint
from contrastive_eval import infer_scan
import viz_paper_outputs

OUT_ROOT = os.path.join("runs", "_per_family")
FAMILIES = {
    "NaPHI":   ["NaPHI_Nadja_SI003", "NaPHI_Nadja_SI004"],
    "MgNaPHI": ["MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI011"],
}


def _multi_get_raw(multi_ds, idx):
    """LoadPRZMulti exposes get_raw via component dispatch. Wrap so
    viz_paper_outputs.render_class_averages_and_examples can use it."""
    return multi_ds.get_raw(idx)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fam, samples in FAMILIES.items():
        train_dir = os.path.join(OUT_ROOT, f"train_{fam}")
        ckpt = os.path.join(train_dir, "best.pth")
        if not os.path.exists(ckpt):
            print(f"[skip] {fam}: no best.pth", flush=True); continue
        eval_dir = os.path.join(train_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        print(f"\n[render] family={fam}  ckpt={ckpt}", flush=True)
        model, _, _, _ = load_contrastive_checkpoint(ckpt, device=device)
        model.eval()

        paths = [SAMPLES[s]["path"] for s in samples]
        # vmax=2 (NaPHI/MgNaPHI default in registry)
        ds = LoadPRZMulti(paths, resize=192, vmax=2)
        print(f"[render] {fam} dataset N={len(ds)}", flush=True)

        # Inference on training data (reads model's own clusters)
        inf = infer_scan(model, ds, device, dense_remap=False,
                          polar_size=192, polar_mask_cols=45,
                          center_crop_size=140,
                          com_centering=True, center_mask_radius=15,
                          eval_temp=0.06, batch_size=128)
        np.savez(os.path.join(eval_dir, "inference.npz"),
                  soft_probs=inf["soft_probs"],
                  assigns=inf["assigns"],
                  embeds=inf["embeds"])
        # minimal run_summary for viz_paper_outputs
        with open(os.path.join(train_dir, "run_summary.json"), "w") as f:
            json.dump({"cfg": {
                "center_mask_radius": 15, "polar_mask_cols": 45,
                "polar_size": 192, "center_crop_size": 140,
            }}, f)

        # Build a "fake" SAMPLES entry for the multi-cube dataset so
        # viz_paper_outputs can read scan_shape (it picks one but won't be
        # used for the class avg / examples since we use ds.get_raw).
        sample_key = f"PERFAM_{fam}"
        SAMPLES[sample_key] = {
            "paths": paths,
            "vmax": 2,
            "scan_shape": (100, 100),  # nominal; not used in class avg
            "center_mask_radius": 15,
            "approved_label": None,
            "is_multi": True,
        }

        # Render class averages + 100 examples per class -- patched
        # render_class_averages_and_examples uses LoadPRZ on cfg.path; we
        # need to bypass that for the multi-cube case. Inline reimplementation:
        from torchvision.transforms import v2 as T
        from torchvision.transforms import InterpolationMode
        import torch.nn.functional as F
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        avg_dir = os.path.join(eval_dir, "class_averages")
        os.makedirs(avg_dir, exist_ok=True)
        H = 192
        cart_pre = T.Compose([
            T.CenterCrop(140),
            T.Resize(H, interpolation=InterpolationMode.BILINEAR, antialias=True),
        ])
        # beam mask in display space: mask_r=15 in Cartesian (full H=192)
        # cropped to 140 then resized back to 192 -> mask r ~15px
        cy = cx = H / 2.0
        yy, xx = np.ogrid[:H, :H]
        bm = ((yy - cy) ** 2 + (xx - cx) ** 2) > 15 ** 2

        K = inf["soft_probs"].shape[1]
        soft_probs = inf["soft_probs"]
        assigns = inf["assigns"]
        counts = np.bincount(assigns, minlength=K)
        for c in range(K):
            idx = np.where(assigns == c)[0]
            if idx.size == 0:
                print(f"  p{c}: empty", flush=True); continue
            scores = soft_probs[idx, c]
            top = idx[np.argsort(-scores)[:min(300, len(idx))]]
            patterns = np.stack([ds.get_raw(int(i)) for i in top], 0).astype(np.float32)
            w = soft_probs[top, c].astype(np.float32)
            wavg = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
            wavg_norm = np.clip(wavg / 2.0, 0.0, 1.0)
            x_full = torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0).float()
            x_full = F.interpolate(x_full, size=(H, H), mode="bilinear",
                                    align_corners=False)
            x_cart = cart_pre(x_full)[0, 0].cpu().numpy()
            ref = (x_cart * bm).flatten()
            ref = ref[ref > 0]
            if ref.size:
                lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
                disp = np.log1p(np.clip(x_cart, lo, hi) - lo) * bm
            else:
                disp = x_cart * bm
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
            ax.set_title(f"{fam}  p{c}  N={int(counts[c])}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            out = os.path.join(avg_dir, f"p{c}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  p{c}: N={counts[c]}  -> {out}", flush=True)

    print(f"\n[done] class averages in runs/_per_family/train_*/eval/class_averages/",
          flush=True)


if __name__ == "__main__":
    main()
