"""run_transfer_na007b_to_na007a.py -- transfer-only eval. Use the
Na007b_K6 model (trained on Na007b) and apply it (inference only, no
retraining) to Na007a. Demonstrates the "train once, transfer" claim.

Output:
    runs/_paper_master/Na007b_K6/transfer/Na007a/
        eval/
            inference.npz
            fig_class_map_paper.png
            class_averages/p{c}.png
            class_examples_200/p{c}/
            paper_attribution/    (GradCAM/IG)
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys, json, time
from datetime import datetime

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

from data import SAMPLES, LoadPRZ
from dino_sr_contrastive_model import load_contrastive_checkpoint
from contrastive_eval import infer_scan

SOURCE_RUN = os.path.join("runs", "_paper_master", "Na007b_K6")
TARGET_SAMPLE = "Na007a"
OUT_DIR = os.path.join(SOURCE_RUN, "transfer", TARGET_SAMPLE)

# match the original Na007b_K6 training settings (vmax=2 from registry)
H = 192
MASK_R = 15
CENTER_CROP = 140
N_TOP_AVG = 300
N_EXAMPLES = 200


def _render_class_outputs(soft_probs, assigns, ds, eval_dir, sample_label,
                            vmax, n_examples=N_EXAMPLES):
    cart_pre = T.Compose([
        T.CenterCrop(CENTER_CROP),
        T.Resize(H, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    cy = cx = H / 2.0
    yy, xx = np.ogrid[:H, :H]
    bm = ((yy - cy) ** 2 + (xx - cx) ** 2) > MASK_R ** 2
    avg_dir = os.path.join(eval_dir, "class_averages")
    ex_root = os.path.join(eval_dir, "class_examples_200")
    os.makedirs(avg_dir, exist_ok=True)
    os.makedirs(ex_root, exist_ok=True)
    K = soft_probs.shape[1]
    counts = np.bincount(assigns, minlength=K)
    for c in range(K):
        idx = np.where(assigns == c)[0]
        if idx.size == 0:
            print(f"  p{c}: empty", flush=True); continue
        scores = soft_probs[idx, c]
        top_avg = idx[np.argsort(-scores)[:min(N_TOP_AVG, len(idx))]]
        patterns = np.stack([ds.get_raw(int(i)) for i in top_avg], 0).astype(np.float32)
        w = soft_probs[top_avg, c].astype(np.float32)
        wavg = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
        wavg_norm = np.clip(wavg / float(vmax), 0.0, 1.0)
        x = torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0).float()
        x = F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)
        x_cart = cart_pre(x)[0, 0].cpu().numpy()
        ref = (x_cart * bm).flatten(); ref = ref[ref > 0]
        if ref.size:
            lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
            disp = np.log1p(np.clip(x_cart, lo, hi) - lo) * bm
        else:
            disp = x_cart * bm
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
        ax.set_title(f"{sample_label}  p{c}  N={int(counts[c])}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(avg_dir, f"p{c}.png"), dpi=200,
                     bbox_inches="tight", facecolor="white")
        plt.close(fig)
        n_ex = min(n_examples, len(idx))
        top_ex = idx[np.argsort(-scores)[:n_ex]]
        d = os.path.join(ex_root, f"p{c}")
        os.makedirs(d, exist_ok=True)
        for old in os.listdir(d):
            try: os.remove(os.path.join(d, old))
            except Exception: pass
        for rank, gi in enumerate(top_ex):
            gi = int(gi)
            raw = ds.get_raw(gi).astype(np.float32)
            raw_norm = np.clip(raw / float(vmax), 0.0, 1.0)
            x = torch.from_numpy(raw_norm).unsqueeze(0).unsqueeze(0).float()
            x = F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)
            x_cart = cart_pre(x)[0, 0].cpu().numpy()
            ref = (x_cart * bm).flatten(); ref = ref[ref > 0]
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
            fig.savefig(os.path.join(d,
                f"rank{rank:03d}_i{gi:06d}_p{float(soft_probs[gi, c]):.2f}.png"),
                         dpi=130, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        print(f"  p{c}: N={counts[c]}  avg + {n_ex} examples", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(SOURCE_RUN, "best.pth")
    if not os.path.exists(ckpt):
        sys.exit(f"missing checkpoint: {ckpt}")
    cfg = SAMPLES[TARGET_SAMPLE]
    vmax = cfg["vmax"]
    print(f"[transfer] {SOURCE_RUN}  ->  {TARGET_SAMPLE}  (vmax={vmax})",
          flush=True)

    eval_dir = os.path.join(OUT_DIR, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    t0 = time.perf_counter()
    model, _, _, _ = load_contrastive_checkpoint(ckpt, device=device)
    model.eval()
    ds = LoadPRZ(cfg["path"], resize=H, vmax=vmax)
    print(f"[transfer] N={len(ds)}  scan={cfg['scan_shape']}", flush=True)
    inf = infer_scan(model, ds, device, dense_remap=False,
                      polar_size=192, polar_mask_cols=45,
                      center_crop_size=140,
                      com_centering=True, center_mask_radius=15,
                      eval_temp=0.06, batch_size=128)
    np.savez(os.path.join(eval_dir, "inference.npz"),
              soft_probs=inf["soft_probs"], assigns=inf["assigns"],
              embeds=inf["embeds"])
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w") as f:
        json.dump({"cfg": {
            "center_mask_radius": 15, "polar_mask_cols": 45,
            "polar_size": 192, "center_crop_size": 140,
            "vmax": vmax,
            "source_model": SOURCE_RUN,
            "target_sample": TARGET_SAMPLE,
        }}, f)
    print(f"[transfer] inference done in {time.perf_counter()-t0:.1f}s",
          flush=True)

    print(f"[transfer] rendering class map (paper)...", flush=True)
    try:
        import viz_paper_outputs
        viz_paper_outputs.render_class_map(OUT_DIR, TARGET_SAMPLE)
    except Exception as e:
        print(f"  [warn] class-map: {e!r}", flush=True)

    print(f"[transfer] rendering class averages + 200 examples...", flush=True)
    _render_class_outputs(inf["soft_probs"], inf["assigns"], ds, eval_dir,
                            sample_label=f"Na007a (transfer from Na007b_K6)",
                            vmax=vmax)

    print(f"[transfer] paper attribution (GradCAM/IG)...", flush=True)
    try:
        import viz_paper_attribution
        viz_paper_attribution.run(OUT_DIR, TARGET_SAMPLE,
                                    n_samples_per_proto=3,
                                    attribution_sigma=2.0,
                                    ig_steps=50,
                                    device=device)
    except Exception as e:
        print(f"  [warn] paper-attribution: {e!r}", flush=True)

    print(f"[transfer] done. output: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
