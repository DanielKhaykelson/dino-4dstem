"""run_K8_Na007b_weight.py — Train Na007b at K=8 with weight-loss (γ=1).

User (2026-04-25): "Na007b weights was good but underclustered a little, so I
hope maybe 8 will work enough for now." K=6+weight gave intra/inter=14.91 (down
from K=6 no-weight winner = 21.49). Try K=8 to see if extra capacity recovers
the inter-class separation while keeping the weight-induced sharpness.

Output: runs/Na007b/winner_K8_weight/
"""
from __future__ import annotations
import os, sys, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from run_contrastive import run_config, evaluate_and_report


WINNER_K8_WEIGHT_KWARGS = dict(
    epochs=50, seed=42, batch_size=128,
    lr=3e-4, weight_decay=1e-6,
    num_prototypes=8,                  # <-- override (K=6 → K=8)
    t0=0.04, tfin=0.07,
    warmup_epochs=20, ramp_epochs=10,
    entropy_gate=False,
    projection_dim=128, projection_hidden=256,
    theta_shift_range=None,
    theta_shift_range_student=192, theta_shift_range_teacher=16,
    center_mask_radius=None,
    center_crop_size=140,
    vmax=None,
    polar_size=192, polar_mask_cols=30,
    pipeline="polar",
    centroid_lambda=0.05, centroid_margin=0.3,
    conf_weight_gamma=1.0,             # <-- weight loss enabled
    entropy_gate_override=None,
    lam_spatial=0.0,
    architecture="resnet", n_layers=1,
)

LABEL = "winner_K8_weight"
SAMPLE = "Na007b"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[K8 weight Na007b] device={device}", flush=True)
    outdir = os.path.join("runs", SAMPLE, LABEL)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {SAMPLE} -> {outdir}", flush=True)
        run_config("c", sample=SAMPLE, outdir=outdir, device=device,
                   **WINNER_K8_WEIGHT_KWARGS)
        print(f"[{datetime.now():%H:%M:%S}] train done ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    else:
        print(f"[skip train] {outdir} (best.pth exists)", flush=True)
    if not os.path.exists(sentinel_eval):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {SAMPLE}", flush=True)
        evaluate_and_report("c", sample=SAMPLE, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    else:
        print(f"[skip eval] {outdir}", flush=True)
    print("\n[K8 weight Na007b] done", flush=True)


if __name__ == "__main__":
    main()
