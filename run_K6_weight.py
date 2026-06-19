"""
run_K6_weight.py — Train K=6 + weight-loss (γ=1) for EuInAs and Na007b,
output full eval report (class map, class avgs, UMAP, GradCAM, IG, etc.).

Configuration: copy of `sweep_polar_centroid` (the Na007b winner) with two
overrides:
    num_prototypes      = 6   (already 6 in winner; no change)
    conf_weight_gamma   = 1.0 (override; winner used 0.0)

Per user (2026-04-25): "I have some salt-and-pepper mostly at the background
in K=10+weight. Let's do EuInAs and Na007b with k=6 + weight. Output results
in full detail including gradcam and then lets discuss."

Output:
    runs/EuInAs_B100/winner_K6_weight/best.pth + eval/...
    runs/Na007b/winner_K6_weight/best.pth + eval/...
"""
from __future__ import annotations
import os, sys, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from run_contrastive import run_config, evaluate_and_report


# Winner config (sweep_polar_centroid) with conf_weight_gamma=1.0 override.
WINNER_K6_WEIGHT_KWARGS = dict(
    epochs=50, seed=42, batch_size=128,
    lr=3e-4, weight_decay=1e-6,
    num_prototypes=6,
    t0=0.04, tfin=0.07,
    warmup_epochs=20, ramp_epochs=10,
    entropy_gate=False,
    projection_dim=128, projection_hidden=256,
    theta_shift_range=None,
    theta_shift_range_student=192, theta_shift_range_teacher=16,
    center_mask_radius=None,   # use SAMPLES default per sample
    center_crop_size=140,
    vmax=None,                 # use SAMPLES default per sample
    polar_size=192, polar_mask_cols=30,
    pipeline="polar",
    centroid_lambda=0.05, centroid_margin=0.3,
    conf_weight_gamma=1.0,     # <-- override (winner = 0.0)
    entropy_gate_override=None,
    lam_spatial=0.0,
    architecture="resnet", n_layers=1,
)

LABEL = "winner_K6_weight"


def run_one(sample: str, device) -> None:
    outdir = os.path.join("runs", sample, LABEL)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample} -> {outdir}", flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **WINNER_K6_WEIGHT_KWARGS)
        print(f"[{datetime.now():%H:%M:%S}] train done ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    else:
        print(f"[skip train] {outdir} (best.pth exists)", flush=True)
    # Run full eval (auto includes GradCAM + IG via run_contrastive.evaluate_and_report)
    if not os.path.exists(sentinel_eval):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {sample}", flush=True)
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    else:
        print(f"[skip eval] {outdir}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[K6+weight] device={device}", flush=True)
    for sample in ("EuInAs_B100", "Na007b"):
        run_one(sample, device)
    print("\n[K6+weight] all done", flush=True)


if __name__ == "__main__":
    main()
