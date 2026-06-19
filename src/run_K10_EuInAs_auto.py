"""run_K10_EuInAs_auto.py — Train EuInAs at K=10 with auto-weight.

The model starts with gamma=0 and at the end of epoch 5 inspects avg_conf:
  avg_conf > 0.85 -> ENABLE weight (gamma -> 1.0)
  avg_conf <= 0.85 -> SKIP weight (gamma stays 0.0)

Expected for EuInAs (memory: ep5 avg_conf ~ 0.96): the detector ENABLES weight
at ep5, training proceeds with gamma=1 from ep6 onward, and the final eval
should match the existing `winner_weight_only` benchmark (intra/inter ~ 24.6).

Output: runs/EuInAs_B100/winner_K10_auto/
        + auto_weight_decision.json with the decision audit.
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


WINNER_K10_AUTO_KWARGS = dict(
    epochs=50, seed=42, batch_size=128,
    lr=3e-4, weight_decay=1e-6,
    num_prototypes=10,
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
    conf_weight_gamma=0.0,                              # start off
    conf_weight_auto=dict(detect_at_epoch=5,
                          threshold=0.85,
                          gamma_target=1.0),            # auto decision
    entropy_gate_override=None,
    lam_spatial=0.0,
    architecture="resnet", n_layers=1,
)

LABEL = "winner_K10_auto"
SAMPLE = "EuInAs_B100"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[K10 auto EuInAs] device={device}", flush=True)
    outdir = os.path.join("runs", SAMPLE, LABEL)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {SAMPLE} -> {outdir}", flush=True)
        run_config("c", sample=SAMPLE, outdir=outdir, device=device,
                   **WINNER_K10_AUTO_KWARGS)
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
    decision_path = os.path.join(outdir, "auto_weight_decision.json")
    if os.path.exists(decision_path):
        print("\n[auto-weight decision]")
        with open(decision_path) as f:
            print(f.read())
    print("\n[K10 auto EuInAs] done", flush=True)


if __name__ == "__main__":
    main()
