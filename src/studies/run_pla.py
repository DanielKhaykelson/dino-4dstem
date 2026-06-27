"""
run_pla.py — train DINO4DSTEM on the PLA (polylactic acid) NBED-005 dataset.

Uses the canonical winner recipe (`winner_polar_centroid`):
  - polar pipeline, circular-theta Conv2d, asymmetric theta-roll
  - num_prototypes=10, centroid_lambda=0.05, centroid_margin=0.3
  - T0=0.04 -> Tfin=0.07 (20-epoch warmup + 10-epoch ramp)
  - polar_mask_cols=30 (inner-column beam mask)
  - conf_weight_gamma=0 (no confidence weighting — canonical default)
  - no spatial loss

Writes:
  runs/PLA/winner_polar_centroid/
    best.pth, report.md, run_summary.json, training_log.csv, eval/*
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pla] device = {device}", flush=True)

    from run_contrastive import run_config, evaluate_and_report

    sample = "PLA"
    label = "winner_polar_centroid"
    outdir = os.path.join("runs", sample, label)
    metrics_path = os.path.join(outdir, "eval", "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[skip] {sample}/{label}: already evaluated", flush=True)
        return

    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'=' * 72}", flush=True)
    print(f"[{datetime.now():%H:%M:%S}] START {sample}/{label}", flush=True)
    print('=' * 72, flush=True)
    t0 = time.perf_counter()
    try:
        run_config(
            "c", sample=sample, epochs=50, seed=42, batch_size=128,
            lr=3e-4, weight_decay=1e-6, num_prototypes=10,
            t0=0.04, tfin=0.07, warmup_epochs=20, ramp_epochs=10,
            entropy_gate=False,
            projection_dim=128, projection_hidden=256,
            theta_shift_range=None,
            theta_shift_range_student=None, theta_shift_range_teacher=None,
            center_mask_radius=None, center_crop_size=140,
            vmax=None, polar_size=192,
            polar_mask_cols=30,
            pipeline="polar", centroid_lambda=0.05, centroid_margin=0.3,
            conf_weight_gamma=0.0,
            entropy_gate_override=False,
            lam_spatial=0.0,
            outdir=outdir, device=device,
        )
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] DONE in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    except Exception as exc:
        print(f"[fail] {sample}/{label}: {exc!r}", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
