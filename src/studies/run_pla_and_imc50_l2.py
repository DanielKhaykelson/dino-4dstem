"""
run_pla_and_imc50_l2.py — sequential queue that runs after the IMC
cross-transfer sweep finishes:

  1. PLA / winner_polar_centroid   (canonical winner on the new PLA sample)
  2. IMC_50nm_SI2 / winner_L2      (backbone ablation completeness for the paper)

Waits for the IMC cross-transfer direction-2 metrics.json sentinel so there's
no GPU contention.
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


CROSS_SENTINEL = os.path.join("runs", "IMC_50nm_SI2", "transfer_from_150nm",
                               "eval", "metrics.json")


def wait_for_sentinel(poll_s: int = 15, max_wait_s: int = 3600):
    t0 = time.perf_counter()
    while not os.path.exists(CROSS_SENTINEL):
        elapsed = time.perf_counter() - t0
        if elapsed > max_wait_s:
            print(f"[wait] gave up after {elapsed:.0f}s.", flush=True)
            return False
        print(f"[wait] {datetime.now():%H:%M:%S}  waiting for cross-transfer "
              f"sentinel  (elapsed {elapsed:.0f}s)", flush=True)
        time.sleep(poll_s)
    print(f"[wait] cross-transfer done. Proceeding.", flush=True)
    return True


def run_pla(device):
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
            vmax=None, polar_size=192, polar_mask_cols=30,
            pipeline="polar", centroid_lambda=0.05, centroid_margin=0.3,
            conf_weight_gamma=0.0, entropy_gate_override=False,
            lam_spatial=0.0,
            outdir=outdir, device=device,
        )
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] DONE PLA in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    except Exception as exc:
        print(f"[fail] {sample}/{label}: {exc!r}", flush=True)
        traceback.print_exc()


def run_imc50_l2(device):
    from run_contrastive import run_config, evaluate_and_report
    from compare_maps import compare
    sample = "IMC_50nm_SI2"
    label = "winner_L2"
    outdir = os.path.join("runs", sample, label)
    metrics_path = os.path.join(outdir, "eval", "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[skip] {sample}/{label}: already evaluated", flush=True)
        return
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'=' * 72}", flush=True)
    print(f"[{datetime.now():%H:%M:%S}] START {sample}/{label}  (L2 backbone)",
          flush=True)
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
            vmax=None, polar_size=192, polar_mask_cols=30,
            pipeline="polar", centroid_lambda=0.05, centroid_margin=0.3,
            conf_weight_gamma=0.0, entropy_gate_override=False,
            lam_spatial=0.0,
            architecture="resnet", n_layers=2,
            outdir=outdir, device=device,
        )
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] DONE IMC50/L2 in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        try:
            compare(sample, old_config="winner_polar_centroid",
                    new_config="winner_L2")
        except Exception as exc:
            print(f"  [compare] {exc!r}")
    except Exception as exc:
        print(f"[fail] {sample}/{label}: {exc!r}", flush=True)
        traceback.print_exc()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[queue] device = {device}", flush=True)
    wait_for_sentinel()
    run_pla(device)
    run_imc50_l2(device)
    print(f"\n[{datetime.now():%H:%M:%S}] queue done.", flush=True)


if __name__ == "__main__":
    main()
