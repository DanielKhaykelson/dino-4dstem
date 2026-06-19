"""
run_final_queue.py — last two runs after weight_generalization finishes.

Waits for BOTH weight_generalization outputs to be fully evaluated
(metrics.json exists — that is written at the end of evaluate_and_report,
so it is a true end-of-run sentinel unlike best.pth which updates every
improving epoch). Then runs:

  1. Na006a + conf_weight_gamma = 1.0   (generalize weight_only to this sample too)
  2. IMC_150nm_SI5 + lam_spatial = 0.1  (finish the spatial-regularizer data point)
"""
from __future__ import annotations

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


SENTINELS = [
    os.path.join("runs", "Na007b",         "winner_weight_only", "eval", "metrics.json"),
    os.path.join("runs", "IMC_150nm_SI5",  "winner_weight_only", "eval", "metrics.json"),
]


def wait_for_metrics(poll_s: int = 60, max_wait_s: int = 4 * 3600):
    t0 = time.perf_counter()
    while not all(os.path.exists(p) for p in SENTINELS):
        elapsed = time.perf_counter() - t0
        if elapsed > max_wait_s:
            print(f"[wait] gave up after {elapsed:.0f}s.", flush=True)
            return False
        missing = [p for p in SENTINELS if not os.path.exists(p)]
        print(f"[wait] {datetime.now():%H:%M:%S}  still missing "
              f"{len(missing)} sentinel(s)  (elapsed {elapsed:.0f}s)",
              flush=True)
        time.sleep(poll_s)
    print(f"[wait] all weight_generalization sentinels present.", flush=True)
    return True


def run_one(sample: str, config_folder: str, winner_folder: str,
             conf_weight_gamma: float, lam_spatial: float, device):
    from run_contrastive import run_config, evaluate_and_report

    outdir = os.path.join("runs", sample, config_folder)
    metrics_path = os.path.join(outdir, "eval", "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[skip] {sample}/{config_folder}: already fully evaluated",
              flush=True)
    else:
        print(f"\n{'=' * 72}", flush=True)
        print(f"[{datetime.now():%H:%M:%S}] START {sample}/{config_folder}  "
              f"(γ={conf_weight_gamma}, λ_spatial={lam_spatial})", flush=True)
        print('=' * 72, flush=True)
        os.makedirs(outdir, exist_ok=True)
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
                conf_weight_gamma=conf_weight_gamma,
                entropy_gate_override=False,
                lam_spatial=lam_spatial,
                outdir=outdir, device=device,
            )
            evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
            print(f"[{datetime.now():%H:%M:%S}] DONE in "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        except Exception as exc:
            print(f"[fail] {sample}/{config_folder}: {exc!r}", flush=True)
            traceback.print_exc()
            return

    try:
        from compare_maps import compare
        compare(sample, old_config=winner_folder, new_config=config_folder)
    except Exception as exc:
        print(f"[compare fail] {sample}/{config_folder}: {exc!r}", flush=True)
        traceback.print_exc()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[final_queue] device = {device}", flush=True)
    wait_for_metrics()

    # Na006a + conf_weight_gamma = 1.0.
    run_one(sample="Na006a",
             config_folder="winner_weight_only",
             winner_folder="winner_polar_centroid",
             conf_weight_gamma=1.0, lam_spatial=0.0, device=device)

    # IMC_150nm_SI5 + lam_spatial = 0.1 (the spatial run that got killed
    # during the earlier GPU-contention mess; re-doing on a clean GPU).
    run_one(sample="IMC_150nm_SI5",
             config_folder="winner_plus_spatial",
             winner_folder="winner_polar_centroid",
             conf_weight_gamma=0.0, lam_spatial=0.1, device=device)

    print(f"\n[{datetime.now():%H:%M:%S}] final_queue done.", flush=True)


if __name__ == "__main__":
    main()
