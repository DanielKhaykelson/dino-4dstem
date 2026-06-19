"""
run_spatial_followups.py — after EuInAs 3-way completes, runs:

  1. Na007b with lam_spatial=0.1 (user-requested)
  2. IMC_150nm_SI5 with lam_spatial=0.1 (exploratory — crystalline-rich sample)

Each run:
  - Uses the current winner config + lam_spatial=0.1 (everything else unchanged)
  - Full eval + GradCAM + IG + class averages (automatic via evaluate_and_report)
  - compare_maps against that sample's current winner_polar_centroid / sweep_polar_centroid

Waits for `runs/EUINAS_3WAY_REPORT.md` to exist before starting (written at
the end of run_euinas_3way.py). Polls every 60s.
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


def wait_for_3way(report_path: str, poll_s: int = 60, max_wait_s: int = 4 * 3600):
    """Block until run_euinas_3way.py has written its report."""
    t0 = time.perf_counter()
    while not os.path.exists(report_path):
        elapsed = time.perf_counter() - t0
        if elapsed > max_wait_s:
            print(f"[wait] gave up after {elapsed:.0f}s. Proceeding anyway.",
                  flush=True)
            return False
        print(f"[wait] {datetime.now():%H:%M:%S}  waiting for "
              f"EUINAS_3WAY_REPORT.md  (elapsed {elapsed:.0f}s)",
              flush=True)
        time.sleep(poll_s)
    print(f"[wait] 3-way report arrived. Proceeding.", flush=True)
    return True


def run_one(sample: str, config_folder: str, winner_folder: str,
             lam_spatial: float, device) -> None:
    from run_contrastive import run_config, evaluate_and_report

    outdir = os.path.join("runs", sample, config_folder)
    if os.path.exists(os.path.join(outdir, "best.pth")):
        print(f"[skip] {sample}/{config_folder}: best.pth exists", flush=True)
    else:
        print(f"\n{'=' * 72}\n[{datetime.now():%H:%M:%S}] "
              f"START {sample}/{config_folder}  (lam_spatial={lam_spatial})\n"
              f"{'=' * 72}", flush=True)
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
                conf_weight_gamma=0.0, entropy_gate_override=False,
                lam_spatial=lam_spatial,
                outdir=outdir, device=device,
            )
            evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
            print(f"[{datetime.now():%H:%M:%S}] DONE {sample}/{config_folder}"
                  f" in {time.perf_counter() - t0:.0f}s",
                  flush=True)
        except Exception as exc:
            print(f"[fail] {sample}/{config_folder}: {exc!r}", flush=True)
            traceback.print_exc()
            return

    # Compare against the sample's existing winner.
    try:
        from compare_maps import compare
        compare(sample, old_config=winner_folder, new_config=config_folder)
    except Exception as exc:
        print(f"[compare fail] {sample}/{config_folder}: {exc!r}", flush=True)
        traceback.print_exc()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[spatial_followups] device = {device}", flush=True)

    report = os.path.join("runs", "EUINAS_3WAY_REPORT.md")
    wait_for_3way(report)

    # Na007b + lam_spatial = 0.1. Winner for Na007b is sweep_polar_centroid.
    run_one(sample="Na007b",
             config_folder="winner_plus_spatial",
             winner_folder="sweep_polar_centroid",
             lam_spatial=0.1, device=device)

    # IMC_150nm_SI5 + lam_spatial = 0.1. Winner is winner_polar_centroid.
    run_one(sample="IMC_150nm_SI5",
             config_folder="winner_plus_spatial",
             winner_folder="winner_polar_centroid",
             lam_spatial=0.1, device=device)

    print(f"\n[{datetime.now():%H:%M:%S}] all spatial followups done.", flush=True)


if __name__ == "__main__":
    main()
