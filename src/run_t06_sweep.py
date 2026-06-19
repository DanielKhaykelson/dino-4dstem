"""
run_t06_sweep.py — T_fin = 0.06 check.

Tests whether a slightly sharper teacher (tau_t: 0.04 -> 0.06 instead of
0.04 -> 0.07) changes the story on Na007b and EuInAs across the three
interesting variants: baseline, + weight, + spatial.

Six runs total, sequential on the GPU:

    Na007b:  {winner_t06, winner_t06_weight, winner_t06_spatial}
    EuInAs:  {winner_t06, winner_t06_weight, winner_t06_spatial}

Each run:
  - Full train + eval + gradcam + IG + class averages
  - compare_maps against the sample's existing baseline winner (T_fin=0.07)

Rationale:
  The user suggested testing T_fin=0.06 to see if a tighter stable band
  (still above the 0.04 collapse edge but below the 0.07 near-divergence
  ceiling) would sharpen the class-3 boundary mixing on EuInAs without
  regressing the already-clean Na007b. Also tests whether the T_fin tweak
  combines well with the other add-ons.
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


# (sample, config_name, gamma, lam_spatial, winner_folder_for_comparison)
RUNS = [
    ("Na007b",      "winner_t06",         0.0, 0.0, "sweep_polar_centroid"),
    ("Na007b",      "winner_t06_weight",  1.0, 0.0, "sweep_polar_centroid"),
    ("Na007b",      "winner_t06_spatial", 0.0, 0.1, "sweep_polar_centroid"),
    ("EuInAs_B100", "winner_t06",         0.0, 0.0, "winner_polar_centroid"),
    ("EuInAs_B100", "winner_t06_weight",  1.0, 0.0, "winner_polar_centroid"),
    ("EuInAs_B100", "winner_t06_spatial", 0.0, 0.1, "winner_polar_centroid"),
]

T_FIN = 0.06  # the whole point of this sweep


def run_one(sample, config_name, gamma, lam_spatial, winner_folder, device):
    from run_contrastive import run_config, evaluate_and_report

    outdir = os.path.join("runs", sample, config_name)
    metrics_path = os.path.join(outdir, "eval", "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[skip] {sample}/{config_name}: already evaluated", flush=True)
    else:
        os.makedirs(outdir, exist_ok=True)
        print(f"\n{'=' * 72}", flush=True)
        print(f"[{datetime.now():%H:%M:%S}] START {sample}/{config_name}  "
              f"(tau_fin={T_FIN}, gamma={gamma}, lam_spatial={lam_spatial})",
              flush=True)
        print('=' * 72, flush=True)
        t0 = time.perf_counter()
        try:
            run_config(
                "c", sample=sample, epochs=50, seed=42, batch_size=128,
                lr=3e-4, weight_decay=1e-6, num_prototypes=10,
                t0=0.04, tfin=T_FIN,
                warmup_epochs=20, ramp_epochs=10,
                entropy_gate=False,
                projection_dim=128, projection_hidden=256,
                theta_shift_range=None,
                theta_shift_range_student=None, theta_shift_range_teacher=None,
                center_mask_radius=None, center_crop_size=140,
                vmax=None, polar_size=192, polar_mask_cols=30,
                pipeline="polar", centroid_lambda=0.05, centroid_margin=0.3,
                conf_weight_gamma=gamma,
                entropy_gate_override=False,
                lam_spatial=lam_spatial,
                outdir=outdir, device=device,
            )
            evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
            print(f"[{datetime.now():%H:%M:%S}] DONE in "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        except Exception as exc:
            print(f"[fail] {sample}/{config_name}: {exc!r}", flush=True)
            traceback.print_exc()
            return

    try:
        from compare_maps import compare
        compare(sample, old_config=winner_folder, new_config=config_name)
    except Exception as exc:
        print(f"[compare fail] {sample}/{config_name}: {exc!r}", flush=True)
        traceback.print_exc()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[t06_sweep] device = {device}", flush=True)
    for row in RUNS:
        run_one(*row, device)
    print(f"\n[{datetime.now():%H:%M:%S}] t06 sweep done.", flush=True)


if __name__ == "__main__":
    main()
