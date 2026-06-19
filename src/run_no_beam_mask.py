"""
run_no_beam_mask.py — EuInAs 3-way test WITHOUT the polar inner-column
(beam) mask.

Rationale from the author: on EuInAs the central beam isn't a significant
source of intensity pollution under standard acquisition conditions.
Removing the mask matters for the paper because it shows the method can
work without a downstream-specific preprocessing hack.

Three EuInAs runs, sequential on GPU:

    winner_nomask           — same as winner, polar_mask_cols=0
    winner_nomask_weight    — + confidence weighting γ=1
    winner_nomask_spatial   — + lam_spatial=0.1

Waits for post_t06 completion so there's no GPU contention. Sentinel is
the post_t06 paper-summary file.
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


POST_T06_SENTINEL = os.path.join("runs", "PAPER_SUMMARY_TABLE.json")

RUNS = [
    ("winner_nomask",         0.0, 0.0),   # label, gamma, lam_spatial
    ("winner_nomask_weight",  1.0, 0.0),
    ("winner_nomask_spatial", 0.0, 0.1),
]


def wait_for_sentinel(poll_s: int = 60, max_wait_s: int = 8 * 3600):
    t0 = time.perf_counter()
    while not os.path.exists(POST_T06_SENTINEL):
        elapsed = time.perf_counter() - t0
        if elapsed > max_wait_s:
            print(f"[wait] gave up after {elapsed:.0f}s. Proceeding anyway.",
                  flush=True)
            return False
        print(f"[wait] {datetime.now():%H:%M:%S}  waiting for post_t06 "
              f"sentinel  (elapsed {elapsed:.0f}s)", flush=True)
        time.sleep(poll_s)
    print(f"[wait] post_t06 done. Proceeding with no-beam-mask sweep.",
          flush=True)
    return True


def run_one(label: str, gamma: float, lam_spatial: float, device):
    from run_contrastive import run_config, evaluate_and_report
    from compare_maps import compare

    sample = "EuInAs_B100"
    outdir = os.path.join("runs", sample, label)
    metrics_path = os.path.join(outdir, "eval", "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[skip] {sample}/{label}: already evaluated", flush=True)
    else:
        os.makedirs(outdir, exist_ok=True)
        print(f"\n{'=' * 72}", flush=True)
        print(f"[{datetime.now():%H:%M:%S}] START {sample}/{label}  "
              f"(NO beam mask, γ={gamma}, λ_spatial={lam_spatial})",
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
                vmax=None, polar_size=192,
                polar_mask_cols=0,                    # <-- the point of the sweep
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
            print(f"[fail] {sample}/{label}: {exc!r}", flush=True)
            traceback.print_exc()
            return

    try:
        compare(sample, old_config="winner_polar_centroid", new_config=label)
    except Exception as exc:
        print(f"[compare fail] {exc!r}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[no_beam_mask] device = {device}", flush=True)
    wait_for_sentinel()
    for label, gamma, lam_spatial in RUNS:
        run_one(label, gamma, lam_spatial, device)
    print(f"\n[{datetime.now():%H:%M:%S}] no-beam-mask sweep done.", flush=True)


if __name__ == "__main__":
    main()
