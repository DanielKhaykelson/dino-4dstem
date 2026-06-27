"""
run_weight_generalization.py — after spatial_followups finishes, confirms
whether conf_weight_gamma=1.0 alone (the EuInAs winner) generalizes to
Na007b and IMC_150nm_SI5.

Waits for `runs/Na007b/winner_plus_spatial/best.pth` AND
`runs/IMC_150nm_SI5/winner_plus_spatial/best.pth` to exist (signal that
spatial_followups has completed both runs).
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


SENTINEL_PATHS = [
    os.path.join("runs", "Na007b",         "winner_plus_spatial", "best.pth"),
    os.path.join("runs", "IMC_150nm_SI5",  "winner_plus_spatial", "best.pth"),
]


def wait_for_sentinels(poll_s: int = 60, max_wait_s: int = 4 * 3600):
    t0 = time.perf_counter()
    while not all(os.path.exists(p) for p in SENTINEL_PATHS):
        elapsed = time.perf_counter() - t0
        if elapsed > max_wait_s:
            print(f"[wait] gave up after {elapsed:.0f}s. Skipping missing "
                  f"sentinels: "
                  f"{[p for p in SENTINEL_PATHS if not os.path.exists(p)]}",
                  flush=True)
            return False
        missing = [p for p in SENTINEL_PATHS if not os.path.exists(p)]
        print(f"[wait] {datetime.now():%H:%M:%S}  still missing {len(missing)} "
              f"spatial sentinel(s)  (elapsed {elapsed:.0f}s)", flush=True)
        time.sleep(poll_s)
    print(f"[wait] spatial followups done. Proceeding with weight_only.",
          flush=True)
    return True


def run_one(sample: str, config_folder: str, winner_folder: str, device) -> None:
    from run_contrastive import run_config, evaluate_and_report

    outdir = os.path.join("runs", sample, config_folder)
    if os.path.exists(os.path.join(outdir, "best.pth")):
        print(f"[skip] {sample}/{config_folder}: best.pth exists", flush=True)
    else:
        print(f"\n{'=' * 72}\n[{datetime.now():%H:%M:%S}] "
              f"START {sample}/{config_folder}  (conf_weight_gamma=1.0)\n"
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
                conf_weight_gamma=1.0, entropy_gate_override=False,
                lam_spatial=0.0,
                outdir=outdir, device=device,
            )
            evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
            print(f"[{datetime.now():%H:%M:%S}] DONE {sample}/{config_folder}"
                  f" in {time.perf_counter() - t0:.0f}s", flush=True)
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
    print(f"[weight_generalization] device = {device}", flush=True)
    wait_for_sentinels()
    # Na007b winner is sweep_polar_centroid.
    run_one("Na007b", "winner_weight_only", "sweep_polar_centroid", device)
    # IMC_150nm winner is winner_polar_centroid.
    run_one("IMC_150nm_SI5", "winner_weight_only", "winner_polar_centroid",
             device)
    print(f"\n[{datetime.now():%H:%M:%S}] weight-only generalization done.",
          flush=True)


if __name__ == "__main__":
    main()
