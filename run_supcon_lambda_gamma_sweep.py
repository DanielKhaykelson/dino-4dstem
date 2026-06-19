"""run_supcon_lambda_gamma_sweep.py -- sweep SupCon weight (lambda) x DINO
confidence weighting (gamma) at K=6 with the tightened SAXS radial gate.

Motivation (per 2026-04-27 discussion):
  The newgate run at K=10, lambda=0.05, gamma=0 showed:
    - Na007b: 3 redundancy clusters in centroid cosine matrix, noisy maps
    - EuInAs: clean strata at K_act=8 (good)
  Inspecting the loss-ratio trajectory revealed SupCon already dominates at
  lambda=0.05 (52% Na007b, 62% EuInAs by ep30). DINO CE decays exponentially
  while SupCon InfoNCE has a ~ln(64) floor -- so even small lambda becomes
  dominant. The right sweep range is much smaller than originally planned.

Knobs being optimized (the two real loss knobs):
    lambda_supcon in {0.01, 0.02, 0.05}
    conf_weight_gamma in {0.0, 0.5}

Other knobs FROZEN at known-good values:
    K = 6                     (Na-mixed needs K<=6; EuInAs may over-merge,
                               revisit at K=10 only if needed)
    epochs = 30               (sweep speed; promote winner to 50ep after)
    SAXS gate: Q=70..140      (tightened from 60..150 per profile inspection)
    supcon_temperature = 0.3
    no contrastive, no repel, no centroid, no w_ent
    COM-centering on
    aug_disable = [hflip, vflip, colorjitter]

Schedule fractions (so the schedule is invariant under EPOCHS rescaling):
    teacher temp: warmup_frac = 0.2 (already fractional in model)
    lambda ramp:  WARMUP_FRAC = 2/3, RAMP_FRAC = 1/3
                  -> warmup_epochs = round(2/3 * EPOCHS), same for ramp.
                  (Lambda ramp only affects contrastive/centroid, both 0
                  here, but kept consistent so 50ep promotion is clean.)

Output: runs/_supcon_lg_sweep/<sample>/<label>/
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
from data import SAMPLES
from run_contrastive import run_config, evaluate_and_report

EPOCHS = 30
K = 6
MASK_R = 15
POLAR_MASK_COLS = 45
SUPCON_TEMP = 0.3

# Schedule fractions (relative to EPOCHS)
WARMUP_FRAC = 2.0 / 3.0       # lambda ramp: 0 contribution for first 2/3
RAMP_FRAC = 1.0 / 3.0         # lambda ramp: linear over remaining 1/3
WARMUP_EPOCHS = int(round(WARMUP_FRAC * EPOCHS))
RAMP_EPOCHS = int(round(RAMP_FRAC * EPOCHS))

OUT_ROOT = os.path.join("runs", "_supcon_lg_sweep")

# (label, supcon_lambda, conf_weight_gamma)
# Ordered so smallest-effect runs come first; smallest lambda + gamma=0 is
# the cleanest baseline against which to read the others.
CONFIGS = [
    ("L01_g00", 0.01, 0.0),
    ("L01_g05", 0.01, 0.5),
    ("L02_g00", 0.02, 0.0),
    ("L02_g05", 0.02, 0.5),
    ("L05_g00", 0.05, 0.0),
    ("L05_g05", 0.05, 0.5),
]

SAMPLE_LIST = ["Na007b", "EuInAs_B100"]


def _kwargs(sample, sup_lam, gamma):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    rad_path = base + ".radial.npy"
    th_path = base + ".gate_thresholds.json"
    return dict(
        epochs=EPOCHS, seed=42, batch_size=128,
        lr=3e-4, weight_decay=1e-6,
        num_prototypes=K,
        t0=0.04, tfin=0.07,
        warmup_epochs=WARMUP_EPOCHS, ramp_epochs=RAMP_EPOCHS,
        entropy_gate=False,
        projection_dim=128, projection_hidden=256,
        theta_shift_range=None,
        theta_shift_range_student=192, theta_shift_range_teacher=16,
        center_mask_radius=MASK_R,
        center_crop_size=140,
        vmax=None,
        polar_size=192, polar_mask_cols=POLAR_MASK_COLS,
        pipeline="polar",
        centroid_lambda=0.0, centroid_margin=0.3,
        conf_weight_gamma=gamma,
        entropy_gate_override=None,
        lam_spatial=0.0,
        architecture="resnet", n_layers=1,
        w_ent=0.0,
        com_centering=True,
        com_search_radius_factor=2.0,
        aug_disable=["hflip", "vflip", "colorjitter"],
        supcon_radials_path=rad_path,
        supcon_thresholds_path=th_path,
        supcon_lambda=sup_lam,
        supcon_temperature=SUPCON_TEMP,
        contrastive_lambda_override=0.0,
        proto_repel_lambda=0.0,
        proto_repel_threshold=0.5,
    )


def run_one(sample: str, label: str, sup_lam: float, gamma: float, device):
    outdir = os.path.join(OUT_ROOT, sample, label)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample}/{label}  "
              f"lam={sup_lam} gamma={gamma}  K={K} ep={EPOCHS}", flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(sample, sup_lam, gamma))
        print(f"[{datetime.now():%H:%M:%S}] train done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        print(f"[skip train] {outdir}", flush=True)
    if not os.path.exists(sentinel_eval):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {sample}/{label}", flush=True)
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[supcon lambda x gamma sweep] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}", flush=True)
    print(f"K={K}  epochs={EPOCHS}  "
          f"warmup_ep={WARMUP_EPOCHS} (frac={WARMUP_FRAC:.3f})  "
          f"ramp_ep={RAMP_EPOCHS} (frac={RAMP_FRAC:.3f})", flush=True)
    print(f"configs: {[c[0] for c in CONFIGS]}", flush=True)
    print(f"samples: {SAMPLE_LIST}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for label, sup_lam, gamma in CONFIGS:
        for sample in SAMPLE_LIST:
            try:
                run_one(sample, label, sup_lam, gamma, device)
            except Exception as e:
                print(f"[FAIL] {sample}/{label}: {e!r}", flush=True)
                import traceback; traceback.print_exc()
    # Sentinel file the watchdog watches for to shut down
    with open(os.path.join(OUT_ROOT, "_done.flag"), "w") as f:
        f.write(datetime.now().isoformat())
    print(f"\n[supcon lambda x gamma sweep] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
