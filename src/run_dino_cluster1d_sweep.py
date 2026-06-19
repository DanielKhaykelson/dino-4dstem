"""run_dino_cluster1d_sweep.py -- DINO + 1D-cluster loss ONLY (no SupCon).

Tests whether the new physics-gated CLUSTERING loss (L_intra: members
close to their cluster's 1D centroid; L_inter: centroids of different
clusters must differ) by itself fixes the centroid-redundancy problem
that DINO+SupCon at K=6 left behind (Na007b max_pair ~0.97 in every
config of the previous lambda x gamma sweep).

Sweep matrix (12 runs):
    lambda_1d in {0.1, 0.3, 1.0}
    gamma     in {0.0, 0.5}
    samples   in {Na007b, EuInAs_B100}

Frozen at known-good values:
    K = 6, epochs = 30
    SAXS gate Q=70..140 (radials reused as the 1D source)
    margin = 0.4 (above tau_neg=0.13, below tau_pos~0.55)
    no SupCon, no contrastive, no centroid, no repel, no w_ent
    COM-centering on, augs minus hflip/vflip/colorjitter
    fractional schedules: cluster1d_warmup_frac=0, ramp_frac=0
        (loss active from ep0 -- DINO has anti-collapse via centering,
         and the new term's stats showed centroid separation kicks in
         within 3 epochs in the sanity run)

Output: runs/_dino_c1d_sweep/<sample>/<label>/
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
CLUSTER1D_MARGIN = 0.4

# Schedule fractions (relative to EPOCHS) -- kept consistent with prior sweep.
WARMUP_FRAC = 2.0 / 3.0
RAMP_FRAC = 1.0 / 3.0
WARMUP_EPOCHS = int(round(WARMUP_FRAC * EPOCHS))
RAMP_EPOCHS = int(round(RAMP_FRAC * EPOCHS))

OUT_ROOT = os.path.join("runs", "_dino_c1d_sweep")

# (label, cluster1d_lambda, conf_weight_gamma)
CONFIGS = [
    ("L1d01_g00", 0.1, 0.0),
    ("L1d01_g05", 0.1, 0.5),
    ("L1d03_g00", 0.3, 0.0),
    ("L1d03_g05", 0.3, 0.5),
    ("L1d10_g00", 1.0, 0.0),
    ("L1d10_g05", 1.0, 0.5),
]

SAMPLE_LIST = ["Na007b", "EuInAs_B100"]


def _kwargs(sample, lam_1d, gamma):
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
        supcon_lambda=0.0,                  # OFF -- isolating cluster1d
        supcon_temperature=0.3,
        contrastive_lambda_override=0.0,
        proto_repel_lambda=0.0,
        proto_repel_threshold=0.5,
        cluster1d_lambda=lam_1d,
        cluster1d_margin=CLUSTER1D_MARGIN,
        cluster1d_min_cluster_mass=1.0,
        cluster1d_warmup_frac=0.0,
        cluster1d_ramp_frac=0.0,
    )


def run_one(sample: str, label: str, lam_1d: float, gamma: float, device):
    outdir = os.path.join(OUT_ROOT, sample, label)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample}/{label}  "
              f"lam_1d={lam_1d} gamma={gamma}  K={K} ep={EPOCHS}", flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(sample, lam_1d, gamma))
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
    print(f"[dino + cluster1d sweep] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}", flush=True)
    print(f"K={K}  epochs={EPOCHS}  margin={CLUSTER1D_MARGIN}", flush=True)
    print(f"configs: {[c[0] for c in CONFIGS]}", flush=True)
    print(f"samples: {SAMPLE_LIST}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for label, lam_1d, gamma in CONFIGS:
        for sample in SAMPLE_LIST:
            try:
                run_one(sample, label, lam_1d, gamma, device)
            except Exception as e:
                print(f"[FAIL] {sample}/{label}: {e!r}", flush=True)
                import traceback; traceback.print_exc()
    # Sentinel for the watchdog
    with open(os.path.join(OUT_ROOT, "_done.flag"), "w") as f:
        f.write(datetime.now().isoformat())
    print(f"\n[dino + cluster1d sweep] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
