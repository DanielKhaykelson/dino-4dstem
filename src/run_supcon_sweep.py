"""run_supcon_sweep.py — SupCon hyperparam sweep with contrastive ablation.

Phase order per user 2026-04-26: BOTH samples without contrastive (3 configs),
then BOTH samples with contrastive=0.2 (3 configs).

Configs (3 SupCon points, repeated for each contrastive setting):
    C1: supcon_lambda=0.05, supcon_temperature=0.3
    C2: supcon_lambda=0.10, supcon_temperature=0.3
    C3: supcon_lambda=0.05, supcon_temperature=0.5

Output:
    runs/_supcon_sweep/<sample>/<config_label>/
        e.g. runs/_supcon_sweep/Na007b/C1_noc/  (no contrastive)
             runs/_supcon_sweep/Na007b/C1_wc/   (with contrastive=0.2)

Each run: 30 epochs, K=10, mask_r=15, polar_mask_cols=45, COM-centering on,
augs minus hflip/vflip/colorjitter, no centroid loss.
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
K = 10
MASK_R = 15
POLAR_MASK_COLS = 45
OUT_ROOT = os.path.join("runs", "_supcon_sweep")

# 3 SupCon points
SUPCON_POINTS = [
    ("C1", 0.05, 0.3),
    ("C2", 0.10, 0.3),
    ("C3", 0.05, 0.5),
]

# Phase 1: no contrastive | Phase 2: contrastive=0.2
PHASES = [
    ("noc", 0.0),
    ("wc",  0.2),
]

# Sample order per user: Na007b first, then EuInAs (in each phase)
SAMPLE_LIST = ["Na007b", "EuInAs_B100"]


def _kwargs(sample: str, supcon_lambda: float, supcon_temp: float,
            contrastive_lambda: float):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    radials_path = base + ".radial.npy"
    thresholds_path = base + ".gate_thresholds.json"
    if not os.path.exists(radials_path):
        raise FileNotFoundError(
            f"radial missing for {sample}: {radials_path}\n"
            f"Run: python compute_radial_profile.py --sample {sample}")
    return dict(
        epochs=EPOCHS, seed=42, batch_size=128,
        lr=3e-4, weight_decay=1e-6,
        num_prototypes=K,
        t0=0.04, tfin=0.07,
        warmup_epochs=20, ramp_epochs=10,
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
        conf_weight_gamma=0.0,
        entropy_gate_override=None,
        lam_spatial=0.0,
        architecture="resnet", n_layers=1,
        w_ent=0.0,
        com_centering=True,
        com_search_radius_factor=2.0,
        aug_disable=["hflip", "vflip", "colorjitter"],
        supcon_radials_path=radials_path,
        supcon_thresholds_path=thresholds_path,
        supcon_lambda=supcon_lambda,
        supcon_temperature=supcon_temp,
        contrastive_lambda_override=contrastive_lambda,
    )


def run_one(sample: str, label: str, supcon_lambda: float, supcon_temp: float,
            contrastive_lambda: float, device):
    outdir = os.path.join(OUT_ROOT, sample, label)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample}/{label}  "
              f"supcon_λ={supcon_lambda} supcon_τ={supcon_temp}  "
              f"contrastive_λ={contrastive_lambda}", flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(sample, supcon_lambda, supcon_temp,
                              contrastive_lambda))
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
    print(f"[supcon sweep] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}  epochs={EPOCHS}", flush=True)
    print(f"phases: {PHASES}", flush=True)
    print(f"supcon points: {SUPCON_POINTS}", flush=True)
    print(f"samples: {SAMPLE_LIST}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for phase_label, contrastive_lambda in PHASES:
        for sample in SAMPLE_LIST:
            for cname, sup_lam, sup_t in SUPCON_POINTS:
                label = f"{cname}_{phase_label}"
                try:
                    run_one(sample, label, sup_lam, sup_t,
                             contrastive_lambda, device)
                except Exception as e:
                    print(f"[FAIL] {sample}/{label}: {e!r}", flush=True)
                    import traceback; traceback.print_exc()
    print(f"\n[supcon sweep] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
