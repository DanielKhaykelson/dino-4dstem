"""run_paper_master.py -- the full paper run.

Recipe (locked):
    DINO + cluster1d (lambda=0.1, gamma=0.5, margin=0.4)
    K = sample-dependent (per the user's spec)
    epochs = 30
    SAXS gate Q=70..140 (radials precomputed)
    deterministic settings (cudnn benchmark off, deterministic algorithms,
    Tensor.max(dim=) global pool replacement)
    seed = 42

Configs (13 runs):

    Na007b:               K=6, K=7, K=8
    EuInAs_B100:          K=5, K=6, K=7
    Na007a:               K=6
    Na006a:               K=6
    IMC_50nm_SI2:         K=6
    IMC_150nm_SI5:        K=6
    MgNaPHI_remeas_SI007: K=6
    MgNaPHI_remeas_SI010: K=6
    NaPHI_Nadja_SI004:    K=6

Output: runs/_paper_master/<sample>_K{K}/
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
OUT_ROOT = os.path.join("runs", "_paper_master")

# (sample, K)  -- order: small / cheap first; multi-K samples in K-ascending
# NOTE: NaPHI_Nadja_SI004, MgNaPHI_remeas_SI007, MgNaPHI_remeas_SI010 are NOT
# in this list -- they're handled by run_line_coverage_pipeline.py (combined
# training + multi-eval), which is the proper analysis for those samples.
CONFIGS = [
    ("Na007b",               6),
    ("Na007b",               7),
    ("Na007b",               8),
    ("Na007a",               6),
    ("Na006a",               6),
    ("IMC_50nm_SI2",         6),
    ("IMC_150nm_SI5",        6),
    ("EuInAs_B100",          5),
    ("EuInAs_B100",          6),
    ("EuInAs_B100",          7),
]


def _kwargs(sample, K):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    rad_path = base + ".radial.npy"
    th_path = base + ".gate_thresholds.json"
    warmup_epochs = int(round((2.0 / 3.0) * EPOCHS))
    ramp_epochs = int(round((1.0 / 3.0) * EPOCHS))
    mask_r = int(cfg.get("center_mask_radius", 15))
    return dict(
        epochs=EPOCHS, seed=42, batch_size=128,
        lr=3e-4, weight_decay=1e-6,
        num_prototypes=K,
        t0=0.04, tfin=0.07,
        warmup_epochs=warmup_epochs, ramp_epochs=ramp_epochs,
        entropy_gate=False,
        projection_dim=128, projection_hidden=256,
        theta_shift_range=None,
        theta_shift_range_student=192, theta_shift_range_teacher=16,
        center_mask_radius=mask_r,
        center_crop_size=140,
        vmax=None,
        polar_size=192, polar_mask_cols=45,
        pipeline="polar",
        centroid_lambda=0.0, centroid_margin=0.3,
        conf_weight_gamma=0.5,
        entropy_gate_override=None,
        lam_spatial=0.0,
        architecture="resnet", n_layers=1,
        w_ent=0.0,
        com_centering=True,
        com_search_radius_factor=2.0,
        aug_disable=["hflip", "vflip", "colorjitter"],
        supcon_radials_path=rad_path,
        supcon_thresholds_path=th_path,
        supcon_lambda=0.0,
        supcon_temperature=0.3,
        contrastive_lambda_override=0.0,
        proto_repel_lambda=0.0,
        proto_repel_threshold=0.5,
        cluster1d_lambda=0.1,
        cluster1d_margin=0.4,
        cluster1d_min_cluster_mass=1.0,
        cluster1d_warmup_frac=0.0,
        cluster1d_ramp_frac=0.0,
    )


def run_one(sample, K, device):
    label = f"{sample}_K{K}"
    outdir = os.path.join(OUT_ROOT, label)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {label}", flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(sample, K))
        print(f"[{datetime.now():%H:%M:%S}] train done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        print(f"[skip train] {outdir}", flush=True)
    if not os.path.exists(sentinel_eval):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {label}", flush=True)
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[paper master] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}  epochs={EPOCHS}", flush=True)
    print(f"configs ({len(CONFIGS)}): "
          f"{[f'{s}_K{k}' for s, k in CONFIGS]}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for sample, K in CONFIGS:
        try:
            run_one(sample, K, device)
        except Exception as e:
            print(f"[FAIL] {sample}_K{K}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
    with open(os.path.join(OUT_ROOT, "_done.flag"), "w") as f:
        f.write(datetime.now().isoformat())
    print(f"\n[paper master] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
