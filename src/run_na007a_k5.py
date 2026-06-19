"""run_na007a_k5.py -- Na007a at K=5, otherwise matching the existing
runs/_paper_master/Na007a_K6 recipe (vmax=2, 30 ep, deterministic,
DINO + cluster1d lambda=0.1 gamma=0.5 margin=0.4).

Output: runs/_paper_master/Na007a_K5/
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

OUT_DIR = os.path.join("runs", "_paper_master", "Na007a_K5")
SAMPLE = "Na007a"
K = 5
EPOCHS = 30
SEED = 42


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SAMPLES[SAMPLE]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    rad_path = base + ".radial.npy"
    th_path = base + ".gate_thresholds.json"
    warmup = int(round((2.0 / 3.0) * EPOCHS))
    ramp = int(round((1.0 / 3.0) * EPOCHS))

    if not os.path.exists(os.path.join(OUT_DIR, "best.pth")):
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {SAMPLE} K={K} ep={EPOCHS}",
              flush=True)
        run_config("c", sample=SAMPLE, outdir=OUT_DIR, device=device,
            epochs=EPOCHS, seed=SEED, batch_size=128,
            lr=3e-4, weight_decay=1e-6,
            num_prototypes=K,
            t0=0.04, tfin=0.07,
            warmup_epochs=warmup, ramp_epochs=ramp,
            entropy_gate=False,
            projection_dim=128, projection_hidden=256,
            theta_shift_range=None,
            theta_shift_range_student=192, theta_shift_range_teacher=16,
            center_mask_radius=cfg.get("center_mask_radius", 15),
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
    else:
        print(f"[skip train] {OUT_DIR} already done", flush=True)

    if not os.path.exists(os.path.join(OUT_DIR, "eval", "metrics.json")):
        t0 = time.perf_counter()
        print(f"[{datetime.now():%H:%M:%S}] EVAL {SAMPLE} K={K}", flush=True)
        evaluate_and_report("c", sample=SAMPLE, outdir=OUT_DIR, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
