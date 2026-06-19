"""run_supcon_repel_sweep.py — Add prototype-repel to C1 / C2, ablate
contrastive on/off. Per user 2026-04-27:

    "do C1 with and without contrastive, and add the repulsion loss as well.
     I worry that I will end up having too many loss terms so withoutC will
     be better if it turns out to the best. You can also do C2.
     EuInAs and Na007b first."

Configs (4 total):
    C1_repel_noc:  supcon_λ=0.05, τ=0.3,  contrastive=0.0, proto_repel_λ=0.1
    C1_repel_wc:   supcon_λ=0.05, τ=0.3,  contrastive=0.2, proto_repel_λ=0.1
    C2_repel_noc:  supcon_λ=0.10, τ=0.3,  contrastive=0.0, proto_repel_λ=0.1
    C2_repel_wc:   supcon_λ=0.10, τ=0.3,  contrastive=0.2, proto_repel_λ=0.1

All share: K=10, mask_r=15, polar_mask_cols=45, COM-centering on,
augs minus hflip/vflip/colorjitter, no centroid, no w_ent, 30 ep,
proto_repel_threshold=0.5.

Order: BOTH samples per config, doing 'noc' first to satisfy the
"prefer fewer loss terms" preference if it works.

Output: runs/_supcon_repel/<sample>/<config_label>/
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
PROTO_REPEL_LAMBDA = 0.1
PROTO_REPEL_THRESHOLD = 0.5
OUT_ROOT = os.path.join("runs", "_supcon_repel")

# (label, supcon_lambda, supcon_temp, contrastive_lambda)
# Order so that the simpler-loss configs (noc) come first.
CONFIGS = [
    ("C1_repel_noc", 0.05, 0.3, 0.0),
    ("C2_repel_noc", 0.10, 0.3, 0.0),
    ("C1_repel_wc",  0.05, 0.3, 0.2),
    ("C2_repel_wc",  0.10, 0.3, 0.2),
]

SAMPLE_LIST = ["Na007b", "EuInAs_B100"]


def _kwargs(sample, sup_lam, sup_t, contrastive_lambda):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    rad_path = base + ".radial.npy"
    th_path = base + ".gate_thresholds.json"
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
        supcon_radials_path=rad_path,
        supcon_thresholds_path=th_path,
        supcon_lambda=sup_lam,
        supcon_temperature=sup_t,
        contrastive_lambda_override=contrastive_lambda,
        proto_repel_lambda=PROTO_REPEL_LAMBDA,
        proto_repel_threshold=PROTO_REPEL_THRESHOLD,
    )


def run_one(sample: str, label: str, sup_lam: float, sup_t: float,
            contrastive_lambda: float, device):
    outdir = os.path.join(OUT_ROOT, sample, label)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample}/{label}  "
              f"supcon_λ={sup_lam} τ={sup_t}  contrastive_λ={contrastive_lambda}  "
              f"repel_λ={PROTO_REPEL_LAMBDA}", flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(sample, sup_lam, sup_t, contrastive_lambda))
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
    print(f"[supcon+repel sweep] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}  epochs={EPOCHS}", flush=True)
    print(f"proto_repel_lambda={PROTO_REPEL_LAMBDA}  "
          f"proto_repel_threshold={PROTO_REPEL_THRESHOLD}", flush=True)
    print(f"configs: {[c[0] for c in CONFIGS]}", flush=True)
    print(f"samples: {SAMPLE_LIST}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for label, sup_lam, sup_t, contrastive_lambda in CONFIGS:
        for sample in SAMPLE_LIST:
            try:
                run_one(sample, label, sup_lam, sup_t,
                         contrastive_lambda, device)
            except Exception as e:
                print(f"[FAIL] {sample}/{label}: {e!r}", flush=True)
                import traceback; traceback.print_exc()
    print(f"\n[supcon+repel sweep] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
