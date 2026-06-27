"""run_supcon_newgate.py — C1 with the SAXS-style radial gate.

Single config, minimum loss terms:
    DINO + SupCon (λ=0.05, τ=0.3)
    contrastive_λ=0  (drop the redundant self-consistency loss)
    proto_repel_λ=0  (drop -- previous run showed proto matrix is already
                       distinct; redundancy was at class-centroid level,
                       which the new gate should fix directly)

K=10, mask_r=15, polar_mask_cols=45, COM-centering on,
augs minus hflip/vflip/colorjitter, no centroid loss, 30 ep.

Output: runs/_supcon_newgate/<sample>/
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
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
SUPCON_LAMBDA = 0.05
SUPCON_TEMP = 0.3
OUT_ROOT = os.path.join("runs", "_supcon_newgate")

SAMPLE_LIST = ["Na007b", "EuInAs_B100"]


def _kwargs(sample):
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
        supcon_lambda=SUPCON_LAMBDA,
        supcon_temperature=SUPCON_TEMP,
        contrastive_lambda_override=0.0,
        proto_repel_lambda=0.0,
        proto_repel_threshold=0.5,
    )


def run_one(sample, device):
    outdir = os.path.join(OUT_ROOT, sample)
    if not os.path.exists(os.path.join(outdir, "best.pth")):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample}  "
              f"supcon_λ={SUPCON_LAMBDA} τ={SUPCON_TEMP}  "
              f"contrastive_λ=0  proto_repel_λ=0", flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(sample))
        print(f"[{datetime.now():%H:%M:%S}] train done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    if not os.path.exists(os.path.join(outdir, "eval", "metrics.json")):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {sample}", flush=True)
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[supcon newgate] device={device}", flush=True)
    print(f"output: {OUT_ROOT}  K={K}  epochs={EPOCHS}", flush=True)
    print(f"loss recipe: DINO + SupCon(λ={SUPCON_LAMBDA}, τ={SUPCON_TEMP})  "
          f"NO contrastive, NO repel, NO centroid", flush=True)
    print(f"samples: {SAMPLE_LIST}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for s in SAMPLE_LIST:
        try:
            run_one(s, device)
        except Exception as e:
            print(f"[FAIL] {s}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
    print(f"\n[supcon newgate] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
