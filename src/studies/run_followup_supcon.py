"""run_followup_supcon.py — sequential test of the new design:
    DINO + radial-gated SupCon, no centroid, no aux losses, no contrastive.

Order: Na007b first, then EuInAs_B100.
Both use Na007b's center mask (mask_r=15, polar_mask_cols=45) per user.
COM-centered, augs trimmed (no hflip/vflip/colorjitter).

Pre-requisites (handled outside):
    <sample>.radial.npy
    <sample>.gate_thresholds.json

Output: runs/_followup_supcon/<sample>/
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

K = 10
OUT_ROOT = os.path.join("runs", "_followup_supcon")

# Per user 2026-04-26: use Na007b mask geometry for both samples.
# Na007b default in SAMPLES has mask_r=15 (already 1.5x of original 10).
MASK_R = 15
POLAR_MASK_COLS = 45

SAMPLE_LIST = ["Na007b", "EuInAs_B100"]

SUPCON_LAMBDA = 0.5
SUPCON_TEMP = 0.1


def _kwargs(sample: str):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    radials_path = base + ".radial.npy"
    thresholds_path = base + ".gate_thresholds.json"
    if not os.path.exists(radials_path):
        raise FileNotFoundError(
            f"radial profile missing for {sample}: {radials_path}\n"
            f"Run: python compute_radial_profile.py --sample {sample}")
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(
            f"gate thresholds missing for {sample}: {thresholds_path}")
    return dict(
        epochs=50, seed=42, batch_size=128,
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
        # No centroid loss in the new design
        centroid_lambda=0.0, centroid_margin=0.3,
        # No conf-weight loss in the new design
        conf_weight_gamma=0.0,
        entropy_gate_override=None,
        lam_spatial=0.0,
        architecture="resnet", n_layers=1,
        w_ent=0.0,
        com_centering=True,
        com_search_radius_factor=2.0,
        aug_disable=["hflip", "vflip", "colorjitter"],
        # NEW: physics-gated SupCon
        supcon_radials_path=radials_path,
        supcon_thresholds_path=thresholds_path,
        supcon_lambda=SUPCON_LAMBDA,
        supcon_temperature=SUPCON_TEMP,
    )


def run_one(sample: str, device):
    outdir = os.path.join(OUT_ROOT, sample, "main")
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample} -> {outdir}",
              flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(sample))
        print(f"[{datetime.now():%H:%M:%S}] train done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        print(f"[skip train] {outdir}", flush=True)
    if not os.path.exists(sentinel_eval):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {sample}", flush=True)
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        print(f"[skip eval] {outdir}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[supcon followup] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}", flush=True)
    print(f"K={K}  mask_r={MASK_R}  polar_mask_cols={POLAR_MASK_COLS}  "
          f"supcon_lambda={SUPCON_LAMBDA}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for s in SAMPLE_LIST:
        try:
            run_one(s, device)
        except Exception as e:
            print(f"[FAIL] {s}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
    print(f"\n[supcon followup] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
