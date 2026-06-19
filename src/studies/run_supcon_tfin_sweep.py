"""run_supcon_tfin_sweep.py — Tfin ablation on top of the SupCon sweep.

Per user 2026-04-26 night: liked C2 no-contrastive but still has over-clustering
concern. Try lower Tfin to make teacher distribution sharper -> potentially
fewer redundant prototypes.

3 SupCon configs (C1, C2, C3) × 2 NEW Tfin (0.05, 0.06) × 2 samples = 12 runs.
All without contrastive (contrastive_lambda=0). 30 epochs each.

Output: runs/_supcon_sweep/<sample>/<config_label>/
        e.g.  C1_noc_t05  (Tfin=0.05)
              C2_noc_t06  (Tfin=0.06)
        T07 already covered by run_supcon_sweep.py with label C{1,2,3}_noc.
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
OUT_ROOT = os.path.join("runs", "_supcon_sweep")

# Per user preference: limit Tfin sweep to C2 (their preferred config) to keep
# total compute bounded. C1/C3 at lower Tfin can be added as a follow-up if
# C2-Tfin sweep suggests temperature is the dominant axis.
SUPCON_POINTS = [
    ("C2", 0.10, 0.3),
]

# NEW Tfin values (0.07 already in main sweep with label C{i}_noc)
TFIN_POINTS = [
    ("t05", 0.05),
    ("t06", 0.06),
]

SAMPLE_LIST = ["Na007b", "EuInAs_B100"]


def _kwargs(sample: str, supcon_lambda: float, supcon_temp: float, tfin: float):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    radials_path = base + ".radial.npy"
    thresholds_path = base + ".gate_thresholds.json"
    if not os.path.exists(radials_path):
        raise FileNotFoundError(f"radial missing: {radials_path}")
    return dict(
        epochs=EPOCHS, seed=42, batch_size=128,
        lr=3e-4, weight_decay=1e-6,
        num_prototypes=K,
        t0=0.04, tfin=tfin,
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
        contrastive_lambda_override=0.0,
    )


def run_one(sample: str, label: str, supcon_lambda: float, supcon_temp: float,
             tfin: float, device):
    outdir = os.path.join(OUT_ROOT, sample, label)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample}/{label}  "
              f"supcon_λ={supcon_lambda} τ={supcon_temp}  Tfin={tfin}",
              flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(sample, supcon_lambda, supcon_temp, tfin))
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
    print(f"[supcon Tfin sweep] device={device}", flush=True)
    print(f"new Tfin values: {TFIN_POINTS}", flush=True)
    print(f"supcon points: {SUPCON_POINTS}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for sample in SAMPLE_LIST:
        for cname, sup_lam, sup_t in SUPCON_POINTS:
            for tname, tfin in TFIN_POINTS:
                label = f"{cname}_noc_{tname}"
                try:
                    run_one(sample, label, sup_lam, sup_t, tfin, device)
                except Exception as e:
                    print(f"[FAIL] {sample}/{label}: {e!r}", flush=True)
                    import traceback; traceback.print_exc()
    print(f"\n[supcon Tfin sweep] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
