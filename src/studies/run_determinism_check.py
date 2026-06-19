"""run_determinism_check.py -- L1d01_g05 K=6 30ep on Na007b + EuInAs with
the new deterministic settings (CUBLAS workspace + cudnn benchmark off +
use_deterministic_algorithms + Tensor.max(dim=) maxpool replacement).

Compare to the previous runs/_dino_c1d_sweep/<sample>/L1d01_g05 to see
the deviation between the old (non-deterministic) and new (deterministic)
trajectories at the same seed.

Output: runs/_determinism_check/<sample>/
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
K = 6
OUT_ROOT = os.path.join("runs", "_determinism_check")
SAMPLES_LIST = ["Na007b", "EuInAs_B100"]


def _kwargs(sample):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    rad_path = base + ".radial.npy"
    th_path = base + ".gate_thresholds.json"
    warmup_epochs = int(round((2.0 / 3.0) * EPOCHS))
    ramp_epochs = int(round((1.0 / 3.0) * EPOCHS))
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
        center_mask_radius=15,
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


def run_one(sample, device):
    outdir = os.path.join(OUT_ROOT, sample)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample}  L1d01_g05 (deterministic)",
              flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(sample))
        print(f"[{datetime.now():%H:%M:%S}] train done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    if not os.path.exists(sentinel_eval):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {sample}", flush=True)
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[determinism check] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}  K={K}  epochs={EPOCHS}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for s in SAMPLES_LIST:
        run_one(s, device)
    with open(os.path.join(OUT_ROOT, "_done.flag"), "w") as f:
        f.write(datetime.now().isoformat())
    print(f"\n[determinism check] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
