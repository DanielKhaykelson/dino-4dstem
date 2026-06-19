"""run_winner_followup.py -- 3 follow-up runs at the cluster1d-sweep winner
settings (lambda_1d=0.1, gamma=0.5, margin=0.4):

  1. Na007b   K=6,  50 ep   -- promote to 50ep, see if still better
  2. EuInAs   K=6,  50 ep   -- same
  3. Na007b   K=10, 30 ep   -- robustness: does extra K over-segment or
                                 stay self-regulated?

Waits for the current cluster1d sweep's sentinel
(runs/_dino_c1d_sweep/_done.flag) before launching, so it can be queued
in the background without fighting for the GPU.

Output: runs/_winner_followup/<label>/
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

OUT_ROOT = os.path.join("runs", "_winner_followup")
WAIT_FOR = os.path.join("runs", "_dino_c1d_sweep", "_done.flag")
MASK_R = 15
POLAR_MASK_COLS = 45

# (label, sample, K, epochs)
CONFIGS = [
    ("Na007b_K6_50ep",  "Na007b",      6,  50),
    ("EuInAs_K6_50ep",  "EuInAs_B100", 6,  50),
    ("Na007b_K10_30ep", "Na007b",      10, 30),
]


def _kwargs(sample, K, epochs):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    rad_path = base + ".radial.npy"
    th_path = base + ".gate_thresholds.json"
    # Schedule fractions kept invariant: warmup 2/3, ramp 1/3 of total epochs
    warmup_epochs = int(round((2.0 / 3.0) * epochs))
    ramp_epochs = int(round((1.0 / 3.0) * epochs))
    return dict(
        epochs=epochs, seed=42, batch_size=128,
        lr=3e-4, weight_decay=1e-6,
        num_prototypes=K,
        t0=0.04, tfin=0.07,
        warmup_epochs=warmup_epochs, ramp_epochs=ramp_epochs,
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


def run_one(label, sample, K, epochs, device):
    outdir = os.path.join(OUT_ROOT, label)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {label}  "
              f"(sample={sample} K={K} ep={epochs})", flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(sample, K, epochs))
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


def wait_for_sentinel(path: str, poll_sec: int = 60):
    if os.path.exists(path):
        return
    print(f"[follow-up] waiting for {path}", flush=True)
    t0 = time.perf_counter()
    while not os.path.exists(path):
        time.sleep(poll_sec)
    print(f"[follow-up] sentinel found after "
          f"{(time.perf_counter()-t0)/60:.1f} min, launching", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[winner follow-up] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}", flush=True)
    print(f"configs: {[c[0] for c in CONFIGS]}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    wait_for_sentinel(WAIT_FOR)
    t_total = time.perf_counter()
    for label, sample, K, epochs in CONFIGS:
        try:
            run_one(label, sample, K, epochs, device)
        except Exception as e:
            print(f"[FAIL] {label}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
    with open(os.path.join(OUT_ROOT, "_done.flag"), "w") as f:
        f.write(datetime.now().isoformat())
    print(f"\n[winner follow-up] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
