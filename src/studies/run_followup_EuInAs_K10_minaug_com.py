"""run_followup_EuInAs_K10_minaug_com.py — EuInAs K=10 vanilla with:
    - mask 1.5x original (mask_r=15, polar_mask_cols=45)
    - COM centering (search radius = 2x mask_r in cropped frame)
    - augmentations: drop hflip, vflip, colorjitter (keep blur, theta-roll)

Per user 2026-04-26: hflip/vflip add unphysical mirror invariance, colorjitter's
additive brightness shift was confusing vacuum-vs-support distinction; mask
should be 1.5x original (not half), and centered on COM constrained to
2x-mask-radius from the geometric center.

Output: runs/_followup_EuInAs_K10_minaug_com/
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
from run_contrastive import run_config, evaluate_and_report

K = 10
SAMPLE = "EuInAs_B100"
OUT_ROOT = os.path.join("runs", "_followup_EuInAs_K10_minaug_com")
LABEL = "main"


KWARGS = dict(
    epochs=50, seed=42, batch_size=128,
    lr=3e-4, weight_decay=1e-6,
    num_prototypes=K,
    t0=0.04, tfin=0.07,
    warmup_epochs=20, ramp_epochs=10,
    entropy_gate=False,
    projection_dim=128, projection_hidden=256,
    theta_shift_range=None,
    theta_shift_range_student=192, theta_shift_range_teacher=16,
    center_mask_radius=15,             # 1.5x original 10
    center_crop_size=140,
    vmax=None,                         # SAMPLES default (30 for EuInAs)
    polar_size=192, polar_mask_cols=45,    # 1.5x original 30
    pipeline="polar",
    centroid_lambda=0.05, centroid_margin=0.3,
    conf_weight_gamma=0.0,             # vanilla
    entropy_gate_override=None,
    lam_spatial=0.0,
    architecture="resnet", n_layers=1,
    w_ent=0.0,
    com_centering=True,
    com_search_radius_factor=2.0,
    aug_disable=["hflip", "vflip", "colorjitter"],
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[minaug+com] device={device}", flush=True)
    outdir = os.path.join(OUT_ROOT, LABEL)
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(os.path.join(outdir, "best.pth")):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {SAMPLE} -> {outdir}",
              flush=True)
        run_config("c", sample=SAMPLE, outdir=outdir, device=device, **KWARGS)
        print(f"[{datetime.now():%H:%M:%S}] train done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        print(f"[skip train] {outdir}", flush=True)
    if not os.path.exists(os.path.join(outdir, "eval", "metrics.json")):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {SAMPLE}", flush=True)
        evaluate_and_report("c", sample=SAMPLE, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"\n[minaug+com] done", flush=True)


if __name__ == "__main__":
    main()
