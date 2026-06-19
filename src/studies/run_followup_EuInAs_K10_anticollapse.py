"""run_followup_EuInAs_K10_anticollapse.py — Try to recover the "great"
EuInAs class map (3 clean horizontal layers + middle-layer shadow sub-classes,
crisp single-pixel interfaces, K_active >= 7).

Strategy: K=10 vanilla, vary w_ent (entropy regularization) to keep more
prototypes alive against the centering's collapse pressure.

Per user 2026-04-26: vmax=30 (don't change), mask_r=10 (don't change),
gamma=0 (vanilla, no weight loss), focus on w_ent first.

Run order (by user): w_ent=0.1 first (prime candidate), then 0.0 (control),
then 0.2 (aggressive). Each run is independent so we can inspect mid-pipeline.

Output: runs/_followup_EuInAs_K10_anticollapse/w_ent_<value>/
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
OUT_ROOT = os.path.join("runs", "_followup_EuInAs_K10_anticollapse")

# User-specified order: 0.1 first, then 0.0 control, then 0.2.
W_ENTS = [0.1, 0.0, 0.2]


def _kwargs(w_ent: float):
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
        center_mask_radius=None,    # use SAMPLES default (10 for EuInAs)
        center_crop_size=140,
        vmax=None,                  # use SAMPLES default (30 for EuInAs)
        polar_size=192, polar_mask_cols=30,
        pipeline="polar",
        centroid_lambda=0.05, centroid_margin=0.3,
        conf_weight_gamma=0.0,      # vanilla per user request
        entropy_gate_override=None,
        lam_spatial=0.0,
        architecture="resnet", n_layers=1,
        w_ent=w_ent,
    )


def run_one(w_ent: float, device):
    label = f"w_ent_{w_ent:.1f}"
    outdir = os.path.join(OUT_ROOT, label)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {SAMPLE}/{label} "
              f"(K={K}, gamma=0, w_ent={w_ent})", flush=True)
        run_config("c", sample=SAMPLE, outdir=outdir, device=device,
                   **_kwargs(w_ent))
        print(f"[{datetime.now():%H:%M:%S}] train done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        print(f"[skip train] {outdir}", flush=True)
    if not os.path.exists(sentinel_eval):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {SAMPLE}/{label}",
              flush=True)
        evaluate_and_report("c", sample=SAMPLE, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        print(f"[skip eval] {outdir}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[followup K=10 anticollapse] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}", flush=True)
    print(f"w_ent sweep order: {W_ENTS}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for w in W_ENTS:
        try:
            run_one(w, device)
        except Exception as e:
            print(f"[FAIL] w_ent={w}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
    print(f"\n[followup K=10 anticollapse] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
