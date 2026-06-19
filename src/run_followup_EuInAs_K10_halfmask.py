"""run_followup_EuInAs_K10_halfmask.py — same EuInAs K=10 vanilla sweep
but with the central mask cut in half:

    center_mask_radius: 10 -> 5     (Cartesian beam mask)
    polar_mask_cols:    30 -> 15    (polar low-r mask)

Effective central mask in detector pixels: r <= 11 (default) -> r <= ~5.5
(roughly the central beam saturated blob only, no halo).

Per user 2026-04-26: "No that is a huge mask. cut it in half at least.
rerun ent = 0, 0.05, 0.1 with half the mask"

Output: runs/_followup_EuInAs_K10_halfmask/w_ent_<value>/
Run order: 0.0 -> 0.05 -> 0.1 (per user's listed order)
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
from run_contrastive import run_config, evaluate_and_report

K = 10
SAMPLE = "EuInAs_B100"
OUT_ROOT = os.path.join("runs", "_followup_EuInAs_K10_halfmask")

W_ENTS = [0.0, 0.05, 0.1]   # user-listed order

# halved masks
CENTER_MASK_RADIUS = 5      # 10 -> 5
POLAR_MASK_COLS    = 15     # 30 -> 15


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
        center_mask_radius=CENTER_MASK_RADIUS,
        center_crop_size=140,
        vmax=None,
        polar_size=192, polar_mask_cols=POLAR_MASK_COLS,
        pipeline="polar",
        centroid_lambda=0.05, centroid_margin=0.3,
        conf_weight_gamma=0.0,      # vanilla
        entropy_gate_override=None,
        lam_spatial=0.0,
        architecture="resnet", n_layers=1,
        w_ent=w_ent,
    )


def run_one(w_ent: float, device):
    label = f"w_ent_{w_ent:.2f}"
    outdir = os.path.join(OUT_ROOT, label)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {SAMPLE}/{label} "
              f"(K={K}, gamma=0, w_ent={w_ent}, "
              f"mask_r={CENTER_MASK_RADIUS}, polar_mask_cols={POLAR_MASK_COLS})",
              flush=True)
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
    print(f"[half-mask sweep] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}", flush=True)
    print(f"masks: center_mask_radius={CENTER_MASK_RADIUS}  "
          f"polar_mask_cols={POLAR_MASK_COLS}", flush=True)
    print(f"w_ent order: {W_ENTS}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for w in W_ENTS:
        try:
            run_one(w, device)
        except Exception as e:
            print(f"[FAIL] w_ent={w}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
    print(f"\n[half-mask sweep] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
