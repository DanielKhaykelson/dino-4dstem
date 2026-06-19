"""run_paper_K3_EuInAs.py — Probe-and-commit for EuInAs at K=3.

Tail-end ablation requested by user (2026-04-25): after the K=6 paper pipeline
finishes, also run EuInAs at K=3 to test the lower-K limit. K=3 forces the
model into very few classes; if the layered structure is truly low-dimensional
this should still work; if it's not, K=3 will fail visibly.

Output: dino_sr_contrastive/runs/_paper_K3_EuInAs/EuInAs_B100/
        (probe/, main/, probe_decision.json, main/eval/...)
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys, time, json, csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from run_contrastive import run_config, evaluate_and_report

THRESHOLD = 0.85
PROBE_EPOCHS = 5
MAIN_EPOCHS = 50
K = 3
SAMPLE = "EuInAs_B100"
OUT_ROOT = os.path.join("runs", "_paper_K3_EuInAs")


def _kwargs(epochs: int, gamma: float, stop_at_epoch: "int | None" = None):
    return dict(
        epochs=epochs, seed=42, batch_size=128,
        lr=3e-4, weight_decay=1e-6,
        num_prototypes=K,
        t0=0.04, tfin=0.07,
        warmup_epochs=20, ramp_epochs=10,
        entropy_gate=False,
        projection_dim=128, projection_hidden=256,
        theta_shift_range=None,
        theta_shift_range_student=192, theta_shift_range_teacher=16,
        center_mask_radius=None,
        center_crop_size=140,
        vmax=None,
        polar_size=192, polar_mask_cols=30,
        pipeline="polar",
        centroid_lambda=0.05, centroid_margin=0.3,
        conf_weight_gamma=gamma,
        entropy_gate_override=None,
        lam_spatial=0.0,
        architecture="resnet", n_layers=1,
        stop_at_epoch=stop_at_epoch,
    )


def read_ep5(probe_dir: str) -> float:
    path = os.path.join(probe_dir, "training_log.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if int(r["epoch"]) == PROBE_EPOCHS:
            return float(r["avg_conf"])
    raise RuntimeError(f"no ep{PROBE_EPOCHS} in {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[K=3 EuInAs probe-and-commit] device={device}", flush=True)
    sample_dir = os.path.join(OUT_ROOT, SAMPLE)
    probe_dir = os.path.join(sample_dir, "probe")
    main_dir = os.path.join(sample_dir, "main")
    os.makedirs(sample_dir, exist_ok=True)

    if not os.path.exists(os.path.join(probe_dir, "training_log.csv")):
        os.makedirs(probe_dir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] PROBE {SAMPLE} K=3 "
              f"(stop after {PROBE_EPOCHS} ep, scheduler T_max={MAIN_EPOCHS}, gamma=0)",
              flush=True)
        run_config("c", sample=SAMPLE, outdir=probe_dir, device=device,
                   **_kwargs(MAIN_EPOCHS, 0.0, stop_at_epoch=PROBE_EPOCHS))
        print(f"[{datetime.now():%H:%M:%S}] probe done ({time.perf_counter()-t0:.0f}s)",
              flush=True)

    avg_conf = read_ep5(probe_dir)
    gamma_main = 1.0 if avg_conf > THRESHOLD else 0.0
    decision = {
        "sample": SAMPLE, "K": K, "probe_epochs": PROBE_EPOCHS,
        "threshold": THRESHOLD,
        "ep5_avg_conf": avg_conf,
        "decision": "enable_weight" if gamma_main > 0 else "skip_weight",
        "gamma_main": gamma_main, "main_epochs": MAIN_EPOCHS,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(sample_dir, "probe_decision.json"), "w") as f:
        json.dump(decision, f, indent=2)
    print(f"[DECISION] {SAMPLE} K=3: ep5 avg_conf={avg_conf:.4f} -> "
          f"{'+weight' if gamma_main>0 else 'vanilla'} (gamma={gamma_main})",
          flush=True)

    if not os.path.exists(os.path.join(main_dir, "best.pth")):
        os.makedirs(main_dir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] MAIN {SAMPLE} K=3 "
              f"({MAIN_EPOCHS} ep, gamma={gamma_main})", flush=True)
        run_config("c", sample=SAMPLE, outdir=main_dir, device=device,
                   **_kwargs(MAIN_EPOCHS, gamma_main))
        print(f"[{datetime.now():%H:%M:%S}] main done ({time.perf_counter()-t0:.0f}s)",
              flush=True)

    if not os.path.exists(os.path.join(main_dir, "eval", "metrics.json")):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {SAMPLE} K=3", flush=True)
        evaluate_and_report("c", sample=SAMPLE, outdir=main_dir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    print("\n[K=3 EuInAs] done", flush=True)


if __name__ == "__main__":
    main()
