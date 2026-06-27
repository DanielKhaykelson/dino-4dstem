"""run_winner_other_samples.py — read winner.json from analyze_supcon_sweep,
then run that exact config on Na007a and Na006a.
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys, json, time
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

ROOT = os.path.join("runs", "_supcon_sweep")
EPOCHS = 30
K = 10
MASK_R = 15
POLAR_MASK_COLS = 45

# Map config label like "C2_noc_t06" -> (supcon_lambda, supcon_temp, contrastive_lambda, tfin)
SUPCON_POINTS = {"C1": (0.05, 0.3), "C2": (0.10, 0.3), "C3": (0.05, 0.5)}
PHASES = {"noc": 0.0, "wc": 0.2}
TFINS = {"t05": 0.05, "t06": 0.06, "": 0.07}    # "" = main sweep (no t-suffix)


def parse_label(label: str):
    parts = label.split("_")
    cname = parts[0]
    phase = parts[1]
    tname = parts[2] if len(parts) > 2 else ""
    sup_lam, sup_t = SUPCON_POINTS[cname]
    contrastive = PHASES[phase]
    tfin = TFINS[tname]
    return sup_lam, sup_t, contrastive, tfin


def _kwargs(sample, sup_lam, sup_t, contrastive_lambda, tfin):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    rad_path = base + ".radial.npy"
    th_path = base + ".gate_thresholds.json"
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
        supcon_radials_path=rad_path,
        supcon_thresholds_path=th_path,
        supcon_lambda=sup_lam,
        supcon_temperature=sup_t,
        contrastive_lambda_override=contrastive_lambda,
    )


def main():
    winner_path = os.path.join(ROOT, "winner.json")
    if not os.path.exists(winner_path):
        print(f"ERROR: no winner.json at {winner_path}"); sys.exit(1)
    winner = json.load(open(winner_path))
    label = winner["winner_label"]
    sup_lam, sup_t, contrastive, tfin = parse_label(label)
    print(f"[winner phase] applying {label}: "
          f"supcon_λ={sup_lam} τ={sup_t} contrastive_λ={contrastive} Tfin={tfin}",
          flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for sample in ("Na007a", "Na006a"):
        outdir = os.path.join(ROOT, sample, label)
        if not os.path.exists(os.path.join(outdir, "best.pth")):
            os.makedirs(outdir, exist_ok=True)
            t0 = time.perf_counter()
            print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample}/{label}",
                  flush=True)
            try:
                run_config("c", sample=sample, outdir=outdir, device=device,
                           **_kwargs(sample, sup_lam, sup_t, contrastive, tfin))
                print(f"[{datetime.now():%H:%M:%S}] train done "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[FAIL train] {sample}/{label}: {e!r}", flush=True)
                import traceback; traceback.print_exc()
                continue
        if not os.path.exists(os.path.join(outdir, "eval", "metrics.json")):
            t0 = time.perf_counter()
            print(f"\n[{datetime.now():%H:%M:%S}] EVAL {sample}", flush=True)
            try:
                evaluate_and_report("c", sample=sample, outdir=outdir,
                                     device=device)
                print(f"[{datetime.now():%H:%M:%S}] eval done "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[FAIL eval] {sample}/{label}: {e!r}", flush=True)
                import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
