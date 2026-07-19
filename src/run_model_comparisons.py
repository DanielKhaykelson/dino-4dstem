"""run_model_comparisons.py — SI ablations for the paper, all at the latest-default
recipe (from the 2026-05-25 m/K sweep): DINO + theta-roll + cluster1d(0.1), K=60,
30 epochs, m=0.97, EMA 0.99->0.999, contrastive/supcon OFF, polar pipeline, seed 42,
COM off. Per-sample preprocessing + vmax taken from that sweep.

Experiments (-> runs/ModelComparisons/):
  01_vit_vs_resnet : ViT vs ResNet(L2)         on Na007b, IMC_SI5
  02_resnet_depth  : ResNet n_layers 1..4      on Na007b, IMC_SI5
  03_cluster1d     : cluster1d_lambda 0 vs 0.1 on EuInAs, IMC_SI5

Usage:
  python src/run_model_comparisons.py --smoke           # 2 cells, 2 epochs
  python src/run_model_comparisons.py                   # full set, 30 epochs
  python src/run_model_comparisons.py --only Na007b_L3  # one cell
Windows: DataLoader is num_workers=0 inside the trainer; this guard is required.
"""
from __future__ import annotations
import os, sys, time, argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import torch
from data import SAMPLES
from run_contrastive import run_config, evaluate_and_report

ROOT = os.path.join("runs", "ModelComparisons")
# per-sample preprocessing + vmax (from the latest m/K sweep)
PREP = {
    "Na007b":        dict(center_crop_size=120, polar_mask_cols=20, center_mask_radius=10, vmax=5.0),
    "IMC_150nm_SI5": dict(center_crop_size=120, polar_mask_cols=40, center_mask_radius=20, vmax=5.0),
    "EuInAs_B100":   dict(center_crop_size=120, polar_mask_cols=0,  center_mask_radius=0,  vmax=30.0),
}


def shared(epochs):
    we = 20 if epochs >= 30 else max(1, int(round(epochs * 0.66)))
    re = 10 if epochs >= 30 else max(1, int(round(epochs * 0.33)))
    return dict(
        epochs=epochs, seed=42, batch_size=128, lr=3e-4, weight_decay=1e-6,
        num_prototypes=60, t0=0.04, tfin=0.07, warmup_epochs=we, ramp_epochs=re,
        entropy_gate=False, projection_dim=128, projection_hidden=256,
        theta_shift_range=None, theta_shift_range_student=192, theta_shift_range_teacher=16,
        polar_size=192,
        pipeline="polar", centroid_lambda=0.0, centroid_margin=0.3, conf_weight_gamma=0.0,
        entropy_gate_override=None, lam_spatial=0.0, w_ent=0.0,
        com_centering=False, com_search_radius_factor=2.0,
        aug_disable=["hflip", "vflip", "colorjitter"],
        center_momentum=0.97, EMA0=0.99, EMAfin=0.999, warmup_frac=0.2,
        supcon_lambda=0.0, contrastive_lambda_override=0.0,
        cluster1d_lambda=0.1, cluster1d_margin=0.4)


def cell_kwargs(sample, epochs, architecture, n_layers, cluster1d_lambda):
    k = shared(epochs); k.update(PREP[sample])
    k["architecture"] = architecture; k["n_layers"] = n_layers
    k["cluster1d_lambda"] = cluster1d_lambda
    base = SAMPLES[sample]["path"]; base = base[:-4] if base.endswith(".prz") else base
    rp = base + ".radial.npy"
    if os.path.exists(rp):
        k["supcon_radials_path"] = rp
    return k


# (exp, label, sample, architecture, n_layers, cluster1d_lambda)
CELLS = [
    ("01_vit_vs_resnet", "vit_Na007b",  "Na007b",        "vit",    1, 0.1),
    ("01_vit_vs_resnet", "vit_IMC_SI5", "IMC_150nm_SI5", "vit",    1, 0.1),
    ("02_resnet_depth",  "Na007b_L1",   "Na007b",        "resnet", 1, 0.1),
    ("02_resnet_depth",  "Na007b_L2",   "Na007b",        "resnet", 2, 0.1),
    ("02_resnet_depth",  "Na007b_L3",   "Na007b",        "resnet", 3, 0.1),
    ("02_resnet_depth",  "Na007b_L4",   "Na007b",        "resnet", 4, 0.1),
    ("02_resnet_depth",  "IMC_SI5_L1",  "IMC_150nm_SI5", "resnet", 1, 0.1),
    ("02_resnet_depth",  "IMC_SI5_L2",  "IMC_150nm_SI5", "resnet", 2, 0.1),
    ("02_resnet_depth",  "IMC_SI5_L3",  "IMC_150nm_SI5", "resnet", 3, 0.1),
    ("02_resnet_depth",  "IMC_SI5_L4",  "IMC_150nm_SI5", "resnet", 4, 0.1),
    ("03_cluster1d",     "EuInAs_c1d_off", "EuInAs_B100", "resnet", 1, 0.0),
    ("03_cluster1d",     "EuInAs_c1d_on",  "EuInAs_B100", "resnet", 1, 0.1),
    ("03_cluster1d",     "IMC_SI5_c1d_off","IMC_150nm_SI5","resnet",1, 0.0),
    # IMC_SI5 cluster1d ON == 02_resnet_depth/IMC_SI5_L1 (reused in analysis)
]


def run_cell(exp, label, sample, arch, n_layers, c1d, epochs, device, smoke=False):
    sub = "_smoke" if smoke else exp
    outdir = os.path.join(ROOT, sub, label)
    # restart-proof: a cell counts as done only if it has eval/metrics.json.
    # A bare best.pth (interrupted mid-train) does NOT count -> retrain fresh.
    if os.path.exists(os.path.join(outdir, "eval", "metrics.json")) and not smoke:
        print(f"[skip] {label}: already evaluated", flush=True)
        return outdir
    os.makedirs(outdir, exist_ok=True)
    t0 = time.perf_counter()
    print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {label}  ({sample}, {arch}, L{n_layers}, c1d={c1d}, ep={epochs})", flush=True)
    run_config("c", sample=sample, outdir=outdir, device=device,
               **cell_kwargs(sample, epochs, arch, n_layers, c1d))
    print(f"[{datetime.now():%H:%M:%S}] train done ({time.perf_counter()-t0:.0f}s)", flush=True)
    if not os.path.exists(os.path.join(outdir, "eval", "metrics.json")):
        t0 = time.perf_counter()
        print(f"[{datetime.now():%H:%M:%S}] EVAL {label}", flush=True)
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done ({time.perf_counter()-t0:.0f}s)", flush=True)
    return outdir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--only", type=str, default=None)
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model_comparisons] device={device}", flush=True)
    if a.smoke:
        epochs = a.epochs or 2
        cells = [c for c in CELLS if c[1] in ("vit_Na007b", "Na007b_L2")]
        print(f"[SMOKE] {len(cells)} cells, {epochs} epochs", flush=True)
    else:
        epochs = a.epochs or 30
        cells = [c for c in CELLS if (a.only is None or c[1] == a.only)]
    ok, fail = [], []
    for exp, label, sample, arch, n_layers, c1d in cells:
        try:
            run_cell(exp, label, sample, arch, n_layers, c1d, epochs, device, smoke=a.smoke)
            ok.append(label)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[FAIL] {label}: {e!r}", flush=True); fail.append(label)
    print(f"\n[done] ok={ok}  fail={fail}", flush=True)


if __name__ == "__main__":
    main()
