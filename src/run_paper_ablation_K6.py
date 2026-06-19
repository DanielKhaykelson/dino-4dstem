"""run_paper_ablation_K6.py — Comprehensive K=6 ablation across all 6 samples
with both gamma=0 (vanilla) and gamma=1 (weight) per sample.

User (2026-04-25): "Run all samples in a way that lets you get the most
information regarding what may hint at the weight. Run with and without weight.
Output everything (umap, IG etc). Load IMC with vmax=10."

Pipeline per sample (no probe -- we want both conditions for ground truth):
    1. Train vanilla (gamma=0) at K=6 for MAIN_EPOCHS -> vanilla/
    2. Eval (full report including class_quality, GradCAM, IG, UMAP, ...)
    3. Train weight (gamma=1) at K=6 for MAIN_EPOCHS -> weight/
    4. Eval

Output tree:
    dino_sr_contrastive/runs/_paper_ablation_K6/
        EuInAs_B100/
            vanilla/    (gamma=0, full eval + class_quality + gradcam + IG)
            weight/     (gamma=1, full eval ...)
        IMC_150nm_SI5/...
        Na007b/...
        IMC_50nm_SI2/...
        Na007a/...
        Na006a/...
        REPORT.md (cross-sample summary -- written by make_ablation_summary.py)

Sample order (longest first): EuInAs, IMC_150nm, Na007b, IMC_50nm, Na007a, Na006a.
IMC samples loaded with vmax=10 per user; others use SAMPLES defaults.
"""
from __future__ import annotations
import os, sys, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from run_contrastive import run_config, evaluate_and_report


MAIN_EPOCHS = 50
K = 6

# Sample order, longest training first
SAMPLES = [
    "EuInAs_B100",
    "IMC_150nm_SI5",
    "Na007b",
    "IMC_50nm_SI2",
    "Na007a",
    "Na006a",
]

# Per-sample vmax override (None = use SAMPLES dict default in data.py)
VMAX_OVERRIDE = {
    "IMC_50nm_SI2":  10.0,
    "IMC_150nm_SI5": 10.0,
}

CONDITIONS = [("vanilla", 0.0), ("weight", 1.0)]
OUT_ROOT = os.path.join("runs", "_paper_ablation_K6")


def _kwargs(gamma: float, vmax: "float | None"):
    return dict(
        epochs=MAIN_EPOCHS, seed=42, batch_size=128,
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
        vmax=vmax,
        polar_size=192, polar_mask_cols=30,
        pipeline="polar",
        centroid_lambda=0.05, centroid_margin=0.3,
        conf_weight_gamma=gamma,
        entropy_gate_override=None,
        lam_spatial=0.0,
        architecture="resnet", n_layers=1,
    )


def run_one_condition(sample: str, label: str, gamma: float, device) -> dict:
    outdir = os.path.join(OUT_ROOT, sample, label)
    sentinel_train = os.path.join(outdir, "best.pth")
    sentinel_eval = os.path.join(outdir, "eval", "metrics.json")
    vmax = VMAX_OVERRIDE.get(sample, None)
    if not os.path.exists(sentinel_train):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {sample}/{label} "
              f"(gamma={gamma}, K={K}, vmax_override={vmax})", flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device,
                   **_kwargs(gamma, vmax))
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
    else:
        print(f"[skip eval] {outdir}", flush=True)
    metrics = json.load(open(sentinel_eval))
    return {
        "sample": sample, "label": label, "gamma": gamma, "vmax": vmax,
        "intra_over_inter": metrics.get("intra_over_inter"),
        "effective_K": metrics.get("effective_K"),
        "active_prototypes": metrics.get("active_prototypes"),
        "KNN_purity_k10": metrics.get("KNN_purity_k10"),
        "stripe_max": metrics.get("stripe_max"),
        "outdir": outdir,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation K={K}] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}", flush=True)
    print(f"samples: {SAMPLES}  conditions: {[c[0] for c in CONDITIONS]}",
          flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    t_total = time.perf_counter()
    for sample in SAMPLES:
        for label, gamma in CONDITIONS:
            try:
                r = run_one_condition(sample, label, gamma, device)
                results.append(r)
                vmax_note = (f" (vmax={r['vmax']})" if r['vmax'] is not None
                             else "")
                print(f"[OK] {sample}/{label}{vmax_note}: "
                      f"intra/inter={r['intra_over_inter']:.2f}",
                      flush=True)
            except Exception as e:
                print(f"[FAIL] {sample}/{label}: {e!r}", flush=True)
                import traceback; traceback.print_exc()
    # Write a thin index file (cross-sample analysis lives in
    # make_ablation_summary.py, run after this completes).
    with open(os.path.join(OUT_ROOT, "results_index.json"), "w") as f:
        json.dump(results, f, indent=2)
    elapsed_min = (time.perf_counter() - t_total) / 60
    print(f"\n[ablation K={K}] all conditions done in {elapsed_min:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
