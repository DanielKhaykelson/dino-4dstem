"""
run_roi_split_trainall.py — The "ceiling" comparison for the ROI-split test.

Trains on all 4 SIs combined (instead of just 1), then evaluates on each SI
individually. Compared against run_roi_split.py's train-on-1 results this
tells us whether training on MORE DATA materially changes the class map.

If train-1 ≈ trainALL per-SI → training on 1 ROI is sufficient (transfer works).
If train-1 << trainALL                    → more-data helps, but maybe still OK.

Requires ~40GB of RAM (4 × 10GB PRZ cubes held simultaneously via LoadPRZMulti).
Must run with NO OTHER Python processes holding GPU/RAM.

Prereq: run_roi_split.py has already finished (so GPU/RAM free).
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

import numpy as np
import torch
from run_contrastive import run_config, evaluate_and_report
from run_roi_split import WINNER_KWARGS, compare_inference


def train(train_sample: str, outdir: str, device) -> None:
    sentinel = os.path.join(outdir, "best.pth")
    if os.path.exists(sentinel):
        print(f"[skip] {outdir}  (best.pth exists)", flush=True)
        return
    os.makedirs(outdir, exist_ok=True)
    t0 = time.perf_counter()
    print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {train_sample} -> {outdir}",
          flush=True)
    run_config("c", sample=train_sample, outdir=outdir, device=device, **WINNER_KWARGS)
    print(f"[{datetime.now():%H:%M:%S}] train done ({time.perf_counter()-t0:.0f}s)",
          flush=True)


def eval_on(eval_sample: str, ckpt_outdir: str, eval_outdir: str, device) -> None:
    sentinel = os.path.join(eval_outdir, "eval", "metrics.json")
    if os.path.exists(sentinel):
        print(f"[skip] {eval_outdir}", flush=True)
        return
    os.makedirs(eval_outdir, exist_ok=True)
    ckpt = os.path.join(ckpt_outdir, "best.pth")
    t0 = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] EVAL {eval_sample}", flush=True)
    evaluate_and_report("c", sample=eval_sample, outdir=eval_outdir,
                        device=device, ckpt_path=ckpt)
    print(f"  done ({time.perf_counter()-t0:.0f}s)", flush=True)


def run_family(family_name: str,
               trainALL_sample: str,
               trainALL_dirname: str,
               train1_dirname: str,
               test_samples: list[str],
               device) -> dict:
    base = os.path.join("runs", "roi_split", family_name)
    trainALL_dir = os.path.join(base, trainALL_dirname)
    train1_dir   = os.path.join(base, train1_dirname)

    # Train-on-all-4.
    train(trainALL_sample, trainALL_dir, device)

    # Eval on each test SI.
    for tsi in test_samples:
        eval_on(tsi, trainALL_dir, os.path.join(trainALL_dir, f"eval_{tsi}"),
                device)

    # Compare train1-on-SI-j vs trainALL-on-SI-j for each j.
    comparisons = {}
    for tsi in test_samples:
        old = os.path.join(train1_dir,   f"eval_{tsi}", "eval", "inference.npz")
        new = os.path.join(trainALL_dir, f"eval_{tsi}", "eval", "inference.npz")
        if not (os.path.exists(old) and os.path.exists(new)):
            print(f"[cmp] missing npz for {tsi}")
            continue
        out = os.path.join(base, f"compare_train1_vs_trainALL_on_{tsi}.json")
        r = compare_inference(old, new,
                              label_ref=f"train1_on_{tsi}",
                              label_tgt=f"trainALL_on_{tsi}",
                              outpath=out)
        comparisons[tsi] = r
        print(f"  {tsi}: meanIoU={r['mean_matched_iou']:.3f}  "
              f"NMI={r['NMI']:.3f}  agree={r.get('agreement_fraction'):.3f}  "
              f"histL1={r['hist_L1_distance']:.3f}", flush=True)

    summary = {
        "family": family_name, "trainALL_sample": trainALL_sample,
        "train1_dirname": train1_dirname, "trainALL_dirname": trainALL_dirname,
        "test_samples": test_samples,
        "train1_vs_trainALL_per_SI": comparisons,
    }
    out = os.path.join(base, "train1_vs_trainALL_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[summary] {out}", flush=True)
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[roi_split_trainall] device={device}", flush=True)

    # NaPHI: 4 SIs combined = 40k patterns, ~40GB RAM peak.
    naphi = run_family(
        family_name="NaPHI_Nadja",
        trainALL_sample="NaPHI_Nadja_4Q_all4",
        trainALL_dirname="trainALL_4Q",
        train1_dirname="train1_SI5",
        test_samples=[f"NaPHI_Nadja_SI{i:03d}" for i in (5, 6, 7, 8)],
        device=device,
    )

    # MgNaPHI: 4 SIs combined
    mgnaphi = run_family(
        family_name="MgNaPHI_remeas",
        trainALL_sample="MgNaPHI_remeas_4Q_all4",
        trainALL_dirname="trainALL_4Q",
        train1_dirname="train1_SI3",
        test_samples=[f"MgNaPHI_remeas_SI{i:03d}" for i in (3, 4, 5, 6)],
        device=device,
    )

    print("\n" + "=" * 72)
    print("ROI-SPLIT TRAIN-1 vs TRAIN-ALL SUMMARY")
    print("=" * 72)
    for fam in (naphi, mgnaphi):
        print(f"\n[{fam['family']}]")
        for tsi, c in fam["train1_vs_trainALL_per_SI"].items():
            print(f"  {tsi}:  meanIoU={c['mean_matched_iou']:.3f}  "
                  f"NMI={c['NMI']:.3f}  histL1={c['hist_L1_distance']:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
