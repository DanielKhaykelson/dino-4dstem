"""
run_imc_cross_transfer.py — train on one IMC film, apply to the other.

Produces two cross-transfer evals:
  - runs/IMC_150nm_SI5/transfer_from_50nm/  : 50nm-winner checkpoint applied
                                               to 150nm scan
  - runs/IMC_50nm_SI2/transfer_from_150nm/  : 150nm-winner checkpoint applied
                                               to 50nm scan

Uses evaluate_and_report() with ckpt_path overriding the in-outdir lookup.
Then runs compare_maps.compare against the native winner on the same target
sample, which gives agreement %, NMI, ARI, Hungarian-matched IoU table, and
a disagreement map.

The paper point (per user): this is crystal-level comparison, not statistical.
For each source-model prototype, the class_averages figure on the target tells
us whether the same diffraction motif is present in the target film; the IoU
table tells us which native-target class(es) it maps onto.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch


# source sample, target sample, label under target dir, source ckpt path
PAIRS = [
    ("IMC_50nm_SI2",  "IMC_150nm_SI5", "transfer_from_50nm",
     os.path.join("runs", "IMC_50nm_SI2",  "winner_polar_centroid", "best.pth")),
    ("IMC_150nm_SI5", "IMC_50nm_SI2",  "transfer_from_150nm",
     os.path.join("runs", "IMC_150nm_SI5", "winner_polar_centroid", "best.pth")),
]


def one_direction(source: str, target: str, label: str, ckpt_path: str,
                  device) -> None:
    from run_contrastive import evaluate_and_report
    from compare_maps import compare

    outdir = os.path.join("runs", target, label)
    metrics_path = os.path.join(outdir, "eval", "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[skip] {target}/{label}: already evaluated", flush=True)
    else:
        if not os.path.exists(ckpt_path):
            print(f"[fail] source checkpoint missing: {ckpt_path}", flush=True)
            return
        os.makedirs(outdir, exist_ok=True)
        print(f"\n{'=' * 72}", flush=True)
        print(f"[{datetime.now():%H:%M:%S}] CROSS {source} -> {target}  "
              f"(label={label})", flush=True)
        print(f"    ckpt: {ckpt_path}", flush=True)
        print('=' * 72, flush=True)
        t0 = time.perf_counter()
        try:
            evaluate_and_report(
                "c", sample=target, outdir=outdir, device=device,
                ckpt_path=ckpt_path,
            )
            print(f"[{datetime.now():%H:%M:%S}] DONE in "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        except Exception as exc:
            print(f"[fail] {target}/{label}: {exc!r}", flush=True)
            traceback.print_exc()
            return

    try:
        compare(target, old_config="winner_polar_centroid", new_config=label)
    except Exception as exc:
        print(f"[compare fail] {exc!r}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[imc_cross_transfer] device = {device}", flush=True)
    for (src, tgt, lab, ckpt) in PAIRS:
        one_direction(src, tgt, lab, ckpt, device)
    print(f"\n[{datetime.now():%H:%M:%S}] IMC cross-transfer sweep done.",
          flush=True)


if __name__ == "__main__":
    main()
