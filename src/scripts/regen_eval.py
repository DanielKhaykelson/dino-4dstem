"""
regen_eval.py — re-run inference + figures + metrics on already-trained runs,
with the new dense-remap and class-averages always included. No retraining.

Good runs (as of 2026-04-22 noon):
  - Na007b/sweep_polar_centroid       (overnight winner)
  - Na007b/config_c_K6_asym_tempsched (polar pairwise baseline)
  - Na006a/winner_polar_centroid
  - EuInAs_B100/winner_polar_centroid
  - Na007a/transfer_from_winner       (uses Na007b winner's checkpoint)

IMC runs are still training in background — re-run this script for them once
they finish.
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import os
import sys
import time
import traceback
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_contrastive import evaluate_and_report


GOOD_RUNS = [
    # (sample,                config_folder,                ckpt_path_override or None)
    ("Na007b",          "sweep_polar_centroid",             None),
    ("Na007b",          "config_c_K6_asym_tempsched",       None),
    ("Na006a",          "winner_polar_centroid",            None),
    ("EuInAs_B100",     "winner_polar_centroid",            None),
    ("IMC_50nm_SI2",    "winner_polar_centroid",            None),
    ("IMC_150nm_SI5",   "winner_polar_centroid",            None),
    ("Na007a",          "transfer_from_winner",
     os.path.join("runs", "Na007b", "sweep_polar_centroid", "best.pth")),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)
    base = os.path.dirname(os.path.abspath(__file__))
    for sample, config, ckpt in GOOD_RUNS:
        outdir = os.path.join(base, "runs", sample, config)
        if not os.path.isdir(outdir):
            print(f"[skip] {sample}/{config}  (dir missing)", flush=True)
            continue
        ckpt_path = os.path.abspath(ckpt) if ckpt else None
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] REGEN {sample}/{config}", flush=True)
        try:
            evaluate_and_report(
                config_key="c", sample=sample, outdir=outdir,
                device=device, ckpt_path=ckpt_path,
            )
            print(f"[{datetime.now():%H:%M:%S}] REGEN done in "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        except Exception as exc:
            print(f"[{datetime.now():%H:%M:%S}] REGEN FAILED "
                  f"{sample}/{config}: {exc!r}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
