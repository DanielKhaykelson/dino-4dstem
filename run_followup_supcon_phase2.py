"""run_followup_supcon_phase2.py — Phase 2: Na007a, Na006a with same params
as Na007b run.

Output goes to runs/_followup_supcon/<sample>/main, same dir naming as phase 1.
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

# Import the helper from phase 1
from run_followup_supcon import _kwargs, run_one, OUT_ROOT, SUPCON_LAMBDA

SAMPLE_LIST = ["Na007a", "Na006a"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[supcon phase 2] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}", flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()
    for s in SAMPLE_LIST:
        try:
            run_one(s, device)
        except Exception as e:
            print(f"[FAIL] {s}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
    print(f"\n[supcon phase 2] done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
