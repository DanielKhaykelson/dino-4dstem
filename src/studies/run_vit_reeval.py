"""
run_vit_reeval.py — re-run evaluate_and_report on the Na007b ViT checkpoint
that failed earlier due to the state_dict-architecture detection bug
(now fixed in load_contrastive_checkpoint).
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
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import torch
from run_contrastive import evaluate_and_report
from compare_maps import compare


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vit_reeval] device = {device}", flush=True)
    sample = "Na007b"
    label = "winner_vit"
    outdir = os.path.join("runs", sample, label)
    t0 = time.perf_counter()
    print(f"\n[{datetime.now():%H:%M:%S}] START re-eval {sample}/{label}", flush=True)
    evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
    print(f"[{datetime.now():%H:%M:%S}] DONE in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    try:
        compare(sample, old_config="sweep_polar_centroid", new_config="winner_vit")
    except Exception as exc:
        print(f"  [compare] {exc!r}")


if __name__ == "__main__":
    main()
