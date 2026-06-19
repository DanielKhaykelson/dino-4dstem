"""run_paper_figures.py -- post-training figure generation for the
paper master sweep. For each run dir under runs/_paper_master/:
    1. viz_paper_outputs.py -- adaptive class map, per-class avg single
       image, 100 examples per class in subfolders.
    2. viz_paper_attribution.py -- multipanel GradCAM/IG figure
       (class avg + 3 samples per class, smoothed).

Auto-detects the sample from the run-dir label "<sample>_K{K}".
"""
from __future__ import annotations
import os, sys, time, traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import viz_paper_outputs
import viz_paper_attribution

ROOT = os.path.join("runs", "_paper_master")


def _label_to_sample(label: str) -> str:
    return label.rsplit("_K", 1)[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[paper figures] device={device}  root={ROOT}", flush=True)
    labels = sorted(d for d in os.listdir(ROOT)
                     if os.path.isdir(os.path.join(ROOT, d))
                     and "_K" in d)
    print(f"labels ({len(labels)}): {labels}", flush=True)
    t_total = time.perf_counter()
    for label in labels:
        run_dir = os.path.join(ROOT, label)
        sample = _label_to_sample(label)
        if not os.path.exists(os.path.join(run_dir, "eval", "metrics.json")):
            print(f"[skip] {label}: eval not done yet", flush=True)
            continue
        print(f"\n[{datetime.now():%H:%M:%S}] === {label} ({sample}) ===",
              flush=True)
        t0 = time.perf_counter()
        try:
            viz_paper_outputs.render_class_map(run_dir, sample)
            viz_paper_outputs.render_class_averages_and_examples(
                run_dir, sample, n_examples=100)
        except Exception as e:
            print(f"[FAIL paper-outputs] {label}: {e!r}", flush=True)
            traceback.print_exc()
        try:
            viz_paper_attribution.run(run_dir, sample,
                                       n_samples_per_proto=3,
                                       attribution_sigma=2.0,
                                       ig_steps=50,
                                       device=device)
        except Exception as e:
            print(f"[FAIL paper-attribution] {label}: {e!r}", flush=True)
            traceback.print_exc()
        print(f"[{datetime.now():%H:%M:%S}] {label} done in "
              f"{time.perf_counter()-t0:.0f}s", flush=True)
    print(f"\n[paper figures] all done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
