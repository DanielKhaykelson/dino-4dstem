"""sweep_stage2_seed7.py -- run Stage 2 cells at seed=7 for every
sample, reusing the existing top-m candidates from each sample's
seed=42 Stage 2 directory.

The main sweep launches Stage 2 at a single seed (42).  This adds a
parallel set of Stage 2 runs at seed=7 so cross-seed variance can be
read off the plateau plot directly.

The script picks each sample's existing top-m candidates by listing
``<root>/<sample>/stage2/`` (whatever m values are present from the
seed=42 run).  Same K grid.
"""
from __future__ import annotations
import argparse, csv, os, sys, time, traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

from sweep_m_K import (                               # noqa: E402
    _launch_one, _register_sample, _ensure_radials,
    _extract_metrics, _append_progress, _render_run_classmap,
    SWEEP_SPEC, SAMPLE_SPECS, SAMPLES,
)
from cross_sample_classmaps import _top_m_for_sample  # noqa: E402


def _process_one_sample(root: str, sample: str, seed: int = 7):
    _register_sample(sample)
    rad_path, th_path = _ensure_radials(sample)
    sample_dir = os.path.join(root, sample)
    stage_dir = os.path.join(sample_dir, "stage2")
    os.makedirs(stage_dir, exist_ok=True)
    top_ms = _top_m_for_sample(root, sample)
    if not top_ms:
        print(f"[skip-sample] {sample}: no stage2 top-m dirs",
              flush=True)
        return
    K_grid = SWEEP_SPEC["K_grid_stage2"]
    print(f"\n{'='*72}\n[seed7-stage2] {sample}  top-m={top_ms}  "
          f"K_grid={K_grid}\n{'='*72}", flush=True)
    for m in top_ms:
        for K in K_grid:
            name = f"m{m:.4f}_seed{seed}_K{K}"
            outdir = os.path.join(stage_dir, name)
            if (os.path.exists(os.path.join(outdir, "best.pth")) and
                    os.path.exists(os.path.join(outdir,
                                                 "eval/inference.npz"))):
                print(f"[skip] {name} already done.", flush=True)
                continue
            stop = os.path.join(root, "STOP_SWEEP")
            if os.path.exists(stop):
                print(f"[stop] STOP_SWEEP detected -- exiting.",
                      flush=True)
                return
            print(f"\n[run] {sample}  m={m:g}  K={K}  seed={seed} "
                  f"  -> {outdir}", flush=True)
            r = _launch_one(sample, m, K, seed, outdir,
                                dry_run=False,
                                rad_path=rad_path, th_path=th_path)
            r["stage"] = "stage2"
            _append_progress(root, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--samples", nargs="*", default=None,
                    help="Limit to these samples (default: all that "
                         "have stage2 dirs in the root).")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"root not found: {root}")
    samples = (args.samples
               if args.samples
               else [s for s in SWEEP_SPEC["sample_order"]
                     if os.path.isdir(os.path.join(root, s, "stage2"))])
    print(f"[main] root={root}  seed={args.seed}  samples={samples}",
          flush=True)
    t0 = time.time()
    for s in samples:
        try:
            _process_one_sample(root, s, seed=args.seed)
        except KeyboardInterrupt:
            print("[stop] keyboard interrupt"); break
        except Exception as e:
            print(f"[sample-FAIL] {s}: {e!r}", flush=True)
            traceback.print_exc()
    print(f"\n[main] done in {(time.time() - t0) / 3600:.1f} h",
          flush=True)


if __name__ == "__main__":
    main()
