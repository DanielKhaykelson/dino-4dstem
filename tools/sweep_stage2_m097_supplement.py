"""sweep_stage2_m097_supplement.py -- ensure every sample has a
full K-plateau test at m=0.97 (both seeds).

The main sweep's Stage 2 only ran whichever m's were in each sample's
top-3 from Stage 1.  m=0.97 turned out to be the best universal-m
candidate but it's missing from EuInAs (top-3 was 0.85/0.9/0.95) and
IMC (top-3 was 0.5/0.85/0.9).  This script fills in those cells.

Skips anything that already has best.pth + eval/inference.npz.
"""
from __future__ import annotations
import argparse, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

from sweep_m_K import (                               # noqa: E402
    _launch_one, _register_sample, _ensure_radials,
    _append_progress, SWEEP_SPEC, SAMPLE_SPECS, SAMPLES,
)

TARGET_M = 0.97
SEEDS    = (42, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--samples", nargs="*",
                    default=["EuInAs_B100", "IMC_SI5", "Na007b"])
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"root not found: {root}")
    K_grid = SWEEP_SPEC["K_grid_stage2"]
    print(f"[main] m={TARGET_M}  K={K_grid}  seeds={SEEDS}  "
          f"samples={args.samples}", flush=True)

    t_start = time.time()
    for s in args.samples:
        try:
            _register_sample(s)
            rad_path, th_path = _ensure_radials(s)
        except Exception as e:
            print(f"[skip-sample] {s}: register/radials failed: {e!r}",
                  flush=True)
            continue
        stage_dir = os.path.join(root, s, "stage2")
        os.makedirs(stage_dir, exist_ok=True)
        for seed in SEEDS:
            for K in K_grid:
                name = f"m{TARGET_M:.4f}_seed{seed}_K{K}"
                outdir = os.path.join(stage_dir, name)
                done = (os.path.exists(os.path.join(outdir, "best.pth"))
                          and os.path.exists(os.path.join(
                              outdir, "eval/inference.npz")))
                if done:
                    print(f"[skip] {s} {name} already complete.",
                          flush=True)
                    continue
                if os.path.exists(os.path.join(root, "STOP_SWEEP")):
                    print("[stop] STOP_SWEEP detected", flush=True)
                    return
                print(f"\n[run] {s}  m={TARGET_M}  K={K}  "
                      f"seed={seed}  -> {outdir}", flush=True)
                try:
                    r = _launch_one(s, TARGET_M, K, seed, outdir,
                                        dry_run=False,
                                        rad_path=rad_path,
                                        th_path=th_path)
                    r["stage"] = "stage2"
                    _append_progress(root, r)
                except Exception as e:
                    print(f"[run-FAIL] {outdir}: {e!r}", flush=True)
                    traceback.print_exc()
    dt = (time.time() - t_start) / 3600
    print(f"\n[main] done in {dt:.1f} h", flush=True)


if __name__ == "__main__":
    main()
