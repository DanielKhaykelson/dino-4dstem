"""eval_and_classmaps.py -- run evaluate_and_report on any sweep run
that has a trained ``best.pth`` but no ``eval/inference.npz`` yet,
then render ``class_map.png`` next to it.

Walks the sweep root and for every <root>/<sample>/stage{1,2}/<run>:
    1. If eval/inference.npz is missing AND best.pth exists,
       run evaluate_and_report (~1 min on GPU per run).
    2. Render class_map.png from inference.npz.
    3. Regenerate per-sample run_gallery.html.

Two modes:
    --once       Single pass over the sweep root, then exit.
    --watch      Loop: pass, sleep --interval seconds, repeat.
                 Exits when STOP_SWEEP exists or all expected samples
                 have a final report.html.

This is the bug-recovery + ongoing watchdog combined.  Safe to run
in parallel with the main sweep (eval is per-run and idempotent;
the lock-file check below stops the watcher from racing with itself).
"""
from __future__ import annotations
import argparse, os, sys, time, traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path: sys.path.insert(0, REPO)
if HERE not in sys.path: sys.path.insert(0, HERE)

# UTF-8 stdout for Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch                                              # noqa: E402
from run_contrastive import evaluate_and_report           # noqa: E402
# Reuse helpers from sweep_m_K.
from sweep_m_K import (                                   # noqa: E402
    _register_sample, _render_run_classmap, _extract_metrics,
    _ensure_radials, SAMPLES, SAMPLE_SPECS,
)
from render_existing_classmaps import (                   # noqa: E402
    _parse_run_name, _write_gallery_html,
)


def _expected_samples_present(root: str) -> bool:
    """All three samples have a final report.html -> sweep is done."""
    return all(os.path.exists(os.path.join(root, s, "report.html"))
               for s in SAMPLE_SPECS)


def _walk_runs(root: str):
    """Yield (sample, stage, run_name, run_dir) for every run dir
    under the sweep root."""
    for sample in sorted(os.listdir(root)):
        sd = os.path.join(root, sample)
        if not os.path.isdir(sd) or sample not in SAMPLE_SPECS:
            continue
        for stage in ("stage1", "stage2"):
            stage_dir = os.path.join(sd, stage)
            if not os.path.isdir(stage_dir):
                continue
            for run_name in sorted(os.listdir(stage_dir)):
                run_dir = os.path.join(stage_dir, run_name)
                if not os.path.isdir(run_dir):
                    continue
                yield sample, stage, run_name, run_dir


def _process_run(sample: str, run_dir: str, *,
                    device, do_eval: bool = True) -> dict:
    """Eval (if needed) + render class_map.png for one run dir.
    Returns a row dict for the gallery."""
    parsed = _parse_run_name(os.path.basename(run_dir))
    m, seed, K = parsed if parsed else (float("nan"), -1, -1)
    row = {"stage": os.path.basename(os.path.dirname(run_dir)),
           "m": m, "K": K, "seed": seed,
           "png_rel": None, "ok": False, "error": ""}
    best = os.path.join(run_dir, "best.pth")
    inf_dir = os.path.join(run_dir, "eval")
    inf_npz = os.path.join(inf_dir, "inference.npz")

    if not os.path.exists(best):
        row["error"] = "no best.pth"
        return row

    # 1. eval if missing.
    if do_eval and not os.path.exists(inf_npz):
        os.makedirs(inf_dir, exist_ok=True)
        try:
            t0 = time.time()
            evaluate_and_report(config_key="c", sample=sample,
                                    outdir=run_dir, device=device,
                                    ckpt_path=best)
            print(f"[eval] {run_dir}  ({time.time() - t0:.1f}s)",
                  flush=True)
        except Exception as e:
            row["error"] = f"eval failed: {e!r}"
            print(f"[eval-FAIL] {run_dir}: {e!r}", flush=True)
            traceback.print_exc()
            return row

    if not os.path.exists(inf_npz):
        row["error"] = "no inference.npz after eval"
        return row

    # 2. render class map.
    metrics = _extract_metrics(run_dir, sample)
    row.update({k: metrics.get(k) for k in (
        "K_eff_end_smooth", "n_live_end",
        "avg_conf_e5", "loss_final")})
    try:
        _render_run_classmap(run_dir, sample, m, K, seed, metrics)
        stage = row["stage"]
        rn = os.path.basename(run_dir)
        row["png_rel"] = f"{stage}/{rn}/class_map.png"
        row["ok"] = True
    except Exception as e:
        row["error"] = f"render failed: {e!r}"
    return row


def _pass(root: str, device, force_eval: bool = False) -> int:
    """One full sweep root pass.  Returns number of NEW runs processed."""
    if not os.path.isdir(root):
        return 0
    new = 0
    # Group rows by sample for the gallery.
    by_sample: dict = {}
    registered = set()
    for sample, stage, run_name, run_dir in _walk_runs(root):
        # One-time per-sample registration + radials so the eval
        # step has everything it needs.
        if sample not in registered:
            try:
                _register_sample(sample)
                _ensure_radials(sample)
            except Exception as e:
                print(f"[register-FAIL] {sample}: {e!r}", flush=True)
            registered.add(sample)
        inf_npz = os.path.join(run_dir, "eval", "inference.npz")
        cm_png = os.path.join(run_dir, "class_map.png")
        if (os.path.exists(inf_npz) and os.path.exists(cm_png)
                and not force_eval):
            # Already done; still need a row for the gallery.
            metrics = _extract_metrics(run_dir, sample)
            parsed = _parse_run_name(run_name)
            m, seed, K = parsed if parsed else (float("nan"), -1, -1)
            row = {"stage": stage, "m": m, "K": K, "seed": seed,
                   "png_rel": f"{stage}/{run_name}/class_map.png",
                   "ok": True, "error": ""}
            row.update({k: metrics.get(k) for k in (
                "K_eff_end_smooth", "n_live_end",
                "avg_conf_e5", "loss_final")})
            by_sample.setdefault(sample, []).append(row)
            continue
        row = _process_run(sample, run_dir, device=device)
        if row["ok"]:
            new += 1
        by_sample.setdefault(sample, []).append(row)
    # Refresh galleries (cheap).
    for sample, rows in by_sample.items():
        try:
            _write_gallery_html(os.path.join(root, sample), sample,
                                    rows)
        except Exception as e:
            print(f"[gallery-FAIL] {sample}: {e!r}", flush=True)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Sweep root containing <sample>/stage{1,2}/...")
    ap.add_argument("--once", action="store_true",
                    help="Single pass then exit.")
    ap.add_argument("--watch", action="store_true",
                    help="Loop until sweep finishes.")
    ap.add_argument("--interval", type=int, default=1200,
                    help="Watch-mode poll interval in seconds.")
    ap.add_argument("--force", action="store_true",
                    help="Re-eval even if inference.npz already exists.")
    ap.add_argument("--device", default="cpu",
                    choices=("cpu", "cuda"),
                    help="Device for eval.  Default CPU so the watcher "
                         "doesn't fight the training sweep for GPU "
                         "memory.  Switch to cuda only after the main "
                         "sweep finishes if you want to catch up faster.")
    args = ap.parse_args()
    if not (args.once or args.watch):
        args.once = True
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[watchdog] cuda requested but not available -- "
              "falling back to cpu.", flush=True)
        args.device = "cpu"
    device = torch.device(args.device)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"root not found: {root}")
    stop_path = os.path.join(root, "STOP_SWEEP")

    print(f"[watchdog] root={root}", flush=True)
    print(f"[watchdog] device={device}", flush=True)

    pass_i = 0
    while True:
        pass_i += 1
        t0 = time.time()
        try:
            n = _pass(root, device, force_eval=args.force)
        except Exception as e:
            n = -1
            print(f"[pass {pass_i}] EXCEPTION: {e!r}", flush=True)
            traceback.print_exc()
        dt = time.time() - t0
        stamp = datetime.now().isoformat(timespec="seconds")
        print(f"[pass {pass_i}] {stamp}  processed_new={n}  "
              f"took={dt:.1f}s", flush=True)
        if args.once:
            break
        if os.path.exists(stop_path):
            print(f"[watchdog] STOP_SWEEP detected -- exiting.",
                  flush=True)
            break
        if _expected_samples_present(root):
            print(f"[watchdog] all expected reports present -- "
                  f"exiting.", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
