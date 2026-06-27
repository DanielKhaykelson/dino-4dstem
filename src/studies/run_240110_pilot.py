"""
run_240110_pilot.py — Zero-shot transfer of Na007b winner checkpoint
to two new SIs from the 240110-MgNaphi session (X: network drive).

Goal (Nano Letters pilot):
  Apply Na007b sweep_polar_centroid/best.pth to:
    NaPHI240110_SI006  — 115mm CL, 115k mag, focused probe, 10eV slit
    NaPHI240110_SI020  — same conditions, different flake/position

After eval: compare line-fraction per sample to assess whether the
Na007b-trained vocabulary generalises to new NaPHI/MgNaPHI material.

Run from: dino_sr_contrastive/  (conda env py4DSTEM_SAM)
    python run_240110_pilot.py
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from run_contrastive import evaluate_and_report

# ── config ────────────────────────────────────────────────────────────────────
CKPT_PATH = os.path.join(
    "runs", "Na007b", "sweep_polar_centroid", "best.pth"
)
PILOT_SAMPLES = [
    ("NaPHI240110_SI006", "240110_SI006_zs"),
    ("NaPHI240110_SI020", "240110_SI020_zs"),
]
BASE_OUTDIR = os.path.join("runs", "MgNaPHI_pilot")
# ──────────────────────────────────────────────────────────────────────────────


def extract_line_fraction(metrics_path: str) -> dict:
    """Read metrics.json and compute approximate line-fraction.

    For NaPHI family the 'line' classes tend to be those with high stripe_score.
    We flag prototypes with stripe_score > threshold as 'line-like'.
    Returns fraction of pixels in line-like classes.
    """
    if not os.path.exists(metrics_path):
        return {"error": "metrics.json not found"}
    with open(metrics_path) as f:
        content = f.read().replace("NaN", "null")
    m = json.loads(content)
    proto_counts = m.get("proto_counts", [])
    stripe_scores = m.get("stripe_scores", [])
    if not proto_counts or not stripe_scores:
        return {"error": "missing proto_counts or stripe_scores"}
    total = sum(proto_counts)
    if total == 0:
        return {"error": "zero total count"}
    # line-like = stripe_score > 2.0 (empirical; Na007b winner classes have
    # stripe_scores ~1.9-2.4; spot-like/amorphous classes tend to be lower)
    LINE_THRESH = 2.0
    line_count = sum(
        cnt for cnt, sc in zip(proto_counts, stripe_scores) if sc > LINE_THRESH
    )
    return {
        "total_pixels": total,
        "line_pixel_count": line_count,
        "line_fraction": line_count / total,
        "stripe_scores": stripe_scores,
        "proto_counts": proto_counts,
        "active_K": m.get("K_active", m.get("active_prototypes")),
        "KNN_purity": m.get("KNN_purity_k10"),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pilot] device = {device}", flush=True)
    print(f"[pilot] checkpoint = {CKPT_PATH}", flush=True)

    if not os.path.exists(CKPT_PATH):
        print(f"[pilot] ERROR: checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    os.makedirs(BASE_OUTDIR, exist_ok=True)
    results = {}

    for sample_key, label in PILOT_SAMPLES:
        outdir = os.path.join(BASE_OUTDIR, label)
        metrics_path = os.path.join(outdir, "eval", "metrics.json")

        if os.path.exists(metrics_path):
            print(f"\n[pilot] {label}: already done, reading metrics...", flush=True)
        else:
            print(f"\n[{datetime.now():%H:%M:%S}] START {sample_key} → {label}", flush=True)
            t0 = time.perf_counter()
            try:
                evaluate_and_report(
                    "c",
                    sample=sample_key,
                    outdir=outdir,
                    device=device,
                    ckpt_path=CKPT_PATH,
                )
                print(f"[{datetime.now():%H:%M:%S}] DONE in "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
            except Exception as exc:
                print(f"[pilot] ERROR on {label}: {exc!r}")
                results[label] = {"error": repr(exc)}
                continue

        lf = extract_line_fraction(metrics_path)
        results[label] = lf
        print(f"  line_fraction = {lf.get('line_fraction', 'N/A'):.3f}  "
              f"  active_K = {lf.get('active_K', 'N/A')}  "
              f"  KNN_purity = {lf.get('KNN_purity', 'N/A')}", flush=True)

    # ── summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PILOT SUMMARY")
    print("=" * 60)
    print(f"{'Sample':<30}  {'Line%':>7}  {'K_active':>8}  {'KNN_pur':>8}")
    print("-" * 60)
    for label, r in results.items():
        lf   = r.get("line_fraction")
        k    = r.get("active_K")
        knn  = r.get("KNN_purity")
        err  = r.get("error")
        if err:
            print(f"{label:<30}  ERROR: {err}")
        else:
            print(f"{label:<30}  {lf*100:>6.1f}%  {str(k):>8}  {knn:>8.4f}")
    print("=" * 60)

    # Save summary JSON
    summary_path = os.path.join(BASE_OUTDIR, "pilot_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[pilot] summary → {summary_path}")


if __name__ == "__main__":
    main()
