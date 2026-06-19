"""run_paper_K6_probe_commit.py — Probe-and-commit pipeline at K=6 across
all 5 samples for the paper.

K=6 standardization rationale: Na007b's validated winner (sweep_polar_centroid)
uses K=6, with K_active=K=6 -- every prototype is in use, no spare capacity.
At K=10 we observed basin-flip instability for EuInAs (effK=5 with 5 spare
prototypes; small init noise pushed the model between two stable equilibria,
intra/inter ranged 7.4 vs 24.6 for the same config). K=6 keeps every sample's
training in a saturated regime which is robust to noise. EuInAs at K=6+weight
already gave intra/inter=25.5, matching K=10+weight=24.6 -- no quality lost.

Pipeline per sample (uses train_contrastive's stop_at_epoch param so the probe
shares the LR / temperature schedule of a 50-epoch run):
    1. Train probe with epochs=50, stop_at_epoch=5, gamma=0 -> probe/
    2. Read ep5 avg_conf -> decide gamma_main (threshold 0.85)
    3. Train main 50-epoch run from scratch with decided gamma -> main/
    4. Save probe_decision.json
    5. Run evaluation (improved GradCAMs)

Output tree:
    dino_sr_contrastive/runs/_paper_K6_probe_commit/
        EuInAs_B100/
            probe/
            main/
                best.pth
                eval/
                    fig_*.png
                    gradcam/
            probe_decision.json
        Na007b/...
        Na006a/...
        IMC_50nm_SI2/...
        IMC_150nm_SI5/...
        REPORT.md

Sample order (per user 2026-04-25): EuInAs first, Na007b second, rest after.
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys, time, json, csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from run_contrastive import run_config, evaluate_and_report


THRESHOLD = 0.85
PROBE_EPOCHS = 5
MAIN_EPOCHS = 50
K = 6

# Order per user (2026-04-25): EuInAs, Na007b, then the rest
SAMPLES = ["EuInAs_B100", "Na007b", "Na006a", "IMC_50nm_SI2", "IMC_150nm_SI5"]

OUT_ROOT = os.path.join("runs", "_paper_K6_probe_commit")


def _kwargs(epochs: int, gamma: float, stop_at_epoch: int | None = None):
    return dict(
        epochs=epochs, seed=42, batch_size=128,
        lr=3e-4, weight_decay=1e-6,
        num_prototypes=K,
        t0=0.04, tfin=0.07,
        warmup_epochs=20, ramp_epochs=10,
        entropy_gate=False,
        projection_dim=128, projection_hidden=256,
        theta_shift_range=None,
        theta_shift_range_student=192, theta_shift_range_teacher=16,
        center_mask_radius=None,
        center_crop_size=140,
        vmax=None,
        polar_size=192, polar_mask_cols=30,
        pipeline="polar",
        centroid_lambda=0.05, centroid_margin=0.3,
        conf_weight_gamma=gamma,
        entropy_gate_override=None,
        lam_spatial=0.0,
        architecture="resnet", n_layers=1,
        stop_at_epoch=stop_at_epoch,
    )


def read_ep5_avg_conf(probe_dir: str) -> float:
    """Read avg_conf at ep5 from the probe's training_log.csv."""
    path = os.path.join(probe_dir, "training_log.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if int(row["epoch"]) == PROBE_EPOCHS:
            return float(row["avg_conf"])
    raise RuntimeError(f"epoch {PROBE_EPOCHS} not found in {path} "
                       f"(rows={len(rows)})")


def run_one_sample(sample: str, device) -> dict:
    sample_dir = os.path.join(OUT_ROOT, sample)
    os.makedirs(sample_dir, exist_ok=True)
    probe_dir = os.path.join(sample_dir, "probe")
    main_dir = os.path.join(sample_dir, "main")
    decision_path = os.path.join(sample_dir, "probe_decision.json")

    # ----- PROBE -----
    # The probe is the first PROBE_EPOCHS of a MAIN_EPOCHS-length run so that
    # the cosine LR / teacher-temperature schedules match what the main run
    # would experience. We pass epochs=MAIN_EPOCHS and stop_at_epoch=PROBE_EPOCHS
    # so the trainer breaks out cleanly after writing ep5's CSV row.
    probe_done = os.path.exists(os.path.join(probe_dir, "training_log.csv"))
    if not probe_done:
        os.makedirs(probe_dir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] PROBE {sample} "
              f"(stop after {PROBE_EPOCHS} ep, scheduler T_max={MAIN_EPOCHS}, gamma=0)",
              flush=True)
        run_config("c", sample=sample, outdir=probe_dir, device=device,
                   **_kwargs(MAIN_EPOCHS, 0.0, stop_at_epoch=PROBE_EPOCHS))
        print(f"[{datetime.now():%H:%M:%S}] probe done ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    else:
        print(f"[skip probe] {probe_dir}", flush=True)

    # ----- DECIDE -----
    avg_conf = read_ep5_avg_conf(probe_dir)
    gamma_main = 1.0 if avg_conf > THRESHOLD else 0.0
    decision = {
        "sample": sample,
        "K": K,
        "probe_epochs": PROBE_EPOCHS,
        "threshold": THRESHOLD,
        "ep5_avg_conf": avg_conf,
        "decision": "enable_weight" if gamma_main > 0 else "skip_weight",
        "gamma_main": gamma_main,
        "main_epochs": MAIN_EPOCHS,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(decision_path, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"[DECISION] {sample}: ep5 avg_conf={avg_conf:.4f}, "
          f"threshold={THRESHOLD} -> "
          f"{'+weight' if gamma_main>0 else 'vanilla'} (gamma={gamma_main})",
          flush=True)

    # ----- MAIN -----
    main_done = os.path.exists(os.path.join(main_dir, "best.pth"))
    if not main_done:
        os.makedirs(main_dir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] MAIN {sample} "
              f"({MAIN_EPOCHS} ep, gamma={gamma_main})", flush=True)
        run_config("c", sample=sample, outdir=main_dir, device=device,
                   **_kwargs(MAIN_EPOCHS, gamma_main))
        print(f"[{datetime.now():%H:%M:%S}] main done ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    else:
        print(f"[skip main] {main_dir}", flush=True)

    # ----- EVAL -----
    eval_sentinel = os.path.join(main_dir, "eval", "metrics.json")
    if not os.path.exists(eval_sentinel):
        t0 = time.perf_counter()
        print(f"\n[{datetime.now():%H:%M:%S}] EVAL {sample}", flush=True)
        evaluate_and_report("c", sample=sample, outdir=main_dir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    else:
        print(f"[skip eval] {main_dir}", flush=True)

    # Pull final intra/inter for the report
    metrics = json.load(open(eval_sentinel))
    return {
        "sample": sample,
        "decision": decision,
        "intra_over_inter": metrics.get("intra_over_inter"),
        "effective_K": metrics.get("effective_K"),
        "active_prototypes": metrics.get("active_prototypes"),
        "KNN_purity_k10": metrics.get("KNN_purity_k10"),
        "main_dir": main_dir,
    }


def write_report(results: list, root: str):
    path = os.path.join(root, "REPORT.md")
    lines = [
        f"# Paper run: K=10 probe-and-commit",
        f"",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"",
        f"Pipeline: 5-epoch gamma=0 probe -> threshold {THRESHOLD} on ep5 avg_conf -> 50-epoch main.",
        f"All samples at K=10.",
        f"",
        f"## Decision table",
        f"",
        f"| Sample | ep5 avg_conf | Decision | gamma | intra/inter | effK / K_active | KNN purity |",
        f"|---|---:|:---:|:---:|---:|:---:|---:|",
    ]
    for r in results:
        d = r["decision"]
        lines.append(
            f"| {r['sample']} | {d['ep5_avg_conf']:.4f} | "
            f"{'+weight' if d['gamma_main']>0 else 'vanilla'} | "
            f"{d['gamma_main']:.1f} | "
            f"{r['intra_over_inter']:.2f} | "
            f"{r['effective_K']:.2f} / {r['active_prototypes']} | "
            f"{r['KNN_purity_k10']:.4f} |"
        )
    lines += [
        f"",
        f"## Files",
        f"",
    ]
    for r in results:
        lines.append(f"- {r['sample']}: `{r['main_dir']}/eval/`  (decision: `{os.path.dirname(r['main_dir'])}/probe_decision.json`)")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {path}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[paper K=6 probe-and-commit] device={device}", flush=True)
    print(f"output root: {OUT_ROOT}", flush=True)
    print(f"samples: {SAMPLES}", flush=True)

    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    t_total = time.perf_counter()
    for sample in SAMPLES:
        try:
            res = run_one_sample(sample, device)
            results.append(res)
            print(f"[OK] {sample}: intra/inter={res['intra_over_inter']:.2f}",
                  flush=True)
        except Exception as e:
            print(f"[FAIL] {sample}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
    write_report(results, OUT_ROOT)
    print(f"\n[paper K=10] all samples done in "
          f"{(time.perf_counter()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
