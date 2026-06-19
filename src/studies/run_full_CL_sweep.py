"""
run_full_CL_sweep.py — Apply the ROI-split-trained checkpoint to ALL
same-CL measurements per sample family and collect per-SI statistics.

Runs only if the ROI-split transfer test passed (i.e. user has reviewed
runs/roi_split/*/transfer_summary.json and confirmed). This script
doesn't re-check; it just does inference.

For each sample family:
    apply trained-on-1-SI checkpoint to all same-CL SIs
    save per-SI metrics + class map + class averages
    aggregate into a summary table

Target datasets (58mm CL, matching the ROI-split training CL):
    NaPHI   (240312-NaPHI-Nadja-remeasure):      SI-001..SI-010
    MgNaPHI (240312-MgNaPHI-remeasure):          SI-001..SI-012

Outputs:
    runs/full_CL_sweep/NaPHI_Nadja/eval_NaPHI_Nadja_SI00X/eval/...
    runs/full_CL_sweep/NaPHI_Nadja/sweep_summary.json
    runs/full_CL_sweep/NaPHI_Nadja/fig_prototype_usage_across_SIs.png
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

import numpy as np
import torch
from run_contrastive import evaluate_and_report


def eval_on(eval_sample: str, ckpt_path: str, eval_outdir: str, device) -> None:
    sentinel = os.path.join(eval_outdir, "eval", "metrics.json")
    if os.path.exists(sentinel):
        print(f"[skip] {eval_outdir}", flush=True)
        return
    os.makedirs(eval_outdir, exist_ok=True)
    t0 = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] EVAL {eval_sample}", flush=True)
    try:
        evaluate_and_report("c", sample=eval_sample, outdir=eval_outdir,
                             device=device, ckpt_path=ckpt_path)
        print(f"  done ({time.perf_counter()-t0:.0f}s)", flush=True)
    except Exception as exc:
        print(f"  ERROR on {eval_sample}: {exc!r}", flush=True)


def run_family(family_name: str, ckpt_path: str, sample_keys: list[str],
               device) -> dict:
    """Apply ckpt to each sample in sample_keys, collect metrics."""
    base = os.path.join("runs", "full_CL_sweep", family_name)
    os.makedirs(base, exist_ok=True)

    per_si = {}
    for sk in sample_keys:
        eval_dir = os.path.join(base, f"eval_{sk}")
        eval_on(sk, ckpt_path, eval_dir, device)
        mj = os.path.join(eval_dir, "eval", "metrics.json")
        if not os.path.exists(mj):
            print(f"  no metrics for {sk}"); continue
        with open(mj) as f:
            content = f.read().replace("NaN", "null")
        m = json.loads(content)
        per_si[sk] = {
            "K_active": m.get("K_active", m.get("active_prototypes")),
            "K_original": m.get("K_original"),
            "effective_K": m.get("effective_K"),
            "KNN_purity_k10": m.get("KNN_purity_k10"),
            "intra_over_inter": m.get("intra_over_inter"),
            "proto_counts": m.get("proto_counts"),
            "stripe_scores": m.get("stripe_scores"),
            "dense_new_to_old_id": m.get("dense_new_to_old_id"),
            "dead_prototypes": m.get("dead_prototypes"),
        }

    # Aggregate: per-original-prototype fraction across all SIs, so we can
    # plot a prototype-usage heatmap (rows=SIs, cols=original prototype ids).
    all_orig_ids = set()
    for r in per_si.values():
        for oid in (r.get("dense_new_to_old_id") or []):
            all_orig_ids.add(oid)
    orig_ids_sorted = sorted(all_orig_ids)
    heat = []
    si_labels = []
    for sk, r in per_si.items():
        counts = r.get("proto_counts") or []
        mapping = r.get("dense_new_to_old_id") or []
        total = sum(counts) or 1
        row = [0.0] * len(orig_ids_sorted)
        for count, oid in zip(counts, mapping):
            if oid in orig_ids_sorted:
                row[orig_ids_sorted.index(oid)] = count / total
        heat.append(row)
        si_labels.append(sk)

    # Save summary
    summary = {
        "family": family_name,
        "ckpt_path": ckpt_path,
        "per_si_metrics": per_si,
        "orig_proto_ids": orig_ids_sorted,
        "si_labels": si_labels,
        "usage_heatmap": heat,  # rows=SIs, cols=orig_proto_ids
    }
    out = os.path.join(base, "sweep_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib.pyplot as plt
        heat_arr = np.asarray(heat, dtype=np.float32)
        fig, ax = plt.subplots(figsize=(1.2 * max(4, len(orig_ids_sorted)),
                                         0.35 * max(4, len(si_labels)) + 2))
        im = ax.imshow(heat_arr, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(orig_ids_sorted)))
        ax.set_xticklabels([f"p{i}" for i in orig_ids_sorted], fontsize=9)
        ax.set_yticks(range(len(si_labels)))
        ax.set_yticklabels(si_labels, fontsize=8)
        ax.set_title(f"{family_name}: prototype usage fraction per SI\n"
                      f"(original-prototype IDs from training checkpoint)",
                      fontsize=10)
        for i in range(len(si_labels)):
            for j in range(len(orig_ids_sorted)):
                v = heat_arr[i, j]
                if v > 0.02:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if v < 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="fraction")
        fig_path = os.path.join(base, "fig_prototype_usage_across_SIs.png")
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[heatmap] {fig_path}")
    except Exception as exc:
        print(f"  [heatmap] {exc!r}")

    print(f"[summary] {out}")
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sweep] device={device}", flush=True)

    # NaPHI family
    naphi_ckpt = os.path.join("runs", "roi_split", "NaPHI_Nadja",
                                "train1_SI5", "best.pth")
    if not os.path.exists(naphi_ckpt):
        print(f"[skip NaPHI] no ckpt at {naphi_ckpt}")
    else:
        naphi_sis = [f"NaPHI_Nadja_SI{i:03d}" for i in range(1, 11)]
        run_family("NaPHI_Nadja", naphi_ckpt, naphi_sis, device)

    # MgNaPHI family
    mgnaphi_ckpt = os.path.join("runs", "roi_split", "MgNaPHI_remeas",
                                 "train1_SI3", "best.pth")
    if not os.path.exists(mgnaphi_ckpt):
        print(f"[skip MgNaPHI] no ckpt at {mgnaphi_ckpt}")
    else:
        # MgNaPHI has SI-001..SI-012 (all 58mm CL).
        mgnaphi_sis = [f"MgNaPHI_remeas_SI{i:03d}" for i in range(1, 13)]
        run_family("MgNaPHI_remeas", mgnaphi_ckpt, mgnaphi_sis, device)

    print("\n[sweep] all done", flush=True)


if __name__ == "__main__":
    main()
