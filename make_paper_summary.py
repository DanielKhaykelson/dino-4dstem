"""make_paper_summary.py — Generate cross-sample summary figures and the
final REPORT.md after the paper pipeline finishes.

Reads per-sample outputs from:
    runs/_paper_K10_probe_commit/<sample>/probe_decision.json
    runs/_paper_K10_probe_commit/<sample>/main/eval/metrics.json
    runs/_paper_K10_probe_commit/<sample>/probe/training_log.csv

Produces:
    runs/_paper_K10_probe_commit/fig_probe_decision.png    (avg_conf trajectories
                                                            + 0.85 threshold)
    runs/_paper_K10_probe_commit/fig_metric_bars.png       (intra/inter bars,
                                                            colored by decision)
    runs/_paper_K10_probe_commit/REPORT.md                 (overwrites the
                                                            simple report)
"""
from __future__ import annotations
import os, sys, json, csv
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "runs", "_paper_K6_probe_commit")

# Predicted ground-truth: where existing trainings indicate weight helps
GROUND_TRUTH = {
    "EuInAs_B100":  ("helps",   "5.14 -> 24.63 (5x)"),
    "Na007b":       ("hurts",   "21.49 -> 11.38 (-47%)"),
    "Na006a":       ("hurts",   "15.99 -> 14.98 (slight)"),
    "IMC_50nm_SI2": ("unknown", "no weight ablation pre-paper"),
    "IMC_150nm_SI5":("hurts",   "11.13 -> 8.95 (-20%)"),
}

THRESHOLD = 0.85


def load_sample(sample: str) -> dict | None:
    sd = os.path.join(ROOT, sample)
    dec_p = os.path.join(sd, "probe_decision.json")
    metrics_p = os.path.join(sd, "main", "eval", "metrics.json")
    probe_log = os.path.join(sd, "probe", "training_log.csv")
    if not (os.path.exists(dec_p) and os.path.exists(metrics_p)):
        print(f"[skip] {sample}: missing decision or metrics", flush=True)
        return None
    decision = json.load(open(dec_p))
    metrics = json.load(open(metrics_p))

    # probe avg_conf trajectory
    conf_traj = []
    if os.path.exists(probe_log):
        with open(probe_log, newline="") as f:
            for row in csv.DictReader(f):
                conf_traj.append((int(row["epoch"]),
                                  float(row["avg_conf"])))
    return {
        "sample": sample,
        "decision": decision,
        "metrics": metrics,
        "conf_traj": conf_traj,
    }


def fig_probe_decision(samples: list[dict], out_path: str):
    """avg_conf trajectories during probe, with 0.85 threshold line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(samples):
        traj = s["conf_traj"]
        if not traj: continue
        eps, conf = zip(*traj)
        d = s["decision"]
        marker = "o" if d["gamma_main"] > 0 else "s"
        ls = "-" if d["gamma_main"] > 0 else "--"
        label = (f"{s['sample']}: ep5 conf={d['ep5_avg_conf']:.3f} -> "
                 f"{'+weight' if d['gamma_main']>0 else 'vanilla'}")
        ax.plot(eps, conf, marker=marker, linestyle=ls, color=cmap(i),
                 markersize=8, linewidth=2, label=label)
    ax.axhline(THRESHOLD, color="k", linestyle=":", linewidth=1.5,
                label=f"threshold = {THRESHOLD}")
    ax.set_xlabel("Probe epoch", fontsize=12)
    ax.set_ylabel("avg_conf (mean teacher peak prob)", fontsize=12)
    ax.set_title("Probe-and-commit: avg_conf trajectories during 5-epoch probe",
                  fontsize=12)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def fig_metric_bars(samples: list[dict], out_path: str):
    """Bar chart of intra/inter, colored by decision (vanilla / +weight)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [s["sample"] for s in samples]
    vals = [s["metrics"]["intra_over_inter"] for s in samples]
    decisions = [s["decision"]["gamma_main"] > 0 for s in samples]
    confs = [s["decision"]["ep5_avg_conf"] for s in samples]
    colors = ["#1f77b4" if d else "#ff7f0e" for d in decisions]
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)
    for bar, v, c in zip(bars, vals, confs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                f"{v:.1f}\n(conf={c:.2f})",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("intra / inter (cosine ratio)", fontsize=12)
    ax.set_title("Final intra/inter per sample, K=10, decision via probe",
                  fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#1f77b4", edgecolor="black", label="+weight (γ=1)"),
              Patch(facecolor="#ff7f0e", edgecolor="black", label="vanilla (γ=0)")]
    ax.legend(handles=legend, fontsize=10, loc="upper right")
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def write_report(samples: list[dict], out_path: str):
    lines = [
        f"# Paper run: K=10 probe-and-commit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"**Pipeline:** 5-epoch γ=0 probe → threshold {THRESHOLD} on ep5 avg_conf → 50-epoch main (γ from probe).",
        "**All samples at K=10** (uniform, paper standard).",
        "",
        "## Decision table",
        "",
        "| Sample | ep5 avg_conf | Decision | γ | intra/inter | effK / K_active | KNN purity | Stripe max | Ground truth |",
        "|---|---:|:---:|:---:|---:|:---:|---:|---:|:---:|",
    ]
    for s in samples:
        d = s["decision"]; m = s["metrics"]
        gt = GROUND_TRUTH.get(s["sample"], ("?", ""))
        lines.append(
            f"| {s['sample']} | {d['ep5_avg_conf']:.4f} | "
            f"{'+weight' if d['gamma_main']>0 else 'vanilla'} | "
            f"{d['gamma_main']:.1f} | "
            f"{m['intra_over_inter']:.2f} | "
            f"{m.get('effective_K',0):.2f} / {m.get('active_prototypes', m.get('K_active','?'))} | "
            f"{m.get('KNN_purity_k10',0):.4f} | "
            f"{m.get('stripe_max',0):.2f} | "
            f"{gt[0]} |"
        )

    lines += [
        "",
        "## Validation",
        "",
        "Decision rule (γ=1 if ep5 avg_conf > 0.85) classifies samples as:",
        "",
    ]
    correct = 0; counted = 0
    for s in samples:
        d = s["decision"]
        gt, gt_note = GROUND_TRUTH.get(s["sample"], ("?", ""))
        wants_weight = d["gamma_main"] > 0
        if gt == "helps":
            ok = wants_weight; label = ("✓" if ok else "✗")
            lines.append(f"- **{s['sample']}** — ground truth: weight HELPS ({gt_note}); decision: {'+weight' if wants_weight else 'vanilla'} {label}")
            if ok: correct += 1
            counted += 1
        elif gt == "hurts":
            ok = not wants_weight; label = ("✓" if ok else "✗")
            lines.append(f"- **{s['sample']}** — ground truth: weight HURTS ({gt_note}); decision: {'+weight' if wants_weight else 'vanilla'} {label}")
            if ok: correct += 1
            counted += 1
        else:
            lines.append(f"- **{s['sample']}** — ground truth: {gt_note}; decision: {'+weight' if wants_weight else 'vanilla'} (new validation point)")

    if counted > 0:
        lines += ["", f"**Decision rule accuracy on samples with ground truth: {correct}/{counted}.**"]

    lines += [
        "",
        "## Figures",
        "",
        "- ![Probe decision trajectories](fig_probe_decision.png)",
        "- ![Final intra/inter](fig_metric_bars.png)",
        "",
        "Per-sample evaluation outputs:",
        "",
    ]
    for s in samples:
        sample = s["sample"]
        lines.append(f"### {sample}")
        sd = os.path.join(ROOT, sample, "main", "eval")
        for fname in ("fig_class_map.png", "fig_umap.png",
                      "fig_class_averages.png", "fig_prototype_usage.png",
                      "gradcam/fig_gradcam_summary.png"):
            p = os.path.join(sd, fname)
            if os.path.exists(p):
                rel = os.path.relpath(p, ROOT)
                rel = rel.replace(os.sep, "/")
                lines.append(f"- [{fname}]({rel})")
        lines.append(f"- [decision]({sample}/probe_decision.json)")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}", flush=True)


def main():
    if not os.path.isdir(ROOT):
        print(f"ROOT does not exist: {ROOT}")
        sys.exit(1)
    SAMPLE_LIST = ["EuInAs_B100", "Na007b", "Na006a", "IMC_50nm_SI2", "IMC_150nm_SI5"]
    samples = []
    for s in SAMPLE_LIST:
        d = load_sample(s)
        if d is not None:
            samples.append(d)
    if not samples:
        print("no samples loaded")
        sys.exit(1)
    fig_probe_decision(samples, os.path.join(ROOT, "fig_probe_decision.png"))
    fig_metric_bars(samples,    os.path.join(ROOT, "fig_metric_bars.png"))
    write_report(samples,       os.path.join(ROOT, "REPORT.md"))
    print("\n[paper summary] done")


if __name__ == "__main__":
    main()
