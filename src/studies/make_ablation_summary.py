"""make_ablation_summary.py — Cross-sample analysis after run_paper_ablation_K6
finishes. Reads per-(sample, condition) outputs and generates:

    fig_ablation_metrics.png   - intra/inter, effK, KNN purity per sample x condition
    fig_ep5_discriminator.png  - ep5 metric (conf, ent, effK) vs Δ(intra/inter)
                                  to test which signals at ep5 predict whether
                                  weight helps
    fig_class_balance.png      - class fraction distributions
    REPORT.md                  - the table the user will read first
"""
from __future__ import annotations
import os, sys, json, csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "runs", "_paper_ablation_K6")
SAMPLES = ["EuInAs_B100", "IMC_150nm_SI5", "Na007b", "IMC_50nm_SI2",
           "Na007a", "Na006a"]
CONDITIONS = ["vanilla", "weight"]


def _read(sample: str, label: str):
    sd = os.path.join(ROOT, sample, label)
    metrics_p = os.path.join(sd, "eval", "metrics.json")
    log_p = os.path.join(sd, "training_log.csv")
    inf_p = os.path.join(sd, "eval", "inference.npz")
    if not (os.path.exists(metrics_p) and os.path.exists(log_p)):
        return None
    metrics = json.load(open(metrics_p))
    rows = list(csv.DictReader(open(log_p)))
    ep5 = next((r for r in rows if int(r["epoch"]) == 5), None)
    ep_last = rows[-1]
    inf = np.load(inf_p) if os.path.exists(inf_p) else None
    fracs = None
    K = metrics.get("K_original", metrics.get("K_active"))
    if inf is not None:
        cnt = np.bincount(inf["assigns"], minlength=int(K) if K else 1)
        fracs = (cnt / cnt.sum()).tolist()
    return {
        "sample": sample, "label": label,
        "metrics": metrics,
        "ep5": ep5,
        "ep_last": ep_last,
        "K": K,
        "class_fractions": fracs,
    }


def _fig_metrics_bars(data: dict, out_path: str):
    """Bar chart of intra/inter for vanilla vs weight per sample."""
    samples_present = [s for s in SAMPLES
                        if data.get((s, "vanilla")) and data.get((s, "weight"))]
    n = len(samples_present)
    if n == 0: return
    x = np.arange(n)
    width = 0.4
    van = [data[(s, "vanilla")]["metrics"]["intra_over_inter"]
           for s in samples_present]
    wei = [data[(s, "weight")]["metrics"]["intra_over_inter"]
           for s in samples_present]
    fig, ax = plt.subplots(figsize=(max(8, n*1.4), 5))
    bars1 = ax.bar(x - width/2, van, width, label="vanilla (γ=0)",
                    color="#ff7f0e", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + width/2, wei, width, label="weight (γ=1)",
                    color="#1f77b4", edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars1, van):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars2, wei):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(samples_present, rotation=20, ha="right")
    ax.set_ylabel("intra / inter (cosine ratio)")
    ax.set_title("K=6 ablation: intra/inter for vanilla vs weight per sample")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fig_ep5_discriminator(data: dict, out_path: str):
    """For each candidate ep5 metric (conf, ent, effK), plot vs delta(intra/inter)
    to see which predicts whether weight helps."""
    candidates = ["avg_conf", "teacher_ent_mean", "effK", "active_classes"]
    samples_present = [s for s in SAMPLES
                        if data.get((s, "vanilla")) and data.get((s, "weight"))]
    if len(samples_present) < 2: return
    fig, axes = plt.subplots(1, len(candidates),
                              figsize=(4*len(candidates), 4.5),
                              squeeze=False)
    for ax, key in zip(axes[0], candidates):
        xs, ys, labels = [], [], []
        for s in samples_present:
            van = data[(s, "vanilla")]
            wei = data[(s, "weight")]
            if van["ep5"] is None: continue
            x = float(van["ep5"][key])
            dy = (wei["metrics"]["intra_over_inter"]
                   - van["metrics"]["intra_over_inter"])
            xs.append(x); ys.append(dy); labels.append(s)
        if not xs: continue
        colors = ["#1f77b4" if y > 0 else "#ff7f0e" for y in ys]
        ax.scatter(xs, ys, c=colors, s=80, edgecolors="black", linewidth=0.7)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y),
                         xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xlabel(f"ep5 {key} (vanilla probe)")
        ax.set_ylabel("Δ intra/inter (weight - vanilla)")
        ax.set_title(f"discriminator: {key}", fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle("Which ep5 metric predicts whether weight helps?",
                  fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fig_class_balance(data: dict, out_path: str):
    samples_present = [s for s in SAMPLES
                        if data.get((s, "vanilla")) and data.get((s, "weight"))]
    n = len(samples_present)
    if n == 0: return
    fig, axes = plt.subplots(n, 2, figsize=(8, n*1.3), squeeze=False)
    for r, s in enumerate(samples_present):
        for c, label in enumerate(CONDITIONS):
            d = data[(s, label)]
            fracs = sorted(d["class_fractions"] or [], reverse=True)
            x = np.arange(len(fracs))
            axes[r, c].bar(x, fracs,
                           color="#ff7f0e" if label == "vanilla" else "#1f77b4",
                           edgecolor="black", linewidth=0.5)
            axes[r, c].set_xticks(x)
            axes[r, c].set_ylim(0, max(fracs)*1.2 if fracs else 1)
            axes[r, c].set_title(f"{s} {label}", fontsize=10)
            axes[r, c].set_ylabel("fraction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_report(data: dict, out_path: str):
    samples_present = [s for s in SAMPLES
                        if data.get((s, "vanilla")) and data.get((s, "weight"))]
    lines = [
        "# K=6 ablation: vanilla vs weight, all samples",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Main table: ground truth + ep5 metrics",
        "",
        "| sample | γ=0 intra/inter | γ=1 intra/inter | Δ | weight | ep5 conf (γ=0) | ep5 ent (γ=0) | ep5 effK (γ=0) |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for s in samples_present:
        v = data[(s, "vanilla")]; w = data[(s, "weight")]
        van_ii = v["metrics"]["intra_over_inter"]
        wei_ii = w["metrics"]["intra_over_inter"]
        delta = wei_ii - van_ii
        verdict = "**helps**" if delta > 1.0 else (
            "**hurts**" if delta < -1.0 else "neutral")
        ep5 = v["ep5"]
        if ep5 is not None:
            conf = float(ep5["avg_conf"])
            ent = float(ep5["teacher_ent_mean"])
            effK = float(ep5["effK"])
            ep_str = f"{conf:.4f} | {ent:.4f} | {effK:.2f}"
        else:
            ep_str = "— | — | —"
        lines.append(
            f"| {s} | {van_ii:.2f} | {wei_ii:.2f} | "
            f"{delta:+.2f} | {verdict} | {ep_str} |"
        )

    lines += ["",
              "## Per-sample artifacts",
              ""]
    for s in samples_present:
        lines.append(f"### {s}")
        for c in CONDITIONS:
            sd_rel = f"{s}/{c}/eval"
            lines.append(f"- **{c}**: "
                         f"[class map](../{sd_rel}/fig_class_map.png), "
                         f"[BF overlay](../{sd_rel}/fig_overlay_BF.png), "
                         f"[HAADF overlay](../{sd_rel}/fig_overlay_HAADF.png), "
                         f"[UMAP](../{sd_rel}/fig_umap.png), "
                         f"[gradcam](../{sd_rel}/gradcam/), "
                         f"[class quality](../{sd_rel}/class_quality/)")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    if not os.path.isdir(ROOT):
        print(f"ROOT does not exist: {ROOT}"); sys.exit(1)
    data = {}
    for s in SAMPLES:
        for c in CONDITIONS:
            d = _read(s, c)
            if d is not None:
                data[(s, c)] = d
            else:
                print(f"[skip] {s}/{c} (incomplete)")
    if not data:
        print("no data"); sys.exit(1)
    _fig_metrics_bars(data, os.path.join(ROOT, "fig_ablation_metrics.png"))
    _fig_ep5_discriminator(data, os.path.join(ROOT, "fig_ep5_discriminator.png"))
    _fig_class_balance(data, os.path.join(ROOT, "fig_class_balance.png"))
    _write_report(data, os.path.join(ROOT, "REPORT.md"))
    print(f"summary done: see {ROOT}/REPORT.md and fig_*.png")


if __name__ == "__main__":
    main()
