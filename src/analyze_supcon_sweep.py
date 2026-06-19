"""analyze_supcon_sweep.py — read all _supcon_sweep runs, compute composite
score addressing over-clustering, surface winner, generate report.

Composite score balances:
  - KNN purity (cluster coherence)
  - prototype distinctness (anti-over-clustering — counts prototype pairs
    with cosine < 0.5 in the embedding space; more distinct = fewer
    redundant prototypes = better)
  - prototype usage entropy (well-spread vs degenerate)

Output: runs/_supcon_sweep/REPORT.md + summary figs.
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "runs", "_supcon_sweep")


def _proto_distinctness(ckpt_path: str) -> dict:
    """Load model from ckpt, compute prototype pairwise cosine, return stats:
        n_distinct_pairs (cos < 0.5)
        n_redundant_pairs (cos > 0.9 — likely the same prototype)
        max_off_diag, mean_off_diag, min_off_diag
    """
    from dino_sr_contrastive_model import load_contrastive_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = load_contrastive_checkpoint(ckpt_path, device=device)
    p = model.prototypes.prototypes.detach()
    p_norm = p / (p.norm(dim=1, keepdim=True) + 1e-12)
    cos = (p_norm @ p_norm.t()).cpu().numpy()
    K = cos.shape[0]
    eye = np.eye(K, dtype=bool)
    off = cos[~eye]
    n_distinct = int((off < 0.5).sum() // 2)
    n_redundant = int((off > 0.9).sum() // 2)
    return {
        "K": K,
        "n_distinct_pairs": n_distinct,
        "n_redundant_pairs": n_redundant,
        "max_off_diag": float(off.max()),
        "mean_off_diag": float(off.mean()),
        "min_off_diag": float(off.min()),
        "cos_matrix": cos.tolist(),
    }


def _read_run(sample: str, label: str) -> dict | None:
    rd = os.path.join(ROOT, sample, label)
    metrics_p = os.path.join(rd, "eval", "metrics.json")
    summary_p = os.path.join(rd, "run_summary.json")
    ckpt_p = os.path.join(rd, "best.pth")
    if not (os.path.exists(metrics_p) and os.path.exists(ckpt_p)):
        return None
    metrics = json.load(open(metrics_p))
    summary = json.load(open(summary_p)) if os.path.exists(summary_p) else {}
    cfg = summary.get("cfg", {})
    proto = _proto_distinctness(ckpt_p)
    return {
        "sample": sample, "label": label,
        "rundir": rd,
        "intra_over_inter": metrics.get("intra_over_inter", float("nan")),
        "effective_K": metrics.get("effective_K", float("nan")),
        "active_prototypes": metrics.get("active_prototypes", 0),
        "KNN_purity_k10": metrics.get("KNN_purity_k10", float("nan")),
        "prototype_usage_entropy": metrics.get("prototype_usage_entropy", float("nan")),
        "stripe_max": metrics.get("stripe_max", float("nan")),
        "supcon_lambda": cfg.get("supcon_lambda", float("nan")),
        "supcon_temperature": cfg.get("supcon_temperature", float("nan")),
        "contrastive_lambda_eff": cfg.get("contrastive_lambda_eff",
                                           cfg.get("contrastive_lambda",
                                                    float("nan"))),
        "tfin": cfg.get("Tfin", cfg.get("tfin", float("nan"))),
        "proto_K": proto["K"],
        "proto_n_distinct_pairs": proto["n_distinct_pairs"],
        "proto_n_redundant_pairs": proto["n_redundant_pairs"],
        "proto_max_off": proto["max_off_diag"],
        "proto_mean_off": proto["mean_off_diag"],
        "proto_cos_matrix": proto["cos_matrix"],
    }


def composite_score(r: dict) -> float:
    """Higher = better.
    KNN_purity (0..1, weight 1)
    + 0.5 * (n_distinct / total_pairs)
    - 0.5 * (n_redundant / total_pairs)
    + 0.2 * (proto_usage_entropy / log(K))
    """
    K = r["proto_K"]
    total_pairs = (K * (K - 1)) // 2
    knn = float(r["KNN_purity_k10"])
    distinct_frac = r["proto_n_distinct_pairs"] / max(total_pairs, 1)
    redundant_frac = r["proto_n_redundant_pairs"] / max(total_pairs, 1)
    proto_ent_norm = float(r["prototype_usage_entropy"]) / max(np.log(K), 1)
    return knn + 0.5 * distinct_frac - 0.5 * redundant_frac + 0.2 * proto_ent_norm


def main():
    if not os.path.isdir(ROOT):
        print(f"ROOT does not exist: {ROOT}"); sys.exit(1)
    rows = []
    for sample in sorted(os.listdir(ROOT)):
        sd = os.path.join(ROOT, sample)
        if not os.path.isdir(sd): continue
        for label in sorted(os.listdir(sd)):
            d = _read_run(sample, label)
            if d is not None:
                d["composite_score"] = composite_score(d)
                rows.append(d)
                print(f"  {sample}/{label}: score={d['composite_score']:.4f}",
                      flush=True)

    # Generate report.
    out_md = os.path.join(ROOT, "REPORT.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# SupCon sweep results\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("## Composite-score ranking (higher = better)\n\n")
        f.write("Formula: `KNN_purity + 0.5·distinct_frac − 0.5·redundant_frac + 0.2·proto_ent_norm`\n")
        f.write("- distinct_frac = fraction of prototype pairs with cosine < 0.5 (anti-over-clustering)\n")
        f.write("- redundant_frac = fraction with cosine > 0.9\n\n")
        # Sort by score
        ranked = sorted(rows, key=lambda r: r["composite_score"], reverse=True)
        f.write("| sample | label | score | KNN | distinct/total | redundant | intra/inter | effK | K_act | proto_ent | supcon_λ | τ | tfin | contrastive_λ |\n")
        f.write("|---|---|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in ranked:
            K = r["proto_K"]
            total_pairs = (K * (K - 1)) // 2
            f.write(f"| {r['sample']} | {r['label']} | "
                    f"{r['composite_score']:.4f} | "
                    f"{r['KNN_purity_k10']:.3f} | "
                    f"{r['proto_n_distinct_pairs']}/{total_pairs} | "
                    f"{r['proto_n_redundant_pairs']} | "
                    f"{r['intra_over_inter']:.2f} | "
                    f"{r['effective_K']:.2f} | "
                    f"{r['active_prototypes']} | "
                    f"{r['prototype_usage_entropy']:.3f} | "
                    f"{r['supcon_lambda']} | {r['supcon_temperature']} | "
                    f"{r['tfin']} | {r['contrastive_lambda_eff']} |\n")

        # Per-sample winners
        f.write("\n## Per-sample top 3\n\n")
        for sample in sorted(set(r["sample"] for r in rows)):
            f.write(f"### {sample}\n")
            sample_rows = sorted([r for r in rows if r["sample"] == sample],
                                  key=lambda r: r["composite_score"], reverse=True)
            for i, r in enumerate(sample_rows[:3]):
                f.write(f"{i+1}. **{r['label']}** — score {r['composite_score']:.4f}, "
                        f"KNN={r['KNN_purity_k10']:.3f}, "
                        f"distinct={r['proto_n_distinct_pairs']}, "
                        f"redundant={r['proto_n_redundant_pairs']}\n")
            f.write("\n")

        # Cross-sample winner: best label averaged across both samples
        f.write("\n## Cross-sample winner (averaged composite score)\n\n")
        labels = set(r["label"] for r in rows)
        avg_scores = {}
        for lab in labels:
            scores = [r["composite_score"] for r in rows if r["label"] == lab]
            n_samples = len(scores)
            if n_samples == 0: continue
            avg_scores[lab] = (sum(scores) / n_samples, n_samples)
        ranked_labs = sorted(avg_scores.items(),
                              key=lambda kv: kv[1][0], reverse=True)
        f.write("| label | avg score | n_samples |\n|---|---:|---:|\n")
        for lab, (s, n) in ranked_labs:
            f.write(f"| {lab} | {s:.4f} | {n} |\n")

    # Save raw results JSON for downstream use
    with open(os.path.join(ROOT, "results.json"), "w") as jf:
        json.dump(rows, jf, indent=2, default=float)

    # Pick the winner label (highest avg score with at least 2 samples)
    winners = [(lab, sn) for lab, sn in avg_scores.items() if sn[1] >= 2]
    winner_label = max(winners, key=lambda x: x[1][0])[0] if winners else None
    if winner_label is None:
        # fallback: highest single-sample score
        winner_label = max(rows, key=lambda r: r["composite_score"])["label"]
    with open(os.path.join(ROOT, "winner.json"), "w") as wf:
        json.dump({"winner_label": winner_label,
                    "avg_score": avg_scores.get(winner_label, [0])[0],
                    "selected_at": datetime.now().isoformat(timespec="seconds")},
                   wf, indent=2)
    print(f"\nWINNER: {winner_label}", flush=True)
    print(f"REPORT: {out_md}", flush=True)


if __name__ == "__main__":
    main()
