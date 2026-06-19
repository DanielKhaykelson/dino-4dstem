"""
run_overnight.py — unattended sweep driver.

Phase 1: Na007b ablation (3 new variants + reuse existing)
  reused:  config_c_K6_asym_tempsched        (polar, pairwise-only, T0=0.04→Tfin=0.07)
  new:     sweep_polar_centroid              (polar, pairwise + centroid)
  new:     sweep_cart_nocent                 (Cartesian, pairwise-only)
  new:     sweep_cart_centroid               (Cartesian, pairwise + centroid)

Phase 2: Pick winner via composite score, apply to Na006a + EuInAs_B100 at K=10.
Phase 3: If winner is Cartesian, also run polar-pairwise fallbacks for Na006a / EuInAs.
Phase 4: Na007a transfer — load Na007b winner checkpoint, run inference + viz.
Phase 5: Write OVERNIGHT_REPORT.md at runs/ root.

Robust to per-run failures: each run is wrapped in try/except so one bad
training doesn't kill the sweep. Partial results are still written.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(BASE_DIR, "runs")
sys.path.insert(0, BASE_DIR)

from data import SAMPLES
from run_contrastive import run_config, evaluate_and_report


# -----------------------------------------------------------------------
# Shared defaults (mirror what the temp-schedule run used)
# -----------------------------------------------------------------------
COMMON = dict(
    lr=3e-4, weight_decay=1e-6,
    t0=0.04, tfin=0.07,                  # DINO v1-canonical schedule
    warmup_epochs=20, ramp_epochs=10,
    entropy_gate=False,
    projection_dim=128, projection_hidden=256,
    theta_shift_range=None,              # asymmetric defaults
    theta_shift_range_student=None,      # -> polar_size = 192 = ±180°
    theta_shift_range_teacher=None,      # -> ±15° for polar, or same mapping for Cartesian
    center_mask_radius=None, center_crop_size=140,
    vmax=None, polar_size=192, polar_mask_cols=30,
    centroid_margin=0.3,
)


def _safe_call(fn, *args, **kwargs):
    """Run `fn`; log + swallow exceptions so the sweep continues."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[run_overnight] {fn.__name__} FAILED: {exc!r}", flush=True)
        traceback.print_exc()
        return None


# -----------------------------------------------------------------------
# One run + eval unit
# -----------------------------------------------------------------------
def _run_and_eval(label: str, sample: str, K: int, epochs: int,
                   batch_size: int, pipeline: str,
                   centroid_lambda: float,
                   device, reference_assigns=None,
                   seed: int = 42) -> dict | None:
    outdir = os.path.join(RUNS_ROOT, sample, label)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.perf_counter()
    print(f"\n{'=' * 72}\n[{datetime.now():%H:%M:%S}] STARTING {sample}/{label}\n{'=' * 72}",
          flush=True)
    try:
        run_config(
            "c",  # ablation preset (c) = λ=0.2 pairwise + θ-roll
            sample=sample, epochs=epochs, seed=seed, batch_size=batch_size,
            num_prototypes=K,
            pipeline=pipeline, centroid_lambda=centroid_lambda,
            outdir=outdir, device=device,
            **COMMON,
        )
        metrics = evaluate_and_report(
            "c", sample=sample, outdir=outdir,
            device=device, reference_assigns=reference_assigns)
        metrics["_label"] = label
        metrics["_sample"] = sample
        metrics["_outdir"] = outdir
        metrics["_elapsed_s"] = time.perf_counter() - t0
        print(f"[{datetime.now():%H:%M:%S}] DONE {sample}/{label}  "
              f"in {metrics['_elapsed_s']:.0f}s", flush=True)
        return metrics
    except Exception as exc:
        print(f"[{datetime.now():%H:%M:%S}] FAILED {sample}/{label}: {exc!r}",
              flush=True)
        traceback.print_exc()
        return None


# -----------------------------------------------------------------------
# Composite winner-selection score
# -----------------------------------------------------------------------
def composite_score(m: dict) -> float:
    if m is None:
        return -1.0
    knn = m.get("KNN_purity_k10", 0.0) or 0.0
    ratio = m.get("intra_over_inter", 0.0) or 0.0
    stripe = m.get("stripe_max", 1.0) or 1.0
    proto_counts = m.get("proto_counts", []) or []
    K = len(proto_counts)
    dead = len(m.get("dead_prototypes", []) or [])
    # Normalize ratio; 20 is already "excellent"
    ratio_norm = min(ratio / 20.0, 1.0)
    stripe_penalty = min(max(stripe - 2.0, 0.0) / 8.0, 1.0)  # penalize ratio > 2
    dead_frac = dead / max(K, 1)
    return (0.4 * knn
            + 0.3 * ratio_norm
            + 0.2 * (1.0 - stripe_penalty)
            + 0.1 * (1.0 - dead_frac))


# -----------------------------------------------------------------------
# Transfer inference on Na007a (no training)
# -----------------------------------------------------------------------
def transfer_inference(winner_outdir: str, target_sample: str,
                        device) -> dict | None:
    """Load the winner's best.pth and run eval on `target_sample` WITHOUT
    training. Class identities inherit from the training sample.
    """
    import viz_gradcam
    from dino_sr_contrastive_model import load_contrastive_checkpoint
    import contrastive_eval as CE

    cfg_target = SAMPLES[target_sample]
    outdir = os.path.join(RUNS_ROOT, target_sample, "transfer_from_winner")
    os.makedirs(outdir, exist_ok=True)
    print(f"\n[{datetime.now():%H:%M:%S}] TRANSFER inference on "
          f"{target_sample} using {winner_outdir}/best.pth",
          flush=True)
    try:
        from data import LoadPRZ
        dataset = LoadPRZ(cfg_target["path"], resize=192, vmax=cfg_target["vmax"])
        scan_shape = cfg_target["scan_shape"]

        ckpt_path = os.path.join(winner_outdir, "best.pth")
        model, eval_temp, _, train_cfg = load_contrastive_checkpoint(
            ckpt_path, device=device)
        # Use the training-time polar_mask_cols if recorded, else 30 default.
        polar_mask_cols = int(train_cfg.get("polar_mask_cols", 30))
        inf = CE.infer_scan(model, dataset, device, eval_temp=eval_temp,
                             polar_mask_cols=polar_mask_cols,
                             center_crop_size=140, polar_size=192)
        eval_dir = os.path.join(outdir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        np.savez_compressed(os.path.join(eval_dir, "inference.npz"),
                             soft_probs=inf["soft_probs"],
                             teacher_probs=inf["teacher_probs"],
                             embeds=inf["embeds"],
                             assigns=inf["assigns"])
        assigns = inf["assigns"]
        soft_probs = inf["soft_probs"]
        embeds = inf["embeds"]
        t_probs = inf["teacher_probs"]
        K = soft_probs.shape[1]
        ent, eff_k, counts = CE.prototype_usage_entropy(soft_probs)
        intra, inter = CE.intra_inter_cosine(embeds, assigns)
        ratio = intra / inter if inter and inter != 0 and not np.isnan(inter) else float("nan")
        knn = CE.knn_purity(embeds, assigns, k=10, max_query=1500)
        cos_mat, centroids = CE.centroid_cosine_matrix(embeds, assigns, K)
        dead = [int(c) for c in range(K) if counts[c] < max(5, 0.005 * len(assigns))]
        stripe_scores = CE.stripe_metric(assigns, scan_shape, K)

        # Figures.
        CE.plot_centroid_cosine(cos_mat, os.path.join(eval_dir, "fig_centroid_cosine.png"))
        CE.plot_prototype_usage(counts, os.path.join(eval_dir, "fig_prototype_usage.png"))
        umap_method = CE.plot_umap(embeds, assigns, os.path.join(eval_dir, "fig_umap.png"))
        CE.plot_representative_patterns(dataset, assigns, soft_probs,
                                         os.path.join(eval_dir, "fig_representative.png"),
                                         per_proto=6)
        CE.plot_boundary_samples(dataset, t_probs,
                                  os.path.join(eval_dir, "fig_boundary.png"), n=16)
        CE.plot_class_map(assigns, scan_shape,
                           os.path.join(eval_dir, "fig_class_map.png"))
        CE.plot_embedding_norms(embeds,
                                 os.path.join(eval_dir, "fig_embedding_norms.png"))

        metrics = {
            "config": "transfer",
            "source_ckpt": ckpt_path,
            "target_sample": target_sample,
            "NMI_vs_self": 0.0,
            "KNN_purity_k10": float(knn),
            "intra_class_cosine": float(intra),
            "inter_class_cosine": float(inter),
            "intra_over_inter": float(ratio) if ratio == ratio else None,
            "prototype_usage_entropy": float(ent),
            "effective_K": float(eff_k),
            "active_prototypes": int((counts > 0).sum()),
            "dead_prototypes": dead,
            "proto_counts": counts.tolist(),
            "stripe_scores": stripe_scores,
            "stripe_max": float(max(stripe_scores)) if stripe_scores else 1.0,
            "embed_norm_mean": float(np.linalg.norm(embeds, axis=1).mean()),
            "umap_method": umap_method,
        }
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=float)

        # GradCAM + IG on the transferred model.
        _safe_call(viz_gradcam.run, sample=target_sample,
                    config="transfer_from_winner",
                    n_samples_per_proto=10, device=device)
        print(f"[{datetime.now():%H:%M:%S}] TRANSFER done: {outdir}", flush=True)
        return metrics
    except Exception as exc:
        print(f"[{datetime.now():%H:%M:%S}] TRANSFER failed: {exc!r}", flush=True)
        traceback.print_exc()
        return None


# -----------------------------------------------------------------------
# Final overnight report writer
# -----------------------------------------------------------------------
def write_overnight_report(phase1_metrics, winner_label, winner_score,
                            phase2_metrics, phase3_metrics, transfer_metrics):
    path = os.path.join(RUNS_ROOT, "OVERNIGHT_REPORT.md")
    L = []
    L.append(f"# Overnight sweep — {datetime.now():%Y-%m-%d %H:%M}")
    L.append("")
    L.append("## Phase 1 — Na007b ablation")
    L.append("")
    L.append("| label | KNN_k10 | intra/inter | effK | stripe_max | dead | composite |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, m in phase1_metrics.items():
        if m is None:
            L.append(f"| {label} | FAILED | | | | | |")
            continue
        sc = composite_score(m)
        L.append(f"| {label} | {m.get('KNN_purity_k10', 0):.3f} | "
                  f"{m.get('intra_over_inter', 0) or 0:.2f} | "
                  f"{m.get('effective_K', 0):.2f} | "
                  f"{m.get('stripe_max', 1):.1f} | "
                  f"{len(m.get('dead_prototypes', []) or [])} | "
                  f"{sc:.3f} |")
    L.append("")
    L.append(f"**Winner:** `{winner_label}`  (composite {winner_score:.3f})")
    L.append("")

    L.append("## Phase 2 — Winner applied to other samples")
    L.append("")
    L.append("| sample | label | KNN_k10 | intra/inter | effK | stripe_max | dead |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for key, m in phase2_metrics.items():
        if m is None:
            L.append(f"| {key} | FAILED | | | | | |")
            continue
        L.append(f"| {m.get('_sample')} | {m.get('_label')} | "
                  f"{m.get('KNN_purity_k10', 0):.3f} | "
                  f"{m.get('intra_over_inter', 0) or 0:.2f} | "
                  f"{m.get('effective_K', 0):.2f} | "
                  f"{m.get('stripe_max', 1):.1f} | "
                  f"{len(m.get('dead_prototypes', []) or [])} |")
    L.append("")

    if phase3_metrics:
        L.append("## Phase 3 — Polar fallback (because winner was Cartesian)")
        L.append("")
        L.append("| sample | label | KNN_k10 | intra/inter | effK | stripe_max |")
        L.append("|---|---|---:|---:|---:|---:|")
        for key, m in phase3_metrics.items():
            if m is None:
                L.append(f"| {key} | FAILED | | | | |")
                continue
            L.append(f"| {m.get('_sample')} | {m.get('_label')} | "
                      f"{m.get('KNN_purity_k10', 0):.3f} | "
                      f"{m.get('intra_over_inter', 0) or 0:.2f} | "
                      f"{m.get('effective_K', 0):.2f} | "
                      f"{m.get('stripe_max', 1):.1f} |")
        L.append("")

    L.append("## Phase 4 — Transfer inference on Na007a")
    L.append("")
    if transfer_metrics is None:
        L.append("Transfer FAILED or was skipped.")
    else:
        L.append(f"- Source checkpoint: `{transfer_metrics['source_ckpt']}`")
        L.append(f"- KNN purity (self, using transferred prototypes): "
                  f"{transfer_metrics['KNN_purity_k10']:.3f}")
        L.append(f"- intra/inter cosine ratio: "
                  f"{transfer_metrics.get('intra_over_inter') or float('nan'):.2f}")
        L.append(f"- Effective K: {transfer_metrics['effective_K']:.2f}")
        L.append(f"- Dead prototypes: {transfer_metrics['dead_prototypes']}")
        L.append(f"- Prototype counts: {transfer_metrics['proto_counts']}")
        L.append("")
        L.append("**Interpretation note:** cluster identities are inherited "
                  "from the training sample (Na007b). Assignments are "
                  "'closest Na007b phase' for each Na007a pattern, not a "
                  "native clustering of Na007a.")
    L.append("")

    L.append("## Interpretive notes")
    L.append("")
    L.append("- Composite score = 0.4·KNN + 0.3·norm(intra/inter) "
              "+ 0.2·(1 − stripe penalty) + 0.1·(1 − dead/K).")
    L.append("- Stripe penalty > 0 when any prototype has λ_max/λ_min > 2 in "
              "its 2-D scan-position covariance (line-like cluster → likely "
              "scan-drift artifact).")
    L.append("- All runs used τ_t: 0.04 → 0.07 warmup over 10 epochs, τ_s=0.1, "
              "center momentum 0.9, teacher EMA cosine 0.994 → 0.999, "
              "student ±180° / teacher ±15° rotation asymmetry.")
    L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n[{datetime.now():%H:%M:%S}] Wrote {path}", flush=True)


# -----------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    # --------------------------------------------------------------------
    # Phase 1: Na007b ablation — 3 new + 1 reused
    # --------------------------------------------------------------------
    na007b_ref_assigns = None
    phase1_metrics: dict = {}
    # Include the existing temp-schedule run as the polar-pairwise baseline.
    existing_ref = os.path.join(RUNS_ROOT, "Na007b",
                                  "config_c_K6_asym_tempsched", "eval",
                                  "metrics.json")
    if os.path.exists(existing_ref):
        with open(existing_ref) as f:
            existing_m = json.load(f)
        existing_m["_label"] = "polar_pairwise (reuse)"
        existing_m["_sample"] = "Na007b"
        existing_m["_outdir"] = os.path.join(RUNS_ROOT, "Na007b",
                                               "config_c_K6_asym_tempsched")
        # stripe metric may not be in old metrics.json; compute on the fly
        if "stripe_max" not in existing_m:
            import contrastive_eval as CE
            scan_shape = SAMPLES["Na007b"]["scan_shape"]
            inf = np.load(os.path.join(RUNS_ROOT, "Na007b",
                                         "config_c_K6_asym_tempsched",
                                         "eval", "inference.npz"))
            K = inf["soft_probs"].shape[1]
            stripes = CE.stripe_metric(inf["assigns"], scan_shape, K)
            existing_m["stripe_scores"] = stripes
            existing_m["stripe_max"] = float(max(stripes))
        phase1_metrics["polar_pairwise"] = existing_m
        na007b_ref_assigns = np.load(os.path.join(RUNS_ROOT, "Na007b",
                                                    "config_c_K6_asym_tempsched",
                                                    "eval", "inference.npz"))["assigns"]

    # Three new variants.
    variants = [
        dict(label="sweep_polar_centroid",  pipeline="polar",     centroid_lambda=0.05),
        dict(label="sweep_cart_nocent",     pipeline="cartesian", centroid_lambda=0.0),
        dict(label="sweep_cart_centroid",   pipeline="cartesian", centroid_lambda=0.05),
    ]
    for v in variants:
        m = _run_and_eval(
            label=v["label"], sample="Na007b",
            K=6, epochs=50, batch_size=128,
            pipeline=v["pipeline"], centroid_lambda=v["centroid_lambda"],
            device=device, reference_assigns=na007b_ref_assigns)
        phase1_metrics[v["label"]] = m

    # --------------------------------------------------------------------
    # Winner selection
    # --------------------------------------------------------------------
    scores = {k: composite_score(v) for k, v in phase1_metrics.items()}
    winner_label = max(scores, key=scores.get)
    winner = phase1_metrics[winner_label]
    winner_pipeline = "polar"
    winner_centroid_lambda = 0.0
    if winner_label == "polar_pairwise":
        winner_pipeline = "polar"; winner_centroid_lambda = 0.0
    elif winner_label == "sweep_polar_centroid":
        winner_pipeline = "polar"; winner_centroid_lambda = 0.05
    elif winner_label == "sweep_cart_nocent":
        winner_pipeline = "cartesian"; winner_centroid_lambda = 0.0
    elif winner_label == "sweep_cart_centroid":
        winner_pipeline = "cartesian"; winner_centroid_lambda = 0.05
    winner_outdir = winner.get("_outdir") or os.path.join(
        RUNS_ROOT, "Na007b", winner_label)
    print(f"\n[winner] {winner_label}  score={scores[winner_label]:.3f}  "
          f"pipeline={winner_pipeline} centroid_lambda={winner_centroid_lambda}",
          flush=True)

    # --------------------------------------------------------------------
    # Phase 2: Apply winner to Na006a + EuInAs_B100  (K=10 fuzzy-c ceiling)
    # --------------------------------------------------------------------
    phase2_metrics: dict = {}
    for sample in ("Na006a", "EuInAs_B100"):
        m = _run_and_eval(
            label=f"winner_{winner_pipeline}"
                   + ("_centroid" if winner_centroid_lambda > 0 else ""),
            sample=sample, K=10, epochs=50, batch_size=128,
            pipeline=winner_pipeline,
            centroid_lambda=winner_centroid_lambda,
            device=device)
        phase2_metrics[sample] = m

    # --------------------------------------------------------------------
    # Phase 3: Polar fallback ONLY if winner is Cartesian
    # --------------------------------------------------------------------
    phase3_metrics: dict = {}
    if winner_pipeline == "cartesian":
        for sample in ("Na006a", "EuInAs_B100"):
            m = _run_and_eval(
                label="polar_fallback",
                sample=sample, K=10, epochs=50, batch_size=128,
                pipeline="polar",
                centroid_lambda=0.0,
                device=device)
            phase3_metrics[sample] = m

    # --------------------------------------------------------------------
    # Phase 4: Transfer inference on Na007a using winner checkpoint
    # --------------------------------------------------------------------
    transfer_metrics = transfer_inference(winner_outdir, "Na007a", device)

    # --------------------------------------------------------------------
    # Final report
    # --------------------------------------------------------------------
    write_overnight_report(phase1_metrics, winner_label,
                             scores[winner_label], phase2_metrics,
                             phase3_metrics, transfer_metrics)
    # Also dump a raw JSON for debugging
    dump = {
        "winner_label": winner_label,
        "winner_score": scores[winner_label],
        "winner_pipeline": winner_pipeline,
        "winner_centroid_lambda": winner_centroid_lambda,
        "phase1": {k: (v if not isinstance(v, dict) else
                        {kk: vv for kk, vv in v.items()})
                    for k, v in phase1_metrics.items()},
        "phase2": phase2_metrics,
        "phase3": phase3_metrics,
        "transfer": transfer_metrics,
    }
    with open(os.path.join(RUNS_ROOT, "OVERNIGHT_DUMP.json"), "w") as f:
        json.dump(dump, f, indent=2, default=float)


if __name__ == "__main__":
    main()
