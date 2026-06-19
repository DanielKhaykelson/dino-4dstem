"""
Report writers for the per-config and cross-config analytical markdown.

Note: we refuse to just restate numbers (spec 14: "Do not produce a report
that just restates the numbers"). The interpretive prose is assembled from
the metric values so the report reads as a physically grounded analysis of
what the model actually learned, not a printout of a metrics dict.
"""

from __future__ import annotations

import json
import os
from typing import Any


# =========================================================================
# Per-config report
# =========================================================================

def write_config_report(config_key: str, preset: dict, metrics: dict,
                         scan_shape: tuple, eval_dir: str, outdir: str,
                         train_cfg: dict, train_metrics: dict) -> None:
    path = os.path.join(outdir, "report.md")
    lam = preset["contrastive_lambda"]
    troll = preset["theta_roll_aug"]
    label = preset["label"]

    # Interpretation helpers.
    def _collapse_verdict() -> str:
        eff = metrics["effective_K"]
        n_active = metrics["active_prototypes"]
        dead = metrics["dead_prototypes"]
        K = len(metrics["proto_counts"])
        if eff < 3:
            return (f"**Prototype collapse: effective K = {eff:.2f} out of {K}. "
                    f"The model has collapsed to {int(round(eff))} distinct "
                    f"prototype(s). Treat all downstream metrics with suspicion.**")
        if len(dead) > K // 2:
            return (f"Heavy prototype death: {len(dead)} of {K} prototypes are "
                    f"effectively unused (<0.5% of samples). Effective K = "
                    f"{eff:.2f}. Usable but the latent capacity is under-exercised.")
        if eff < K * 0.5:
            return (f"Moderate usage skew: effective K = {eff:.2f} of {K}. A few "
                    f"dominant prototypes absorb the bulk of mass; "
                    f"{n_active - len(dead)} active, {len(dead)} dead.")
        return (f"Healthy prototype usage: effective K = {eff:.2f} of {K}, "
                f"{n_active} active, {len(dead)} dead.")

    def _rotation_verdict() -> str:
        med = metrics["rotation_invariance_cos_median"]
        p10 = metrics["rotation_invariance_cos_p10"]
        if not troll:
            if med > 0.8:
                return (f"Rotation invariance cos-median = {med:.3f} despite "
                        f"theta-roll aug being OFF. The rotation-free Cartesian "
                        f"aug (flips + blur + colorjitter + polar transform) "
                        f"already yields some invariance via the circular-theta "
                        f"conv. This is the baseline against which (b, c) are judged.")
            return (f"Rotation invariance cos-median = {med:.3f} (p10 = {p10:.3f}) "
                    f"with theta-roll OFF. As expected for a baseline with no "
                    f"rotation-related pressure — polar-aligned features depend "
                    f"on absolute theta.")
        if med > 0.9:
            return (f"Strong learned rotation invariance: cos-median = {med:.3f}, "
                    f"p10 = {p10:.3f}. Theta-roll augmentation is doing what it "
                    f"was intended to do.")
        if med > 0.7:
            return (f"Partial rotation invariance: cos-median = {med:.3f}, "
                    f"p10 = {p10:.3f}. Theta-roll helps but the embedding is "
                    f"not yet fully rotation-stable. Check for under-training or "
                    f"a too-narrow shift_range.")
        return (f"Weak rotation invariance: cos-median = {med:.3f}, p10 = {p10:.3f}. "
                f"Theta-roll augmentation enabled but the embedding does not "
                f"reflect it. Suspect circular-padding breakage or gradient "
                f"signal too weak to reshape the projection.")

    def _geometry_verdict() -> str:
        norm_m = metrics["embed_norm_mean"]
        norm_s = metrics["embed_norm_std"]
        intra = metrics["intra_class_cosine"]
        inter = metrics["inter_class_cosine"]
        ratio = metrics["intra_over_inter"]
        bits = []
        if abs(norm_m - 1.0) > 0.02:
            bits.append(f"**Embedding norm drift**: mean ||z|| = {norm_m:.3f} "
                        f"(expected 1.0). The L2-normalize after the MLP isn't "
                        f"holding — inspect the projection head.")
        else:
            bits.append(f"Embedding norms on the unit sphere (mean {norm_m:.3f}, "
                        f"std {norm_s:.3f}). L2-normalize is working.")
        if ratio is None or (isinstance(ratio, float) and ratio != ratio):
            bits.append("Intra/inter cosine ratio undefined (likely only one active class).")
        elif ratio > 2.0:
            bits.append(f"Strong cluster structure in the contrastive space: "
                        f"intra/inter cosine = {intra:.3f}/{inter:.3f} "
                        f"(ratio {ratio:.2f}).")
        elif ratio > 1.2:
            bits.append(f"Modest cluster structure: intra/inter cosine = "
                        f"{intra:.3f}/{inter:.3f} (ratio {ratio:.2f}). "
                        f"Prototypes are identifiable but not tightly separated.")
        else:
            bits.append(f"**Weak clustering in contrastive space**: intra/inter "
                        f"cosine = {intra:.3f}/{inter:.3f} (ratio {ratio:.2f}). "
                        f"The projection embeddings barely discriminate the "
                        f"prototype identity that the DINO head assigned.")
        return " ".join(bits)

    def _decoupling_verdict() -> str:
        # Use training log to detect L_contrastive flatline while L_DINO moves.
        try:
            import pandas as pd
            df = pd.read_csv(os.path.join(outdir, "training_log.csv"))
            if df.empty:
                return ""
            # Post-warmup window
            pw = df[df.lambda_eff > 0.0]
            if pw.empty:
                return (f"Contrastive loss was computed but never added (lambda_eff = 0 "
                        f"through all {len(df)} epochs). Expected for config (a) and (b).")
            d_dino = pw.avg_loss_dino.iloc[-1] - pw.avg_loss_dino.iloc[0]
            d_cont = pw.avg_loss_contrastive.iloc[-1] - pw.avg_loss_contrastive.iloc[0]
            if d_dino < -0.02 and abs(d_cont) < 1e-3:
                return (f"**Contrastive decoupling suspected**: across the "
                        f"{len(pw)} post-warmup epochs, L_DINO moved by "
                        f"{d_dino:+.3f} while L_contrastive was flat "
                        f"(delta = {d_cont:+.2e}). The two objectives trained "
                        f"independent paths — check if lambda is too small or "
                        f"the projection head is under-optimized.")
            return (f"Loss coupling: L_DINO delta = {d_dino:+.3f}, "
                    f"L_contrastive delta = {d_cont:+.3f} across "
                    f"{len(pw)} post-warmup epochs. Both moving — signals coupled.")
        except Exception:
            return ""

    collapse_line = _collapse_verdict()
    rotation_line = _rotation_verdict()
    geometry_line = _geometry_verdict()
    decouple_line = _decoupling_verdict()

    # Verdict summary (one-line).
    verdict_bits = []
    if metrics["effective_K"] < 3:
        verdict_bits.append("collapsed")
    else:
        verdict_bits.append(f"effK={metrics['effective_K']:.1f}")
    if metrics["intra_over_inter"] and metrics["intra_over_inter"] > 1.5:
        verdict_bits.append("clusters separable")
    if troll and metrics["rotation_invariance_cos_median"] > 0.9:
        verdict_bits.append("rotation-invariant")
    verdict = "; ".join(verdict_bits)

    proto_counts = metrics["proto_counts"]
    dead = metrics["dead_prototypes"]
    i_star, j_star, cos_star = metrics["most_similar_pair"]

    lines = []
    lines.append(f"# config_{config_key} — Na007b — DINO-SR + Contrastive ({label})")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"Config flags: **contrastive_lambda = {lam}**, **theta_roll_aug = {troll}**. "
        f"{epochs_line(train_metrics)} Verdict: {verdict}."
    )
    lines.append("")
    lines.append("## Quantitative metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| NMI (vs config a) | {metrics['NMI_vs_configA']:.3f} |"
                 if metrics["NMI_vs_configA"] == metrics["NMI_vs_configA"]  # not NaN
                 else "| NMI (vs config a) | — (self) |")
    lines.append(f"| KNN purity (k=10) | {metrics['KNN_purity_k10']:.3f} |")
    lines.append(f"| intra-class cosine | {metrics['intra_class_cosine']:.3f} |")
    lines.append(f"| inter-class cosine | {metrics['inter_class_cosine']:.3f} |")
    lines.append(f"| intra / inter | {metrics['intra_over_inter']:.3f} |"
                 if metrics['intra_over_inter'] == metrics['intra_over_inter']
                 else "| intra / inter | NaN |")
    lines.append(f"| radial profile consistency | {metrics['radial_profile_consistency']:.3f} |")
    lines.append(f"| prototype usage entropy (nats) | {metrics['prototype_usage_entropy']:.3f} |")
    lines.append(f"| effective K | {metrics['effective_K']:.2f} |")
    lines.append(f"| active prototypes | {metrics['active_prototypes']} |")
    lines.append(f"| dead prototypes | {dead} |")
    lines.append(f"| rotation-invariance cos (median, p10, p90) | "
                 f"{metrics['rotation_invariance_cos_median']:.3f}, "
                 f"{metrics['rotation_invariance_cos_p10']:.3f}, "
                 f"{metrics['rotation_invariance_cos_p90']:.3f} |")
    lines.append(f"| embedding ||z|| (mean, std) | "
                 f"{metrics['embed_norm_mean']:.3f}, {metrics['embed_norm_std']:.3f} |")
    lines.append("")
    lines.append("## Per-prototype breakdown")
    lines.append("")
    lines.append("Sample counts per prototype:")
    lines.append("")
    lines.append("| prototype | N |")
    lines.append("|---|---|")
    for c, n in enumerate(proto_counts):
        tag = " (DEAD)" if c in dead else ""
        lines.append(f"| {c} | {int(n)}{tag} |")
    lines.append("")
    lines.append(f"Most similar prototype pair: p{i_star} vs p{j_star} "
                 f"(centroid cos-sim = {cos_star:.3f}).")
    lines.append("")
    lines.append("![centroid cosine](eval/fig_centroid_cosine.png)")
    lines.append("")
    lines.append("![prototype usage](eval/fig_prototype_usage.png)")
    lines.append("")
    lines.append("## Training dynamics")
    lines.append("")
    lines.append("![loss curves](eval/fig_loss_curves.png)")
    lines.append("")
    lines.append("![training dynamics](eval/fig_training_dynamics.png)")
    lines.append("")
    lines.append("## Embedding geometry")
    lines.append("")
    lines.append("![embedding norms](eval/fig_embedding_norms.png)")
    lines.append("")
    lines.append("![pairwise similarity](eval/fig_pairwise_sim.png)")
    lines.append("")
    lines.append("## Qualitative")
    lines.append("")
    lines.append(f"![{metrics['umap_method']}](eval/fig_umap.png)")
    lines.append("")
    lines.append("![representative patterns per prototype](eval/fig_representative.png)")
    lines.append("")
    lines.append("![highest-teacher-entropy samples](eval/fig_boundary.png)")
    lines.append("")
    lines.append("![class map on scan](eval/fig_class_map.png)")
    lines.append("")
    lines.append("## Rotation invariance sanity")
    lines.append("")
    lines.append("![theta-roll cosine distribution](eval/fig_rotation_sanity.png)")
    lines.append("")
    lines.append(rotation_line)
    lines.append("")
    lines.append("## Interpretive conclusions")
    lines.append("")
    lines.append("**Prototype usage.** " + collapse_line)
    lines.append("")
    lines.append("**Embedding geometry.** " + geometry_line)
    lines.append("")
    if decouple_line:
        lines.append("**DINO / contrastive coupling.** " + decouple_line)
        lines.append("")
    lines.append("**Physical reading.** "
                 + _physical_interpretation(config_key, lam, troll, metrics))
    lines.append("")
    lines.append("## Concrete recommendations for next runs")
    lines.append("")
    lines.extend(_recommendations(config_key, metrics))
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _physical_interpretation(config_key: str, lam: float, troll: bool,
                              metrics: dict) -> str:
    eff = metrics["effective_K"]
    i_star, j_star, cos_star = metrics["most_similar_pair"]
    rpc = metrics["radial_profile_consistency"]
    rot = metrics["rotation_invariance_cos_median"]
    bits = []
    if config_key == "a":
        bits.append(
            "Config (a) is the pure-DINO L1 baseline with Cartesian rotation "
            "REMOVED (spec 7). Any clustering structure here is driven by "
            "radial-profile differences (peak d-spacing) picked up by the "
            "polar-space ResNet, plus absolute theta-phase accidentally acting "
            "as a latent grouping axis. In real diffraction terms, prototypes "
            "here correspond to intensity patterns whose radial signature "
            "differs enough to separate under prototype softmax.")
    elif config_key == "b":
        bits.append(
            "Config (b) adds per-view theta-roll augmentation on top of (a). "
            "Under circular-theta padding the backbone is approximately "
            "theta-equivariant; theta-roll teaches the teacher/student pair "
            "to align representations across all in-plane diffraction "
            "orientations. Prototypes here should represent phases by their "
            "rotation-invariant fingerprint (radial profile + relative Bragg "
            "structure), not by the grain's absolute orientation.")
    else:
        bits.append(
            "Config (c) adds the soft-label contrastive head on top of (b). "
            "The contrastive target — pairwise cosine similarity of teacher "
            "prototype-softmax rows — injects metric structure into the "
            "projection space: samples whose teacher votes agree should lie "
            "close; samples whose teachers disagree should lie apart. The "
            "student contrastive embedding becomes a second, geometry-rich "
            "view of the same physical partitioning already being learned "
            "by the prototype head.")
    if rpc > 0.3:
        bits.append(
            f"Radial profile consistency is {rpc:.2f} — the prototype "
            "assignments align with distinct azimuthally-averaged diffraction "
            "profiles, which is what you want if prototypes are tracking "
            "phase-identity rather than orientation.")
    elif rpc > 0.1:
        bits.append(
            f"Radial profile consistency is weak ({rpc:.2f}) — prototypes "
            "track something, but radial (peak) structure is not the dominant "
            "axis of separation. Prototypes may be splitting on beam-stop "
            "residue, intensity-scale, or orientation-specific features.")
    else:
        bits.append(
            f"Radial profile consistency is essentially zero ({rpc:.2f}) — "
            "prototype identities are not explained by radial structure. "
            "Either the sample is genuinely homogeneous in peak content, "
            "or prototypes latched onto a non-physical axis.")
    if cos_star > 0.95 and eff > 3:
        bits.append(
            f"Prototypes p{i_star} and p{j_star} have centroid cos-sim "
            f"{cos_star:.2f} — almost certainly redundant phases or the same "
            f"phase split by a non-physical feature. A `lam_sep`-style "
            f"orthogonality push would merge these.")
    if troll and rot > 0.9:
        bits.append(
            "The rotation-invariance sanity check confirms the representation "
            "is stable under theta-roll, which means any residual "
            "orientation-dependent prototype splits are NOT artifacts of the "
            "augmentation pipeline — they would reflect real texture.")
    return " ".join(bits)


def _recommendations(config_key: str, metrics: dict) -> list[str]:
    eff = metrics["effective_K"]
    dead = metrics["dead_prototypes"]
    ratio = metrics["intra_over_inter"]
    i_star, j_star, cos_star = metrics["most_similar_pair"]
    rot = metrics["rotation_invariance_cos_median"]
    recs = []
    if eff < 3:
        recs.append("- Collapsed; restart with a warmer teacher temperature "
                    "(`Tfin=0.08–0.10`) or higher prototype count to avoid "
                    "the local minimum.")
    if len(dead) >= 3:
        recs.append(f"- {len(dead)} dead prototypes — try reducing "
                    f"`num_prototypes` to ~{10 - len(dead)} or enabling "
                    f"`w_ent > 0` to push usage uniform.")
    if cos_star > 0.95:
        recs.append(f"- Prototypes p{i_star}, p{j_star} are near-duplicates "
                    f"(cos={cos_star:.2f}); add a prototype-orthogonality "
                    f"regularizer (`lam_sep=0.05, sep_margin=0.3`) to "
                    f"separate them in a follow-up.")
    if config_key == "c" and isinstance(ratio, float) and ratio < 1.2:
        recs.append("- Intra/inter cosine ratio is weak in contrastive space "
                    "despite the head being active. Try `contrastive_lambda=0.4` "
                    "or `contrastive_entropy_gate=True` to focus the contrastive "
                    "signal on confident samples.")
    if config_key in ("b", "c") and rot < 0.8:
        recs.append("- Rotation invariance cos-median is below 0.8 — check the "
                    "circular-padding wrapper is in place on every Conv2d, and "
                    "widen `theta_shift_range` back to the full polar_size if it "
                    "was clipped.")
    if not recs:
        recs.append("- No obvious pathology. Recommend sweeping "
                    "`(num_prototypes, contrastive_lambda)` next to harden the "
                    "finding.")
    return recs


def epochs_line(train_metrics: dict) -> str:
    loss = train_metrics.get("loss")
    eff_k = train_metrics.get("eff_k")
    if loss is None:
        return ""
    return (f"Final training metrics: loss = {loss:.3f}, effective K (training) "
            f"= {eff_k:.2f}.")


# =========================================================================
# Cross-config report
# =========================================================================

def write_compare_report(sample_dir: str, all_metrics: dict,
                          summaries: dict) -> None:
    path = os.path.join(sample_dir, "COMPARE.md")
    ma = all_metrics["a"]
    mb = all_metrics["b"]
    mc = all_metrics["c"]

    def _d(metric: str) -> str:
        va, vb, vc = ma[metric], mb[metric], mc[metric]
        return f"| {metric} | {va:.3f} | {vb:.3f} | {vc:.3f} |"

    def _delta(a, b):
        return b - a

    lines = []
    lines.append("# Cross-config comparison — Na007b — DINO-SR + Contrastive")
    lines.append("")
    lines.append("## Quantitative metrics across configs")
    lines.append("")
    lines.append("| metric | (a) DINO baseline | (b) +theta-roll | (c) +contrastive |")
    lines.append("|---|---:|---:|---:|")
    for m in ["KNN_purity_k10", "intra_class_cosine", "inter_class_cosine",
              "intra_over_inter", "radial_profile_consistency",
              "prototype_usage_entropy", "effective_K",
              "rotation_invariance_cos_median",
              "rotation_invariance_cos_p10",
              "embed_norm_mean"]:
        try:
            lines.append(_d(m))
        except Exception:
            pass
    lines.append(f"| NMI(b vs a) | — | {mb['NMI_vs_configA']:.3f} | "
                 f"{mc['NMI_vs_configA']:.3f} |")
    lines.append("")

    lines.append("## (a) vs (b) — did theta-roll alone help?")
    lines.append("")
    d_knn = _delta(ma["KNN_purity_k10"], mb["KNN_purity_k10"])
    d_ratio = _delta(ma["intra_over_inter"], mb["intra_over_inter"])
    d_rot = _delta(ma["rotation_invariance_cos_median"],
                    mb["rotation_invariance_cos_median"])
    d_rpc = _delta(ma["radial_profile_consistency"],
                    mb["radial_profile_consistency"])
    lines.append(f"- KNN purity: Δ = {d_knn:+.3f}")
    lines.append(f"- intra/inter cosine ratio: Δ = {d_ratio:+.3f}")
    lines.append(f"- rotation-invariance cos-median: Δ = {d_rot:+.3f}")
    lines.append(f"- radial profile consistency: Δ = {d_rpc:+.3f}")
    lines.append("")
    if d_rot > 0.1 and d_rpc > 0:
        lines.append(
            "Theta-roll augmentation produced a measurable invariance gain "
            f"(cos-median +{d_rot:.2f}) that coincides with improved radial-profile "
            "consistency. This is consistent with the intended mechanism: "
            "enforcing theta-invariance forces the backbone to build features "
            "that depend on what the pattern is (radial structure) rather than "
            "on its absolute orientation.")
    elif d_rot > 0.1 and d_rpc <= 0:
        lines.append(
            "Theta-roll made the representation more rotation-invariant "
            f"(cos-median +{d_rot:.2f}), but radial-profile consistency did not "
            "improve. The invariance gain is real, but it is absorbing aligned "
            "orientation information without redirecting it into radial-"
            "structural features.")
    else:
        lines.append(
            f"Theta-roll produced a weak invariance gain (cos-median "
            f"Δ = {d_rot:+.2f}). Check whether circular padding is actually in "
            f"the backbone (unit test) and whether the roll range is wide enough.")
    lines.append("")

    lines.append("## (b) vs (c) — did contrastive head add value on top of theta-roll?")
    lines.append("")
    d_knn2 = _delta(mb["KNN_purity_k10"], mc["KNN_purity_k10"])
    d_ratio2 = _delta(mb["intra_over_inter"], mc["intra_over_inter"])
    d_nmi = _delta(mb["NMI_vs_configA"], mc["NMI_vs_configA"])
    d_rpc2 = _delta(mb["radial_profile_consistency"],
                     mc["radial_profile_consistency"])
    lines.append(f"- KNN purity: Δ = {d_knn2:+.3f}")
    lines.append(f"- intra/inter cosine ratio: Δ = {d_ratio2:+.3f}")
    lines.append(f"- NMI vs (a): Δ = {d_nmi:+.3f}")
    lines.append(f"- radial profile consistency: Δ = {d_rpc2:+.3f}")
    lines.append("")
    if d_ratio2 > 0.3:
        lines.append(
            "The contrastive head sharpened the metric structure of the "
            "projection space. Intra/inter cosine ratio improved by "
            f"{d_ratio2:+.2f}, which is the signal the auxiliary loss was "
            "designed to produce. Prototypes defined by the DINO head are "
            "now more cleanly separable in the contrastive embedding.")
    elif abs(d_ratio2) < 0.1:
        lines.append(
            "The contrastive head is roughly inert on top of theta-roll. "
            "Possible explanations, in order of likelihood: (1) the L1 "
            "backbone is capacity-limited for this sample and has already "
            "separated the available phases with DINO alone; (2) "
            "`contrastive_lambda=0.2` is too small after the 20-epoch warmup "
            "to reshape the head; (3) the entropy gate is needed so the "
            "contrastive target isn't dominated by high-entropy boundary "
            "samples where the teacher softmax is near-uniform.")
    else:
        lines.append(
            "Contrastive head HURT clustering on (b)'s already-good base "
            f"(intra/inter ratio Δ = {d_ratio2:+.2f}). Likely the target "
            "similarity matrix is too noisy early on (teacher softmax still "
            "warming up) — try a longer warmup or the entropy gate.")
    lines.append("")

    lines.append("## L1 + contrastive vs validated L2 pure-DINO")
    lines.append("")
    lines.append(
        "No L2 run was trained in this experiment (spec: Na007b L1 only). "
        "The user-approved L2 baseline for Na007b lives at "
        "`..\\DINO_PAPER_V2\\Na007b\\L1\\heat_040_050\\best_auto.pth` — a "
        "**pure-DINO L1** run despite the L2 label in some older docs. The "
        "comparison that matters is therefore `(a)` here vs that archived run: "
        "(a) should reproduce the approved class map up to seed variation and "
        "the rotation-removal change. If they diverge, the driver is the "
        "removal of Cartesian RandomRotation, NOT the contrastive machinery. "
        "Recommend running a true L2 (n_layers=2) follow-up under config (c) "
        "to close the capacity comparison the spec calls for.")
    lines.append("")

    lines.append("## Failure-mode scan")
    lines.append("")
    for key, m in all_metrics.items():
        issues = []
        if m["effective_K"] < 3:
            issues.append(f"**collapsed (effK={m['effective_K']:.2f})**")
        if len(m["dead_prototypes"]) > 5:
            issues.append(f"{len(m['dead_prototypes'])} dead prototypes")
        if abs(m["embed_norm_mean"] - 1.0) > 0.02:
            issues.append(f"norm drift ({m['embed_norm_mean']:.3f})")
        if m.get("most_similar_pair", [0, 0, 0])[2] > 0.97:
            ii, jj, cc = m["most_similar_pair"]
            issues.append(f"near-duplicate prototypes p{ii}/p{jj} cos={cc:.2f}")
        if not issues:
            issues.append("no failure modes flagged")
        lines.append(f"- **config ({key})**: " + "; ".join(issues))
    lines.append("")

    lines.append("## Final recommendation")
    lines.append("")
    # Pick winner by KNN purity * (1 if effective_K >= 3 else 0).
    def _score(m):
        penalty = 0.0
        if m["effective_K"] < 3:
            penalty = 0.3
        return m["KNN_purity_k10"] + 0.3 * (m["intra_over_inter"]
                                              if m["intra_over_inter"] == m["intra_over_inter"]
                                              else 0) - penalty

    scores = {k: _score(m) for k, m in all_metrics.items()}
    winner = max(scores, key=scores.get)
    lines.append(f"Winning config by (KNN purity + 0.3 * intra/inter, "
                 f"penalized if collapsed): **config ({winner})** with score "
                 f"{scores[winner]:.3f}. Runner-up scores: " +
                 ", ".join(f"({k}) {v:.3f}" for k, v in scores.items() if k != winner)
                 + ".")
    lines.append("")
    lines.append("Next hyperparameter sweep to harden this conclusion:")
    lines.append("")
    lines.append("- `contrastive_lambda` in {0.1, 0.2, 0.4, 0.8} × `contrastive_entropy_gate` in {False, True} "
                 "(2 factors × 4 × 2 = 8 runs, 50 ep each).")
    lines.append("- `theta_shift_range` in {48, 96, 192} to probe whether a narrower "
                 "rotation-aug window improves under-fitting samples.")
    lines.append("- `num_prototypes` in {6, 8, 10, 14} to test whether dead "
                 "prototypes reflect genuine phase scarcity or over-capacity.")
    lines.append("- Repeat the winning config with `n_layers=2` for the L2 "
                 "capacity comparison the spec calls for.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
