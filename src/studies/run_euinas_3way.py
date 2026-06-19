"""
run_euinas_3way.py — EuInAs 3-way ablation around the pre-gating winner.

Existing baselines (from earlier runs, NOT retrained here):
  winner_polar_centroid       — no gate, no weight, no spatial
  winner_conf_gated           — BOTH gate + weight active (over-corrected)

This script adds three new EuInAs runs:
  winner_gate_only            — entropy_gate=True, gamma=0, lam_spatial=0
  winner_weight_only          — entropy_gate=False, gamma=1, lam_spatial=0
  winner_spatial_only         — entropy_gate=False, gamma=0, lam_spatial=0.1

For each new run:
  1. Train 50 epochs (winner hyperparameters otherwise).
  2. Full eval + class averages + GradCAM + IG.
  3. compare_maps against the previous winner.
  4. analyze_class on whichever class captures the "red dots in middle layer".

Finally, writes EUINAS_3WAY_REPORT.md synthesizing all four configs
(pre-existing winner + 3 new).
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch

# Force UTF-8 on stdout so our markdown/print paths don't crash on the
# Windows cp1252 console when logs contain non-ASCII.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_contrastive import run_config, evaluate_and_report


BASE = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(BASE, "runs")

COMMON = dict(
    sample="EuInAs_B100",
    epochs=50, seed=42, batch_size=128,
    lr=3e-4, weight_decay=1e-6,
    num_prototypes=10,
    t0=0.04, tfin=0.07,
    warmup_epochs=20, ramp_epochs=10,
    projection_dim=128, projection_hidden=256,
    theta_shift_range=None,
    theta_shift_range_student=None, theta_shift_range_teacher=None,
    center_mask_radius=None, center_crop_size=140,
    vmax=None, polar_size=192, polar_mask_cols=30,
    pipeline="polar", centroid_lambda=0.05, centroid_margin=0.3,
)

ABLATIONS = [
    dict(label="winner_gate_only",
         conf_weight_gamma=0.0, entropy_gate_override=True,  lam_spatial=0.0),
    dict(label="winner_weight_only",
         conf_weight_gamma=1.0, entropy_gate_override=False, lam_spatial=0.0),
    dict(label="winner_spatial_only",
         conf_weight_gamma=0.0, entropy_gate_override=False, lam_spatial=0.1),
]


def run_one(ab: dict, device) -> dict | None:
    outdir = os.path.join(RUNS_ROOT, "EuInAs_B100", ab["label"])
    os.makedirs(outdir, exist_ok=True)
    t0 = time.perf_counter()
    print(f"\n{'=' * 72}", flush=True)
    print(f"[{datetime.now():%H:%M:%S}] START EuInAs_B100/{ab['label']}",
          flush=True)
    print(f"  conf_weight_gamma={ab['conf_weight_gamma']} "
          f"entropy_gate={ab['entropy_gate_override']} "
          f"lam_spatial={ab['lam_spatial']}", flush=True)
    print('=' * 72, flush=True)
    try:
        run_config(
            "c", outdir=outdir, device=device,
            entropy_gate=False,                   # default preset gate (unused)
            conf_weight_gamma=ab["conf_weight_gamma"],
            entropy_gate_override=ab["entropy_gate_override"],
            lam_spatial=ab["lam_spatial"],
            **COMMON,
        )
        m = evaluate_and_report("c", sample="EuInAs_B100", outdir=outdir,
                                 device=device)
        elapsed = time.perf_counter() - t0
        m["_label"] = ab["label"]
        m["_elapsed_s"] = elapsed
        print(f"[{datetime.now():%H:%M:%S}] DONE {ab['label']} in "
              f"{elapsed:.0f}s", flush=True)
        return m
    except Exception as exc:
        print(f"[{datetime.now():%H:%M:%S}] FAILED {ab['label']}: {exc!r}",
              flush=True)
        traceback.print_exc()
        return None


def compare_and_analyze(label: str):
    """Run compare_maps + analyze_class on the run vs the winner."""
    try:
        from compare_maps import compare
        compare("EuInAs_B100", old_config="winner_polar_centroid",
                 new_config=label)
    except Exception as exc:
        print(f"[analyze] compare_maps failed for {label}: {exc!r}")
    # analyze_class on class 3 of the new run (if class 3 exists).
    try:
        from analyze_class import analyze
        new_eval = os.path.join(RUNS_ROOT, "EuInAs_B100", label, "eval",
                                 "inference.npz")
        if os.path.exists(new_eval):
            a = np.load(new_eval)["assigns"]
            # Find the class that's MOST visually the "red dots in middle" —
            # proxy: the class whose occupancy spans multiple y-bands in a
            # scattered way. Heuristic: take the class with the highest
            # stripe score (elongated) that isn't the top-y layer.
            from data import SAMPLES
            scan_shape = SAMPLES["EuInAs_B100"]["scan_shape"]
            # Run analyze_class on class with max stripe_metric.
            with open(os.path.join(RUNS_ROOT, "EuInAs_B100", label, "eval",
                                     "metrics.json")) as f:
                m = json.load(f)
            stripes = m.get("stripe_scores", [])
            if stripes:
                # Skip the top layer's class (usually dense id 0 = largest).
                # Pick the class with the highest stripe score that ISN'T
                # dense id 0 (typically top) or dense id 1 (typically middle).
                scored = list(enumerate(stripes))
                scored.sort(key=lambda t: -t[1])
                for c_try, _ in scored:
                    if c_try not in (0, 1):
                        analyze("EuInAs_B100", label, c_try)
                        break
    except Exception as exc:
        print(f"[analyze] analyze_class failed for {label}: {exc!r}")


def _load_metrics(label: str) -> dict | None:
    p = os.path.join(RUNS_ROOT, "EuInAs_B100", label, "eval", "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _load_compare(label: str) -> dict | None:
    p = os.path.join(RUNS_ROOT, "EuInAs_B100", label, "eval",
                      f"compare_winner_polar_centroid_vs_{label}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def write_report(all_metrics: dict):
    out = os.path.join(RUNS_ROOT, "EUINAS_3WAY_REPORT.md")
    labels = list(all_metrics.keys())
    L = []
    L.append(f"# EuInAs 3-way ablation — {datetime.now():%Y-%m-%d %H:%M}")
    L.append("")
    L.append("Probing which failure-mode fix cleans up the class-3 mixing "
              "without over-clustering. All runs: K=10 ceiling, "
              "T₀=0.04 → T_fin=0.07 warmup, asymmetric θ-roll, centroid λ=0.05, "
              "polar pipeline.")
    L.append("")
    L.append("## Summary table")
    L.append("")
    L.append("| label | gate | γ | λ_spatial | K_active | KNN | intra/inter | "
              "effK | stripe_max | most-sim pair | agreement vs winner | "
              "NMI vs winner |")
    L.append("|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|")
    base_descriptors = {
        "winner_polar_centroid":   ("off", "0",  "0.0"),
        "winner_conf_gated":       ("on",  "1",  "0.0"),
        "winner_gate_only":        ("on",  "0",  "0.0"),
        "winner_weight_only":      ("off", "1",  "0.0"),
        "winner_spatial_only":     ("off", "0",  "0.1"),
    }
    for label in labels:
        m = all_metrics[label]
        if m is None:
            L.append(f"| {label} | ? | ? | ? | FAILED | | | | | | | |")
            continue
        g, w, s = base_descriptors.get(label, ("?", "?", "?"))
        comp = _load_compare(label) if label != "winner_polar_centroid" else None
        agree = "(self)" if comp is None else f"{comp['agreement_fraction']:.1%}"
        nmi = "(self)" if comp is None else f"{comp['NMI']:.3f}"
        L.append(
            f"| {label} | {g} | {w} | {s} | {m.get('K_active', m.get('active_prototypes', '?'))} | "
            f"{m['KNN_purity_k10']:.3f} | {m['intra_over_inter']:.2f} | "
            f"{m['effective_K']:.2f} | {m.get('stripe_max', float('nan')):.1f} | "
            f"{m['most_similar_pair'][2]:.2f} | {agree} | {nmi} |"
        )
    L.append("")

    L.append("## Per-label readings")
    L.append("")
    for label in labels:
        m = all_metrics[label]
        if m is None:
            L.append(f"### {label} — FAILED"); L.append(""); continue
        L.append(f"### `{label}`")
        L.append("")
        g, w, s = base_descriptors.get(label, ("?", "?", "?"))
        L.append(f"- flags: **entropy_gate = {g}**, "
                  f"**conf_weight_gamma = {w}**, **lam_spatial = {s}**")
        L.append(f"- K_active = {m.get('K_active', m.get('active_prototypes','?'))}, "
                  f"dead = {m.get('dead_prototypes', [])}")
        L.append(f"- proto counts: {m['proto_counts']}")
        L.append(f"- stripe scores: {[round(x, 1) for x in m.get('stripe_scores', [])]}")
        comp = _load_compare(label) if label != "winner_polar_centroid" else None
        if comp is not None:
            L.append(f"- matched-pair IoU vs winner (Hungarian, sorted):")
            for p in sorted(comp["matched_pairs"], key=lambda pp: -pp["iou"]):
                L.append(f"    - old p{p['old']} vs new p{p['new']}: "
                          f"IoU = {p['iou']:.3f}")
            if comp.get("unmatched_new_classes"):
                L.append(f"- unmatched new classes: "
                          f"{comp['unmatched_new_classes']}")
        L.append("")

    L.append("## Interpretive conclusions")
    L.append("")
    # Compute direction of change for key metrics.
    w = all_metrics.get("winner_polar_centroid")
    if w is not None:
        baseline_ii = w["intra_over_inter"]
        baseline_k = w.get("K_active", w.get("active_prototypes"))
    else:
        baseline_ii = baseline_k = None
    L.append("Treating `winner_polar_centroid` as baseline. Each new run's "
              "delta on intra/inter cosine ratio and K_active:")
    L.append("")
    L.append("| run | ΔK_active | Δintra/inter | reads as |")
    L.append("|---|---:|---:|---|")
    for label in labels:
        if label == "winner_polar_centroid": continue
        m = all_metrics[label]
        if m is None: continue
        dK = m.get("K_active", m.get("active_prototypes", 0)) - (baseline_k or 0)
        dii = m["intra_over_inter"] - (baseline_ii or 0)
        if dK > 0 and dii < 0:
            read = "over-clusters / splits phases, centroids less separated"
        elif dK > 0 and dii > 0:
            read = "over-clusters AND sharpens centroids — ok if new classes are real"
        elif dK < 0 and dii > 0:
            read = "consolidates + sharpens — the ideal direction"
        elif dK == 0 and dii > 0:
            read = "same partition, sharper embedding — purely beneficial"
        elif dK == 0 and dii < 0:
            read = "same partition, softer embedding — unexpected, probably no effect"
        else:
            read = "consolidates but softens — borderline"
        L.append(f"| {label} | {dK:+d} | {dii:+.2f} | {read} |")
    L.append("")
    L.append("### Spatial (λ=0.1) reading")
    L.append("If `winner_spatial_only` has **lower K_active** (or same) and "
              "**higher spatial agreement with the winner on the three clean "
              "layer bands** while REDUCING the scattered-red-dot mixing in "
              "the middle and bottom, that's the fix for layered-material "
              "mixing. The class-3 (or its new equivalent) diagnostic figure "
              "should show a cleaner intra-class cosine and lower entropy-"
              "region fraction.")
    L.append("")
    L.append("### Gate-only vs weight-only reading")
    L.append("- **Gate-only** should produce the gentlest correction: it only "
              "affects WHICH PAIRS contribute to the contrastive loss; DINO "
              "loss is untouched. Expect minor changes vs baseline.")
    L.append("- **Weight-only** should be more aggressive: it reshapes the "
              "backbone's DINO training so confident patterns dominate. Expect "
              "sharper centroids but potentially more over-clustering if there's "
              "any hierarchical structure the confident samples expose.")
    L.append("- **Both** (already run) is what over-corrected.")
    L.append("")
    L.append("See per-run compare figures at "
              "`runs/EuInAs_B100/<label>/eval/fig_compare_winner_polar_centroid_vs_<label>.png`.")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n[report] wrote {out}", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    # Load the pre-existing baselines so the report covers all 5 corners.
    all_metrics: dict = {}
    for label in ("winner_polar_centroid", "winner_conf_gated"):
        all_metrics[label] = _load_metrics(label)

    # Train the 3 new ablations. Skip any whose best.pth already exists
    # (resume-friendly — allows re-running after a mid-script crash).
    for ab in ABLATIONS:
        ckpt = os.path.join(RUNS_ROOT, "EuInAs_B100", ab["label"], "best.pth")
        if os.path.exists(ckpt):
            print(f"[resume] {ab['label']} already trained; loading metrics.",
                  flush=True)
            m = _load_metrics(ab["label"])
        else:
            m = run_one(ab, device=device)
        if m is not None:
            all_metrics[ab["label"]] = m
            compare_and_analyze(ab["label"])
        else:
            all_metrics[ab["label"]] = None

    # Also run compare_maps for winner_conf_gated vs winner (if not done yet).
    if _load_compare("winner_conf_gated") is None:
        try:
            from compare_maps import compare
            compare("EuInAs_B100",
                     old_config="winner_polar_centroid",
                     new_config="winner_conf_gated")
        except Exception as exc:
            print(f"[analyze] retro-compare of winner_conf_gated failed: {exc!r}")

    write_report(all_metrics)


if __name__ == "__main__":
    main()
