"""
run_post_t06.py — runs everything needed for the arXiv manuscript after
the t06 sweep finishes. Waits for the last t06 sentinel, then proceeds.

Sequence (sequential on GPU to avoid contention):

  (1) NMF baseline on remaining samples (EuInAs, Na006a, IMC_150nm) — CPU
      work, kicked off in parallel with (2).
  (2) EuInAs at K=12 with the winner config (no gating) — surfaces the
      orientation-domain substructure that was in Chapter 3's K=6 version.
  (3) L2 backbone benchmark on Na007b, EuInAs, Na006a, IMC_150nm.
  (4) ViT backbone ablation on Na007b.
  (5) Strain analysis between EuInAs K=12 prototypes that look
      orientation-related (applies analyze_strain.py automatically).
  (6) IMC within-film affine analysis (for the crystal-level IMC story):
      apply analyze_strain between pairs of crystalline prototypes in
      IMC_150nm_SI5.
  (7) Write PAPER_READY_SUMMARY.md + a top-level table of all DINO4DSTEM
      results ready for the paper.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch


PY = sys.executable


# Sentinels — t06 is done when all 6 t06 runs have metrics.json.
T06_SENTINELS = [
    os.path.join("runs", "Na007b",      "winner_t06",         "eval", "metrics.json"),
    os.path.join("runs", "Na007b",      "winner_t06_weight",  "eval", "metrics.json"),
    os.path.join("runs", "Na007b",      "winner_t06_spatial", "eval", "metrics.json"),
    os.path.join("runs", "EuInAs_B100", "winner_t06",         "eval", "metrics.json"),
    os.path.join("runs", "EuInAs_B100", "winner_t06_weight",  "eval", "metrics.json"),
    os.path.join("runs", "EuInAs_B100", "winner_t06_spatial", "eval", "metrics.json"),
]


def wait_for_t06(poll_s: int = 60, max_wait_s: int = 4 * 3600) -> bool:
    t0 = time.perf_counter()
    while not all(os.path.exists(p) for p in T06_SENTINELS):
        elapsed = time.perf_counter() - t0
        if elapsed > max_wait_s:
            print(f"[wait] gave up after {elapsed:.0f}s. Proceeding with whatever is done.",
                  flush=True)
            return False
        missing = [p for p in T06_SENTINELS if not os.path.exists(p)]
        print(f"[wait] {datetime.now():%H:%M:%S}  missing {len(missing)} t06 sentinels "
              f"(elapsed {elapsed:.0f}s)", flush=True)
        time.sleep(poll_s)
    print(f"[wait] t06 sweep done; proceeding.", flush=True)
    return True


# =========================================================================
# Step 1: NMF baselines on remaining samples (CPU only, parallel with GPU steps)
# =========================================================================

def run_nmf_baselines_in_background():
    """Kick off NMF baselines for the remaining samples as a detached
    subprocess so it can run in parallel with the GPU work."""
    print(f"[{datetime.now():%H:%M:%S}] launching NMF baselines (CPU) in background", flush=True)
    # We chain them in one python call so we own one subprocess.
    samples = [
        ("EuInAs_B100",   "winner_polar_centroid"),
        ("Na006a",        "winner_polar_centroid"),
        ("IMC_150nm_SI5", "winner_polar_centroid"),
    ]
    script = "; ".join([
        f"python baseline_nmf_kmeans.py --sample {s} --dino-config {c}"
        for s, c in samples
    ])
    # Writing a tiny helper rather than shelling out in a weird way.
    log_path = os.path.join("runs", "nmf_baselines_remaining.log")
    f = open(log_path, "w")
    # Launch each sample sequentially in a single python process per sample.
    pids = []
    for s, c in samples:
        cmd = [PY, "baseline_nmf_kmeans.py", "--sample", s, "--dino-config", c]
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        pids.append(p)
        print(f"  launched NMF on {s} (PID {p.pid})", flush=True)
        # NMF runs sequentially on CPU to avoid RAM explosion; wait for each.
        p.wait()
    f.close()
    print(f"[{datetime.now():%H:%M:%S}] NMF baselines done", flush=True)


# =========================================================================
# Step 2: EuInAs K=12 run
# =========================================================================

def run_euinas_K12(device) -> dict | None:
    from run_contrastive import run_config, evaluate_and_report

    outdir = os.path.join("runs", "EuInAs_B100", "winner_K12")
    if os.path.exists(os.path.join(outdir, "eval", "metrics.json")):
        print(f"[skip] EuInAs K=12 already done", flush=True)
        with open(os.path.join(outdir, "eval", "metrics.json")) as f:
            return json.load(f)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.perf_counter()
    print(f"\n[{datetime.now():%H:%M:%S}] START EuInAs K=12 (winner config, higher K ceiling)",
          flush=True)
    try:
        run_config(
            "c", sample="EuInAs_B100", epochs=50, seed=42, batch_size=128,
            lr=3e-4, weight_decay=1e-6, num_prototypes=12,
            t0=0.04, tfin=0.07, warmup_epochs=20, ramp_epochs=10,
            entropy_gate=False,
            projection_dim=128, projection_hidden=256,
            theta_shift_range=None,
            theta_shift_range_student=None, theta_shift_range_teacher=None,
            center_mask_radius=None, center_crop_size=140,
            vmax=None, polar_size=192, polar_mask_cols=30,
            pipeline="polar", centroid_lambda=0.05, centroid_margin=0.3,
            conf_weight_gamma=0.0, entropy_gate_override=False,
            lam_spatial=0.0,
            outdir=outdir, device=device,
        )
        m = evaluate_and_report("c", sample="EuInAs_B100", outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] DONE EuInAs K=12 in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        return m
    except Exception as exc:
        print(f"[fail] EuInAs K=12: {exc!r}", flush=True)
        traceback.print_exc()
        return None


# =========================================================================
# Step 3: L2 backbone benchmark (4 samples)
# =========================================================================

def run_L2_benchmark(device):
    from run_contrastive import run_config, evaluate_and_report
    samples = [
        ("Na007b",        "sweep_polar_centroid"),
        ("EuInAs_B100",   "winner_polar_centroid"),
        ("Na006a",        "winner_polar_centroid"),
        ("IMC_150nm_SI5", "winner_polar_centroid"),
    ]
    for sample, winner_folder in samples:
        outdir = os.path.join("runs", sample, "winner_L2")
        metrics_path = os.path.join(outdir, "eval", "metrics.json")
        if os.path.exists(metrics_path):
            print(f"[skip] L2 / {sample}: already evaluated", flush=True)
            continue
        os.makedirs(outdir, exist_ok=True)
        print(f"\n[{datetime.now():%H:%M:%S}] START L2 / {sample}", flush=True)
        t0 = time.perf_counter()
        try:
            run_config(
                "c", sample=sample, epochs=50, seed=42, batch_size=128,
                lr=3e-4, weight_decay=1e-6, num_prototypes=10,
                t0=0.04, tfin=0.07, warmup_epochs=20, ramp_epochs=10,
                entropy_gate=False,
                projection_dim=128, projection_hidden=256,
                theta_shift_range=None,
                theta_shift_range_student=None, theta_shift_range_teacher=None,
                center_mask_radius=None, center_crop_size=140,
                vmax=None, polar_size=192, polar_mask_cols=30,
                pipeline="polar", centroid_lambda=0.05, centroid_margin=0.3,
                conf_weight_gamma=0.0, entropy_gate_override=False,
                lam_spatial=0.0,
                architecture="resnet", n_layers=2,
                outdir=outdir, device=device,
            )
            evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
            print(f"[{datetime.now():%H:%M:%S}] DONE L2/{sample} in "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
            try:
                from compare_maps import compare
                compare(sample, old_config=winner_folder, new_config="winner_L2")
            except Exception as exc:
                print(f"  [compare] {exc!r}")
        except Exception as exc:
            print(f"[fail] L2/{sample}: {exc!r}", flush=True)
            traceback.print_exc()


# =========================================================================
# Step 4: ViT backbone ablation
# =========================================================================

def run_vit_ablation(device):
    from run_contrastive import run_config, evaluate_and_report
    sample = "Na007b"
    outdir = os.path.join("runs", sample, "winner_vit")
    if os.path.exists(os.path.join(outdir, "eval", "metrics.json")):
        print(f"[skip] ViT/{sample}: already evaluated", flush=True)
        return
    os.makedirs(outdir, exist_ok=True)
    print(f"\n[{datetime.now():%H:%M:%S}] START ViT / {sample}", flush=True)
    t0 = time.perf_counter()
    try:
        run_config(
            "c", sample=sample, epochs=50, seed=42, batch_size=128,
            lr=3e-4, weight_decay=1e-6, num_prototypes=10,
            t0=0.04, tfin=0.07, warmup_epochs=20, ramp_epochs=10,
            entropy_gate=False,
            projection_dim=128, projection_hidden=256,
            theta_shift_range=None,
            theta_shift_range_student=None, theta_shift_range_teacher=None,
            center_mask_radius=None, center_crop_size=140,
            vmax=None, polar_size=192, polar_mask_cols=30,
            pipeline="polar", centroid_lambda=0.05, centroid_margin=0.3,
            conf_weight_gamma=0.0, entropy_gate_override=False,
            lam_spatial=0.0,
            architecture="vit", n_layers=1,
            outdir=outdir, device=device,
        )
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] DONE ViT/{sample} in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        try:
            from compare_maps import compare
            compare(sample, old_config="sweep_polar_centroid", new_config="winner_vit")
        except Exception as exc:
            print(f"  [compare] {exc!r}")
    except Exception as exc:
        print(f"[fail] ViT/{sample}: {exc!r}", flush=True)
        traceback.print_exc()


# =========================================================================
# Step 5: Strain analysis on EuInAs K=12
# =========================================================================

def run_strain_analysis_euinas():
    """On the EuInAs K=12 run, auto-pick the two most-similar film-class
    prototypes (by radial profile) and run the LoG+RANSAC pipeline.
    Also run it on every adjacent-by-spatial-proximity pair as a sweep.
    """
    from data import SAMPLES, LoadPRZ
    from analyze_strain import strain_between_classes

    sample = "EuInAs_B100"
    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    run_dir = os.path.join("runs", sample, "winner_K12")
    inf_path = os.path.join(run_dir, "eval", "inference.npz")
    if not os.path.exists(inf_path):
        print(f"[skip] strain: K=12 inference missing", flush=True)
        return
    inf = np.load(inf_path)
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]
    K = soft_probs.shape[1]

    # Compute class means for every active class.
    means = {}
    for c in range(K):
        idx = np.where(assigns == c)[0]
        if len(idx) < 20:
            continue
        scores = soft_probs[idx, c]
        top = idx[np.argsort(-scores)[:min(300, len(idx))]]
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0).astype(np.float32)
        w = soft_probs[top, c].astype(np.float32)
        means[c] = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
    classes = sorted(means.keys())

    # Rank pairs by radial-profile similarity so we focus on plausibly
    # orientation-related pairs, not e.g. vacuum vs substrate.
    from analyze_imc import radial_profile
    profiles = {c: radial_profile(means[c], r_min=15, n_bins=80)[0] for c in classes}
    def _norm(p):
        p = p - p.min()
        return p / (np.linalg.norm(p) + 1e-12)
    profN = {c: _norm(profiles[c]) for c in classes}

    pair_scores = []
    for i, ci in enumerate(classes):
        for cj in classes[i + 1:]:
            s = float(profN[ci] @ profN[cj])
            pair_scores.append((ci, cj, s))
    pair_scores.sort(key=lambda t: -t[2])

    # Take the top-5 most similar pairs (candidates for orientation-related).
    top_pairs = pair_scores[:5]
    print(f"[strain] top-5 most-similar prototype pairs by radial profile:",
          flush=True)
    for ci, cj, s in top_pairs:
        print(f"  p{ci} vs p{cj}: radial-profile cos = {s:.3f}", flush=True)
    summary = []
    for ci, cj, s in top_pairs:
        out_dir = os.path.join(run_dir, "eval", f"strain_p{ci}_vs_p{cj}")
        res = strain_between_classes(
            means[ci], means[cj], out_dir,
            label_A=f"p{ci}", label_B=f"p{cj}",
            blob_threshold=0.03, match_gate_px=15,
        )
        res["radial_cos"] = s
        summary.append(res)
    with open(os.path.join(run_dir, "eval", "strain_sweep.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[{datetime.now():%H:%M:%S}] strain sweep done -> "
          f"{run_dir}/eval/strain_sweep.json", flush=True)


# =========================================================================
# Step 6: IMC within-film crystal-identity analysis
# =========================================================================

def run_imc_affine():
    """For IMC_150nm, apply the affine analysis pipeline between pairs of
    its crystalline prototypes. Two prototypes whose affine is near-identity
    (just rotation + small shear, near-unit scale) are the same polymorph
    in different orientations. Pairs whose affine needs large scaling are
    different polymorphs.
    """
    from data import SAMPLES, LoadPRZ
    from analyze_strain import strain_between_classes
    from analyze_imc import radial_profile, crystallinity_score

    sample = "IMC_150nm_SI5"
    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    run_dir = os.path.join("runs", sample, "winner_polar_centroid")
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]
    K = soft_probs.shape[1]

    # Compute class means.
    means = {}
    xtal = {}
    for c in range(K):
        idx = np.where(assigns == c)[0]
        if len(idx) < 50:
            continue
        scores = soft_probs[idx, c]
        top = idx[np.argsort(-scores)[:min(300, len(idx))]]
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0).astype(np.float32)
        w = soft_probs[top, c].astype(np.float32)
        means[c] = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
        prof, _ = radial_profile(means[c], r_min=15, n_bins=80)
        xtal[c] = crystallinity_score(prof)
    # Only run affine on crystalline prototypes.
    crystal_classes = [c for c, x in xtal.items() if x >= 1.5]
    print(f"[imc_affine] crystalline classes: {crystal_classes}", flush=True)
    summary = []
    for i, ci in enumerate(crystal_classes):
        for cj in crystal_classes[i + 1:]:
            out_dir = os.path.join(run_dir, "eval", f"strain_p{ci}_vs_p{cj}")
            res = strain_between_classes(
                means[ci], means[cj], out_dir,
                label_A=f"p{ci}", label_B=f"p{cj}",
                blob_threshold=0.03, match_gate_px=12,
            )
            summary.append(res)
    with open(os.path.join(run_dir, "eval", "imc_affine_sweep.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[{datetime.now():%H:%M:%S}] IMC affine sweep done -> "
          f"{run_dir}/eval/imc_affine_sweep.json", flush=True)


# =========================================================================
# Step 7: Paper-ready summary
# =========================================================================

def write_paper_summary():
    """Consolidate a top-level table with every DINO4DSTEM result variant
    vs NMF baseline for the paper Methods/Results Table 1."""
    rows = []
    # Collect all runs we have metrics.json for.
    base = "runs"
    for sample in os.listdir(base):
        sdir = os.path.join(base, sample)
        if not os.path.isdir(sdir):
            continue
        for cfg in os.listdir(sdir):
            p = os.path.join(sdir, cfg, "eval", "metrics.json")
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        m = json.load(f)
                    rows.append(dict(
                        sample=sample, config=cfg,
                        KNN=m.get("KNN_purity_k10"),
                        intra_inter=m.get("intra_over_inter"),
                        K_active=m.get("K_active", m.get("active_prototypes")),
                        effK=m.get("effective_K"),
                        stripe_max=m.get("stripe_max"),
                    ))
                except Exception:
                    continue
    with open(os.path.join(base, "PAPER_SUMMARY_TABLE.json"), "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"[{datetime.now():%H:%M:%S}] wrote {base}/PAPER_SUMMARY_TABLE.json "
          f"({len(rows)} rows)", flush=True)


# =========================================================================
# Driver
# =========================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[post_t06] device = {device}", flush=True)

    wait_for_t06()

    # NMF baselines for other samples — CPU only, can interleave with GPU work.
    # We run them after t06 to keep CPU/RAM sane (t06 already uses CPU for
    # data loading).
    try:
        run_nmf_baselines_in_background()
    except Exception as exc:
        print(f"[fail] NMF batch: {exc!r}", flush=True)
        traceback.print_exc()

    # EuInAs K=12 — first because strain analysis depends on it.
    run_euinas_K12(device)

    # L2 benchmark on 4 samples (the largest GPU block).
    run_L2_benchmark(device)

    # ViT ablation on Na007b.
    run_vit_ablation(device)

    # Strain pipeline sweep on EuInAs K=12.
    try:
        run_strain_analysis_euinas()
    except Exception as exc:
        print(f"[fail] strain sweep: {exc!r}", flush=True)
        traceback.print_exc()

    # IMC within-film affine sweep.
    try:
        run_imc_affine()
    except Exception as exc:
        print(f"[fail] imc affine: {exc!r}", flush=True)
        traceback.print_exc()

    write_paper_summary()
    print(f"\n[{datetime.now():%H:%M:%S}] post_t06 all done.", flush=True)


if __name__ == "__main__":
    main()
