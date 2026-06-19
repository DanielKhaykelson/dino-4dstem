"""run_line_coverage_pipeline.py -- end-to-end NaPHI / MgNaPHI line-phase
outlier analysis.

Goal: show that MgNaPHI SI-007 (predicted-low-Mg, EDS-confirmed) lands
between bulk NaPHI and bulk MgNaPHI on a quantitative line-phase coverage
axis. Strong unsupervised outlier detection + transfer-learning claim.

Pipeline:
    1. Wait for runs/_paper_master/_done.flag (so we don't fight master
        sweep for the GPU). Skip wait if --no-wait.
    2. Build a combined training set: 2 NaPHI + 2 bulk-MgNaPHI cubes (via
        LoadPRZMulti). Compute or load per-cube radials, concatenate, and
        recalibrate gate thresholds on the combined.
    3. Train one model on the combined cube with the locked recipe
        (DINO + cluster1d, lambda_1d=0.1, gamma=0.5, K=8, deterministic).
    4. Identify the LINE prototype automatically:
        - Compute class-average diffraction for each prototype (top-N
          confidence-weighted).
        - Polar-transform.
        - Score line-ness as theta_Gini / radial_Gini (lines: high theta
          concentration + low radial concentration -> high score; spots:
          ~1; rings: low; vacuum: small total intensity, gated out).
        - Pick argmax. Save fig_line_phase_pick.png with all class-avg
          panels annotated by score, picked one starred.
    5. Eval on all NaPHI + MgNaPHI samples that meet the README criteria
        (115k mag, 58mm CL). For each: line_frames / total_frames.
    6. Plot 1D scatter colored by family + write metrics_summary.json.

Output: runs/_linecov/
    train/                          (training run dir, model + ckpts + eval)
    fig_line_phase_pick.png         (which prototype is the line phase)
    fig_coverage_scatter.png        (final paper plot)
    metrics_summary.json            (all per-sample numbers)
"""
from __future__ import annotations
import argparse, os, sys, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from data import SAMPLES, LoadPRZ, LoadPRZMulti
from dino_sr_contrastive_model import load_contrastive_checkpoint
from contrastive_eval import infer_scan
from compute_radial_profile import (
    compute_radial as compute_radial_for_sample,
    calibrate_thresholds,
)

# =========================================================================
# Configuration
# =========================================================================

OUT_ROOT = os.path.join("runs", "_linecov")
TRAIN_DIR = os.path.join(OUT_ROOT, "train")
WAIT_FOR = os.path.join("runs", "_paper_master", "_done.flag")

# Training set: pick 2 NaPHI (line-rich) + 2 bulk MgNaPHI (spot-rich) so
# the trained model has prototypes for BOTH regimes.
TRAIN_SAMPLES = [
    "NaPHI_Nadja_SI002",
    "NaPHI_Nadja_SI003",
    "MgNaPHI_remeas_SI004",
    "MgNaPHI_remeas_SI005",
]
COMBINED_RADIAL_BASE = r"D:\DINOSR\data\_linecov_combined"

# Targets: every NaPHI / MgNaPHI sample that meets README criteria
# (115k mag, 58mm CL), categorized for the scatter plot.
# (sample_key, family_label, marker)
#   family_label is a small human-readable string (used for plot legend);
#   the README inclusion / exclusion is hardcoded based on README parsing.
TARGETS = [
    # Bulk NaPHI: known-line samples
    ("NaPHI_Nadja_SI002", "NaPHI bulk"),
    ("NaPHI_Nadja_SI003", "NaPHI bulk"),
    ("NaPHI_Nadja_SI004", "NaPHI bulk"),
    ("NaPHI_Nadja_SI009", "NaPHI bulk"),
    ("NaPHI_Nadja_SI010", "NaPHI bulk"),
    # 4-quarter same ROI (reproducibility check; same expected coverage):
    ("NaPHI_Nadja_SI005", "NaPHI 4Q"),
    ("NaPHI_Nadja_SI006", "NaPHI 4Q"),
    ("NaPHI_Nadja_SI007", "NaPHI 4Q"),
    ("NaPHI_Nadja_SI008", "NaPHI 4Q"),
    # Bulk MgNaPHI (spot-rich)
    ("MgNaPHI_remeas_SI001", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI003", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI004", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI005", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI006", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI010", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI011", "MgNaPHI bulk"),
    # Predicted outlier + beam-damage controls (same area, different scans)
    ("MgNaPHI_remeas_SI007", "MgNaPHI SI-007 (outlier)"),
    ("MgNaPHI_remeas_SI008", "MgNaPHI SI-008 (=SI-007 area)"),
    ("MgNaPHI_remeas_SI009", "MgNaPHI SI-009 (=SI-007 area)"),
]

K_TRAIN = 8                      # combined dataset has line + spot + vacuum + supports
EPOCHS = 30
SEED = 42

# Recipe (locked)
LAM_1D = 0.1
GAMMA = 0.5
MARGIN = 0.4


# =========================================================================
# Helpers
# =========================================================================

def _radial_path_for(sample: str) -> str:
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    return base + ".radial.npy"


def _wait_for_sentinel(path: str, poll_sec: int = 60):
    if os.path.exists(path):
        print(f"[linecov] sentinel already present", flush=True)
        return
    print(f"[linecov] waiting for {path}", flush=True)
    t0 = time.perf_counter()
    while not os.path.exists(path):
        time.sleep(poll_sec)
    print(f"[linecov] sentinel found after "
          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)


def _ensure_radials_for(sample: str, device):
    rad = _radial_path_for(sample)
    if os.path.exists(rad):
        return rad
    print(f"[linecov] computing radials for {sample}", flush=True)
    radials = compute_radial_for_sample(sample, device=device)
    np.save(rad, radials)
    print(f"  saved {rad}  shape={radials.shape}", flush=True)
    return rad


def _build_combined_radial_set(device):
    """Concatenate per-cube radials in the order LoadPRZMulti loads them."""
    combined_npy = COMBINED_RADIAL_BASE + ".radial.npy"
    combined_th = COMBINED_RADIAL_BASE + ".gate_thresholds.json"
    if os.path.exists(combined_npy) and os.path.exists(combined_th):
        print(f"[linecov] combined radial set already exists", flush=True)
        return combined_npy, combined_th

    parts = []
    for s in TRAIN_SAMPLES:
        rad_path = _ensure_radials_for(s, device)
        parts.append(np.load(rad_path))
    combined = np.concatenate(parts, axis=0).astype(np.float32)
    np.save(combined_npy, combined)
    print(f"[linecov] combined radial: shape={combined.shape}  "
          f"(parts={[p.shape[0] for p in parts]})", flush=True)

    th = calibrate_thresholds(combined, n_pairs=50_000,
                                frac_pos=0.15, frac_neg=0.50)
    th["sample"] = "LINECOV_COMBINED"
    with open(combined_th, "w") as f:
        json.dump(th, f, indent=2)
    print(f"[linecov] tau_pos={th['tau_pos']:.4f}  "
          f"tau_neg={th['tau_neg']:.4f}", flush=True)
    return combined_npy, combined_th


def _register_combined_sample():
    """Add a multi-cube entry to SAMPLES so run_config picks it up."""
    SAMPLES["LINECOV_COMBINED"] = {
        "paths": [SAMPLES[s]["path"] for s in TRAIN_SAMPLES],
        "vmax": 2,
        "scan_shape": (100, 100),  # nominal; not used at training
        "center_mask_radius": 15,
        "approved_label": None,
        "is_multi": True,
    }


def _train_combined(combined_radial: str, combined_th: str, device):
    if os.path.exists(os.path.join(TRAIN_DIR, "best.pth")):
        print(f"[linecov] train already done at {TRAIN_DIR}", flush=True)
        return
    from run_contrastive import run_config
    os.makedirs(TRAIN_DIR, exist_ok=True)
    print(f"\n[{datetime.now():%H:%M:%S}] TRAIN combined  K={K_TRAIN}  "
          f"ep={EPOCHS}", flush=True)
    warmup = int(round((2.0 / 3.0) * EPOCHS))
    ramp = int(round((1.0 / 3.0) * EPOCHS))
    run_config("c", sample="LINECOV_COMBINED", outdir=TRAIN_DIR, device=device,
        epochs=EPOCHS, seed=SEED, batch_size=128,
        lr=3e-4, weight_decay=1e-6,
        num_prototypes=K_TRAIN,
        t0=0.04, tfin=0.07,
        warmup_epochs=warmup, ramp_epochs=ramp,
        entropy_gate=False,
        projection_dim=128, projection_hidden=256,
        theta_shift_range=None,
        theta_shift_range_student=192, theta_shift_range_teacher=16,
        center_mask_radius=15,
        center_crop_size=140,
        vmax=None,
        polar_size=192, polar_mask_cols=45,
        pipeline="polar",
        centroid_lambda=0.0, centroid_margin=0.3,
        conf_weight_gamma=GAMMA,
        entropy_gate_override=None,
        lam_spatial=0.0,
        architecture="resnet", n_layers=1,
        w_ent=0.0,
        com_centering=True,
        com_search_radius_factor=2.0,
        aug_disable=["hflip", "vflip", "colorjitter"],
        supcon_radials_path=combined_radial,
        supcon_thresholds_path=combined_th,
        supcon_lambda=0.0,
        supcon_temperature=0.3,
        contrastive_lambda_override=0.0,
        proto_repel_lambda=0.0,
        proto_repel_threshold=0.5,
        cluster1d_lambda=LAM_1D,
        cluster1d_margin=MARGIN,
        cluster1d_min_cluster_mass=1.0,
        cluster1d_warmup_frac=0.0,
        cluster1d_ramp_frac=0.0,
    )


# =========================================================================
# Line-phase identification
# =========================================================================

def _polar_warp(img: np.ndarray, n_theta: int = 360,
                 n_r: "int | None" = None) -> np.ndarray:
    import cv2
    H, W = img.shape
    cx, cy = W / 2.0, H / 2.0
    if n_r is None:
        n_r = int(min(cx, cy))
    polar = cv2.warpPolar(img.astype(np.float32), (n_r, n_theta), (cx, cy),
                            n_r, cv2.WARP_POLAR_LINEAR)
    return polar


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).clip(min=0))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    return (2 * np.sum(np.arange(1, n + 1) * x)) / (n * x.sum()) - (n + 1) / n


def _beam_mask(H, W, radius):
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    return ((yy - cy) ** 2 + (xx - cx) ** 2) > radius ** 2


def line_score(class_avg: np.ndarray, beam_mask_radius: int = 40):
    """Line-ness score: theta_gini / radial_gini.

      LINE   : narrow theta + extended radial -> theta_gini high, radial low
      SPOT   : narrow theta + narrow radial   -> both high; ratio ~1
      RING   : wide theta   + narrow radial   -> theta low, radial high; low
      VACUUM : low intensity                  -> small total, gated separately
    """
    H, W = class_avg.shape
    bm = _beam_mask(H, W, beam_mask_radius)
    img = class_avg * bm
    polar = _polar_warp(img)
    I_theta = polar.sum(axis=1)
    I_r = polar.sum(axis=0)
    tg = _gini(I_theta)
    rg = _gini(I_r)
    return float(tg / max(rg, 1e-6)), float(tg), float(rg), float(img.sum())


def _confidence_weighted_class_avg(dataset, soft_probs, assigns, c, n_top=300):
    idx = np.where(assigns == c)[0]
    if idx.size == 0:
        return None, 0
    scores = soft_probs[idx, c]
    top = idx[np.argsort(-scores)[:min(n_top, len(idx))]]
    patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0).astype(np.float32)
    w = soft_probs[top, c].astype(np.float32)
    return ((patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12),
            int(idx.size))


def _identify_line_prototype(model, train_dataset, device, train_dir):
    """Compute class avg for each of K prototypes, score line-ness, pick max.
    Run inference on the TRAINING dataset (the one the model was trained on).
    Save fig_line_phase_pick.png annotating each panel with its score.
    """
    print(f"\n[linecov] identifying line prototype...", flush=True)
    K = model.prototypes.prototypes.shape[0]
    inf = infer_scan(model, train_dataset, device, dense_remap=False,
                      polar_size=192, polar_mask_cols=45,
                      center_crop_size=140,
                      com_centering=True, center_mask_radius=15,
                      eval_temp=0.06, batch_size=128)
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]
    counts = np.bincount(assigns, minlength=K)
    print(f"  per-prototype counts: {counts.tolist()}", flush=True)

    avgs = {}
    scores = {}
    intensities = {}
    for c in range(K):
        avg, n = _confidence_weighted_class_avg(train_dataset, soft_probs,
                                                  assigns, c)
        if avg is None or counts[c] < 10:
            avgs[c] = None
            scores[c] = 0.0
            intensities[c] = 0.0
            continue
        avgs[c] = avg
        s, tg, rg, total_int = line_score(avg)
        scores[c] = s
        intensities[c] = total_int
    # Vacuum gate: if a prototype's avg total intensity is less than 30% of
    # the largest prototype's intensity, it is likely vacuum -> exclude.
    max_int = max(intensities.values()) if intensities else 1.0
    valid = {c: s for c, s in scores.items()
              if avgs[c] is not None and intensities[c] >= 0.3 * max_int}
    if not valid:
        raise RuntimeError("no valid (non-vacuum) prototypes found")
    line_proto = max(valid, key=valid.get)
    print(f"  line prototype = p{line_proto}  (score={valid[line_proto]:.3f})",
          flush=True)
    print(f"  all scores: " +
          ", ".join([f"p{c}:{scores[c]:.2f}" for c in range(K)]), flush=True)

    # Render annotated panels
    H = avgs[next(iter([c for c in avgs if avgs[c] is not None]))].shape[0]
    bm = _beam_mask(H, H, 40)
    cols = min(K, 4)
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.6),
                              squeeze=False)
    for c in range(K):
        ax = axes[c // cols, c % cols]
        if avgs[c] is None:
            ax.set_axis_off()
            continue
        disp = avgs[c] * bm
        # log + percentile clip for display
        ref = disp[bm]
        if ref.size:
            lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
            disp = np.log1p(np.clip(disp, lo, hi) - lo)
        star = "*" if c == line_proto else " "
        ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
        ax.set_title(f"{star} p{c}  N={counts[c]}\n"
                      f"line-score={scores[c]:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(K, rows * cols):
        axes[j // cols, j % cols].set_axis_off()
    fig.suptitle(f"Line-prototype identification "
                  f"(line proto = p{line_proto})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_ROOT, "fig_line_phase_pick.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[linecov] wrote {out}", flush=True)
    return line_proto, scores, counts


# =========================================================================
# Multi-sample evaluation
# =========================================================================

def _eval_target(model, sample, line_proto, device, render_paper_outputs=True):
    cfg = SAMPLES[sample]
    ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    inf = infer_scan(model, ds, device, dense_remap=False,
                      polar_size=192, polar_mask_cols=45,
                      center_crop_size=140,
                      com_centering=True, center_mask_radius=15,
                      eval_temp=0.06, batch_size=128)
    assigns = inf["assigns"]
    K = inf["soft_probs"].shape[1]
    counts = np.bincount(assigns, minlength=K).tolist()
    line_count = int((assigns == line_proto).sum())
    total = int(len(assigns))

    if render_paper_outputs:
        # Persist per-sample inference + run viz_paper_outputs against it so
        # the user can inspect per-sample class composition (especially the
        # MgNaPHI samples). We synthesize a minimal run_dir layout that the
        # paper-outputs scripts expect: <run_dir>/eval/inference.npz +
        # run_summary.json (the latter only for the polar mask values).
        per_sample_dir = os.path.join(OUT_ROOT, "per_sample", sample)
        eval_dir = os.path.join(per_sample_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        np.savez(os.path.join(eval_dir, "inference.npz"),
                  soft_probs=inf["soft_probs"], assigns=assigns,
                  embeds=inf["embeds"])
        # minimal run_summary so viz scripts pick the right mask + crop
        with open(os.path.join(per_sample_dir, "run_summary.json"), "w") as f:
            json.dump({"cfg": {
                "center_mask_radius": 15, "polar_mask_cols": 45,
                "polar_size": 192, "center_crop_size": 140,
            }}, f)
        try:
            import viz_paper_outputs
            viz_paper_outputs.render_class_map(per_sample_dir, sample)
            viz_paper_outputs.render_class_averages_and_examples(
                per_sample_dir, sample, n_examples=100)
        except Exception as e:
            print(f"  [warn] paper-outputs failed for {sample}: {e!r}",
                  flush=True)

    return {
        "sample": sample,
        "K": int(K),
        "counts": counts,
        "line_count": line_count,
        "total": total,
        "coverage": float(line_count / max(total, 1)),
    }


def _plot_scatter(results, out):
    """1D scatter, x=coverage, y=jittered category."""
    families = []
    seen = []
    for r in results:
        if r["family"] not in seen:
            seen.append(r["family"])
        families.append(r["family"])
    fam_to_y = {f: i for i, f in enumerate(seen)}
    rng = np.random.default_rng(0)
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(11, 0.8 * len(seen) + 2))
    for r in results:
        y = fam_to_y[r["family"]] + (rng.random() - 0.5) * 0.2
        ax.scatter([r["coverage"]], [y], s=140,
                    color=colors[fam_to_y[r["family"]] % len(colors)],
                    edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(r["sample"].replace("_remeas_", "_").replace("_Nadja_", "_"),
                     (r["coverage"], y), xytext=(6, 0),
                     textcoords="offset points", fontsize=8, va="center")
    ax.set_yticks(range(len(seen)))
    ax.set_yticklabels(seen)
    ax.set_xlabel("line-phase coverage = line_frames / total_frames")
    ax.set_xlim(-0.02, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Line-phase coverage across NaPHI / MgNaPHI samples\n"
                  "(MgNaPHI SI-007 is predicted-low-Mg, EDS-confirmed)")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-wait", action="store_true",
                     help="Don't wait for the master sweep sentinel")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    if not args.no_wait:
        _wait_for_sentinel(WAIT_FOR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[linecov] device={device}", flush=True)

    # Step 2-3: combined radials + register multi-cube sample
    combined_npy, combined_th = _build_combined_radial_set(device)
    _register_combined_sample()

    # Step 4: train (skips if best.pth exists)
    _train_combined(combined_npy, combined_th, device)

    # Step 5-6: load model + identify line prototype
    ckpt_path = os.path.join(TRAIN_DIR, "best.pth")
    model, _, _, _ = load_contrastive_checkpoint(ckpt_path, device=device)
    train_ds = LoadPRZMulti(SAMPLES["LINECOV_COMBINED"]["paths"],
                              resize=192, vmax=2)
    line_proto, scores_dict, train_counts = _identify_line_prototype(
        model, train_ds, device, TRAIN_DIR)
    line_info = {
        "line_proto": int(line_proto),
        "all_scores": {f"p{c}": float(s) for c, s in scores_dict.items()},
        "training_counts": [int(x) for x in train_counts],
    }

    # Step 7: eval on every target
    print(f"\n[linecov] evaluating on {len(TARGETS)} target samples...",
          flush=True)
    rows = []
    for sample, family in TARGETS:
        if sample not in SAMPLES:
            print(f"  [SKIP] {sample}: not in SAMPLES registry", flush=True)
            continue
        path = SAMPLES[sample]["path"]
        if not os.path.exists(path):
            print(f"  [SKIP] {sample}: file missing ({path})", flush=True)
            continue
        try:
            r = _eval_target(model, sample, line_proto, device)
            r["family"] = family
            rows.append(r)
            print(f"  {sample:<28} family={family:<28}  "
                  f"coverage={r['coverage']:.4f}  "
                  f"line/total={r['line_count']}/{r['total']}", flush=True)
        except Exception as e:
            print(f"  [FAIL] {sample}: {e!r}", flush=True)
            import traceback; traceback.print_exc()

    # Step 8: plot + JSON
    out_scatter = os.path.join(OUT_ROOT, "fig_coverage_scatter.png")
    _plot_scatter(rows, out_scatter)
    print(f"[linecov] wrote {out_scatter}", flush=True)

    summary = {
        "line_info": line_info,
        "training_subset": TRAIN_SAMPLES,
        "K_train": K_TRAIN,
        "epochs": EPOCHS,
        "lambda_1d": LAM_1D,
        "gamma": GAMMA,
        "results": rows,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT_ROOT, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[linecov] wrote {os.path.join(OUT_ROOT, 'metrics_summary.json')}",
          flush=True)


if __name__ == "__main__":
    main()
