"""compute_line_coverage_perfam.py -- POST-process for run_per_family_pipeline.

Once the two per-family models exist at runs/_per_family/train_{NaPHI,MgNaPHI}/best.pth,
this script:

    1. Calibrates a "is-this-pattern-line-like" threshold from the NaPHI
        training distribution: pick the 10th percentile of max_cos under
        Model N evaluated on its own training data. This means ~90% of
        training NaPHI patterns ARE counted as line.

    2. For each test sample, evaluates through Model N once and computes
        line_coverage = fraction of patterns with max_cos > threshold.

    3. Plots: a 1D scatter of line_coverage per sample, color-coded by
        family. The SI-007 trio should land tightly together; bulk MgNaPHI
        should be near zero; bulk NaPHI should be near one.

Output: runs/_per_family/
    fig_line_coverage.png
    line_coverage_summary.json
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from data import SAMPLES, LoadPRZ, LoadPRZMulti
from dino_sr_contrastive_model import load_contrastive_checkpoint
from contrastive_eval import infer_scan

OUT_ROOT = os.path.join("runs", "_per_family")
TRAIN_NAPHI_DIR = os.path.join(OUT_ROOT, "train_NaPHI")
TRAIN_MgNaPHI_DIR = os.path.join(OUT_ROOT, "train_MgNaPHI")

NaPHI_TRAIN_SAMPLES = ["NaPHI_Nadja_SI003", "NaPHI_Nadja_SI004"]
MgNaPHI_TRAIN_SAMPLES = ["MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI011"]

# Calibration: at what percentile of NaPHI-training max_cos do we set the
# threshold? 10 means 90% of training NaPHI patterns count as line.
CALIB_PCT = 10.0

TARGETS = [
    ("NaPHI_Nadja_SI003", "NaPHI bulk (train-N)"),
    ("NaPHI_Nadja_SI004", "NaPHI bulk (train-N)"),
    ("NaPHI_Nadja_SI009", "NaPHI bulk"),
    ("NaPHI_Nadja_SI010", "NaPHI bulk"),
    ("NaPHI_Nadja_SI005", "NaPHI 4Q"),
    ("NaPHI_Nadja_SI006", "NaPHI 4Q"),
    ("NaPHI_Nadja_SI007", "NaPHI 4Q"),
    ("NaPHI_Nadja_SI008", "NaPHI 4Q"),
    ("MgNaPHI_remeas_SI001", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI003", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI004", "MgNaPHI bulk (train-M)"),
    ("MgNaPHI_remeas_SI005", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI006", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI010", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI011", "MgNaPHI bulk (train-M)"),
    ("MgNaPHI_remeas_SI007", "MgNaPHI SI-007 (outlier)"),
    ("MgNaPHI_remeas_SI008", "MgNaPHI SI-008 (=SI-007 area)"),
    ("MgNaPHI_remeas_SI009", "MgNaPHI SI-009 (=SI-007 area)"),
]


def _max_cos_proto(model, dataset, device):
    inf = infer_scan(model, dataset, device, dense_remap=False,
                      polar_size=192, polar_mask_cols=45,
                      center_crop_size=140,
                      com_centering=True, center_mask_radius=15,
                      eval_temp=0.06, batch_size=128)
    embeds = inf["embeds"]
    P = model.prototypes.prototypes.detach().cpu().numpy()
    P = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)
    cos_to_proto = embeds @ P.T
    return cos_to_proto.max(axis=-1)


def _max_softprob(model, dataset, device):
    inf = infer_scan(model, dataset, device, dense_remap=False,
                      polar_size=192, polar_mask_cols=45,
                      center_crop_size=140,
                      com_centering=True, center_mask_radius=15,
                      eval_temp=0.06, batch_size=128)
    return inf["soft_probs"].max(axis=-1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[line-coverage] device={device}", flush=True)

    # Load both trained per-family models
    print("[line-coverage] loading models...", flush=True)
    naphi_model, _, _, _ = load_contrastive_checkpoint(
        os.path.join(TRAIN_NAPHI_DIR, "best.pth"), device=device)
    naphi_model.eval()
    mgnaphi_model, _, _, _ = load_contrastive_checkpoint(
        os.path.join(TRAIN_MgNaPHI_DIR, "best.pth"), device=device)
    mgnaphi_model.eval()

    # ---- Calibrate threshold from NaPHI training distribution ----
    print(f"[line-coverage] calibrating threshold "
          f"from NaPHI training (pct={CALIB_PCT})...", flush=True)
    train_paths = [SAMPLES[s]["path"] for s in NaPHI_TRAIN_SAMPLES]
    train_ds = LoadPRZMulti(train_paths, resize=192, vmax=2)
    train_max_cos = _max_cos_proto(naphi_model, train_ds, device)
    train_max_softprob = _max_softprob(naphi_model, train_ds, device)
    cos_thresh = float(np.percentile(train_max_cos, CALIB_PCT))
    softprob_thresh = float(np.percentile(train_max_softprob, CALIB_PCT))
    print(f"  cos threshold     = {cos_thresh:.4f}  "
          f"(median train cos = {np.median(train_max_cos):.4f})", flush=True)
    print(f"  softprob threshold= {softprob_thresh:.4f}  "
          f"(median train sp = {np.median(train_max_softprob):.4f})",
          flush=True)

    # Also compute the NEGATIVE control: max_cos under Model N for MgNaPHI
    # training data. This tells us how far MgNaPHI patterns score under the
    # NaPHI vocabulary.
    mg_train_paths = [SAMPLES[s]["path"] for s in MgNaPHI_TRAIN_SAMPLES]
    mg_train_ds = LoadPRZMulti(mg_train_paths, resize=192, vmax=2)
    mg_max_cos = _max_cos_proto(naphi_model, mg_train_ds, device)
    print(f"  (sanity) MgNaPHI training under Model N: "
          f"median cos = {np.median(mg_max_cos):.4f}, "
          f"frac > thresh = {(mg_max_cos > cos_thresh).mean():.3f}",
          flush=True)

    # ---- Per-sample line coverage ----
    print(f"\n[line-coverage] evaluating {len(TARGETS)} samples...", flush=True)
    rows = []
    for sample, family in TARGETS:
        if sample not in SAMPLES or not os.path.exists(SAMPLES[sample]["path"]):
            print(f"  [SKIP] {sample}: missing", flush=True)
            continue
        try:
            cfg = SAMPLES[sample]
            ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
            mc = _max_cos_proto(naphi_model, ds, device)
            mp = _max_softprob(naphi_model, ds, device)
            cov_cos = float((mc > cos_thresh).mean())
            cov_sp = float((mp > softprob_thresh).mean())
            row = {
                "sample": sample, "family": family,
                "n_patterns": int(len(mc)),
                "line_coverage_cos": cov_cos,
                "line_coverage_softprob": cov_sp,
                "median_max_cos": float(np.median(mc)),
                "median_max_softprob": float(np.median(mp)),
            }
            rows.append(row)
            print(f"  {sample:<32} {family:<32}  "
                  f"cov(cos)={cov_cos:.3f}  cov(sp)={cov_sp:.3f}  "
                  f"med_cos={np.median(mc):.3f}", flush=True)
        except Exception as e:
            print(f"  [FAIL] {sample}: {e!r}", flush=True)
            import traceback; traceback.print_exc()

    # ---- Plot scatter (cos-coverage primary) ----
    fams = list({r["family"] for r in rows})
    fam_color = {f: plt.get_cmap("tab10").colors[i % 10] for i, f in enumerate(fams)}
    rows_sorted = sorted(rows, key=lambda r: r["line_coverage_cos"])
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 0.8 * len(fams) + 2))
    fam_y = {f: i for i, f in enumerate(fams)}
    for r in rows_sorted:
        y = fam_y[r["family"]] + (rng.random() - 0.5) * 0.2
        ax.scatter([r["line_coverage_cos"]], [y], s=140,
                    color=fam_color[r["family"]],
                    edgecolors="black", linewidths=0.6, zorder=3)
        short = r["sample"].replace("_remeas_", "_").replace("_Nadja_", "_")
        ax.annotate(short, (r["line_coverage_cos"], y),
                     xytext=(6, 0), textcoords="offset points",
                     fontsize=8, va="center")
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("line coverage = fraction of patterns with "
                  f"max cos(z, NaPHI proto) > {cos_thresh:.3f}\n"
                  f"(threshold = {CALIB_PCT}th percentile of NaPHI training)")
    ax.set_title("Line-phase coverage (Model N: NaPHI-only) — sample\n"
                  "SI-007 / SI-008 / SI-009 are remeasures of the same area "
                  "and should cluster")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_ROOT, "fig_line_coverage.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[line-coverage] wrote {out}", flush=True)

    summary = {
        "naphi_train_samples": NaPHI_TRAIN_SAMPLES,
        "mgnaphi_train_samples": MgNaPHI_TRAIN_SAMPLES,
        "calib_percentile": CALIB_PCT,
        "cos_threshold": cos_thresh,
        "softprob_threshold": softprob_thresh,
        "naphi_train_median_max_cos": float(np.median(train_max_cos)),
        "mgnaphi_train_median_max_cos_under_N": float(np.median(mg_max_cos)),
        "results": rows,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT_ROOT, "line_coverage_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[line-coverage] wrote line_coverage_summary.json", flush=True)


if __name__ == "__main__":
    main()
