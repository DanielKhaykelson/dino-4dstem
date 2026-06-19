"""run_per_family_pipeline.py -- Two per-family models for the line-vs-spot
discrimination problem (replaces the failed combined-training approach).

Models:
    Model N (NaPHI):   train on NaPHI_Nadja_SI003 + SI004    K=6
    Model M (MgNaPHI): train on MgNaPHI_remeas_SI004 + SI011 K=6
    (locked recipe: DINO + cluster1d lambda=0.1 gamma=0.5 margin=0.4 30ep
     deterministic)

Per test-sample metrics (no cluster-membership counting -- avoids the
within-family-orientation-split problem we saw before):

    conf_NaPHI    = mean over patterns of max(softmax(logits_N))
    conf_MgNaPHI  = mean over patterns of max(softmax(logits_M))
    cosproto_NaPHI    = mean over patterns of max(cos(z, proto_N))   (no temp)
    cosproto_MgNaPHI  = mean over patterns of max(cos(z, proto_M))

A pattern that fits well in the NaPHI vocabulary scores high on conf_NaPHI
(peaked softmax + close prototype). A spot pattern that has no matching
prototype in Model N scores low (uniform softmax + far prototype).

Output: runs/_per_family/
    train_NaPHI/     train_MgNaPHI/    (the two models)
    per_sample/<sample>/eval/    (assignments + paper-output figures)
    metrics_summary.json          (all numbers)
    fig_2d_scatter.png            (2D conf_NaPHI vs conf_MgNaPHI per sample)
    fig_1d_likeness.png           (NaPHI-likeness = conf_NaPHI - conf_MgNaPHI)
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

OUT_ROOT = os.path.join("runs", "_per_family")

FAMILIES = {
    "NaPHI":   ["NaPHI_Nadja_SI003", "NaPHI_Nadja_SI004"],
    "MgNaPHI": ["MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI011"],
}
COMBINED_RADIAL_BASE = {
    "NaPHI":   r"D:\DINOSR\data\_perfam_NaPHI",
    "MgNaPHI": r"D:\DINOSR\data\_perfam_MgNaPHI",
}

K_TRAIN = 6
EPOCHS = 30
SEED = 42
LAM_1D = 0.1
GAMMA = 0.5
MARGIN = 0.4

# Test set
# (NaPHI SI-001 and SI-002 are excluded per user instruction -- SI-001 had
# "wrong FOV, weak signals", SI-002 was "too thick" per the README.)
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


# =========================================================================
# Helpers
# =========================================================================

def _radial_path_for(sample: str) -> str:
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    return base + ".radial.npy"


def _ensure_radials_for(sample: str, device):
    rad = _radial_path_for(sample)
    if os.path.exists(rad):
        return rad
    print(f"[perfam] computing radials for {sample}", flush=True)
    radials = compute_radial_for_sample(sample, device=device)
    np.save(rad, radials)
    return rad


def _build_combined_radials(family: str, samples: list, device):
    base = COMBINED_RADIAL_BASE[family]
    npy = base + ".radial.npy"
    th = base + ".gate_thresholds.json"
    if os.path.exists(npy) and os.path.exists(th):
        print(f"[perfam {family}] combined radials already exist", flush=True)
        return npy, th
    parts = [np.load(_ensure_radials_for(s, device)) for s in samples]
    combined = np.concatenate(parts, axis=0).astype(np.float32)
    np.save(npy, combined)
    th_d = calibrate_thresholds(combined, n_pairs=50_000,
                                 frac_pos=0.15, frac_neg=0.50)
    th_d["sample"] = f"PERFAM_{family}"
    with open(th, "w") as f:
        json.dump(th_d, f, indent=2)
    print(f"[perfam {family}] combined radials shape={combined.shape}  "
          f"tau_pos={th_d['tau_pos']:.4f}  tau_neg={th_d['tau_neg']:.4f}",
          flush=True)
    return npy, th


def _register_combined_sample(family: str, samples: list):
    key = f"PERFAM_{family}"
    SAMPLES[key] = {
        "paths": [SAMPLES[s]["path"] for s in samples],
        "vmax": 2,
        "scan_shape": (100, 100),  # nominal; not used at training
        "center_mask_radius": 15,
        "approved_label": None,
        "is_multi": True,
    }
    return key


def _train_family(family: str, samples: list, npy: str, th: str, device):
    train_dir = os.path.join(OUT_ROOT, f"train_{family}")
    if os.path.exists(os.path.join(train_dir, "best.pth")):
        print(f"[perfam {family}] train already done at {train_dir}", flush=True)
        return train_dir
    from run_contrastive import run_config
    os.makedirs(train_dir, exist_ok=True)
    sample_key = f"PERFAM_{family}"
    print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {family}  K={K_TRAIN}  "
          f"ep={EPOCHS}", flush=True)
    warmup = int(round((2.0 / 3.0) * EPOCHS))
    ramp = int(round((1.0 / 3.0) * EPOCHS))
    run_config("c", sample=sample_key, outdir=train_dir, device=device,
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
        supcon_radials_path=npy,
        supcon_thresholds_path=th,
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
    return train_dir


# =========================================================================
# Per-sample evaluation + likeness metrics
# =========================================================================

def _eval_through_model(model, sample, device):
    """Return:
        soft_probs   (N, K)
        assigns      (N,)
        max_softprob (N,)
        max_cos      (N,)  -- cosine to NEAREST prototype (raw, no temperature)
    """
    cfg = SAMPLES[sample]
    ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    inf = infer_scan(model, ds, device, dense_remap=False,
                      polar_size=192, polar_mask_cols=45,
                      center_crop_size=140,
                      com_centering=True, center_mask_radius=15,
                      eval_temp=0.06, batch_size=128)
    soft = inf["soft_probs"]
    embeds = inf["embeds"]
    P = model.prototypes.prototypes.detach().cpu().numpy()
    P = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)
    cos_to_proto = embeds @ P.T  # (N, K) -- embeds are already L2-normed
    return {
        "soft_probs": soft,
        "assigns": inf["assigns"],
        "embeds": embeds,
        "max_softprob": soft.max(axis=-1),
        "max_cos": cos_to_proto.max(axis=-1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[perfam] device={device}", flush=True)

    # Build radials + register multi-cube samples + train each family
    train_dirs = {}
    for family, samples in FAMILIES.items():
        npy, th = _build_combined_radials(family, samples, device)
        _register_combined_sample(family, samples)
        train_dirs[family] = _train_family(family, samples, npy, th, device)

    # Load both trained models
    print(f"\n[{datetime.now():%H:%M:%S}] loading both models...", flush=True)
    models = {}
    for family in FAMILIES:
        ckpt = os.path.join(train_dirs[family], "best.pth")
        m, _, _, _ = load_contrastive_checkpoint(ckpt, device=device)
        m.eval()
        models[family] = m

    # Eval each test sample through both models
    print(f"\n[{datetime.now():%H:%M:%S}] evaluating "
          f"{len(TARGETS)} test samples through both models...", flush=True)
    rows = []
    for sample, family_label in TARGETS:
        if sample not in SAMPLES or not os.path.exists(SAMPLES[sample]["path"]):
            print(f"  [SKIP] {sample}: missing", flush=True)
            continue
        try:
            r_N = _eval_through_model(models["NaPHI"], sample, device)
            r_M = _eval_through_model(models["MgNaPHI"], sample, device)
        except Exception as e:
            print(f"  [FAIL] {sample}: {e!r}", flush=True)
            continue
        N = len(r_N["max_softprob"])
        row = {
            "sample": sample,
            "family": family_label,
            "n_patterns": int(N),
            "conf_NaPHI":    float(r_N["max_softprob"].mean()),
            "conf_MgNaPHI":  float(r_M["max_softprob"].mean()),
            "cosproto_NaPHI":   float(r_N["max_cos"].mean()),
            "cosproto_MgNaPHI": float(r_M["max_cos"].mean()),
            "softprob_naphi_minus_mgnaphi":
                float(r_N["max_softprob"].mean() - r_M["max_softprob"].mean()),
            "cosproto_naphi_minus_mgnaphi":
                float(r_N["max_cos"].mean() - r_M["max_cos"].mean()),
        }
        rows.append(row)
        print(f"  {sample:<32} {family_label:<32} "
              f"conf_N={row['conf_NaPHI']:.3f} conf_M={row['conf_MgNaPHI']:.3f}  "
              f"cos_N={row['cosproto_NaPHI']:.3f} cos_M={row['cosproto_MgNaPHI']:.3f}",
              flush=True)

    # ---- 2D scatter: x=conf_NaPHI, y=conf_MgNaPHI ----
    fams = list({r["family"] for r in rows})
    fam_color = {f: plt.get_cmap("tab10").colors[i % 10] for i, f in enumerate(fams)}
    fig, ax = plt.subplots(figsize=(10, 8))
    for r in rows:
        ax.scatter(r["conf_NaPHI"], r["conf_MgNaPHI"],
                    s=140, color=fam_color[r["family"]],
                    edgecolors="black", linewidths=0.6, zorder=3)
        short = r["sample"].replace("_remeas_", "_").replace("_Nadja_", "_")
        ax.annotate(short, (r["conf_NaPHI"], r["conf_MgNaPHI"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=8)
    # diagonal y=x
    lo, hi = 0.1, 1.0
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, zorder=1)
    ax.set_xlabel("conf_NaPHI = mean max(softmax(logits)) under Model N")
    ax.set_ylabel("conf_MgNaPHI = mean max(softmax(logits)) under Model M")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                            markerfacecolor=fam_color[f],
                            markeredgecolor="black", markersize=10, label=f)
                for f in fams]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_title(f"Per-family confidence: train-N = {FAMILIES['NaPHI']}, "
                  f"train-M = {FAMILIES['MgNaPHI']}\n"
                  "Below diagonal: more NaPHI-like.  Above: more MgNaPHI-like.",
                  fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "fig_2d_scatter.png"), dpi=180,
                 bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 1D likeness ----
    rows_sorted = sorted(rows, key=lambda r: r["softprob_naphi_minus_mgnaphi"])
    fams_y = list({r["family"] for r in rows_sorted})
    fam_y = {f: i for i, f in enumerate(fams_y)}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 0.8 * len(fams_y) + 2))
    for r in rows_sorted:
        y = fam_y[r["family"]] + (rng.random() - 0.5) * 0.2
        ax.scatter([r["softprob_naphi_minus_mgnaphi"]], [y], s=140,
                    color=fam_color[r["family"]],
                    edgecolors="black", linewidths=0.6, zorder=3)
        short = r["sample"].replace("_remeas_", "_").replace("_Nadja_", "_")
        ax.annotate(short, (r["softprob_naphi_minus_mgnaphi"], y),
                     xytext=(6, 0), textcoords="offset points",
                     fontsize=8, va="center")
    ax.axvline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_yticks(range(len(fams_y)))
    ax.set_yticklabels(fams_y)
    ax.set_xlabel("NaPHI-likeness = conf_NaPHI - conf_MgNaPHI")
    ax.set_title("NaPHI-likeness across NaPHI and MgNaPHI samples")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "fig_1d_likeness.png"), dpi=180,
                 bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # JSON summary
    summary = {
        "families": FAMILIES,
        "K_train": K_TRAIN,
        "epochs": EPOCHS,
        "lambda_1d": LAM_1D,
        "gamma": GAMMA,
        "results": rows,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT_ROOT, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[perfam] wrote outputs to {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
