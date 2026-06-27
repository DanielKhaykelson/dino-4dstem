"""run_per_family_v5_pipeline.py -- end-to-end per-family pipeline at
vmax=5, K=8.

Steps:
    1. Compute per-sample radials at vmax=5 (saved with v5 suffix to
       avoid clobbering the v2 cache).
    2. Concatenate per family, calibrate gate thresholds.
    3. Train Model N (NaPHI SI-003+004) and Model M (MgNaPHI SI-004+011)
       at vmax=5, K=8, 30 ep, deterministic, DINO + cluster1d
       (lambda=0.1, gamma=0.5, margin=0.4).
    4. Render class averages + 200 examples per class for both source
       models on their training data.
    5. Eval each model on its OWN family's test samples at vmax=5;
       save inference.npz, class map, class averages, 200 examples
       per class per test sample.

Output dir: runs/_per_family_v5/{NaPHI,MgNaPHI}_combined_K8_30ep/
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys, time, json
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
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

from data import SAMPLES, LoadPRZ, LoadPRZMulti
from dino_sr_contrastive_model import load_contrastive_checkpoint
from contrastive_eval import infer_scan
from compute_radial_profile import (
    _build_pre as _build_polar_pre, calibrate_thresholds,
    Q_LOW, Q_HIGH, EPS, POLY_ORDER, POLAR_SIZE, POLAR_MASK_COLS, CENTER_CROP,
)


# =========================================================================
# Configuration
# =========================================================================

VMAX = 5
K = 8
EPOCHS = 30
SEED = 42
LAM_1D = 0.1
GAMMA = 0.5
MARGIN = 0.4
RADIAL_TAG = f"v{VMAX}"

OUT_ROOT = os.path.join("runs", "_per_family_v5")
SRC_DIR = {
    "NaPHI":   os.path.join(OUT_ROOT, "NaPHI_combined_K8_30ep"),
    "MgNaPHI": os.path.join(OUT_ROOT, "MgNaPHI_combined_K8_30ep"),
}

FAMILIES = {
    "NaPHI":   ["NaPHI_Nadja_SI003",   "NaPHI_Nadja_SI004"],
    "MgNaPHI": ["MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI011"],
}
TEST_SETS = {
    "NaPHI":   ["NaPHI_Nadja_SI003", "NaPHI_Nadja_SI004",
                 "NaPHI_Nadja_SI009", "NaPHI_Nadja_SI010",
                 "NaPHI_Nadja_SI005", "NaPHI_Nadja_SI006",
                 "NaPHI_Nadja_SI007", "NaPHI_Nadja_SI008"],
    "MgNaPHI": ["MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI011",
                 "MgNaPHI_remeas_SI001", "MgNaPHI_remeas_SI003",
                 "MgNaPHI_remeas_SI005", "MgNaPHI_remeas_SI006",
                 "MgNaPHI_remeas_SI010", "MgNaPHI_remeas_SI007",
                 "MgNaPHI_remeas_SI008", "MgNaPHI_remeas_SI009"],
}

COMBINED_RADIAL_BASE = {
    "NaPHI":   rf"D:\DINOSR\data\_perfam_NaPHI_v{VMAX}_K{K}",
    "MgNaPHI": rf"D:\DINOSR\data\_perfam_MgNaPHI_v{VMAX}_K{K}",
}


# =========================================================================
# Per-sample radials at vmax=VMAX
# =========================================================================

def _radial_path_for(sample):
    cfg = SAMPLES[sample]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    return base + f".radial_{RADIAL_TAG}.npy"


def _compute_one_sample_radial(sample, device):
    """Like compute_radial_profile.compute_radial but with explicit vmax=VMAX."""
    rad_path = _radial_path_for(sample)
    if os.path.exists(rad_path):
        return rad_path
    cfg = SAMPLES[sample]
    print(f"[radial v{VMAX}] {sample}: computing ...", flush=True)
    ds = LoadPRZ(cfg["path"], resize=POLAR_SIZE, vmax=VMAX)
    pre = _build_polar_pre(POLAR_SIZE, POLAR_MASK_COLS, CENTER_CROP)
    N = len(ds)
    n_bins = POLAR_SIZE
    radials = np.zeros((N, n_bins), dtype=np.float32)
    batch = 256
    with torch.no_grad():
        for i in range(0, N, batch):
            j = min(i + batch, N)
            xs = torch.stack([ds[k] for k in range(i, j)]).to(device).float()
            xs = pre(xs)
            prof = xs.sum(dim=(1, 2))
            radials[i:j] = prof.cpu().numpy()
    # SAXS treatment: keep mid-q, log, polynomial baseline subtraction
    keep = radials[:, Q_LOW:Q_HIGH]
    sums = keep.sum(axis=1, keepdims=True); sums[sums < EPS] = EPS
    norm = keep / sums
    log_I = np.log(norm + EPS)
    n_keep = keep.shape[1]
    q = np.arange(n_keep, dtype=np.float64)
    residuals = np.zeros_like(log_I, dtype=np.float32)
    for i in range(log_I.shape[0]):
        y = log_I[i].astype(np.float64)
        coef = np.polyfit(q, y, POLY_ORDER); bg = np.polyval(coef, q)
        r = y - bg
        med = np.median(r); mad = np.median(np.abs(r - med)) + 1e-12
        peak_mask = (r - med) > 2.0 * mad
        if peak_mask.sum() < n_keep - POLY_ORDER - 1:
            non_peak = ~peak_mask
            coef = np.polyfit(q[non_peak], y[non_peak], POLY_ORDER)
            bg = np.polyval(coef, q)
        r2 = y - bg
        residuals[i] = (r2 - r2.mean()).astype(np.float32)
    np.save(rad_path, residuals)
    print(f"  saved {rad_path}  shape={residuals.shape}", flush=True)
    return rad_path


def _build_combined_radials(family, samples, device):
    base = COMBINED_RADIAL_BASE[family]
    npy = base + ".radial.npy"
    th = base + ".gate_thresholds.json"
    if os.path.exists(npy) and os.path.exists(th):
        print(f"[radial v{VMAX} {family}] combined exists", flush=True)
        return npy, th
    parts = []
    for s in samples:
        p = _compute_one_sample_radial(s, device)
        parts.append(np.load(p))
    combined = np.concatenate(parts, axis=0).astype(np.float32)
    np.save(npy, combined)
    th_d = calibrate_thresholds(combined, n_pairs=50_000,
                                 frac_pos=0.15, frac_neg=0.50)
    th_d["sample"] = f"PERFAM_{family}_v{VMAX}"
    with open(th, "w") as f:
        json.dump(th_d, f, indent=2)
    print(f"[radial v{VMAX} {family}] tau_pos={th_d['tau_pos']:.4f} "
          f"tau_neg={th_d['tau_neg']:.4f}", flush=True)
    return npy, th


# =========================================================================
# Training
# =========================================================================

def _register_combined_sample(family, samples):
    SAMPLES[f"PERFAM_{family}_V{VMAX}"] = {
        "paths": [SAMPLES[s]["path"] for s in samples],
        "vmax": VMAX,
        "scan_shape": (100, 100),
        "center_mask_radius": 15,
        "approved_label": None,
        "is_multi": True,
    }


def _train_family(family, samples, npy, th, device):
    train_dir = SRC_DIR[family]
    if os.path.exists(os.path.join(train_dir, "best.pth")):
        print(f"[train {family}] best.pth exists, skipping", flush=True)
        return train_dir
    from run_contrastive import run_config
    os.makedirs(train_dir, exist_ok=True)
    sample_key = f"PERFAM_{family}_V{VMAX}"
    print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {family}  K={K}  ep={EPOCHS}  "
          f"vmax={VMAX}", flush=True)
    warmup = int(round((2.0 / 3.0) * EPOCHS))
    ramp = int(round((1.0 / 3.0) * EPOCHS))
    run_config("c", sample=sample_key, outdir=train_dir, device=device,
        epochs=EPOCHS, seed=SEED, batch_size=128,
        lr=3e-4, weight_decay=1e-6,
        num_prototypes=K,
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
# Class avg + 200 examples per class (works for multi-cube training data)
# =========================================================================

H = 192
MASK_R = 15
CENTER_CROP_DISP = 140


def _render_class_outputs(soft_probs, assigns, ds, eval_dir, sample_label,
                            n_examples=200):
    cart_pre = T.Compose([
        T.CenterCrop(CENTER_CROP_DISP),
        T.Resize(H, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])
    cy = cx = H / 2.0
    yy, xx = np.ogrid[:H, :H]
    bm = ((yy - cy) ** 2 + (xx - cx) ** 2) > MASK_R ** 2
    avg_dir = os.path.join(eval_dir, "class_averages")
    ex_root = os.path.join(eval_dir, "class_examples_200")
    os.makedirs(avg_dir, exist_ok=True)
    os.makedirs(ex_root, exist_ok=True)
    K_local = soft_probs.shape[1]
    counts = np.bincount(assigns, minlength=K_local)
    for c in range(K_local):
        idx = np.where(assigns == c)[0]
        if idx.size == 0:
            print(f"  p{c}: empty", flush=True); continue
        scores = soft_probs[idx, c]
        # ----- avg from top-300 -----
        top_avg = idx[np.argsort(-scores)[:min(300, len(idx))]]
        patterns = np.stack([ds.get_raw(int(i)) for i in top_avg], 0).astype(np.float32)
        w = soft_probs[top_avg, c].astype(np.float32)
        wavg = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
        wavg_norm = np.clip(wavg / float(VMAX), 0.0, 1.0)
        x = torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0).float()
        x = F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)
        x_cart = cart_pre(x)[0, 0].cpu().numpy()
        ref = (x_cart * bm).flatten(); ref = ref[ref > 0]
        if ref.size:
            lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
            disp = np.log1p(np.clip(x_cart, lo, hi) - lo) * bm
        else:
            disp = x_cart * bm
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
        ax.set_title(f"{sample_label}  p{c}  N={int(counts[c])}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(avg_dir, f"p{c}.png"), dpi=200,
                     bbox_inches="tight", facecolor="white")
        plt.close(fig)
        # ----- 200 examples -----
        n_ex = min(n_examples, len(idx))
        top_ex = idx[np.argsort(-scores)[:n_ex]]
        d = os.path.join(ex_root, f"p{c}")
        os.makedirs(d, exist_ok=True)
        for old in os.listdir(d):
            try: os.remove(os.path.join(d, old))
            except Exception: pass
        for rank, gi in enumerate(top_ex):
            gi = int(gi)
            raw = ds.get_raw(gi).astype(np.float32)
            raw_norm = np.clip(raw / float(VMAX), 0.0, 1.0)
            x = torch.from_numpy(raw_norm).unsqueeze(0).unsqueeze(0).float()
            x = F.interpolate(x, size=(H, H), mode="bilinear", align_corners=False)
            x_cart = cart_pre(x)[0, 0].cpu().numpy()
            ref = (x_cart * bm).flatten(); ref = ref[ref > 0]
            if ref.size:
                lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
                disp = np.log1p(np.clip(x_cart, lo, hi) - lo) * bm
            else:
                disp = x_cart * bm
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
            ax.set_title(f"p{c}  i={gi}  p={float(soft_probs[gi, c]):.2f}",
                          fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(os.path.join(d, f"rank{rank:03d}_i{gi:06d}_p{float(soft_probs[gi, c]):.2f}.png"),
                         dpi=130, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        print(f"  p{c}: N={counts[c]}  avg + {n_ex} examples", flush=True)


# =========================================================================
# Per-family inference on training data + class avgs/examples
# =========================================================================

def _eval_training(family, train_samples, train_dir, device):
    eval_dir = os.path.join(train_dir, "eval")
    if os.path.exists(os.path.join(eval_dir, "inference.npz")):
        print(f"[eval-train {family}] already done", flush=True)
        return
    os.makedirs(eval_dir, exist_ok=True)
    model, _, _, _ = load_contrastive_checkpoint(
        os.path.join(train_dir, "best.pth"), device=device)
    model.eval()
    paths = [SAMPLES[s]["path"] for s in train_samples]
    ds = LoadPRZMulti(paths, resize=H, vmax=VMAX)
    print(f"[eval-train {family}] N={len(ds)}", flush=True)
    inf = infer_scan(model, ds, device, dense_remap=False,
                      polar_size=192, polar_mask_cols=45,
                      center_crop_size=140,
                      com_centering=True, center_mask_radius=15,
                      eval_temp=0.06, batch_size=128)
    np.savez(os.path.join(eval_dir, "inference.npz"),
              soft_probs=inf["soft_probs"], assigns=inf["assigns"],
              embeds=inf["embeds"])
    with open(os.path.join(train_dir, "run_summary.json"), "w") as f:
        json.dump({"cfg": {
            "center_mask_radius": 15, "polar_mask_cols": 45,
            "polar_size": 192, "center_crop_size": 140,
            "vmax": VMAX, "K": K,
        }}, f)
    print(f"[eval-train {family}] rendering class avgs + 200 examples...",
          flush=True)
    _render_class_outputs(inf["soft_probs"], inf["assigns"], ds, eval_dir,
                            f"{family}_train", n_examples=200)


# =========================================================================
# Per-test-sample inference
# =========================================================================

def _eval_test_samples(family, train_dir, device):
    model, _, _, _ = load_contrastive_checkpoint(
        os.path.join(train_dir, "best.pth"), device=device)
    model.eval()
    print(f"\n[eval-test {family}] running on {len(TEST_SETS[family])} samples",
          flush=True)
    for sample in TEST_SETS[family]:
        if sample not in SAMPLES or not os.path.exists(SAMPLES[sample]["path"]):
            print(f"  [SKIP] {sample}", flush=True); continue
        transfer_root = os.path.join(train_dir, "transfer", sample)
        eval_dir = os.path.join(transfer_root, "eval")
        if os.path.exists(os.path.join(eval_dir, "class_examples_200")):
            print(f"  [skip] {sample} already done", flush=True)
            continue
        os.makedirs(eval_dir, exist_ok=True)
        cfg = SAMPLES[sample]
        ds = LoadPRZ(cfg["path"], resize=H, vmax=VMAX)
        inf = infer_scan(model, ds, device, dense_remap=False,
                          polar_size=192, polar_mask_cols=45,
                          center_crop_size=140,
                          com_centering=True, center_mask_radius=15,
                          eval_temp=0.06, batch_size=128)
        np.savez(os.path.join(eval_dir, "inference.npz"),
                  soft_probs=inf["soft_probs"], assigns=inf["assigns"],
                  embeds=inf["embeds"])
        with open(os.path.join(transfer_root, "run_summary.json"), "w") as f:
            json.dump({"cfg": {
                "center_mask_radius": 15, "polar_mask_cols": 45,
                "polar_size": 192, "center_crop_size": 140,
                "vmax": VMAX, "K": K,
            }}, f)
        print(f"  {sample}: rendering class avgs + 200 examples...",
              flush=True)
        _render_class_outputs(inf["soft_probs"], inf["assigns"], ds, eval_dir,
                                sample, n_examples=200)


# =========================================================================
# Main
# =========================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[per-family v{VMAX}] device={device}  K={K}  EPOCHS={EPOCHS}",
          flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)
    t_total = time.perf_counter()

    # 1. radials
    fam_radials = {}
    for fam, samples in FAMILIES.items():
        npy, th = _build_combined_radials(fam, samples, device)
        fam_radials[fam] = (npy, th)
        _register_combined_sample(fam, samples)

    # 2. train
    for fam, samples in FAMILIES.items():
        npy, th = fam_radials[fam]
        _train_family(fam, samples, npy, th, device)

    # 3. eval on training data + class avgs + 200 examples
    for fam, samples in FAMILIES.items():
        _eval_training(fam, samples, SRC_DIR[fam], device)

    # 4. eval on each family's test samples
    for fam in FAMILIES:
        _eval_test_samples(fam, SRC_DIR[fam], device)

    print(f"\n[done] in {(time.perf_counter() - t_total) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
