"""sweep_m_K.py — long-running background sweep over center-momentum m
and prototype count K, executed sample-by-sample.

Goal: identify a value of `m` (and a confidence band) such that K
emerges as a data-driven quantity — i.e. K_eff is stable across K
when m = m*.

Protocol (per sample, in this order):
  Stage 1   K = 30, m x seed grid  -> identify top-3 m candidates
  Stage 2   top-3 m x K grid x 1 seed -> confirm K_eff plateau
  Reports   written AS SOON AS A STAGE FINISHES (not at the end of
            the whole sweep) — user's explicit request.

Output layout
-------------
runs/_sweep_m_K_<ts>/
    SWEEP_SPEC.json                 frozen spec at launch
    SWEEP_PROGRESS.csv              rolling per-run log (append-on-done)
    STOP_SWEEP                      ← touch this file to abort gracefully
    <sample>/
        sanity_frame.png            pre-flight frame check
        SAMPLE_LOCK.json            locked pre-processing kwargs
        stage1/
            m{m:.4f}_seed{s}_K{K}/  ← standard run_contrastive outdir
            ...
        stage2/
            m{m:.4f}_seed{s}_K{K}/
            ...
        report_stage1.html          ← written after Stage 1 completes
        report.html                 ← final, written after Stage 2
        figures/
            fig_stage1_K_eff_vs_m.png
            fig_stage1_avg_conf_e5.png
            fig_stage2_K_eff_vs_K.png
            fig_class_maps_stage1.png
            fig_class_avgs_stage1.png
            ...
        tables/
            stage1.csv
            stage2.csv

Invocation
----------
    python tools/sweep_m_K.py                # run all 3 samples
    python tools/sweep_m_K.py --sample Na007b   # just one
    python tools/sweep_m_K.py --dry-run      # 1-epoch sanity, no real sweep

Determinism: ``run_contrastive`` already sets the deterministic env
vars; seeds are passed through verbatim.  All other knobs come from
PAPER_DEFAULTS so the only varying axes are (m, K, seed, sample).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime

# Windows console defaults to cp1252, which can't encode common Unicode
# arrows / multiplication signs.  Force UTF-8 so our progress prints
# don't crash the sweep mid-run.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

# Make repo modules importable when launched directly.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch                                      # noqa: E402
from data import SAMPLES, register_runtime_sample # noqa: E402
from run_contrastive import (                     # noqa: E402
    run_config, evaluate_and_report,
)
# `run_config` only TRAINS; the eval step (inference over the full
# scan, embeddings + soft_probs + assigns -> eval/inference.npz)
# lives in evaluate_and_report.  The sweep calls both — see _launch_one.
# PAPER_DEFAULTS lives in the GUI training panel; importing it does
# not bring up tkinter (the dict is at module top).  We re-use it so
# the sweep launches with EXACTLY the same hyperparameter envelope as
# the user's validated GUI runs — only (m, K, seed) and per-sample
# pre-processing vary.
from gui_app.train_panel import PAPER_DEFAULTS    # noqa: E402


# =========================================================================
# 1.  Spec
# =========================================================================
SWEEP_SPEC: dict = {
    "m_grid":            [0.5, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999],
    "K_stage1":          30,
    "seeds_stage1":      [42, 7],
    "K_grid_stage2":     [10, 15, 30, 60, 120],
    "seed_stage2":       42,
    "epochs":            30,
    "top_m_picks":       3,
    "sample_order":      ["Na007b", "EuInAs_B100", "IMC_SI5"],
    "K_true_band":       {                         # used by the report
        "Na007b":      [6, 6],
        "EuInAs_B100": [4, 6],
        "IMC_SI5":     [10, 25],                   # broad — unknown
    },
}

# Per-sample pre-processing (from the user spec).  ``com_centering=False``
# everywhere, ``polar_mask_cols`` and ``center_crop_size`` per sample.
# vmax also per sample.  These are LOCKED — every run in the sweep
# uses these exact values so the only varying axes are (m, K, seed).
SAMPLE_SPECS: dict = {
    "Na007b": {
        "cube_path":         "D:/DINOSR/data/Na007b_nbed.cube.npy",
        "vmax":              5.0,
        "center_crop_size":  120,
        "polar_mask_cols":   20,
        "com_centering":     False,
    },
    "EuInAs_B100": {
        "cube_path":         "D:/DINOSR/data/EuInAs_B100.cube.npy",
        "vmax":              30.0,
        "center_crop_size":  120,
        "polar_mask_cols":   0,
        "com_centering":     False,
    },
    "IMC_SI5": {
        "cube_path":         "D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
        "vmax":              5.0,
        "center_crop_size":  120,
        "polar_mask_cols":   40,
        "com_centering":     False,
    },
}


# =========================================================================
# 2.  Sample registration & sanity check
# =========================================================================
def _register_sample(sample_key: str) -> tuple[str, dict]:
    """Register the cube as a SAMPLES entry under the canonical key
    (`Na007b`, `EuInAs_B100`, `IMC_SI5`) so ``run_config`` can address
    it.  Stamps the locked pre-processing values too.
    """
    spec = SAMPLE_SPECS[sample_key]
    p = spec["cube_path"]
    if not os.path.exists(p):
        raise RuntimeError(f"cube not found: {p}")
    # Use the canonical sample_key (no runtime prefix) so any
    # checkpoint that references it by name keeps working.
    cube = np.load(p, mmap_mode="r", allow_pickle=True)
    Ny, Nx = cube.shape[:2]
    # center_mask_radius: cart-space CenterMask is REDUNDANT with
    # polar_mask_cols (both mask the low-r region; only the former
    # before the polar transform, only the latter after).  Since the
    # user's spec gives polar_mask_cols per sample and com_centering
    # is OFF (so the COM-search-radius coupling doesn't matter), we
    # set center_mask_radius = polar_mask_cols // 2 EXACTLY — no
    # floor — so polar_mask_cols=0 (EuInAs) -> center_mask_radius=0,
    # meaning the CenterMask step is skipped entirely.  This matches
    # the user's "polar mask r 0 = no central masking" intent.
    derived_cmr = int(spec["polar_mask_cols"]) // 2
    SAMPLES[sample_key] = {
        "path":                 os.path.abspath(p),
        "scan_shape":           (int(Ny), int(Nx)),
        "vmax":                 float(spec["vmax"]),
        "center_mask_radius":   int(derived_cmr),
        "blur_sigma":           0.0,
        "log_stretch":          False,
        "ellipticity_ab":       1.0,
        "ellipticity_theta_deg": 0.0,
        "approved_label":       None,
        "_runtime":             True,
    }
    print(f"[register] {sample_key}: scan {Ny}x{Nx}, "
          f"vmax={spec['vmax']}, crop={spec['center_crop_size']}, "
          f"polar_mask_cols={spec['polar_mask_cols']}, "
          f"com={spec['com_centering']}", flush=True)
    return p, spec


def _ensure_radials(sample_key: str) -> tuple[str, str]:
    """Compute (and cache) the 1D radial profiles + gate-threshold
    calibration for this sample.  These are REQUIRED for the
    cluster1d_lambda_intra / cluster1d_lambda_inter terms to fire;
    without them ``run_contrastive`` silently zeros the cluster1d
    loss (``lambda_cluster1d_eff = -1``), which is what produced the
    first 14 runs of vanilla-DINO-only training (the user spotted
    this immediately).  Mirrors the GUI's auto-compute step.

    Convention (same as the GUI): files live next to the cube as
        ``<cube_path>.radial.npy``
        ``<cube_path>.gate_thresholds.json``
    """
    import json as _json
    from compute_radial_profile import (
        compute_radial as _compute_radial,
        calibrate_thresholds as _calibrate_thresholds,
    )
    cube_path = SAMPLES[sample_key]["path"]
    rad_path = cube_path + ".radial.npy"
    th_path  = cube_path + ".gate_thresholds.json"
    if not os.path.exists(rad_path):
        print(f"[radials] {sample_key}: computing 1D radial profiles "
              f"(this is a one-shot cost; cached at {rad_path}) ...",
              flush=True)
        t0 = time.time()
        rad = _compute_radial(sample_key)
        np.save(rad_path, rad)
        print(f"[radials] {sample_key}: radial profile written "
              f"({time.time() - t0:.1f}s, shape={rad.shape}).",
              flush=True)
    else:
        rad = np.load(rad_path)
        print(f"[radials] {sample_key}: re-using cached radials "
              f"({rad_path}, shape={rad.shape}).", flush=True)
    if not os.path.exists(th_path):
        print(f"[radials] {sample_key}: calibrating gate thresholds "
              f"...", flush=True)
        t0 = time.time()
        th = _calibrate_thresholds(rad, n_pairs=50_000,
                                       frac_pos=0.15, frac_neg=0.50)
        th["sample"] = sample_key
        with open(th_path, "w", encoding="utf-8") as f:
            _json.dump(th, f, indent=2)
        print(f"[radials] {sample_key}: thresholds written "
              f"({time.time() - t0:.1f}s).", flush=True)
    return rad_path, th_path


def _sanity_render(sample_key: str, out_png: str):
    """Render a 1x3 panel showing the same frame at three pipeline
    stages so the user can sanity-check what the model actually sees:

      [raw frame]   [cart-processed, 192x192]   [polar input, mask applied]

    The polar pane is the EXACT input the encoder receives (after
    CenterCrop -> Resize -> optional CenterMask -> PolarTransform ->
    PolarMaskLeft).  Anything weird visible there will be in every
    sample of the run.
    """
    from data import LoadPRZ
    from dino_sr_contrastive_model import PolarTransform, PolarMaskLeft
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import cv2

    spec = SAMPLE_SPECS[sample_key]
    ds = LoadPRZ(spec["cube_path"], resize=192, vmax=spec["vmax"])
    idx = len(ds) // 2
    # Raw frame: ds.get_raw bypasses the 192-resize and filters so
    # we can show the detector-resolution image.
    raw = ds.get_raw(idx).astype(np.float32)
    # Cart-processed: ds[idx] gives the (1, 192, 192) tensor the
    # cart pipeline produces (after vmax norm, resize, blur/log, etc.).
    cart = ds[idx][0].cpu().numpy()
    # Polar pane: apply CenterCrop -> Resize -> CenterMask (if >0)
    # -> PolarTransform -> PolarMaskLeft, exactly as the training
    # transforms do (see dino_sr_contrastive_model.get_contrastive_transforms).
    crop = int(spec["center_crop_size"])
    pmc  = int(spec["polar_mask_cols"])
    cmr  = pmc // 2                       # derived in _register_sample
    # CenterCrop on 192 -> Resize back to 192:
    y0 = max(0, (192 - crop) // 2); x0 = y0
    cropped = cart[y0:y0 + crop, x0:x0 + crop]
    cart_resized = cv2.resize(cropped, (192, 192),
                                 interpolation=cv2.INTER_LINEAR)
    cart_for_polar = cart_resized.copy()
    if cmr > 0:
        # CenterMask: zero a disk of radius cmr at the centre.
        yy, xx = np.ogrid[:192, :192]
        cy = cx = 95.5
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) > cmr ** 2
        cart_for_polar = cart_for_polar * mask
    import torch as _torch
    tt = _torch.from_numpy(cart_for_polar).unsqueeze(0).unsqueeze(0).float()
    tt = PolarTransform(output_size=192)(tt)
    if pmc > 0:
        tt = PolarMaskLeft(k_cols=pmc)(tt)
    polar = tt[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), dpi=130)
    axes[0].imshow(raw, cmap="inferno",
                      vmax=float(spec["vmax"]))
    axes[0].set_title(f"raw frame   #{idx}\n{raw.shape[0]}x{raw.shape[1]} px,  "
                         f"vmax={spec['vmax']}", fontsize=10)
    axes[1].imshow(cart, cmap="inferno")
    axes[1].set_title(f"cart-processed   192x192\n"
                         f"crop={crop}, COM=off", fontsize=10)
    axes[2].imshow(polar, cmap="inferno", aspect="auto")
    axes[2].set_title(f"polar (model input)\n"
                         f"polar_mask_cols={pmc}, "
                         f"center_mask_r={cmr}", fontsize=10)
    axes[2].set_xlabel("r  (low -> high)", fontsize=9)
    axes[2].set_ylabel(r"$\theta$  (0 to 2$\pi$)", fontsize=9)
    for a in axes[:2]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{sample_key} -- sanity panel", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor="white",
                  bbox_inches="tight")
    plt.close(fig)


def _render_run_classmap(outdir: str, sample: str,
                              m: float, K: int, seed: int,
                              metrics: dict):
    """Save a class-map PNG next to the run's checkpoint immediately
    after the run finishes.  Used INSTEAD of waiting for the stage-
    level grid.  The user can browse `<outdir>/class_map.png` of every
    completed run while later runs are still going.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    inf = os.path.join(outdir, "eval", "inference.npz")
    if not os.path.exists(inf):
        return
    try:
        d = np.load(inf, allow_pickle=True)
        assigns = np.asarray(d["assigns"])
        soft = np.asarray(d["soft_probs"])
        K_pred = int(soft.shape[1])
        Ny, Nx = SAMPLES[sample]["scan_shape"]
        cm = assigns.reshape(Ny, Nx)
        # K_eff already in the metrics dict.
        keff = metrics.get("K_eff_end_smooth", float("nan"))
        nlive = metrics.get("n_live_end", 0)
        avg_conf5 = metrics.get("avg_conf_e5", float("nan"))
        # Aspect ratio matches scan.
        h = 5.0
        w = max(3.0, h * Nx / max(Ny, 1))
        fig, ax = plt.subplots(figsize=(w, h), dpi=130)
        cmap = plt.get_cmap("tab20" if K_pred <= 20 else "viridis",
                                K_pred)
        ax.imshow(cm, cmap=cmap, vmin=-0.5, vmax=K_pred - 0.5,
                    interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"{sample}  m={m:g}  K={K}  seed={seed}\n"
            f"K_eff={keff:.2f}   n_live={nlive}   "
            f"avg_conf_e5={avg_conf5:.2f}",
            fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "class_map.png"),
                      dpi=140, facecolor="white",
                      bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[classmap] {outdir}: render failed: {e!r}",
              flush=True)


# =========================================================================
# 3.  Per-run launch
# =========================================================================
def _paper_kwargs(sample: str, m: float, K: int, seed: int,
                      epochs: int,
                      supcon_radials_path: "str | None" = None,
                      supcon_thresholds_path: "str | None" = None,
                      ) -> dict:
    """Build the full kwargs dict that ``run_config`` expects, mirroring
    the GUI training-panel's "1D" variant defaults exactly.  Only the
    sweep axes (m, K, seed) and the per-sample pre-processing
    (vmax, center_crop_size, polar_mask_cols, com_centering) vary.
    """
    spec = SAMPLE_SPECS[sample]
    d = PAPER_DEFAULTS
    # GUI's split (matches train_panel.py:966–967):
    warmup_epochs = int(round((2.0 / 3.0) * epochs))
    ramp_epochs   = int(round((1.0 / 3.0) * epochs))
    # GUI's aug_disable: defaults have hflip/vflip/colorjitter OFF and
    # blur ON, so only the three OFF tags are disabled.
    aug_disable = []
    for tag, key in (("hflip", "aug_hflip"),
                       ("vflip", "aug_vflip"),
                       ("colorjitter", "aug_colorjitter"),
                       ("blur", "aug_blur")):
        if not bool(d.get(key, False)):
            aug_disable.append(tag)
    return dict(
        epochs=int(epochs), seed=int(seed),
        batch_size=int(d["batch_size"]),
        lr=float(d["lr"]),
        weight_decay=float(d["weight_decay"]),
        num_prototypes=int(K),
        t0=float(d["T0"]), tfin=float(d["Tfin"]),
        center_momentum=float(m),                          # SWEPT
        EMA0=float(d["EMA0"]), EMAfin=float(d["EMAfin"]),
        warmup_frac=float(d["warmup_frac"]),
        warmup_epochs=warmup_epochs, ramp_epochs=ramp_epochs,
        entropy_gate=False,
        projection_dim=128, projection_hidden=256,
        theta_shift_range=None,
        theta_shift_range_student=int(d["theta_shift_student"]),
        theta_shift_range_teacher=int(d["theta_shift_teacher"]),
        # CenterMask redundant with polar_mask_cols; pass exact
        # derived value (0 means no cart-space mask at all).
        center_mask_radius=int(spec["polar_mask_cols"]) // 2,
        center_crop_size=int(spec["center_crop_size"]),
        vmax=float(spec["vmax"]),
        polar_size=int(d["polar_size"]),
        polar_mask_cols=int(spec["polar_mask_cols"]),
        pipeline="polar",
        centroid_lambda=0.0,                                # 1D variant: off
        centroid_margin=float(d["centroid_margin"]),
        conf_weight_gamma=float(d["conf_weight_gamma"]),    # 1D variant: ON
        entropy_gate_override=None,
        lam_spatial=0.0,                                     # 1D variant: off
        spatial_tau_pos=float(d["spatial_tau_pos"]),
        spatial_tau_neg=float(d["spatial_tau_neg"]),
        architecture="resnet",
        n_layers=int(d["n_layers"]),
        w_ent=0.0,
        com_centering=bool(spec["com_centering"]),
        com_search_radius_factor=float(d["com_search_radius_factor"]),
        aug_disable=aug_disable,
        cj_brightness=float(d["cj_brightness"]),
        cj_contrast=float(d["cj_contrast"]),
        blur_kernel_max=int(d["blur_kernel_max"]),
        blur_sigma_max=float(d["blur_sigma_max"]),
        # REQUIRED for cluster1d_lambda_intra / inter to fire — the
        # training loop silently disables those terms when these are
        # missing.  Set via _ensure_radials() at the top of each
        # sample's processing.
        supcon_radials_path=supcon_radials_path,
        supcon_thresholds_path=supcon_thresholds_path,
        supcon_lambda=0.0,
        supcon_temperature=float(d["supcon_temperature"]),
        target_mode="dino",
        sinkhorn_eps=0.05, sinkhorn_iters=3,
        contrastive_lambda_override=0.0,
        proto_repel_lambda=0.0, proto_repel_threshold=0.5,
        cluster1d_lambda=float(d["cluster1d_lambda"]),
        cluster1d_lambda_intra=float(d["cluster1d_lambda_intra"]),
        cluster1d_lambda_inter=float(d["cluster1d_lambda_inter"]),
        cluster1d_margin=float(d["cluster1d_margin"]),
        cluster1d_min_cluster_mass=1.0,
        cluster1d_warmup_frac=float(d["cluster1d_warmup_frac"]),
        cluster1d_ramp_frac=float(d["cluster1d_ramp_frac"]),
        pair_labels_path=None,
        lambda_pair=0.0,
        pair_entropy_reg=0.0,
        pair_per_batch=int(d["pair_per_batch"]),
        save_every=int(d["save_every"]),
    )


def _launch_one(sample: str, m: float, K: int, seed: int,
                  outdir: str, *, dry_run: bool = False,
                  rad_path: "str | None" = None,
                  th_path: "str | None" = None) -> dict:
    """Invoke run_config with the sweep's locked kwargs.  Returns a
    dict of post-run metrics extracted from training_log.csv +
    inference.npz."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1 if dry_run else int(SWEEP_SPEC["epochs"])
    os.makedirs(outdir, exist_ok=True)
    kw = _paper_kwargs(sample, m, K, seed, epochs,
                          supcon_radials_path=rad_path,
                          supcon_thresholds_path=th_path)
    t0 = time.time()
    best_path = os.path.join(outdir, "best.pth")
    inf_npz   = os.path.join(outdir, "eval", "inference.npz")

    # ---- 1. Train (unless a best.pth already exists from a prior
    #         partial-progress launch with the same --root) ----
    if os.path.exists(best_path):
        print(f"[resume] {outdir}: best.pth present -- "
              f"skipping training, jumping to eval.", flush=True)
        run_ok, run_err = True, ""
    else:
        try:
            run_config(config_key="c", sample=sample,
                           outdir=outdir, device=device, **kw)
            run_ok, run_err = True, ""
        except Exception as e:
            run_ok, run_err = False, repr(e)
            traceback.print_exc()
    train_time_s = time.time() - t0

    # ---- 2. Eval (always, unless already done) ----
    # `run_config` only trains; the inference step has to be called
    # explicitly.  Without it there's no eval/inference.npz, which
    # means no K_eff, no class map, no n_live -- the run is opaque.
    # The user explicitly asked for eval per run, so we do it here.
    if run_ok and not os.path.exists(inf_npz):
        t1 = time.time()
        try:
            os.makedirs(os.path.dirname(inf_npz), exist_ok=True)
            evaluate_and_report(config_key="c", sample=sample,
                                    outdir=outdir, device=device,
                                    ckpt_path=best_path
                                      if os.path.exists(best_path)
                                      else None)
            print(f"[eval] {outdir}  ({time.time() - t1:.1f}s)",
                  flush=True)
        except Exception as e:
            print(f"[eval-FAIL] {outdir}: {e!r}", flush=True)
            traceback.print_exc()
            # Don't flip run_ok -- training succeeded; eval is salvageable
            # later via the watchdog.

    # Always extract metrics that exist; missing ones are NaN.
    metrics = _extract_metrics(outdir, sample)
    metrics.update(dict(
        sample=sample, m=float(m), K=int(K), seed=int(seed),
        outdir=outdir, train_time_s=float(train_time_s),
        ok=bool(run_ok), error=run_err,
    ))
    # Per-run class map — dropped next to the checkpoint as soon as
    # the run finishes, so the user can browse incremental results
    # without waiting for the whole stage.
    if run_ok:
        try:
            _render_run_classmap(outdir, sample, m, K, seed, metrics)
        except Exception as e:
            print(f"[classmap] {outdir}: failed to render: {e!r}",
                  flush=True)
    return metrics


def _extract_metrics(outdir: str, sample: str) -> dict:
    """Pull K_eff_end, n_live, avg_conf@5, final loss, effective_rank
    from the run's training_log + inference.npz.
    """
    out = dict(K_eff_end=float("nan"), K_eff_end_smooth=float("nan"),
                  n_live_end=int(0), avg_conf_e5=float("nan"),
                  loss_final=float("nan"), effective_rank=float("nan"),
                  has_nan=False)
    # 1) training_log.csv
    log_path = os.path.join(outdir, "training_log.csv")
    if os.path.exists(log_path):
        try:
            rows = []
            with open(log_path) as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    rows.append(r)
            if rows:
                last = rows[-1]
                ep5  = next((r for r in rows
                                if int(float(r.get("epoch", -1))) == 5), None)
                # K_eff can appear as 'effK', 'k_eff', or be derived
                # from the entropy column.
                for k in ("effK", "k_eff", "K_eff"):
                    if k in last and last[k] not in ("", None):
                        out["K_eff_end"] = float(last[k])
                        break
                # Smoothed: mean of last 5 epochs.
                tail = rows[-5:]
                tail_keff = [float(r[k]) for r in tail
                                for k in ("effK", "k_eff", "K_eff")
                                if k in r and r[k] not in ("", None)]
                if tail_keff:
                    out["K_eff_end_smooth"] = float(np.mean(tail_keff))
                if ep5 is not None:
                    for k in ("avg_conf", "avgconf"):
                        if k in ep5 and ep5[k] not in ("", None):
                            out["avg_conf_e5"] = float(ep5[k])
                            break
                for k in ("loss", "total_loss"):
                    if k in last and last[k] not in ("", None):
                        out["loss_final"] = float(last[k])
                        break
                # NaN guard
                for r in rows:
                    v = r.get("loss") or r.get("total_loss") or ""
                    try:
                        if v and not math.isfinite(float(v)):
                            out["has_nan"] = True; break
                    except Exception:
                        pass
        except Exception as e:
            print(f"[metrics] {outdir}: log parse failed: {e!r}", flush=True)
    # 2) inference.npz — load if present, compute K_eff_end, n_live,
    #    effective rank.
    inf_path = os.path.join(outdir, "eval", "inference.npz")
    if os.path.exists(inf_path):
        try:
            d = np.load(inf_path, allow_pickle=True)
            soft = np.asarray(d["soft_probs"])
            assigns = np.asarray(d["assigns"])
            embeds = np.asarray(d["embeds"]) if "embeds" in d.files else None
            K = int(soft.shape[1])
            p_bar = soft.mean(axis=0)
            pb = np.clip(p_bar, 1e-12, 1.0)
            keff = float(np.exp(-(pb * np.log(pb)).sum()))
            if not math.isfinite(out["K_eff_end"]):
                out["K_eff_end"] = keff
                out["K_eff_end_smooth"] = keff
            n_live = int((p_bar > 1.0 / max(K, 1)).sum())
            out["n_live_end"] = n_live
            if embeds is not None and embeds.shape[0] > 8:
                # Effective rank of L2-normed student embeddings:
                # (Σλ_i)² / Σλ_i² over the covariance eigenvalues.
                X = embeds.astype(np.float64)
                X -= X.mean(axis=0, keepdims=True)
                cov = (X.T @ X) / max(X.shape[0] - 1, 1)
                evals = np.linalg.eigvalsh(cov).clip(min=0)
                s = float(evals.sum())
                s2 = float((evals * evals).sum())
                out["effective_rank"] = (s * s) / s2 if s2 > 0 else 0.0
        except Exception as e:
            print(f"[metrics] {outdir}: inference parse failed: {e!r}",
                  flush=True)
    return out


# =========================================================================
# 4.  Aggregation, figures, report
# =========================================================================
def _picks_top_m(stage1_df, top_n: int = 3) -> list[float]:
    """Pick the top-N m candidates from Stage 1 results.

    Criterion: cross-seed std of K_eff_end_smooth (lower = more
    stable), broken by mean K_eff_end_smooth being inside the band
    [3, K_stage1] (i.e. neither full collapse nor no-death).
    """
    if not stage1_df:
        return []
    from collections import defaultdict
    by_m: dict = defaultdict(list)
    for r in stage1_df:
        if not r.get("ok"): continue
        if r.get("has_nan"): continue
        v = r.get("K_eff_end_smooth")
        if v is None or not np.isfinite(v): continue
        by_m[float(r["m"])].append(float(v))
    candidates = []
    for m, vals in by_m.items():
        if len(vals) == 0: continue
        mean_k = float(np.mean(vals))
        std_k  = float(np.std(vals))
        # Filter: collapse / no-death pathological cases.
        K_stage1 = SWEEP_SPEC["K_stage1"]
        if mean_k < 2.0 or mean_k > K_stage1 - 1.0:
            continue
        candidates.append((m, mean_k, std_k))
    candidates.sort(key=lambda t: (t[2], -t[1]))   # std asc; tiebreak: higher K_eff first
    return [m for (m, _, _) in candidates[:top_n]]


def _write_stage1_figures(sample: str, df, fig_dir: str, table_dir: str):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    # Aggregate per m.
    from collections import defaultdict
    by_m: dict = defaultdict(lambda: {"keff": [], "rank": [],
                                          "avg_conf_e5": [],
                                          "loss_final": [],
                                          "n_live": []})
    for r in df:
        if not r.get("ok"): continue
        m = float(r["m"])
        for k_src, k_dst in (("K_eff_end_smooth", "keff"),
                                 ("effective_rank",    "rank"),
                                 ("avg_conf_e5",       "avg_conf_e5"),
                                 ("loss_final",        "loss_final"),
                                 ("n_live_end",        "n_live")):
            v = r.get(k_src)
            if v is not None and np.isfinite(float(v)):
                by_m[m][k_dst].append(float(v))
    ms = sorted(by_m.keys())
    K_stage1 = SWEEP_SPEC["K_stage1"]
    K_true_band = SWEEP_SPEC["K_true_band"].get(sample, [None, None])

    def _ms_arr(field):
        return (np.array([float(np.mean(by_m[m][field]))
                          if by_m[m][field] else np.nan for m in ms]),
                np.array([float(np.std(by_m[m][field]))
                          if by_m[m][field] else 0.0 for m in ms]))

    # --- Fig 1: K_eff vs m
    mean_k, std_k = _ms_arr("keff")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
    ax.errorbar(ms, mean_k, yerr=std_k, marker="o", capsize=4,
                  color="#1f77b4", lw=1.4)
    ax.axhline(K_stage1, color="#999", ls=":", lw=1,
                 label=f"K = {K_stage1} (max)")
    ax.axhline(1, color="#999", ls=":", lw=1, label="K_eff = 1 (collapse)")
    if K_true_band[0] is not None:
        ax.axhspan(K_true_band[0], K_true_band[1],
                     color="#2ca02c", alpha=0.15,
                     label=f"expected K_true band [{K_true_band[0]}, {K_true_band[1]}]")
    ax.set_xscale("function",
                    functions=(lambda x: 1 - x, lambda y: 1 - y))
    ax.set_xticks(ms)
    ax.set_xticklabels([f"{m:g}" for m in ms], rotation=0)
    ax.set_xlabel("center momentum  m")
    ax.set_ylabel(r"$K_{\mathrm{eff}}$ at end of training  (mean ± std over seeds)")
    ax.set_title(f"{sample} — Stage 1: K = {K_stage1}, "
                  f"seeds = {SWEEP_SPEC['seeds_stage1']}",
                  fontsize=11)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_stage1_K_eff_vs_m.png"),
                  dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: avg_conf at epoch 5 vs m (early-collapse detector)
    mean_c, std_c = _ms_arr("avg_conf_e5")
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=130)
    ax.errorbar(ms, mean_c, yerr=std_c, marker="s", capsize=4,
                  color="#d62728", lw=1.4)
    ax.axhspan(0.2, 0.6, color="#2ca02c", alpha=0.15,
                 label="healthy band [0.2, 0.6]")
    ax.axhline(0.85, color="#999", ls=":", lw=1,
                 label="auto-detector threshold 0.85")
    ax.set_xticks(ms); ax.set_xticklabels([f"{m:g}" for m in ms])
    ax.set_xlabel("center momentum  m")
    ax.set_ylabel("avg_conf at epoch 5")
    ax.set_title(f"{sample} — Stage 1: early collapse diagnostic",
                  fontsize=11)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_stage1_avg_conf_e5.png"),
                  dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # --- Fig 3: effective rank vs m
    mean_r, std_r = _ms_arr("rank")
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=130)
    ax.errorbar(ms, mean_r, yerr=std_r, marker="^", capsize=4,
                  color="#9467bd", lw=1.4)
    ax.set_xticks(ms); ax.set_xticklabels([f"{m:g}" for m in ms])
    ax.set_xlabel("center momentum  m")
    ax.set_ylabel("effective rank of student CLS embeddings")
    ax.set_title(f"{sample} — Stage 1: feature-collapse diagnostic",
                  fontsize=11)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_stage1_eff_rank.png"),
                  dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # --- CSV
    tbl_path = os.path.join(table_dir, "stage1.csv")
    with open(tbl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "n_seeds_ok",
                    "K_eff_mean", "K_eff_std",
                    "n_live_mean",
                    "avg_conf_e5_mean", "avg_conf_e5_std",
                    "loss_final_mean",
                    "effective_rank_mean"])
        for m in ms:
            keff = by_m[m]["keff"]
            rank = by_m[m]["rank"]
            ac   = by_m[m]["avg_conf_e5"]
            ll   = by_m[m]["loss_final"]
            nl   = by_m[m]["n_live"]
            w.writerow([f"{m:.4f}", len(keff),
                          f"{np.mean(keff):.3f}" if keff else "",
                          f"{np.std(keff):.3f}"  if keff else "",
                          f"{np.mean(nl):.1f}"    if nl   else "",
                          f"{np.mean(ac):.3f}"    if ac   else "",
                          f"{np.std(ac):.3f}"     if ac   else "",
                          f"{np.mean(ll):.4f}"    if ll   else "",
                          f"{np.mean(rank):.2f}"  if rank else "",
                          ])


def _write_stage2_figures(sample: str, df, fig_dir: str, table_dir: str,
                              top_m: list):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    K_grid = sorted(SWEEP_SPEC["K_grid_stage2"])
    K_true_band = SWEEP_SPEC["K_true_band"].get(sample, [None, None])

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=140)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for ci, m in enumerate(sorted(top_m)):
        keffs = []
        for K in K_grid:
            matching = [r for r in df if r.get("ok")
                          and float(r["m"]) == float(m)
                          and int(r["K"]) == int(K)]
            v = (float(matching[0]["K_eff_end_smooth"])
                   if matching and
                   np.isfinite(matching[0].get("K_eff_end_smooth",
                                                  float("nan")))
                   else float("nan"))
            keffs.append(v)
        ax.plot(K_grid, keffs, marker="o", lw=1.6,
                  color=colors[ci % len(colors)],
                  label=f"m = {m:g}")
    # Diagonal "no death" reference.
    ax.plot(K_grid, K_grid, color="#aaaaaa", ls=":", lw=1,
              label=r"$K_{\mathrm{eff}} = K$  (no death)")
    if K_true_band[0] is not None:
        ax.axhspan(K_true_band[0], K_true_band[1],
                     color="#2ca02c", alpha=0.15,
                     label=f"expected K_true [{K_true_band[0]}, {K_true_band[1]}]")
    ax.set_xscale("log")
    ax.set_xticks(K_grid); ax.set_xticklabels([str(K) for K in K_grid])
    ax.set_xlabel("prototype-head size  K")
    ax.set_ylabel(r"$K_{\mathrm{eff}}$ at end of training")
    ax.set_title(f"{sample} — Stage 2: data-driven K plateau test",
                  fontsize=11)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_stage2_K_eff_vs_K.png"),
                  dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # CSV
    tbl_path = os.path.join(table_dir, "stage2.csv")
    with open(tbl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "K",
                    "K_eff_end", "K_eff_end_smooth",
                    "n_live_end", "avg_conf_e5",
                    "loss_final", "effective_rank",
                    "outdir"])
        for r in df:
            if not r.get("ok"): continue
            w.writerow([f"{r['m']:.4f}", r["K"],
                          f"{r.get('K_eff_end', '')}",
                          f"{r.get('K_eff_end_smooth', '')}",
                          r.get("n_live_end", ""),
                          f"{r.get('avg_conf_e5', '')}",
                          f"{r.get('loss_final', '')}",
                          f"{r.get('effective_rank', '')}",
                          r.get("outdir", "")])


def _render_class_grid(sample: str, df, fig_dir: str, *,
                            stage: str, kind: str = "classmap",
                            cols: int = 4):
    """Render a grid of class-map PNGs (or class-average PNGs) — one
    cell per run in df.  Used for stage1 and stage2 thumbnails."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    runs = [r for r in df if r.get("ok")]
    if not runs:
        return
    n = len(runs)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows),
                                dpi=120, squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_axis_off()
    Ny, Nx = SAMPLES[sample]["scan_shape"]
    for i, r in enumerate(runs):
        ax = axes[i // cols][i % cols]
        outdir = r["outdir"]
        inf = os.path.join(outdir, "eval", "inference.npz")
        if not os.path.exists(inf):
            continue
        try:
            d = np.load(inf, allow_pickle=True)
            assigns = np.asarray(d["assigns"]).reshape(Ny, Nx)
            soft = np.asarray(d["soft_probs"])
            K = int(soft.shape[1])
            if kind == "classmap":
                cmap = plt.get_cmap("tab20" if K <= 20 else "viridis", K)
                ax.imshow(assigns, cmap=cmap, vmin=-0.5, vmax=K - 0.5,
                            interpolation="nearest", aspect="equal")
                ax.set_axis_on()
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(
                    f"m={r['m']:g}  K={r['K']}  s={r['seed']}\n"
                    f"K_eff={r.get('K_eff_end_smooth', float('nan')):.2f}",
                    fontsize=8, pad=2)
            else:  # avg
                from data import LoadPRZ
                ds = LoadPRZ(SAMPLES[sample]["path"], resize=192,
                               vmax=SAMPLES[sample]["vmax"])
                a_flat = np.asarray(d["assigns"]).ravel()
                K_show = min(K, 12)
                inner_cmap = plt.get_cmap("tab20", K)
                rows_in = int(math.ceil(K_show / 4))
                # one composite tile per run (4xK/4 grid)
                from matplotlib.gridspec import GridSpecFromSubplotSpec
                gs = GridSpecFromSubplotSpec(rows_in, 4,
                                                subplot_spec=ax.get_subplotspec(),
                                                hspace=0.05, wspace=0.05)
                fig_in_axes = []
                for k in range(K_show):
                    sub = fig.add_subplot(gs[k // 4, k % 4])
                    idx_k = np.where(a_flat == k)[0]
                    if idx_k.size < 1:
                        sub.set_axis_off(); continue
                    pick = idx_k[:min(64, idx_k.size)]
                    avg = np.mean([ds.get_raw(int(i)) for i in pick], 0)
                    sub.imshow(avg, cmap="inferno",
                                 interpolation="nearest")
                    sub.set_xticks([]); sub.set_yticks([])
                ax.remove()
                # the outer label below the grid:
                fig.text((i % cols + 0.5) / cols,
                          1.0 - (i // cols + 0.95) / rows,
                          f"m={r['m']:g}  K={r['K']}  s={r['seed']}\n"
                          f"K_eff={r.get('K_eff_end_smooth', float('nan')):.2f}",
                          ha="center", va="top", fontsize=7)
        except Exception as e:
            ax.text(0.5, 0.5, f"err: {e!r}", ha="center", va="center",
                      fontsize=7, transform=ax.transAxes)
    fig.suptitle(f"{sample} — {stage} — {kind}", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(fig_dir,
                               f"fig_{stage}_{kind}_grid.png")
    fig.savefig(out_png, dpi=140, facecolor="white",
                  bbox_inches="tight")
    plt.close(fig)


def _write_html_report(sample: str, sample_dir: str,
                          stage1_df, stage2_df: "list | None",
                          top_m: list, *, final: bool):
    """Scientific-paper-style HTML report.  Written after Stage 1
    (partial) and after Stage 2 (final)."""
    fig_dir = os.path.join(sample_dir, "figures")
    table_dir = os.path.join(sample_dir, "tables")
    K_stage1 = SWEEP_SPEC["K_stage1"]
    K_grid = SWEEP_SPEC["K_grid_stage2"]
    band = SWEEP_SPEC["K_true_band"].get(sample, [None, None])
    title = f"Sweep over center momentum (m) and prototype count (K) — {sample}"

    def _img_tag(rel_path: str, w: int = 720) -> str:
        return (f'<img src="{rel_path}" style="width:{w}px;max-width:100%;'
                f'display:block;margin:0 auto;border:1px solid #ccc;'
                f'background:#fff" />')

    lines = []
    lines.append("<!doctype html><html><head><meta charset=utf-8>")
    lines.append(f"<title>{title}</title>")
    lines.append("<style>"
                 "body{font-family:Georgia,serif;max-width:920px;"
                 "margin:24px auto;padding:0 16px;color:#222;line-height:1.55}"
                 "h1{font-size:22px;border-bottom:2px solid #444;padding-bottom:6px}"
                 "h2{font-size:16px;margin-top:24px;border-bottom:1px solid #ccc}"
                 "code{background:#f4f4f4;padding:1px 4px;border-radius:3px}"
                 "table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}"
                 "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}"
                 "th{background:#f0f0f0;font-weight:bold}"
                 ".muted{color:#666;font-size:13px}"
                 ".caption{font-size:12px;color:#444;text-align:center;"
                 "margin:4px auto 16px auto;max-width:760px}"
                 "</style></head><body>")
    lines.append(f"<h1>{title}</h1>")
    stamp = datetime.now().isoformat(timespec='seconds')
    lines.append(f"<p class=muted>Report generated {stamp}. "
                  f"Sample {sample!r}; cube path {SAMPLES[sample]['path']!r}; "
                  f"scan {SAMPLES[sample]['scan_shape']}.</p>")

    lines.append("<h2>Methods</h2>")
    lines.append(f"<p>We trained the DINO-SR self-supervised model on the "
                  f"<code>{sample}</code> diffraction cube for {SWEEP_SPEC['epochs']} "
                  f"epochs over an 8 x 2 (center-momentum m x seed) grid at fixed "
                  f"prototype-head size K = {K_stage1} (Stage 1), and selected "
                  f"the top {SWEEP_SPEC['top_m_picks']} m candidates for a Stage 2 "
                  f"sweep over K ∈ {{{', '.join(map(str, K_grid))}}} at a single "
                  f"seed.  Pre-processing was held fixed across the entire sweep "
                  f"(vmax = {SAMPLE_SPECS[sample]['vmax']}, "
                  f"center crop = {SAMPLE_SPECS[sample]['center_crop_size']}, "
                  f"polar_mask_cols = {SAMPLE_SPECS[sample]['polar_mask_cols']}, "
                  f"COM-centering = "
                  f"{'on' if SAMPLE_SPECS[sample]['com_centering'] else 'off'}); "
                  f"all other hyperparameters were left at the validated paper "
                  f"defaults (T_t = 0.04, EMA τ = 0.99 -> 0.999, warmup_frac = 0.2).")
    lines.append("Per-run reporting includes the effective number of used "
                  f"prototypes <em>K<sub>eff</sub></em> = exp(H(<span style='font-style:italic'>p̄</span>)), "
                  "the number of live prototypes <em>n<sub>live</sub></em> (defined as "
                  "<span style='font-style:italic'>p̄<sub>k</sub></span> &gt; 1/K), the "
                  "avg-confidence-at-epoch-5 (an early-collapse detector), and the "
                  "effective rank of the student CLS embeddings (a feature-collapse "
                  "diagnostic, defined as (Σ λ<sub>i</sub>)² / Σ λ<sub>i</sub>² over "
                  "the embedding covariance spectrum).</p>")
    if band[0] is not None:
        lines.append(f"<p>For <code>{sample}</code> the expected number of phases "
                      f"is in the band [{band[0]}, {band[1]}], based on "
                      f"prior validated runs of the same data.</p>")

    # Stage 1 section.
    lines.append("<h2>Stage 1 — m sweep at K = "
                  f"{K_stage1}</h2>")
    lines.append(f"<p>{len([r for r in stage1_df if r.get('ok')])} of "
                  f"{len(stage1_df)} runs completed without error.</p>")
    lines.append(_img_tag("figures/fig_stage1_K_eff_vs_m.png"))
    lines.append("<div class=caption>Figure 1.  Effective prototype count "
                  "<em>K<sub>eff</sub></em> at the end of training, as a "
                  "function of center momentum m, averaged across seeds.  "
                  "Error bars show ±1 standard deviation.  The green band "
                  "indicates the expected <em>K<sub>true</sub></em> for "
                  "this sample, when known.</div>")
    lines.append(_img_tag("figures/fig_stage1_avg_conf_e5.png"))
    lines.append("<div class=caption>Figure 2.  Average teacher confidence "
                  "at epoch 5, used as an early collapse diagnostic.  Values "
                  "above ~0.85 indicate pathological collapse during the "
                  "warmup phase.</div>")
    lines.append(_img_tag("figures/fig_stage1_eff_rank.png"))
    lines.append("<div class=caption>Figure 3.  Effective rank of the "
                  "student CLS-embedding covariance.  Low values (≲ K<sub>eff</sub>) "
                  "indicate feature-space collapse hidden behind a deceptively "
                  "stable <em>K<sub>eff</sub></em>.</div>")
    lines.append("<h3>Stage 1 table</h3>")
    # Inline the CSV as an HTML table
    csv_path = os.path.join(table_dir, "stage1.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rdr = csv.reader(f)
            header = next(rdr, None)
            lines.append("<table><thead><tr>"
                          + "".join(f"<th>{h}</th>" for h in (header or []))
                          + "</tr></thead><tbody>")
            for row in rdr:
                lines.append("<tr>"
                              + "".join(f"<td>{c}</td>" for c in row)
                              + "</tr>")
            lines.append("</tbody></table>")
    lines.append("<p>Class maps and class-average diffraction patterns "
                  "for every Stage 1 run are shown below.</p>")
    fig_cm = "figures/fig_stage1_classmap_grid.png"
    if os.path.exists(os.path.join(sample_dir, fig_cm)):
        lines.append(_img_tag(fig_cm))
        lines.append("<div class=caption>Figure 4. Class maps for every "
                      "Stage 1 run.  Cell title shows m, K, seed, and the "
                      "smoothed end-of-training <em>K<sub>eff</sub></em>.</div>")
    lines.append(f"<p><strong>Selected top-{SWEEP_SPEC['top_m_picks']} m "
                  f"candidates</strong> for Stage 2 "
                  f"(cross-seed stability of <em>K<sub>eff</sub></em>): "
                  f"{', '.join(f'{m:g}' for m in top_m) or '—'}</p>")

    # Stage 2 section (only present when final=True).
    if final and stage2_df:
        lines.append(f"<h2>Stage 2 — plateau test, K ∈ "
                      f"{{{', '.join(map(str, K_grid))}}}</h2>")
        lines.append(_img_tag("figures/fig_stage2_K_eff_vs_K.png"))
        lines.append("<div class=caption>Figure 5.  Plateau test: "
                      "<em>K<sub>eff</sub></em> at the end of training as "
                      "a function of the prototype-head size K, for each "
                      "of the top m candidates.  A horizontal line "
                      "indicates that K<sub>eff</sub> is independent of K "
                      "— i.e. the model is converging to an intrinsic "
                      "<em>K<sub>true</sub></em> set by the data, not by "
                      "the user-chosen K.  The dotted diagonal "
                      "(<em>K<sub>eff</sub></em> = K) is the no-death "
                      "limit.</div>")
        lines.append("<h3>Stage 2 table</h3>")
        csv_path = os.path.join(table_dir, "stage2.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                rdr = csv.reader(f)
                header = next(rdr, None)
                lines.append("<table><thead><tr>"
                              + "".join(f"<th>{h}</th>" for h in (header or []))
                              + "</tr></thead><tbody>")
                for row in rdr:
                    lines.append("<tr>"
                                  + "".join(f"<td>{c}</td>" for c in row)
                                  + "</tr>")
                lines.append("</tbody></table>")
        fig_cm = "figures/fig_stage2_classmap_grid.png"
        if os.path.exists(os.path.join(sample_dir, fig_cm)):
            lines.append(_img_tag(fig_cm))
            lines.append("<div class=caption>Figure 6.  Class maps from "
                          "every Stage 2 run.</div>")

    # Discussion stub (filled by hand later if needed).
    lines.append("<h2>Discussion</h2>")
    if final and stage2_df:
        lines.append("<p>If the curves in Figure 5 flatten at a common "
                      "<em>K<sub>eff</sub></em> value across K, the result "
                      "supports the data-driven-K claim: with m at the "
                      "plateau, K becomes an output of the model rather "
                      "than a user input.  Discrepancies between the "
                      "candidate m values indicate sensitivity to the "
                      "centering schedule and should be reported as the "
                      "uncertainty in <em>K<sub>true</sub></em>.</p>")
    else:
        lines.append("<p>Stage 2 has not yet completed.  The plateau "
                      "test will be appended to this report as soon as "
                      "the K sweep finishes for the top-m candidates "
                      "selected above.</p>")
    lines.append("</body></html>")

    name = "report.html" if final else "report_stage1.html"
    out = os.path.join(sample_dir, name)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] wrote {out}", flush=True)


# =========================================================================
# 5.  Main loop
# =========================================================================
def _stop_requested(sweep_root: str) -> bool:
    return os.path.exists(os.path.join(sweep_root, "STOP_SWEEP"))


def _append_progress(sweep_root: str, row: dict):
    p = os.path.join(sweep_root, "SWEEP_PROGRESS.csv")
    is_new = not os.path.exists(p)
    keys = ["timestamp", "sample", "stage", "m", "K", "seed",
            "outdir", "ok", "K_eff_end_smooth", "n_live_end",
            "avg_conf_e5", "loss_final", "effective_rank",
            "has_nan", "train_time_s", "error"]
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if is_new: w.writeheader()
        out = {k: row.get(k, "") for k in keys}
        out["timestamp"] = datetime.now().isoformat(timespec="seconds")
        w.writerow(out)


def _run_stage1(sample: str, sample_dir: str, sweep_root: str,
                  *, dry_run: bool = False,
                  rad_path: "str | None" = None,
                  th_path: "str | None" = None) -> list:
    out = []
    K = SWEEP_SPEC["K_stage1"]
    for m in SWEEP_SPEC["m_grid"]:
        for seed in SWEEP_SPEC["seeds_stage1"]:
            if _stop_requested(sweep_root):
                print("[stop] STOP_SWEEP detected — exiting Stage 1.",
                      flush=True); return out
            name = f"m{m:.4f}_seed{seed}_K{K}"
            outdir = os.path.join(sample_dir, "stage1", name)
            # Full skip ONLY if BOTH training and eval are done.
            # If best.pth exists but inference.npz doesn't, fall through
            # to _launch_one which will detect this and run eval-only.
            if (os.path.exists(os.path.join(outdir, "best.pth")) and
                    os.path.exists(os.path.join(outdir,
                                                 "eval/inference.npz"))):
                print(f"[skip] stage1 {name} already complete.",
                      flush=True)
                metrics = _extract_metrics(outdir, sample)
                metrics.update(dict(sample=sample, m=float(m), K=int(K),
                                       seed=int(seed), outdir=outdir,
                                       ok=True, error="",
                                       stage="stage1",
                                       train_time_s=0.0))
                # Make sure the class map exists too, in case this
                # run was completed by an older launcher version.
                if not os.path.exists(os.path.join(outdir,
                                                       "class_map.png")):
                    try:
                        _render_run_classmap(outdir, sample, m, K,
                                                seed, metrics)
                    except Exception as e:
                        print(f"[classmap-skip-path] {outdir}: "
                              f"{e!r}", flush=True)
                out.append(metrics)
                _append_progress(sweep_root, metrics)
                continue
            print(f"\n[run] stage1  {sample}  m={m:g}  seed={seed}  K={K} "
                  f"  -> {outdir}", flush=True)
            r = _launch_one(sample, m, K, seed, outdir,
                                dry_run=dry_run,
                                rad_path=rad_path, th_path=th_path)
            r["stage"] = "stage1"
            out.append(r)
            _append_progress(sweep_root, r)
    return out


def _run_stage2(sample: str, sample_dir: str, sweep_root: str,
                  top_m: list, *, dry_run: bool = False,
                  rad_path: "str | None" = None,
                  th_path: "str | None" = None) -> list:
    out = []
    seed = SWEEP_SPEC["seed_stage2"]
    for m in top_m:
        for K in SWEEP_SPEC["K_grid_stage2"]:
            if _stop_requested(sweep_root):
                print("[stop] STOP_SWEEP detected — exiting Stage 2.",
                      flush=True); return out
            name = f"m{m:.4f}_seed{seed}_K{K}"
            outdir = os.path.join(sample_dir, "stage2", name)
            if (os.path.exists(os.path.join(outdir, "best.pth")) and
                    os.path.exists(os.path.join(outdir,
                                                 "eval/inference.npz"))):
                print(f"[skip] stage2 {name} already complete.",
                      flush=True)
                metrics = _extract_metrics(outdir, sample)
                metrics.update(dict(sample=sample, m=float(m), K=int(K),
                                       seed=int(seed), outdir=outdir,
                                       ok=True, error="",
                                       stage="stage2",
                                       train_time_s=0.0))
                if not os.path.exists(os.path.join(outdir,
                                                       "class_map.png")):
                    try:
                        _render_run_classmap(outdir, sample, m, K,
                                                seed, metrics)
                    except Exception as e:
                        print(f"[classmap-skip-path] {outdir}: "
                              f"{e!r}", flush=True)
                out.append(metrics)
                _append_progress(sweep_root, metrics)
                continue
            print(f"\n[run] stage2  {sample}  m={m:g}  K={K}  seed={seed} "
                  f"  -> {outdir}", flush=True)
            r = _launch_one(sample, m, K, seed, outdir,
                                dry_run=dry_run,
                                rad_path=rad_path, th_path=th_path)
            r["stage"] = "stage2"
            out.append(r)
            _append_progress(sweep_root, r)
    return out


def _process_sample(sample: str, sweep_root: str, *,
                       dry_run: bool = False):
    sample_dir = os.path.join(sweep_root, sample)
    os.makedirs(sample_dir, exist_ok=True)
    fig_dir = os.path.join(sample_dir, "figures")
    table_dir = os.path.join(sample_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)
    # Register sample + sanity render
    _register_sample(sample)
    with open(os.path.join(sample_dir, "SAMPLE_LOCK.json"),
                "w", encoding="utf-8") as f:
        json.dump(SAMPLE_SPECS[sample], f, indent=2)
    try:
        _sanity_render(sample, os.path.join(sample_dir, "sanity_frame.png"))
    except Exception as e:
        print(f"[sanity] {sample}: failed: {e!r}", flush=True)

    # Ensure 1D radial profiles + gate thresholds exist BEFORE Stage 1.
    # cluster1d_lambda_intra / inter are silently disabled by the
    # training loop when these are missing, which would re-run the
    # vanilla-DINO bug.
    rad_path = th_path = None
    try:
        rad_path, th_path = _ensure_radials(sample)
    except Exception as e:
        print(f"[radials] {sample}: FATAL — could not produce radials/"
              f"thresholds: {e!r}\n             cluster1d losses would "
              f"be disabled.  Aborting this sample.", flush=True)
        traceback.print_exc()
        return

    # ---- Stage 1 ----
    print(f"\n{'='*72}\n[sweep] {sample} :: STAGE 1\n{'='*72}", flush=True)
    stage1 = _run_stage1(sample, sample_dir, sweep_root,
                            dry_run=dry_run,
                            rad_path=rad_path, th_path=th_path)
    print(f"\n[sweep] {sample} :: Stage 1 done.  rendering figures + "
          f"partial report …", flush=True)
    _write_stage1_figures(sample, stage1, fig_dir, table_dir)
    _render_class_grid(sample, stage1, fig_dir,
                          stage="stage1", kind="classmap")
    top_m = _picks_top_m(stage1, top_n=SWEEP_SPEC["top_m_picks"])
    print(f"[sweep] {sample} :: top-m candidates for Stage 2: "
          f"{top_m}", flush=True)
    _write_html_report(sample, sample_dir, stage1, None,
                          top_m, final=False)
    if _stop_requested(sweep_root):
        print("[stop] STOP_SWEEP — aborting after Stage 1.", flush=True)
        return

    # ---- Stage 2 ----
    if not top_m:
        print(f"[sweep] {sample} :: no viable top-m candidates "
              f"(all collapsed or no-death) — skipping Stage 2.",
              flush=True)
        return
    print(f"\n{'='*72}\n[sweep] {sample} :: STAGE 2\n{'='*72}", flush=True)
    stage2 = _run_stage2(sample, sample_dir, sweep_root, top_m,
                            dry_run=dry_run,
                            rad_path=rad_path, th_path=th_path)
    print(f"\n[sweep] {sample} :: Stage 2 done.  rendering figures + "
          f"final report …", flush=True)
    _write_stage2_figures(sample, stage2, fig_dir, table_dir, top_m)
    _render_class_grid(sample, stage2, fig_dir,
                          stage="stage2", kind="classmap")
    _write_html_report(sample, sample_dir, stage1, stage2, top_m,
                          final=True)
    print(f"\n[sweep] {sample} :: report at "
          f"{os.path.join(sample_dir, 'report.html')}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=list(SAMPLE_SPECS.keys()),
                    default=None,
                    help="Process only this sample (default: all).")
    ap.add_argument("--dry-run", action="store_true",
                    help="1-epoch runs for plumbing check.")
    ap.add_argument("--root", default=None,
                    help="Override sweep root (default: "
                          "runs/_sweep_m_K_<timestamp>).")
    args = ap.parse_args(argv)

    if args.root:
        sweep_root = args.root
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_root = os.path.join(REPO, "runs", f"_sweep_m_K_{ts}")
    os.makedirs(sweep_root, exist_ok=True)
    # Freeze the spec next to the data.
    with open(os.path.join(sweep_root, "SWEEP_SPEC.json"),
                "w", encoding="utf-8") as f:
        json.dump({**SWEEP_SPEC,
                       "samples": SAMPLE_SPECS,
                       "epochs":  (1 if args.dry_run
                                     else SWEEP_SPEC["epochs"])},
                      f, indent=2)
    print(f"[sweep] root = {sweep_root}", flush=True)
    print(f"[sweep] STOP file = "
          f"{os.path.join(sweep_root, 'STOP_SWEEP')}", flush=True)
    # Echo the locked per-sample pre-processing so the user can
    # eyeball the values BEFORE the long run starts.
    print("\n[sweep] locked per-sample pre-processing (passed to every "
          "run; varies only m, K, seed):", flush=True)
    print(f"  {'sample':<14s} {'vmax':>6s} {'crop':>5s} "
          f"{'pmc':>5s} {'cmr':>5s} {'com':>5s}  cube_path", flush=True)
    for sname, sspec in SAMPLE_SPECS.items():
        derived_cmr = int(sspec["polar_mask_cols"]) // 2
        print(f"  {sname:<14s} {sspec['vmax']:>6g} "
              f"{sspec['center_crop_size']:>5d} "
              f"{sspec['polar_mask_cols']:>5d} "
              f"{derived_cmr:>5d} "
              f"{'OFF' if not sspec['com_centering'] else 'ON':>5s}  "
              f"{sspec['cube_path']}", flush=True)
    print("  (cmr = center_mask_radius derived as pmc // 2; "
          "cart-space CenterMask is redundant with the polar "
          "mask, so 0 means no extra mask.)\n", flush=True)

    order = ([args.sample] if args.sample
                else list(SWEEP_SPEC["sample_order"]))
    for sample in order:
        if _stop_requested(sweep_root):
            print("[stop] STOP_SWEEP detected — exiting before "
                  f"sample {sample}.", flush=True)
            break
        try:
            _process_sample(sample, sweep_root, dry_run=args.dry_run)
        except KeyboardInterrupt:
            print("[stop] keyboard interrupt — exiting.", flush=True)
            break
        except Exception as e:
            print(f"[sweep] {sample} FAILED at the sample level: "
                  f"{e!r}", flush=True)
            traceback.print_exc()
    print("\n[sweep] all done.", flush=True)


if __name__ == "__main__":
    main()
