# %% [markdown]
# # DINO-SR + Contrastive Head -- training / evaluation entry point
#
# This script can be run:
#   1. As a plain CLI:
#        python run_contrastive.py --config a --sample Na007b --epochs 50
#   2. As a Jupytext notebook (open in VS Code with the Jupyter extension,
#      or jupytext --to notebook). The `# %%` markers are cell boundaries.
#
# Ablation table (spec, SPEC.md section "Ablation protocol"):
#
# | Config | contrastive_lambda | theta_roll_aug | Purpose |
# |--------|-------------------:|:--------------:|---------|
# | a      | 0.0                | False          | Pure DINO L1 baseline  |
# | b      | 0.0                | True           | Isolates theta-roll    |
# | c      | 0.2                | True           | Full joint model       |
#
# Target sample: Na007b (user explicitly said Na007b only for the 3x50-epoch
# protocol; do NOT expand to other samples).

# %% imports
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from datetime import datetime

# Determinism: set BEFORE any torch / CUDA / cuBLAS call so the cuBLAS
# workspace is allocated in deterministic mode. setdefault means we don't
# clobber a value the user may have already exported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(BASE_DIR, "runs")


# %% config presets
CONFIG_PRESETS = {
    "a": dict(contrastive_lambda=0.0, theta_roll_aug=False,
              label="config_a_dino_baseline"),
    "b": dict(contrastive_lambda=0.0, theta_roll_aug=True,
              label="config_b_thetaroll"),
    "c": dict(contrastive_lambda=0.2, theta_roll_aug=True,
              label="config_c_full"),
}


# %% unit test — theta-roll equivariance (spec item 8)
def run_theta_roll_unit_test(device) -> dict:
    """Build a fresh (untrained) encoder with circular-theta conv, push a
    random input and its theta-rolled copy through, and check the feature
    maps agree on interior r-columns.

    Tolerance per NEW_SESSION_PROMPT.md Q6: < 1e-4 on the central r-region,
    essentially bit-exact (<= 1e-6) on the interior.
    """
    from dino_sr_contrastive_model import (
        create_encoder_resnet18_variableN,
        encoder_total_theta_stride,
        theta_roll_equivariance_test,
    )

    torch.manual_seed(0)
    enc = create_encoder_resnet18_variableN(
        n_layers=1, use_maxpool=True, circular_theta=True).to(device).eval()
    stride = encoder_total_theta_stride(1)
    x = torch.randn(2, 1, 192, 192, device=device)
    # Zero low-r cols so the input matches PolarMaskLeft output.
    x[..., :30] = 0.0
    k_theta = 16  # multiple of stride so output roll is exact.
    interior, full = theta_roll_equivariance_test(enc, x, k_theta, stride, r_edge=3)
    ok = interior < 1e-4
    print(f"[theta-roll-unit-test] stride={stride}  k_theta={k_theta}  "
          f"interior_max_abs={interior:.3e}  full_max_abs={full:.3e}  "
          f"ok={ok}")
    return dict(interior=interior, full=full, ok=bool(ok),
                stride=stride, k_theta=k_theta)


# %% run a single config
def run_config(config_key: str, sample: str, epochs: int, seed: int,
               batch_size: int, lr: float, weight_decay: float,
               num_prototypes: int,
               t0: float, tfin: float,
               warmup_epochs: int, ramp_epochs: int,
               entropy_gate: bool,
               projection_dim: int, projection_hidden: int,
               theta_shift_range: int | None,
               center_mask_radius: int | None,
               center_crop_size: int,
               vmax: float | None,
               polar_size: int, polar_mask_cols: int,
               outdir: str, device,
               theta_shift_range_student: int | None = None,
               theta_shift_range_teacher: int | None = None,
               pipeline: str = "polar",
               centroid_lambda: float = 0.0,
               centroid_margin: float = 0.3,
               conf_weight_gamma: float = 0.0,
               entropy_gate_override: bool | None = None,
               # DINO hyperparameters (paper defaults). Previously
               # hardcoded inside the model construction; now exposed
               # so the Training panel can surface them.  warmup_frac
               # is the fraction of epochs used for the teacher-temp
               # warmup ramp.
               center_momentum: float = 0.9,
               EMA0: float = 0.990,
               EMAfin: float = 0.999,
               warmup_frac: float = 0.2,
               lam_spatial: float = 0.0,
               # Gated 4-neighbor spatial loss thresholds (1D radial
               # agreement). Defaults pull when cos > 0.5, push when
               # cos < 0.0; sentinel -2 means "auto-load from
               # supcon_thresholds_path" (the per-sample calibrated
               # gate_thresholds.json).
               spatial_tau_pos: float = -2.0,
               spatial_tau_neg: float = -2.0,
               architecture: str = "resnet",
               n_layers: int = 1,
               conf_weight_auto: "dict | None" = None,
               stop_at_epoch: "int | None" = None,
               w_ent: float = 0.0,
               com_centering: bool = False,
               com_search_radius_factor: float = 2.0,
               aug_disable: "list | None" = None,
               # Per-augmentation params (used only when the matching
               # tag is NOT in `aug_disable`).
               cj_brightness: float = 0.2,
               cj_contrast:   float = 0.2,
               blur_kernel_max: int = 5,
               blur_sigma_max:  float = 0.3,
               supcon_radials_path: "str | None" = None,
               supcon_thresholds_path: "str | None" = None,
               supcon_lambda: float = 0.0,
               supcon_temperature: float = 0.1,
               contrastive_lambda_override: "float | None" = None,
               proto_repel_lambda: float = 0.0,
               proto_repel_threshold: float = 0.5,
               cluster1d_lambda: float = 0.0,
               # Per-term cluster1d weights.  -1 sentinel means "use
               # the unified cluster1d_lambda for both" (backward
               # compat).  Setting `intra > 0, inter = 0` is the
               # "let weak prototypes drift naturally" recipe.
               cluster1d_lambda_intra: float = -1.0,
               cluster1d_lambda_inter: float = -1.0,
               cluster1d_margin: float = 0.4,
               cluster1d_min_cluster_mass: float = 1.0,
               cluster1d_warmup_frac: float = 0.0,
               cluster1d_ramp_frac: float = 0.0,
               # Pair-supervised assignment loss (semi-supervised).
               # `pair_labels_path` is the canonical .pair_labels.json
               # sidecar; if set and non-empty, training adds
               # `lambda_pair · L_pair` per batch.
               pair_labels_path: "str | None" = None,
               # ISO-format cutoff. Only pairs whose `t` field is
               # STRICTLY GREATER than this are kept. Used for
               # fine-tune "only labels added since parent run".
               # Default None = use all pairs.
               pair_labels_min_timestamp: "str | None" = None,
               lambda_pair: float = 0.0,
               pair_entropy_reg: float = 0.0,
               pair_per_batch: int = 32,
               # ---- target-distribution mode -------------------------
               # "dino"     : centering + softmax(/temp_teacher)  (default)
               # "sinkhorn" : SwAV / DINOv2-SK style equipartition.
               #              Centering buffer is unused; sinkhorn_eps
               #              controls sharpness, sinkhorn_iters the
               #              SK iterations. Vanilla SwAV-style by
               #              setting cluster1d_lambda=0; SK + 1D-radial
               #              regularizer by setting it >0.
               target_mode: str = "dino",
               sinkhorn_eps: float = 0.05,
               sinkhorn_iters: int = 3,
               # ---- fine-tune kwargs (Phase B) -----------------------
               # Initialise model weights from this checkpoint *before*
               # training starts. Any layer not present in the ckpt
               # state-dict keeps its random init.
               init_from_checkpoint: "str | None" = None,
               # Freeze the student encoder backbone (not the projector
               # or prototypes). Useful for short fine-tunes where you
               # want the labels to nudge the head, not retrain the
               # representation.
               freeze_encoder: bool = False,
               # If set, restrict the DataLoader to a random subset of
               # `subsample_n` patterns. Shuffles deterministically
               # under `seed`. Pair-loss labels are NOT subsampled —
               # all of them are still seen each batch.
               subsample_n: "int | None" = None,
               save_every: int = 10):
    # Snapshot every argument the caller passed in, BEFORE any
    # reassignment / shadowing happens below.  Used at write-time to
    # dump the full DINO/contrastive/SupCon/pair/SK knob set into
    # run_summary.json so downstream tools (eval, post-hoc, GradCAM,
    # the HTML inspector) never have to guess what training actually
    # used.  Excludes objects that aren't JSON-safe (device, supcon
    # path strings stay as strings).
    _arg_snapshot = {
        k: locals()[k]
        for k in inspect.signature(run_config).parameters
        if k in locals() and k not in ("device", "outdir")
    }
    from data import SAMPLES as EVAL_SAMPLES, LoadPRZ, LoadPRZMulti
    from dino_sr_contrastive_model import (
        ContrastiveDINOModel,
        get_contrastive_transforms,
        get_cartesian_transforms,
        train_contrastive,
    )
    preset = CONFIG_PRESETS[config_key]
    cfg = EVAL_SAMPLES[sample]
    effective_vmax = vmax if vmax is not None else cfg["vmax"]
    mask_r = (center_mask_radius if center_mask_radius is not None
              else cfg.get("center_mask_radius", 15))
    scan_shape = cfg["scan_shape"]
    is_multi = bool(cfg.get("is_multi", False))

    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'#' * 72}")
    print(f"# config={config_key}  sample={sample}  epochs={epochs}  "
          f"outdir={outdir}")
    print(f"# contrastive_lambda={preset['contrastive_lambda']}  "
          f"theta_roll_aug={preset['theta_roll_aug']}")
    print(f"# T0={t0} Tfin={tfin} num_proto={num_prototypes}  "
          f"vmax={effective_vmax}  mask_r={mask_r}")
    print(f"{'#' * 72}\n", flush=True)

    if is_multi:
        dataset = LoadPRZMulti(cfg["paths"], resize=polar_size,
                               vmax=effective_vmax)
        N = len(dataset)
        print(f"[multi] {sample}: combined N={N} from {len(cfg['paths'])} PRZ", flush=True)
        # scan_shape is meaningless for combined training; disable spatial loss.
        if lam_spatial > 0:
            print(f"[multi] lam_spatial={lam_spatial} disabled for multi-SI training")
            lam_spatial = 0.0
    else:
        dataset = LoadPRZ(cfg["path"], resize=polar_size, vmax=effective_vmax)
        N = len(dataset)
        assert N == scan_shape[0] * scan_shape[1], \
            f"dataset size {N} != scan_shape {scan_shape} product"
    # `subsample_n` is plumbed straight through to train_contrastive
    # (which builds a SubsetRandomSampler instead of subsetting the
    # dataset, so pair-loss can still fetch arbitrary scan indices
    # from the full cube).

    if pipeline == "cartesian":
        # Cartesian pipeline: translate theta-roll ranges into degree ranges
        # (polar_size rows span 360 deg, so shift=s rows ≈ s/polar_size * 360 deg).
        eff_student = (theta_shift_range_student
                        if theta_shift_range_student is not None
                        else (theta_shift_range if theta_shift_range is not None
                              else polar_size))
        eff_teacher = (theta_shift_range_teacher
                        if theta_shift_range_teacher is not None
                        else (theta_shift_range if theta_shift_range is not None
                              else max(2, int(round(polar_size * 15 / 180)))))
        rot_student_deg = (eff_student / polar_size) * 180.0
        rot_teacher_deg = (eff_teacher / polar_size) * 180.0
        student_aug, teacher_aug, student_roll, teacher_roll = get_cartesian_transforms(
            center_mask_radius=mask_r,
            center_crop_size=center_crop_size,
            polar_size=polar_size,
            rotation_range_student=rot_student_deg,
            rotation_range_teacher=rot_teacher_deg,
        )
        circular_theta = False
    else:
        student_aug, teacher_aug, student_roll, teacher_roll = get_contrastive_transforms(
            disable=aug_disable,
            center_mask_radius=mask_r,
            center_crop_size=center_crop_size,
            polar_size=polar_size,
            polar_mask_cols=polar_mask_cols,
            theta_roll_aug=preset["theta_roll_aug"],
            theta_shift_range=theta_shift_range,
            theta_shift_range_student=theta_shift_range_student,
            theta_shift_range_teacher=theta_shift_range_teacher,
            com_centering=com_centering,
            com_search_radius_factor=com_search_radius_factor,
            cj_brightness=cj_brightness,
            cj_contrast=cj_contrast,
            blur_kernel_max=blur_kernel_max,
            blur_sigma_max=blur_sigma_max,
        )
        circular_theta = True

    # For the ViT backbone the circular-theta Conv2d wrapping is moot
    # (there's no conv trunk to wrap), so silently disable it to avoid
    # user confusion.
    if architecture == "vit":
        circular_theta = False
    # Reseed BEFORE model construction so the parameter init is identical
    # regardless of how much RNG was consumed by prior runs in the same
    # Python process. (train_contrastive reseeds again at the start of its
    # training loop; we want the model init itself to be deterministic too.)
    import random as _random
    _random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Under auto-weight, force the starting gamma to 0 regardless of the static
    # value — the trainer flips it at the configured detection epoch.
    initial_gamma = 0.0 if conf_weight_auto is not None else conf_weight_gamma
    model = ContrastiveDINOModel(
        architecture=architecture,
        n_layers=n_layers,
        num_prototypes=num_prototypes,
        contrastive_hidden=projection_hidden,
        contrastive_out=projection_dim,
        center_momentum=float(center_momentum),
        EMA0=float(EMA0), EMAfin=float(EMAfin),
        use_maxpool=True, w_ent=w_ent,
        T0=t0, Tfin=tfin, warmup_frac=float(warmup_frac),
        circular_theta=circular_theta,
        use_centroid_loss=(centroid_lambda > 0.0),
        conf_weight_gamma=initial_gamma,
        target_mode=target_mode,
        sinkhorn_eps=sinkhorn_eps,
        sinkhorn_iters=sinkhorn_iters,
    ).to(device)

    # Fine-tune: optionally warm-start from a previous checkpoint.
    # We accept missing or extra keys so a checkpoint trained at K=6
    # can warm-start a K=8 head (only the matching prototypes load;
    # the new ones stay random — sometimes useful, often you'll want
    # to keep K identical between parent and fine-tune).
    if init_from_checkpoint:
        if not os.path.exists(init_from_checkpoint):
            raise FileNotFoundError(
                f"init_from_checkpoint not found: "
                f"{init_from_checkpoint}")
        try:
            from dino_sr_contrastive_model import (
                load_contrastive_checkpoint)
            print(f"[finetune] warm-starting from "
                   f"{init_from_checkpoint}", flush=True)
            loaded, *_ = load_contrastive_checkpoint(
                init_from_checkpoint, device=device)
            # Copy state-dict tensors that match shape; report drop-ins.
            src = loaded.state_dict()
            tgt = model.state_dict()
            n_loaded = 0; n_skipped = 0; n_missing = 0
            for k, v in src.items():
                if k in tgt and tgt[k].shape == v.shape:
                    tgt[k].copy_(v); n_loaded += 1
                else:
                    n_skipped += 1
            for k in tgt:
                if k not in src: n_missing += 1
            model.load_state_dict(tgt, strict=False)
            print(f"[finetune] state-dict transfer: "
                   f"{n_loaded} keys loaded, {n_skipped} shape-mismatch "
                   f"skipped, {n_missing} target keys not in source.",
                   flush=True)
            del loaded
        except Exception as _e:
            raise RuntimeError(
                f"failed to warm-start from "
                f"{init_from_checkpoint}: {_e}") from _e

    # Fine-tune: optionally freeze the encoder so only the projector
    # + prototype head update. ~5-8× speedup per step + much less
    # representation drift.
    if freeze_encoder:
        n_frozen = 0
        for p in model.student_encoder.parameters():
            p.requires_grad_(False); n_frozen += 1
        for p in model.teacher_encoder.parameters():
            p.requires_grad_(False)
        print(f"[finetune] froze encoder ({n_frozen} student-encoder "
              f"param tensors). Projector + prototypes still trainable.",
              flush=True)

    # Allow explicit override of the entropy gate (used by the confidence-gated
    # ablation without touching the original `entropy_gate` preset handle).
    effective_entropy_gate = entropy_gate if entropy_gate_override is None else bool(entropy_gate_override)
    # Load radial profiles + gate thresholds for SupCon if requested.
    supcon_radials_arr = None
    supcon_tau_pos = 0.5
    supcon_tau_neg = 0.0
    if (supcon_lambda > 0 or cluster1d_lambda > 0
            or lam_spatial > 0) and supcon_radials_path is not None:
        if is_multi:
            # The dataset is the concatenation of cfg["paths"]; the radial
            # sidecar must be concatenated the SAME way (LoadPRZMulti
            # loads cubes in cfg["paths"] order).  A single cube's radial
            # file is N_one rows, not the combined N — concatenate.
            parts = []
            for p in cfg["paths"]:
                base = p[:-4] if p.endswith(".prz") else p
                rp = base + ".radial.npy"
                if not os.path.exists(rp):
                    raise FileNotFoundError(
                        f"[multi] per-cube radial profile missing:\n  {rp}\n"
                        f"A radial-gated variant (SupCon / cluster1d / "
                        f"spatial) needs a `<cube>.radial.npy` for EVERY "
                        f"cube in the multi set.  Generate radials for each "
                        f"cube first (or train a non-radial variant).")
                parts.append(np.load(rp, allow_pickle=True))
            supcon_radials_arr = np.concatenate(parts, axis=0)
            print(f"[multi] combined radials from {len(parts)} cubes -> "
                  f"{supcon_radials_arr.shape}", flush=True)
        else:
            if not os.path.exists(supcon_radials_path):
                raise FileNotFoundError(
                    f"supcon_radials_path not found: {supcon_radials_path}")
            supcon_radials_arr = np.load(supcon_radials_path,
                                              allow_pickle=True)
        if supcon_radials_arr.shape[0] != N:
            raise ValueError(
                f"radial profile count {supcon_radials_arr.shape[0]} != "
                f"dataset N={N}")
        if is_multi:
            # A single cube's gate_thresholds.json is calibrated on that
            # cube alone; the combined set's pair statistics differ, so
            # recalibrate tau_pos/tau_neg from the CONCATENATED radials.
            from compute_radial_profile import calibrate_thresholds
            _th = calibrate_thresholds(supcon_radials_arr)
            supcon_tau_pos = float(_th["tau_pos"])
            supcon_tau_neg = float(_th["tau_neg"])
            print(f"[multi] recalibrated combined thresholds "
                  f"tau_pos={supcon_tau_pos:.4f}  tau_neg={supcon_tau_neg:.4f}  "
                  f"frac_pos={_th.get('actual_frac_pos', 0):.3f}  "
                  f"frac_neg={_th.get('actual_frac_neg', 0):.3f}", flush=True)
        elif supcon_thresholds_path is not None and os.path.exists(
                supcon_thresholds_path):
            with open(supcon_thresholds_path) as _tf:
                _th = json.load(_tf)
            supcon_tau_pos = float(_th["tau_pos"])
            supcon_tau_neg = float(_th["tau_neg"])
            print(f"[supcon] tau_pos={supcon_tau_pos:.4f}  "
                  f"tau_neg={supcon_tau_neg:.4f}  "
                  f"frac_pos={_th.get('actual_frac_pos', 0):.3f}  "
                  f"frac_neg={_th.get('actual_frac_neg', 0):.3f}", flush=True)
        else:
            print(f"[supcon] using default thresholds "
                  f"tau_pos={supcon_tau_pos} tau_neg={supcon_tau_neg}",
                  flush=True)

    # Resolve spatial-gate sentinels.  -2 means "use the same calibrated
    # tau_pos/tau_neg as the radial-gated SupCon loss" (per-sample
    # auto-calibrated from gate_thresholds.json).  Explicit values
    # override.
    if lam_spatial > 0:
        if spatial_tau_pos < -1.0:
            spatial_tau_pos = float(supcon_tau_pos)
        if spatial_tau_neg < -1.0:
            spatial_tau_neg = float(supcon_tau_neg)
        print(f"[L_spatial] tau_pos={spatial_tau_pos:.4f}  "
              f"tau_neg={spatial_tau_neg:.4f}  "
              f"(gate {'inherited from supcon' if (supcon_radials_arr is not None) else 'WILL BE UNGATED — no radials'})",
              flush=True)

    t0_train = time.perf_counter()
    # Resolve effective contrastive_lambda: explicit override > preset.
    eff_contrastive_lambda = (preset["contrastive_lambda"]
                                if contrastive_lambda_override is None
                                else float(contrastive_lambda_override))
    # PRE-FLIGHT loss recipe banner (so the active loss is always visible).
    # Resolve fractional cluster1d schedule into absolute epoch counts.
    cluster1d_warmup_epochs = int(round(cluster1d_warmup_frac * epochs))
    cluster1d_ramp_epochs = int(round(cluster1d_ramp_frac * epochs))

    # Pair-supervised assignment loss: load labels from sidecar JSON
    # (if provided) and convert to (a, b, y) numpy arrays. Disable the
    # term silently if the file is missing or empty so existing recipes
    # without labels are unaffected.
    pair_idx_a = pair_idx_b = pair_y = None
    pair_label_count_total = 0
    pair_label_count_used  = 0  # after timestamp filter
    if pair_labels_path and lambda_pair > 0:
        try:
            from gui_app.pair_labels import (
                load_pair_labels, labels_to_arrays)
            _pl = load_pair_labels(pair_labels_path)
            pairs_all = _pl.get("pairs", [])
            pair_label_count_total = len(pairs_all)
            # Optional timestamp filter (used by Phase-B fine-tune
            # 'only labels added since parent run').
            if pair_labels_min_timestamp:
                _cut = str(pair_labels_min_timestamp)
                pairs_kept = [p for p in pairs_all
                              if str(p.get("t", "")) > _cut]
                _pl_view = dict(_pl); _pl_view["pairs"] = pairs_kept
                print(f"[pair-loss] timestamp filter > {_cut!r}: "
                      f"{len(pairs_kept)}/{len(pairs_all)} pairs "
                      f"kept.", flush=True)
            else:
                _pl_view = _pl
            arrs = labels_to_arrays(_pl_view)
            if arrs is not None:
                pair_idx_a, pair_idx_b, pair_y = arrs
                pair_label_count_used = int(len(pair_y))
                print(f"[pair-loss] using {pair_label_count_used} "
                      f"pairs from {pair_labels_path}", flush=True)
            else:
                print(f"[pair-loss] no pairs in "
                      f"{pair_labels_path} — skipping.", flush=True)
        except Exception as _e:
            print(f"[pair-loss] could not load {pair_labels_path}: "
                   f"{_e!r} — skipping.", flush=True)
    print(f"\n[loss recipe] DINO=1.0  contrastive={eff_contrastive_lambda}  "
          f"centroid={centroid_lambda}  "
          f"supcon={supcon_lambda}@τ={supcon_temperature}  "
          f"proto_repel={proto_repel_lambda}@thr={proto_repel_threshold}  "
          f"cluster1d={cluster1d_lambda}@margin={cluster1d_margin}"
          f"(warmup={cluster1d_warmup_epochs},ramp={cluster1d_ramp_epochs})  "
          f"spatial={lam_spatial}@tau_pos={spatial_tau_pos}/tau_neg={spatial_tau_neg}  "
          f"w_ent={w_ent}  "
          f"conf_weight_gamma_init={initial_gamma}  "
          f"lam_spatial={lam_spatial}", flush=True)

    model, timing = train_contrastive(
        model, dataset,
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        device=device, outdir=outdir, save_every=save_every, seed=seed,
        batch_size=batch_size,
        student_aug=student_aug, teacher_aug=teacher_aug,
        student_roll=student_roll, teacher_roll=teacher_roll,
        contrastive_lambda=eff_contrastive_lambda,
        contrastive_warmup_epochs=warmup_epochs,
        contrastive_ramp_epochs=ramp_epochs,
        contrastive_entropy_gate=effective_entropy_gate,
        centroid_lambda=centroid_lambda,
        centroid_margin=centroid_margin,
        lam_spatial=lam_spatial,
        scan_shape=scan_shape if lam_spatial > 0 else None,
        spatial_tau_pos=float(spatial_tau_pos),
        spatial_tau_neg=float(spatial_tau_neg),
        conf_weight_auto=conf_weight_auto,
        stop_at_epoch=stop_at_epoch,
        supcon_radials=supcon_radials_arr,
        supcon_tau_pos=supcon_tau_pos,
        supcon_tau_neg=supcon_tau_neg,
        supcon_lambda=supcon_lambda,
        supcon_temperature=supcon_temperature,
        proto_repel_lambda=proto_repel_lambda,
        proto_repel_threshold=proto_repel_threshold,
        cluster1d_lambda=cluster1d_lambda,
        cluster1d_lambda_intra=cluster1d_lambda_intra,
        cluster1d_lambda_inter=cluster1d_lambda_inter,
        cluster1d_margin=cluster1d_margin,
        cluster1d_min_cluster_mass=cluster1d_min_cluster_mass,
        cluster1d_warmup_epochs=cluster1d_warmup_epochs,
        cluster1d_ramp_epochs=cluster1d_ramp_epochs,
        pair_idx_a=pair_idx_a, pair_idx_b=pair_idx_b, pair_y=pair_y,
        lambda_pair=float(lambda_pair),
        pair_entropy_reg=float(pair_entropy_reg),
        pair_per_batch=int(pair_per_batch),
        subsample_n=(int(subsample_n) if subsample_n else None),
    )
    print(f"\nTraining done in {time.perf_counter() - t0_train:.0f}s "
          f"({timing['train_time_per_epoch_s']:.1f}s/ep).", flush=True)

    # Merge the entry-time argument snapshot (every knob the caller
    # passed) with the curated derived keys below.  Curated keys win on
    # collision so the existing readers keep seeing the same values.
    _curated_cfg = {
        "contrastive_lambda": preset["contrastive_lambda"],
        "theta_roll_aug": preset["theta_roll_aug"],
        "theta_shift_range_student": student_roll.shift_range,
        "theta_shift_range_teacher": teacher_roll.shift_range,
        "theta_shift_range": (student_roll.shift_range),  # legacy — kept for backward-compat readers
        "pipeline": pipeline,
        "circular_theta": circular_theta,
        "centroid_lambda": centroid_lambda,
        "centroid_margin": centroid_margin,
        "conf_weight_gamma": conf_weight_gamma,
        "conf_weight_auto": conf_weight_auto,
        "target_mode": target_mode,
        "sinkhorn_eps": sinkhorn_eps,
        "sinkhorn_iters": sinkhorn_iters,
        "w_ent": w_ent,
        "com_centering": com_centering,
        "com_search_radius_factor": com_search_radius_factor,
        "aug_disable": list(aug_disable) if aug_disable else None,
        "cj_brightness": cj_brightness,
        "cj_contrast":   cj_contrast,
        "blur_kernel_max": blur_kernel_max,
        "blur_sigma_max":  blur_sigma_max,
        "supcon_lambda": supcon_lambda,
        "supcon_temperature": supcon_temperature,
        "supcon_tau_pos": supcon_tau_pos,
        "supcon_tau_neg": supcon_tau_neg,
        "supcon_radials_path": supcon_radials_path,
        "contrastive_lambda_eff": eff_contrastive_lambda,
        "proto_repel_lambda": proto_repel_lambda,
        "proto_repel_threshold": proto_repel_threshold,
        "cluster1d_lambda": cluster1d_lambda,
        "cluster1d_lambda_intra": cluster1d_lambda_intra,
        "cluster1d_lambda_inter": cluster1d_lambda_inter,
        "cluster1d_margin": cluster1d_margin,
        "cluster1d_min_cluster_mass": cluster1d_min_cluster_mass,
        "cluster1d_warmup_frac": cluster1d_warmup_frac,
        "cluster1d_ramp_frac": cluster1d_ramp_frac,
        "cluster1d_warmup_epochs": cluster1d_warmup_epochs,
        "cluster1d_ramp_epochs": cluster1d_ramp_epochs,
        "pair_labels_path": pair_labels_path,
        "pair_labels_min_timestamp": pair_labels_min_timestamp,
        "pair_label_count": pair_label_count_total,
        "pair_label_count_used": pair_label_count_used,
        "lambda_pair": lambda_pair,
        "pair_entropy_reg": pair_entropy_reg,
        "pair_per_batch": pair_per_batch,
        "init_from_checkpoint": init_from_checkpoint,
        "freeze_encoder": bool(freeze_encoder),
        "subsample_n": (int(subsample_n) if subsample_n else None),
        "contrastive_entropy_gate": effective_entropy_gate,
        "lam_spatial": lam_spatial,
        "spatial_tau_pos": float(spatial_tau_pos),
        "spatial_tau_neg": float(spatial_tau_neg),
        "center_momentum": float(center_momentum),
        "EMA0": float(EMA0),
        "EMAfin": float(EMAfin),
        "warmup_frac": float(warmup_frac),
        "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "weight_decay": weight_decay,
        "num_prototypes": num_prototypes,
        "T0": t0, "Tfin": tfin,
        "warmup_epochs": warmup_epochs, "ramp_epochs": ramp_epochs,
        "entropy_gate": entropy_gate,
        "projection_dim": projection_dim,
        "projection_hidden": projection_hidden,
        "center_mask_radius": mask_r,
        "center_crop_size": center_crop_size,
        "vmax": effective_vmax,
        "polar_size": polar_size, "polar_mask_cols": polar_mask_cols,
        "seed": seed,
    }
    # _arg_snapshot first → curated derived keys override.  This way
    # every parameter in run_config's signature is recorded (even ones
    # not in the curated list, e.g. architecture, n_layers,
    # save_every, stop_at_epoch, entropy_gate_override,
    # supcon_thresholds_path, contrastive_lambda_override, …).
    cfg_full = {**_arg_snapshot, **_curated_cfg}

    # ------------------------------------------------------------------
    # Pre-processing block — every knob that affects WHAT THE MODEL SAW,
    # with explicit names, units, and source.  Lives at the top level
    # of run_summary.json (alongside `cfg`) so post-hoc/eval/GradCAM
    # tools never have to disambiguate "which mask did this run use".
    # ------------------------------------------------------------------
    # vmax source: did the caller pass an explicit arg, or did we fall
    # back to the SAMPLES[sample]["vmax"] default?
    if vmax is None:
        vmax_source = f"sample_default ({sample}.vmax)"
    else:
        vmax_source = "explicit_arg"
    # center_mask_radius source: same logic — None means "use sample
    # default" (which may itself have been set by the GUI pre-panel).
    if center_mask_radius is None:
        mask_source = f"sample_default ({sample}.center_mask_radius)"
    else:
        mask_source = "explicit_arg"
    # Filters that LoadPRZ reads from SAMPLES at __getitem__ time.
    # If the GUI pre-panel set blur / log-stretch on this sample, they
    # were applied to every batch the model saw; record them here.
    _samp = EVAL_SAMPLES.get(sample, {}) if isinstance(EVAL_SAMPLES,
                                                          dict) else {}
    _blur_sigma  = float(_samp.get("blur_sigma", 0.0) or 0.0)
    _log_stretch = bool(_samp.get("log_stretch", False))

    pre_pipeline = {
        "vmax_used":                          float(effective_vmax),
        "vmax_source":                        vmax_source,
        "center_crop_size_192px":             int(center_crop_size),
        "center_mask_radius_cart_192px":      int(mask_r),
        "center_mask_radius_source":          mask_source,
        "polar_mask_cols_low_r_polar_frame":  int(polar_mask_cols),
        "polar_size":                         int(polar_size),
        "com_centering":                      bool(com_centering),
        "com_search_radius_factor":           float(com_search_radius_factor),
        "blur_sigma_loadprz":                 _blur_sigma,
        "log_stretch_loadprz":                _log_stretch,
        # Note on the two masks (cart vs polar) so the JSON itself
        # documents the convention.
        "_legend": {
            "center_mask_radius_cart_192px":
                "Disk radius (px in the 192×192 cart frame) zeroed "
                "BEFORE the polar transform.  Hides the direct beam.",
            "polar_mask_cols_low_r_polar_frame":
                "Number of leftmost columns of the polar image "
                "(= low-radius region) zeroed AFTER the polar "
                "transform.  Back-projects to a dark disk of radius "
                "~ (polar_mask_cols/192) × (center_crop_size / 2) "
                "raw-detector px in cart space.",
        },
    }

    summary = {
        "sample": sample,
        "config": config_key,
        "label": preset["label"],
        "outdir": outdir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pre_pipeline": pre_pipeline,
        "cfg": cfg_full,
        "timing": {
            "train_seconds": timing["train_time_s"],
            "per_epoch_seconds": timing["train_time_per_epoch_s"],
        },
        "asymmetry_report": timing.get("asymmetry_report"),
    }
    # JSON-safe default: numpy scalars → python scalars, paths → str,
    # anything else → str(repr).  The old default=float crashed on
    # non-float values (None survived but strings did not).
    def _json_default(o):
        try:
            import numpy as _np
            if isinstance(o, (_np.integer,)): return int(o)
            if isinstance(o, (_np.floating,)): return float(o)
            if isinstance(o, (_np.ndarray,)): return o.tolist()
        except Exception:
            pass
        try:
            import pathlib as _pl
            if isinstance(o, _pl.PurePath): return str(o)
        except Exception:
            pass
        # Last-resort: stringify so the dump never crashes a run.
        return repr(o)

    with open(os.path.join(outdir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    return summary


# %% post-training evaluation and report
def evaluate_and_report(config_key: str, sample: str, outdir: str, device,
                         reference_assigns: np.ndarray | None = None,
                         ckpt_path: str | None = None):
    """Produce per-config figures and report.md. Returns metrics dict.

    `ckpt_path` defaults to `<outdir>/best.pth`; pass an explicit path for
    transfer-inference runs where the checkpoint lives elsewhere.
    """
    from data import SAMPLES as EVAL_SAMPLES, LoadPRZ
    from dino_sr_contrastive_model import load_contrastive_checkpoint
    import contrastive_eval as CE

    preset = CONFIG_PRESETS[config_key]
    cfg = EVAL_SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    scan_shape = cfg["scan_shape"]

    if ckpt_path is None:
        ckpt_path = os.path.join(outdir, "best.pth")
    model, eval_temp, train_metrics, train_cfg = load_contrastive_checkpoint(
        ckpt_path, device=device)

    eval_dir = os.path.join(outdir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Read training-time polar/mask/com settings from run_summary.json so eval
    # uses identical preprocessing (otherwise eval defaults clobber the
    # actual values used in training -- subtle bug).
    rs_path = os.path.join(outdir, "run_summary.json")
    eval_polar_mask_cols = 30
    eval_center_crop = 140
    eval_polar_size = 192
    eval_center_mask_radius = 15
    eval_com_centering = False
    eval_com_search_radius_factor = 2.0
    if os.path.exists(rs_path):
        with open(rs_path) as _rsf:
            _rs = json.load(_rsf)
        _rcfg = _rs.get("cfg", {})
        eval_polar_mask_cols = int(_rcfg.get("polar_mask_cols", 30))
        eval_center_crop = int(_rcfg.get("center_crop_size", 140))
        eval_polar_size = int(_rcfg.get("polar_size", 192))
        eval_center_mask_radius = int(_rcfg.get("center_mask_radius", 15))
        eval_com_centering = bool(_rcfg.get("com_centering", False))
        eval_com_search_radius_factor = float(_rcfg.get(
            "com_search_radius_factor", 2.0))
    print(f"[eval {config_key}] inferring scan "
          f"(polar_mask_cols={eval_polar_mask_cols}, "
          f"center_crop={eval_center_crop}, mask_r={eval_center_mask_radius}, "
          f"com_centering={eval_com_centering})...", flush=True)
    inf = CE.infer_scan(model, dataset, device, eval_temp=eval_temp,
                         polar_mask_cols=eval_polar_mask_cols,
                         center_crop_size=eval_center_crop,
                         polar_size=eval_polar_size,
                         com_centering=eval_com_centering,
                         com_search_radius_factor=eval_com_search_radius_factor,
                         center_mask_radius=eval_center_mask_radius)
    soft_probs = inf["soft_probs"]
    t_probs = inf["teacher_probs"]
    embeds = inf["embeds"]
    assigns = inf["assigns"]
    original_ids = inf.get("K_original_ids", [])
    K_original = inf.get("K_original", int(soft_probs.shape[1]))

    # Save inference arrays for cross-config comparison. Includes the
    # dense-remap metadata so downstream code can recover original ids.
    np.savez_compressed(os.path.join(eval_dir, "inference.npz"),
                        soft_probs=soft_probs, teacher_probs=t_probs,
                        embeds=embeds, assigns=assigns,
                        K_original_ids=np.asarray(original_ids, dtype=np.int64),
                        K_original=np.int64(K_original))

    # Quantitative metrics.
    K = soft_probs.shape[1]
    ent, eff_k, counts = CE.prototype_usage_entropy(soft_probs)
    intra, inter = CE.intra_inter_cosine(embeds, assigns)
    ratio = intra / inter if (inter and inter != 0 and not np.isnan(inter)) else float("nan")
    knn = CE.knn_purity(embeds, assigns, k=10, max_query=1500)
    cos_mat, centroids = CE.centroid_cosine_matrix(embeds, assigns, K)
    # Dead / near-dead prototypes.
    dead = [int(c) for c in range(K) if counts[c] < max(5, 0.005 * len(assigns))]
    # Pair with highest off-diagonal cos-sim.
    off = cos_mat.copy(); np.fill_diagonal(off, -2.0)
    i_star, j_star = np.unravel_index(np.argmax(off), off.shape)
    most_similar_pair = (int(i_star), int(j_star), float(off[i_star, j_star]))
    # Stripe metric: ratio of spatial-covariance eigenvalues per class.
    stripe_scores = CE.stripe_metric(assigns, scan_shape, K)
    stripe_max = float(max(stripe_scores)) if stripe_scores else 1.0
    stripe_worst_c = int(np.argmax(stripe_scores)) if stripe_scores else -1

    # Radial profile consistency on raw patterns (subsample for speed).
    rng = np.random.default_rng(0)
    N = len(dataset)
    idx = rng.choice(N, size=min(1500, N), replace=False)
    raw_sub = np.stack([dataset.get_raw(int(i)) for i in idx], 0)
    rpc = CE.radial_profile_consistency(raw_sub, assigns[idx])

    # NMI against reference (config a assignments, if provided).
    if reference_assigns is not None and len(reference_assigns) == len(assigns):
        nmi_ref = CE.nmi(assigns, reference_assigns)
    else:
        nmi_ref = float("nan")

    # Figures. Loss curves / training dynamics require training_log.csv,
    # which does not exist for pure transfer-inference runs (no training
    # was done in this outdir). Skip those panels when the log is missing.
    training_log = os.path.join(outdir, "training_log.csv")
    if os.path.exists(training_log):
        CE.plot_loss_curves(training_log,
                             os.path.join(eval_dir, "fig_loss_curves.png"))
        CE.plot_training_dynamics(training_log,
                                   os.path.join(eval_dir, "fig_training_dynamics.png"))
    else:
        print(f"[eval {config_key}] no training_log.csv at {training_log} -- "
              f"skipping loss/dynamics plots (transfer-inference run)",
              flush=True)
    CE.plot_centroid_cosine(cos_mat,
                             os.path.join(eval_dir, "fig_centroid_cosine.png"))
    CE.plot_prototype_usage(counts,
                             os.path.join(eval_dir, "fig_prototype_usage.png"))
    umap_method = CE.plot_umap(embeds, assigns,
                                os.path.join(eval_dir, "fig_umap.png"))
    CE.plot_representative_patterns(dataset, assigns, soft_probs,
                                     os.path.join(eval_dir, "fig_representative.png"),
                                     per_proto=6)
    CE.plot_boundary_samples(dataset, t_probs,
                              os.path.join(eval_dir, "fig_boundary.png"), n=16)
    CE.plot_class_map(assigns, scan_shape,
                       os.path.join(eval_dir, "fig_class_map.png"),
                       original_ids=list(original_ids))
    CE.plot_class_averages(dataset, soft_probs, assigns,
                            os.path.join(eval_dir, "fig_class_averages.png"),
                            N_top=200, original_ids=list(original_ids))
    CE.plot_embedding_norms(embeds,
                             os.path.join(eval_dir, "fig_embedding_norms.png"))
    CE.plot_pairwise_similarity(embeds,
                                 os.path.join(eval_dir, "fig_pairwise_sim.png"))

    # Rotation invariance sanity (always compute — it's informative for (a) too).
    cosines = CE.theta_roll_cosine_distribution(
        model, dataset, device, shift_range=192, batch_size=64, n_batches=4)
    CE.plot_rotation_sanity(cosines,
                             os.path.join(eval_dir, "fig_rotation_sanity.png"))
    rot_median = float(np.median(cosines))
    rot_p10 = float(np.percentile(cosines, 10))
    rot_p90 = float(np.percentile(cosines, 90))

    metrics = {
        "config": config_key,
        "label": preset["label"],
        "NMI_vs_configA": nmi_ref,
        "KNN_purity_k10": knn,
        "intra_class_cosine": intra,
        "inter_class_cosine": inter,
        "intra_over_inter": ratio,
        "radial_profile_consistency": rpc,
        "prototype_usage_entropy": ent,
        "effective_K": eff_k,
        "active_prototypes": int((counts > 0).sum()),
        "dead_prototypes": dead,
        "most_similar_pair": most_similar_pair,
        "rotation_invariance_cos_median": rot_median,
        "rotation_invariance_cos_p10": rot_p10,
        "rotation_invariance_cos_p90": rot_p90,
        "embed_norm_mean": float(np.linalg.norm(embeds, axis=1).mean()),
        "embed_norm_std": float(np.linalg.norm(embeds, axis=1).std()),
        "umap_method": umap_method,
        "train_epochs_per_sec": (1.0 / train_metrics.get("loss", 1.0))
        if False else None,
        "proto_counts": counts.tolist(),
        "stripe_scores": stripe_scores,
        "stripe_max": stripe_max,
        "stripe_worst_prototype": stripe_worst_c,
        "K_original": int(K_original),
        "K_active": int(len(original_ids)) if len(original_ids) else int(K),
        "dense_new_to_old_id": list(original_ids),
    }
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    # GradCAM + Integrated Gradients (auto, every run).
    try:
        import viz_gradcam
        viz_gradcam.run(sample=sample, config=os.path.basename(outdir),
                         n_samples_per_proto=10, device=device,
                         ckpt_path=ckpt_path, run_dir=outdir)
    except Exception as exc:
        print(f"[eval {config_key}] gradcam+IG skipped: {exc!r}")
        import traceback; traceback.print_exc()

    # Write report.md.
    from _report_templates import write_config_report
    write_config_report(config_key, preset, metrics, scan_shape, eval_dir,
                        outdir, train_cfg, train_metrics)
    return metrics


# %% top-level driver
def run_all(sample: str, epochs: int, seed: int, batch_size: int,
            device) -> None:
    os.makedirs(RUNS_ROOT, exist_ok=True)
    sample_dir = os.path.join(RUNS_ROOT, sample)
    os.makedirs(sample_dir, exist_ok=True)

    # Unit test first — non-negotiable per spec item 8.
    unit = run_theta_roll_unit_test(device)
    with open(os.path.join(sample_dir, "unit_test.json"), "w") as f:
        json.dump(unit, f, indent=2)
    if not unit["ok"]:
        raise RuntimeError(
            f"theta-roll unit test failed: interior={unit['interior']:.3e} "
            f">= 1e-4. Circular padding is broken.")

    summaries = {}
    for key in ("a", "b", "c"):
        outdir = os.path.join(sample_dir, f"config_{key}")
        summaries[key] = run_config(
            key, sample=sample, epochs=epochs, seed=seed,
            batch_size=batch_size, lr=3e-4, weight_decay=1e-6,
            num_prototypes=10,
            t0=0.07, tfin=0.07,
            warmup_epochs=20, ramp_epochs=10,
            entropy_gate=False,
            projection_dim=128, projection_hidden=256,
            theta_shift_range=None,
            center_mask_radius=None, center_crop_size=140,
            vmax=None, polar_size=192, polar_mask_cols=30,
            outdir=outdir, device=device,
        )
    # Evaluate. Use config (a) assignments as the NMI reference.
    ref_assigns = None
    all_metrics = {}
    for key in ("a", "b", "c"):
        outdir = os.path.join(sample_dir, f"config_{key}")
        m = evaluate_and_report(key, sample=sample, outdir=outdir,
                                  device=device,
                                  reference_assigns=ref_assigns)
        all_metrics[key] = m
        if key == "a":
            ref_assigns = np.load(os.path.join(outdir, "eval",
                                                    "inference.npz"),
                                       allow_pickle=True)["assigns"]

    # Cross-config compare.
    from _report_templates import write_compare_report
    write_compare_report(sample_dir, all_metrics, summaries)
    print(f"\n[run_all] done. See {sample_dir}/COMPARE.md")


# %% CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="Na007b",
                    help="Sample key from eval_all.SAMPLES (default Na007b)")
    ap.add_argument("--config", default="all",
                    help="One of {a,b,c,all} (default all)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: epochs=10, config=c, no full eval.")
    ap.add_argument("--unit-test-only", action="store_true",
                    help="Only run the theta-roll unit test and exit.")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.unit_test_only:
        unit = run_theta_roll_unit_test(device)
        sys.exit(0 if unit["ok"] else 1)

    if args.smoke:
        # Smoke: theta-roll-on, lambda=0.2 (config c), 10 epochs, no eval.
        unit = run_theta_roll_unit_test(device)
        if not unit["ok"]:
            sys.exit(1)
        sample_dir = os.path.join(RUNS_ROOT, args.sample)
        os.makedirs(sample_dir, exist_ok=True)
        outdir = os.path.join(sample_dir, "smoke_config_c")
        run_config(
            "c", sample=args.sample, epochs=10, seed=args.seed,
            batch_size=args.batch_size, lr=3e-4, weight_decay=1e-6,
            num_prototypes=10,
            t0=0.07, tfin=0.07,
            warmup_epochs=20, ramp_epochs=10, entropy_gate=False,
            projection_dim=128, projection_hidden=256,
            theta_shift_range=None,
            center_mask_radius=None, center_crop_size=140,
            vmax=None, polar_size=192, polar_mask_cols=30,
            outdir=outdir, device=device,
        )
        return 0

    if args.config == "all":
        run_all(args.sample, args.epochs, args.seed, args.batch_size, device)
    else:
        sample_dir = os.path.join(RUNS_ROOT, args.sample)
        os.makedirs(sample_dir, exist_ok=True)
        outdir = os.path.join(sample_dir, f"config_{args.config}")
        run_config(
            args.config, sample=args.sample, epochs=args.epochs, seed=args.seed,
            batch_size=args.batch_size, lr=3e-4, weight_decay=1e-6,
            num_prototypes=10,
            t0=0.07, tfin=0.07,
            warmup_epochs=20, ramp_epochs=10, entropy_gate=False,
            projection_dim=128, projection_hidden=256,
            theta_shift_range=None,
            center_mask_radius=None, center_crop_size=140,
            vmax=None, polar_size=192, polar_mask_cols=30,
            outdir=outdir, device=device,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
