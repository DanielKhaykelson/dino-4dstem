"""
run_with_scorecard.py — train a DINO-SR run with the scorecard hooked into
the training loop (Stage 1 of the evaluation-optimization pipeline).

Behavior per sample:
  1. Build the model, dataset, and training transforms.
  2. Build a scorecard callback that precomputes virtual HAADF once and then
     scores every `eval_every` epochs (default 10) from `eval_min_epoch` on.
  3. Call train_ablation with the callback wired in. Training saves ONLY:
       - latest.pth        : most recent epoch (for resume)
       - best.pth          : epoch with the highest scorecard overall
       - ckpt_ep*.pth      : periodic snapshots every --save-every epochs
       - scorecard_log.json: full trajectory
  4. After training, run the full standalone scorecard on best.pth and write
     scorecard.json + scorecard.png to the run's eval/ dir.

New in v4:
  - --lam-sep (default 0.05): prototype-orthogonality regularizer. 0 disables.
  - Single best.pth (was: 11 competing best_*.pth files).
  - Flat-top EffK score (was: bell curve peaking at K/2).
  - w_ent default 0 (unchanged, just flagging: uniform-usage pressure is off).

Outputs go to `DINO_PAPER_SC/{sample}/L1/{label}/` by default to keep the
pristine `DINO_PAPER_V2/` approved runs untouched during development.

CLI:
  python run_with_scorecard.py --sample Na007b \
      --label v4_test --t0 0.04 --tfin 0.05 --epochs 30 --lam-sep 0.05
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SC_OUT_ROOT = os.path.join(BASE_DIR, "DINO_PAPER_SC")

# Lazy imports so --help works without a full torch stack initialization.


def _build_run(sample: str, label: str, *,
               t0: float, tfin: float, vmax: float | None,
               center_mask_radius: int | None,
               num_prototypes: int, epochs: int, batch_size: int,
               center_crop_size: int, eval_every: int, eval_min_epoch: int,
               lam_sep: float, sep_margin: float,
               lam_spatial: float, adjust_t: bool,
               use_polar: bool, use_elliptical: bool,
               polar_mask_cols: int,
               disable_aug: list | None,
               seed: int, outdir: str, device) -> dict:
    """Prepare everything and run training. Returns a summary dict."""
    from eval_all import SAMPLES as EVAL_SAMPLES, LoadPRZ
    from dino_sr_ablation import (AblationDINOModelSR, train_ablation,
                                    get_ablation_transforms,
                                    load_ablation_checkpoint)
    from scorecard import (SAMPLE_TYPES, make_scorecard_callback,
                            compute_scorecard_from_arrays, save_scorecard_png,
                            load_settings)

    if sample not in EVAL_SAMPLES:
        raise KeyError(f"{sample} not in eval_all.SAMPLES; "
                       f"available: {list(EVAL_SAMPLES)}")

    cfg = dict(EVAL_SAMPLES[sample])
    cfg.update(SAMPLE_TYPES.get(sample, {"sample_type": "non_layered"}))
    effective_vmax = vmax if vmax is not None else cfg["vmax"]
    scan_shape = cfg["scan_shape"]
    mask_r = (center_mask_radius if center_mask_radius is not None
              else cfg.get("center_mask_radius", 15))
    sample_type = cfg["sample_type"]
    layer_bounds = cfg.get("layer_bounds")

    # Elliptical correction: fit ellipse on PACBED once, pass the affine
    # matrix into the aug pipeline for use at every training/eval step.
    ellipse_affine = None
    ellipse_meta = None

    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, "run_summary.json")

    print(f"\n{'#' * 72}")
    print(f"# {sample} / {label}")
    print(f"# output   : {outdir}")
    print(f"# config   : vmax={effective_vmax} T0={t0} Tfin={tfin} "
          f"num_proto={num_prototypes} epochs={epochs} "
          f"lam_sep={lam_sep} lam_spatial={lam_spatial} adjust_t={adjust_t} "
          f"sample_type={sample_type} layer_bounds={layer_bounds}")
    print(f"# scorecard: every {eval_every} epochs starting ep{eval_min_epoch}")
    print(f"{'#' * 72}\n", flush=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"Loading dataset from {cfg['path']} ...", flush=True)
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=effective_vmax)
    N = len(dataset)
    assert N == scan_shape[0] * scan_shape[1], \
        f"dataset size {N} != scan_shape {scan_shape} product"

    # ── Elliptical fit (once per sample) ─────────────────────────────────
    if use_elliptical:
        from elliptical_correction import compute_pacbed, fit_ellipse_affine
        print("Fitting elliptical distortion on PACBED ...", flush=True)
        pacbed = compute_pacbed(dataset, n_samples=1000, seed=seed)
        ellipse_affine, ellipse_meta = fit_ellipse_affine(pacbed, mask_radius=mask_r)
        print(f"  axis ratio b/a = {ellipse_meta['axis_ratio']:.4f}  "
              f"phi = {ellipse_meta['phi_deg']:+.2f} deg  "
              f"(1 = perfectly circular; smaller = more elliptical)")

    # ── Model ────────────────────────────────────────────────────────────────
    model = AblationDINOModelSR(
        n_layers=1, num_prototypes=num_prototypes,
        center_momentum=0.9, EMA0=0.990, EMAfin=0.999,
        use_maxpool=True, w_ent=0.0,
        T0=t0, Tfin=tfin, warmup_frac=0.2,
    ).to(device)

    student_aug, teacher_aug = get_ablation_transforms(
        disable=disable_aug, center_mask_radius=mask_r,
        center_crop_size=center_crop_size,
        use_polar=use_polar,
        ellipse_affine=ellipse_affine,
        polar_mask_cols=polar_mask_cols)

    # ── Scorecard callback ───────────────────────────────────────────────────
    t_cb = time.perf_counter()
    scorecard_cb = make_scorecard_callback(
        dataset, scan_shape, sample_type,
        layer_bounds=layer_bounds,
        num_prototypes=num_prototypes,
        center_mask_radius=mask_r,
        center_crop_size=center_crop_size,
        batch_size=batch_size, device=device,
        sample_name=sample, run_label=label,
        use_polar=use_polar,
        ellipse_affine=ellipse_affine,
        polar_mask_cols=polar_mask_cols,
    )
    print(f"Scorecard callback ready in {time.perf_counter() - t_cb:.1f}s "
          f"(virtual HAADF cached).", flush=True)

    # ── Train ────────────────────────────────────────────────────────────────
    # Pass scan_shape only when L_spatial is active (avoids unnecessary
    # sampling of the spatial tile when the Potts term isn't being used).
    training_scan_shape = scan_shape if lam_spatial > 0.0 else None
    t0_train = time.perf_counter()
    model, timing = train_ablation(
        model, dataset,
        epochs=epochs, device=device,
        outdir=outdir, save_every=10, seed=seed, batch_size=batch_size,
        student_aug=student_aug, teacher_aug=teacher_aug,
        scan_shape=training_scan_shape,
        center_mask_radius=mask_r, center_crop_size=center_crop_size,
        eval_callback=scorecard_cb,
        eval_every=eval_every, eval_min_epoch=eval_min_epoch,
        lam_sep=lam_sep, sep_margin=sep_margin,
        lam_spatial=lam_spatial,
        adjust_t=adjust_t,
    )
    train_elapsed = time.perf_counter() - t0_train
    print(f"\nTraining done in {train_elapsed:.0f}s "
          f"({timing['train_time_per_epoch_s']:.1f}s/ep).", flush=True)

    # ── Final verdict on best.pth ────────────────────────────────────────────
    best_path = os.path.join(outdir, "best.pth")
    final_result = None
    if os.path.exists(best_path):
        print("\nRunning final scorecard on best.pth ...", flush=True)
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        if "scorecard" in ckpt:
            final_result = ckpt["scorecard"]
            print(f"  best.pth (ep {ckpt.get('epoch','?')})  "
                  f"overall={final_result['overall']:.3f}  "
                  f"verdict={final_result['verdict']}  "
                  f"weakest={final_result.get('weakest_component','?')}",
                  flush=True)
    else:
        print("\nNo best.pth was saved (callback never fired or errored).",
              flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "sample": sample, "label": label, "outdir": outdir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "vmax": float(effective_vmax),
            "T0": t0, "Tfin": tfin,
            "num_prototypes": num_prototypes,
            "epochs": epochs, "batch_size": batch_size,
            "center_mask_radius": mask_r,
            "center_crop_size": center_crop_size,
            "seed": seed,
            "scan_shape": list(scan_shape),
            "sample_type": sample_type,
            "layer_bounds": layer_bounds,
        },
        "timing": {
            "train_seconds": timing.get("train_time_s"),
            "per_epoch_seconds": timing.get("train_time_per_epoch_s"),
            "total_seconds": train_elapsed,
        },
        "scorecard_trajectory": timing.get("scorecard_log", []),
        "best_scorecard_overall": timing.get("best_scorecard_overall"),
        "final_verdict": final_result,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote {summary_path}", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, help="Sample name from eval_all.SAMPLES")
    ap.add_argument("--label",  required=True, help="Run label (output subdir)")
    ap.add_argument("--t0",     type=float, default=0.04)
    ap.add_argument("--tfin",   type=float, default=0.05)
    ap.add_argument("--vmax",   type=float, default=None,
                    help="Override vmax from SAMPLES (default: use sample default)")
    ap.add_argument("--center-mask-radius", type=int, default=None,
                    help="Override center_mask_radius from SAMPLES")
    ap.add_argument("--num-prototypes", type=int, default=10)
    ap.add_argument("--epochs",   type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--center-crop", type=int, default=135)
    ap.add_argument("--eval-every",     type=int, default=10)
    ap.add_argument("--eval-min-epoch", type=int, default=10)
    ap.add_argument("--lam-sep",        type=float, default=0.05,
                    help="Weight for prototype-orthogonality regularizer (0 disables)")
    ap.add_argument("--sep-margin",     type=float, default=0.3,
                    help="Cosine margin above which prototype pairs are penalized")
    ap.add_argument("--lam-spatial",    type=float, default=0.0,
                    help="Weight for local-Potts spatial regularizer (0 disables). "
                         "Recommended 0.1 for fragmented class maps.")
    ap.add_argument("--adjust-t",       action="store_true",
                    help="Enable scorecard-driven T_fin adjustment during training")
    ap.add_argument("--polar",          action="store_true",
                    help="Apply polar transform as the last step in both "
                         "student and teacher augmentation pipelines. Peak "
                         "d-spacing maps to column position; rotation maps "
                         "to row shift.")
    ap.add_argument("--elliptical",     action="store_true",
                    help="Fit a per-sample elliptical distortion on PACBED "
                         "and pre-warp every pattern to circularize the "
                         "rings BEFORE aug/polar. Makes rotation a true "
                         "symmetry in Cartesian space.")
    ap.add_argument("--polar-mask-cols", type=int, default=20,
                    help="When --polar is on, zero the first N columns of "
                         "the polar output (replaces Cartesian CenterMask). "
                         "0 disables (useful for ablation). Default 20 ≈ "
                         "mask radius ~10 px, covers the direct-beam + "
                         "low-r off-center wave region.")
    ap.add_argument("--disable",        type=str, default="",
                    help="Comma-separated aug names to disable: "
                         "hflip, vflip, colorjitter, rotation, blur, centermask")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--outdir", default=None,
                    help="Override output dir (default: DINO_PAPER_SC/{sample}/L1/{label})")
    args = ap.parse_args(argv)

    # Default outdir encodes T0/Tfin in the folder name so the run is
    # self-documenting: "v5_polar_T040_050" = label "v5_polar" at T0=0.040, Tfin=0.050.
    # Pass --outdir explicitly to override.
    default_label = f"{args.label}_T{int(round(args.t0*1000)):03d}_{int(round(args.tfin*1000)):03d}"
    outdir = args.outdir or os.path.join(
        SC_OUT_ROOT, args.sample, "L1", default_label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    _build_run(
        args.sample, args.label,
        t0=args.t0, tfin=args.tfin, vmax=args.vmax,
        center_mask_radius=args.center_mask_radius,
        num_prototypes=args.num_prototypes,
        epochs=args.epochs, batch_size=args.batch_size,
        center_crop_size=args.center_crop,
        eval_every=args.eval_every, eval_min_epoch=args.eval_min_epoch,
        lam_sep=args.lam_sep, sep_margin=args.sep_margin,
        lam_spatial=args.lam_spatial,
        adjust_t=args.adjust_t,
        use_polar=args.polar,
        use_elliptical=args.elliptical,
        polar_mask_cols=args.polar_mask_cols,
        disable_aug=[s.strip() for s in args.disable.split(",") if s.strip()] or None,
        seed=args.seed, outdir=outdir, device=device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
