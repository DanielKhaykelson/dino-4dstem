"""
viz_pipeline.py — visualize what the student and teacher actually see, plus
per-prototype class averages (mean diffraction patterns).

Produces two figures under `runs/<sample>/<config>/eval/`:

  1. `fig_view_examples.png`
     - 8 rows = 8 sample indices (one per prototype when possible)
     - 4 cols = [raw Cartesian, student Cartesian view, teacher Cartesian view,
                 student-then-polar, teacher-then-polar]
     The Cartesian view comes from the Cartesian-only prefix of the aug
     pipeline (CenterCrop + flips + ColorJitter + Blur + Resize for student;
     CenterCrop + Resize for teacher). The polar view adds PolarTransform
     + PolarMaskLeft + PolarThetaRoll on top. This makes the aug effects
     directly readable: flips / intensity / blur visible in Cartesian,
     theta-shift visible in polar.

  2. `fig_class_averages.png`
     - One row per prototype, N_show columns showing:
         - mean raw Cartesian pattern (across all scan positions assigned
           to that prototype)
         - median raw Cartesian pattern
         - confidence-weighted mean (weights = teacher softmax prob)
     Prototypes are plotted in descending size order for readability.

Usage:
    python viz_pipeline.py --sample Na007b --config config_c_K6_asym
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

from data import SAMPLES, LoadPRZ
from dino_sr_contrastive_model import (
    PolarTransform, PolarMaskLeft, PolarThetaRoll,
)


# =========================================================================
# 1. Rebuild the student/teacher aug pipelines in SPLIT form.
# =========================================================================

def _build_split_pipelines(center_crop_size: int = 140,
                            polar_size: int = 192,
                            polar_mask_cols: int = 30,
                            theta_shift_range_student: int = 192,
                            theta_shift_range_teacher: int = 16,
                            seed: int = 0):
    """Build Cartesian-only and full (Cartesian + polar + roll) pipelines
    for both student and teacher. Returns a dict so the caller can run
    each stage independently on the same input.
    """
    # Deterministic flips/colorjitter/blur so repeated calls give the same
    # aug — handy for visualization. Since we want DIFFERENT looks, we'll
    # seed with `seed` before each stage call in the viz driver.
    student_cart = T.Compose([
        T.CenterCrop(center_crop_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 0.3)),
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR,
                 antialias=True),
    ])
    teacher_cart = T.Compose([
        T.CenterCrop(center_crop_size),
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR,
                 antialias=True),
    ])
    polar = T.Compose([
        PolarTransform(output_size=polar_size),
        PolarMaskLeft(k_cols=polar_mask_cols),
    ])
    student_roll = PolarThetaRoll(shift_range=theta_shift_range_student)
    teacher_roll = PolarThetaRoll(shift_range=theta_shift_range_teacher)
    return dict(
        student_cart=student_cart,
        teacher_cart=teacher_cart,
        polar=polar,
        student_roll=student_roll,
        teacher_roll=teacher_roll,
    )


# =========================================================================
# 2. Pick one representative sample per prototype (highest confidence).
# =========================================================================

def _pick_one_per_prototype(soft_probs: np.ndarray, assigns: np.ndarray,
                              K: int) -> list[int]:
    picks = []
    for c in range(K):
        mask = (assigns == c)
        if not mask.any():
            picks.append(-1)
            continue
        idx = np.where(mask)[0]
        scores = soft_probs[idx, c]
        picks.append(int(idx[np.argmax(scores)]))
    return picks


# =========================================================================
# 3. Figure 1 — view-example gallery
# =========================================================================

def plot_view_examples(dataset, soft_probs, assigns, out_path: str,
                        center_crop_size=140, polar_size=192,
                        polar_mask_cols=30,
                        student_roll=192, teacher_roll=16,
                        seed: int = 0):
    import matplotlib.pyplot as plt

    K = soft_probs.shape[1]
    picks = _pick_one_per_prototype(soft_probs, assigns, K)
    picks = [i for i in picks if i >= 0]
    # If fewer than 8, pad with random picks.
    rng = np.random.default_rng(seed)
    while len(picks) < 8:
        picks.append(int(rng.integers(len(dataset))))
    picks = picks[:8]

    pipes = _build_split_pipelines(
        center_crop_size=center_crop_size, polar_size=polar_size,
        polar_mask_cols=polar_mask_cols,
        theta_shift_range_student=student_roll,
        theta_shift_range_teacher=teacher_roll,
        seed=seed)

    n_rows = len(picks)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.1, n_rows * 2.1))
    col_titles = ["raw Cartesian",
                  f"student Cartesian\n(flips + CJ + blur)",
                  "teacher Cartesian\n(crop + resize)",
                  f"student polar + θ-roll\n(range ±{student_roll//2} rows)",
                  f"teacher polar + θ-roll\n(range ±{teacher_roll//2} rows)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8)

    for r, idx in enumerate(picks):
        t = dataset[idx]
        x = t if torch.is_tensor(t) else torch.from_numpy(t)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.float()
        # col 0: raw (post-LoadPRZ normalize, pre-aug).
        axes[r, 0].imshow(x[0].numpy(), cmap="gray")
        # col 1: student Cartesian
        torch.manual_seed(seed + r * 7 + 1)
        sc = pipes["student_cart"](x)
        axes[r, 1].imshow(sc[0].numpy(), cmap="gray")
        # col 2: teacher Cartesian
        torch.manual_seed(seed + r * 7 + 2)
        tc = pipes["teacher_cart"](x)
        axes[r, 2].imshow(tc[0].numpy(), cmap="gray")
        # col 3: student polar + roll
        torch.manual_seed(seed + r * 7 + 3)
        sc2 = pipes["student_cart"](x)
        sp = pipes["polar"](sc2)
        sp = pipes["student_roll"](sp.unsqueeze(0))[0]
        sh = int(pipes["student_roll"].last_shifts[0].item())
        axes[r, 3].imshow(sp[0].numpy(), cmap="gray", aspect="auto")
        axes[r, 3].text(0.02, 0.96, f"shift = {sh:+d} rows ({sh*360/polar_size:+.0f}°)",
                        transform=axes[r, 3].transAxes, color="yellow",
                        fontsize=7, va="top", ha="left",
                        bbox=dict(facecolor="black", alpha=0.6, pad=1))
        # col 4: teacher polar + roll
        torch.manual_seed(seed + r * 7 + 4)
        tc2 = pipes["teacher_cart"](x)
        tp = pipes["polar"](tc2)
        tp = pipes["teacher_roll"](tp.unsqueeze(0))[0]
        th = int(pipes["teacher_roll"].last_shifts[0].item())
        axes[r, 4].imshow(tp[0].numpy(), cmap="gray", aspect="auto")
        axes[r, 4].text(0.02, 0.96, f"shift = {th:+d} rows ({th*360/polar_size:+.0f}°)",
                        transform=axes[r, 4].transAxes, color="yellow",
                        fontsize=7, va="top", ha="left",
                        bbox=dict(facecolor="black", alpha=0.6, pad=1))
        axes[r, 0].set_ylabel(f"p{assigns[idx]}  i={idx}", fontsize=7)
        for j in range(n_cols):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    fig.suptitle("student vs teacher views of the same pattern "
                  "(each row is one sample, seeds independent per stage)",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# 4. Figure 2 — per-prototype class averages
# =========================================================================

def plot_class_averages(dataset, soft_probs, teacher_probs, assigns,
                          out_path: str, N_top: int = 200,
                          beam_mask_radius: int = 40,
                          pct_lo: float = 2.0, pct_hi: float = 99.5):
    """For each prototype, show mean / median / confidence-weighted mean
    raw Cartesian pattern of the top-N_top nearest-to-centroid members.

    Scaling: the direct beam is many orders of magnitude brighter than any
    Bragg feature, so a naive imshow washes out the diffraction structure.
    We (1) zero the central disk (radius `beam_mask_radius` px in raw pixel
    units -- 40 px is well beyond the typical 15 px beam in these samples
    so we cover the near-beam wings too), then (2) clip the display range
    to the [pct_lo, pct_hi] percentile band of the masked pattern. This
    stretches contrast onto the diffuse + Bragg structure.
    """
    import matplotlib.pyplot as plt

    K = soft_probs.shape[1]
    counts = np.bincount(assigns, minlength=K)
    order = np.argsort(-counts)

    # Build a single beam mask sized for the raw patterns.
    H0, W0 = dataset.H, dataset.W
    yy, xx = np.ogrid[:H0, :W0]
    cy, cx = H0 / 2.0, W0 / 2.0
    bmask = ((yy - cy) ** 2 + (xx - cx) ** 2) > beam_mask_radius ** 2  # 1 outside beam

    def _disp(arr):
        """Beam-mask + percentile-clip + log1p for display."""
        masked = arr * bmask
        vals = masked[bmask]
        if vals.size == 0:
            return masked
        lo = np.percentile(vals, pct_lo)
        hi = np.percentile(vals, pct_hi)
        clipped = np.clip(masked, lo, hi)
        # log1p on (clipped - lo) keeps Bragg/ring contrast visible without
        # crushing the low end. Result is [0, log(1 + hi - lo)].
        return np.log1p(clipped - lo)

    fig, axes = plt.subplots(K, 3, figsize=(6.5, K * 2.2))
    if K == 1:
        axes = axes[None, :]
    col_titles = [f"mean (top {N_top})",
                  f"median (top {N_top})",
                  f"conf-weighted mean (top {N_top})"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=9)

    for row_idx, c in enumerate(order):
        mask = (assigns == c)
        if not mask.any():
            for j in range(3):
                axes[row_idx, j].set_axis_off()
            axes[row_idx, 0].set_ylabel(f"p{c} (0)",
                                         fontsize=7, rotation=0,
                                         labelpad=18, ha="right", va="center")
            continue
        idx = np.where(mask)[0]
        scores = soft_probs[idx, c]
        top = idx[np.argsort(-scores)[:min(N_top, len(idx))]]
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0)
        weights = soft_probs[top, c]
        mean = patterns.mean(axis=0)
        median = np.median(patterns, axis=0)
        wmean = (patterns * weights[:, None, None]).sum(axis=0) / (weights.sum() + 1e-12)
        for j, arr in enumerate((mean, median, wmean)):
            axes[row_idx, j].imshow(_disp(arr), cmap="inferno")
            axes[row_idx, j].set_xticks([]); axes[row_idx, j].set_yticks([])
        axes[row_idx, 0].set_ylabel(f"p{c}\n(N={int(counts[c])})",
                                     fontsize=7, rotation=0,
                                     labelpad=22, ha="right", va="center")
    fig.suptitle(f"per-prototype class averages "
                  f"(raw Cartesian, beam-masked r<{beam_mask_radius}px, "
                  f"{pct_lo:.0f}-{pct_hi:.0f} pct clip, log1p)",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# 5. Driver
# =========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="Na007b")
    ap.add_argument("--config", default="config_c_K6_asym")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-top", type=int, default=200,
                    help="Top-N highest-confidence samples per class to average.")
    ap.add_argument("--student-shift", type=int, default=None,
                    help="Override student theta shift range (useful for runs whose "
                         "run_summary.json predates the per-view plumbing).")
    ap.add_argument("--teacher-shift", type=int, default=None,
                    help="Override teacher theta shift range.")
    args = ap.parse_args(argv)

    cfg = SAMPLES[args.sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "runs", args.sample, args.config)

    data = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft_probs = data["soft_probs"]
    t_probs = data["teacher_probs"]
    assigns = data["assigns"]

    # Pull asymmetric shift ranges from the run summary so visualization
    # matches exactly what the model saw during training.
    with open(os.path.join(run_dir, "run_summary.json")) as f:
        s = json.load(f)
    # Runs launched before the asymmetric-arg plumbing recorded a single
    # symmetric `theta_shift_range`. Fall back gracefully.
    student_shift = s["cfg"].get("theta_shift_range_student",
                                  s["cfg"].get("theta_shift_range", 192))
    teacher_shift = s["cfg"].get("theta_shift_range_teacher",
                                  s["cfg"].get("theta_shift_range", 192))
    if student_shift is None:
        student_shift = 192
    if teacher_shift is None:
        teacher_shift = 16  # asymmetric default when the field is absent
    if args.student_shift is not None:
        student_shift = args.student_shift
    if args.teacher_shift is not None:
        teacher_shift = args.teacher_shift
    print(f"[viz_pipeline] student_shift={student_shift}, teacher_shift={teacher_shift}")

    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    plot_view_examples(
        dataset, soft_probs, assigns,
        out_path=os.path.join(eval_dir, "fig_view_examples.png"),
        student_roll=student_shift, teacher_roll=teacher_shift,
        seed=args.seed,
    )
    plot_class_averages(
        dataset, soft_probs, t_probs, assigns,
        out_path=os.path.join(eval_dir, "fig_class_averages.png"),
        N_top=args.n_top,
    )
    print(f"[viz_pipeline] wrote {eval_dir}/fig_view_examples.png")
    print(f"[viz_pipeline] wrote {eval_dir}/fig_class_averages.png")


if __name__ == "__main__":
    main()
