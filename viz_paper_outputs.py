"""viz_paper_outputs.py -- paper-quality outputs per run:

  1. fig_class_map_paper.png             -- class map with adaptive K colorbar
                                              (no wasted slots beyond K_active)
  2. eval/class_averages/p{c}.png        -- per-class average diffraction
                                              (single image, log-stretched
                                              Cartesian, beam-mask applied)
  3. eval/class_examples/p{c}/i{idx}_p{p:.2f}.png ... 100 per class
                                              -- top-100 confidence members,
                                              each saved as its own PNG

All Cartesian display matches the rest of the paper figures
(centerCrop -> resize -> log1p stretch, beam mask from the trained mask_r).
"""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from torchvision.transforms import v2 as T
from torchvision.transforms import InterpolationMode

from data import SAMPLES, LoadPRZ


def _read_train_cfg(run_dir: str):
    rs_path = os.path.join(run_dir, "run_summary.json")
    out = dict(mask_r=15, polar_size=192, center_crop=140)
    if os.path.exists(rs_path):
        with open(rs_path) as f:
            rs = json.load(f)
        rcfg = rs.get("cfg", {})
        out["mask_r"] = int(rcfg.get("center_mask_radius", 15))
        out["polar_size"] = int(rcfg.get("polar_size", 192))
        out["center_crop"] = int(rcfg.get("center_crop_size", 140))
    return out


def _beam_mask_disp(H, W, radius):
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    return ((yy - cy) ** 2 + (xx - cx) ** 2) > radius ** 2


def _log_disp(cart, beam_mask, pct_lo=2.0, pct_hi=99.5):
    ref = cart[beam_mask]
    if ref.size == 0:
        return cart
    lo, hi = np.percentile(ref, pct_lo), np.percentile(ref, pct_hi)
    clipped = np.clip(cart, lo, hi) * beam_mask
    return np.log1p(clipped - lo)


def _adaptive_cmap(K_active: int):
    """Pick discrete cmap with exactly K_active colors."""
    if K_active <= 10:
        base = list(plt.get_cmap("tab10").colors[:K_active])
    elif K_active <= 20:
        base = list(plt.get_cmap("tab20").colors[:K_active])
    else:
        base = [plt.get_cmap("turbo")(i / max(K_active - 1, 1))
                for i in range(K_active)]
    return ListedColormap(base, name=f"K{K_active}")


def render_class_map(run_dir: str, sample: str):
    """Adaptive-K class map. K_active colors only, no padding."""
    cfg = SAMPLES[sample]
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    assigns = inf["assigns"]
    Ny, Nx = cfg["scan_shape"]
    K_active = int(np.unique(assigns).size)
    # densely re-map assignments to 0..K_active-1
    unique_ids = sorted(np.unique(assigns).tolist())
    remap = {old: new for new, old in enumerate(unique_ids)}
    dense = np.array([remap[v] for v in assigns], dtype=np.int32).reshape(Ny, Nx)

    cmap = _adaptive_cmap(K_active)
    norm = BoundaryNorm(np.arange(K_active + 1) - 0.5, K_active)

    aspect = Nx / max(Ny, 1)
    if aspect > 1:
        fig_h = 5.0; fig_w = min(15, fig_h * aspect)
    else:
        fig_w = 5.0; fig_h = min(12, fig_w / aspect)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(dense, cmap=cmap, norm=norm, aspect="equal",
                    interpolation="nearest")
    ax.set_title(f"{sample}  -  prototype assignment "
                  f"(K_active={K_active}, shape={Ny}x{Nx})", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                         ticks=list(range(K_active)), shrink=0.9)
    cbar.ax.set_yticklabels([f"p{old}" for old in unique_ids], fontsize=10)
    cbar.set_label("class", fontsize=11)
    fig.tight_layout()
    out = os.path.join(run_dir, "eval", "fig_class_map_paper.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[class-map] wrote {out}", flush=True)
    return unique_ids


def _build_cart_pre(polar_size, center_crop):
    return T.Compose([
        T.CenterCrop(center_crop),
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ])


def render_class_averages_and_examples(run_dir, sample, n_examples=100,
                                          n_top_for_avg=300):
    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]
    assigns = inf["assigns"]

    tcfg = _read_train_cfg(run_dir)
    cart_pre = _build_cart_pre(tcfg["polar_size"], tcfg["center_crop"])

    SCALE = tcfg["center_crop"] / tcfg["polar_size"]
    eff_cart_r = tcfg["mask_r"] * SCALE
    H = tcfg["polar_size"]
    bm_disp = _beam_mask_disp(H, H, eff_cart_r * H / tcfg["polar_size"])

    avg_dir = os.path.join(run_dir, "eval", "class_averages")
    ex_root = os.path.join(run_dir, "eval", "class_examples")
    os.makedirs(avg_dir, exist_ok=True)
    os.makedirs(ex_root, exist_ok=True)

    K_full = soft_probs.shape[1]
    counts = np.bincount(assigns, minlength=K_full)
    for c in range(K_full):
        idx = np.where(assigns == c)[0]
        if idx.size == 0:
            continue

        scores = soft_probs[idx, c]
        # ----- single-image class average ----- (top-N confidence-weighted)
        n_for_avg = min(n_top_for_avg, len(idx))
        top_avg = idx[np.argsort(-scores)[:n_for_avg]]
        patterns = np.stack([dataset.get_raw(int(i)) for i in top_avg], 0).astype(np.float32)
        w = soft_probs[top_avg, c].astype(np.float32)
        wavg = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
        wavg_norm = np.clip(wavg / max(float(cfg["vmax"]), 1e-6), 0.0, 1.0)
        x_full = (torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0).float())
        x_full = F.interpolate(x_full, size=(H, H), mode="bilinear",
                                align_corners=False)
        x_cart = cart_pre(x_full)[0, 0].cpu().numpy()
        disp = _log_disp(x_cart, bm_disp)

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
        ax.set_title(f"{sample}  p{c}  class average  (N={int(counts[c])})",
                      fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        avg_out = os.path.join(avg_dir, f"p{c}.png")
        fig.savefig(avg_out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # ----- 100 example patterns (top-confidence) -----
        n_ex = min(n_examples, len(idx))
        ex_dir = os.path.join(ex_root, f"p{c}")
        os.makedirs(ex_dir, exist_ok=True)
        # clear previous examples to avoid stale files
        for old in os.listdir(ex_dir):
            try:
                os.remove(os.path.join(ex_dir, old))
            except Exception:
                pass

        top_ex = idx[np.argsort(-scores)[:n_ex]]
        for rank, gi in enumerate(top_ex):
            gi = int(gi)
            raw = dataset.get_raw(gi).astype(np.float32)
            raw_norm = np.clip(raw / max(float(cfg["vmax"]), 1e-6), 0.0, 1.0)
            x_full = (torch.from_numpy(raw_norm).unsqueeze(0).unsqueeze(0).float())
            x_full = F.interpolate(x_full, size=(H, H), mode="bilinear",
                                    align_corners=False)
            x_cart = cart_pre(x_full)[0, 0].cpu().numpy()
            disp = _log_disp(x_cart, bm_disp)
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.imshow(disp, cmap="inferno", aspect="equal", interpolation="nearest")
            ax.set_title(f"p{c}  i={gi}  p={float(soft_probs[gi, c]):.2f}",
                          fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            ex_out = os.path.join(ex_dir,
                                    f"rank{rank:03d}_i{gi:06d}_p{float(soft_probs[gi, c]):.2f}.png")
            fig.savefig(ex_out, dpi=140, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        print(f"[paper-out] p{c}  N={int(counts[c])}  avg + {n_ex} examples",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--n-examples", type=int, default=100)
    args = ap.parse_args()
    render_class_map(args.run_dir, args.sample)
    render_class_averages_and_examples(args.run_dir, args.sample,
                                          n_examples=args.n_examples)


if __name__ == "__main__":
    main()
