"""render_class_averages.py -- per-class mean diffraction patterns.

For every run matching the glob (default: m=0.97 in the latest sweep),
load eval/inference.npz, average raw patterns per class, render a
grid PNG showing all class-average diffraction patterns side-by-side.

A "fake" class produced by half-finished collapse will look mushy /
non-structured / repeated across multiple classes.  A real phase
shows sharp Bragg disks or a distinct radial profile.
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

# UTF-8 stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# Per-sample vmax (display cap).  Same as the sweep spec.
VMAX = {"Na007b": 5.0, "EuInAs_B100": 30.0, "IMC_SI5": 5.0}
CUBE = {
    "Na007b":      "D:/DINOSR/data/Na007b_nbed.cube.npy",
    "EuInAs_B100": "D:/DINOSR/data/EuInAs_B100.cube.npy",
    "IMC_SI5":     "D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
}


def render_one(run_dir: str, max_per_class: int = 256,
                  log_stretch: bool = False):
    inf = os.path.join(run_dir, "eval", "inference.npz")
    if not os.path.exists(inf):
        print(f"[skip] no inference.npz in {run_dir}", flush=True)
        return False
    # Parse sample name from path
    norm = run_dir.replace("\\", "/")
    parts = norm.rstrip("/").split("/")
    # .../_sweep_m_K_*/<sample>/stage*/<run_name>
    if len(parts) < 4: return False
    sample = parts[-3]
    if sample not in CUBE:
        print(f"[skip] unknown sample {sample!r}", flush=True)
        return False
    cube = np.load(CUBE[sample], mmap_mode="r", allow_pickle=True)
    Ny, Nx, H, W = cube.shape
    flat = cube.reshape(-1, H, W)

    d = np.load(inf, allow_pickle=True)
    assigns = np.asarray(d["assigns"])
    soft = np.asarray(d["soft_probs"])
    K = int(soft.shape[1])

    # Weighted average using soft_probs as weights for each prototype.
    # Use the TOP-N most confident patterns per class so the mean is
    # not diluted by low-confidence (mixed) members.
    class_avgs = np.zeros((K, H, W), dtype=np.float32)
    class_counts = np.zeros(K, dtype=int)
    for k in range(K):
        # All patterns hard-assigned to class k, sorted by p(k) desc.
        idx = np.where(assigns == k)[0]
        class_counts[k] = len(idx)
        if len(idx) == 0:
            continue
        # Take top-N most confident.
        if len(idx) > max_per_class:
            scores = soft[idx, k]
            idx = idx[np.argsort(-scores)[:max_per_class]]
        # Average RAW counts, then put it through the SAME pipeline
        # LoadPRZ uses for the model: clip to [0, vmax] / vmax -> [0, 1].
        # This matches what the network actually sees during inference.
        batch = flat[idx].astype(np.float32)
        raw_mean = batch.mean(axis=0)
        vm = float(VMAX[sample])
        norm = np.clip(raw_mean, 0.0, vm) / max(vm, 1e-9)
        if log_stretch:
            norm = (np.log1p(np.clip(norm, 0.0, None) * 50.0)
                      / float(np.log1p(50.0)))
        class_avgs[k] = norm.astype(np.float32)

    # Layout: 5 columns.
    cols = 5
    rows = (K + cols - 1) // cols
    fig_w = 2.0 * cols
    fig_h = 2.2 * rows + 0.6
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                                squeeze=False, dpi=130)
    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        ax.set_xticks([]); ax.set_yticks([])
        if k >= K:
            ax.set_axis_off(); continue
        if class_counts[k] == 0:
            ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                       fontsize=9, transform=ax.transAxes,
                       color="#888")
            ax.set_axis_off()
            continue
        # Now in [0, 1] (post vmax-normalisation / optional log).
        ax.imshow(class_avgs[k], cmap="inferno",
                    vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"p{k}  N={class_counts[k]}",
                       fontsize=8)
    rn = parts[-1]
    tag = "vmax-norm + log1p" if log_stretch else "vmax-norm"
    fig.suptitle(f"{sample} :: {parts[-2]} :: {rn}  "
                   f"(K={K}, top-{max_per_class}/class, "
                   f"vmax={VMAX[sample]}, {tag})",
                   fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(run_dir, "class_averages.png")
    fig.savefig(out_png, dpi=140, facecolor="white",
                  bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_png}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Sweep root or a single run dir.")
    ap.add_argument("--pattern", default="m0.9700_*",
                    help="Glob over run names to filter "
                         "(default: m=0.97 only).")
    ap.add_argument("--max-per-class", type=int, default=256)
    ap.add_argument("--log", action="store_true",
                    help="Apply log1p(50x) stretch post vmax-norm "
                         "(matches the GUI's log-stretch toggle).")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    # If root IS a single run dir
    if os.path.exists(os.path.join(root, "eval", "inference.npz")):
        render_one(root, args.max_per_class, args.log)
        return
    # Otherwise walk the sweep root
    pat = os.path.join(root, "*", "stage*", args.pattern)
    runs = sorted(glob.glob(pat))
    if not runs:
        print(f"[main] no matches for {pat}", flush=True)
        return
    print(f"[main] {len(runs)} matches", flush=True)
    ok = 0
    for r in runs:
        if render_one(r, args.max_per_class, args.log):
            ok += 1
    print(f"\n[main] rendered={ok}/{len(runs)}", flush=True)


if __name__ == "__main__":
    main()
