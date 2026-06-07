"""interpret_imc_averages.py — per-DINO-class mean diffraction pattern +
azimuthal radial profile.  If classes separate cleanly in their radial
profile, the model is using the radial signature.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = sys.argv[1] if len(sys.argv) > 1 else (
    r"runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60")
CUBE = r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"
OUT = os.path.join(RUN, "_interpretability")
os.makedirs(OUT, exist_ok=True)


def main():
    assigns = np.load(os.path.join(RUN, "eval", "inference.npz"),
                      allow_pickle=True)["assigns"].astype(int)
    cube = np.load(CUBE, mmap_mode="r")            # (Ny,Nx,H,W)
    Ny, Nx, H, W = cube.shape
    K = assigns.max() + 1
    assn = assigns.reshape(Ny, Nx)
    sums = np.zeros((K, H, W), np.float64)
    cnts = np.zeros(K, np.int64)
    for rx in range(Ny):
        row = np.asarray(cube[rx], dtype=np.float32)   # (Nx,H,W)
        for ry in range(Nx):
            c = int(assn[rx, ry])
            sums[c] += row[ry]; cnts[c] += 1
    means = np.array([sums[c] / max(cnts[c], 1) for c in range(K)])

    # radial profiles
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W))
    rb = np.clip(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int),
                 0, min(H, W) // 2 - 1)
    nb = rb.max() + 1
    npix = np.bincount(rb.ravel(), None, nb)
    profs = np.array([np.bincount(rb.ravel(), means[c].ravel(), nb)
                       / np.maximum(npix, 1) for c in range(K)])
    np.save(os.path.join(OUT, "class_mean_radial_profiles.npy"), profs)

    # ---- figure: class-mean patterns grid ----
    ncol = 5; nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for c in range(K):
        ax = axes[c]
        img = np.log1p(np.clip(means[c], 0, None))
        out = img[rb > 10]
        vmax = float(np.percentile(out, 99.5)) if out.size else img.max()
        ax.imshow(img, cmap="inferno", vmax=max(vmax, 1e-6),
                  interpolation="nearest", aspect="equal")
        ax.set_title(f"class {c}  ({cnts[c]}px)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for c in range(K, len(axes)):
        axes[c].axis("off")
    fig.suptitle("DINO IMC_SI5 — per-class mean diffraction", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "class_mean_patterns.png"), dpi=150)
    plt.close(fig)

    # ---- figure: overlaid radial profiles (lin-log, beam trimmed) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = max(int(0.10 * nb), 8)
    cmap = plt.get_cmap("tab20")
    for c in range(K):
        ax.semilogy(np.arange(lo, nb), np.clip(profs[c, lo:], 1e-3, None),
                    color=cmap(c % 20), lw=1.3, label=f"c{c}")
    ax.set_xlabel("radial bin (px, beam trimmed)")
    ax.set_ylabel("mean I(r)  (log)")
    ax.set_title("per-class mean radial profile — do classes separate "
                 "by radial signature?")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "class_radial_profiles.png"), dpi=150)
    plt.close(fig)
    print(f"[averages] K={K} classes, sizes={cnts.tolist()} -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
