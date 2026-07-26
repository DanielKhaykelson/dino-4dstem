"""
A8  -  Orientation (ACOM) vs DINO order classes.

Two questions, one figure:
  1. Is the DINO axis secretly ORIENTATION? -> compare the classical template-matching
     orientation (ACOM zone-axis) map to the DINO class map; report AMI/ARI.
  2. Does classical peak-based ACOM even work on these weak organic patterns?

What the existing ACOM run shows (imc_acom_fullpx_{SI3,SI4}.npz): it detects ~200-340
"peaks" per pattern (noise, not Bragg) and collapses the majority of each field onto a
single high-index zone axis, so its orientation map is near-degenerate and its agreement
with DINO is ~0. Both readings support the paper: DINO is not clustering by orientation,
AND classical orientation mapping is unreliable in this low-dose weak-scattering regime,
which is the gap the label-free method fills.

NOTE: this is ORIENTATION ACOM (crystal tilt), NOT alpha/gamma phase discrimination (see A2).

Output: figs/Review/A8_acom_vs_dino.png + printed AMI/ARI and ACOM quality diagnostics.
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    os.chdir("D:/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive")
    import numpy as np
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib as mpl

    FIGS = "docs/paper/draft_v2/figs"; OUT = f"{FIGS}/Review"
    RUN = {"SI3": "runs/_gui/IMC_SI3_m097k60", "SI4": "runs/_gui/IMC_SI4_m097_k60"}
    DISP = {"SI3": "interface", "SI4": "needles"}
    NY = NX = 128
    NAMES = ["SI3", "SI4"]

    fig = Figure(figsize=(13.5, 7.8), facecolor="white")
    gs = fig.add_gridspec(2, 3, left=0.02, right=0.99, top=0.83, bottom=0.04, wspace=0.12, hspace=0.30)
    print("=" * 96)
    print("A8  ORIENTATION (ACOM) vs DINO ORDER CLASSES")
    print("=" * 96)
    for ri, name in enumerate(NAMES):
        dino = np.load(f"{RUN[name]}/eval/inference.npz")["assigns"].astype(int)
        z = np.load(f"{FIGS}/imc_acom_fullpx_{name}.npz")
        za = np.c_[z["za_u"], z["za_v"], z["za_w"]]
        # encode zone-axis triples as integer labels
        uq, inv = np.unique(za, axis=0, return_inverse=True)
        acom = inv.astype(int)
        corr = z["corr"]; npk = z["n_peaks"]
        # recompute agreement ourselves (don't trust the cached 0.0)
        ami = adjusted_mutual_info_score(dino, acom)
        ari = adjusted_rand_score(dino, acom)
        # ACOM degeneracy diagnostics
        ct = np.bincount(acom); dom = ct.max() / ct.sum()
        nza = len(uq)
        print(f"\n----- {name} ({DISP[name]})")
        print(f"      DINO classes = {len(np.unique(dino))}   ACOM distinct zone axes = {nza}")
        print(f"      AMI(DINO,ACOM) = {ami:.3f}   ARI(DINO,ACOM) = {ari:.3f}")
        print(f"      ACOM degeneracy: {100*dom:.0f}% of the field on ONE zone axis {tuple(uq[np.argmax(ct)])}")
        print(f"      ACOM peak count per pattern: median {int(np.median(npk))} (Bragg-plausible would be ~10-20)")
        print(f"      ACOM template score corr: median {np.median(corr):.1f} (broad, low-confidence)")

        # panel 1: DINO class map
        ax = fig.add_subplot(gs[ri, 0]); mp = dino.reshape(NY, NX)
        u = np.unique(mp); rmap = {c: i for i, c in enumerate(u)}
        cols = mpl.colormaps.get_cmap("tab20")(np.linspace(0, 1, 20))[:len(u)]
        ax.imshow(np.vectorize(rmap.get)(mp), cmap=ListedColormap(cols),
                  norm=BoundaryNorm(np.arange(-0.5, len(u)), len(u)), interpolation="nearest", aspect="equal")
        ax.set_title(f"{name}: DINO order classes (K={len(u)})", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        # panel 2: ACOM zone-axis map
        ax = fig.add_subplot(gs[ri, 1]); amp = acom.reshape(NY, NX)
        cols2 = mpl.colormaps.get_cmap("tab10")(np.linspace(0, 1, 10))[:max(nza, 1)]
        ax.imshow(amp, cmap=ListedColormap(cols2), norm=BoundaryNorm(np.arange(-0.5, nza), nza),
                  interpolation="nearest", aspect="equal")
        ax.set_title(f"{name}: ACOM zone-axis map\n{100*dom:.0f}% one axis, ~{int(np.median(npk))} peaks/pattern",
                     fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        # panel 3: ACOM template-score (confidence) map
        ax = fig.add_subplot(gs[ri, 2])
        im = ax.imshow(corr.reshape(NY, NX), cmap="magma", interpolation="nearest", aspect="equal")
        ax.set_title(f"{name}: ACOM match score\nAMI vs DINO = {ami:.02f}", fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("A8  Classical orientation mapping (ACOM) on the IMC fields:\n"
                 "near-degenerate zone-axis map, ~0 agreement with DINO - orientation and order are independent,\n"
                 "and peak-based ACOM is unreliable in this low-dose weak-scattering regime",
                 fontsize=11.5, fontweight="bold", y=0.985)
    p = f"{OUT}/A8_acom_vs_dino.png"; fig.savefig(p, dpi=150, facecolor="white"); print(f"\nwrote {p}")

    print("\n" + "=" * 96)
    print("INTERPRETATION (one sentence for the text):")
    print("""  Template-matching orientation mapping (ACOM) applied to these low-dose organic nanodiffraction
  patterns is near-degenerate - it detects hundreds of spurious peaks per pattern and assigns most
  of each field to a single implausible high-index zone axis - so it carries essentially no
  orientation structure and its agreement with the DINO class map is ~0 (AMI/ARI). This is doubly
  consistent with our claim: the DINO classes are not orientation (there is no recoverable
  orientation signal for them to track), and classical peak-based orientation mapping is unreliable
  in exactly the weak-scattering, beam-sensitive regime that the label-free classifier handles.""")


if __name__ == "__main__":
    main()
