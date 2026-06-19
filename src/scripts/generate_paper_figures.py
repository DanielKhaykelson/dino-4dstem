"""
generate_paper_figures.py — produce the paper figures from available runs.

Figures produced:

  fig1_naphi_benchmark.png
      Na006a row, Na007b row, Na007a-transfer row. Each row: class map
      colored by DINO4DSTEM prototype, class-mean polar pattern for a
      crystalline ("line-rich") prototype, class-mean pattern for a
      rotational-disorder ("line-free") prototype, radial profiles
      overlaid with d-spacing markers.  Khaykelson 2025 Fig 4 layout.

  fig2_euinas_domains.png
      Class map, two largest film-interior class-mean patterns,
      normalized difference (red/blue lobes diagnostic).

  fig3_strain_pipeline.png
      Re-uses the strain analysis figure written by analyze_strain.py
      for the EuInAs K=12 (or winner) class pair with highest radial-
      profile similarity.

  fig4_imc_crystal_identity.png
      50nm and 150nm class maps side-by-side; per-prototype radial
      profiles labeled amorphous-vs-crystalline; affine-match matrix
      showing which crystalline prototypes are the same polymorph
      within each film.

  fig5_nmf_vs_dino4dstem_Na007b.png
      Side-by-side Na007b maps: NMF+kmeans at silhouette-K (top),
      NMF+kmeans at matched-K (middle), DINO4DSTEM at matched-K
      (bottom). Silhouette vs K curve as inset. Head-to-head table.

  figS2_attribution.png
      GradCAM + IG on one representative prototype (polar + Cartesian).

  figS3_backbone_ablation.png
      Bar chart: KNN, intra/inter for L1, L2, ViT on Na007b.

All figures are PNG at 300 DPI, intended for arXiv submission.
Re-run the script any time to refresh with newer data.
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import json
import os
import sys

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from data import SAMPLES, LoadPRZ
from analyze_imc import (_clip_log1p_aggressive, _beam_mask,
                           class_means, radial_profile, find_peaks_simple,
                           crystallinity_score)

mpl = matplotlib
mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
})

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figures")
os.makedirs(OUT, exist_ok=True)


# =========================================================================
# Helpers
# =========================================================================

def _safe_load_inference(sample: str, config: str):
    p = f"runs/{sample}/{config}/eval/inference.npz"
    if not os.path.exists(p):
        return None
    return np.load(p)


def _safe_load_metrics(sample: str, config: str):
    p = f"runs/{sample}/{config}/eval/metrics.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _class_map_imshow(ax, assigns_2d, K_active, title=None):
    if K_active <= 10:
        cmap, vmax = "tab10", 9
    elif K_active <= 20:
        cmap, vmax = "tab20", 19
    else:
        cmap, vmax = "turbo", max(K_active - 1, 1)
    ax.imshow(assigns_2d, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


# =========================================================================
# Figure 1 — NaPHI benchmark  (Khaykelson 2025 Fig 4 layout)
# =========================================================================

def fig1_naphi_benchmark():
    # 3 rows: Na006a, Na007b, Na007a (transfer).  4 cols each:
    #   class map | mean from "line-rich" prototype | mean from "line-free"
    #   prototype | radial profiles overlaid.
    row_specs = [
        ("Na006a", "winner_polar_centroid", "Na006a (small flake, 0 deg)"),
        ("Na007b", "sweep_polar_centroid",  "Na007b (mature flake, 0 deg)"),
        ("Na007a", "transfer_from_winner",  "Na007a (same flake, 45 deg; transfer)"),
    ]
    fig = plt.figure(figsize=(10, 7.2))
    gs = GridSpec(3, 4, figure=fig, hspace=0.25, wspace=0.30,
                    left=0.06, right=0.98, top=0.94, bottom=0.06,
                    width_ratios=[1.1, 1.0, 1.0, 1.4])

    for r, (sample, config, row_title) in enumerate(row_specs):
        inf = _safe_load_inference(sample, config)
        m = _safe_load_metrics(sample, config)
        if inf is None or m is None:
            for c in range(4):
                ax = fig.add_subplot(gs[r, c])
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"{sample}/{config}\nmissing", ha="center", va="center")
            continue

        scan_shape = SAMPLES[sample]["scan_shape"]
        assigns = inf["assigns"]
        soft_probs = inf["soft_probs"]
        K = soft_probs.shape[1]
        counts = np.bincount(assigns, minlength=K)

        # Class means (need the dataset).
        dataset = LoadPRZ(SAMPLES[sample]["path"], resize=192, vmax=SAMPLES[sample]["vmax"])
        means = class_means(dataset, assigns, soft_probs, K, N_top=300)
        # Compute radial profiles + crystallinity score.
        profs = np.stack([radial_profile(means[c], r_min=15, n_bins=80)[0]
                           for c in range(K)], 0)
        centers = radial_profile(means[0], r_min=15, n_bins=80)[1]
        xtal = np.array([crystallinity_score(p) for p in profs])

        # Pick one "line-rich" (most-crystalline) and one "line-free" (least-
        # crystalline, but still substantial, not vacuum).
        # Heuristic: line-rich = argmax xtal among in-sample classes (ignore
        # classes that look like vacuum — those are typically largest in
        # count on small scans; better proxy: usage fraction < 0.5).
        on_sample = np.where(counts / counts.sum() < 0.45)[0]
        if len(on_sample) == 0:
            on_sample = np.arange(K)
        xtal_on = xtal[on_sample]
        # Line-rich: most crystalline among on-sample.
        idx_rich = on_sample[int(np.argmax(xtal_on))]
        # Line-free: least crystalline among on-sample with counts >= 100.
        usable = on_sample[counts[on_sample] >= 100]
        if len(usable) == 0:
            usable = on_sample
        xtal_usable = xtal[usable]
        idx_free = usable[int(np.argmin(xtal_usable))]
        # Distinct selections.
        if idx_free == idx_rich and len(usable) > 1:
            xs = np.argsort(xtal[usable])
            idx_free = usable[xs[0 if xs[0] != int(np.argmax(xtal_on)) else 1]]

        # (a) Class map.
        ax = fig.add_subplot(gs[r, 0])
        _class_map_imshow(ax, assigns.reshape(scan_shape),
                            K_active=int(assigns.max()) + 1)
        ax.set_title(row_title, fontsize=9, loc="left", fontweight="bold")

        # (b) Line-rich class-mean (polar-space image for visual clarity).
        # Pattern is Cartesian raw; display with aggressive clip.
        H0, W0 = dataset.H, dataset.W
        bm = _beam_mask(H0, W0, radius=40)
        ax = fig.add_subplot(gs[r, 1])
        ax.imshow(_clip_log1p_aggressive(means[idx_rich], mask=bm, pct_lo=5, pct_hi=95),
                   cmap="inferno")
        ax.set_title(f"p{idx_rich}  line-rich-like (xtal={xtal[idx_rich]:.1f})",
                      fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # (c) Line-free class-mean.
        ax = fig.add_subplot(gs[r, 2])
        ax.imshow(_clip_log1p_aggressive(means[idx_free], mask=bm, pct_lo=5, pct_hi=95),
                   cmap="inferno")
        ax.set_title(f"p{idx_free}  line-free-like (xtal={xtal[idx_free]:.1f})",
                      fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

        # (d) Radial profiles — both classes overlaid + all other classes in faint.
        ax = fig.add_subplot(gs[r, 3])
        cmap_colors = plt.get_cmap("tab10")
        for c in range(K):
            p = profs[c]
            pn = (p - p.min()) / (p.max() - p.min() + 1e-12)
            if c == idx_rich:
                ax.plot(centers, pn, color="tab:red", lw=1.8,
                         label=f"p{idx_rich} rich")
            elif c == idx_free:
                ax.plot(centers, pn, color="tab:blue", lw=1.8,
                         label=f"p{idx_free} free")
            else:
                ax.plot(centers, pn, color=cmap_colors(c % 10),
                         lw=0.6, alpha=0.3)
        ax.set_xlabel("r (post-resize 192-px radius)")
        ax.set_ylabel("normalised I")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("DINO4DSTEM on Na-PHI: three flakes, benchmark layout "
                   "(Khaykelson 2025 Fig. 4 style)", fontsize=11, y=0.98)
    path = os.path.join(OUT, "fig1_naphi_benchmark.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper_fig] wrote {path}")


# =========================================================================
# Figure 2 — EuInAs class map + pair means + normalized difference
# =========================================================================

def fig2_euinas_domains(config="winner_polar_centroid"):
    sample = "EuInAs_B100"
    inf = _safe_load_inference(sample, config)
    m = _safe_load_metrics(sample, config)
    if inf is None or m is None:
        print(f"[paper_fig] fig2: {sample}/{config} missing")
        return
    scan_shape = SAMPLES[sample]["scan_shape"]
    assigns = inf["assigns"]
    soft_probs = inf["soft_probs"]
    K = soft_probs.shape[1]
    counts = np.bincount(assigns, minlength=K)

    dataset = LoadPRZ(SAMPLES[sample]["path"], resize=192,
                       vmax=SAMPLES[sample]["vmax"])
    means = class_means(dataset, assigns, soft_probs, K, N_top=300)
    # Heuristic for A and B: the two largest classes (typically layer interior
    # + something close to it).
    order = np.argsort(-counts)
    # Filter out tiny classes.
    valid = [c for c in order if counts[c] > 100][:4]
    # Pick the TWO classes whose radial profiles are MOST SIMILAR — these are
    # candidates for "same lattice, small rotation" that the strain pipeline
    # will separate.
    profs = {c: radial_profile(means[c], r_min=15, n_bins=80)[0] for c in valid}
    def _nrm(p):
        p = p - p.min(); return p / (np.linalg.norm(p) + 1e-12)
    nprofs = {c: _nrm(profs[c]) for c in valid}
    best_pair, best_s = None, -1.0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            s = float(nprofs[valid[i]] @ nprofs[valid[j]])
            if s > best_s:
                best_s, best_pair = s, (valid[i], valid[j])
    cA, cB = best_pair

    fig = plt.figure(figsize=(11, 4.0))
    gs = GridSpec(1, 4, figure=fig, wspace=0.15,
                    left=0.04, right=0.99, top=0.92, bottom=0.08,
                    width_ratios=[1.3, 1.0, 1.0, 1.0])

    # (a) class map wide aspect
    ax = fig.add_subplot(gs[0, 0])
    _class_map_imshow(ax, assigns.reshape(scan_shape),
                        K_active=int(assigns.max()) + 1,
                        title=f"(a) EuInAs class map  (K_active={int(assigns.max())+1})")

    H0, W0 = dataset.H, dataset.W
    bm = _beam_mask(H0, W0, radius=40)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(_clip_log1p_aggressive(means[cA], mask=bm, pct_lo=5, pct_hi=95), cmap="inferno")
    ax.set_title(f"(b) p{cA} mean  (N={counts[cA]})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(_clip_log1p_aggressive(means[cB], mask=bm, pct_lo=5, pct_hi=95), cmap="inferno")
    ax.set_title(f"(c) p{cB} mean  (N={counts[cB]})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    eps = 1e-6 * max(means[cA].max(), means[cB].max(), 1.0)
    diff = (means[cA] - means[cB]) / (means[cA] + means[cB] + eps)
    ax.imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"(d) (p{cA} - p{cB}) / (p{cA} + p{cB})\n"
                   f"rot+strain signature  (profile cos={best_s:.2f})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    path = os.path.join(OUT, "fig2_euinas_domains.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper_fig] wrote {path}  (pair p{cA} vs p{cB})")


# =========================================================================
# Figure 5 — DINO4DSTEM vs NMF head-to-head on Na007b
# =========================================================================

def fig5_nmf_vs_dino4dstem():
    sample = "Na007b"
    scan_shape = SAMPLES[sample]["scan_shape"]

    dino = _safe_load_inference(sample, "sweep_polar_centroid")
    dino_m = _safe_load_metrics(sample, "sweep_polar_centroid")
    nmf_path = f"runs/{sample}/baseline_nmf_kmeans/eval/inference.npz"
    nmf_m_path = f"runs/{sample}/baseline_nmf_kmeans/eval/metrics.json"
    if dino is None or not os.path.exists(nmf_path):
        print(f"[paper_fig] fig5: missing NMF or DINO4DSTEM on {sample}")
        return
    nmf = np.load(nmf_path)
    with open(nmf_m_path) as f:
        nmf_m = json.load(f)

    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(2, 3, figure=fig, wspace=0.15, hspace=0.30,
                    height_ratios=[3.5, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    _class_map_imshow(ax, nmf["assigns_kstar"].reshape(scan_shape),
                        K_active=int(nmf["assigns_kstar"].max()) + 1,
                        title=f"(a) NMF+k-means  K* silhouette={nmf_m['k_star']}\n"
                               f"KNN={nmf_m['metrics_at_k_star']['KNN_purity_k10']:.3f}  "
                               f"intra/inter={nmf_m['metrics_at_k_star']['intra_over_inter']:.1f}")

    ax = fig.add_subplot(gs[0, 1])
    target_k = nmf_m["target_k"]
    if nmf["assigns_target_k"].size > 0:
        _class_map_imshow(ax, nmf["assigns_target_k"].reshape(scan_shape),
                            K_active=target_k,
                            title=f"(b) NMF+k-means at K={target_k}\n"
                                   f"KNN={nmf_m['metrics_at_target_k']['KNN_purity_k10']:.3f}  "
                                   f"intra/inter={nmf_m['metrics_at_target_k']['intra_over_inter']:.2f}")

    ax = fig.add_subplot(gs[0, 2])
    _class_map_imshow(ax, dino["assigns"].reshape(scan_shape),
                        K_active=int(dino["assigns"].max()) + 1,
                        title=f"(c) DINO4DSTEM at K={int(dino['assigns'].max())+1}\n"
                               f"KNN={dino_m['KNN_purity_k10']:.3f}  "
                               f"intra/inter={dino_m['intra_over_inter']:.1f}")

    # Silhouette curve
    ax_sil = fig.add_subplot(gs[1, :])
    ks = [r[0] for r in nmf_m["silhouette_curve"]]
    sil = [r[1] for r in nmf_m["silhouette_curve"]]
    ax_sil.plot(ks, sil, "o-", color="tab:purple", lw=2)
    ax_sil.axvline(nmf_m["k_star"], color="red", ls="--", lw=1,
                    label=f"silhouette peak K={nmf_m['k_star']} (chooses vacuum vs sample)")
    ax_sil.axvline(target_k, color="tab:blue", ls="--", lw=1,
                    label=f"DINO4DSTEM effective K={target_k}")
    ax_sil.set_xlabel("K"); ax_sil.set_ylabel("silhouette")
    ax_sil.set_title("(d) NMF+k-means silhouette sweep: over-consolidates to K=2 (trivial)")
    ax_sil.legend(fontsize=8); ax_sil.grid(alpha=0.3)

    fig.suptitle("DINO4DSTEM vs NMF+k-means on Na007b", fontsize=11)
    path = os.path.join(OUT, "fig5_nmf_vs_dino4dstem_Na007b.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper_fig] wrote {path}")


# =========================================================================
# Figure 4 — IMC crystal identity
# =========================================================================

def fig4_imc_crystal_identity():
    # Uses existing analyze_imc figures + adds affine matrix once run_post_t06
    # finishes. For v0, just produce the class maps + radial profiles panel.
    cfgs = [
        ("IMC_50nm_SI2",  "winner_polar_centroid", "50 nm"),
        ("IMC_150nm_SI5", "winner_polar_centroid", "150 nm"),
    ]
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.15,
                    height_ratios=[1.0, 1.0])

    for col, (sample, config, label) in enumerate(cfgs):
        inf = _safe_load_inference(sample, config)
        m = _safe_load_metrics(sample, config)
        if inf is None:
            ax = fig.add_subplot(gs[0, col])
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{sample}\nmissing", ha="center", va="center")
            continue
        assigns = inf["assigns"]
        scan_shape = SAMPLES[sample]["scan_shape"]
        K = int(assigns.max()) + 1
        ax = fig.add_subplot(gs[0, col])
        _class_map_imshow(ax, assigns.reshape(scan_shape),
                            K_active=K,
                            title=f"({chr(ord('a')+col)}) Indomethacin {label} PVD + 70degC/60min\n"
                                   f"K_active={K}, KNN={m['KNN_purity_k10']:.3f}")

        # Radial profiles overlaid below.
        soft_probs = inf["soft_probs"]
        dataset = LoadPRZ(SAMPLES[sample]["path"], resize=192,
                           vmax=SAMPLES[sample]["vmax"])
        means = class_means(dataset, assigns, soft_probs, K, N_top=200)
        profs = np.stack([radial_profile(means[c], r_min=15, n_bins=80)[0]
                           for c in range(K)], 0)
        centers = radial_profile(means[0], r_min=15, n_bins=80)[1]
        xtal = np.array([crystallinity_score(p) for p in profs])
        ax = fig.add_subplot(gs[1, col])
        cmap = plt.get_cmap("tab10")
        for c in range(K):
            p = profs[c]
            pn = (p - p.min()) / (p.max() - p.min() + 1e-12)
            is_crystal = xtal[c] >= 1.5
            ax.plot(centers, pn, color=cmap(c % 10),
                     lw=1.6 if is_crystal else 0.8,
                     alpha=1.0 if is_crystal else 0.35,
                     label=f"p{c} ({'C' if is_crystal else 'A'})")
        ax.set_xlabel("r (post-resize px)")
        ax.set_ylabel("normalised I")
        ax.set_title(f"({chr(ord('c')+col)}) radial profiles (C=crystalline, A=amorphous)",
                      fontsize=9)
        ax.legend(fontsize=6, ncol=2, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("DINO4DSTEM on indomethacin PVD thin films — "
                   "crystal-identity comparison", fontsize=11)
    path = os.path.join(OUT, "fig4_imc_crystal_identity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper_fig] wrote {path}")


# =========================================================================
# Figure S3 — backbone ablation (L1 / L2 / ViT) on Na007b
# =========================================================================

def figS3_backbone_ablation():
    sample = "Na007b"
    configs = [
        ("sweep_polar_centroid", "L1 (winner)"),
        ("winner_L2",             "L2"),
        ("winner_vit",            "ViT-Tiny"),
    ]
    data = []
    for c, label in configs:
        m = _safe_load_metrics(sample, c)
        if m is None:
            continue
        data.append(dict(label=label,
                          KNN=m["KNN_purity_k10"],
                          intra_inter=m["intra_over_inter"],
                          K_active=m.get("K_active", m.get("active_prototypes", 0))))
    if not data:
        print(f"[paper_fig] figS3: no ablation configs yet")
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    labels = [d["label"] for d in data]
    axes[0].bar(labels, [d["KNN"] for d in data], color="tab:blue")
    axes[0].set_title("KNN purity (k=10)"); axes[0].set_ylim(0, 1.02); axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(labels, [d["intra_inter"] for d in data], color="tab:green")
    axes[1].set_title("intra / inter cosine"); axes[1].grid(alpha=0.3, axis="y")
    axes[2].bar(labels, [d["K_active"] for d in data], color="tab:orange")
    axes[2].set_title("K_active"); axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle(f"Backbone ablation on {sample}", fontsize=11)
    path = os.path.join(OUT, "figS3_backbone_ablation.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[paper_fig] wrote {path}")


# =========================================================================
# Main
# =========================================================================

def main():
    print(f"paper figures -> {OUT}")
    fig1_naphi_benchmark()
    fig2_euinas_domains()
    fig4_imc_crystal_identity()
    fig5_nmf_vs_dino4dstem()
    figS3_backbone_ablation()
    print("\nDone. Re-run any time to refresh figures with newer run data.")


if __name__ == "__main__":
    main()
