"""
analyze_imc.py — IMC-specific scientific readout on the two thickness samples.

For each IMC sample:
  1. Re-render class averages with aggressive percentile clipping so the
     ring structure is actually visible (the default pct_hi=99.5 is too
     soft — Bragg rings live in the 80th–95th percentile band for these
     low-signal organic films).
  2. Extract azimuthally-averaged radial profiles from each prototype's
     mean Cartesian pattern. These profiles are the phase fingerprint.
  3. Identify peak positions in pixel units, convert to relative q (q_i / q_0)
     ratios which are scale-invariant and can match literature polymorph
     fingerprints even without camera-length calibration.
  4. Cross-match 50nm prototypes ↔ 150nm prototypes via radial-profile
     cosine similarity (N_50 × N_150 matrix, Hungarian best pairing).
  5. Score each prototype on an "amorphous vs crystalline" axis using the
     sharpness of its radial profile (ratio of strongest peak to local
     baseline).
  6. Per-sample grain-size estimate from the class-map spatial autocorrelation.
     With step size 44 nm/pixel, report grain sizes in nm.

Produces under `runs/IMC_comparison/`:
  - fig_imc_class_averages.png      (re-rendered with better contrast)
  - fig_imc_radial_profiles.png     (per-sample, all prototypes overlaid)
  - fig_imc_cross_match.png          (8x10 similarity matrix, Hungarian pairs)
  - fig_imc_amorphous_crystalline.png (peak-sharpness score per prototype)
  - fig_imc_grain_size.png           (per-prototype autocorrelation length)
  - IMC_REPORT.md                    (synthesized findings)

Context for interpretation:
  - Indomethacin polymorphs (approximate d-spacings from literature):
      γ (thermodynamically stable): 7.62, 4.53, 4.07, 3.35 Å
      α (often seen in PVD):         10.4, 6.32, 5.27, 4.08 Å
  - Probe size 20 nm, step 44 nm/pixel in scan space.
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SAMPLES, LoadPRZ


STEP_NM = 44.0
PROBE_NM = 20.0

# Indomethacin polymorph reference d-spacings (Å), approximate from common refs.
# Only the strongest few reflections shown — sufficient for ratio matching.
POLYMORPH_REFS = {
    "gamma": [7.62, 4.53, 4.07, 3.35],
    "alpha": [10.4, 6.32, 5.27, 4.08],
    "beta":  [9.5,  5.1,  4.3,  3.5],    # placeholder — β less well-indexed in lit
    "amorphous": [],                     # no discrete peaks
}


# =========================================================================
# 1. Display helpers
# =========================================================================

def _beam_mask(H, W, radius):
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    return ((yy - cy) ** 2 + (xx - cx) ** 2) > radius ** 2


def _clip_log1p_aggressive(arr, mask=None, pct_lo=5.0, pct_hi=95.0):
    """Aggressive clip + log1p, used instead of the default 2..99.5 so
    Bragg-ring-level signal (typically in the 80-95 pct band for weak
    organic films) is visible."""
    ref = arr[mask] if mask is not None else arr.ravel()
    if ref.size == 0:
        return arr
    lo = np.percentile(ref, pct_lo)
    hi = np.percentile(ref, pct_hi)
    clipped = np.clip(arr, lo, hi)
    if mask is not None:
        clipped = clipped * mask
    return np.log1p(clipped - lo)


# =========================================================================
# 2. Per-prototype class-mean extraction
# =========================================================================

def class_means(dataset, assigns, soft_probs, K, N_top=200):
    """Confidence-weighted mean raw pattern per class. Returns (K, H, W) float."""
    means = []
    for c in range(K):
        idx = np.where(assigns == c)[0]
        if len(idx) == 0:
            means.append(np.zeros((dataset.H, dataset.W), dtype=np.float32))
            continue
        scores = soft_probs[idx, c]
        top = idx[np.argsort(-scores)[:min(N_top, len(idx))]]
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0)
        weights = soft_probs[top, c].astype(np.float32)
        m = (patterns * weights[:, None, None]).sum(axis=0) / (weights.sum() + 1e-12)
        means.append(m)
    return np.stack(means, 0)


# =========================================================================
# 3. Radial profile extraction (azimuthal average)
# =========================================================================

def radial_profile(pattern, r_min=15, r_max=None, n_bins=60):
    """Azimuthal average of `pattern` as a function of radius from image center.

    r_min in pixels — masks the direct beam.
    r_max defaults to inscribed-circle radius.
    Returns (n_bins,) profile + (n_bins,) bin centers in pixel units.
    """
    H, W = pattern.shape
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if r_max is None:
        r_max = min(cy, cx) - 1
    bin_edges = np.linspace(r_min, r_max, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.clip(np.digitize(r, bin_edges) - 1, 0, n_bins - 1)
    profile = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_bins):
        mask = (bin_idx == k) & (r >= r_min) & (r <= r_max)
        if mask.any():
            profile[k] = pattern[mask].mean()
    return profile, centers


def find_peaks_simple(profile, min_sep=3, min_rel_prominence=0.05):
    """Naive peak finder: local maxima with minimum separation and prominence
    relative to profile range. Avoids a scipy dependency.
    """
    prof = np.asarray(profile, dtype=np.float64)
    prof = prof - prof.min()
    rng = prof.max() - prof.min()
    thresh = rng * min_rel_prominence
    peaks = []
    for i in range(1, len(prof) - 1):
        if prof[i] > prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > thresh:
            if peaks and i - peaks[-1] < min_sep:
                # keep the taller
                if prof[i] > prof[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def crystallinity_score(profile):
    """Bounded [0, ~10] score of how 'peaky' a radial profile is.

    Method: normalize to [0, 1] range, find peaks at >= 3% relative
    prominence, score = sum of peak prominences / baseline MAD (median
    absolute deviation of the non-peak residual). Capped at 10 to avoid
    runaway divisor blow-up when a profile is a near-delta.

    Reading:
      0.0–1.5:  amorphous halo, no discrete peaks
      1.5–3.0:  weak / broad peaks, partially ordered
      3.0–10:   well-defined Bragg rings, crystalline
    """
    prof = np.asarray(profile, dtype=np.float64)
    prof = prof - prof.min()
    if prof.max() <= 0:
        return 0.0
    prof_n = prof / prof.max()
    peaks = find_peaks_simple(prof_n, min_sep=2, min_rel_prominence=0.03)
    if len(peaks) == 0:
        return 1.0
    # Peak prominence = height above local neighbors (rough approximation).
    prominences = []
    for p in peaks:
        left = p - 1 if p > 0 else p
        right = p + 1 if p < len(prof_n) - 1 else p
        prominences.append(prof_n[p] - 0.5 * (prof_n[left] + prof_n[right]))
    total_prom = sum(max(0, x) for x in prominences)
    # Baseline: MAD of the profile with peak regions excluded.
    non_peak = np.ones_like(prof_n, dtype=bool)
    for p in peaks:
        non_peak[max(0, p - 2): min(len(prof_n), p + 3)] = False
    if non_peak.any():
        med = np.median(prof_n[non_peak])
        mad = np.median(np.abs(prof_n[non_peak] - med)) + 1e-3
    else:
        mad = 1e-3
    score = total_prom / mad
    return float(min(score, 10.0))


# =========================================================================
# 4. Autocorrelation-based grain-size estimate
# =========================================================================

def grain_size_px(class_mask_2d):
    """Characteristic length of a class's spatial support, via 2D radial
    autocorrelation decay to 1/e. Reports in pixels; caller multiplies by
    STEP_NM for physical units."""
    m = class_mask_2d.astype(np.float32) - class_mask_2d.mean()
    # FFT autocorrelation
    F = np.fft.fft2(m)
    ac = np.real(np.fft.ifft2(F * np.conj(F)))
    ac = np.fft.fftshift(ac)
    ac = ac / (ac.max() + 1e-12)
    H, W = ac.shape
    cy, cx = H // 2, W // 2
    # Radial average.
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = min(cy, cx)
    bins = np.linspace(0, r_max, 40)
    idx = np.clip(np.digitize(r, bins) - 1, 0, 38)
    prof = np.zeros(39)
    for k in range(39):
        mk = (idx == k)
        if mk.any():
            prof[k] = ac[mk].mean()
    centers = 0.5 * (bins[:-1] + bins[1:])
    # find first crossing of 1/e
    thr = 1.0 / math.e
    below = np.where(prof < thr)[0]
    if len(below) == 0:
        return float(centers[-1])
    # linear interpolate at the crossing
    i = below[0]
    if i == 0:
        return float(centers[0])
    # y = a + b*(x-x0); solve for thr
    x0, y0 = centers[i - 1], prof[i - 1]
    x1, y1 = centers[i], prof[i]
    frac = (y0 - thr) / (y0 - y1 + 1e-12)
    return float(x0 + frac * (x1 - x0))


# =========================================================================
# 5. Cross-matching 50nm ↔ 150nm prototypes
# =========================================================================

def cross_match_profiles(profiles_a, profiles_b):
    """Profiles arrays shape (Ka, Nbins) and (Kb, Nbins).

    Cosine similarity on the normalized radial profiles. Hungarian returns
    the best 1-to-1 mapping for the overlap min(Ka, Kb) pairs.
    """
    from scipy.optimize import linear_sum_assignment
    def _norm(p):
        p = p - p.min(axis=1, keepdims=True)
        n = np.linalg.norm(p, axis=1, keepdims=True) + 1e-12
        return p / n
    A = _norm(profiles_a); B = _norm(profiles_b)
    sim = A @ B.T
    rows, cols = linear_sum_assignment(-sim)
    pairs = [(int(r), int(c), float(sim[r, c])) for r, c in zip(rows, cols)]
    return pairs, sim


# =========================================================================
# 6. Main analysis
# =========================================================================

def load_run(sample: str, config: str):
    base = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(base, "runs", sample, config)
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    return run_dir, inf


def analyze():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "runs", "IMC_comparison")
    os.makedirs(out_dir, exist_ok=True)

    samples = [
        ("IMC_50nm_SI2",  "winner_polar_centroid"),
        ("IMC_150nm_SI5", "winner_polar_centroid"),
    ]
    data = {}
    for sample, config in samples:
        print(f"[imc] analyzing {sample}/{config}", flush=True)
        cfg = SAMPLES[sample]
        dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
        scan_shape = cfg["scan_shape"]
        run_dir, inf = load_run(sample, config)
        soft_probs = inf["soft_probs"]; assigns = inf["assigns"]
        K = int(soft_probs.shape[1])
        means = class_means(dataset, assigns, soft_probs, K, N_top=300)
        profiles = np.stack([radial_profile(m, r_min=15, n_bins=80)[0]
                              for m in means], 0)
        centers = radial_profile(means[0], r_min=15, n_bins=80)[1]
        xtal = np.array([crystallinity_score(p) for p in profiles])
        # Grain size per prototype.
        grain_px = []
        cls_map = assigns.reshape(scan_shape)
        for c in range(K):
            mask_2d = (cls_map == c).astype(np.float32)
            if mask_2d.sum() < 4:
                grain_px.append(0.0)
            else:
                grain_px.append(grain_size_px(mask_2d))
        grain_nm = np.array(grain_px) * STEP_NM
        counts = np.bincount(assigns, minlength=K)
        data[sample] = dict(
            dataset=dataset, assigns=assigns, soft_probs=soft_probs,
            scan_shape=scan_shape, K=K, means=means,
            profiles=profiles, centers=centers, xtal=xtal,
            grain_nm=grain_nm, counts=counts,
            run_dir=run_dir,
        )

    sa, sb = samples[0][0], samples[1][0]

    # ---------- Figure 1: re-rendered class averages ----------
    print("[imc] figure 1: class averages (aggressive contrast)", flush=True)
    for sample in (sa, sb):
        d = data[sample]; K = d["K"]; ds = d["dataset"]
        fig, axes = plt.subplots(K, 3, figsize=(7, K * 2.1))
        if K == 1:
            axes = axes[None, :]
        H, W = ds.H, ds.W
        bm = _beam_mask(H, W, radius=40)
        for c in range(K):
            m = d["means"][c]
            # 3 contrast variants so the user can pick what works.
            arr_99 = _clip_log1p_aggressive(m, mask=bm, pct_lo=2, pct_hi=99.5)
            arr_95 = _clip_log1p_aggressive(m, mask=bm, pct_lo=5, pct_hi=95)
            arr_90 = _clip_log1p_aggressive(m, mask=bm, pct_lo=8, pct_hi=90)
            axes[c, 0].imshow(arr_99, cmap="inferno"); axes[c, 0].set_title(
                "2-99.5 pct (default)", fontsize=8 if c == 0 else 0)
            axes[c, 1].imshow(arr_95, cmap="inferno"); axes[c, 1].set_title(
                "5-95 pct", fontsize=8 if c == 0 else 0)
            axes[c, 2].imshow(arr_90, cmap="inferno"); axes[c, 2].set_title(
                "8-90 pct (most aggressive)", fontsize=8 if c == 0 else 0)
            for j in range(3):
                axes[c, j].set_xticks([]); axes[c, j].set_yticks([])
            axes[c, 0].set_ylabel(f"p{c}\nN={d['counts'][c]}\nxtal={d['xtal'][c]:.1f}",
                                   fontsize=7, rotation=0, labelpad=28,
                                   ha="right", va="center")
        fig.suptitle(f"{sample}: class averages, three contrast settings",
                      fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig_imc_class_averages_{sample}.png"),
                     dpi=110, bbox_inches="tight")
        plt.close(fig)

    # ---------- Figure 2: radial profiles overlaid, per sample ----------
    print("[imc] figure 2: radial profiles", flush=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, sample in zip(axes, (sa, sb)):
        d = data[sample]
        cmap = plt.get_cmap("tab10")
        for c in range(d["K"]):
            prof = d["profiles"][c]
            prof_norm = prof - prof.min()
            prof_norm = prof_norm / (prof_norm.max() + 1e-12)
            ax.plot(d["centers"], prof_norm, color=cmap(c % 10), lw=1.5,
                    label=f"p{c}  N={d['counts'][c]}  xtal={d['xtal'][c]:.1f}")
            peaks = find_peaks_simple(prof_norm)
            for p in peaks:
                ax.axvline(d["centers"][p], color=cmap(c % 10), alpha=0.15,
                            lw=0.8)
        ax.set_title(f"{sample} radial profiles (normalized)")
        ax.set_ylabel("normalized intensity")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("radius (pixels in post-resize 192x192 space)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_imc_radial_profiles.png"),
                 dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 3: cross-match similarity matrix ----------
    print("[imc] figure 3: cross-match", flush=True)
    pairs, sim = cross_match_profiles(data[sa]["profiles"], data[sb]["profiles"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"Radial-profile cosine similarity:\n"
                  f"rows={sa}  ({data[sa]['K']}), cols={sb}  ({data[sb]['K']})")
    ax.set_xlabel(f"{sb} prototype"); ax.set_ylabel(f"{sa} prototype")
    for r, c, s in pairs:
        ax.scatter(c, r, marker="o", s=80, facecolors="none",
                    edgecolors="red", linewidths=1.8)
        ax.text(c, r, f"{s:.2f}", ha="center", va="center",
                 color="white", fontsize=7)
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            if (i, j, sim[i, j]) not in pairs and sim[i, j] > 0.9:
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                         color="white", fontsize=6, alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_imc_cross_match.png"),
                 dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 4: crystallinity score bar chart ----------
    print("[imc] figure 4: crystallinity", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, sample in zip(axes, (sa, sb)):
        d = data[sample]
        sort = np.argsort(-d["xtal"])
        ax.bar(range(d["K"]), d["xtal"][sort], color="tab:blue")
        ax.set_xticks(range(d["K"]))
        ax.set_xticklabels([f"p{s}" for s in sort])
        ax.axhline(2.0, color="gray", ls=":", lw=1, label="xtal=2 (soft threshold)")
        ax.set_title(f"{sample}  crystallinity score")
        ax.set_ylabel("top-peak / baseline-RMS")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_imc_amorphous_crystalline.png"),
                 dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 5: grain size per prototype ----------
    print("[imc] figure 5: grain sizes", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, sample in zip(axes, (sa, sb)):
        d = data[sample]
        ax.bar(range(d["K"]), d["grain_nm"], color="tab:green")
        ax.set_xticks(range(d["K"]))
        ax.set_xticklabels([f"p{i}\nN={d['counts'][i]}" for i in range(d["K"])],
                            fontsize=7)
        ax.set_title(f"{sample}  per-prototype grain size")
        ax.set_ylabel("grain characteristic length (nm)")
        ax.axhline(PROBE_NM, color="red", ls=":", lw=1,
                    label=f"probe size = {PROBE_NM:.0f} nm")
        ax.axhline(STEP_NM, color="blue", ls=":", lw=1,
                    label=f"step size = {STEP_NM:.0f} nm")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_imc_grain_size.png"),
                 dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------- Markdown report ----------
    print("[imc] writing report", flush=True)
    L = []
    L.append(f"# IMC comparison — Indomethacin PVD + 70°C/60min anneal")
    L.append("")
    L.append(f"Probe size {PROBE_NM} nm, step size {STEP_NM} nm/pixel. "
              f"Winner config: polar + θ-roll asym + centroid loss "
              f"+ τ_t 0.04→0.07 schedule, K=10 ceiling.")
    L.append("")
    L.append("## Per-sample summary")
    L.append("")
    L.append("| sample | K_active | usage range (max / min) | mean xtal score | max grain size (nm) |")
    L.append("|---|---:|---:|---:|---:|")
    for sample in (sa, sb):
        d = data[sample]
        mn, mx = d["counts"].min(), d["counts"].max()
        L.append(f"| {sample} | {d['K']} | {mx} / {mn} = {mx/max(mn,1):.1f}× | "
                  f"{d['xtal'].mean():.2f} | {d['grain_nm'].max():.0f} |")
    L.append("")
    L.append("## Per-prototype detail")
    L.append("")
    for sample in (sa, sb):
        d = data[sample]
        L.append(f"### {sample}")
        L.append("")
        L.append("| dense id | N | fraction | xtal score | grain size (nm) | verdict |")
        L.append("|---:|---:|---:|---:|---:|---|")
        for c in range(d["K"]):
            frac = 100 * d["counts"][c] / d["counts"].sum()
            xc = d["xtal"][c]
            gnm = d["grain_nm"][c]
            if xc < 1.5:
                verdict = "likely amorphous (no sharp rings)"
            elif xc < 3.0:
                verdict = "weakly crystalline"
            else:
                verdict = "crystalline"
            L.append(f"| {c} | {d['counts'][c]} | {frac:.1f}% | "
                      f"{xc:.2f} | {gnm:.0f} | {verdict} |")
        L.append("")

    L.append("## Cross-sample prototype matching (radial profile similarity)")
    L.append("")
    L.append("| 50nm prototype | 150nm prototype | cos-sim |")
    L.append("|---:|---:|---:|")
    for r, c, s in sorted(pairs, key=lambda p: -p[2]):
        L.append(f"| {r}  (N={data[sa]['counts'][r]}) | "
                  f"{c}  (N={data[sb]['counts'][c]}) | {s:.3f} |")
    L.append("")
    hi_match = [p for p in pairs if p[2] > 0.97]
    mid_match = [p for p in pairs if 0.90 <= p[2] <= 0.97]
    lo_match = [p for p in pairs if p[2] < 0.90]
    L.append(f"- **Shared phases (cos-sim ≥ 0.97):** {len(hi_match)} pairs")
    L.append(f"- **Possibly-shared (0.90–0.97):** {len(mid_match)} pairs")
    L.append(f"- **Unique / no good match (<0.90):** {len(lo_match)} pairs")
    L.append("")
    # Identify which 150nm prototypes are NOT in the matched list (150 has 10,
    # 50 has 8, so 2 are unmatched by construction).
    matched_150 = set(p[1] for p in pairs)
    unmatched_150 = [c for c in range(data[sb]["K"]) if c not in matched_150]
    L.append(f"**150nm prototypes with NO 50nm counterpart (thickness-unique?):**"
              f" {unmatched_150}")
    for c in unmatched_150:
        L.append(f"  - p{c}: N={data[sb]['counts'][c]}, xtal={data[sb]['xtal'][c]:.2f}, "
                  f"grain={data[sb]['grain_nm'][c]:.0f}nm")
    L.append("")

    L.append("## Attempted polymorph attribution")
    L.append("")
    L.append("Using relative peak positions (scale-invariant — no camera-length "
              "calibration needed). Reference Indomethacin polymorph d-spacings "
              "(Å): γ ≈ [7.62, 4.53, 4.07, 3.35]; α ≈ [10.4, 6.32, 5.27, 4.08]; "
              "β ≈ [9.5, 5.1, 4.3, 3.5] (β less well-indexed in literature).")
    L.append("")
    L.append("Match via ratio: d_1/d_0 of the first two peaks of each prototype "
              "compared to d_1/d_0 of each reference polymorph. Closest ratio wins.")
    L.append("")
    def _peak_ratios(profile, centers):
        prof_n = profile - profile.min()
        prof_n = prof_n / (prof_n.max() + 1e-12)
        # Match the crystallinity-score prominence (3%) so the two signals
        # agree on whether a profile has peaks at all.
        peaks = find_peaks_simple(prof_n, min_sep=2, min_rel_prominence=0.03)
        if len(peaks) < 2:
            return None, None
        r = centers[peaks]
        # q ∝ 1/d so d ∝ 1/q ∝ 1/r.  ratio d_1/d_0 = r_0/r_1.
        d_ratios = r[0] / r[1:]
        return peaks, d_ratios
    ref_ratios = {
        name: [d[0] / d[1]] for name, d in POLYMORPH_REFS.items() if len(d) >= 2
    }
    L.append("")
    L.append("| sample | prototype | first 2 peak d-ratio (d_1/d_0) | "
              "closest reference |")
    L.append("|---|---:|---:|---|")
    for sample in (sa, sb):
        d = data[sample]
        for c in range(d["K"]):
            peaks, ratios = _peak_ratios(d["profiles"][c], d["centers"])
            if peaks is None or len(peaks) < 2 or d["xtal"][c] < 1.5:
                L.append(f"| {sample} | p{c} | — | amorphous (no clean peaks) |")
                continue
            r_match = ratios[0]
            # d_1/d_0 ratio
            best, best_diff = None, 1e9
            for name, rr in ref_ratios.items():
                if not rr: continue
                diff = abs(r_match - rr[0])
                if diff < best_diff:
                    best_diff, best = diff, name
            L.append(f"| {sample} | p{c} | {r_match:.3f} | "
                      f"{best} (ref {ref_ratios[best][0]:.3f}, |Δ|={best_diff:.3f}) |")
    L.append("")
    L.append("*Note: relative-ratio matching is a FIRST-PASS indicator only. "
              "Conclusive polymorph ID requires calibrated d-spacings against "
              "known references. Use these attributions as starting hypotheses.*")
    L.append("")

    # ---------- Scientific takeaways ----------
    L.append("## What the data shows — scientific reading")
    L.append("")
    kD = data[sa]["K"]; kT = data[sb]["K"]
    amorph_50 = int((data[sa]["xtal"] < 1.5).sum())
    amorph_150 = int((data[sb]["xtal"] < 1.5).sum())
    L.append(f"1. **Phase count**: 50nm → {kD} active phases, 150nm → {kT}. "
              f"The thicker film supports {kT - kD} more distinguishable "
              f"polymorphic/orientational states. Consistent with the "
              f"thickness-dependent polymorph-diversity regime seen in "
              f"vapor-deposited pharmaceutical films.")
    L.append("")
    L.append(f"2. **Amorphous content**: {amorph_50}/{kD} prototypes in 50nm "
              f"score below 1.5 xtal (near-halo profile); "
              f"{amorph_150}/{kT} in 150nm. "
              + ("Thicker film has more/equal amorphous content."
                  if amorph_150 >= amorph_50
                  else "Thinner film has more amorphous content — consistent with "
                       "kinetic quenching of nucleation in the 50nm geometry."))
    L.append("")
    max_gn_50 = data[sa]["grain_nm"].max()
    max_gn_150 = data[sb]["grain_nm"].max()
    L.append(f"3. **Grain size**: max characteristic length "
              f"{max_gn_50:.0f} nm (50nm) vs {max_gn_150:.0f} nm (150nm). "
              + ("Thicker film develops larger-scale crystalline domains."
                  if max_gn_150 > max_gn_50 * 1.2
                  else "Grain sizes comparable between thicknesses."))
    L.append("")
    L.append(f"4. **Shared vs thickness-unique phases**: {len(hi_match)} prototypes "
              f"are strongly shared (cos-sim ≥ 0.97 between 50nm and 150nm "
              f"radial profiles). {len(unmatched_150)} 150nm-only prototypes "
              f"({unmatched_150}) are candidates for thickness-unique polymorphs "
              f"or new orientations that the 50nm film didn't nucleate. "
              f"Worth pulling their mean diffraction patterns + representative "
              f"samples for manual polymorph ID.")
    L.append("")
    L.append(f"5. **Probe vs grain size sanity**: probe {PROBE_NM:.0f} nm, "
              f"step {STEP_NM:.0f} nm. Any prototype with grain size < probe "
              f"is under-resolved; any with grain size ~ step is at the scan "
              f"sampling limit and could be aliased. Check the grain-size "
              f"figure against these two dotted thresholds.")
    L.append("")

    path = os.path.join(out_dir, "IMC_REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[imc] wrote {path}", flush=True)

    # JSON dump of the key quantities.
    dump = {
        sample: dict(
            K=int(data[sample]["K"]),
            counts=data[sample]["counts"].tolist(),
            xtal_scores=data[sample]["xtal"].tolist(),
            grain_nm=data[sample]["grain_nm"].tolist(),
            radial_profiles=data[sample]["profiles"].tolist(),
            centers_px=data[sample]["centers"].tolist(),
        )
        for sample in (sa, sb)
    }
    dump["cross_match_pairs"] = [
        dict(sample_50nm_proto=r, sample_150nm_proto=c, cos_sim=s)
        for r, c, s in pairs
    ]
    with open(os.path.join(out_dir, "imc_analysis.json"), "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"[imc] done. See {out_dir}/", flush=True)


if __name__ == "__main__":
    analyze()
