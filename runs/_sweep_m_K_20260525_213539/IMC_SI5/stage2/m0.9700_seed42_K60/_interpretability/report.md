# What does the DINO IMC_SI5 model base its clustering on?

Run: `m0.9700_seed42_K60` (IMC_SI5, K_active = 13).
Method: correlate the frozen embedding / class map against physical factors
(**test 1**), causally ablate the input and re-infer (**tests 2 & 4**),
and read per-class physical signatures + mean patterns (**test 5**).

## TL;DR (the nuanced answer)

**The model clusters on BOTH (a) the overall amount of scattered/diffracted
intensity — a crystallinity + mass-thickness signal — AND (b) the 2-D
low-q diffraction structure (the arrangement of rings/spots).
It does NOT base its decision on crystallographic orientation, on the raw
Bragg-peak count, or on high-q detail; and it is invariant to in-plane
rotation (by design: θ-roll augmentation + radial-gated SupCon).**

- (a) is the **dominant, coarse** axis: removing the overall scattered
  intensity collapses the class map almost entirely (ARI 1.00 → **0.01**).
- (b) provides the **fine** 13-way structure: the 1-D radial profile alone
  recovers only ~4 coarse groups (ARI 0.08, K→4); the full 2-D pattern is
  needed for the rest, and it lives mostly at **low q** (inner rings/spots).

## Test 1 — what the embedding encodes / what the classes separate

`probe R²` = 5-fold CV ridge regression *embedding → factor* (how linearly
decodable the factor is). `η²` = correlation ratio (how much of the
factor's variance the DINO partition explains).

| factor | probe R² (encoded) | η² (separated) |
|---|---|---|
| **scattered intensity** | **0.71** | **0.34** |
| **crystallinity (peak/halo)** | **0.24** | 0.15 |
| spottiness (azim-variance) | 0.15 | 0.02 |
| ACOM correlation | ≈ 0 | ≈ 0 |
| n Bragg peaks | ≈ 0 | ≈ 0 |
| ACOM zone axis (auto plan) | — | AMI 0.05, ARI 0.02 |
| ACOM phase (α/γ/neither) | — | AMI ≈ 0, ARI ≈ 0 |

→ The embedding strongly represents **scattered intensity** and
**crystallinity**, and essentially ignores **orientation** and **peak
count**.  (Correlational only — the ablations below test causality.)
*Figure:* `test1_factor_ranking.png`.

## Tests 2 & 4 — causal ablations (re-infer, ARI vs the original map)

High ARI = that information was **not needed**; low ARI = the model
**depends** on it.

| ablation | ARI | K | reading |
|---|---|---|---|
| baseline (sanity) | 1.00 | 13 | pipeline reproduces the saved map exactly |
| **scattered_norm** (post-beam energy removed) | **0.01** | 8 | **overall scattered intensity is a major driver** |
| radial_only (azimuthal average) | 0.08 | 4 | 1-D radial alone → only ~4 coarse groups; needs 2-D |
| blur σ=2 (kill sharp spots) | 0.27 | 7 | sharp spot structure matters |
| q-mask **low** (inner band → 0) | 0.23 | 11 | **low q carries most of the decision** |
| q-mask high (outer band → 0) | 0.53 | 13 | high q matters less |
| perframe_norm (per-frame min-max) | 1.00 | 13 | **artefact** — the beam pins each frame's max, so this barely changes the rings; use `scattered_norm` instead |

*Figure:* `test2_4_ablation_maps.png` (+ `test4_scattered_norm.json`).

**Synthesis of the ablations:** the decision needs (i) the overall scattered
intensity, (ii) 2-D angular structure (not just the radial profile),
(iii) sharp spots, (iv) the low-q region. Orientation is absent from every
view.

## Test 5 — per-class signatures + mean patterns

- `test5_class_signatures.png` — z-scored median of each factor per class.
  Classes order primarily along the **scattered-intensity / crystallinity**
  axis, not along ACOM corr.
- `class_mean_patterns.png` — per-class mean diffraction (13 classes,
  2481 → 351 px). Classes form a progression from diffuse/amorphous-looking
  to sharp, ring-rich crystalline patterns.
- `class_radial_profiles.png` — per-class mean radial profiles separate
  mainly by **overall level + ring contrast**, consistent with a
  crystallinity/intensity ordering, with finer 2-D differences not visible
  in 1-D (why `radial_only` collapses to ~4).

## GradCAM / Integrated-Gradients — *where* in the pattern it looks

Per-prototype class-average attribution (GradCAM + IG, projected from the
polar input back to the Cartesian detector). Files:
`eval/paper_attribution/fig_paper_attribution.png` (all 13 prototypes) and
`fig_paper_attribution_p0..p12.png` (standalone, paper-ready).

- Attribution concentrates on the **inner / low-q rings and spots**, with
  little weight in the outer high-q corners — the *spatial* version of the
  q-mask ablation (low-q dominant, ARI 0.23 vs high-q 0.53).
- The attended structure is **azimuthally distributed** (it lights up rings,
  not a single Bragg direction), consistent with rotation invariance and
  with the model keying on the radial/ring arrangement rather than on an
  absolute orientation.
- Higher-intensity / more-crystalline prototypes show stronger, sharper
  low-q attribution than the diffuse/amorphous-looking ones — corroborating
  the scattered-intensity + crystallinity axis from tests 1, 4 and 5.

## Why this makes sense (training design)

The model is DINO + **radial-gated SupCon** with **θ-roll (azimuthal)
augmentation**:
- θ-roll → trained to be **in-plane-rotation invariant** → cannot key on
  absolute orientation (matches ACOM ≈ 0).
- The SupCon gate compares **radial** profiles → biases grouping toward the
  radial/crystallinity signature, which is exactly the dominant axis found.

## Caveats / what's NOT yet disentangled

- **Scattered intensity bundles crystallinity AND thickness/mass.** The
  ablation can't separate them; both raise the total diffracted signal. A
  thickness map (e.g. from a t/λ estimate) would be needed to split them.
- **Orientation re-confirmed with a valid library.** The earlier ACOM used
  the **`corners`** plan (cubic triangle), which under-covers triclinic IMC.
  Re-run with an **`auto`** plan (pymatgen symmetry) on α+γ, your detection
  params (threshold 0.1, min_sigma 2, k_max 0.5, GPU) and `min_peaks=4`,
  `corr_threshold=0.03`: **367 positions indexed across 43 distinct
  (phase, zone-axis) labels, and DINO-class vs zone-axis AMI = 0.051,
  ARI = 0.017** — i.e. still essentially zero. Orientation is confirmed
  **not** a driver, consistent with the θ-roll design and the radial_only
  collapse. (`test_orientation_auto.json`, `orient_*.npy`.)

## Files
`test1_ranking.json`, `test1_factor_ranking.png`,
`test2_4_ablations.json`, `test4_scattered_norm.json`,
`test2_4_ablation_maps.png`, `test5_class_signatures.png`,
`class_mean_patterns.png`, `class_radial_profiles.png`,
`class_mean_radial_profiles.npy`,
`test_orientation_auto.json`, `orient_phase_id.npy`,
`orient_winning_corr.npy`,
`eval/paper_attribution/fig_paper_attribution.png` (+ `_p0..p12.png`).
