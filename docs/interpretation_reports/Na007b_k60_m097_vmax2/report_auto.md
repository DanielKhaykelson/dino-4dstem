# Interpretation report — loaded__Na007b_nbed.cube

Run: `Na007b_k60_m097_vmax2`  ·  K_active = 12  ·  N = 12600

## What the embedding encodes / separates (probing)

| factor | probe R² | η² | MI |
|---|---|---|---|
| scattered intensity | 0.9503 | 0.892 | 0.956 |
| spottiness (azim-var) | 0.931 | 0.8735 | 0.8252 |
| crystallinity (peak/halo) | 0.89 | 0.7948 | 0.8731 |


## Causal ablations (ARI vs original map)

| ablation | ARI | K |
|---|---|---|
| baseline | 1.0 | 12 |
| scattered_norm | 0.0019 | 3 |
| radial_only | 0.1103 | 5 |
| blur(s=2) | 0.3898 | 11 |
| qmask_low | 0.1121 | 5 |
| qmask_high | 0.9378 | 12 |

High ARI = that information was not needed; low ARI = the model depends on it.

## Could a classical method reproduce the DINO map?

| classical feature | ARI vs DINO | AMI vs DINO |
|---|---|---|
| pattern NMF (non-neg decomp.) | 0.4791 | 0.4734 |
| azimuthal profile (radial) → PCA | 0.4735 | 0.5001 |
| pattern PCA (linear decomp.) | 0.4632 | 0.4584 |
| combination (DF + radial + pattern) | 0.453 | 0.4715 |
| virtual DF (total scattered I) | 0.4245 | 0.421 |

**Uniqueness:** classical features **substantially capture** the partition (best ARI=0.4791, AMI=0.5); DINO refines them but does not depart strongly — it behaves like a non-linear blend of classical descriptors.

## Reading

Summary: The embedding most strongly encodes **scattered intensity, crystallinity (peak/halo), spottiness (azim-var)**; removing the overall **scattered intensity** collapses the map (ARI=0.0019) — a major driver; the **1-D radial profile alone** is insufficient (ARI=0.1103) — 2-D structure is needed; **low-q** matters more than high-q (ARI 0.1121 vs 0.9378); classical features **substantially capture** the partition (best ARI=0.4791, AMI=0.5); DINO refines them but does not depart strongly — it behaves like a non-linear blend of classical descriptors.

> **ACOM not available** for this run, so the orientation/phase factors and the zone-axis cross-check were skipped. Run ACOM (Diffraction ▸ ACOM, full-dataset mode) first to include them.
