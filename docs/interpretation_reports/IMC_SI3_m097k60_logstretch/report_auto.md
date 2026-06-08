# Interpretation report — loaded__Survey_CH2_1_nbed.cube

Run: `IMC_SI3_m097k60_logstretch`  ·  K_active = 14  ·  N = 16384

## What the embedding encodes / separates (probing)

| factor | probe R² | η² | MI |
|---|---|---|---|
| scattered intensity | 0.8412 | 0.5356 | 0.6045 |
| crystallinity (peak/halo) | 0.8283 | 0.6918 | 0.6202 |
| spottiness (azim-var) | 0.8224 | 0.6309 | 0.6896 |
| ACOM correlation | 0.0743 | 0.0687 | 0.0626 |

ACOM zone-axis vs classes: AMI=0.0593, ARI=0.0176 (2993 indexed, 182 zone axes).

## Causal ablations (ARI vs original map)

| ablation | ARI | K |
|---|---|---|
| baseline | 1.0 | 14 |
| scattered_norm | 0.0 | 1 |
| radial_only | 0.068 | 3 |
| blur(s=2) | 0.2229 | 8 |
| qmask_low | 0.2717 | 11 |
| qmask_high | 0.3927 | 14 |

High ARI = that information was not needed; low ARI = the model depends on it.

## Could a classical method reproduce the DINO map?

| classical feature | ARI vs DINO | AMI vs DINO |
|---|---|---|
| pattern PCA (linear decomp.) | 0.1776 | 0.303 |
| pattern NMF (non-neg decomp.) | 0.1715 | 0.3129 |
| azimuthal profile (radial) → PCA | 0.1657 | 0.3018 |
| combination (DF + radial + pattern) | 0.1352 | 0.2703 |
| virtual DF (total scattered I) | 0.1195 | 0.2332 |

**Uniqueness:** **no classical baseline reproduces the DINO partition** (best ARI=0.1776, AMI=0.31) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map.

## Reading

Summary: The embedding most strongly encodes **scattered intensity, crystallinity (peak/halo), spottiness (azim-var)**; removing the overall **scattered intensity** collapses the map (ARI=0.0) — a major driver; the **1-D radial profile alone** is insufficient (ARI=0.068) — 2-D structure is needed; **low-q** matters more than high-q (ARI 0.2717 vs 0.3927); it is **not** based on crystallographic orientation (zone-axis AMI=0.0593); **no classical baseline reproduces the DINO partition** (best ARI=0.1776, AMI=0.31) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map.
