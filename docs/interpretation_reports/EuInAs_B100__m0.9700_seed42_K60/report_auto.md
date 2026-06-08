# Interpretation report — EuInAs_B100

Run: `m0.9700_seed42_K60`  ·  K_active = 9  ·  N = 26136

## What the embedding encodes / separates (probing)

| factor | probe R² | η² | MI |
|---|---|---|---|
| spottiness (azim-var) | 0.9277 | 0.7438 | 1.0655 |
| scattered intensity | 0.9262 | 0.8664 | 1.0811 |
| crystallinity (peak/halo) | 0.8982 | 0.7745 | 0.791 |
| ACOM correlation | 0.0976 | 0.1029 | 0.1638 |

ACOM phase vs classes: AMI=0.0828, ARI=0.0449.
ACOM zone-axis vs classes: AMI=0.3048, ARI=0.1424 (4448 indexed, 63 zone axes).

## Causal ablations (ARI vs original map)

| ablation | ARI | K |
|---|---|---|
| baseline | 1.0 | 9 |
| scattered_norm | 0.0 | 1 |
| radial_only | 0.3825 | 3 |
| blur(s=2) | 0.3575 | 5 |
| qmask_low | 0.0067 | 2 |
| qmask_high | 0.9953 | 9 |

High ARI = that information was not needed; low ARI = the model depends on it.

## Could a classical method reproduce the DINO map?

| classical feature | ARI vs DINO | AMI vs DINO |
|---|---|---|
| combination (DF + radial + pattern) | 0.4373 | 0.6131 |
| azimuthal profile (radial) → PCA | 0.4117 | 0.5982 |
| pattern NMF (non-neg decomp.) | 0.4077 | 0.6054 |
| pattern PCA (linear decomp.) | 0.3444 | 0.5919 |
| virtual DF (total scattered I) | 0.3202 | 0.5252 |

**Uniqueness:** classical features **substantially capture** the partition (best ARI=0.4373, AMI=0.61); DINO refines them but does not depart strongly — it behaves like a non-linear blend of classical descriptors.

## Reading

Summary: The embedding most strongly encodes **scattered intensity, crystallinity (peak/halo), spottiness (azim-var)**; removing the overall **scattered intensity** collapses the map (ARI=0.0) — a major driver; **low-q** matters more than high-q (ARI 0.0067 vs 0.9953); classical features **substantially capture** the partition (best ARI=0.4373, AMI=0.61); DINO refines them but does not depart strongly — it behaves like a non-linear blend of classical descriptors.
