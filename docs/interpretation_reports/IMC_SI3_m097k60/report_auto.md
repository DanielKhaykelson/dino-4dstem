# Interpretation report — loaded__Survey_CH2_1_nbed.cube

Run: `IMC_SI3_m097k60`  ·  K_active = 12  ·  N = 16384

## What the embedding encodes / separates (probing)

| factor | probe R² | η² | MI |
|---|---|---|---|
| scattered intensity | 0.5868 | 0.3882 | 0.4079 |
| crystallinity (peak/halo) | 0.5687 | 0.4618 | 0.3892 |
| spottiness (azim-var) | 0.5434 | 0.4046 | 0.4675 |
| ACOM correlation | 0.0726 | 0.0658 | 0.0619 |

ACOM zone-axis vs classes: AMI=0.0401, ARI=0.0076 (2993 indexed, 182 zone axes).

## Causal ablations (ARI vs original map)

| ablation | ARI | K |
|---|---|---|
| baseline | 1.0 | 12 |
| scattered_norm | 0.1104 | 12 |
| radial_only | 0.0771 | 3 |
| blur(s=2) | 0.1596 | 7 |
| qmask_low | 0.3202 | 10 |
| qmask_high | 0.28 | 12 |

High ARI = that information was not needed; low ARI = the model depends on it.

## Could a classical method reproduce the DINO map?

| classical feature | ARI vs DINO | AMI vs DINO |
|---|---|---|
| pattern NMF (non-neg decomp.) | 0.1148 | 0.2422 |
| azimuthal profile (radial) → PCA | 0.1129 | 0.2187 |
| pattern PCA (linear decomp.) | 0.0991 | 0.1994 |
| combination (DF + radial + pattern) | 0.0943 | 0.2323 |
| virtual DF (total scattered I) | 0.0844 | 0.1634 |

**Uniqueness:** **no classical baseline reproduces the DINO partition** (best ARI=0.1148, AMI=0.24) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map.

## Reading

Summary: The embedding most strongly encodes **scattered intensity, crystallinity (peak/halo), spottiness (azim-var)**; removing the overall **scattered intensity** collapses the map (ARI=0.1104) — a major driver; the **1-D radial profile alone** is insufficient (ARI=0.0771) — 2-D structure is needed; it is **not** based on crystallographic orientation (zone-axis AMI=0.0401); **no classical baseline reproduces the DINO partition** (best ARI=0.1148, AMI=0.24) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map.
