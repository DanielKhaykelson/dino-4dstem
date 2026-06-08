# Interpretation report — loaded__Survey_CH2_0_1_nbed.cube

Run: `IMC_SI4_m097_k60`  ·  K_active = 9  ·  N = 16384

## What the embedding encodes / separates (probing)

| factor | probe R² | η² | MI |
|---|---|---|---|
| crystallinity (peak/halo) | 0.8016 | 0.5729 | 0.4631 |
| scattered intensity | 0.6986 | 0.2311 | 0.2925 |
| spottiness (azim-var) | 0.6146 | 0.2894 | 0.4704 |
| ACOM correlation | 0.1331 | 0.1191 | 0.0755 |

ACOM zone-axis vs classes: AMI=0.0606, ARI=0.0113 (1589 indexed, 137 zone axes).

## Causal ablations (ARI vs original map)

| ablation | ARI | K |
|---|---|---|
| baseline | 1.0 | 9 |
| scattered_norm | 0.013 | 8 |
| radial_only | 0.106 | 6 |
| blur(s=2) | 0.2607 | 8 |
| qmask_low | 0.2501 | 6 |
| qmask_high | 0.575 | 9 |

High ARI = that information was not needed; low ARI = the model depends on it.

## Could a classical method reproduce the DINO map?

| classical feature | ARI vs DINO | AMI vs DINO |
|---|---|---|
| pattern PCA (linear decomp.) | 0.1499 | 0.2521 |
| combination (DF + radial + pattern) | 0.1396 | 0.2224 |
| pattern NMF (non-neg decomp.) | 0.1182 | 0.2132 |
| azimuthal profile (radial) → PCA | 0.0912 | 0.1753 |
| virtual DF (total scattered I) | 0.0651 | 0.1254 |

**Uniqueness:** **no classical baseline reproduces the DINO partition** (best ARI=0.1499, AMI=0.25) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map.

## Reading

Summary: The embedding most strongly encodes **scattered intensity, crystallinity (peak/halo), spottiness (azim-var)**; removing the overall **scattered intensity** collapses the map (ARI=0.013) — a major driver; the **1-D radial profile alone** is insufficient (ARI=0.106) — 2-D structure is needed; **low-q** matters more than high-q (ARI 0.2501 vs 0.575); it is **not** based on crystallographic orientation (zone-axis AMI=0.0606); **no classical baseline reproduces the DINO partition** (best ARI=0.1499, AMI=0.25) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map.
