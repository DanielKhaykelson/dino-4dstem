# Interpretation report — IMC_SI5

Run: `m0.9700_seed42_K60`  ·  K_active = 13  ·  N = 16384

## What the embedding encodes / separates (probing)

| factor | probe R² | η² | MI |
|---|---|---|---|
| scattered intensity | 0.7054 | 0.3396 | 0.3856 |
| crystallinity (peak/halo) | 0.237 | 0.1505 | 0.1145 |
| spottiness (azim-var) | 0.1549 | 0.0191 | 0.0592 |
| n Bragg peaks | -0.0031 | 0.0087 | 0.0078 |
| ACOM corr | -0.0064 | 0.0065 | 0.0077 |

ACOM phase vs classes: AMI=0.0007, ARI=-0.0005.
ACOM zone-axis vs classes: AMI=0.0512, ARI=0.0172 (367 indexed, 43 zone axes).

## Causal ablations (ARI vs original map)

| ablation | ARI | K |
|---|---|---|
| baseline | 1.0 | 13 |
| radial_only | 0.0821 | 4 |
| blur(s=2) | 0.2733 | 7 |
| perframe_norm | 1.0 | 13 |
| qmask_low | 0.2281 | 11 |
| qmask_high | 0.5311 | 13 |
| scattered_norm | 0.0142 | 8 |

High ARI = that information was not needed; low ARI = the model depends on it.

## Could a classical method reproduce the DINO map?

| classical feature | ARI vs DINO | AMI vs DINO |
|---|---|---|
| pattern NMF (non-neg decomp.) | 0.2144 | 0.2737 |
| pattern PCA (linear decomp.) | 0.2127 | 0.2225 |
| azimuthal profile (radial) → PCA | 0.1962 | 0.2315 |
| combination (DF + radial + pattern) | 0.1714 | 0.2978 |
| virtual DF (total scattered I) | 0.1031 | 0.1563 |

**Uniqueness:** **no classical baseline reproduces the DINO partition** (best ARI=0.2144, AMI=0.3) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map.

## Reading

Summary: The embedding most strongly encodes **crystallinity (peak/halo), scattered intensity**; removing the overall **scattered intensity** collapses the map (ARI=0.0142) — a major driver; the **1-D radial profile alone** is insufficient (ARI=0.0821) — 2-D structure is needed; **low-q** matters more than high-q (ARI 0.2281 vs 0.5311); it is **not** based on crystallographic orientation (zone-axis AMI=0.0512); **no classical baseline reproduces the DINO partition** (best ARI=0.2144, AMI=0.3) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map.

Grad-CAM / IG per-prototype attention written to `eval/paper_attribution/`.
