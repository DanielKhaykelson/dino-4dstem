# Interpretation report — loaded__Na007b_nbed.cube

Run: `Na007b_k60_m097`  ·  K_active = 14  ·  N = 12600

## What the embedding encodes / separates (probing)

| factor | probe R² | η² | MI |
|---|---|---|---|
| scattered intensity | 0.9391 | 0.9004 | 0.8744 |
| spottiness (azim-var) | 0.9173 | 0.9261 | 0.7567 |
| crystallinity (peak/halo) | 0.8558 | 0.8552 | 0.8048 |


## Causal ablations (ARI vs original map)

| ablation | ARI | K |
|---|---|---|
| baseline | 1.0 | 14 |
| scattered_norm | 0.167 | 9 |
| radial_only | 0.1943 | 8 |
| blur(s=2) | 0.381 | 9 |
| qmask_low | 0.1521 | 5 |
| qmask_high | 0.9964 | 14 |

High ARI = that information was not needed; low ARI = the model depends on it.

## Could a classical method reproduce the DINO map?

| classical feature | ARI vs DINO | AMI vs DINO |
|---|---|---|
| pattern PCA (linear decomp.) | 0.5161 | 0.4151 |
| pattern NMF (non-neg decomp.) | 0.5094 | 0.4278 |
| azimuthal profile (radial) → PCA | 0.5069 | 0.4282 |
| combination (DF + radial + pattern) | 0.4905 | 0.4136 |
| virtual DF (total scattered I) | 0.4495 | 0.372 |

**Uniqueness:** classical features **substantially capture** the partition (best ARI=0.5161, AMI=0.43); DINO refines them but does not depart strongly — it behaves like a non-linear blend of classical descriptors.

## Reading

Summary: The embedding most strongly encodes **scattered intensity, crystallinity (peak/halo), spottiness (azim-var)**; removing the overall **scattered intensity** collapses the map (ARI=0.167) — a major driver; the **1-D radial profile alone** is insufficient (ARI=0.1943) — 2-D structure is needed; **low-q** matters more than high-q (ARI 0.1521 vs 0.9964); classical features **substantially capture** the partition (best ARI=0.5161, AMI=0.43); DINO refines them but does not depart strongly — it behaves like a non-linear blend of classical descriptors.

> **ACOM not available** for this run, so the orientation/phase factors and the zone-axis cross-check were skipped. Run ACOM (Diffraction ▸ ACOM, full-dataset mode) first to include them.
