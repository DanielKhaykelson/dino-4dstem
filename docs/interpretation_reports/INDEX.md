# Interpretation reports — all runs

Auto-collected by `tools/collect_interpretation_reports.py`. Each subfolder mirrors that run's `_interpretability/` (report + figures).

| sample | run | K | N | classical-uniqueness verdict |
|---|---|---|---|---|
| EuInAs B100 | [EuInAs_B100__m0.9700_seed42_K60](EuInAs_B100__m0.9700_seed42_K60/report_auto.md) | 9 | 26136 | classical features substantially capture the partition (best ARI=0.4373, AMI=0.61); DINO refines them but does not depart strongly — it behaves like a non-linear blend of classical descriptors. |
| IMC SI3 | [IMC_SI3_m097k60](IMC_SI3_m097k60/report_auto.md) | 12 | 16384 | no classical baseline reproduces the DINO partition (best ARI=0.1148, AMI=0.24) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map. |
| IMC SI3 | [IMC_SI3_m097k60_logstretch](IMC_SI3_m097k60_logstretch/report_auto.md) | 14 | 16384 | no classical baseline reproduces the DINO partition (best ARI=0.1776, AMI=0.31) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map. |
| IMC SI4 | [IMC_SI4_m097_k60](IMC_SI4_m097_k60/report_auto.md) | 9 | 16384 | no classical baseline reproduces the DINO partition (best ARI=0.1499, AMI=0.25) — the clustering is distinctive, not a re-labelling of a virtual-DF / radial / PCA / NMF map. |
| IMC SI5 | [IMC_SI5__m0.9700_seed42_K60](IMC_SI5__m0.9700_seed42_K60/report.md) | 13 | ? | see report |
| NaPHI Na007b | [Na007b_k60_m097](Na007b_k60_m097/report_auto.md) | 14 | 12600 | classical features substantially capture the partition (best ARI=0.5161, AMI=0.43); DINO refines them but does not depart strongly — it behaves like a non-linear blend of classical descriptors. |
| NaPHI Na007b | [Na007b_k60_m097_vmax2](Na007b_k60_m097_vmax2/report_auto.md) | 12 | 12600 | classical features substantially capture the partition (best ARI=0.4791, AMI=0.5); DINO refines them but does not depart strongly — it behaves like a non-linear blend of classical descriptors. |

## Documents (`_documents/`)

- [IMC_SI5_interpretability_manuscript.pdf](_documents/IMC_SI5_interpretability_manuscript.pdf)
- [cross_sample_summary.pdf](_documents/cross_sample_summary.pdf)
- [explainer_for_PI.pdf](_documents/explainer_for_PI.pdf)
- [explainer_for_PI.pptx](_documents/explainer_for_PI.pptx)

## Notes

- *distinctive* = classical methods (virtual-DF / radial / PCA / NMF) do **not** reproduce the DINO map; *substantially captured* = they get most of the way.
- All runs share the same physical basis: scattered intensity + low-q 2-D diffraction structure; not orientation.
- IMC_SI5 carries a hand-curated `report.md` (+ manuscript) rather than the auto report.