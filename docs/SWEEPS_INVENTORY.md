# Sweeps & ablations inventory

Survey from `D:\DINOSR\Claude` down to `...\PaperRun_claude\dino_sr_contrastive\runs`.

**Note on terminology:** the DINO encoder *is* a truncated ResNet-18; "depth" = number of
ResNet stages kept (`n_layers` = L1–L4). The "DINO vs ResNet-18 at 4 depths" comparison is
the DINO-encoder depth ablation (L1–L4). The only non-DINO baseline present is classical
NMF (cepstral), not a plain ResNet classifier.

## Master table

| # | Sweep / ablation | Location (under `D:\DINOSR\Claude\`) | Samples | What is varied | #runs | Key outputs |
|---|---|---|---|---|---|---|
| 1 | Depth × temperature × augmentation (resize 192) | `AblationSweep\ABLATION_TEMP_SWEEP_resize192\` | EuInAs_B100, Na007b | depth L1–L4; teacher-temp (const 0.65/0.70/0.75, cool 0.80→0.65/0.70, heat 0.40→0.60 / 0.60→0.70 / 0.70→0.75); per-aug leave-one-out @L2 (noBlur, noCenterMask, noColorJitter, noHFlip, noRotation, noVFlip, allaugs) | 78 | `master_summary.csv/.json`, `heatmap_*`, `best_per_depth_*`, `aug_ablation_*` |
| 2 | Depth × temperature × augmentation (resize 256) | `AblationSweep\ABLATION_TEMP_SWEEP_resize256\` | EuInAs_B100, Na007b, PLA_NoAnn_SI05 | same grid as #1 | 117 | same as #1 |
| 3 | CBAM attention ablation | `AblationSweep_CBAM\ABLATION_SWEEP\` | EuInAs_B100, Na007b | CBAM on/off at depth L1–L4; per-aug leave-one-out @L2_CBAM | 30 | `master_summary.csv/.json`, `depth_cbam_comparison_*`, `grand_comparison_*`, `aug_ablation_comparison_*` |
| 4 | **Latest sweep:** momentum × K (2-stage) | `...\dino_sr_contrastive\runs\_sweep_m_K_20260525_213539\` | Na007b, EuInAs_B100, IMC_SI5 | **Stage1:** m ∈ {0.5,0.85,0.9,0.95,0.97,0.99,0.995,0.999}, K=30, seeds {42,7}. **Stage2:** top-3 m × K ∈ {10,15,30,60,120}, seed 42. 30 epochs | 240 (118 leaf models) | `SWEEP_SPEC.json`, `SWEEP_PROGRESS.csv`, `cross_sample_report.html`, `_cross_figs\` |
| 4b | m×K sweep (aborted predecessors) | `runs\_sweep_m_K_20260525_180932\`, `..._181055\` | Na007b only | partial momentum sweep (superseded by #4) | partial | `SWEEP_SPEC.json`, `STOP_SWEEP` |
| 5 | K=6 loss ablation (vanilla vs cluster1d-weight) | `runs\_paper_ablation_K6\` | EuInAs_B100, Na007b, Na007a, Na006a, IMC_50nm_SI2, IMC_150nm_SI5 | γ=0 (vanilla) vs γ=1 (radial-weight); ep5 discriminator metrics | 6 samples × 2 = 12 | `REPORT.md`, `fig_ablation_metrics.png`, `fig_ep5_discriminator.png`, `fig_class_balance.png` |
| 6 | cluster1d-λ sweep | `runs\_dino_c1d_sweep\` | (DINO 1-D radial loss strength) | cluster1d_lambda | 2 | per-run eval |
| 7 | SupCon loss-design sweeps | `runs\_supcon_sweep\` (4), `_supcon_lg_sweep\` (2), `_supcon_newgate\` (2), `_supcon_repel\` (2), `_followup_supcon\` (4) | mixed | supervised-contrastive gate / large-batch / new-gate / repel variants | 14 total | per-run eval |
| 8 | Per-family runs | `runs\_per_family\` (2), `_per_family_v5\` (2) | per material family | family-grouped training | 4 | per-run eval |
| 9 | Classical baseline (non-DINO) | `runs\_baselines\` | EuInAs_B100, Na007b | cepstral-NMF at K=6 and K=auto | 4 | baseline class maps |
| 10 | EuInAs follow-ups | `runs\_followup_EuInAs_K10_anticollapse\`, `_halfmask\`, `_minaug_com\` | EuInAs | anti-collapse / half-mask / min-aug+COM | 3 dirs | per-run eval |
| 11 | First DINO sweep | `DINO_SWEEP_1strun\` | Na006a, Na007a, Na007b | early DINO hyperparameter sweep | 3 sample-dirs | per-run eval |
| 12 | Temperature sweeps (early) | `TempSweep\` → DINO_SWEEP, DINO_SWEEP_Eu100_0p04to0p06best, DINO_SWEEP_Eu100_111_one, DINO_SWEEP_Na_Mg | EuInAs, Na, Mg | teacher-temp exploration | 4 sub-sweeps | per-run eval |
| 13 | k-sweep (scripts only) | `k-sweep\` | — | `ablation_sweep.py`, `dino_sr_ablation.py`, `explainability.py` (+`k-sweep.zip`) | scripts | — |

## CSV columns (ablations #1–#3)
`sample, run_name, label, n_layers, [use_cbam], disable_aug, [T0, Tfin, warmup_frac], checkpoint, eval_temp,`
`effK, n_active, avg_conf, auto_score, total_score, spatial_coherence, flip_rate, sharpness, max_sim,`
`total_params, trainable_params, train_time_s, train_time_per_epoch_s, inference_time_s, active_classes,`
`boundary_frac, conf_bimodality, conf_median, conf_p10, conf_p90, eff_k, intensity_corr, isolated_disagree,`
`loss, max_class_flip, mdp_max_xcorr, spatial_flip`

## Latest m/K sweep — progress columns
`timestamp, sample, stage, m, K, seed, outdir, ok, K_eff_end_smooth, n_live_end, avg_conf_e5,`
`loss_final, effective_rank, has_nan, train_time_s, error`

## Reports already on disk (deep `runs\`)
`OVERNIGHT_REPORT.md`, `EUINAS_3WAY_REPORT.md`, `PAPER_SUMMARY_TABLE.json`, `inference_benchmark.json`,
`_paper_ablation_K6\REPORT.md`.
