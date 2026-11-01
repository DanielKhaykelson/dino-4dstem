# DINO4DSTEM — session handoff

**Purpose**: everything a fresh session (or anyone picking up this project) needs to continue cleanly. Read this first. Then read `SESSION_LOG.md` for the full run index.

---

## Project one-liner

Self-distillation-based single-step clustering of 4D-STEM diffraction data, with rotation-invariant polar pipeline + contrastive head + prototype assignment. Target: arXiv preprint, then Communications Materials / M&M / Ultramicroscopy (NOT npj Comp Materials). User is Daniel Khaykelson (Weizmann); this is a follow-up to his Nano Lett. 2025 SAM paper on NaPHI.

**Method name: DINO4DSTEM** (not DINOp, not DINO-SR — those are deprecated).

---

## Current winner configs (per sample as of 2026-04-23 afternoon)

| sample | winner config (folder) | K_act | KNN | intra/inter | notes |
|---|---|---:|---:|---:|---|
| Na007b | `sweep_polar_centroid` (overnight 4/22) | 6 | 0.986 | 21.49 | default config, τ_t=0.04→0.07 |
| EuInAs_B100 | **`winner_t06_weight`** (t06 sweep result) | **6** | **0.987** | **23.28** | τ_t=0.04→0.06, γ=1, most-sim 0.12 ← NEW WINNER |
| Na006a | `winner_polar_centroid` | 8 | 0.962 | 16.0 | default |
| IMC_50nm_SI2 | `winner_polar_centroid` | 8 | 0.974 | 11.26 | default |
| IMC_150nm_SI5 | `winner_polar_centroid` | 10 | 0.962 | 11.13 | default |
| Na007a | `transfer_from_winner` (loaded from Na007b sweep_polar_centroid) | 6 | 0.979 | 12.71 | transfer, no retrain |

**User has NOT yet locked in winners — they said "I will tell you which are the winners" after all runs finish.** My current recommendation (subject to user override): use `sweep_polar_centroid` for Na007b, `winner_t06_weight` for EuInAs, default for rest.

---

## Background tasks still running

Check with `tail -3 runs/<logname>.log` to see progress.

| ID | purpose | log file | ETA from now |
|---|---|---|---|
| `b3xpv7o23` | post_t06 driver: NMF baselines on 3 samples + EuInAs K=12 + L2 ×4 samples + ViT Na007b + strain sweep on EuInAs K=12 + IMC within-film affine sweep | `runs/post_t06.log` | ~4 h |
| `bwo7ulcos` | no-beam-mask EuInAs 3-way (waits for post_t06 completion sentinel `runs/PAPER_SUMMARY_TABLE.json`). 3 EuInAs runs at `polar_mask_cols=0`: winner / +weight / +spatial. | `runs/no_beam_mask.log` | waits, then ~2 h |

Both tasks self-contained — they'll complete whether or not this conversation continues.

---

## Completed tasks with artifacts

| ID | what it produced |
|---|---|
| `btxvsrqcg` | t06 sweep — 3 Na007b + 3 EuInAs runs at τ_t=0.04→0.06. **EuInAs t06+γ=1 is the new winner for that sample.** Na007b t07 default still wins. |
| `bdxnm20cs` | NMF+kmeans baseline Na007b. Silhouette picks K=2 (trivial); at K=6 gives KNN 0.872, intra/inter 1.81. |
| `ba7jtsjon` | NMF+agglomerative+polar-cross-correlation baseline Na007b (Yoo 2024 style). Silhouette picks K=6 (correct); KNN 0.880, intra/inter 1.56. **~14× worse intra/inter than DINO4DSTEM.** |

---

## Key findings (critical to preserve)

1. **Core winner config** (applies to most samples): polar pipeline + circular-θ Conv2d + asymmetric θ-roll (student ±180°/teacher ±15°) + contrastive head λ=0.2 + centroid head λ_cent=0.05 + τ_t schedule 0.04→0.07 + L1 ResNet backbone + K=10 ceiling.

2. **EuInAs needs different knobs**: τ_t=0.04→0.06 + γ=1 (confidence-weighted DINO loss) gives KNN 0.987 and intra/inter 23.3 — dramatic improvement over the default winner. This is sample-adaptive, not universal.

3. **Spatial regularization (lam_spatial) is NOT a universal fix** — it helps visually on some samples but creates dead prototypes on Na007b. Document as optional.

4. **Confidence-weighted DINO loss (γ=1)**: sharpens centroids on EuInAs (bimodal teacher confidence), regresses slightly on NaPHI (continuous distribution). Sample-adaptive.

5. **NMF+k-means comparison on Na007b**: silhouette picks K=2 (trivial), intra/inter 1.81 at matched K=6. NMF+agglomerative+polar-cross-correlation (Yoo 2024 style) picks K=6 correctly but intra/inter only 1.56. **DINO4DSTEM delivers ~14× better cluster geometry**.

6. **EuInAs strain pipeline** (blob+RANSAC affine): works, but current K=5 winner doesn't surface orientation domains clearly. K=12 run in post_t06 will retry.

7. **Na007a transfer** (tilted view of same Na007b flake): KNN 0.979 with zero retraining. Published claim.

8. **t06 doesn't generalize**: sharper teacher schedule helps EuInAs but hurts Na007b (over-activates prototypes from 6 to 9). Reject as universal default.

---

## File organization

```
dino_sr_contrastive/
├── HANDOFF.md                        ← you are here
├── SESSION_LOG.md                    ← full run index + decisions
├── MANUSCRIPT_DRAFT.md               ← root-level manuscript (superseded by paper/)
│
├── paper/
│   ├── manuscript.md                 ← current draft, iteratively updated
│   ├── manuscript.docx               ← Word export of current draft
│   ├── manuscript_v1.docx            ← newer Word export (when v0 is open)
│   ├── references.bib                ← BibTeX with 18 refs
│   ├── md_to_docx.py                 ← converter (python-docx based)
│   └── README.md                     ← paper-package instructions
│
├── paper_figures/                    ← 300 DPI PNGs for arXiv
│   ├── fig1_naphi_benchmark.png       (Na006a / Na007b / Na007a transfer rows, Fig 4 Khaykelson 2025 layout)
│   ├── fig2_euinas_domains.png        (EuInAs class map + 2 class means + diff)
│   ├── fig4_imc_crystal_identity.png  (IMC 50/150nm class maps + radial profiles)
│   ├── fig5_nmf_vs_dino4dstem_Na007b.png (head-to-head baseline)
│   ├── figS1_schematic.png            (DINO4DSTEM architecture block diagram)
│   ├── figS2_attribution_summary.png  (GradCAM + IG)
│   └── figS3_backbone_ablation.png    (L1 vs L2 vs ViT)
│
├── runs/
│   ├── SESSION_LOG.md                ← alternate location of session log
│   ├── PAPER_SUMMARY_TABLE.json       ← summary of every run (post_t06 writes this)
│   ├── <sample>/<config>/            ← one directory per (sample, config)
│   │   ├── best.pth, latest.pth, ckpt_ep50.pth
│   │   ├── training_log.csv
│   │   ├── run_summary.json
│   │   └── eval/
│   │       ├── metrics.json           ← canonical metrics per run
│   │       ├── inference.npz          ← assigns, soft_probs, embeds
│   │       ├── fig_class_map.png
│   │       ├── fig_class_averages.png
│   │       ├── fig_umap.png, fig_centroid_cosine.png, etc.
│   │       ├── gradcam/fig_gradcam_p*.png (per prototype GradCAM + IG)
│   │       └── (optional) fig_compare_*_vs_*.png from compare_maps
│   │   └── (optional) strain_p*_vs_p*/  sub-directory of strain analysis outputs
│   └── IMC_comparison/                ← cross-thickness IMC analysis figures + IMC_REPORT.md
│
├── dino_sr_contrastive_model.py      ← ContrastiveDINOModel + train_contrastive + ThetaCircularConv2d + PolarThetaRoll + all loss terms
├── run_contrastive.py                 ← run_config() + evaluate_and_report() top-level driver
├── contrastive_eval.py                ← NMI, KNN purity, stripe metric, class averages, dense remap, etc.
├── viz_pipeline.py                    ← augmented-view gallery + class averages
├── viz_gradcam.py                     ← GradCAM + Integrated Gradients
├── suggest_k.py                       ← K recommendation
├── compare_maps.py                    ← IoU + Hungarian matching
├── analyze_class.py                   ← per-class deep dive
├── analyze_imc.py                     ← IMC-specific radial profiles + polymorph hints
├── analyze_strain.py                  ← LoG blob + RANSAC affine + decomposition (EuInAs C3/C5-style)
├── baseline_nmf_kmeans.py             ← NMF + k-means baseline (Na007b done)
├── baseline_nmf_agglomerative.py      ← NMF + polar-cross-correlation + agglomerative (Yoo 2024 style) (Na007b done)
├── generate_paper_figures.py          ← paper figure generator
├── generate_schematic.py              ← Fig S1 schematic
├── make_paper_figures_imc.py          ← IMC-specific figure set
├── data.py                            ← minimal SAMPLES + LoadPRZ
│
├── run_overnight.py                   ← first overnight driver (historical)
├── run_euinas_3way.py                 ← EuInAs 3-way ablation (done, weight-only winner for EuInAs was established)
├── run_spatial_followups.py           ← spatial + lam_spatial sweep (done; didn't generalize)
├── run_weight_generalization.py       ← gamma=1 generalization to other samples (done; EuInAs-only)
├── run_final_queue.py                 ← final queue after weight gen (done)
├── run_t06_sweep.py                   ← t06 vs t07 sweep (done; EuInAs new winner here)
├── run_post_t06.py                    ← post-t06 master driver (RUNNING)
└── run_no_beam_mask.py                ← EuInAs no-beam-mask 3-way (QUEUED, waits for post_t06)
```

---

## Python environment

```
python   : C:\Users\danielkh\AppData\Local\anaconda3\envs\py4DSTEM_SAM\python.exe
torch    : 2.7.1+cu118  (CUDA available on RTX 4080 16GB)
numpy    : 1.26.4
deps    : scikit-learn, scikit-image (blob_log, ransac), umap-learn, matplotlib, python-docx
platform : Windows, git-bash shell
data path: D:\DINOSR\data\ (local copies of all .prz)
```

All training loops set `matplotlib.use("Agg", force=True)` at top to avoid Tcl/tkinter crashes from headless matplotlib. Console encoding set via `sys.stdout.reconfigure(encoding="utf-8")` at top of drivers to avoid cp1252 crashes on non-ASCII (esp. `↔`, `γ`).

---

## Pending for completion (in order)

1. Wait for `b3xpv7o23` (post_t06) to finish. Writes `runs/PAPER_SUMMARY_TABLE.json` when done.
2. `bwo7ulcos` (no-beam-mask) will auto-start when post_t06 sentinel appears.
3. When no-beam-mask finishes, **user will review winners and tell us which to lock in.**
4. Run `generate_paper_figures.py` + `generate_schematic.py` to regenerate all figures with final data.
5. Fill remaining manuscript placeholders (EuInAs K=12 strain numbers, L2/ViT ablation, IMC affine polymorph table).
6. Re-export `.docx`: `python paper/md_to_docx.py paper/manuscript.md paper/manuscript_final.docx`
7. Set up arXiv submission: convert markdown → LaTeX, flesh out references, data availability, acknowledgments, author list.

---

## Things user has explicitly decided

- **Method name**: DINO4DSTEM (final).
- **Target venue**: arXiv first. Then Communications Materials preferred; M&M or Ultramicroscopy as backup. **NOT npj Comp Materials** (explicit rejection).
- **IMC framing**: crystal-level identity comparison (same/different polymorph), NOT amount/statistics percentages.
- **Audience**: EM / materials-science community, not ML venues. Tone is "complement to NMF+k-means, not replacement".
- **Respect NMF**: "NMF is older and still the best method available" — don't write the paper as if DINO4DSTEM dunks on NMF.
- **Explainability is important**: GradCAM + IG are mandatory in the paper.
- **Temperature schedule**: 0.04→0.07 is default; 0.04→0.06 only for specific bimodal-confidence samples (EuInAs).
- **Writing style**: reviewer #2 perspective — no fluff, honest about limitations, every claim defensible.
- **Figures**: Fig 4 from Khaykelson 2025 Nano Lett is the template. Row 1 = Na006a, Row 2 = Na007b, Row 3 = Na007a (transfer).
- **ViT ablation**: mandatory, show it's worse than CNN.
- **L2 backbone**: offer as option but don't change default from L1 (since L1 works).

---

## Things still open for user decision

1. Which EuInAs config to use for paper Figure 2/3 — current winner (`winner_polar_centroid`, K=5) or new winner (`winner_t06_weight`, K=6)? I believe new winner is better (dramatically tighter centroids) but user will decide.
2. Whether to include the `no_beam_mask` result in main text or supplement once that finishes.
3. Whether to add a head-to-head class map comparison with Yoo 2025 paper's method or leave that as a reference-only citation.
4. Final title.
5. Author list + acknowledgments + affiliations.

---

## Known bugs / caveats

- Windows cp1252 console can't print UTF-8 `↔` or `γ`. I've replaced these in all printed output. If re-writing, keep ASCII in prints.
- `skimage.measure.ransac` doesn't accept `random_state` kwarg in this version; we seed numpy globally before the call.
- `analyze_strain.py` diff_mean_abs doesn't always decrease after warp even when the fit converges — means the class pair isn't orientation-related despite visually similar radial profiles. This is a FEATURE (no false positives), not a bug.
- `polar_mask_cols=0` disables beam mask cleanly (tested).
- t06 on Na007b over-activates (K 6→9) — use t07 on non-EuInAs samples.

---

## If this conversation gets compressed

Everything above is self-sufficient to continue. Do:
1. Read `HANDOFF.md` (this file).
2. Read `SESSION_LOG.md` for the full run index.
3. Check `runs/PAPER_SUMMARY_TABLE.json` for a machine-readable summary of every run.
4. Check the background-task status via `tail -5 runs/<logname>.log`.
5. If the user tells you "pick up where we left off", the default next action is: wait for remaining background tasks, then regen figures + fill manuscript placeholders + re-export docx.
