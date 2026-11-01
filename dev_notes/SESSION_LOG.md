# Session log — DINO4DSTEM + contrastive on 4D-STEM
*Rolling index of every run and the reason it was made. You can roll any
prior config back by loading the corresponding `best.pth`.*

## Code state (current)

```
dino_sr_contrastive/
  data.py                         minimal LoadPRZ + SAMPLES registry (no explainability dep)
  dino_sr_contrastive_model.py    ContrastiveDINOModel + training loop
  run_contrastive.py              config/driver wrapper
  contrastive_eval.py             NMI, KNN, stripe metric, class averages, dense remap, etc.
  viz_pipeline.py                 augmented-view gallery + class averages
  viz_gradcam.py                  GradCAM + Integrated Gradients
  suggest_k.py                    K recommendation (merge-gap + diagnostics)
  compare_maps.py                 side-by-side + IoU + Hungarian matching
  analyze_class.py                per-class deep dive (entropy, embeddings)
  analyze_imc.py                  IMC-specific radial profiles + polymorph hints
  make_paper_figures_imc.py       2 paper figures for IMC section
  run_overnight.py                multi-phase sweep driver (used once)
  run_euinas_3way.py              EuInAs 3-way ablation driver (current)
  regen_eval.py                   re-run eval-only on existing runs (dense remap)
```

All training loops save: `best.pth`, `latest.pth`, `ckpt_ep50.pth`, full `training_log.csv`, `eval/inference.npz`, `eval/metrics.json`, `eval/` figures, `eval/gradcam/` GradCAM + IG per prototype.

Intermediate `ckpt_ep10/20/30/40.pth` were deleted across all runs on 2026-04-22 to save ~310 MB — `best` + `latest` + `ckpt_ep50` are always preserved.

## Hyperparameter defaults (what "winner config" means)

```
architecture           : L1 ResNet-18 trunk (conv1 + bn1 + relu + layer1)
pipeline               : polar (PolarTransform + PolarMaskLeft + PolarThetaRoll)
circular_theta         : ThetaCircularConv2d wrapping every Conv2d → theta-equivariant trunk
theta_roll asymmetry   : student ±180° (shift_range=192), teacher ±15° (shift_range=16)
center mask            : polar_mask_cols=30 (~ 15 px inner radius)

teacher temp τ_t       : 0.04 → 0.07 warmup over first 10 epochs of 50 (canonical DINOv1)
student temp τ_s       : 0.1 constant
center momentum        : 0.9 (mean-EMA, NOT Sinkhorn — preserves fuzzy-c K ceiling)
teacher EMA momentum   : cosine 0.994 → 0.999
lr                     : 3e-4 AdamW, cosine decay, batch 128, 50 epochs
contrastive_lambda     : 0.2 pairwise, warmup 20, ramp 10
centroid_lambda        : 0.05, margin 0.3, inter weight 0.5
K ceiling              : 10 (the model can go lower via dead prototypes)
```

Tunable knobs added along the way (default 0 / off):
- `conf_weight_gamma` (confidence-weighted DINO loss)
- `entropy_gate_override` (entropy-gated contrastive pairs)
- `lam_spatial` (4-neighbor Potts on scan-adjacent pixels)

## Samples (from data.SAMPLES)

| key | material | shape | vmax | mask_r | scan step | note |
|---|---|---|---:|---:|---:|---|
| Na006a | NaPHI (layered photocatalyst) | (100, 100) | 4 | 15 | ~? | small flake at 0° tilt |
| Na007b | NaPHI | (126, 100) | 2 | 15 | ~? | **training sample** — mature flake at 0° |
| Na007a | NaPHI | (126, 100) | 2 | 15 | ~? | same flake as Na007b, **45° tilt** — for transfer |
| EuInAs_B100 | inorganic layered (EuInAs) | (66, 396) | 30 | 10 | ~? | layer bounds [22, 44] |
| IMC_50nm_SI2 | indomethacin, PVD 50nm + 70°C/60min | (128, 128) | 3 | 15 | 44 nm | probe 20 nm |
| IMC_150nm_SI5 | indomethacin, PVD 150nm + 70°C/60min | (128, 128) | 5 | 15 | 44 nm | probe 20 nm |

## Run index (runs/ subdirs + key metric + verdict)

### Na007b

| label | τ_t | K | flags | KNN | intra/inter | K_act | verdict |
|---|---|---:|---|---:|---:|---:|---|
| `config_a` | 0.07 const | 10 | λ=0, no θ-roll | 0.917 | 3.13 | 8-1dead | pure-DINO baseline |
| `config_b` | 0.07 const | 10 | λ=0, θ-roll ±180 sym | 0.902 | 3.41 | 10 | θ-roll alone gave modest gain |
| `config_c` | 0.07 const | 10 | λ=0.2, θ-roll ±180 sym | 0.956 | 16.78 | 10 | contrastive jumped intra/inter |
| `config_c_K6` | 0.07 const | 6 | as above | 0.978 | 15.37 | 6 | smaller K, cleaner |
| `config_c_K6_asym` | 0.07 const | 6 | asym rotation ±180/±15 | 0.974 | 13.71 | 6 | matches old pipeline asymmetry |
| `config_c_K6_asym_tempsched` | **0.04→0.07** | 6 | + schedule | 0.972 | 19.14 | 6 | **τ_t schedule big win** |
| `sweep_polar_centroid` | 0.04→0.07 | 6 | + centroid loss | 0.986 | 20.22 | 6 | **winner** (as of 2026-04-21 overnight) |
| `sweep_cart_nocent` | 0.04→0.07 | 6 | Cartesian pipeline, no centroid | 0.973 | 4.77 | 6 | polar is load-bearing |
| `sweep_cart_centroid` | 0.04→0.07 | 6 | Cartesian + centroid | 0.978 | 4.12 | 6 | Cartesian doesn't recover |

### EuInAs_B100

| label | τ_t | K | flags | KNN | intra/inter | K_act | agreement vs winner | verdict |
|---|---|---:|---|---:|---:|---:|---:|---|
| `winner_polar_centroid` | 0.04→0.07 | 10 | (winner) | 0.970 | 19.14 | 5 (5 dead) | — | current winner |
| `winner_conf_gated` | 0.04→0.07 | 10 | + γ=1 + entropy_gate | 0.981 | 8.54 | 7 | 53.6% | over-corrected (both filters) |
| `winner_gate_only` | 0.04→0.07 | 10 | + entropy_gate only | 0.964 | 4.60 | 7 (1 dead) | 81.7% | preserves layer structure but splits top into near-duplicates (cos 0.81) |
| **`winner_weight_only`** | **0.04→0.07** | **10** | **+ γ=1 only** | **0.994** | **24.63** | **5** | **66.3%** | **new leader on any separation metric. 5× intra/inter improvement. partition reshuffled** |
| `winner_spatial_only` | 0.04→0.07 | 10 | + lam_spatial=0.1 | 0.980 | 4.43 | 5 | 78.4% | preserves layer structure (IoU 0.98 top!) but doesn't sharpen centroids. stripe_max 271810 = cleaner thin line boundary |

### Na006a

| label | notes |
|---|---|
| `winner_polar_centroid` | winner config applied at K=10 → K_act=8, KNN 0.962, intra/inter 16.21, stripe_max 13.0 |

### EuInAs and IMC from overnight sweep

| sample | label | K_act | KNN | stripe_max | notes |
|---|---|---:|---:|---:|---|
| EuInAs_B100 | `winner_polar_centroid` | 5 (5 dead) | 0.970 | 997.8 | stripe=997 is the REAL layer boundary, not artifact |
| IMC_50nm_SI2 | `winner_polar_centroid` | 8 | 0.974 | 5.7 | ~73% amorphous (!) |
| IMC_150nm_SI5 | `winner_polar_centroid` | 10 | 0.962 | 10.5 | ~72% crystalline |

### Na007a (transfer from Na007b)

| label | notes |
|---|---|
| `transfer_from_winner` | loaded sweep_polar_centroid's best.pth, ran inference on Na007a (45° tilt), K=6 active, no dead, intra/inter=12.71. Same flake transfer working. |

## Cross-sample ablation — final results (2026-04-22 afternoon)

Tested 3 add-ons (entropy gate, conf_weight_gamma=1, lam_spatial=0.1) on 4 samples.
**None generalize.** Each helps only on a specific failure signature.

| config | EuInAs (KNN / intra/inter) | Na007b | Na006a | IMC_150nm |
|---|---|---|---|---|
| baseline (winner) | 0.970 / 5.14 | 0.986 / 21.5 | 0.962 / 16.0 | 0.962 / 11.1 |
| + γ=1 weight | **0.994 / 24.6** ✓ | 0.951 / 11.4 ✗ (K 6→9, dup cos=0.99) | 0.966 / 15.0 ≈ | 0.964 / 8.9 ≈ |
| + entropy gate | 0.964 / 4.60 ✗ | — | — | — |
| + both (γ+gate) | 0.981 / 8.54 ✗ (K=7) | — | — | — |
| + lam_spatial | 0.980 / 4.43 ≈ | 0.929 / 5.3 ✗ | — | 0.949 / 3.79 ✗ (37% in one class) |

**Decision: keep the baseline config as default. Add-ons are sample-adaptive
knobs documented as optional.**

## Next steps

1. (Optional) Try τ_t=0.04→0.06 on top of baseline as user suggested,
   if baseline is judged not-yet-clean-enough visually.
2. Build Figure 4 analog (Na006a / Na007b / Na007a) per user's reference.
3. Final IMC paper figures already exist in runs/IMC_comparison/

## Decisions + reasoning log

- **Kept mean-EMA centering** (not Sinkhorn-Knopp from DINOv2/v3). Reason: we rely on dead prototypes to reveal effective K; Sinkhorn would force all K active.
- **Asymmetric θ-roll ranges** (student ±180°, teacher ±15°) reintroduced after noticing the old validated pipeline used asymmetric `RandomRotation`.
- **τ_t schedule 0.04→0.07** wins over constant 0.07. Matches DINOv1 paper canon.
- **Contrastive head doesn't redraw clusters** — NMI(b vs a) ≈ NMI(c vs a). It sharpens embedding geometry only.
- **Polar is load-bearing** (confirmed by Cartesian ablation). Intra/inter drops 4× on Cartesian.
- **Confidence-weighted DINO + entropy gate combined is too aggressive** (over-clustered EuInAs from 5→7). Testing each alone now.
- **stripe metric (λ_max/λ_min of per-class scan-position covariance)** added to flag line-artifact prototypes. EuInAs's "997" is the real layer boundary, not a fault.

## What to roll back to

If any current direction doesn't pan out, safe fallbacks:

- Best Na007b checkpoint: `runs/Na007b/sweep_polar_centroid/best.pth` (KNN 0.986, intra/inter 20.22)
- Best EuInAs checkpoint: `runs/EuInAs_B100/winner_polar_centroid/best.pth`
- Consensus winner config: the one written in "Hyperparameter defaults" above (defaults = winner config with `conf_weight_gamma=0`, `entropy_gate_override=False`, `lam_spatial=0`).
