# Paper reproducibility document — DINO+cluster1d 4D-STEM phase mapping

This file is intended to be a self-contained record of everything used to
generate the figures and numbers in the manuscript. It documents the
model, the training and inference recipes, every analysis script, the
data registry, and the file layout. A reader 10 years from now should be
able to clone the repository, restore the conda environment, and
reproduce every result by following the per-experiment recipes at the
end.

Last updated: 2026-04-29.

---

## 1. Project goal

Unsupervised phase mapping of 4D-STEM diffraction cubes via self-supervised
representation learning (DINO) regularised by a physics-informed clustering
term (cluster1d) that uses a 1-D radial profile of each diffraction pattern.
The paper claims:

1. The trained model produces interpretable, balanced prototype
    assignments of diffraction patterns into "phases" (= clusters)
    without labels.
2. The model is rotation-invariant by construction (θ-circular
    convolutions + 360° θ-roll augmentation), so grain orientations
    collapse into a single prototype rather than fragmenting.
3. **Transfer**: a single trained model can be applied (forward only) to
    held-out cubes / samples in seconds (~3 ms / pattern), enabling
    multi-sample studies without per-sample retraining.
4. We use this transfer property to flag a chemistry-correlated outlier
    (MgNaPHI SI-007 with EDS-confirmed reduced Mg) on the basis of
    diffraction-pattern morphology alone.

---

## 2. Environment

- OS: Windows 10 (development & training).
- Conda env: `py4DSTEM_SAM`. Key packages:
    - Python 3.10
    - PyTorch 2.7.1 + CUDA 11.8 (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`)
    - numpy, scipy, scikit-learn, scikit-image
    - matplotlib, opencv-python, tqdm
    - umap-learn (used in eval_and_report)
    - py4DSTEM (only for old eval_all paths; not strictly required for
      the modern pipeline)
- GPU: single consumer GPU (e.g. RTX 3060). All training fits in 8 GB.

Anaconda Prompt activation:
```
call C:\Users\<user>\AppData\Local\anaconda3\Scripts\activate.bat py4DSTEM_SAM
cd /d D:\DINOSR\Claude\PaperRun_claude\dino_sr_contrastive
```

Python invocation in scripts uses the explicit interpreter path:
`C:/Users/<user>/AppData/Local/anaconda3/envs/py4DSTEM_SAM/python.exe`.

---

## 3. Data registry & formats

All cubes live under `D:\DINOSR\data\`. Each cube is a `.prz` (numpy
compressed) file containing a key `data` of shape `(Ny, Nx, H, W)`,
`float32`. For very large cubes a `.cube.npy` sidecar (true mmap) can be
created next to the `.prz` to avoid full-RAM loads — see `data.LoadPRZ`.

The registry lives in `data.py` as a `SAMPLES` dict, keys → metadata:

```python
SAMPLES["Na007b"] = {
    "path": "D:/DINOSR/data/Na007b.prz",
    "vmax": 2,                      # intensity normalisation cap
    "scan_shape": (126, 100),       # (Ny, Nx)
    "center_mask_radius": 15,       # px in 192-resized polar
    "approved_label": "...",
}
```

Multi-cube training sets (used to train per-family models) carry
`"is_multi": True` and `"paths": [...]` instead of `"path"`. The
`run_contrastive.run_config` function reads `is_multi` to instantiate
either `LoadPRZ` or `LoadPRZMulti`.

vmax conventions used in this paper:
- Na series, NaPHI, MgNaPHI training: `vmax = 5` (the v5 pipeline).
- Older paper-master runs (Na007a/b, Na006a, IMC×2, EuInAs): `vmax = 2`
  to `vmax = 30` per registry — see `data.SAMPLES` source.

---

## 4. Pre-processing pipeline

### 4.1 Per-pattern (training & inference)

Implemented in `dino_sr_contrastive_model.PolarTransform`,
`PolarMaskLeft`, `dino_sr_ablation.CenterOnCOM`, applied via a
`torchvision.transforms.v2.Compose`:

1. **Normalise** to `[0, 1]` per pattern via
    `data.rescale_like_vmax(x, vmax)` (per-pattern vmin = 0).
2. **CenterCrop** to `center_crop_size = 140` px.
3. **CenterOnCOM** with `search_radius = 2 × center_mask_radius`
    (default 30 px). Finds COM of intensity in a circular search region
    and shifts so the direct beam is centered. Removes residual probe
    drift.
4. **Resize** to 192 × 192 (bilinear, antialias).
5. **PolarTransform(192)**: Cartesian → (θ, r) of shape (192, 192).
6. **PolarMaskLeft(45)**: zero the first 45 columns (central beam).

The model sees `(B, 1, 192, 192)` polar tensors.

### 4.2 1-D radial profile (offline, once per cube + per vmax)

`compute_radial_profile.py` produces, for each pattern, a 70-element
"SAXS-treated" radial profile saved as `<base>.radial.npy` (or
`<base>.radial_v<V>.npy` if non-default vmax) plus a
`<base>.gate_thresholds.json`.

Procedure:
1. Apply the same polar pipeline (steps 1-6 above) at vmax of choice.
2. Sum over θ → I(r) of length 192.
3. Keep bins **70..140** only (`Q_LOW`, `Q_HIGH`) to avoid mask-edge
    artifact and high-q noise floor.
4. Normalise per pattern to unit area, take log.
5. Fit a degree-3 polynomial baseline in log-space using a robust
    two-pass fit (peaks excluded by 2·MAD), subtract.
6. Mean-center the residual.
7. Calibrate gate thresholds by sampling 50 000 random pairs:
    `tau_pos = quantile(cos, 0.85)`, `tau_neg = quantile(cos, 0.50)`.

The 1D profiles + thresholds are passed to the model only as a
*regulariser* — they do not enter at inference.

---

## 5. Model architecture

`dino_sr_contrastive_model.ContrastiveDINOModel` defines:

### 5.1 Encoder

`create_encoder_resnet18_variableN(n_layers=1, use_maxpool=True,
circular_theta=True)` returns a `PlainSequentialEncoder`. Layout:

- Input: `(B, 1, 192, 192)`.
- Stem: Conv2d(1→64, k=7, s=2, p=3) + BN + ReLU.
- ResNet-18 layer1 (or up to layer4 if `n_layers > 1`).
- Global pool: **`x.flatten(2).max(dim=2).values`** (replaces
    `F.adaptive_max_pool2d` for *deterministic* backward; mathematically
    identical forward).

θ-circular convolutions: when `circular_theta=True`, every Conv2d in the
encoder is replaced by a custom `CircularConv2d` that pads cyclically
along the θ axis (axis 2, since polar layout is (θ, r)).

Output: `(B, D)` features with `D = 64` for `n_layers=1`.

### 5.2 Projector

2-layer MLP: `Linear(D, projection_hidden=256) + GELU + Linear(256,
projection_dim=128) + L2-normalise`. Output `z ∈ ℝ^128`, ‖z‖ = 1.

### 5.3 Prototypes

`Linear(128, K, bias=False)` whose weight rows are L2-normalised every
forward pass. Logits = `z · Pᵀ`, i.e. cosines.

### 5.4 EMA teacher

A frozen copy of `(encoder, projector, prototypes)` updated every step
with `θ_t ← m θ_t + (1−m) θ_s`, `m` ramping 0.996 → 1.0 over training
(`get_teacher_momentum`).

---

## 6. Loss functions

### 6.1 DINO loss

For two augmented views (student `x_s`, teacher `x_t`):

1. Compute `logits_s = student(x_t)`, `raw_t = teacher(x_s)` (the swap).
2. Center teacher: `tl = raw_t − center` where `center` is an EMA of
    teacher logits across batches.
3. Sharpen: `p_t = softmax(tl / τ_teacher)`,
    `log p_s = log_softmax(logits_s / τ_student)`.
4. Per-sample CE: `L_i = −Σ_k p_t[i,k] · log p_s[i,k]`.
5. Optional confidence weighting (`conf_weight_gamma > 0`):
    `w_i = (max_k p_t[i,k])^γ`,
    `L_DINO = Σ w_i L_i / Σ w_i`. Otherwise simple mean.

Schedules:
- `τ_teacher`: linear warmup from `T0 = 0.04` to `Tfin = 0.07` over
   `warmup_frac = 0.2` of epochs (model-internal default).
- `τ_student = 0.1` (fixed).

### 6.2 cluster1d loss

`dino_sr_contrastive_model.cluster1d_loss(p_student, radials, margin,
min_cluster_mass)`:

```
N_c   = Σ_i p_i,c                                   # (K,)
r̄_c   = Σ_i p_i,c · R[i] / N_c                       # (K, n_bins)

L_intra = − E_i [ Σ_c p_i,c · cos(R[i], r̄_c.detach()) ]
L_inter = mean_{c≠c′}  ReLU( cos(r̄_c, r̄_c′) − margin )
   masked to clusters with N_c ≥ min_cluster_mass
```

Returns `(L_intra, L_inter, mean_off_diag, max_off_diag)` for logging.

The total loss at training time is

```
L = L_DINO + λ_supcon · L_supcon + λ_1d · L_cluster1d + ...
```

with `λ_supcon = 0`, `λ_1d = 0.1`, `margin = 0.4` for the paper.

---

## 7. Training recipe (locked)

The "paper recipe" used everywhere in the latest experiments:

| Knob | Value |
|---|---|
| K | 6 (paper-master) or 8 (per-family, tilt) |
| epochs | 30 |
| seed | 42 |
| batch size | 128 |
| optimiser | AdamW(lr=3e-4, wd=1e-6) |
| LR schedule | cosine (T_max = epochs, eta_min = 1e-6) |
| τ_teacher schedule | warmup_frac=0.2 (model default), T0=0.04, Tfin=0.07 |
| λ_DINO | 1.0 |
| λ_cluster1d (`cluster1d_lambda`) | 0.1 |
| margin (cluster1d) | 0.4 |
| `conf_weight_gamma` (γ) | 0.5 |
| polar mask cols | 45 |
| center_mask_radius | 15 |
| center_crop_size | 140 |
| augmentations enabled | θ-roll (s=192, t=16), Gaussian noise |
| augmentations disabled | hflip, vflip, colorjitter |
| COM-centering | on (`com_centering=True`, `com_search_radius_factor=2.0`) |
| supcon | OFF (`supcon_lambda=0`) — superseded by cluster1d |
| centroid loss | OFF |
| proto repel | OFF |

Determinism enforced inside `train_contrastive`:
```
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```
plus the explicit `Tensor.max(dim=2)` global pool replacement (Section
5.1). Verified bit-exact reproducibility across two same-seed 3-epoch
runs (see `runs/_det_a` vs `runs/_det_b`).

Training entry point: `run_contrastive.run_config(config_key="c", sample,
outdir, **kwargs)`. It builds the model + augmentations, calls
`train_contrastive`, then `evaluate_and_report` (eval-with-figures) when
the run is single-sample. Multi-cube training (`is_multi=True`) skips
the `evaluate_and_report` call and runs eval explicitly via
`infer_scan` in the wrapper (see Section 12).

---

## 8. Inference / transfer

`contrastive_eval.infer_scan(model, dataset, device, ...)` is the
inference path. It applies the same eval-time geometric pipeline
(CenterCrop, COM, Resize, Polar, MaskLeft) and returns:

- `soft_probs` (N, K_active) — softmax of teacher logits at
   `eval_temp = 0.06`.
- `assigns` (N,) — argmax.
- `embeds` (N, 128) — L2-normed teacher embeddings.

`dense_remap=True` reorders prototypes by descending count and remaps
both probs and assigns; `dense_remap=False` preserves the original
prototype IDs. Use **`dense_remap=False`** when comparing across
samples (so a given prototype index is the *same* learned cluster on
every cube).

For per-sample paper outputs after inference, see Section 9.

---

## 9. Image formation & analysis scripts

All scripts live under `dino_sr_contrastive/`. Each writes a
self-describing PNG / NPY / JSON to a deterministic subdirectory. List
below in alphabetical order.

### 9.1 `analyze_bottom_layers.py`

Strain + separation analysis between two prototype class averages
(typically the two layers nearest the substrate of EuInAs). Optional
overrides via `--classA / --classB` and `--out-name` to write to a
custom subfolder. For each pair:

- Confidence-weighted class average (top-300) per class.
- `analyze_strain.strain_between_classes` runs RANSAC affine alignment
   on detected blob centres (skimage `blob_log`), reports rotation,
   ε_major, ε_minor, shear, n_inliers.
- Polar comparison + angular-trace plots at fixed radii.
- SSIM, raw pixel cosine, embedding centroid cosine, intra-class
   cosine per side, 1-D radial-profile cosine.
- All numbers in `metrics_summary.json`.

Outputs in `<run_dir>/eval/<out-name>/`.

### 9.2 `analyze_strain.py`

Standalone RANSAC strain core. Used by `analyze_bottom_layers`.

### 9.3 `baseline_cepstral_nmf.py`

npj 2024 Yoo-style baseline: cepstrum per pattern → central 32×32
patch → NMF → k-means. Optional auto-K via cosine silhouette over
`--k-range`. Outputs class map, NMF components, silhouette curve, and a
`metrics.json` with timings and cluster sizes. See
`runs/_baselines/cepstral_nmf_<sample>_K{auto,N}_v<vmax>/`.

### 9.4 `baseline_nmf_kmeans.py`

Older polar-vector NMF baseline (kept for reference; superseded by
9.3).

### 9.5 `compute_radial_profile.py`

The SAXS-treated 1D radial profile precomputation (Section 4.2).
Standalone CLI; pass `--sample <key>`. The v5 pipeline calls a
parameterised version inside `run_per_family_v5_pipeline.py`.

### 9.6 `gui_line_labeler.py`

Tk GUI for manual labelling of each (sample, prototype) pair as
`lines / nolines / partial(%)`. Reads
`<source>/transfer/<sample>/eval/inference.npz` and
`<source>/transfer/<sample>/eval/class_averages/p{c}.png`, plus 6-10
random examples from `class_examples_200/p{c}/`. Writes
`<source>/line_labels.json` (autosave) plus
`line_labels_per_class.csv` and `line_labels_summary.csv` on export.

### 9.7 `gui_vmax_picker.py`

Tk GUI to inspect raw cube patterns at different vmax values, decide a
vmax for each cube. Lazy-load via mmap.

### 9.8 `midrun_class_map.py`

Standalone tool to render the class map from any saved checkpoint
(used by mid-training watchdogs). Same display style as the eval
class map.

### 9.9 `overlay_classes_on_haadf.py`

Overlays the union of a chosen prototype set (`--classes 0 2 6`) on the
HAADF virtual image, in a single user-chosen colour
(`--color "#2ca02c"`) and alpha. Also calls
`plot_class_averages_nomask.render_aggregate` to produce the matching
unmasked aggregate diffraction average.

Outputs:
- `<run_dir>/eval/virtual/fig_HAADF_overlay_<cls_tag>.png`
- `<run_dir>/eval/class_averages_nomask/aggregate_only_<cls_tag>.png`

### 9.10 `plot_aggregate_avg.py`

Older, masked-version of (9.9 second half). The unmasked version inside
`plot_class_averages_nomask.render_aggregate` is preferred for paper
figures.

### 9.11 `plot_attribution_per_proto.py`

Two-panel (GradCAM | IG) per-prototype figure. For every test sample
under a per-family source model, computes class-average GradCAM and IG
on the polar-domain logits, projects back to Cartesian, smooths
(σ=2). Saves
`<source>/transfer/<sample>/eval/attribution_per_proto/p{c}.png`.

### 9.12 `plot_baseline_class_averages.py`

Computes per-cluster class averages for the cepstral+NMF baseline
(Section 9.3). Top-N membership ranked by Euclidean distance from
cluster mean in NMF-W space. Same paper display recipe as the DINO
class averages (log1p, percentile clip, beam mask r<15px).

### 9.13 `plot_class_averages_nomask.py`

Same as the pipeline's class-average rendering but **without the
central beam mask**. Library:
- `render_one(run_dir, sample, vmax, n_top_avg=300)` — per-prototype
   averages.
- `render_aggregate(run_dir, sample, include=[...] or exclude=[...],
   vmax, n_top=2000)` — confidence-weighted average over a chosen
   prototype subset.

Output dir: `<run_dir>/eval/class_averages_nomask/`.

### 9.14 `plot_line_coverage_combined.py`

Two-row scatter (NaPHI top, MgNaPHI bottom) of per-sample line
coverage from `line_labels_summary.csv`. Track-based label staggering
prevents name overlap.

### 9.15 `plot_manual_line_coverage.py`

Earlier version of (9.14); still used to produce
`fig_manual_line_coverage.png`.

### 9.16 `plot_model_schematic.py`

Generates the methods schematic figure
(`runs/_per_family_v5/fig_model_schematic.{png,pdf}`).

### 9.17 `plot_per_family_transfer_classmaps.py`

Multi-panel class maps for every test sample per family (K=8 fixed
tab10 palette across panels). Output:
`<source>/fig_all_transfer_classmaps.png`.

### 9.18 `plot_transfer_classmaps_lineness.py`

Same multi-panel layout but **recoloured by per-pixel line score**
in `[0, 1]` (0 = no-lines prototype, 1 = lines prototype, partial =
pct/100). Reads `line_labels.json` from the source-model dir. Also
produces a NaPHI + MgNaPHI combined panel
(`runs/_per_family_v5/fig_all_transfer_classmaps_lineness_combined.png`).

### 9.19 `render_per_family_averages.py`

Post-processes a per-family pipeline run: for each trained source
model (Model N or Model M) loads `best.pth`, runs `infer_scan` on the
training data, and renders class averages so the user can pick the
line prototype indices.

### 9.20 `render_naphi_examples.py`

Generic CLI to render N example patterns per class for a per-family
model on its training data. `--family NaPHI|MgNaPHI`,
`N_PER_CLASS = 200` by default.

### 9.21 `viz_paper_outputs.py`

Library used by both single-sample and transfer pipelines.
- `render_class_map(run_dir, sample)` — adaptive K colorbar (no
   wasted slots beyond K_active), saved as
   `eval/fig_class_map_paper.png`.
- `render_class_averages_and_examples(run_dir, sample, n_examples=100)`
   — class averages + N example patterns per class. Output:
   `eval/class_averages/p{c}.png` and
   `eval/class_examples/p{c}/...png`.

### 9.22 `viz_paper_attribution.py`

Per-run paper-quality attribution figure. For each prototype computes
class-average GradCAM (hooks the last residual block of the student
encoder) and Integrated Gradients (50 steps), projects back to
Cartesian, smooths (σ=2), and renders a multi-row figure
(class avg + GradCAM + IG + 3 exemplars + GradCAM-on-exemplar) plus
per-prototype panels.

### 9.23 `virtual_bf_haadf.py`

Virtual BF + HAADF detector images from raw cubes. Defaults
proportional to frame size:
- BF disk: `r ≤ 0.06 · H` (covers central beam halo).
- HAADF annulus: `0.18 · H ≤ r ≤ 0.45 · H`.

Outputs `BF.npy`, `HAADF.npy`, `fig_BF.png`, `fig_HAADF.png`,
`fig_BF_HAADF.png` in `<run_dir>/eval/virtual/`.

### 9.24 `write_paper_line_coverage.py`

Consolidates manual labels into two paper-ready CSVs
(`paper_line_coverage_per_sample.csv` with notes column for duplicate
groups, `paper_line_coverage_summary.csv` with group-level mean ± std
+ outlier gap).

---

## 10. Per-experiment recipes

### 10.1 Paper master sweep — `run_paper_master.py`

13 single-sample trainings at K = 5–8 across all main samples
(Na007a/b, Na006a, IMC×2, EuInAs at multiple K). Each uses the locked
recipe and `evaluate_and_report` for the standard suite of figures.
Output: `runs/_paper_master/<sample>_K<K>/`.

### 10.2 v5 + per-family pipeline — `run_per_family_v5_pipeline.py`

For both NaPHI (training cubes SI-003 + SI-004) and MgNaPHI (SI-004 +
SI-011):

1. Compute per-cube radials at vmax = 5 (saved as
    `<base>.radial_v5.npy`).
2. Concatenate them into a multi-cube radials NPY +
    `gate_thresholds.json`.
3. Register a `PERFAM_<family>_V5` multi-cube entry in `SAMPLES`.
4. Train one model with K = 8, 30 ep, deterministic, locked recipe.
5. Run `infer_scan` on the training data, write
    `inference.npz` + class avgs + 200 examples per class +
    `fig_class_map_paper.png`.
6. For each held-out test sample under that family, run `infer_scan`,
    write `inference.npz` + class map + class avgs + 200 examples per
    class to
    `<source>/transfer/<sample>/eval/`.

Output root: `runs/_per_family_v5/{NaPHI,MgNaPHI}_combined_K8_30ep/`.

### 10.3 Per-family attribution — `plot_attribution_per_proto.py`

After 10.2 finishes, run twice (once per family) with
`--root runs/_per_family_v5 --model-dir <family>_combined_K8_30ep
--family <NaPHI|MgNaPHI>` to build the GradCAM | IG figures.

### 10.4 Manual labelling

1. Open `gui_line_labeler.py`.
2. Browse to `runs/_per_family_v5/<family>_combined_K8_30ep`.
3. Label each (sample, prototype) pair as Lines / NoLines / Partial%.
4. Done & Export CSV.
5. Run `write_paper_line_coverage.py` to consolidate.

### 10.5 Coverage plots

1. `plot_per_family_transfer_classmaps.py` — multi-panel class maps
    (one panel per test sample, K=8 tab10 palette).
2. `plot_transfer_classmaps_lineness.py` — same panels recoloured by
    line score (0..1), uses `line_labels.json`.
3. `plot_manual_line_coverage.py` — two-row scatter of all samples.

### 10.6 MgPhi tilt — `run_mgphi_tilt_pipeline.py`

Per-sample (NOT per-family) training for the tilt-series cubes
NBED-001a / b / c. K = 8, vmax = 5, 30 ep. Each sample gets its own
`<run_dir>/eval/` with class averages + 200 examples + paper
attribution. Class maps added by `viz_paper_outputs.render_class_map`.
Virtual BF / HAADF images produced afterwards by `virtual_bf_haadf.py`.

### 10.7 Transfer demo — `run_transfer_na007b_to_na007a.py`

Inference-only application of the Na007b_K6 model to Na007a. Writes
`runs/_paper_master/Na007b_K6/transfer/Na007a/eval/...`, including
class map, class averages, 200 examples per class.

### 10.8 EuInAs strain — see `analyze_bottom_layers.py`

Pair-wise strain analysis between film prototypes. Used in the EuInAs
methods section for the rotation/strain story (p0 vs p3 = 1.07°
rotation + ~0.3% strain; p1 vs p4 = pure intensity scaling).

### 10.9 Cepstral baseline

`baseline_cepstral_nmf.py --sample <key> --vmax <V> --k-range 2 12`
auto-K, then `plot_baseline_class_averages.py --run-dir <out>` for the
class averages.

---

## 11. Key numerical results & where they live

| Result | Path |
|---|---|
| Master sweep eval JSONs | `runs/_paper_master/<sample>_K<K>/eval/metrics.json` |
| Per-family models | `runs/_per_family_v5/<family>_combined_K8_30ep/best.pth` |
| Per-family transfer inferences | `runs/_per_family_v5/<family>_combined_K8_30ep/transfer/<sample>/eval/inference.npz` |
| Manual labels (raw) | `runs/_per_family_v5/<family>_combined_K8_30ep/line_labels.json` |
| Paper CSVs (per-sample + summary) | `runs/_per_family_v5/paper_line_coverage_{per_sample,summary}.csv` |
| Coverage scatter (paper Fig 3) | `runs/_per_family_v5/fig_manual_line_coverage.{png,pdf}` |
| Lineness class-map grids | `runs/_per_family_v5/fig_all_transfer_classmaps_lineness_combined.png` |
| Model schematic | `runs/_per_family_v5/fig_model_schematic.{png,pdf}` |
| MgPhi tilt outputs | `runs/_mgphi_tilt/NBED001{a,b,c}_K8_30ep_v5/eval/` |
| Cepstral baseline | `runs/_baselines/cepstral_nmf_<sample>_<K>_v<vmax>/` |

Headline numbers (manuscript main text):
- NaPHI line coverage: 0.80 ± 0.12 (n = 8, range 0.57 – 1.00).
- MgNaPHI bulk line coverage: 0.48 ± 0.41 (n = 7, range 0.00 – 0.90).
- MgNaPHI SI-007 trio: 1.00 ± 0.00 (n = 3 remeasures of same area).
- Outlier gap to next-highest bulk: 0.10 (vs SI-003 at 0.896).
- Method noise floor (NaPHI 4-quarter set): 0.79 ± 0.07.
- EDS Na:Mg ratios — SI-007 ≈ 1.2 : 1, SI-010 ≈ 1 : 1.
- Train cost (Na007b K = 6, deterministic, 30 ep): 929 s ≈ 15.5 min.
- Eval cost (Na007a transfer): 39.8 s for 12 600 patterns ≈
   3.16 ms / pattern.

---

## 12. Reproducibility checklist

To reproduce any single result:

1. **Restore env**: `conda env create -f environment.yml` (or pip
    install the packages in Section 2). Verify with
    `python -c "import torch; print(torch.cuda.is_available())"`.
2. **Restore data**: place the `.prz` cubes at the paths in `data.py`.
    For the multi-cube samples ensure all components exist.
3. **Recompute radials** (one-shot): run `compute_radial_profile.py
    --sample <key>`. The radials are deterministic — same input,
    same output.
4. **Train** by invoking `run_paper_master.py` (single-sample) or
    `run_per_family_v5_pipeline.py` (per-family). Both are
    deterministic with `seed = 42`. Expect bit-exact reproducibility on
    the same GPU.
5. **Evaluate / transfer** as described in Section 10. All steps after
    training are deterministic and CPU-bound when not using the GPU.
6. **Plot** with the relevant `plot_*.py` script. All plots are
    pixel-stable for a given input.

Hyperparameter changes: the locked recipe in Section 7 is what produced
every result in the manuscript. Changing K, λ_1d, γ, margin, augmentation
flags, vmax, or the polar pipeline parameters will produce different
outputs and is not part of the reproducibility claim.

---

## 13. Code map (one-line summary per file)

```
data.py                                  — cube loaders + SAMPLES registry
dino_sr_contrastive_model.py             — model, losses, train_contrastive
contrastive_eval.py                      — infer_scan + evaluate_and_report
run_contrastive.py                       — single-sample run_config + eval
compute_radial_profile.py                — 1-D radial precompute + gate calibration

# experiment drivers
run_paper_master.py                      — 13-config paper-master sweep
run_per_family_v5_pipeline.py            — Model N + Model M training + transfer
run_mgphi_tilt_pipeline.py               — NBED-001a/b/c per-sample
run_transfer_na007b_to_na007a.py         — single-shot Na007b -> Na007a transfer
run_na007a_k5.py                         — Na007a re-run at K=5

# baselines
baseline_cepstral_nmf.py                 — npj 2024 (cepstral+NMF+kmeans)
baseline_nmf_kmeans.py                   — older polar-vector NMF (reference)

# image-formation analysis
viz_paper_outputs.py                     — class map (paper) + class avgs + examples
viz_paper_attribution.py                 — per-run GradCAM/IG paper figure
viz_gradcam.py                           — low-level GradCAM + IG
plot_class_averages_nomask.py            — class avgs without beam mask + aggregate
plot_aggregate_avg.py                    — older masked aggregate (kept)
plot_baseline_class_averages.py          — baseline class avgs
plot_attribution_per_proto.py            — 2-panel (GradCAM|IG) per (sample, proto)
plot_per_family_transfer_classmaps.py    — multi-panel class maps (tab10)
plot_transfer_classmaps_lineness.py      — same, recoloured by per-pixel line score
plot_line_coverage_combined.py           — two-row coverage scatter
plot_manual_line_coverage.py             — earlier coverage plot (paper Fig 3)
plot_model_schematic.py                  — methods schematic
virtual_bf_haadf.py                      — virtual BF + HAADF images
overlay_classes_on_haadf.py              — class overlay on HAADF + aggregate avg
analyze_strain.py                        — RANSAC affine strain
analyze_bottom_layers.py                 — pair strain + separation metrics

# manual labelling
gui_line_labeler.py                      — Tk GUI for line/nolines/partial%
write_paper_line_coverage.py             — consolidate labels into paper CSVs
gui_vmax_picker.py                       — Tk GUI for picking vmax

# helpers + post-processes
midrun_class_map.py                      — render class map from any ckpt
render_per_family_averages.py            — render Model N/M training-data class avgs
render_naphi_examples.py                 — render 200 examples per class

# watchdogs (poll-and-render during training)
watchdog_midrun_paper_master.py
watchdog_midrun_c1d_sweep.py
watchdog_midrun_lg_sweep.py
watchdog_midrun_winner_followup.py
watchdog_midrun_determinism.py
```

---

## 14. Conventions

- All GPU work is on a single CUDA device (`torch.device("cuda")`).
- Outputs in `runs/<experiment>/<run_label>/` with sub-dirs:
   `eval/`, `eval/class_averages/`, `eval/class_examples_200/`,
   `eval/paper_attribution/`, `eval/virtual/`, `transfer/<sample>/eval/`.
- `dense_remap=False` in any cross-sample comparison; `dense_remap=True`
   only for single-sample reports (where prototype IDs do not need to
   align across samples).
- vmax is sample-specific. Newer experiments use `vmax = 5` for Na/Mg
   cubes; older ones (paper master) used the registry default
   (typically 2 for Na/NaPHI/MgNaPHI, 30 for EuInAs, 3-5 for IMC).
- Prototype index "p{c}" in figure titles refers to the *original*
   training-time index, not a dense-remap.

---

End of document. If something is missing, the most likely place to look
is the docstring at the top of the relevant `*.py` file — every script
in the repo has one.
