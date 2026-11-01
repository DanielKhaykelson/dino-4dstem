# Hand-off prompt — DINO-SR + Contrastive Head

You are picking up a new task in a fresh session. Read this file end-to-end before touching anything.

---

## What to do

Implement and evaluate the spec in `SPEC.md` (also in `dino sr contrastive prompt.pdf`).

**Summary:** add a soft-label contrastive head to the existing DINO-SR L1 pipeline, replace Cartesian rotation aug with exact polar θ-roll, run 3 ablations, write 3 deep analytical reports.

## Working directory

You work inside:

```
D:\DINOSR\Claude\PaperRun_claude\dino_sr_contrastive\
```

All files you need are here. **Edit only copies in this folder.** Do not touch anything in `..\`.

## Files already in this folder

See `README.md` for the full list. Key ones:

- `SPEC.md` — the authoritative spec.
- `dino_sr_ablation.py` — current trainer. Contains `AblationDINOModelSR`, `train_ablation()`, `PolarTransform`, `PolarMaskLeft`, `CenterMask` (imported from `dino_sr_fixed`), `get_ablation_transforms()`. Fork this.
- `dino_sr_fixed.py` — shared components (`Prototypes`, `ProjectionHead`, aug transforms).
- `eval_all.py` — `LoadPRZ` + sample registry.
- `elliptical_correction.py`, `scorecard.py` — present but **not required** by the spec. Ignore unless helpful.
- `run_with_scorecard.py` — old CLI entry. Fork it into a new `run_contrastive.py` rather than editing.

## Environment

- **Python**: `C:\Users\danielkh\AppData\Local\anaconda3\envs\py4DSTEM_SAM\python.exe` (Windows + git-bash shell). Use this path in all `Bash` tool calls.
- **GPU**: RTX 4080, 16 GB. CUDA available.
- **Torch**: 2.7.1+cu118, numpy 1.26.4, torchvision v2 API in use.
- **`umap-learn`**: may or may not be installed. Install if missing. User said yes to UMAP.

## Data

- **Target sample for the 3 ablations**: **Na007b** (user explicitly said Na007b only — do NOT run the 3×50-epoch protocol on other samples).
- **Path**: `D:\DINOSR\data\Na007b.prz` (local copy, ~13 GB, fast read). Already in `eval_all.SAMPLES["Na007b"]`.
- **Shape**: scan (126, 100), raw patterns (512, 512), vmax=2, center_mask_radius=15, non_layered.
- **Dataset size**: 12600 patterns.

### All available local sample paths (for reference / sanity checks only)

All 6 samples are already copied to local disk under `D:\DINOSR\data\` (fast, no network I/O) and registered in `eval_all.SAMPLES`. The sample-type metadata (layered vs non-layered, layer bounds for the layered EuInAs sample) is in `scorecard.SAMPLE_TYPES` if you need it.

| key | path | scan_shape | vmax | mask_r | type | approved_label |
|---|---|---|---|---|---|---|
| `Na007a`       | `D:\DINOSR\data\Na007a.prz`       | (126, 100) | 2  | 15 | non_layered | *(none yet)*      |
| `Na007b`       | `D:\DINOSR\data\Na007b.prz`       | (126, 100) | 2  | 15 | non_layered | `heat_040_050`    |
| `Na006a`       | `D:\DINOSR\data\Na006a.prz`       | (100, 100) | 4  | 15 | non_layered | `heat_040_070`    |
| `EuInAs_B100`  | `D:\DINOSR\data\EuInAs_B100.prz`  | (66, 396)  | 30 | 10 | **layered** (layer_bounds=[22, 44]) | `heat_040_070` |
| `IMC_50nm_SI2` | `D:\DINOSR\data\IMC_50nm_SI2.prz` | (128, 128) | 3  | 15 | non_layered | `const_065`       |
| `IMC_150nm_SI5`| `D:\DINOSR\data\IMC_150nm_SI5.prz`| (128, 128) | 5  | 15 | non_layered | `const_040`       |

**When to use other samples**:
- Spot-checking that the θ-roll-invariance unit test (spec item 8) holds on a structurally different sample (e.g. EuInAs which is layered with anisotropic texture) — 1 forward pass, not training.
- Sanity-verifying that the "`contrastive_lambda=0 + theta_roll_aug=False` reproduces pure DINO" test holds on a sample beyond Na007b.

**When NOT to use other samples**: don't expand the 3×50-epoch ablation to all 6. That's a 9-hour run the user did not ask for.

**Approved-label checkpoints for cross-reference**: each sample's `approved_label` corresponds to a pre-existing trained run in `..\DINO_PAPER_V2\{sample}\L1\{approved_label}\best_auto.pth` — the pure-DINO L2 baseline the user has historically trusted. Useful ONLY for the "comparison to validated L2 baseline" section of the report (spec 14, interpretive conclusions). Do not reuse these checkpoints for training.

## Q&A — clarifications from the user (answers baked in)

The previous session asked these before handing off. The user's answers:

1. **Q**: Which training notebook to copy?
   **A**: There's no canonical notebook in active use — the current path is CLI (`run_with_scorecard.py`). The `dino sr contrastive prompt.pdf` mentions Jupytext `# %%` cells, but the user's working style is CLI-first. **Produce a Jupytext-compatible `.py` file** (with `# %%` cell markers) as `run_contrastive.py` in this subfolder. It runs as a plain script AND can be opened as a notebook via VS Code / Jupytext. Best of both worlds.

2. **Q**: Which samples for the 3 ablation configs?
   **A**: **Na007b only**. All 3 configs, 50 epochs each. Total ≈ 60 min of GPU time for training + setup.

3. **Q**: "Center mask applied as zeroed low-r rows" — rows or cols?
   **A**: The existing `PolarTransform` produces `(θ=rows, r=cols)`. "Low-r" means **low-r columns** (`x[:, :, :, :k_cols] = 0`). The existing `PolarMaskLeft(k_cols=...)` is already correct — use it. `k_cols = 30` matches `center_mask_radius = 15` px (polar output 192 cols spanning r=0..96 px, so 15/96 × 192 ≈ 30).

4. **Q**: L1 backbone output dim?
   **A**: 64 (`LAYER_OUT_DIMS[1] = 64` in existing code). So projection MLP = `64 → 256 → 128`.

5. **Q**: UMAP?
   **A**: Yes. Install `umap-learn` if missing. If install fails (offline env), fall back to sklearn PCA + annotate that in the report; but prefer UMAP.

6. **Q**: θ-roll unit test tolerance?
   **A**: Your judgment. My recommendation: after forwarding `x` and `torch.roll(x, k, dim=-2)` through the same encoder, the feature maps should match under the same roll-k shift. Test: **max absolute diff < 1e-4 on the CENTRAL r-region** (skip first/last 3 columns on each r-edge where zero-padding produces legitimate drift). Interior should be essentially bit-exact (≤ 1e-6). If you can't hit 1e-4 on the interior, something is wrong with the circular padding.

## Critical design points — do not miss

- **Circular padding along θ (rows), zero/reflect along r (cols).** `nn.Conv2d` takes a single `padding_mode`, so you need a wrapper. Cleanest: a `ThetaCircularConv2d` module that does `F.pad(x, (px, px, 0, 0), 'constant')` for r and `F.pad(..., (0, 0, py, py), 'circular')` for θ, then calls a `padding=0` Conv2d. Recursively swap all `nn.Conv2d` in the ResNet with this wrapper at model construction.

- **Student/teacher asymmetry assertion (spec item 9)** — on the first forward pass, log and assert that the student's and teacher's θ-roll amounts differ for every sample in the batch. Equal values = RNG sharing bug. This bug has burned users before; the assertion is cheap insurance.

- **Lambda warmup logic**: ep < 20 → λ = 0; 20 ≤ ep < 30 → linear ramp 0 → target; ep ≥ 30 → target. `L_contrastive` is still *computed* every step (for logging), it just isn't added to the loss during warmup.

- **`contrastive_lambda=0 + theta_roll_aug=False` must bit-exactly reproduce current pure-DINO L1**. This is the validation that the contrastive branch is purely additive. Test this before running all three ablations: save best.pth from config (a), compare to a pure-DINO L1 reference run (same seed) — weights should be identical.

## Recommended execution order

1. **Skeleton + smoke test (30 min)**
   - Copy `dino_sr_ablation.py` → `dino_sr_contrastive_model.py`, strip what's not needed, add:
     - `ProjectionMLP` (64 → 256 → 128, BN+GELU, L2-norm)
     - `PolarThetaRoll` aug module (`torch.roll(x, k, dim=-2)`, k drawn per view)
     - `ThetaCircularConv2d` wrapper + recursive replacement of Conv2d in the L1 backbone
     - Soft-label contrastive loss (teacher-softmax-cos vs student-embed-cos, MSE, diagonal masked)
     - Config plumbing (items 12 in spec)
     - Logging (item 11)
   - Write `run_contrastive.py` (Jupytext `# %%` cells)
   - **Unit test (spec item 8)**: feed a sample + its `torch.roll(x, k, dim=-2)` through the encoder, check `F.conv2d` outputs match under same roll. Max abs diff < 1e-4 on interior r-cols.
   - **Smoke run**: 10 epochs, config (c), verify asymmetry assertion passes, λ-warmup logged (=0 for ep 1–10), losses sane, no NaN, no prototype collapse.

2. **Validate contrastive branch is additive**
   - Run config (a) — `contrastive_lambda=0, theta_roll_aug=False` — for 5 epochs with the new code.
   - Check the `best.pth` weights differ negligibly from a pure-DINO L1 reference run at same seed. (Or verify by running the same code path with a `no-contrastive-head` flag and comparing.)

3. **Three ablations × 50 epochs**
   - Config (a): `lambda=0, theta_roll=False`
   - Config (b): `lambda=0, theta_roll=True`
   - Config (c): `lambda=0.2, theta_roll=True`
   - Each ≈ 20 min. Save to `dino_sr_contrastive/runs/Na007b/config_{a,b,c}/`.

4. **Evaluation + reports**
   - Build eval helpers for each required metric (NMI, KNN purity, intra/inter-class cosine, radial-profile consistency, prototype usage entropy). KNN purity = fraction of k nearest embedding neighbors with the same prototype label; pick k=10.
   - Per-config figures: loss curves, prototype-usage-entropy curves, teacher-softmax-entropy distribution curves, prototype centroid cosine matrix, UMAP (color by prototype), per-prototype representative patterns (4–8 closest), boundary-sample panels, θ-roll-invariance sanity scatter.
   - Write 3 markdown reports into `runs/Na007b/config_{a,b,c}/report.md`. Also write a top-level `runs/Na007b/COMPARE.md` that synthesizes (a) vs (b) vs (c) interpretive conclusions.
   - All PNGs saved with `bbox_inches='tight'`.

## Report structure (template — use this for all 3 configs)

```
# config_{x} — Na007b — DINO-SR + Contrastive

## Summary (1 paragraph)
What this config is (flags) and the one-line verdict.

## Quantitative metrics
| metric | value |
|---|---|
| NMI | ... |
| KNN purity (k=10) | ... |
| intra-class cosine | ... |
| inter-class cosine | ... |
| intra/inter ratio | ... |
| radial profile consistency | ... |
| prototype usage entropy | ... |
| effective K | ... |

## Per-prototype breakdown
- Usage histogram (N samples per prototype, sorted)
- Dead/near-dead prototypes (listed)
- Centroid cosine matrix (figure)
- Most-similar prototype pair (a, b, cos)

## Training dynamics
- Loss curves: L_DINO, L_contrastive, L_total (figure)
- Prototype usage entropy over epochs (figure)
- Teacher softmax entropy distribution over epochs (figure)

## Embedding geometry
- Projection embedding norm distribution (figure, sanity check that L2-norm holds)
- Pairwise similarity histogram on held-out batch (figure)

## Qualitative
- UMAP of student embeddings, colored by prototype (figure)
- Representative patterns per prototype (figure, 4–8 closest)
- Highest-teacher-entropy samples (figure, top 16)

## Rotation invariance sanity (configs b, c only)
- Cosine similarity between student embedding of x vs x with θ-roll k, across held-out batch
- Expected: high for (b, c); random for (a)

## Interpretive conclusions
(Prose. Physical, specific to diffraction / phases / prototypes.)
- What this config learned
- Where it wins
- Where it loses
- Failure-mode check (collapse? norm drift? contrastive decoupled from DINO?)

## Concrete recommendations for next runs
...
```

And a cross-config `COMPARE.md` at `runs/Na007b/`:
```
## (a) vs (b) — did θ-roll alone help?
- Quantify deltas on each metric
- Is the improvement consistent with learned rotation invariance (sanity test)?

## (b) vs (c) — did contrastive head add value on top of θ-roll?
- Quantify. If not, hypothesize why (capacity, λ tuning, warmup length, entropy gating needed).

## L1 + contrastive vs validated L2 pure-DINO
- If an L2 run is available, compare.
- If not, note the gap and recommend running L2 as follow-up.

## Failure-mode scan across configs
- Prototype collapse: any config K_effective < 3?
- Embedding norm drift: any config drifting away from 1.0?
- Contrastive decoupling: L_contrastive plateaued while L_DINO improved?

## Final recommendation
- Which config to use going forward? Why?
- Next hyperparameter sweep (specific values).
```

## What NOT to do

- Don't touch anything in the parent folder.
- Don't add loss terms or scorecard components that aren't in the spec. The spec is precise; stick to it.
- Don't use multi-crop, MAE, contrastive-negatives-from-queue, or other SSL gadgetry not in the spec.
- Don't rotate the Cartesian image after removing the rotation aug — θ-roll is the only rotation mechanism now.
- Don't silently skip the unit test (spec item 8). If circular conv isn't truly circular, the whole point of θ-roll-as-rotation is nullified.
- Don't write a report that restates numbers. The user was emphatic: the value is in the physical interpretation.

## Style

- Markdown tables for metrics, prose for interpretation.
- Refer to "phases", "prototypes", "Bragg spots", "radial profile", "grain orientation" — not abstract "clusters" or "classes" where a physical name applies.
- PNG figures only, `bbox_inches='tight'`, numbered and captioned.
- tqdm-only progress. No `print("epoch ...")` spam.

## If something is ambiguous

Default to the most conservative, spec-literal interpretation. If blocked, ask once clearly, then proceed with your best guess.

## Deliverable summary

```
dino_sr_contrastive/
├── SPEC.md                     (already here)
├── README.md                   (already here)
├── NEW_SESSION_PROMPT.md       (this file)
├── run_contrastive.py          (NEW — your training entry point, Jupytext # %% cells)
├── dino_sr_contrastive_model.py (NEW — fork of dino_sr_ablation.py with contrastive head)
├── contrastive_eval.py         (NEW — NMI / KNN / cosine / UMAP helpers)
├── [any other helper files you create]
└── runs/
    └── Na007b/
        ├── config_a/
        │   ├── best.pth, latest.pth, training_log.csv
        │   ├── eval/   (figures, UMAP, representative patterns)
        │   └── report.md
        ├── config_b/   (same)
        ├── config_c/   (same)
        └── COMPARE.md
```
