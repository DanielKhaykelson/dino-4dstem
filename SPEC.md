# DINO-SR + Contrastive Head — Spec (verbatim from PDF)

## Context

I have a DINO-SR pipeline for 4D-STEM diffraction clustering (self-supervised, prototype-based, ResNet backbone). Pipeline order per sample: Cartesian augment → polar transform → center mask (zero low-r rows) → network. Single global view per sample (no multi-crop — local crops of diffraction patterns are not semantically equivalent to globals). Student and teacher each see one view, asymmetry coming entirely from independent augmentation draws.

I want to add a soft-label contrastive auxiliary head and retrain from scratch.

## File management

- Create a new subfolder (e.g. `dino_sr_contrastive/`) next to the existing files.
- Copy only the files needed for training (training notebook, model definitions, loss, data loading, augmentations) into this subfolder.
- Edit ONLY the copies in the new subfolder. Do not touch originals.
- Keep the `# %%` Jupytext cell structure, figure saves with `bbox_inches='tight'`, tqdm-only progress.

## Requirements

### 1. Backbone
ResNet **L1** depth. Keep other validated hyperparameters: RESIZE=192, CENTER_CROP=140, `center_mask_radius=15` (applied in polar as zeroed low-r rows, deterministic, shared across views), temperature in [0.04, 0.075], CBAM off.

### 2. Projection head
Add in parallel to the existing prototype head:
- 2-layer MLP: `backbone_out_dim → 256 → 128`
- BN + GELU between layers
- L2-normalize the output
- Shared backbone; only the heads differ

### 3. Soft-label contrastive loss
- Target similarity matrix: cosine similarity between teacher prototype softmax distributions (detached, no grad).
- Predicted similarity matrix: cosine similarity between student projection embeddings.
- Loss: MSE between predicted and target matrices, diagonal masked out.
- Teacher softmax for targets, student embeddings for predictions.

### 4. Total loss
`L = L_DINO + lambda * L_contrastive`
- lambda warmup: 0 for `warmup_epochs` (default 20), then linear ramp to target over `ramp_epochs` (default 10).
- Rationale: early prototype softmax is near-uniform, targets are noise until DINO has learned something.

### 5. Optional entropy gating (config flag, default off)
- Per batch, compute teacher softmax entropy per sample.
- Include only samples below the batch-median entropy in the contrastive loss.
- Per-batch quantile, no fixed threshold.

### 6. Polar theta-roll augmentation
- Random integer shift along theta axis, circular wrap.
- Applied AFTER polar transform and AFTER center mask, before network forward.
- Drawn independently per view (student and teacher get different shifts).
- Config: `theta_shift_range` (int, default = `theta_dim` for full range). Range is `[-theta_shift_range//2, +theta_shift_range//2]`. Reduce if L1 underfits.

### 7. Remove Cartesian rotation augmentation
REMOVE Cartesian rotation from the existing pipeline. Rotation is now handled exclusively by theta-roll in polar space — exact, interpolation-free, no beamstop coupling, no center-offset artifacts. Retain all other Cartesian augmentations (translations, intensity scaling, noise, etc.).

### 8. Conv padding
Use circular padding on the theta axis, zero or reflect on r axis. Add a unit test: forward a sample and a theta-rolled copy through the network; feature maps at the corresponding theta offset should match within numerical tolerance (modulo edge effects on r).

### 9. CRITICAL — student/teacher augmentation asymmetry
- Every stochastic augmentation (Cartesian stage + theta-roll) drawn independently per view. Do not share RNG state or parameters.
- Assertion during first forward pass: theta-roll amounts (and other stochastic aug params) for student and teacher views of the same sample must differ. Log a warning if they match — vanishingly unlikely by chance, indicates a bug.
- Center mask stays deterministic and shared (it is preprocessing, not augmentation).

### 10. Seeding
If using a global seed, ensure augmentation RNG advances between student and teacher draws, or derive distinct per-view seeds. Avoid the common bug of seeding once per sample and getting identical augmentations for both views.

### 11. Logging (tqdm postfix + epoch log file)
- L_DINO, L_contrastive, total
- Prototype usage entropy (catch silent prototype collapse)
- Teacher softmax entropy distribution: mean, p10, p90
- Projection embedding norm stats (sanity check L2-norm holds)
- Fraction of batch used in contrastive loss if entropy gate is on

### 12. Config additions (per-sample dict)
- `contrastive_lambda` (float, default 0.2)
- `contrastive_warmup_epochs` (int, default 20)
- `contrastive_ramp_epochs` (int, default 10)
- `contrastive_entropy_gate` (bool, default False)
- `projection_dim` (int, default 128)
- `projection_hidden` (int, default 256)
- `theta_roll_aug` (bool, default True)
- `theta_shift_range` (int, default = `theta_dim`)

### 13. Preserve existing behavior
Do NOT modify the existing DINO loss, prototype head, or EMA teacher update. The contrastive branch is purely additive. Setting `contrastive_lambda=0` must exactly reproduce a pure-DINO run.

## Ablation protocol (comment block at top of notebook)

| Config | `contrastive_lambda` | `theta_roll_aug` | Purpose |
|--------|----------------------|------------------|---------|
| (a)    | 0                    | False            | Pure DINO L1 baseline |
| (b)    | 0                    | True             | Isolates theta-roll contribution |
| (c)    | 0.2                  | True             | Full joint model |

### 14. Analysis and conclusions

After training completes for each ablation config (a, b, c), produce a **deep analytical report** — not just metric tables. The report must go beyond "numbers went up/down" and interpret what the results mean for the method.

**Required analysis components:**

**Quantitative:**
- Full post-eval metrics per config: NMI, KNN purity, intra-class cosine, inter-class cosine, intra/inter ratio, radial profile consistency, prototype usage entropy.
- Per-prototype breakdown: which prototypes are well-used, which are dead or near-dead, which pairs of prototypes are most similar (centroid cosine matrix).
- Training dynamics: loss curves (DINO, contrastive, total) overlaid across the three configs. Prototype usage entropy over epochs. Teacher softmax entropy distribution over epochs.
- Embedding geometry: projection embedding norm distribution, pairwise similarity histogram on a held-out batch. Compare across configs.

**Qualitative:**
- UMAP of student embeddings per config, colored by prototype assignment. Visually inspect cluster separation, connectivity, presence of sub-structure. (UMAP only — do not use t-SNE.)
- Representative samples per prototype: show 4–8 patterns closest to each prototype centroid. Do they look physically coherent?
- Boundary/edge samples: patterns with highest teacher softmax entropy. Are they genuinely ambiguous (phase boundaries, weak signal) or pipeline artifacts?
- Rotation invariance sanity check: forward a sample and its theta-rolled copy, report cosine similarity distribution across a held-out set. Should be high for configs (b) and (c); baseline comparison for (a).

**Interpretive conclusions (the core deliverable):**
- Did the theta-roll augmentation alone (b vs a) improve clustering? By how much on which metrics? Is the improvement consistent with learned rotation invariance (check the sanity test) or does it come from a different mechanism?
- Did the contrastive head (c vs b) add value on top of theta-roll? Quantify. If not, hypothesize why — overclustering absorbed by soft labels, capacity limits of L1, warmup/ramp schedule mistuned, entropy gating needed?
- Are there signs of silent failure modes: prototype collapse, embedding norm drift, contrastive loss decoupling from DINO loss, overconfident teacher softmax?
- Where does each config win and lose? E.g. config (c) may have cleaner centroids but more dead prototypes; (b) may have better edge-sample handling. Trade-offs, not just rankings.
- Comparison to the validated L2 baseline from prior work: does L1 + contrastive recover or exceed L2 pure-DINO performance, or is the capacity gap too large?
- Concrete recommendations for next runs: specific hyperparameters to sweep, architectural changes to try, ablations that would sharpen the conclusions.

**Format:**
- Generate a markdown report saved alongside the training outputs.
- All supporting figures saved as PNG with `bbox_inches='tight'`, referenced by filename in the report.
- Tables for metrics, prose for interpretation.
- Be concrete and physical — refer to diffraction patterns, phases, and prototypes specifically, not abstract "clusters." This work is for a paper, so conclusions should be defensible and specific.
- **Do not produce a report that just restates the numbers.** The value is in the interpretation — what the results imply about the method, what's working, what's not, and why.

## Deliverable

Modified training notebook and supporting files in the new subfolder, runnable end-to-end for all three ablation configs by flipping config flags. Markdown analytical report produced per config, with figures.
