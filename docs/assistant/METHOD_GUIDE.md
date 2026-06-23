# DINO-4DSTEM — Method & Workflow Guide (for the assistant)

This is the assistant's domain reference: when to use each method in the GUI, how
to set parameters for 4D-STEM diffraction, how to choose clustering, and how to
judge whether a result is trustworthy. Written for nanobeam electron diffraction
(NBED) / 4D-STEM scanning datasets.

---

## 1. The pipeline at a glance

A 4D-STEM scan is a diffraction pattern (DP) at every probe position
`(Ny × Nx × H × W)`. The goal is an unsupervised **class map** over the scan +
**class-average DPs** that a microscopist can interpret as phases / crystallinity
regimes / orientations.

Canonical order in the GUI:

1. **Load data** (Pre-processing tab) → set field of view, beam mask, centering.
2. **Train** the DINO model (Training tab) → learns a label-free embedding.
3. **Eval** → class map + class-average patterns from a checkpoint.
4. **Score** the run → quantitative quality metrics.
5. **Interpretation** → explain *why* classes split (Grad-CAM, ablations, radial).
6. **Compare** to classical baselines: **NMF + clustering** and **ACOM**.

You rarely need every step. Pick by the question being asked (Section 6).

---

## 2. Pre-processing (the single biggest lever)

Diffraction is dominated by the intense central (000) beam. Good preprocessing is
what makes any downstream method work.

- **center_mask_radius (beam mask, px)** — zeroes the central disk so clustering is
  driven by scattered intensity, not by the beam. Set just larger than the central
  disk + its tails. Too small → everything clusters by beam intensity/thickness;
  too large → you delete low-angle structure (diffuse scattering, first ring).
- **center_crop_size (FOV, px)** — crops the detector to the informative region.
  Smaller = faster + focuses on low/medium angles; too small clips real Bragg
  reflections.
- **com_centering** — re-centers each DP on its center of mass before masking.
  Essential when the beam wanders across the scan (descan error, sample tilt).
  Almost always **on** for NBED.
- **polar transform (polar_size / polar_mask_cols)** — remaps each DP to
  (radius, angle). This is powerful for diffraction because: (a) rotation of a
  pattern becomes a *shift* in the angular axis (so the θ-roll augmentation makes
  the model rotation-robust), and (b) radial (crystallinity / d-spacing) and
  angular (texture / spottiness) information separate cleanly.
  `polar_mask_cols` zeroes the inner radial columns (a beam mask in polar space).
- **vmax** — display/contrast only for viewing; it is **not** used by training
  normalization. Don't expect vmax to change model results.
- **binning (real-space n×n)** — offered at load for very large cubes. Bins scan
  positions (not the detector) to fit memory. Reduces spatial resolution of the map.

Rule of thumb: tune mask + crop + COM on the Pre-processing tab while watching the
DP-max and a few frames, then push them to training with **"Load parameters to
model"**. Note: the Training "use defaults" toggle governs **model** params only —
data/preprocessing params always follow the loader.

---

## 3. DINO self-supervised classifier (the primary method)

DINO trains a student/teacher network so that two augmented views of the same DP
get the same soft assignment over `K` prototypes — **no labels**. The result is an
embedding + a `K`-way soft assignment per probe position.

Use it when: you want data-driven classes that capture subtle, non-obvious
structure (mixtures of texture + crystallinity), beyond what a single hand-chosen
descriptor (a ring intensity, a spot count) captures.

Key knobs:
- **K (num prototypes / classes)** — upper bound on classes; the model collapses
  unused ones, so set K generously (the validated recipe uses a large K and reads
  off the *active* classes). Start K≈6 for a focused map, larger (e.g. 60) when you
  want fine structure and will merge afterward.
- **epochs** — ~50 is the validated working point for these datasets.
- **center_momentum / EMA (teacher)** — stabilize training; the validated values are
  center_momentum ≈ 0.97, EMA 0.99→0.999. Higher center_momentum fights collapse.
- **polar pipeline + θ-roll augmentation** — makes classes orientation-invariant
  (group by *structure*, not by how the grain is rotated). Use for powder-like /
  textured samples where orientation is a nuisance variable.
- **SupCon / cluster1d / centroid / spatial** auxiliary losses — optional regularizers
  (contrastive pull/push, 1-D separation, spatial smoothness). Add only if the plain
  DINO map is fragmented or collapses; otherwise keep them off for simplicity.

Reading the output: a good class map is **spatially coherent** (contiguous regions,
not salt-and-pepper) and the **class-average DPs are visibly distinct** (different
ring strength, spottiness, or symmetry).

---

## 4. Clustering the embedding (and NMF clustering)

Both the DINO embedding and the NMF loadings (`W`) are clustered with the same menu.
Choosing the algorithm:

| Method | Use when | Notes |
|---|---|---|
| **K-means** | default; roughly balanced, blob-like classes | fast, needs K; sensitive to scale → features are normalized first |
| **Agglomerative (Ward)** | classes of unequal size / nested structure | deterministic; good for a dendrogram view |
| **HDBSCAN** | unknown #clusters, want an explicit "noise" class | density-based; great when some pixels are junk/amorphous; tune min_cluster_size |
| **Fuzzy-c-means (FCM)** | mixed pixels / gradual transitions | soft memberships; good for continuous crystallization gradients |

Choosing **K**: use the silhouette score (auto-K picks the K with the best
silhouette over a range) as a guide, but always sanity-check against the class map's
spatial coherence and the distinctness of class averages — silhouette alone can
prefer trivial splits. For DINO, K is set at train time; for NMF you can re-cluster
without re-fitting (the **Cluster** button) to try K / methods quickly.

---

## 5. Classical baselines

- **NMF + clustering** — factorizes the stack of DPs into `n_components` non-negative
  components and clusters the per-pixel loadings. Fast, interpretable, a strong
  baseline. Use it to (a) get a quick map without training, and (b) show that DINO
  finds structure NMF misses (or agrees with it). Choose `n_components` with the
  auto heuristic, then re-cluster to explore K. **NMF on a polar+masked stack** is
  the most directly comparable baseline to the DINO map.
- **ACOM (py4DSTEM)** — template-matches each DP against simulated diffraction of a
  known crystal structure to get **orientation** (and **strain**). Use it only for
  **crystalline** samples with a known/assumed structure (CIF). It answers
  "what orientation / strain", not "what phase by appearance". For amorphous or
  unknown-structure material, ACOM is not the right tool — use DINO / NMF.

---

## 6. Which method should I use? (decision guide)

Pick by the **question**, then by **sample type**.

- "Map the distinct regions / phases without labels" → **DINO** (primary). Compare
  with **NMF** to show robustness.
- "Quick map, no training time" → **NMF + K-means**.
- "Orientation or strain map of a crystal" → **ACOM** (needs structure/CIF).
- "Why did the model split these classes?" → **Interpretation** (Grad-CAM +
  ablations + radial profiles + classical concordance).
- "Is my model any good?" → **Score the run** + Section 7.

Sample-type modifiers:
- **Layered materials (e.g. EuInAs)** — orientation/zone-axis dominates; expect ACOM
  to be informative and judge DINO by zone-axis agreement (AMI vs ACOM). These
  samples often need a **confidence/weight loss** (see avg_conf rule, Section 7).
- **Non-layered (e.g. IMC, NaPHI)** — the class parameter is **crystallinity +
  azimuthal spottiness** (2-D Bragg excess), *not* orientation. Judge DINO against
  classical clustering (low ARI = DINO is finding something classical methods miss,
  which is the point) and against SAM masks where available
  (IoU / Dice / count correlation).

---

## 7. Is the result trustworthy? (validity assessment)

No single number suffices; combine:

1. **Spatial coherence** of the class map — real microstructure is contiguous;
   salt-and-pepper means the model keyed on noise.
2. **Class-average distinctness** — the average DP of each class should look
   physically different (ring strength, spottiness, symmetry). Indistinct averages
   = over-clustering.
3. **Silhouette / separation** in embedding space — supportive, not decisive.
4. **Concordance with a baseline** — agreement with NMF (or ACOM zone axis for
   crystals) builds confidence; strong *disagreement* needs a physical explanation.
5. **Confidence at early epochs** — `avg_conf` measured at ~epoch 5 separates samples
   that need a confidence/weight loss (avg_conf ≳ 0.85, e.g. layered/zone-axis cases)
   from those that don't (≲ 0.3). High early confidence with a collapsing map → add
   the weight/confidence loss.
6. **Stability** — re-running (different seed) should give substantially the same map.

Red flags: one class swallows most pixels (collapse), the map tracks only thickness
/ beam intensity (preprocessing/mask too weak), or classes are spatially random.

---

## 8. Parameter starting points

- **General validated recipe:** polar pipeline + θ-roll aug, center_momentum ≈ 0.97,
  EMA 0.99→0.999, ~50 epochs, COM-centering on, beam mask sized to the central disk.
- **Focused map:** K ≈ 6. **Fine structure:** K large (e.g. 60), then merge.
- **Layered / zone-axis sample:** enable the confidence/weight loss; cross-check with
  ACOM.
- **Non-layered / crystallinity sample:** plain polar DINO; compare to NMF + the
  classical crystallinity/spottiness descriptors and SAM masks.
- Always: set mask + crop + COM on the loader, then **Load parameters to model**.

---

## 9. Troubleshooting (symptom → fix)

**Over-clustering** (one real class split into several; class-averages look alike):
- *Data:* strengthen invariances so nuisance variation stops spawning classes —
  COM-centering on, beam mask covering the central disk, polar pipeline + θ-roll aug
  (so rotated grains don't split), mild blur to kill shot-noise splits; check crop /
  vmax aren't clipping signal.
- *Model:* **lower K** (primary); train longer; raise center_momentum (~0.97) + EMA;
  add a consolidation loss (centroid_lambda or cluster1d); lower conf_weight_gamma if
  it over-sharpens.
- *Quick (no retrain):* merge classes in Post-hoc, or re-cluster with smaller K
  (NMF "Cluster" button; or agglomerative + cut the dendrogram).

**Collapse / under-clustering** (one class swallows most pixels):
- *Data:* reduce over-aggressive augmentation; ensure the mask isn't deleting the
  discriminative signal.
- *Model:* raise K; enable the confidence/weight loss (esp. layered/zone-axis,
  avg_conf >~0.85 @ epoch ~5); train longer.

**Salt-and-pepper / spatially incoherent:** stronger preprocessing + mild blur;
add the spatial-smoothness loss (lam_spatial); train longer; consider lower K.

**Map tracks thickness / beam only:** enlarge the beam mask, COM-center, log-stretch;
confirm polar + masking are active.

**Unstable run-to-run:** fix the seed, train longer, raise EMA/center_momentum, add a
consolidation loss — and revisit preprocessing/K (instability ⇒ poorly separated
classes).

After any change: retrain (or re-cluster for NMF), then score + re-check spatial
coherence and class-average distinctness.
