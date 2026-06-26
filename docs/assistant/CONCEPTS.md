# DINO-4DSTEM — Concepts & Background (for the assistant)

Plain, accurate background the assistant can draw on when a user asks about the
methods. Grounded reference — prefer these explanations over improvising.

---

## 1. The problem: 4D-STEM / unsupervised phase mapping

A 4D-STEM scan records a 2-D diffraction pattern (DP) at every probe position,
giving a 4-D array (Ny × Nx × H × W). Each DP encodes the local structure
(crystallinity, phase, orientation, texture). The unsupervised goal: group the
probe positions into a small number of structurally distinct classes — a
"class map" over the scan — **without labels**, plus a representative
(class-average) DP per class.

Two broad families do this here: **linear matrix factorization** (NMF, PCA) and a
**self-supervised neural network** (DINO). Both feed a **clustering** step.

---

## 2. PCA — Principal Component Analysis

**What:** a linear, orthogonal transform that finds directions (principal
components) of **maximum variance** in the data. Mathematically, the components
are the eigenvectors of the data covariance matrix (equivalently the right
singular vectors from the SVD of the mean-centered data), ordered by explained
variance.

**Key properties**
- Requires **mean-centering**; often features are standardized first.
- Components are **orthogonal** and can have **negative** entries.
- Gives the **optimal linear reconstruction** for a given number of components
  (Eckart–Young theorem) and is essentially unique (up to sign).
- Holistic, not parts-based: a component is a global pattern mixing + and −
  contributions, which is often **hard to interpret physically**.

**Used for:** denoising, whitening, compression, visualization, and as a
preprocessing/baseline step.

---

## 3. NMF — Non-negative Matrix Factorization

**What:** factor a non-negative data matrix V (m×n, all ≥ 0) into two non-negative
matrices, V ≈ W·H, with W (m×k) the **components/basis** and H (k×n) the
**coefficients/loadings**, k ≪ m,n. Minimizes a reconstruction cost — Frobenius
norm ‖V − WH‖² or KL divergence — subject to W,H ≥ 0.

**Key properties**
- **Non-negativity → additive, "parts-based"** representation: components combine
  only by addition, so for diffraction each component is itself viewable as a
  non-negative DP (directly interpretable).
- Solved by **multiplicative updates** (Lee & Seung) or alternating/coordinate
  descent; the problem is non-convex, so results depend on **initialization**
  (NNDSVD helps) and can hit local minima.
- **Not unique** (scaling/permutation ambiguity); sensitive to feature scaling.
- **Linear**, like PCA.

**Choosing k (n_components):** look for a knee in reconstruction error vs k, or
judge by how interpretable/stable the components are. In this app you can re-cluster
the loadings without refitting to explore the downstream cluster count.

**Used for:** a fast, interpretable, label-free baseline map; the per-pixel
loadings (H) are clustered to produce a class map.

---

## 4. PCA vs NMF (quick comparison)

| | PCA | NMF |
|---|---|---|
| Constraint | orthogonal components | non-negativity (W,H ≥ 0) |
| Sign | + and − allowed | non-negative only |
| Interpretability | holistic, often abstract | parts-based, components look like DPs |
| Uniqueness | unique (up to sign), variance-optimal | non-unique, init-dependent |
| Centering | requires mean-centering | uses raw non-negative data |
| Both | linear; sensitive to nuisance variation (e.g. in-plane rotation splits one phase into several components) |

Neither is rotation-invariant or nonlinear — which is the gap DINO fills.

---

## 5. Self-supervised learning (SSL)

**Idea:** learn useful representations from **unlabeled** data by solving a
"pretext" task whose targets come from the data itself, then use the learned
embedding for downstream tasks (clustering, classification) — no human labels.

**Families**
- **Contrastive** (SimCLR, MoCo): pull together two augmented views of the same
  sample (positives), push apart different samples (negatives).
- **Self-distillation / non-contrastive** (BYOL, **DINO**): match a student to a
  teacher across views; **no negatives** needed.
- **Masked modeling** (MAE, BERT-style): reconstruct hidden parts of the input.
- **Clustering-based** (DeepCluster, SwAV): alternate clustering and feature
  learning.

Why SSL here: diffraction data is unlabeled and the classes of interest are subtle
mixtures of features; SSL learns a representation where structurally similar DPs
sit close together, beyond any single hand-chosen descriptor.

---

## 6. DINO (self-distillation with no labels)

**What:** two networks of the same architecture — a **student** and a **teacher**.
The teacher's weights are an **exponential moving average (EMA)** of the student's
(no gradient on the teacher). Both see **augmented views** of the same input
(DINO's "multi-crop": several global + local crops). Each network maps a view to a
probability vector over **K prototypes** (a softmax with a temperature). The loss
is the **cross-entropy** between the teacher's distribution and the student's, so
the student learns to predict the (sharper) teacher output across views.

**Avoiding collapse** (the danger that all outputs become identical): DINO balances
two operations on the teacher output —
- **centering** (subtract a running EMA mean) prevents one prototype dominating,
- **sharpening** (low teacher temperature) prevents the uniform distribution.
Together they keep the output distribution informative.

**Why it's powerful:** no labels, no negatives; the learned features are
semantically meaningful and the soft assignment over K prototypes acts as a
clustering.

**This project's adaptation (DINO-4DSTEM):**
- Operates on diffraction patterns; a **polar transform** maps (kx,ky)→(radius,
  angle) so an in-plane **rotation becomes a shift** along the angle axis.
- A **θ-roll augmentation** then makes classes **rotation-invariant** (group by
  structure, not grain orientation) — directly addressing the linear-method weakness.
- Beam masking + COM-centering remove the dominating central (000) beam.
- K prototypes → a soft class assignment per probe position → the class map; the
  prototypes are learned **jointly** with the features (clustering and
  representation are coupled, unlike NMF→cluster).
- Key knobs: K, epochs, teacher EMA, center momentum (fights collapse). A trained
  model can **transfer** to sister samples/orientations — something a
  sample-specific NMF/PCA cannot do.

---

## 7. Clustering methods

Both the DINO embedding and the NMF loadings are partitioned by a clustering
algorithm. Normalize/scale features first; choice of distance (Euclidean vs cosine)
matters.

| Method | Idea | Good when | Watch out |
|---|---|---|---|
| **K-means** | minimize within-cluster squared distance to centroids | roughly spherical, similar-size clusters; fast default | must set K; sensitive to init (use k-means++) and scale |
| **Agglomerative (Ward)** | merge points bottom-up into a dendrogram | unequal sizes, nested structure, want a hierarchy | O(n²) memory-ish; pick K or a cut height |
| **DBSCAN / HDBSCAN** | density-based; dense regions = clusters, sparse = noise | unknown #clusters, arbitrary shapes, explicit outlier/noise class | tune min_cluster_size / eps; HDBSCAN handles varying density |
| **Fuzzy c-means (FCM)** | each point gets soft membership in every cluster | gradual transitions / mixed pixels (e.g. crystallization gradients) | still needs c; memberships need thresholding to label |
| **Gaussian Mixture (GMM)** | probabilistic elliptical clusters via EM | elliptical clusters, want soft/probabilistic assignment | assumes Gaussian; can overfit; needs #components |
| **Spectral** | cluster the graph Laplacian eigenvectors | non-convex / manifold-shaped clusters | builds an n×n affinity; costly for large n |

**Choosing the number of clusters K:** silhouette score (−1…1, higher = better
separation), elbow on inertia, gap statistic, Davies–Bouldin (lower better),
Calinski–Harabasz (higher better); BIC/AIC for GMM. These are guides only — always
cross-check against **spatial coherence** of the class map and the **distinctness of
class-average DPs**; a metric can prefer a trivial split.

**Evaluating a clustering**
- *Internal* (no labels): silhouette, Calinski–Harabasz, Davies–Bouldin.
- *External* (vs a reference labeling): **ARI** (adjusted Rand index),
  **AMI/NMI** (adjusted/normalized mutual information), homogeneity & completeness.
  Note: a *low* ARI between DINO and a classical method can be the point — it means
  DINO captured structure the classical method missed.

---

## 8. Dimensionality reduction for visualization

- **PCA** — linear; quick 2-D/3-D view + denoising (see §2).
- **t-SNE / UMAP** — nonlinear embeddings that preserve local neighborhoods, used to
  **visualize** high-dimensional embeddings (e.g. the DINO features) as a 2-D
  scatter. Great for inspection, but distances/," cluster sizes" in t-SNE/UMAP plots
  are not quantitatively meaningful — don't read cluster validity off them.

---

## 9. Glossary

- **Embedding / latent space:** the vector representation a model maps each input to;
  similar inputs → nearby vectors.
- **Prototype:** a learned reference direction; DINO outputs a soft assignment over K
  prototypes (≈ cluster centers learned during training).
- **Soft vs hard assignment:** soft = a probability over classes per pixel; hard =
  the single argmax class.
- **Class map:** the hard (or soft) class assigned to each probe position, shown on
  the scan grid.
- **Class-average DP:** the (confidence-weighted) mean diffraction pattern of a class
  — what a class "looks like".
- **Collapse:** degenerate solution where the model assigns (almost) everything to
  one class / identical outputs.
- **EMA (teacher):** exponential moving average of the student weights; a slowly
  updated, more stable target network.

---

## 10. When to use what (summary)

- **Quick, interpretable, no training** → **NMF** (+ clustering). Components read as DPs.
- **Linear denoising / variance view / preprocessing** → **PCA**.
- **Label-free classes robust to rotation, transferable across samples, capturing
  subtle mixed features** → **DINO** (then cluster its embedding).
- **Orientation / strain of a known crystal** → **ACOM** (template matching; not in
  this file — it's a different, structure-based method).
- **Pick the clustering** by cluster shape/size and whether you need a noise class or
  soft memberships (see §7), and validate K with silhouette **and** the map's spatial
  coherence + class-average distinctness.
