# Interpreting DINO clustering **without ACOM** (no CIF available)

Some datasets cannot be ACOMed — we have no CIF, the phase is unknown, or the
material is poorly crystalline. ACOM only contributes **two** things to the
interpretation battery:

1. orientation/phase **factors** for the probe ranking (ACOM correlation,
   n Bragg peaks, phase id), and
2. the **DINO-vs-zone-axis cross-check** (is the partition just orientation?).

Everything else is **CIF-free**. This document is the plan for the no-ACOM
case: what still runs, how to read it, and which extra tests substitute for
the orientation question.

## What still works without a CIF

| Test | Needs | Question it answers |
|---|---|---|
| **Probing + signatures** | embeds + cube | which physical factor is encoded / separates the classes |
| **Causal ablations** | model + cube | which input signal the clustering *causally* depends on |
| **Class averages** | cube + assigns | what each class' mean pattern / radial profile looks like |
| **Grad-CAM / IG** | model | *where* in the pattern the model attends |
| **Classical-baseline comparison** | cube + assigns | could a non-DINO classical method reproduce the map |

The probe factor set drops the ACOM columns and keeps the **CIF-free physical
factors**: total **scattered intensity** (virtual DF / mass-thickness),
**crystallinity** (1-D peak/halo ratio), and **spottiness** (azimuthal
variance). The ablations (scattered-norm, radial-only, blur, low-q, high-q)
are model-based and never needed ACOM.

## Substituting for the orientation question

Without ACOM we cannot *index* orientation, but we can still test whether the
partition is **orientation-like** (i.e. driven by the angular arrangement of
spots rather than by intensity/radial content):

- **`radial_only` ablation** — replace each pattern by its azimuthal average.
  If the map survives, the decision is purely radial (intensity/ring content),
  i.e. *not* orientation. If it collapses, 2-D angular structure matters
  (which *includes*, but is not specific to, orientation).
- **`theta-roll invariance` (design)** — models trained with azimuthal
  (θ-roll) augmentation are in-plane-rotation invariant by construction, so
  they cannot key on absolute orientation regardless of ACOM. State this from
  the training config (`theta_shift_range_*`).
- **azimuthal-spottiness factor** — high azimuthal variance = spotty
  (oriented single-crystal) rings; if classes do *not* separate on this
  factor (low η²), orientation is unlikely to be the axis.

These three together give a confident "orientation is / isn't a driver"
read **without a CIF**. When a CIF later becomes available, the ACOM
cross-check confirms it quantitatively (AMI/ARI vs zone axis).

## The "is it unique?" test (classical baselines)

To decide whether DINO found something a classical pipeline could not, we
cluster a series of **classical** 4D-STEM feature sets into the same K and
measure agreement (ARI/AMI) with the DINO partition:

1. **Virtual dark-field** — total scattered intensity (one number/pixel).
2. **Azimuthal integration** — the 1-D radial profile → PCA → KMeans.
3. **Pattern PCA** — linear decomposition of the 2-D patterns → KMeans.
4. **Pattern NMF** — non-negative decomposition (the NMF tab) → KMeans.
5. **Combination** — DF + radial + pattern features concatenated → KMeans.

Reading the result:

- best single ARI **≥ 0.8** → a classical method already reproduces the map;
  DINO is not adding distinctive structure.
- combination ARI **≥ 0.6** and **> best single + 0.1** → DINO behaves like a
  *non-linear blend* of classical descriptors (a combination reproduces it).
- best ARI **0.5–0.8** → classical methods partially reproduce it; DINO
  refines beyond them.
- best ARI **< 0.5** → the partition is **distinctive** — not a re-labelling
  of any virtual-DF / radial / PCA / NMF map.

## Extra tests worth adding for no-CIF data

These are CIF-free and sharpen the physical reading; candidates for future
additions to `interpret_core`:

- **Thickness proxy (t/λ)** ≈ `log(I_total / I_unscattered)` per pixel, to
  *disentangle* the scattered-intensity axis into crystallinity vs
  thickness/mass (the one thing the intensity ablation alone cannot split).
- **Rotational-symmetry order** — FFT of the azimuthal profile at fixed q;
  the dominant n-fold order is a CIF-free structural descriptor (amorphous
  halo → no order; 6-fold → hexagonal packing, etc.).
- **Pattern-correlation spectral clustering** — build the pairwise pattern
  correlation matrix on a subsample, spectral-cluster it; another strong
  classical baseline for the uniqueness test.
- **Virtual-aperture panel** — correlate the DINO map against a bank of
  annular virtual detectors (q-band intensities); identifies *which* q-band
  best explains the partition (a coarser cousin of the q-mask ablation).
- **Amorphous/crystalline 1-D peak count** — number of resolved peaks above
  a k·σ baseline in the radial profile, as a crystallinity index independent
  of ACOM peak finding.

## Decision flow

```
linked run + dataset
        │
        ├─ ACOM outputs on disk?  ── yes ─→ add orientation/phase factors
        │                                   + DINO-vs-zone-axis cross-check
        │
        └─ no ─→ CIF-free battery:
                  probing(scattered/crystallinity/spottiness)
                  ablations(scattered-norm/radial-only/blur/low-q/high-q)
                  class averages + Grad-CAM/IG
                  classical-baseline comparison  → uniqueness verdict
                  orientation read via radial_only + θ-roll + spottiness
```

In the GUI (Analysis ▸ Post-hoc ▸ Interpretation) this is automatic: if no
ACOM run is found the panel says so, skips the two ACOM-only pieces, and runs
everything above.
