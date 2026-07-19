# Synthetic 4D-STEM Data — Simulate & Validate (Assistant Guide)

This guide is for the **Synthetic** tab (Data ▸ Pre-processing ▸ Synthetic). It
lets you **simulate** a 4D-STEM phantom with known, ground-truth class labels and
then **validate** a trained model against them — something impossible on real data,
which has no labels. Use it to prove the unsupervised classifier recovers real
structure, and as a runnable example.

The tab has two sub-tabs: **Simulate** (build the phantom) and **Validate** (score
a trained model against it).

## Why use it
On real experimental 4D-STEM data there is no ground truth, so the class map can
only be judged by eye. A simulated phantom has a **known** class for every scan
position, so you can measure accuracy, ARI, and NMI. It also lets you test that the
model is invariant to nuisance variation (in-plane rotation) while separating real
structure (3-D orientation, crystallinity).

## Quick start (the one-click example)
1. On the **Simulate** sub-tab click **⚡ WS₂ example**. This loads several WS₂
   domains (different 3-D orientations) and sensible parameters.
2. Set **Engine → finite** and **potential gpts → 512** for clean spots (see
   Engines below).
3. Click **Simulate ▶**. When it finishes, the phantom cube path is auto-filled in
   the Dataset (Pre-processing) tab.
4. Go to the Dataset tab and **Load** it, then **train** a model on it (Training
   tab), and **Run Inference** (Eval / Post-hoc).
5. Back on **Validate**, choose labeling **main**, click **Score ▶**, and pick that
   model's `inference.npz`. You get accuracy / ARI / NMI + a confusion matrix, and a
   saved report.

## Building structures (Simulate sub-tab)
Each row in the Structures list is one domain "phase". Add structures via:
- **+ from CIF…** — load any crystal from a `.cif` file (complex materials).
- **+ built-in crystal…** — pick from built-in ASE crystals (WS₂, MoS₂, Si, Cu, Au,
  Al, Fe, NaCl) — no file needed.
- **+ from ASE builder…** — advanced: a raw `ase.build` call.
- **⚡ WS₂ example** — one click, pre-fills a full multi-domain WS₂ phantom.
- **🔬 View 3D** — open the selected structure in a rotatable 3-D viewer to inspect
  the atoms / stacking / thickness before simulating.

### The three domain axes
Each grain differs along up to three axes, which map to the ground-truth classes:
- **3-D orientation** — the zone axis ([001], [110], [100], or custom h,k,l). Set in
  the structure editor. Different orientation = different class.
- **Crystallinity** — a clickable control (**crystalline / partial / amorphous**) in
  the structure editor. Physically it applies frozen-phonon displacement disorder:
  crystalline = sharp spots, partial = weakened spots + mild diffuse, amorphous =
  diffuse rings. Different crystallinity = different class.
- **In-plane rotation** — a range (e.g. 0–360°) spun about the beam. This is a
  **nuisance** variable: grains that differ only by in-plane rotation must collapse
  to a **single** class. A good model is invariant to it.

## Scan & acquisition parameters
- **Scan Ny / Nx** — number of probe positions (the class-map size). More = more
  patterns, but *diversity comes from the number of grains, not pixels* (all pixels
  in a grain are the same pattern). To give the model more to learn, raise the
  Voronoi seed count, not just the scan.
- **Step (Å)** — real-space spacing between probe positions.
- **Beam (kV)** — accelerating voltage (e.g. 200).
- **Conv α (mrad)** — probe convergence semi-angle. Sets the disk size: small (~0.2)
  → sharp point-like reflections (SAED-like); ~1.5 → few-pixel disks (NBED). Very
  small convergence makes sub-pixel disks that alias under in-plane rotation — keep
  it ~1–2 mrad unless you want SAED.
- **Detector px / Det α_max (mrad)** — detector grid and its maximum angle. Set
  α_max so the diffraction fills the frame: too large leaves a big blank ring around
  the pattern; too small clips high-order reflections. For WS₂ ~28 mrad works.
- **Dose (e/Å²)** and **Electrons: shot | exact** — dose scales each pattern to
  `Dose × step²` electrons/pattern. **shot** Poisson-samples (realistic shot noise;
  higher dose = cleaner, SNR ~ √counts). **exact** uses the exact expected counts
  with NO noise (a clean ideal pattern). Dose = 0 skips scaling/noise.

## Engines (Engine dropdown)
- **multislice** — scanned probe per pixel, `projection="infinite"` (fast, but
  finite-crystal features can smear into streaks / "diffuse" between spots).
- **finite** — `projection="finite"` + ONE high-quality diffraction pattern per
  grain, broadcast to that grain's pixels. **Removes the inter-spot streaks (clean
  spots)** and is often faster (one sim per grain, not per pixel). **Use potential
  gpts ≥ 512 for the sharpest spots.** This is the recommended engine for quality.
- **PRISM** — scattering-matrix scan (experimental here).

## Advanced simulation controls
- **Layout: block | voronoi** — block tiles N phases in a grid; **voronoi** makes
  `Voronoi seeds` random grains, each with a random (phase, orientation, in-plane
  rotation). Voronoi is more realistic; more seeds = more grain diversity = better
  training data (cheap, because the expensive sim is cached per structure).
- **potential gpts** — reciprocal-space resolution of the potential. Higher = sharper
  spots (use ≥ 512 with the finite engine).
- **vacuum pad (Å)** — vacuum around the (orthogonalised, square) atom box. Small pad
  + big crystal → sharper spots (fewer finite-size "added spots").
- **Tile X,Y,Z** (per structure) — supercell repeats. **Bigger in-plane tiling (X,Y)
  = sharper Bragg spots and fewer added spots**, because a larger crystal patch has a
  narrower shape-function. Z = thickness in unit cells / layers.

## Ground-truth class maps (labelings)
Every sim writes several ground-truth maps, each labeling the *same* phantom at a
different strictness. Pick one in the Validate dropdown:
- **A** — phase only.
- **B** — phase + orientation (zone axis).
- **crys** — phase + crystallinity.
- **main** — phase + orientation + crystallinity, **in-plane-rotation-invariant**.
  This is the primary target: it rewards separating real structure while ignoring
  in-plane rotation.
- **C** — full, including in-plane rotation bins (a negative control: a rotation-
  invariant model should NOT match this).

## Output files (next to the cube)
- `phantom.cube.npy` — the 4D-STEM data (drop-in for the loader).
- `phantom.classmap_A/B/crys/main/C.npy` — ground-truth class per scan pixel.
- `phantom.grain_map.npy` — which grain each pixel belongs to (grains can share a
  class).
- `phantom.sim_meta.json` — all parameters + provenance.
- `phantom_preview.png` — class map + class-average diffraction quick-look.
- Sims are written under `runs/synth/<name>_<timestamp>/`, next to the GUI's runs.

## Validating a model (Validate sub-tab)
Prerequisites: (1) a phantom (from Simulate, or **Load existing phantom…**), and
(2) the model's `inference.npz` (train on the phantom cube, then Run Inference).
1. Choose a **Labeling** (start with **main**).
2. Click **Score ▶** and pick the `inference.npz`.
3. It runs **Hungarian** label matching (the model's cluster numbers are arbitrary),
   then reports **pixel accuracy, ARI, NMI, per-class precision/recall/F1** and a
   **confusion matrix** (bright diagonal = good). The report is saved next to the
   phantom as `validation_<model>_<labeling>.png / .txt / .json`, and opens in a
   scrollable window.
- **ARI / NMI** are the honest headline numbers (independent of label matching).
- A near-diagonal confusion matrix means the model recovered the known domains.

## Troubleshooting & advice
- **Extra / "added" spots (a mesh of weak spots between the real Bragg spots)** —
  finite-box shape-function side-lobes. Fix: **increase in-plane Tile X,Y** (e.g.
  4→14) and **reduce vacuum pad** so the crystal patch fills the box.
- **Elliptical rings instead of circular** — was caused by a non-square simulation
  cell; the builder now squares the in-plane box automatically, so rings are
  circular. If you still see it, re-simulate.
- **Diffuse streaks connecting the spots** — an `infinite`-projection artifact of the
  multislice engine. Fix: switch **Engine → finite** (uses `projection="finite"`) and
  set **potential gpts ≥ 512**.
- **Pattern is mostly the central beam / lots of blank space** — the diffraction
  doesn't fill the frame. Lower **Det α_max (mrad)** so the reflections reach the
  frame edge; keep the viewer's log-stretch on to reveal weak disks.
- **In-plane rotations split into separate classes** — the model isn't invariant, or
  the sim corrupted the rotation. Use the **finite** engine (clean, square patterns),
  ensure enough **Voronoi seeds** (many distinct in-plane angles to learn from), and
  keep convergence ~1–2 mrad (not sub-pixel).
- **Model over-clusters the phantom (many classes for a few domains)** — train with a
  **small K (~6–8)**; a large K over-splits a simple phantom.
- **Model trains poorly on the phantom** — the true diversity is the **number of
  grains**, not pixels. Raise **Voronoi seeds** (this is cheap — the expensive sim is
  cached per structure). A bigger scan alone doesn't help.
- **Amorphous domain is slow** — it averages several frozen-phonon configs (8×), so
  amorphous grains cost more per pattern; reduce seeds or use fewer amorphous grains.

## Tips
- For paper-quality patterns use the **finite** engine with **gpts ≥ 512**.
- Score against the **main** labeling — it is the physically meaningful, in-plane-
  invariant target.
- Save an NMF/clustering result with **Save snapshot** and reload it later with
  **Load snapshot…** instead of recomputing.
