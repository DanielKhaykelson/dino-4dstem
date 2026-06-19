# DINO-4DSTEM

Self-supervised classification of 4D-STEM diffraction data using a **DINO** model,
with a desktop GUI, a plain-English assistant (local LLM), and a full
NMF / ACOM / interpretation pipeline. Companion code for materials-science work on
nanobeam electron diffraction (NBED) datasets.

Given a 4D-STEM scan (a diffraction pattern at every probe position), the model
learns — **without labels** — to group probe positions into structurally distinct
classes (phases, crystallinity regimes, orientations), producing a **class map**
over the scan and a set of **class-average diffraction patterns**.

---

## 🚀 Quick start (Windows)

No programming or `git` knowledge needed — see **[INSTALL.md](INSTALL.md)** for the
full click-by-click guide. In short:

1. **Install Miniconda** (one time): https://www.anaconda.com/download/success
2. **Download this repo** — green **`< > Code`** button → **Download ZIP** → extract
   to e.g. `C:\dino-4dstem`.
3. **Install** — double-click **`install.bat`** (creates the `dino4dstem`
   environment; takes a few minutes the first time).
4. **Make desktop icons** — open PowerShell in the folder and paste:
   ```powershell
   powershell -ExecutionPolicy Bypass -File make_desktop_shortcuts.ps1
   ```
5. **Run** — double-click the **DINO-4DSTEM GUI** or **Assistant** desktop icon.

> Point the app at your own 4D-STEM files (`.prz` / `.npz` / `.h5` master / `.npy`)
> — no data is bundled.

---

## 🖥️ Three ways to run

| Mode | Launch | Use |
|---|---|---|
| **GUI** | `launch_gui.bat` / `python src/gui_dino4dstem.py` | Full interactive app: load, preprocess, train, infer, NMF, ACOM, interpretation. |
| **Assistant (headless)** | `launch_assistant.bat` / `python src/assistant_headless.py <cube>` | Drive the whole pipeline in plain English with no GUI; auto-saves all figures. |
| **In-GUI assistant** | 💬 button in the GUI | The same assistant embedded as a floating window, with a "show me where to click" teacher mode. |

The natural-language assistant runs on a **local** model via
[Ollama](https://ollama.com), which installs itself automatically on first use.

---

## 🧠 Methodology

1. **Preprocessing** — central-beam masking, COM-centering, optional polar transform
   and percentile/`vmax` intensity scaling of each diffraction pattern.
2. **Self-supervised learning** — a DINO student/teacher model is trained on the
   diffraction patterns (no labels), learning an embedding where structurally similar
   patterns cluster together.
3. **Clustering** — embeddings are clustered into `K` classes and mapped back onto the
   scan grid to give the **class map** and **class-average patterns**.
4. **Interpretation** — a battery explains the classes: Grad-CAM attribution, feature
   ablations, radial profiles, and concordance with classical baselines.
5. **Classical baselines** — **NMF** (with K-means / agglomerative / HDBSCAN / fuzzy-c
   clustering) and **ACOM** orientation/strain mapping (via py4DSTEM) for comparison.

Every analysis (GUI or headless) auto-saves its figures next to the data in a
sequential `<data>_assistant/<analysis>_NNN/` folder, so reruns never overwrite.

---

## 📊 Representative results

| Sample | Type | Headline metric |
|---|---|---|
| NaPHI (Na-poly(heptazine imide)) | non-layered | DINO vs SAM masks: IoU ≈ 0.74, Dice ≈ 0.85, count *r* ≈ 0.999 |
| IMC | non-layered | Distinct from classical clustering (ARI ≈ 0.11–0.21) — captures structure classical methods miss |
| EuInAs | layered | Zone-axis agreement with ACOM, AMI ≈ 0.30 |

---

## 🗂️ Repository layout

```
src/             KEY code imported by the app (flat): entry points, model, data, eval
  gui_app/       desktop GUI package (panels + chat assistant)
  studies/       ablation / sweep / experiment runner scripts
  scripts/       standalone analysis / plotting / baseline utilities
  tools/         other one-off batch scripts
docs/            user manuals (.tex/.pdf) + figure drafts
```

See **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** for a fuller overview.

---

## 📜 Citation

The DINO-4DSTEM manuscript is **in preparation** — please contact the authors before
using these results.

**Related work** (Segment Anything pipeline + abTEM simulations for one of the
samples, NaPHI):

> **Elucidating Structural Disorder in a Polymeric Layered Material: The Case of
> Sodium Poly(heptazine imide) Photocatalyst.**
> D. Khaykelson, G. A. A. Diab, S. R. Cohen, T. Kashti, T. Bendikov, I. Pinkas,
> I. F. Teixeira, N. V. Tarakina, L. Houben, and B. Rybtchinski.
> *Nano Letters* **2025**, 25, 49, 17230–17236.
> DOI: [10.1021/acs.nanolett.5c04946](https://pubs.acs.org/doi/10.1021/acs.nanolett.5c04946)
> · Code: [NaPHI_structural-simulations_SAM](https://github.com/DanielKhaykelson/NaPHI_structural-simulations_SAM)
