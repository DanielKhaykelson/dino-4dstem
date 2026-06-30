# DINO-4DSTEM — Project Summary

Self-supervised classification of 4D-STEM diffraction data with a DINO-based
model, plus a full interpretation pipeline, a desktop GUI, and a natural-language
assistant (local LLM via Ollama). Built for paper-level materials-science work on
nanobeam electron diffraction (NBED) datasets.

---

## What it does

Given a 4D-STEM scan (a diffraction pattern at every probe position), the model
learns — **without labels** — to group probe positions into structurally distinct
classes (phases, crystallinity regimes, orientations). The result is a **class
map** over the scan and a set of **class-average diffraction patterns**, which a
microscopist can interpret as physically meaningful regions.

On top of the classifier there is an **interpretation battery** that explains
*why* the model split classes the way it did (Grad-CAM, ablations, radial
profiles, classical baselines such as NMF/ACOM for comparison).

## Three ways to run it

| Entry point | Launch | Use |
|---|---|---|
| **GUI** | `launch_gui.bat` (or `python src/gui_dino4dstem.py`) | Full interactive app: load data, preprocess, train, infer, NMF, ACOM, interpretation. |
| **Assistant (headless)** | `launch_assistant.bat` (or `python src/assistant_headless.py <cube>`) | Drive the whole pipeline in plain English with no GUI; saves all figures automatically. |
| **In-GUI Assistant** | 💬 button inside the GUI | Same natural-language assistant, embedded as a floating window; can also act as a teacher that highlights where to click. |

`make_desktop_shortcuts.ps1` creates two desktop icons (GUI + Assistant).

## Key capabilities

- **Self-supervised DINO classifier** for 4D-STEM diffraction (`dino_sr_contrastive_model.py`).
- **Memory-safe loading** of large cubes (`.prz`, `.npz`, `.h5` master, `.npy`, `.dm4`) via
  zip-offset memmap + optional real-space binning prompt — an 8.6 GB `.prz` loads at
  ~0.7 GB RAM.
- **Classical baselines** for comparison: NMF + clustering (K-means / agglomerative /
  HDBSCAN / fuzzy-c-means) and ACOM orientation/strain mapping (py4DSTEM).
- **Interpretation reports**: class maps, class averages, radial profiles, Grad-CAM
  ladders, ablation maps, classical-vs-DINO concordance.
- **Auto figure saving**: every analysis (GUI or headless) writes figures to a
  sequential `<data>_assistant/<analysis>_NNN/` folder next to the data — reruns never
  overwrite.
- **Natural-language assistant**: local Ollama model (default), agent/tool loop,
  learned knowledge base + semantic RAG over the docs, auto-install of Ollama on
  first use.

## Representative results (paper work)

| Sample | Type | Headline metric |
|---|---|---|
| NaPHI (Na-poly(heptazine imide)) | non-layered | DINO vs SAM masks: IoU ≈ 0.74, Dice ≈ 0.85, count r ≈ 0.999 |
| IMC | non-layered | Distinct from classical clustering (ARI ≈ 0.11–0.21) — captures structure classical methods miss |
| EuInAs | layered | Zone-axis agreement with ACOM, AMI ≈ 0.30 |

(See the manual and figures below for full context.)

## Repository layout

All Python lives under `src/`; the root holds only launchers, docs, and config.

```
src/                          KEY code (imported by the app) lives flat here:
  gui_dino4dstem.py             GUI entry point
  assistant_headless.py         headless assistant entry point
  assistant_io.py               figure/output saving
  data.py, eval_all.py          data loading, evaluation
  dino_sr_contrastive_model.py  model + training
  scorecard.py, contrastive_eval.py, sam_utils.py, ...   core helpers
  gui_app/                      desktop GUI package (all panels + chat assistant)
  studies/                      ablation / sweep / experiment runner scripts
  scripts/                      standalone analysis / plotting / baseline utilities
  tools/                        other one-off batch scripts
docs/                          manuals (.tex/.pdf) + figure drafts
launch_gui.bat, launch_assistant.bat,
  install.bat, make_desktop_shortcuts.ps1     install / entry points / desktop icons
```

`studies/`, `scripts/`, and `tools/` are NOT imported by the GUI/assistant — they
are run directly. The "key" set is whatever the two entry points import; sorting
preserved that flat in `src/` so imports keep working.
- `docs/USER_MANUAL.pdf`, `docs/USER_MANUAL_QUICKSTART.pdf` — **user manuals**.
- `docs/USER_MANUAL.tex`, `docs/USER_MANUAL_QUICKSTART.tex` — manual sources.
- `docs/paper/draft_v2/figs/latest_review/*.png` — **latest figure drafts**.

## Documentation

- **Full manual:** [`docs/USER_MANUAL.pdf`](docs/USER_MANUAL.pdf)
- **Quickstart:** [`docs/USER_MANUAL_QUICKSTART.pdf`](docs/USER_MANUAL_QUICKSTART.pdf)
- **Latest figure drafts:** [`docs/paper/draft_v2/figs/latest_review/`](docs/paper/draft_v2/figs/latest_review/)

## Environment

- Conda env `py4DSTEM_SAM` (Python 3.10). Key deps: PyTorch, py4DSTEM, scikit-learn,
  customtkinter, matplotlib, requests; optional Ollama for the assistant.
- Windows note: torch `DataLoader` is run with `num_workers=0` (avoids a Windows
  multiprocessing hang); process-spawning entry points are guarded with
  `if __name__ == "__main__":`.

## Privacy

This is a **private** repository. Raw 4D-STEM data, model checkpoints, unpublished
manuscript drafts, and third-party PDFs are intentionally **not** tracked (see
`.gitignore`). The manuals and figure drafts here are included for reference only.
