# DINO-4DSTEM

**Find the distinct regions in your 4D-STEM data — automatically.**
Point it at a scan, and it groups the diffraction patterns into structurally
distinct classes (phases, crystallinity, texture) and shows them on a map — no
labels, no hand-tuning required. There's a desktop app and a plain-English
assistant that can run the whole thing for you.

---

## ⬇️ Get it (Windows)

**1. Download** the latest bundle:
👉 **[Download dino-4dstem.zip](https://github.com/DanielKhaykelson/dino-4dstem/releases/latest)**
(then right-click → *Extract All*)

**2. Install the environment** — double-click **`install_1_environment.bat`**
(one time, several minutes).

**3. Install DINO-4DSTEM** — double-click **`install_2_dino4dstem.bat`**
(fast; makes the Desktop icons). Updating the code later? Just re-run this one.

**4. Run** — double-click the **DINO-4DSTEM** icon on your Desktop. 🦕

> In a hurry: `install.bat` runs steps 2 and 3 back-to-back.

> First time? The bundle includes **`GETTING_STARTED.docx`** — a picture-free,
> click-by-click walkthrough. You need **Miniconda** installed first (free):
> https://www.anaconda.com/download/success

---

## What you get

- **GUI app** — load data, preprocess, train, and view the class map + the average
  diffraction pattern of each class.
- **Assistant** — ask in plain English ("load this file, then train", "is my model
  good?", "I think it over-clustered — how do I fix it?"). Runs locally and free
  (or with a free Google Gemini key for speed without a GPU).
- **Comparisons** — NMF and ACOM baselines, plus an interpretation report that
  explains *why* the classes split.

Bring your own data (`.prz` / `.npz` / `.h5` master / `.npy` / `.dm4` / EMPAD `.raw`). Nothing is uploaded.

---

## Learn more

- 📘 **How to use it:** [`GETTING_STARTED.docx`](GETTING_STARTED.docx) · detailed [`INSTALL.md`](INSTALL.md)
- 🧭 **What it does / results:** [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md)
- 🧠 **The methods explained (NMF, DINO, clustering, PCA):** [`docs/assistant/CONCEPTS.md`](docs/assistant/CONCEPTS.md)
- 📕 **Full manual:** [`docs/USER_MANUAL.pdf`](docs/USER_MANUAL.pdf)

Windows only. Companion code for ongoing 4D-STEM / electron-diffraction research.
