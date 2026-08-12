<div align="center">

<img src="assets/dino.png" alt="DINO-4DSTEM" width="72" height="72" style="image-rendering:pixelated">

# DINO-4DSTEM

### See the phases hiding in your 4D-STEM scan.

An unsupervised classifier that groups your diffraction patterns into distinct
structural regions — and shows you **why**. No labels. No coding.

![platform](https://img.shields.io/badge/platform-Windows-0a7bbd?style=flat-square)
![coding](https://img.shields.io/badge/coding-not%20required-0ca678?style=flat-square)
![setup](https://img.shields.io/badge/setup-~15%20min%2C%20once-e8590c?style=flat-square)

<br>

<img src="assets/hero.png" alt="A real NaPHI diffraction pattern is classified into a colored class map" width="820">

<br><br>

### **[⬇  Download &amp; install →](INSTALL.md)**

*Free · Windows · a one-time ~15-minute setup, then double-click to run.*

</div>

---

## What it does

Point it at a scan. It groups every diffraction pattern into structurally
distinct classes — phases, crystallinity, texture — paints them on a map of
your sample, and gives you the evidence behind each class.

<table>
<tr>
<td width="33%" valign="top">

### 🔍 Automatic

Groups every diffraction pattern into distinct classes with **no labels and no
hand-tuning**. Just load a scan and go.

</td>
<td width="33%" valign="top">

### 🧭 Interpretable

See *why* each region is its own class — its diffraction as a **single frame**,
a **grain average**, or the **whole class** — plus attribution maps and an
**NMF** baseline to check the answer.

</td>
<td width="33%" valign="top">

### 🖱️ No code

A **desktop app** plus a **plain-English assistant**. Double-click to launch —
or just ask *“load this file, then train.”*

</td>
</tr>
</table>

---

## See the evidence — at any scale

Every class is checkable. Look at its diffraction as a **single frame**, a
**grain average**, or the **whole-class average** — the noise falls away and
the Bragg spots sharpen as you zoom out.

<div align="center">
<img src="assets/interpret.png" alt="The same NaPHI class as a single frame, a grain average, and a class average — progressively cleaner" width="920">
</div>

---

## Bring your own data

Point it at the files you already have — **nothing is uploaded**, everything
stays on your machine.

| Format | Source / detector | Notes |
|---|---|---|
| `.prz` · `.npz` · `.npy` | py4DSTEM / NumPy cubes | memory-mapped, low RAM |
| `.h5` · `.hdf5` master | Dectris / Eiger / EMPAD masters | stitches external links; asks scan shape if 3-D |
| `.dm4` · `.dm3` | Gatan | reads calibration from metadata |
| `.raw` | EMPAD | metadata rows cropped automatically |
| `.mib` (+ `.hdr`) | Quantum Detectors **Merlin / Medipix3** | counting mode |
| **folder of images** | `.tif` / `.png` / `.jpg` / … | numbered by scan order → cube (greyscale + optional binning) |

---

## Get started — two ways

Both need **[Miniconda](https://www.anaconda.com/download/success)** (free)
installed first. Full step-by-step with screenshots: **[INSTALL.md](INSTALL.md)**.

**① Download &amp; run — for everyone (no git):**
Download the ZIP → double-click `install_1_environment.bat` → double-click
`install_2_dino4dstem.bat` → launch from the Desktop icon.

**② Install with git — for one-command updates:**
```bash
git clone https://github.com/DanielKhaykelson/dino-4dstem
```
then run the two installers. Update anytime with `git pull`.

> First time? The download bundle includes **`GETTING_STARTED.docx`** — a
> picture-by-picture, click-by-click walkthrough.

---

## Learn more

- 🧭 **What it does &amp; results** — [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md)
- 🧠 **The methods explained** (DINO, NMF, clustering, PCA) — [`docs/assistant/CONCEPTS.md`](docs/assistant/CONCEPTS.md)
- 📕 **Full manual** — [`docs/USER_MANUAL.pdf`](docs/USER_MANUAL.pdf)
- 🛠️ **Install guide** — [`INSTALL.md`](INSTALL.md)

<div align="center">
<br>
<sub>🦕 Windows only · companion code for ongoing 4D-STEM / electron-diffraction research.</sub>
</div>
