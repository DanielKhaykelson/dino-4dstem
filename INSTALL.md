# Installing DINO-4DSTEM (Windows)

Three steps: **install once → make icons → run from the desktop.**

## 0. Prerequisite (one time per computer)

Install **Miniconda** (free):
https://docs.conda.io/projects/miniconda/en/latest/ — accept the defaults.
(Full Anaconda works too.)

## 1. Get the code

Download or clone this repository into a folder of your choice, e.g.
`C:\dino-4dstem`. (It's a private repo, so you need access — use
**Code → Download ZIP**, or `git clone <repo-url>`.)

## 2. Install once

Double-click **`install.bat`** (or run it in a terminal).
It creates a conda environment named `dino4dstem` with everything needed.
The first run downloads a few GB and takes several minutes.

## 3. Make the desktop icons (copy-paste one line)

When `install.bat` finishes it prints this exact line — run it from the project
folder in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File make_desktop_shortcuts.ps1
```

Two icons appear on your Desktop:

- **DINO-4DSTEM GUI** — the full interactive app
- **DINO-4DSTEM Assistant** — the no-GUI natural-language assistant

## 4. Run

Double-click an icon. That's it.

---

## Notes

- **Data isn't included.** Point the app at your own 4D-STEM files
  (`.prz` / `.npz` / `.h5` master / `.npy`).
- **Assistant model:** the natural-language assistant uses **Ollama**, which
  installs itself automatically the first time you use it.
- **GPU (optional):** the default install uses **CPU** PyTorch. For an NVIDIA GPU,
  after step 2 run:
  ```
  conda activate dino4dstem
  pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
  ```
- **Updating:** re-run `install.bat` after pulling new changes — it updates the env.
- **Already have a suitable env?** Edit the single line `ENV_NAME=...` in
  `_activate.bat` (and `install.bat`) to point at it.
- **Pip-only (no conda):** create a Python 3.10 env, `pip install -r requirements.txt`,
  then run `python src\gui_dino4dstem.py` (or `src\assistant_headless.py`).
- **Optional features** (pretrained-DINO panel, synthetic diffraction, SAM masks,
  GPU ACOM) need extra packages — see the commented "OPTIONAL extras" block in
  `requirements.txt`.
