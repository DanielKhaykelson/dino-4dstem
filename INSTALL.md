# Installing DINO-4DSTEM on Windows — step by step

This takes about 15 minutes, most of it just waiting. **You do not need to know
anything about programming or "git".** Just follow the steps and copy-paste the
lines exactly.

There are 5 steps:
**1)** install Miniconda · **2)** download the code · **3)** double-click
`install_1_environment.bat` · **4)** double-click `install_2_dino4dstem.bat`
· **5)** double-click the Desktop icon.

> The install is split in two on purpose. **Step 3** builds the Python
> environment — big and slow, but you only ever do it once. **Step 4** installs
> DINO-4DSTEM itself and makes the icons — fast, and it's the *only* step you
> repeat when you get a newer version of the code.
> (In a hurry? `install.bat` just runs both back-to-back.)

---

## Step 1 — Install Miniconda (one time per computer)

Miniconda is the free tool that manages Python for the app.

1. Go to: **https://www.anaconda.com/download/success**
2. Under **Miniconda Installers**, download **"Miniconda3 Windows 64-bit"**.
3. Open the downloaded file and click **Next → I Agree → Next → Install**
   (the **defaults are fine** — don't change anything), then **Finish**.

You only ever do this once.

---

## Step 2 — Download DINO-4DSTEM

1. In your web browser, open the project page (you must be signed in to GitHub
   with access, since it's private):
   **https://github.com/DanielKhaykelson/dino-4dstem**
2. Click the green **`< > Code`** button → **Download ZIP**.
3. Open your **Downloads** folder, **right-click** the `dino-4dstem-*.zip` file →
   **Extract All…**
4. For the location, type a simple path like **`C:\dino-4dstem`** and click
   **Extract**.
5. Open the extracted folder. Inside you should see files like **`install.bat`**,
   **`launch_gui.bat`**, and folders `src`, `docs`.
   > ⚠️ Pick where this folder lives **now** and don't move it later — the desktop
   > icons point to this location. (If you do move it, just redo Step 4.)

---

## Step 3 — Install the environment (one double-click)

1. Double-click **`install_1_environment.bat`**.
2. A black window opens and builds the Python environment. The first time it
   downloads a few GB and takes **several minutes** — let it run until it says
   **"STEP 1 complete"**.
   - If Windows shows **"Windows protected your PC"**, click **More info →
     Run anyway** (this is normal for downloaded scripts).
   - If it says **"Could not find conda"**, Step 1 wasn't finished — install
     Miniconda and try again.

You only ever do this step again if the dependencies change.

---

## Step 4 — Install DINO-4DSTEM (one double-click)

1. Double-click **`install_2_dino4dstem.bat`**.
2. It checks the app can start in the environment (it prints the package
   versions it found), then creates the icons automatically.
3. Two icons now appear on your **Desktop**:
   - **DINO-4DSTEM GUI** — the full app
   - **DINO-4DSTEM Assistant** — the no-window, plain-English assistant

This step is quick — re-run just this one whenever you update the code.

---

## Step 5 — Run

Double-click **DINO-4DSTEM GUI** (or **Assistant**). The first launch can take
~30 seconds while Python starts up. That's it. 🎉

To analyze your data, point the app at your own 4D-STEM files
(`.prz` / `.npz` / `.h5` master / `.npy` / `.dm4`) — no data is included with the app.

---

## Updating later

When there's a new version: download the ZIP again into the **same** folder
(replace the files), then double-click **`install_2_dino4dstem.bat`**. That's
usually all you need — a code update doesn't rebuild the environment, so it
takes seconds and your desktop icons keep working.

Only re-run **`install_1_environment.bat`** if the release notes say the
dependencies changed (or step 2 reports missing packages).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Black window says **"Could not find conda"** | Miniconda isn't installed — do Step 1. |
| **"Windows protected your PC"** popup | Click **More info → Run anyway**. |
| Desktop icon does nothing / error flashes | Re-run **`install.bat`**, then redo Step 4. Make sure you didn't move the folder. |
| Assistant mentions downloading **Ollama** | That's expected — the assistant's AI model installs itself automatically on first use. |
| Training is very slow | Default install uses the **CPU**. For an NVIDIA GPU, see "GPU" below. |

### GPU (optional, advanced)

The default install runs on the CPU (fine for most use; training is slower).
For an NVIDIA GPU, open PowerShell in the project folder and run:

```powershell
powershell -ExecutionPolicy Bypass -Command "conda activate dino4dstem; pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118"
```

### For advanced users (git / pip)

- With git: `git clone https://github.com/DanielKhaykelson/dino-4dstem` instead of
  the ZIP; everything else is the same.
- Pip only (no conda): make a Python 3.10 environment, then
  `pip install -r requirements.txt`, and run `python src\gui_dino4dstem.py`.
- Already have a suitable conda env? Edit the single line `ENV_NAME=...` in
  `_activate.bat` (and `install.bat`) to point at it.
