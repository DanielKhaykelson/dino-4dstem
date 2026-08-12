# Installing DINO-4DSTEM (Windows)

**You do not need to know anything about programming or “git.”** Pick one of the
two methods below and copy-paste the lines exactly. The whole thing takes about
**15 minutes**, most of it just waiting — and you only do it once.

---

## Step 0 — Install Miniconda (one time per computer, both methods)

Miniconda is the free tool that manages Python for the app.

1. Go to **https://www.anaconda.com/download/success**
2. Under **Miniconda Installers**, download **“Miniconda3 Windows 64-bit.”**
3. Open the file → **Next → I Agree → Next → Install** (the **defaults are
   fine** — change nothing) → **Finish**.

You only ever do this once.

---

## Method ①  —  Download &amp; Run  *(recommended · no git, no terminal)*

The simplest path. Nothing but double-clicks.

1. **Download the code.** On the project page, click the green
   **`< > Code`** button → **Download ZIP**.
2. **Extract it.** Open your **Downloads** folder, right-click the
   `dino-4dstem-*.zip` → **Extract All…**, and extract to a simple path like
   **`C:\dino-4dstem`**.
   > ⚠️ Pick where the folder lives **now** and don’t move it later — the
   > Desktop icons point at this location. (If you do move it, just redo step 4.)
3. **Build the environment.** Double-click **`install_1_environment.bat`**.
   A black window downloads a few GB and builds Python — let it run until it
   says **“STEP 1 complete.”** (Several minutes, one time.)
4. **Install the app.** Double-click **`install_2_dino4dstem.bat`**. It checks
   the app starts, then creates two **Desktop icons** — *DINO-4DSTEM GUI* and
   *DINO-4DSTEM Assistant*.
5. **Run.** Double-click the **DINO-4DSTEM** icon. The first launch takes
   ~30 seconds while Python starts. That’s it. 🎉

> In a hurry? **`install.bat`** runs steps 3 and 4 back-to-back.
> If Windows shows *“Windows protected your PC,”* click **More info → Run
> anyway** (normal for downloaded scripts).

---

## Method ②  —  Install with git  *(for one-command updates later)*

A little more setup, but then getting the newest version is a single command.

1. **Install Git** (one time): **https://git-scm.com/download/win** — defaults
   are fine.
2. **Download the code.** Open a terminal (press **Start**, type *PowerShell*,
   Enter) and run:
   ```bash
   git clone https://github.com/DanielKhaykelson/dino-4dstem
   cd dino-4dstem
   ```
   *(This copies the project into a folder named `dino-4dstem`.)*
3. **Install** — run the same two installers (double-click them in the folder,
   or from the terminal):
   ```bash
   .\install_1_environment.bat   # builds the Python environment (slow, once)
   .\install_2_dino4dstem.bat    # installs the app + makes Desktop icons
   ```
4. **Run** — double-click the **DINO-4DSTEM** Desktop icon.

### Updating later (git method)

Whenever there’s a new version, from the project folder:
```bash
git pull                        # fetch the latest code
.\install_2_dino4dstem.bat      # re-install the app (seconds; no env rebuild)
```
Only re-run `install_1_environment.bat` if the release notes say the
dependencies changed.

---

## Point it at your data

No data ships with the app — open your own 4D-STEM files:
`.prz` · `.npz` · `.npy` · `.h5`/`.hdf5` master · `.dm4`/`.dm3` · EMPAD `.raw`
· Merlin/Medipix `.mib` · or a **folder of numbered images** (`.tif`/`.png`/…).

---

## GPU (optional, advanced)

The default install runs on the **CPU** (fine for most use; training is slower).
For an NVIDIA GPU, open PowerShell in the project folder and run:

```powershell
conda activate dino4dstem; pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Black window says **“Could not find conda”** | Miniconda isn’t installed — do Step 0. |
| **“Windows protected your PC”** popup | Click **More info → Run anyway**. |
| Desktop icon does nothing / error flashes | Re-run **`install_2_dino4dstem.bat`**. Make sure you didn’t move the folder. |
| Assistant mentions downloading **Ollama** | Expected — the assistant’s local AI model installs itself on first use. |
| Training is very slow | The default install uses the **CPU** — see **GPU** above. |

### Pip-only (no conda, advanced)

Make a Python 3.10 environment, then `pip install -r requirements.txt`, and run
`python src\gui_dino4dstem.py`.
