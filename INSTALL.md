# Installing DINO-4DSTEM (Windows)

**You don’t need to know anything about programming or “git.”**
Pick one of the two methods below and copy-paste the lines exactly.

The whole thing takes about **15 minutes** — most of it just waiting — and you only do it once.

<br>

---

<br>

## Before you start — install Miniconda *(one time, both methods)*

Miniconda is the free tool that manages Python for the app. You only ever do this once per computer.

<br>

**1.** Go to **[anaconda.com/download/success](https://www.anaconda.com/download/success)**

<br>

**2.** Under **Miniconda Installers**, download **“Miniconda3 Windows 64-bit.”**

<br>

**3.** Open the file and click **Next → I Agree → Next → Install → Finish**.
The defaults are fine — change nothing.

<br>

---

<br>

## Method ①  —  Download &amp; Run

**Recommended. No git, no terminal — nothing but double-clicks.**

<br>

### &nbsp; ⬇ &nbsp; [**Download DINO-4DSTEM (ZIP)**](https://github.com/DanielKhaykelson/dino-4dstem/archive/refs/heads/master.zip)

*(Or, on the repo page, click the green **`< > Code`** button → **Download ZIP**.)*

<br>

**Step 1 — Extract the ZIP**

Open your **Downloads** folder, right-click `dino-4dstem-*.zip` → **Extract All…**

Extract it to a simple path like **`C:\dino-4dstem`**.

> ⚠️ Pick where the folder lives **now** and don’t move it later — the Desktop icons point here. (If you do move it, just redo Step 3.)

<br>

**Step 2 — Build the environment**

Double-click **`install_1_environment.bat`**.

A black window downloads a few GB and builds Python. Let it run until it says **“STEP 1 complete.”**

*Several minutes, one time only.*

<br>

**Step 3 — Install the app**

Double-click **`install_2_dino4dstem.bat`**.

It checks that the app starts, then creates two Desktop icons: **DINO-4DSTEM GUI** and **DINO-4DSTEM Assistant**.

<br>

**Step 4 — Run it**

Double-click the **DINO-4DSTEM** icon on your Desktop.

The first launch takes ~30 seconds while Python starts. That’s it. 🎉

<br>

> **In a hurry?** `install.bat` runs Steps 2 and 3 back-to-back.
>
> If Windows shows *“Windows protected your PC,”* click **More info → Run anyway** — that’s normal for downloaded scripts.

<br>

---

<br>

## Method ②  —  Install with git

**A little more setup, but then getting updates is a single command.**

<br>

**Step 1 — Install Git** *(one time)*

Download from **[git-scm.com/download/win](https://git-scm.com/download/win)** — the defaults are fine.

<br>

**Step 2 — Download the code**

Open a terminal (press **Start**, type *PowerShell*, press **Enter**), then run:

```bash
git clone https://github.com/DanielKhaykelson/dino-4dstem
cd dino-4dstem
```

*This copies the project into a folder named `dino-4dstem`.*

<br>

**Step 3 — Install**

Run the same two installers — double-click them in the folder, or from the terminal:

```bash
.\install_1_environment.bat   # builds the Python environment (slow, once)
.\install_2_dino4dstem.bat    # installs the app + makes the Desktop icons
```

<br>

**Step 4 — Run**

Double-click the **DINO-4DSTEM** Desktop icon.

<br>

### Updating later

From the project folder:

```bash
git pull                        # fetch the latest code
.\install_2_dino4dstem.bat      # re-install the app (seconds — no env rebuild)
```

Only re-run `install_1_environment.bat` if the release notes say the dependencies changed.

<br>

---

<br>

## Point it at your data

No data ships with the app — open your own 4D-STEM files:

`.prz` · `.npz` · `.npy` · `.h5`/`.hdf5` master · `.dm4`/`.dm3` · EMPAD `.raw` · Merlin/Medipix `.mib` · or a **folder of numbered images** (`.tif`/`.png`/…).

<br>

---

<br>

## GPU *(optional, advanced)*

The default install runs on the **CPU** — fine for most use; training is just slower.

For an NVIDIA GPU, open PowerShell in the project folder and run:

```powershell
conda activate dino4dstem; pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

<br>

---

<br>

## Troubleshooting

| Problem | Fix |
|---|---|
| Black window says **“Could not find conda”** | Miniconda isn’t installed — do the first section. |
| **“Windows protected your PC”** popup | Click **More info → Run anyway**. |
| Desktop icon does nothing / error flashes | Re-run **`install_2_dino4dstem.bat`**. Make sure you didn’t move the folder. |
| Assistant mentions downloading **Ollama** | Expected — the assistant’s local AI model installs itself on first use. |
| Training is very slow | The default install uses the **CPU** — see **GPU** above. |

<br>

> **Pip-only (no conda, advanced):** make a Python 3.10 environment, then
> `pip install -r requirements.txt`, and run `python src\gui_dino4dstem.py`.
