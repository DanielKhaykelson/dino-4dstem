# ACOM GPU (cupy) — install & rollback notes

Snapshot taken **2026-06-07** of conda env **`py4DSTEM_SAM`**, BEFORE installing cupy.
Use this to verify a clean install and to roll back if anything breaks.

## Current environment (known-good baseline)

| item | value |
|---|---|
| Python | 3.10.18 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| torch | 2.7.1+cu118  (CUDA **11.8**) |
| py4DSTEM | 0.14.19 |
| GPU | NVIDIA RTX 4080 |
| driver | 536.25 (supports up to CUDA 12.2) |
| cupy | **NOT installed** (ACOM runs on CPU) |

Snapshot files in this folder:
- `pip_freeze_20260607.txt`     — exact pip package versions (190 pkgs)
- `conda_list_20260607.txt`     — full conda list (human-readable)
- `conda_explicit_20260607.txt` — explicit package URLs (exact rebuild)
- `environment_20260607.yml`    — `conda env export` (rebuild from scratch)

## Why this is low-risk

`cupy-cudaXXx` is a **self-contained binary wheel** that bundles its own CUDA
runtime libraries. It does **not** touch torch, numpy, scipy or py4DSTEM — it
only adds two packages: **`cupy-cuda11x`** and its dep **`fastrlock`**. torch
and cupy each load their own CUDA runtime, so they coexist.

The only thing to watch: pip must **not** downgrade/upgrade `numpy`. cupy
supports numpy 1.26, so it shouldn't — but verify with the dry-run below.

## Install (recommended: cupy-cuda11x, matches torch's CUDA 11.8)

All commands use the env's interpreter explicitly so you can't hit the wrong env:

```bat
set PYEXE=C:\Users\danielkh\AppData\Local\anaconda3\envs\py4DSTEM_SAM\python.exe

REM 1) DRY RUN first — confirm it ONLY adds cupy-cuda11x + fastrlock,
REM    and does NOT change numpy / torch / scipy.  If it wants to touch
REM    anything else, STOP.
"%PYEXE%" -m pip install cupy-cuda11x --dry-run

REM 2) Install (RTX 4080 / Ada is supported by CUDA 11.8)
"%PYEXE%" -m pip install cupy-cuda11x
```

> Alternative: `cupy-cuda12x` (matches the driver's CUDA 12.2). Either works on
> the 4080; `cuda11x` is chosen to match torch's runtime and minimise surprises.

## Verify (cupy works AND nothing else broke)

```bat
"%PYEXE%" -c "import cupy; print('cupy', cupy.__version__); print('devices', cupy.cuda.runtime.getDeviceCount()); print('sum', int(cupy.arange(5).sum()))"
"%PYEXE%" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"%PYEXE%" -c "import numpy, scipy, py4DSTEM; print('numpy', numpy.__version__)"
```
Expected: cupy prints a version + `devices 1` + `sum 10`; torch still `cuda True`;
numpy still **1.26.4**.

Then in the GUI: ACOM tab → tick **GPU** → Build all → run. (If cupy is missing
the toggle raises a clear error instead of crashing.)

## Roll back (remove cupy)

```bat
set PYEXE=C:\Users\danielkh\AppData\Local\anaconda3\envs\py4DSTEM_SAM\python.exe
"%PYEXE%" -m pip uninstall -y cupy-cuda11x fastrlock
```
That fully reverts the change (untick GPU in the ACOM tab).

## Nuclear option (rebuild the env from the snapshot)

If the env somehow gets corrupted, recreate it from scratch:

```bat
set CONDA=C:\Users\danielkh\AppData\Local\anaconda3\Scripts\conda.exe
"%CONDA%" env remove -n py4DSTEM_SAM
"%CONDA%" env create -n py4DSTEM_SAM -f environment_20260607.yml
```
(or, for an exact same-platform rebuild:
`"%CONDA%" create -n py4DSTEM_SAM --file conda_explicit_20260607.txt`)

## Expectations

GPU accelerates only the per-pattern FFT correlation in `match_single_pattern`,
not the Python per-pattern loop in `match_orientations`. Expect a solid speedup
but not orders of magnitude. The bigger levers remain: coarser orientation plan
(larger Δ angles / `corners`/`auto` vs `full`) and skipping empty/vacuum pixels
on full-dataset runs.
