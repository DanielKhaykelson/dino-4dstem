"""setup_gpu_torch.py -- make PyTorch in THIS environment use the NVIDIA GPU.

Run by the installers right after the environment is created/updated
(``python tools/setup_gpu_torch.py``); safe to re-run any time.

Why it is needed: requirements.txt installs ``torch`` from PyPI, and on
Windows the PyPI wheels are CPU-only.  The CUDA builds live on PyTorch's own
package index, one per CUDA version, and which one works depends on the
installed NVIDIA DRIVER.  This script:

  1. asks ``nvidia-smi`` which CUDA version the driver supports
     (no NVIDIA GPU / no driver -> keeps the CPU build and says so),
  2. picks the newest PyTorch CUDA build the driver can run
     (cu128 -> cu126 -> cu118),
  3. installs it (same torch / torchvision versions as requirements.txt),
  4. CHECKS that torch actually sees the GPU, falling back to the next older
     build if not, and back to the CPU build if none works.

macOS: nothing to do (no CUDA; the PyPI build already uses Apple's MPS).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

TORCH = "2.7.1"
TORCHVISION = "0.22.1"
# (minimum CUDA version the DRIVER must support, wheel tag) -- newest first
BUILDS = [((12, 8), "cu128"), ((12, 6), "cu126"), ((11, 8), "cu118")]
INDEX = "https://download.pytorch.org/whl/"


def driver_cuda():
    """(major, minor) CUDA version the installed driver supports, or None."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def gpu_names():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=30).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def probe() -> dict:
    """torch state in a FRESH interpreter (the current one may have an old
    torch already imported)."""
    code = ("import json, torch; ok = torch.cuda.is_available(); "
            "print(json.dumps({'v': torch.__version__, "
            "'cuda': torch.version.cuda, 'ok': ok, "
            "'name': torch.cuda.get_device_name(0) if ok else ''}))")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=300)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"v": "?", "cuda": None, "ok": False, "name": "",
                "err": repr(e)}


def pip(*args) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--no-warn-script-location", *args]
    print("   ", " ".join(cmd[4:]), flush=True)
    return subprocess.call(cmd)


def install_build(tag: str) -> bool:
    pkgs = [f"torch=={TORCH}", f"torchvision=={TORCHVISION}"]
    # Swap ONLY torch / torchvision (--no-deps), so the pinned numpy etc. are
    # never touched by PyTorch's index.  Windows wheels bundle the CUDA
    # runtime, so that is all they need.
    if pip("--force-reinstall", "--no-deps", *pkgs,
           "--index-url", INDEX + tag) != 0:
        return False
    if sys.platform.startswith("linux"):
        # Linux CUDA wheels pull the CUDA runtime as separate nvidia-*
        # packages: install whatever is missing, leaving the rest alone.
        if pip(*pkgs, "--index-url", INDEX + tag,
               "--extra-index-url", "https://pypi.org/simple") != 0:
            return False
    return True


def main() -> int:
    print("Checking for an NVIDIA GPU...", flush=True)
    if sys.platform == "darwin":
        print("  macOS: no CUDA -- the standard build (CPU / Apple MPS) is "
              "used.  Nothing to do.")
        return 0
    dc = driver_cuda()
    if dc is None:
        print("  No NVIDIA GPU (or driver) found -- keeping the CPU build.\n"
              "  If this computer HAS an NVIDIA GPU, install/update the NVIDIA "
              "driver and re-run this installer.")
        return 0
    for g in gpu_names():
        print(f"  GPU: {g}")
    print(f"  driver supports CUDA {dc[0]}.{dc[1]}")
    st = probe()
    if st.get("ok"):
        print(f"  torch {st['v']} already uses the GPU ({st['name']}). "
              f"Nothing to do.")
        return 0
    options = [tag for need, tag in BUILDS if dc >= need]
    if not options:
        print(f"  The driver is too old for PyTorch {TORCH} (needs CUDA "
              f">= 11.8).  Update the NVIDIA driver and re-run the "
              f"installer.  Keeping the CPU build for now.")
        return 0
    for tag in options:
        print(f"\nInstalling PyTorch {TORCH} for {tag} (a large download, "
              f"several minutes)...", flush=True)
        if not install_build(tag):
            print(f"  {tag}: install failed, trying the next one.")
            continue
        st = probe()
        if st.get("ok"):
            print(f"\n  OK: torch {st['v']} (CUDA {st['cuda']}) uses the GPU: "
                  f"{st['name']}")
            return 0
        print(f"  {tag} installed but the GPU is not usable "
              f"({st.get('err', 'cuda not available')}); trying the next one.")
    print("\n[WARN] Could not get a working GPU build -- restoring the CPU "
          "build.  The app works, just slower.  Updating the NVIDIA driver "
          "usually fixes this; then re-run the installer.")
    pip("--force-reinstall", "--no-deps", f"torch=={TORCH}",
        f"torchvision=={TORCHVISION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
