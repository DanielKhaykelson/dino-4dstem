#!/usr/bin/env bash
# ============================================================
#  DINO-4DSTEM  --  installer for Linux and macOS
#
#  Creates (or updates) the conda environment "dino4dstem" from
#  environment.yml, verifies the app can import, and offers to
#  create a desktop / applications launcher.
#
#  The Windows equivalent is install.bat -- this does not replace
#  it, it is the Unix sibling.  Prerequisite: Miniforge/Miniconda.
#
#  Usage:   ./install.sh          (create/update env, then verify)
# ============================================================
set -u
DINO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo " DINO-4DSTEM install (Linux/macOS)"
echo "============================================================"

# shellcheck disable=SC1091
source "$DINO_DIR/_find_conda.sh" || exit 1
echo "Using conda: $(command -v conda)"
echo "Environment: $ENV_NAME"
echo

# Does the env already exist?  Ask conda rather than guessing a path.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment \"$ENV_NAME\" exists -- updating it..."
  conda env update -n "$ENV_NAME" -f "$DINO_DIR/environment.yml" --prune || {
    echo; echo "[ERROR] Environment update failed. See messages above."; exit 1; }
else
  echo "Creating environment \"$ENV_NAME\" -- this can take several minutes..."
  conda env create -f "$DINO_DIR/environment.yml" || {
    echo; echo "[ERROR] Environment creation failed. See messages above."
    echo "To start clean:  conda env remove -n $ENV_NAME"; exit 1; }
fi

# On Linux with an NVIDIA GPU you can swap in the CUDA build of torch for
# large speed-ups.  requirements.txt installs the CPU build by default; this
# leaves it untouched unless the user opts in.
if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "An NVIDIA GPU was detected.  For GPU acceleration you can install the"
  echo "CUDA build of PyTorch into the env, e.g.:"
  echo "    conda run -n $ENV_NAME pip install torch==2.7.1 torchvision==0.22.1 \\"
  echo "        --index-url https://download.pytorch.org/whl/cu118"
  echo "(Optional -- the app runs on CPU without it.)"
fi
if [ "$(uname -s)" = "Darwin" ]; then
  echo
  echo "Note: macOS has no CUDA; DINO-4DSTEM runs on CPU here.  Training the"
  echo "denoiser is slower than on an NVIDIA GPU but fully functional."
fi

echo
echo "Verifying the app can import in the environment..."
conda run -n "$ENV_NAME" python -c "import sys, numpy, torch, customtkinter, py4DSTEM; \
print('  python      ', sys.version.split()[0]); \
print('  numpy       ', numpy.__version__); \
print('  torch       ', torch.__version__, '(cuda', torch.cuda.is_available(), ')'); \
print('  py4DSTEM    ', py4DSTEM.__version__); \
print('  customtkinter', customtkinter.__version__)" || {
  echo; echo "[ERROR] The environment is missing packages -- re-run ./install.sh"; exit 1; }

echo
echo "============================================================"
echo " DINO-4DSTEM is installed."
echo
echo "  Launch it:   ./launch_gui.sh"
echo "  Assistant:   ./launch_assistant.sh"
echo "  Notebooks:   ./launch_notebooks.sh"
echo
echo "  Optional desktop/app launcher:  ./make_desktop_shortcuts.sh"
echo "============================================================"

# Make sure the launchers are executable (git may not carry the bit).
chmod +x "$DINO_DIR"/*.sh 2>/dev/null || true
