#!/usr/bin/env bash
# ===== DINO-4DSTEM tutorial-notebooks launcher (Linux/macOS) =====
# Opens Jupyter Notebook in the "notebooks" folder using the DINO-4DSTEM env.
# Sibling of launch_notebooks.bat.
DINO_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONIOENCODING=utf-8
# shellcheck disable=SC1091
source "$DINO_DIR/_activate.sh" || { echo "Press Enter to close."; read -r; exit 1; }

# First run only: make sure Jupyter + the interactive-plot widgets are present.
if ! python -c "import notebook" >/dev/null 2>&1; then
  echo
  echo "Installing Jupyter into the environment (one time, ~1 min)..."
  python -m pip install --quiet notebook ipywidgets ipympl || {
    echo "[ERROR] Could not install Jupyter. Check your internet connection."
    echo "Press Enter to close."; read -r; exit 1; }
fi

cd "$DINO_DIR/notebooks" || exit 1
echo
echo "Launching Jupyter Notebook in: $PWD"
echo "(A browser tab will open. Press Ctrl+C here to stop Jupyter.)"
echo
jupyter notebook
