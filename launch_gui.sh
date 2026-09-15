#!/usr/bin/env bash
# ===== DINO-4DSTEM GUI launcher (Linux/macOS) =====
# Portable: works from wherever this folder lives.  Sibling of launch_gui.bat.
DINO_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONIOENCODING=utf-8
# shellcheck disable=SC1091
source "$DINO_DIR/_activate.sh" || { echo "Press Enter to close."; read -r; exit 1; }
python "$DINO_DIR/src/gui_dino4dstem.py"
status=$?
if [ $status -ne 0 ]; then
  echo
  echo "The GUI exited with an error (code $status). Press Enter to close."
  read -r
fi
