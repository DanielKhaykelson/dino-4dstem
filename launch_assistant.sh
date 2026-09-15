#!/usr/bin/env bash
# ===== DINO-4DSTEM Assistant launcher (Linux/macOS) =====
# Opens the standalone assistant window.  Sibling of launch_assistant.bat.
# Optional: pass a cube path as the first argument to load it on start.
DINO_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONIOENCODING=utf-8
# shellcheck disable=SC1091
source "$DINO_DIR/_activate.sh" || { echo "Press Enter to close."; read -r; exit 1; }
python "$DINO_DIR/src/assistant_gui.py" "$@"
