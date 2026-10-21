#!/usr/bin/env bash
# DINO-4DSTEM uninstaller (Linux / macOS) -- sibling of uninstall.bat.
# Removes what the installer added OUTSIDE this folder:
#   - the conda environment (default "dino4dstem", or the name in env_name.txt)
#   - the desktop / application-menu launchers
# It does NOT delete this folder or any of your data / results.
set -u
DINO_DIR="$(cd "$(dirname "$0")" && pwd)"

ENV_NAME="dino4dstem"
if [ -f "$DINO_DIR/env_name.txt" ]; then
  ENV_NAME="$(tr -d '[:space:]' < "$DINO_DIR/env_name.txt")"
fi

echo "============================================================"
echo " DINO-4DSTEM uninstall"
echo "============================================================"
echo
echo " This will remove:"
echo "   - the conda environment \"$ENV_NAME\""
echo "   - the DINO-4DSTEM GUI / Assistant / Notebooks launchers"
echo
echo " It will NOT touch this folder, your data, or your results."
echo
if [ "$ENV_NAME" = "base" ]; then
  echo "[ERROR] Refusing to remove the conda \"base\" environment (check env_name.txt)."
  exit 1
fi
printf "Type YES to uninstall, anything else to cancel: "
read -r ANSWER
if [ "${ANSWER:-}" != "YES" ] && [ "${ANSWER:-}" != "yes" ]; then
  echo "Cancelled -- nothing was changed."
  exit 0
fi

echo
echo "Removing launchers..."
for name in "DINO-4DSTEM GUI" "DINO-4DSTEM Assistant" "DINO-4DSTEM Notebooks"; do
  for f in "$HOME/.local/share/applications/${name// /-}.desktop" \
           "$HOME/Desktop/${name// /-}.desktop" \
           "$HOME/Desktop/$name.command"; do
    if [ -e "$f" ]; then rm -f "$f" && echo "  removed: $f"; fi
  done
done

echo
# shellcheck disable=SC1091
if source "$DINO_DIR/_find_conda.sh"; then
  echo "Removing the conda environment \"$ENV_NAME\" (can take a minute)..."
  if conda env remove -n "$ENV_NAME" -y; then
    echo "  removed environment \"$ENV_NAME\"."
  else
    echo "[WARN] conda could not remove \"$ENV_NAME\" -- it may not exist, or a"
    echo "       DINO-4DSTEM / Jupyter window is still open. Close them and re-run."
    exit 1
  fi
else
  echo "[WARN] conda not found -- the environment was not removed."
  exit 1
fi

echo
echo "============================================================"
echo " Uninstall complete."
echo
echo " To remove the program files too, delete this folder yourself:"
echo "   $DINO_DIR"
echo " (Keep anything you want from your data folders first -- results are"
echo "  saved next to your data, in <data>_DINO4DSTEM folders.)"
echo "============================================================"
