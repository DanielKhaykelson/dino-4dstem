#!/usr/bin/env bash
# Activates the DINO-4DSTEM conda environment in the current shell.
# SOURCE this (Linux/macOS sibling of _activate.bat): `source _activate.sh`.
# Helper used by the launchers -- not run directly.

_ACT_SRC="${BASH_SOURCE[0]:-${(%):-%N}}"
DINO_DIR="$(cd "$(dirname "$_ACT_SRC")" && pwd)"

# Find conda + set ENV_NAME (returns non-zero and prints guidance if missing).
# shellcheck disable=SC1091
source "$DINO_DIR/_find_conda.sh" || return 1 2>/dev/null || exit 1

conda activate "$ENV_NAME" || {
  echo "[ERROR] Could not activate conda env \"$ENV_NAME\"."
  echo "        Run  ./install.sh  first, or put a different env name in"
  echo "        env_name.txt next to this file."
  return 1 2>/dev/null || exit 1
}
