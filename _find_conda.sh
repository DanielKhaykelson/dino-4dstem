#!/usr/bin/env bash
# Locates conda and sources its shell hook so `conda activate` works, and sets
# ENV_NAME.  SOURCE this (it is the Linux/macOS sibling of _find_conda.bat) --
# it is a helper used by _activate.sh and install.sh, not run directly.
#
# On success: `conda` is usable in the current shell and ENV_NAME is set.
# On failure: prints guidance and returns non-zero.

# Directory this script lives in (works whether sourced from bash or zsh).
_FC_SRC="${BASH_SOURCE[0]:-${(%):-%N}}"
DINO_DIR="$(cd "$(dirname "$_FC_SRC")" && pwd)"

ENV_NAME="dino4dstem"
if [ -f "$DINO_DIR/env_name.txt" ]; then
  ENV_NAME="$(tr -d '[:space:]' < "$DINO_DIR/env_name.txt")"
fi

# Prefer a conda already on PATH; otherwise search the usual install roots for
# etc/profile.d/conda.sh (covers Anaconda, Miniconda, Miniforge, Mambaforge on
# both Linux and macOS, Intel and Apple Silicon Homebrew locations).
_CONDA_SH=""
if command -v conda >/dev/null 2>&1; then
  _base="$(conda info --base 2>/dev/null)"
  if [ -n "$_base" ] && [ -f "$_base/etc/profile.d/conda.sh" ]; then
    _CONDA_SH="$_base/etc/profile.d/conda.sh"
  fi
fi
if [ -z "$_CONDA_SH" ]; then
  for _base in \
    "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
    "$HOME/opt/anaconda3" "$HOME/opt/miniconda3" "$HOME/opt/miniforge3" \
    "/opt/anaconda3" "/opt/miniconda3" "/opt/miniforge3" \
    "/usr/local/anaconda3" "/usr/local/miniconda3" "/usr/local/Caskroom/miniforge/base" \
    "/opt/homebrew/anaconda3" "/opt/homebrew/Caskroom/miniforge/base"; do
    if [ -f "$_base/etc/profile.d/conda.sh" ]; then
      _CONDA_SH="$_base/etc/profile.d/conda.sh"; break
    fi
  done
fi

if [ -z "$_CONDA_SH" ]; then
  echo "[ERROR] Could not find conda. Install Miniforge (recommended) first:"
  echo "        https://github.com/conda-forge/miniforge#install"
  return 1 2>/dev/null || exit 1
fi

# Quiet conda's "a newer version exists" nag (this shell only).
export CONDA_NOTIFY_OUTDATED_CONDA=false
# shellcheck disable=SC1090
source "$_CONDA_SH"
