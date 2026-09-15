#!/usr/bin/env bash
# Creates desktop launchers for DINO-4DSTEM on Linux and macOS.
# Sibling of make_desktop_shortcuts.ps1 (Windows) -- does not replace it.
#
#   Linux : writes .desktop files to ~/.local/share/applications and ~/Desktop
#   macOS : writes double-clickable .command files to ~/Desktop
#
# Run:  ./make_desktop_shortcuts.sh
DINO_DIR="$(cd "$(dirname "$0")" && pwd)"
OS="$(uname -s)"
ICON="$DINO_DIR/assets/dino.png"      # used on Linux if present

make_linux () {
  local name="$1" script="$2"
  local apps="$HOME/.local/share/applications"
  mkdir -p "$apps"
  local f="$apps/${name// /-}.desktop"
  {
    echo "[Desktop Entry]"
    echo "Type=Application"
    echo "Name=$name"
    echo "Comment=DINO-4DSTEM"
    echo "Exec=bash \"$DINO_DIR/$script\""
    echo "Path=$DINO_DIR"
    [ -f "$ICON" ] && echo "Icon=$ICON"
    echo "Terminal=true"
    echo "Categories=Science;Education;"
  } > "$f"
  chmod +x "$f"
  # also drop a copy on the Desktop if there is one
  if [ -d "$HOME/Desktop" ]; then
    cp "$f" "$HOME/Desktop/${name// /-}.desktop"
    chmod +x "$HOME/Desktop/${name// /-}.desktop"
    # mark trusted on GNOME (best-effort; harmless elsewhere)
    gio set "$HOME/Desktop/${name// /-}.desktop" metadata::trusted true 2>/dev/null || true
  fi
  echo "Created: $f"
}

make_macos () {
  local name="$1" script="$2"
  local desk="$HOME/Desktop"
  mkdir -p "$desk"
  local f="$desk/$name.command"
  {
    echo "#!/usr/bin/env bash"
    echo "exec bash \"$DINO_DIR/$script\""
  } > "$f"
  chmod +x "$f"
  echo "Created: $f"
}

chmod +x "$DINO_DIR"/*.sh 2>/dev/null || true

if [ "$OS" = "Darwin" ]; then
  make_macos "DINO-4DSTEM GUI"        "launch_gui.sh"
  make_macos "DINO-4DSTEM Assistant"  "launch_assistant.sh"
  make_macos "DINO-4DSTEM Notebooks"  "launch_notebooks.sh"
  echo
  echo "Done. Three .command files are on your Desktop -- double-click to run."
  echo "(First launch: macOS may ask you to allow it -- right-click > Open once.)"
elif [ "$OS" = "Linux" ]; then
  make_linux "DINO-4DSTEM GUI"        "launch_gui.sh"
  make_linux "DINO-4DSTEM Assistant"  "launch_assistant.sh"
  make_linux "DINO-4DSTEM Notebooks"  "launch_notebooks.sh"
  echo
  echo "Done. Launchers are in your applications menu and on your Desktop."
else
  echo "Unsupported OS: $OS  (this script is for Linux/macOS;"
  echo "on Windows use make_desktop_shortcuts.ps1)."
  exit 1
fi
