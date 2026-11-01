"""Desktop icons for a pip-installed DINO-4DSTEM.

pip never runs code at install time, so icons are made by one extra command,
``dino4dstem shortcuts``.  Every icon runs ``<this python> -m dino4dstem ...``,
so it always starts the environment the package was installed into; no conda
lookup or hard-coded paths are needed.

    Windows : .lnk files on the Desktop (follows OneDrive Desktop redirection)
    Linux   : .desktop files in the applications menu and on ~/Desktop
    macOS   : double-clickable .command files on ~/Desktop
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dino4dstem import paths

# (shortcut name, dino4dstem sub-command)
APPS = [
    ("DINO-4DSTEM GUI", "gui"),
    ("DINO-4DSTEM Assistant", "assistant"),
    ("DINO-4DSTEM Notebooks", "notebooks"),
]


def _python() -> str:
    return sys.executable


def _slug(name: str) -> str:
    return name.replace(" ", "-")


# ---------------------------------------------------------------- Windows
def _ps_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _windows(remove: bool) -> list[Path]:
    ico = paths.icon("ico")
    ws = paths.workspace()
    lines = ['$desk = [Environment]::GetFolderPath("Desktop")',
             '$ws = New-Object -ComObject WScript.Shell']
    for name, cmd in APPS:
        lnk = f'(Join-Path $desk {_ps_quote(name + ".lnk")})'
        if remove:
            lines.append(f'$p = {lnk}; if (Test-Path $p) {{ Remove-Item $p; Write-Output $p }}')
            continue
        lines += [
            f'$p = {lnk}',
            '$s = $ws.CreateShortcut($p)',
            f'$s.TargetPath = {_ps_quote(_python())}',
            f'$s.Arguments = {_ps_quote("-m dino4dstem " + cmd)}',
            f'$s.WorkingDirectory = {_ps_quote(ws)}',
            f'$s.Description = {_ps_quote(name)}',
        ]
        if ico:
            lines.append(f'$s.IconLocation = {_ps_quote(str(ico) + ",0")}')
        lines += ['$s.Save()', 'Write-Output $p']
    out = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-Command", "\n".join(lines)],
        capture_output=True, text=True, check=True)
    return [Path(l.strip()) for l in out.stdout.splitlines() if l.strip()]


# ---------------------------------------------------------------- Linux
def _desktop_dir() -> Path:
    return Path.home() / "Desktop"


def _linux(remove: bool) -> list[Path]:
    apps = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "applications"
    png = paths.icon("png")
    done: list[Path] = []
    for name, cmd in APPS:
        targets = [apps / f"{_slug(name)}.desktop"]
        if _desktop_dir().is_dir():
            targets.append(_desktop_dir() / f"{_slug(name)}.desktop")
        if remove:
            for t in targets:
                if t.exists():
                    t.unlink()
                    done.append(t)
            continue
        body = "\n".join(filter(None, [
            "[Desktop Entry]",
            "Type=Application",
            f"Name={name}",
            "Comment=DINO-4DSTEM",
            f'Exec="{_python()}" -m dino4dstem {cmd}',
            f"Path={paths.workspace()}",
            f"Icon={png}" if png else "",
            "Terminal=true",
            "Categories=Science;Education;",
        ])) + "\n"
        apps.mkdir(parents=True, exist_ok=True)
        for t in targets:
            t.write_text(body, encoding="utf-8")
            t.chmod(0o755)
            if t.parent == _desktop_dir() and _has("gio"):  # GNOME trust, best effort
                subprocess.run(["gio", "set", str(t), "metadata::trusted", "true"],
                               capture_output=True, check=False)
            done.append(t)
    return done


def _has(prog: str) -> bool:
    from shutil import which
    return which(prog) is not None


# ---------------------------------------------------------------- macOS
def _macos(remove: bool) -> list[Path]:
    desk = _desktop_dir()
    done: list[Path] = []
    for name, cmd in APPS:
        f = desk / f"{name}.command"
        if remove:
            if f.exists():
                f.unlink()
                done.append(f)
            continue
        desk.mkdir(parents=True, exist_ok=True)
        f.write_text(
            "#!/usr/bin/env bash\n"
            f'cd "{paths.workspace()}"\n'
            f'exec "{_python()}" -m dino4dstem {cmd}\n', encoding="utf-8")
        f.chmod(0o755)
        done.append(f)
    return done


def _dispatch(remove: bool) -> list[Path]:
    if sys.platform.startswith("win"):
        return _windows(remove)
    if sys.platform == "darwin":
        return _macos(remove)
    return _linux(remove)


def create() -> list[Path]:
    """Create the three Desktop icons; returns the files written."""
    return _dispatch(remove=False)


def remove() -> list[Path]:
    """Delete the icons made by :func:`create`; returns the files removed."""
    return _dispatch(remove=True)
