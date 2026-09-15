"""_platform.py -- small cross-platform helpers.

Keeps the Windows behaviour byte-for-byte (os.startfile) and adds the Linux
and macOS equivalents, so the app opens a results folder in the native file
manager on all three platforms.
"""
from __future__ import annotations

import os
import subprocess
import sys


def open_path(path: str) -> bool:
    """Open a file or folder in the OS file manager / default handler.

    Windows: os.startfile · macOS: `open` · Linux: `xdg-open`.
    Returns True on success, False if it could not be opened (caller may then
    fall back to just showing the path).
    """
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]  # Windows only
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception:
        return False
