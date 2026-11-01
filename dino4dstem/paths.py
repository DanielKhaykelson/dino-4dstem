"""Where the app code lives and where it keeps its working files.

Two layouts are supported:

* **pip install** (a wheel): ``src/``, ``assets/``, ``notebooks/``, ``docs/``
  and ``tools/`` are copied *inside* this package directory.
* **git clone / editable install**: they sit next to this package, at the
  repository root, exactly as the .bat/.sh launchers expect.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def app_root() -> Path:
    """Folder that contains ``src/`` and ``assets/``."""
    for cand in (_HERE, _HERE.parent):
        if (cand / "src" / "gui_dino4dstem.py").is_file():
            return cand
    raise FileNotFoundError(
        "DINO-4DSTEM app files not found next to the dino4dstem package. "
        "Reinstall with: pip install --force-reinstall dino4dstem")


def is_source_checkout() -> bool:
    """True when running from a git clone (or editable install)."""
    return app_root() == _HERE.parent


def workspace() -> Path:
    """Writable folder used as the working directory of the app.

    The app keeps its state in ``runs/`` under the working directory and the
    assistant indexes the ``docs/`` found there.  In a git clone that is the
    repository itself (unchanged behaviour).  For a pip install it is
    ``~/DINO4DSTEM`` (override with the ``DINO4DSTEM_HOME`` variable), seeded
    once with the bundled docs so the assistant can answer from them.
    """
    env = os.environ.get("DINO4DSTEM_HOME")
    if env:
        ws = Path(env).expanduser()
    elif is_source_checkout():
        return app_root()
    else:
        ws = Path.home() / "DINO4DSTEM"
    ws.mkdir(parents=True, exist_ok=True)
    docs_src = app_root() / "docs"
    docs_dst = ws / "docs"
    if docs_src.is_dir() and not docs_dst.exists():
        shutil.copytree(docs_src, docs_dst)
    return ws


def notebooks_dir() -> Path:
    """Tutorial notebooks, copied to a writable place on first use."""
    src = app_root() / "notebooks"
    if is_source_checkout() and not os.environ.get("DINO4DSTEM_HOME"):
        return src
    dst = workspace() / "notebooks"
    if not dst.exists():
        shutil.copytree(src, dst)
    return dst


def icon(kind: str = "png") -> Path | None:
    """App icon: ``ico`` (Windows) or ``png`` (Linux)."""
    name = {"ico": "dino.ico", "png": "dino.png"}[kind]
    p = app_root() / "assets" / name
    return p if p.is_file() else None
