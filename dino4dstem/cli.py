"""The ``dino4dstem`` command.

    dino4dstem                 open the desktop app (same as ``dino4dstem gui``)
    dino4dstem assistant [cube]  open the plain-English assistant
    dino4dstem notebooks       open the tutorial notebooks in Jupyter
    dino4dstem shortcuts       put GUI / Assistant / Notebooks icons on the Desktop
    dino4dstem setup-gpu       switch PyTorch to the CUDA build for your NVIDIA GPU
"""
from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys

from dino4dstem import __version__, paths


def _run_app_script(name: str, args: list[str]) -> int:
    """Run one of the app's entry scripts in this interpreter, from the workspace."""
    root = paths.app_root()
    script = root / "src" / name
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.chdir(paths.workspace())
    sys.path.insert(0, str(root / "src"))
    sys.argv = [str(script), *args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:  # the scripts call sys.exit themselves
        return e.code if isinstance(e.code, int) else 0
    return 0


def cmd_gui(a: argparse.Namespace) -> int:
    return _run_app_script("gui_dino4dstem.py", [])


def cmd_assistant(a: argparse.Namespace) -> int:
    return _run_app_script("assistant_gui.py", [a.cube] if a.cube else [])


def cmd_notebooks(a: argparse.Namespace) -> int:
    nb = paths.notebooks_dir()
    print(f"Launching Jupyter Notebook in: {nb}")
    print("(A browser tab will open. Press Ctrl+C here to stop Jupyter.)")
    return subprocess.call([sys.executable, "-m", "notebook"], cwd=nb)


def cmd_shortcuts(a: argparse.Namespace) -> int:
    from dino4dstem import shortcuts
    if a.remove:
        for p in shortcuts.remove():
            print(f"Removed: {p}")
        return 0
    for p in shortcuts.create():
        print(f"Created: {p}")
    print("\nDone. Double-click an icon on your Desktop to start.")
    if sys.platform == "darwin":
        print("(First launch: macOS may ask you to allow it; right-click > Open once.)")
    return 0


def cmd_setup_gpu(a: argparse.Namespace) -> int:
    script = paths.app_root() / "tools" / "setup_gpu_torch.py"
    return subprocess.call([sys.executable, str(script)])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dino4dstem",
        description="DINO-4DSTEM: self-supervised 4D-STEM classification. "
                    "Run without a command to open the desktop app.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    s = sub.add_parser("gui", help="open the desktop app (default)")
    s.set_defaults(func=cmd_gui)

    s = sub.add_parser("assistant", help="open the plain-English assistant")
    s.add_argument("cube", nargs="?", help="optional data file to load on start")
    s.set_defaults(func=cmd_assistant)

    s = sub.add_parser("notebooks", help="open the tutorial notebooks in Jupyter")
    s.set_defaults(func=cmd_notebooks)

    s = sub.add_parser("shortcuts", help="create Desktop icons for the app")
    s.add_argument("--remove", action="store_true", help="delete the icons instead")
    s.set_defaults(func=cmd_shortcuts)

    s = sub.add_parser("setup-gpu",
                       help="install the CUDA build of PyTorch for an NVIDIA GPU")
    s.set_defaults(func=cmd_setup_gpu)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    func = getattr(a, "func", cmd_gui)
    return func(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
