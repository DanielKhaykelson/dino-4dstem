"""Build the three per-OS release bundles.

    python tools/build_release_zips.py [outdir]

Produces, from the files git is tracking at HEAD:

    dino-4dstem-windows.zip   .bat / .ps1 launchers + GETTING_STARTED.docx
    dino-4dstem-linux.zip     .sh launchers (mode 0755)
    dino-4dstem-macos.zip     .sh launchers (mode 0755)

Two details matter and are the reason this lives in the repo rather than in
somebody's scratch folder:

* **Line endings.** Content is read from the git blob, which is LF-normalised.
  Windows scripts are converted back to CRLF here; .sh files must stay LF or
  their shebang breaks on Linux/macOS.  A release built without this shipped
  LF .bat files.
* **The executable bit.** zipfile drops it unless external_attr is set, so the
  .sh launchers would need a chmod after unzipping.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN_ONLY = (".bat", ".ps1")
NIX_ONLY = (".sh",)
# shipped in the bundle but not tracked by git
EXTRAS = ["GETTING_STARTED.docx", "manual - put into gemini.pdf"]


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO)


def wanted(path: str, os_kind: str) -> bool:
    low = path.lower()
    if os_kind == "windows":
        return not low.endswith(NIX_ONLY)
    if low.endswith(WIN_ONLY):
        return False
    # the Windows click-by-click walkthrough is meaningless on Unix
    return os.path.basename(low) != "getting_started.docx"


def add(z: zipfile.ZipFile, arc: str, data: bytes) -> None:
    low = arc.lower()
    if low.endswith(WIN_ONLY):                 # cmd.exe wants CRLF
        data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    elif low.endswith(NIX_ONLY):               # shebang breaks on CRLF
        data = data.replace(b"\r\n", b"\n")
    zi = zipfile.ZipInfo(arc)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = (0o755 if low.endswith(NIX_ONLY) else 0o644) << 16
    z.writestr(zi, data)


def build(os_kind: str, outdir: str) -> str:
    tracked = _git("ls-files").decode().splitlines()
    out = os.path.join(outdir, f"dino-4dstem-{os_kind}.zip")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in tracked:
            if not wanted(p, os_kind):
                continue
            try:
                add(z, p, _git("show", f":{p}"))
                n += 1
            except subprocess.CalledProcessError:
                pass
        for e in EXTRAS:
            full = os.path.join(REPO, e)
            if os.path.exists(full) and wanted(e, os_kind):
                with open(full, "rb") as fh:
                    add(z, e, fh.read())
                n += 1
    print(f"{os.path.basename(out):28s} {n:3d} files  "
          f"{os.path.getsize(out)/1e6:5.1f} MB")
    return out


def verify(path: str) -> None:
    """Fail loudly rather than shipping a bundle that cannot run."""
    z = zipfile.ZipFile(path)
    names = z.namelist()
    problems = []
    for n in names:
        low = n.lower()
        if low.endswith(WIN_ONLY) and b"\r\n" not in z.read(n):
            problems.append(f"{n}: LF line endings (cmd.exe needs CRLF)")
        if low.endswith(NIX_ONLY) and b"\r\n" in z.read(n):
            problems.append(f"{n}: CRLF line endings (breaks the shebang)")
    for need in ("src/gui_dino4dstem.py", "environment.yml",
                 "requirements.txt", "assets/dino.ico"):
        if need not in names:
            problems.append(f"missing {need}")
    if problems:
        print("  FAILED:")
        for p in problems:
            print(f"    - {p}")
        raise SystemExit(1)
    print("  verified: line endings, launchers and required files OK")


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else REPO
    os.makedirs(outdir, exist_ok=True)
    for kind in ("windows", "linux", "macos"):
        verify(build(kind, outdir))
