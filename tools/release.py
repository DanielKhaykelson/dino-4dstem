"""Cut a release: new version -> git tag -> zips -> GitHub Release -> PyPI.

    python tools/release.py            # 1.0.3 -> 1.0.4
    python tools/release.py minor      # 1.0.3 -> 1.1.0
    python tools/release.py 2.0.0      # explicit version
    python tools/release.py --dry-run  # show what would happen, change nothing

Steps (stops at the first failure):

1. the working tree must have no staged / modified tracked files, and master
   must be pushed;
2. bump ``__version__`` in dino4dstem/__init__.py, commit, tag ``vX.Y.Z``,
   push master and the tag;
3. build the three per-OS zips (tools/build_release_zips.py);
4. create the GitHub Release ``vX.Y.Z`` with the zips attached.  Publishing
   the Release starts .github/workflows/publish.yml, which builds the wheel
   and uploads it to PyPI (trusted publishing, no token in the repo).

The GitHub API is called with the credential git already uses for pushing
(``git credential fill``); it is never printed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT = os.path.join(REPO, "dino4dstem", "__init__.py")
API = "https://api.github.com/repos/DanielKhaykelson/dino-4dstem"
OSES = ("windows", "linux", "macos")


def git(*args: str, capture: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, check=True, text=True,
                       capture_output=capture)
    return r.stdout.strip() if capture else ""


def current_version() -> str:
    m = re.search(r'__version__ = "([^"]+)"', open(INIT, encoding="utf-8").read())
    return m.group(1)


def next_version(cur: str, how: str) -> str:
    if re.fullmatch(r"\d+\.\d+\.\d+", how):
        return how
    major, minor, patch = (int(x) for x in cur.split("."))
    if how == "major":
        return f"{major + 1}.0.0"
    if how == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _headers() -> dict:
    cred = subprocess.run(["git", "credential", "fill"], cwd=REPO, text=True,
                          input="protocol=https\nhost=github.com\n\n",
                          capture_output=True, check=True).stdout
    tok = dict(l.split("=", 1) for l in cred.splitlines() if "=" in l)["password"]
    return {"Authorization": "token " + tok,
            "Accept": "application/vnd.github+json"}


def api(url, method="GET", data=None, extra=None, headers=None):
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    r = urllib.request.Request(url, data=body, method=method,
                               headers={**headers, **(extra or {})})
    with urllib.request.urlopen(r) as f:
        b = f.read()
        return json.loads(b) if b else None


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    args = [a for a in argv if a != "--dry-run"]
    cur = current_version()
    new = next_version(cur, args[0] if args else "patch")
    tag = f"v{new}"
    print(f"version {cur} -> {new}  (tag {tag})")

    dirty = [l for l in git("status", "--porcelain", "-uno").splitlines() if l]
    if dirty:
        print("Commit or stash these first:\n  " + "\n  ".join(dirty))
        return 1
    git("fetch", "-q", "origin")
    if git("rev-list", "--count", "HEAD..origin/master") != "0":
        print("origin/master has commits you do not have - pull first.")
        return 1
    if tag in git("tag").split():
        print(f"tag {tag} already exists")
        return 1
    if dry:
        print("dry run: nothing changed")
        return 0

    src = open(INIT, encoding="utf-8").read()
    open(INIT, "w", encoding="utf-8", newline="").write(
        src.replace(f'__version__ = "{cur}"', f'__version__ = "{new}"'))
    git("commit", "-q", "-m", f"Release {tag}", "--", INIT)
    git("tag", "-a", tag, "-m", f"DINO-4DSTEM {new}")
    git("push", "-q", "origin", "master", tag, capture=False)
    print(f"pushed master + {tag}")

    out = tempfile.mkdtemp(prefix="dino4dstem_rel_")
    subprocess.run([sys.executable, os.path.join(REPO, "tools",
                                                 "build_release_zips.py"), out],
                   cwd=REPO, check=True)

    h = _headers()
    rel = api(API + "/releases", "POST", headers=h, data={
        "tag_name": tag, "name": f"DINO-4DSTEM {new}",
        "body": (f"pip install dino4dstem=={new}\n\n"
                 "Or download the zip for your system below."),
    })
    up = rel["upload_url"].split("{")[0]
    for o in OSES:
        name = f"dino-4dstem-{o}.zip"
        with open(os.path.join(out, name), "rb") as f:
            res = api(f"{up}?name={name}", "POST", f.read(),
                      {"Content-Type": "application/zip"}, headers=h)
        print(f"  {name}: {res['state']} ({res['size']:,} bytes)")
    print(f"GitHub Release {tag}: {rel['html_url']}")
    print("PyPI: the 'publish to PyPI' workflow is now running; "
          f"`pip install -U dino4dstem` gets {new} in a few minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
