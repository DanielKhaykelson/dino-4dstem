"""display_prefs.py — shared, live colour preferences for the GUI.

Two independent, app-wide choices:

  * **class-map palette** — a *categorical* colour set for cluster / class
    maps (tab10, tab20 + variants, ColorBrewer qualitative sets, glasbey,
    or the standard matplotlib cycle).
  * **diffraction colormap** — a *continuous* colormap for diffraction
    patterns / virtual images (gray, viridis, inferno, …).

Panels read the current choice through :func:`class_palette` /
:func:`get_diff_cmap_name`, and may :func:`subscribe` a redraw callback so
they recolour live the moment the user picks a different scheme.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba

# Categorical options offered for CLASS MAPS.  "auto" = tab10 for ≤10
# classes else tab20 (the previous hard-coded behaviour).
CLASS_CMAPS = [
    "auto (tab10/tab20)", "tab10", "tab20", "tab20b", "tab20c",
    "Set1", "Set2", "Set3", "Paired", "Accent", "Dark2",
    "Pastel1", "Pastel2", "glasbey", "standard (C0–C9)",
]

# Continuous options offered for DIFFRACTION patterns / virtual images.
DIFF_CMAPS = [
    "gray", "gray_r", "viridis", "inferno", "magma", "plasma", "cividis",
    "turbo", "hot", "afmhot", "bone", "jet", "nipy_spectral",
]

_state = {"class": "auto (tab10/tab20)", "diff": "inferno"}
_subs: list = []


# ---- getters / setters -------------------------------------------------
def get_class_cmap_name() -> str:
    return _state["class"]


def get_diff_cmap_name() -> str:
    return _state["diff"]


def set_class_cmap_name(name: str) -> None:
    if name and name != _state["class"]:
        _state["class"] = name
        _notify()


def set_diff_cmap_name(name: str) -> None:
    if name and name != _state["diff"]:
        _state["diff"] = name
        _notify()


# ---- live-refresh pub/sub ---------------------------------------------
def subscribe(cb) -> None:
    """Register a no-arg callback fired whenever a colour scheme changes."""
    if cb not in _subs:
        _subs.append(cb)


def unsubscribe(cb) -> None:
    try:
        _subs.remove(cb)
    except ValueError:
        pass


def _notify() -> None:
    for cb in list(_subs):
        try:
            cb()
        except Exception:
            pass


# ---- palette construction ---------------------------------------------
def class_palette(K: int) -> ListedColormap:
    """A ListedColormap of K distinct colours for a class map of K classes,
    using the currently-selected categorical scheme.  Colours cycle if the
    scheme has fewer than K entries."""
    K = max(int(K), 1)
    name = _state["class"]
    if name.startswith("auto"):
        name = "tab20" if K > 10 else "tab10"
    if name.startswith("standard"):
        return ListedColormap([to_rgba(f"C{i % 10}") for i in range(K)])
    if name == "glasbey":
        try:
            import colorcet as cc
            g = cc.glasbey
            return ListedColormap([g[i % len(g)] for i in range(K)])
        except Exception:
            name = "tab20"
    cmap = plt.get_cmap(name)
    n = int(getattr(cmap, "N", 256))
    if n <= 32:                       # qualitative: index + cycle
        return ListedColormap([cmap(i % n) for i in range(K)])
    # continuous fallback: sample evenly
    return ListedColormap([cmap(i / max(K - 1, 1)) for i in range(K)])
