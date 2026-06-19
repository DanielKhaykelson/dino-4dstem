"""pair_labels.py -- pair-label sidecar I/O for semi-supervised
training and fine-tuning.

Each cube (.prz / .cube.npy) gets a sister JSON file:

    <basename>.pair_labels.json

with the schema:

    {
      "sample_path": "...",
      "scan_shape":  [Ny, Nx],
      "K_at_label":  6,                      # K when these labels were
                                              #   committed; semantic ≠
                                              #   prototype identity
      "created":     "2026-05-01T09:12:34",
      "updated":     "2026-05-01T09:34:18",
      "pairs": [
         {"a": 1234, "b": 5678, "y":  +1, "source": "random",
          "t": "..."},
         {"a": 8765, "b": 4321, "y":  -1, "source": "cross_class",
          "t": "..."},
         ...
      ]
    }

`y = +1`  →  same physical phase  (pull together)
`y = -1`  →  different phase       (push apart)

`source` records HOW the pair was selected so we can later analyse
whether random vs active-sampled labels gave more signal:
    "random"       — uniform-random pre-train labelling
    "cross_class"  — pair of high-confidence reps from two different
                     prototypes (active sampling, fine-tune)
    "scan_edge"    — neighbouring scan pixels assigned to different
                     prototypes (active sampling, fine-tune)
    "boundary"     — low-max-softmax sample paired with each top-K rep
    "manual"       — picked by the user from a click on the class map

Indices are flat scan indices: `idx = ry * Nx + rx`.
"""
from __future__ import annotations
import os
import json
from datetime import datetime

LABEL_SAME = +1
LABEL_DIFF = -1


def label_path_for_cube(cube_path: str) -> str:
    """Canonical sidecar path for a cube: <basename>.pair_labels.json.
    Strips .prz / .npz / .cube.npy / .npy from the input path.

    Idempotent: if `cube_path` already ends in `.pair_labels.json`
    (i.e. the caller already has the sidecar path itself), it is
    returned unchanged. This lets every other helper in this module
    accept either a cube path or a labels path interchangeably."""
    base = cube_path
    low = base.lower()
    if low.endswith(".pair_labels.json"):
        return cube_path
    if low.endswith(".cube.npy"):
        base = base[: -len(".cube.npy")]
    elif low.endswith((".prz", ".npz", ".npy")):
        base = os.path.splitext(base)[0]
    return base + ".pair_labels.json"


def load_pair_labels(cube_path: str) -> dict:
    """Load the pair-labels sidecar. Returns a fresh empty dict if the
    file doesn't exist (so callers can always treat the result as
    'the labels for this cube')."""
    p = label_path_for_cube(cube_path)
    if not os.path.exists(p):
        return {
            "sample_path": cube_path,
            "scan_shape":  None,
            "K_at_label":  None,
            "pairs":       [],
            "created":     datetime.now().isoformat(timespec="seconds"),
        }
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        # corrupt sidecar — return empty rather than crashing the GUI
        return {
            "sample_path": cube_path,
            "scan_shape":  None,
            "K_at_label":  None,
            "pairs":       [],
            "created":     datetime.now().isoformat(timespec="seconds"),
            "_load_error": True,
        }
    d.setdefault("pairs", [])
    d.setdefault("sample_path", cube_path)
    return d


def save_pair_labels(cube_path: str, labels: dict) -> str:
    """Write `labels` to the canonical sidecar path. Returns the path
    written."""
    p = label_path_for_cube(cube_path)
    payload = dict(labels)
    payload["sample_path"] = cube_path
    payload["updated"] = datetime.now().isoformat(timespec="seconds")
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    return p


def append_pair(labels: dict, idx_a: int, idx_b: int, y: int,
                  source: str = "random") -> None:
    """Append a single labelled pair. Mutates `labels` in place.
    `y` must be +1 (LABEL_SAME) or -1 (LABEL_DIFF)."""
    if int(y) not in (LABEL_SAME, LABEL_DIFF):
        raise ValueError(
            f"y must be +1 (same) or -1 (different), got {y!r}")
    labels.setdefault("pairs", []).append({
        "a": int(idx_a),
        "b": int(idx_b),
        "y": int(y),
        "source": str(source),
        "t": datetime.now().isoformat(timespec="seconds"),
    })


def label_count(labels: dict) -> dict:
    """Return {'same': n_same, 'diff': n_diff, 'total': N,
                'by_source': {source: count, ...}}."""
    pairs = labels.get("pairs", [])
    same = sum(1 for p in pairs if int(p.get("y", 0)) == LABEL_SAME)
    diff = sum(1 for p in pairs if int(p.get("y", 0)) == LABEL_DIFF)
    by_source: dict = {}
    for p in pairs:
        s = str(p.get("source", "?"))
        by_source[s] = by_source.get(s, 0) + 1
    return {
        "same":      same,
        "diff":      diff,
        "total":     len(pairs),
        "by_source": by_source,
    }


def labels_to_arrays(labels: dict):
    """Return (idx_a, idx_b, y) as three numpy int arrays for
    consumption by the training loop. Returns None if no pairs."""
    import numpy as np
    pairs = labels.get("pairs", [])
    if not pairs:
        return None
    a = np.fromiter((int(p["a"]) for p in pairs), dtype=np.int64,
                     count=len(pairs))
    b = np.fromiter((int(p["b"]) for p in pairs), dtype=np.int64,
                     count=len(pairs))
    y = np.fromiter((int(p["y"]) for p in pairs), dtype=np.int8,
                     count=len(pairs))
    return a, b, y


def sample_random_pair(N: int, rng) -> "tuple[int, int]":
    """Draw a uniform random unordered pair (a, b) with a != b. `rng`
    is a `numpy.random.Generator`."""
    a = int(rng.integers(0, N))
    b = int(rng.integers(0, N))
    while b == a:
        b = int(rng.integers(0, N))
    return a, b


def pop_last_pair(labels: dict) -> "dict | None":
    """Pop and return the most recently appended pair entry (or None
    if the list is empty). Mutates `labels` in place."""
    pairs = labels.setdefault("pairs", [])
    if not pairs:
        return None
    return pairs.pop()


def set_pair_label_at(labels: dict, list_idx: int, y: int) -> bool:
    """Change the label `y` of the pair at position `list_idx`.
    Returns True on success, False if the index is out of range or
    `y` isn't ±1."""
    pairs = labels.get("pairs", [])
    if not (0 <= int(list_idx) < len(pairs)):
        return False
    if int(y) not in (LABEL_SAME, LABEL_DIFF):
        return False
    pairs[int(list_idx)]["y"] = int(y)
    pairs[int(list_idx)]["t"] = datetime.now().isoformat(
        timespec="seconds")
    return True


def delete_pair_at(labels: dict, list_idx: int) -> "dict | None":
    """Remove and return the pair at position `list_idx` (None on
    out-of-range)."""
    pairs = labels.get("pairs", [])
    if not (0 <= int(list_idx) < len(pairs)):
        return None
    return pairs.pop(int(list_idx))


def already_labelled(labels: dict, idx_a: int, idx_b: int) -> bool:
    """Return True iff (idx_a, idx_b) (in either order) is already in
    the labels list. Used to avoid re-asking the user about pairs they
    already answered."""
    a, b = int(idx_a), int(idx_b)
    for p in labels.get("pairs", []):
        pa, pb = int(p["a"]), int(p["b"])
        if (pa == a and pb == b) or (pa == b and pb == a):
            return True
    return False


# ---------------------------------------------------------------------------
# Importers (load externally-provided labels into the cube's sidecar)
# ---------------------------------------------------------------------------

def _flat_idx_for(rx, ry, scan_shape):
    Nx = int(scan_shape[1])
    return int(rx) * Nx + int(ry)


def import_from_json(cube_path: str, src_json_path: str,
                       *, scan_shape=None,
                       skip_duplicates: bool = True) -> dict:
    """Merge labels from another `pair_labels.json` into this cube's
    sidecar.  Returns a stats dict {added, skipped_dup, skipped_oor,
    n_after}.  Out-of-range indices (against `scan_shape`) are
    skipped."""
    import json as _json
    with open(src_json_path, "r", encoding="utf-8") as f:
        src = _json.load(f)
    src_pairs = src.get("pairs", [])
    cur = load_pair_labels(cube_path)
    cur.setdefault("pairs", [])
    if scan_shape is None:
        scan_shape = cur.get("scan_shape") or src.get("scan_shape")
    N = (int(scan_shape[0]) * int(scan_shape[1])) if scan_shape else None
    stats = {"added": 0, "skipped_dup": 0, "skipped_oor": 0,
             "skipped_bad_y": 0}
    for p in src_pairs:
        try:
            a = int(p["a"]); b = int(p["b"]); y = int(p["y"])
        except (KeyError, TypeError, ValueError):
            stats["skipped_bad_y"] += 1; continue
        if y not in (LABEL_SAME, LABEL_DIFF):
            stats["skipped_bad_y"] += 1; continue
        if N is not None and (a < 0 or b < 0 or a >= N or b >= N):
            stats["skipped_oor"] += 1; continue
        if skip_duplicates and already_labelled(cur, a, b):
            stats["skipped_dup"] += 1; continue
        append_pair(cur, a, b, y,
                     source=str(p.get("source", "imported_json")))
        stats["added"] += 1
    save_pair_labels(cube_path, cur)
    stats["n_after"] = len(cur.get("pairs", []))
    return stats


def import_from_csv(cube_path: str, src_csv_path: str,
                      *, scan_shape=None,
                      skip_duplicates: bool = True) -> dict:
    """Merge labels from a CSV.

    Header detection (case-insensitive):
       (a, b, y)                     → flat indices
       (rx_a, ry_a, rx_b, ry_b, y)   → scan coords
    `y` may be +1/-1, 1/0 (1=same, 0=diff), or 'same'/'diff'.
    Lines with unparsable y are silently skipped (counted in stats).
    """
    import csv as _csv
    cur = load_pair_labels(cube_path)
    cur.setdefault("pairs", [])
    if scan_shape is None:
        scan_shape = cur.get("scan_shape")
    N = (int(scan_shape[0]) * int(scan_shape[1])) if scan_shape else None

    def _parse_y(v):
        if isinstance(v, (int, float)):
            iv = int(v)
            if iv in (1, -1): return iv
            if iv == 0: return LABEL_DIFF
            return None
        s = str(v).strip().lower()
        if s in ("1", "+1", "same", "true", "yes", "y"): return LABEL_SAME
        if s in ("-1", "diff", "different", "no", "n"):  return LABEL_DIFF
        if s in ("0", "false"): return LABEL_DIFF
        return None

    stats = {"added": 0, "skipped_dup": 0, "skipped_oor": 0,
             "skipped_bad_y": 0, "rows_seen": 0}
    with open(src_csv_path, "r", encoding="utf-8", newline="") as f:
        rdr = _csv.DictReader(f)
        cols_lower = {c.lower(): c for c in (rdr.fieldnames or [])}
        has_flat = ("a" in cols_lower and "b" in cols_lower
                      and "y" in cols_lower)
        has_xy = (all(k in cols_lower for k in
                       ("rx_a", "ry_a", "rx_b", "ry_b", "y")))
        if not (has_flat or has_xy):
            raise ValueError(
                "CSV header must be either 'a,b,y' or "
                "'rx_a,ry_a,rx_b,ry_b,y' (case-insensitive). Got: "
                f"{rdr.fieldnames}")
        for r in rdr:
            stats["rows_seen"] += 1
            y = _parse_y(r.get(cols_lower["y"]))
            if y is None:
                stats["skipped_bad_y"] += 1; continue
            try:
                if has_flat:
                    a = int(r[cols_lower["a"]])
                    b = int(r[cols_lower["b"]])
                else:
                    if scan_shape is None:
                        # need scan_shape for (rx,ry) → flat conversion
                        raise ValueError(
                            "CSV uses scan-coords format but no "
                            "scan_shape is known for this cube; "
                            "load the cube first.")
                    a = _flat_idx_for(r[cols_lower["rx_a"]],
                                        r[cols_lower["ry_a"]],
                                        scan_shape)
                    b = _flat_idx_for(r[cols_lower["rx_b"]],
                                        r[cols_lower["ry_b"]],
                                        scan_shape)
            except (ValueError, TypeError):
                stats["skipped_bad_y"] += 1; continue
            if N is not None and (a < 0 or b < 0 or a >= N or b >= N):
                stats["skipped_oor"] += 1; continue
            if skip_duplicates and already_labelled(cur, a, b):
                stats["skipped_dup"] += 1; continue
            append_pair(cur, a, b, y, source="imported_csv")
            stats["added"] += 1
    save_pair_labels(cube_path, cur)
    stats["n_after"] = len(cur.get("pairs", []))
    return stats


def import_from_classmap(cube_path: str, src_path: str,
                            *, scan_shape=None,
                            max_pairs: int = 500,
                            same_frac: float = 0.5,
                            ignore_value=-1,
                            seed: int = 42,
                            skip_duplicates: bool = True) -> dict:
    """Derive pair labels from a per-pixel class-id map (`.npy`, `.png`,
    or `.json`).  For every pair (a, b):
       y = +1   iff class[a] == class[b]
       y = -1   iff class[a] != class[b]
    The full pair set is enormous (~N²/2), so we sample `max_pairs`
    total with `same_frac` of them being same-class.

    `.npy`  → 2D int array (Ny, Nx).  ignore_value is dropped.
    `.png`  → grayscale; pixel value = class id.
    `.json` → dict { "(rx,ry)" : class_id, ... } OR list of triples
              [[rx, ry, class_id], ...].
    """
    import os as _os
    import numpy as _np
    if scan_shape is None:
        cur = load_pair_labels(cube_path)
        scan_shape = cur.get("scan_shape")
    if scan_shape is None:
        raise ValueError(
            "scan_shape unknown for this cube; load it on Tab 1 first.")
    Ny, Nx = int(scan_shape[0]), int(scan_shape[1])

    ext = _os.path.splitext(src_path)[1].lower()
    if ext == ".npy":
        cmap = _np.asarray(_np.load(src_path, allow_pickle=True))
        if cmap.shape != (Ny, Nx):
            raise ValueError(
                f"class-map shape {cmap.shape} does not match "
                f"scan_shape ({Ny}, {Nx})")
    elif ext in (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"):
        try:
            from PIL import Image
        except Exception as _e:
            raise RuntimeError(
                f"PIL needed to read image class-maps: {_e}") from _e
        img = Image.open(src_path).convert("I")  # 32-bit int
        cmap = _np.asarray(img)
        if cmap.shape != (Ny, Nx):
            raise ValueError(
                f"class-map image shape {cmap.shape} != "
                f"scan_shape ({Ny}, {Nx})")
    elif ext == ".json":
        import json as _json
        with open(src_path, "r", encoding="utf-8") as f:
            obj = _json.load(f)
        cmap = _np.full((Ny, Nx), ignore_value, dtype=_np.int64)
        if isinstance(obj, dict):
            for k, v in obj.items():
                kk = k.strip("()[] ")
                rx, ry = (int(x) for x in kk.split(",")[:2])
                if 0 <= rx < Ny and 0 <= ry < Nx:
                    cmap[rx, ry] = int(v)
        elif isinstance(obj, list):
            for triple in obj:
                rx, ry, c = int(triple[0]), int(triple[1]), int(triple[2])
                if 0 <= rx < Ny and 0 <= ry < Nx:
                    cmap[rx, ry] = c
        else:
            raise ValueError("class-map .json must be dict or list")
    else:
        raise ValueError(f"unsupported class-map extension: {ext}")

    cmap = cmap.astype(_np.int64)
    valid = (cmap != int(ignore_value))
    rng = _np.random.default_rng(int(seed))
    valid_idx = _np.flatnonzero(valid.ravel())
    if valid_idx.size < 2:
        raise ValueError(
            f"need at least 2 labelled pixels in the class-map; got "
            f"{valid_idx.size} (after dropping ignore={ignore_value}).")

    cur = load_pair_labels(cube_path)
    cur.setdefault("pairs", [])
    flat = cmap.ravel()

    n_same_target = int(round(max_pairs * same_frac))
    n_diff_target = max_pairs - n_same_target

    stats = {"added": 0, "skipped_dup": 0, "skipped_oor": 0,
             "skipped_bad_y": 0, "n_same_added": 0, "n_diff_added": 0,
             "valid_pixels": int(valid_idx.size),
             "n_classes": int(_np.unique(flat[valid_idx]).size)}

    # Sample up to ~10 × target with rejection (need balanced y, no dups,
    # no self-pairs).  Cap attempts so we don't loop forever on a
    # degenerate map.
    max_attempts_same = max_pairs * 20
    max_attempts_diff = max_pairs * 20

    n_same = 0; n_diff = 0; tries = 0
    while (n_same < n_same_target or n_diff < n_diff_target) \
            and tries < max_attempts_same + max_attempts_diff:
        tries += 1
        a = int(valid_idx[rng.integers(0, valid_idx.size)])
        b = int(valid_idx[rng.integers(0, valid_idx.size)])
        if a == b: continue
        ca, cb = int(flat[a]), int(flat[b])
        if ca == cb:
            if n_same >= n_same_target: continue
            y = LABEL_SAME
        else:
            if n_diff >= n_diff_target: continue
            y = LABEL_DIFF
        if skip_duplicates and already_labelled(cur, a, b):
            stats["skipped_dup"] += 1
            continue
        append_pair(cur, a, b, y, source="imported_classmap")
        stats["added"] += 1
        if y == LABEL_SAME:
            n_same += 1; stats["n_same_added"] += 1
        else:
            n_diff += 1; stats["n_diff_added"] += 1

    save_pair_labels(cube_path, cur)
    stats["n_after"] = len(cur.get("pairs", []))
    stats["attempts"] = tries
    return stats
