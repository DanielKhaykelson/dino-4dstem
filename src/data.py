"""
Minimal dataset + SAMPLES registry for the contrastive run.

Carved out of eval_all.py so this subfolder can run without the
`explainability` dependency that eval_all.py imports at the top.

Matches eval_all.SAMPLES exactly for Na007b (the only sample we actually
train on here) plus the other 5 registered samples for reference.
"""

from __future__ import annotations

import os
import numpy as np
import torch

# Register Dectris-style HDF5 filters (bitshuffle + LZ4) if available.
# Nóvina-saved master files use them; without this, h5py raises
# "Filter not registered" on read.  Silent no-op if not installed.
try:
    import hdf5plugin  # noqa: F401
except Exception:
    pass

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python is required for resize; install it.") from exc


LOCAL_DATA_ROOT = r"D:\DINOSR\data"


SAMPLES = {
    "Na007a": {
        "path": os.path.join(LOCAL_DATA_ROOT, "Na007a.prz"),
        "vmax": 2, "scan_shape": (126, 100), "center_mask_radius": 15,
        "approved_label": None,
    },
    "Na007b": {
        "path": os.path.join(LOCAL_DATA_ROOT, "Na007b.prz"),
        "vmax": 2, "scan_shape": (126, 100), "center_mask_radius": 15,
        "approved_label": "heat_040_050",
    },
    "Na006a": {
        "path": os.path.join(LOCAL_DATA_ROOT, "Na006a.prz"),
        "vmax": 4, "scan_shape": (100, 100), "center_mask_radius": 15,
        "approved_label": "heat_040_070",
    },
    "EuInAs_B100": {
        "path": os.path.join(LOCAL_DATA_ROOT, "EuInAs_B100.prz"),
        "vmax": 30, "scan_shape": (66, 396), "center_mask_radius": 10,
        "approved_label": "heat_040_070",
    },
    "IMC_50nm_SI2": {
        "path": os.path.join(LOCAL_DATA_ROOT, "IMC_50nm_SI2.prz"),
        "vmax": 3, "scan_shape": (128, 128), "center_mask_radius": 15,
        "approved_label": "const_065",
    },
    "IMC_150nm_SI5": {
        "path": os.path.join(LOCAL_DATA_ROOT, "IMC_150nm_SI5.prz"),
        "vmax": 5, "scan_shape": (128, 128), "center_mask_radius": 15,
        "approved_label": "const_040",
    },
    "PLA": {
        # Polylactic acid thin film (NBED-005 dataset from Nadav).
        # Copied from X:\Nadav\251029-NYahalom-PLA-no-ann\...\4D-STEM_Single_Image.prz.
        # Cube shape (128, 128, 256, 256) float32. Intensity distribution per
        # pattern: mean ~0.67, p95 ~1.0, p99 ~2.2, per-pattern max 9k-19k
        # (hot center pixel). vmax=3 chosen to match IMC_50nm scale (same
        # detector family, same organic / amorphous-carbon support).
        "path": os.path.join(LOCAL_DATA_ROOT, "PLA.prz"),
        "vmax": 3, "scan_shape": (128, 128), "center_mask_radius": 15,
        "approved_label": None,
    },
}

# ── 240110-MgNaphi (MgNaPHI at 115mm CL, locally copied) ──────────────────────
# All entries: 115mm CL, 100x100 scan, focused probe, 3.55nm step size (115k mag).
# Path: D:\DINOSR\data\240110-MgNaphi\SI-xxx\DataCube_0.prz (PRZ size 4.9 GB each).
_MG_240110_ROOT = r"D:\DINOSR\data\240110-MgNaphi"
# Focused-probe, 115mm CL, 115k mag SIs (the "good" measurements for training).
# SI-002 (115k mag, 10eV),
# SI-005 (115k mag, 7eV slit, first good after grid change),
# SI-006 (115k mag, 10eV),
# SI-008 (115k mag, 10eV, same position as SI-007 different mag),
# SI-009 (115k mag, 10eV, same position as SI-008 different ROI),
# SI-020 (115k mag, 10eV, zoomed into thin flake),
# SI-022 (115k mag, 10eV, very thin flake, same flake as SI-021).
for _si in (2, 5, 6, 8, 9, 20, 22):
    SAMPLES[f"MgNaPHI240110_SI{_si:03d}"] = {
        "path": os.path.join(_MG_240110_ROOT, f"SI-{_si:03d}", "DataCube_0.prz"),
        "vmax": 2, "scan_shape": (100, 100), "center_mask_radius": 15,
        "approved_label": None,
    }
# Defocused-probe 115mm CL SIs (skipped for now; different physics).
# SI-010/011/012/013 = 4 quarters of same ROI at 20k mag with -12um defocused probe.


def rescale_like_vmax(x, vmax, vmin=None, out_range=(0.0, 1.0), dtype=np.float32):
    x = np.asarray(x); lo, hi = map(float, out_range)
    if x.ndim == 2:
        vm = x.min() if vmin is None else float(vmin)
        d = (float(vmax) - vm) or 1e-12; y = (x - vm) / d
    elif x.ndim == 3:
        vm = x.min(axis=(1, 2), keepdims=True) if vmin is None else float(vmin)
        d = float(vmax) - vm
        if not np.isscalar(d):
            d[d == 0] = 1e-12
        else:
            d = d or 1e-12
        y = (x - vm) / d
    else:
        raise ValueError("Expected 2D or 3D")
    return (np.clip(y, 0, 1) * (hi - lo) + lo).astype(dtype, copy=False)


class _H5Cube4D:
    """4D `(Nx, Ny, H, W)` view over an h5py 3D / 4D dataset. For 3D
    inputs the caller must pass scan_shape=(Ny, Nx). Reads lazily and
    holds the file handle for external-link resolution."""
    def __init__(self, h5_dataset, file_handle=None,
                 scan_shape: tuple | None = None,
                 corrections: dict | None = None):
        s = tuple(h5_dataset.shape)
        self._corr = corrections or {}
        if len(s) == 4:
            self.Nx, self.Ny, self.H, self.W = s
            self._mode = "4d"
        elif len(s) == 3:
            N, H, W = s
            if scan_shape is None:
                raise ValueError(
                    f"3D dataset of length {N}: scan_shape required.")
            Ny, Nx = (int(scan_shape[0]), int(scan_shape[1]))
            if Ny * Nx != N:
                raise ValueError(
                    f"scan_shape ({Ny}, {Nx}) → Ny·Nx={Ny*Nx} ≠ "
                    f"N={N}.")
            self.Nx, self.Ny, self.H, self.W = Nx, Ny, H, W
            self._mode = "3d"
        else:
            raise ValueError(f"need 3D or 4D dataset, got shape {s}")
        self._d = h5_dataset
        self._f = file_handle
        self.shape = (self.Nx, self.Ny, self.H, self.W)
        self.dtype = h5_dataset.dtype

    def __getitem__(self, idx):
        if self._mode == "4d":
            arr = np.asarray(self._d[idx])
            if (self._corr and arr.ndim == 2
                    and arr.shape == (self.H, self.W)):
                arr = _apply_dectris_corrections(arr, self._corr)
            return arr
        if isinstance(idx, tuple) and len(idx) >= 2:
            rx = int(idx[0]); ry = int(idx[1])
            i = rx * self.Ny + ry
            frame = np.asarray(self._d[i])
            if self._corr:
                frame = _apply_dectris_corrections(frame, self._corr)
            if len(idx) == 2:
                return frame
            return frame[idx[2:]]
        raise TypeError(
            f"unsupported index for 3D-backed cube: {type(idx)}")


def _npz_member_header(path, member="data"):
    """Read just the (shape, dtype, is_uncompressed) of a .npy member in a
    .npz/.prz WITHOUT loading the data (header bytes only — cheap even for
    compressed archives)."""
    import zipfile
    zf = zipfile.ZipFile(path, "r")
    try:
        npy = [n for n in zf.namelist() if n.lower().endswith(".npy")]
        if not npy:
            raise ValueError("no .npy member")
        name = None
        for n in npy:
            if n.rsplit("/", 1)[-1].lower() in (member.lower() + ".npy",
                                                member.lower()):
                name = n; break
        if name is None:
            name = max(npy, key=lambda n: zf.getinfo(n).file_size)
        stored = zf.getinfo(name).compress_type == zipfile.ZIP_STORED
        with zf.open(name) as fp:
            version = np.lib.format.read_magic(fp)
            shape, fortran, dtype = np.lib.format._read_array_header(
                fp, version)
    finally:
        zf.close()
    return tuple(shape), dtype, stored


def peek_cube_info(path):
    """Cheaply return (shape4d=(Ny,Nx,H,W), dtype, lazy) for .npy/.prz/.npz
    without loading frame data.  (HDF5 is peeked by the caller.)  `lazy`
    means it can be memory-mapped (won't fill RAM)."""
    pl = path.lower()
    if pl.endswith(".npy"):
        mm = np.load(path, mmap_mode="r")
        return tuple(mm.shape), mm.dtype, True
    if pl.endswith((".prz", ".npz")):
        base, _ = os.path.splitext(path)
        cand = base + ".cube.npy"
        if os.path.exists(cand):
            mm = np.load(cand, mmap_mode="r")
            return tuple(mm.shape), mm.dtype, True
        shape, dtype, stored = _npz_member_header(path, "data")
        return tuple(shape), dtype, stored
    if pl.endswith((".dm4", ".dm3")):
        arr = _open_dm4(path)            # memmap — cheap header peek
        return tuple(arr.shape), arr.dtype, True
    raise ValueError("peek_cube_info: unsupported (use h5 peek for hdf5)")


def bin_realspace_to_npy(src, n, out_path, progress=None):
    """Real-space n×n bin a lazy 4D cube `src` (Ny, Nx, H, W) and write a
    uint16 `.cube.npy` of shape (Ny//n, Nx//n, H, W), STREAMING one output
    frame at a time so the whole cube is never held in RAM.

    Each output position is the mean of the n×n block of diffraction
    patterns (full detector detail kept).  Returns out_path.
    """
    n = int(n)
    Ny, Nx, H, W = src.shape
    oy, ox = Ny // n, Nx // n
    if oy < 1 or ox < 1:
        raise ValueError(f"bin factor {n} too large for scan {Ny}x{Nx}")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16,
                                    shape=(oy, ox, H, W))
    total = oy * ox
    done = 0
    try:
        for i in range(oy):
            for j in range(ox):
                acc = np.zeros((H, W), dtype=np.float64)
                for di in range(n):
                    for dj in range(n):
                        acc += np.asarray(src[i * n + di, j * n + dj],
                                          dtype=np.float64)
                acc /= float(n * n)
                out[i, j] = np.rint(acc).astype(np.uint16)
                done += 1
                if progress and (done % 32 == 0 or done == total):
                    progress(done, total, "binning")
    finally:
        out.flush()
        del out
    return out_path


def _npz_member_memmap(path, member="data"):
    """Memory-map an UNCOMPRESSED .npy member stored inside a .npz/.prz
    zip, with ZERO RAM (frames read from disk on demand) — the key to
    loading multi-GB .prz files without filling memory.

    Returns a numpy memmap of the member's array.  Raises if the member
    is compressed (DEFLATE) — those can't be memory-mapped.
    """
    import zipfile
    import struct
    zf = zipfile.ZipFile(path, "r")
    try:
        npy = [n for n in zf.namelist() if n.lower().endswith(".npy")]
        if not npy:
            raise ValueError("no .npy member in archive")
        # Prefer the requested member ('data'), else the largest array.
        name = None
        for n in npy:
            base = n.rsplit("/", 1)[-1].lower()
            if base in (member.lower() + ".npy", member.lower()):
                name = n; break
        if name is None:
            name = max(npy, key=lambda n: zf.getinfo(n).file_size)
        info = zf.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"member '{name}' is compressed; cannot mmap")
        hdr_off = info.header_offset
    finally:
        zf.close()
    # Skip the local file header to reach the raw .npy bytes.
    with open(path, "rb") as f:
        f.seek(hdr_off)
        local = f.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ValueError("bad local zip header")
        namelen = struct.unpack("<H", local[26:28])[0]
        extralen = struct.unpack("<H", local[28:30])[0]
        npy_off = hdr_off + 30 + namelen + extralen
        # Parse the embedded .npy header to get dtype/shape/order.
        f.seek(npy_off)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        data_off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=data_off,
                     shape=shape, order="F" if fortran else "C")


def _dm4_pick(path):
    """Open a Gatan .dm4/.dm3 and return (array, index, is_eels) for the
    largest >=3D dataset (skips the 2D thumbnail). `array` is a lazy memmap
    when possible. `is_eels` is a best-effort flag from the dm metadata."""
    from ncempy.io.dm import fileDM
    dmf = fileDM(path, on_memory=False)
    n = int(getattr(dmf, "numObjects", 1))
    best, best_size, best_i = None, -1, -1
    for i in range(n):
        try:
            mm = dmf.getMemmap(i)
        except Exception:
            mm = None
        if mm is None:
            continue
        if getattr(mm, "ndim", 0) >= 3 and int(getattr(mm, "size", 0) or 0) > best_size:
            best, best_size, best_i = mm, int(mm.size), i
    if best is None:
        from ncempy.io.dm import dmReader
        for ds in range(n):
            try:
                a = np.asarray(dmReader(path, dSetNum=ds)["data"])
            except Exception:
                continue
            if a.ndim >= 3:
                best, best_i = a, ds
                break
        if best is None:
            raise ValueError(
                f"{os.path.basename(path)}: no 3D/4D dataset found in the "
                f"dm file.")
    # Best-effort: detect EELS / spectrum datasets (energy axis) so we don't
    # silently treat them as diffraction cubes.
    is_eels = False
    try:
        meta = dmf.getMetadata(best_i)
        blob = str(meta).lower()
        is_eels = ("eels" in blob) or ("ev" in str(meta.get("EELS", "")).lower()
                                       if isinstance(meta, dict) else False)
    except Exception:
        is_eels = False
    return best, best_i, is_eels


def _dm4_to_4d(arr, scan_shape=None, path=""):
    """Coerce a dm 3D/4D dataset to (Ny, Nx, H, W) (memmap-preserving)."""
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        N, H, W = arr.shape
        if scan_shape is None:
            s = int(round(N ** 0.5))
            if s * s == N:
                scan_shape = (s, s)
            else:
                raise ValueError(
                    f"3D dm dataset of length {N}: scan_shape (Ny, Nx) is "
                    f"required (could not infer a square scan).")
        Ny, Nx = int(scan_shape[0]), int(scan_shape[1])
        if Ny * Nx != N:
            raise ValueError(
                f"scan_shape {Ny}x{Nx} = {Ny * Nx} != {N} frames in dm file.")
        return arr.reshape(Ny, Nx, H, W)
    raise ValueError(
        f"{os.path.basename(path)}: unexpected dm ndim={arr.ndim}.")


def _dm4_shape_warnings(shape4d, is_eels):
    """Sanity warnings for a candidate (Ny, Nx, H, W) dm dataset."""
    w = []
    H, W = int(shape4d[2]), int(shape4d[3])
    if is_eels:
        w.append("dm metadata mentions EELS — this may be a spectrum image, "
                 "not a 4D-STEM diffraction cube.")
    if min(H, W) < 16:
        w.append(f"detector frame is small ({H}×{W}) — may not be diffraction.")
    if max(H, W) > 4 * max(1, min(H, W)):
        w.append(f"detector frame is very non-square ({H}×{W}) — unusual for "
                 f"4D-STEM.")
    return w


def dm4_probe(path, scan_shape=None):
    """Inspect a .dm4/.dm3 WITHOUT loading frame data. Returns
    {shape4d, dtype, raw_ndim, warnings} so the GUI can confirm before load."""
    arr, _idx, is_eels = _dm4_pick(path)
    shape4d = tuple(_dm4_to_4d(arr, scan_shape=scan_shape, path=path).shape)
    return {"shape4d": shape4d, "dtype": np.dtype(arr.dtype),
            "raw_ndim": int(arr.ndim),
            "warnings": _dm4_shape_warnings(shape4d, is_eels)}


def _open_dm4(path, scan_shape=None):
    """Lazy loader for Gatan .dm4/.dm3 4D-STEM cubes via ncempy. Returns a
    4D array (Ny, Nx, H, W) — a memmap (zero-RAM) when possible. Picks the
    largest >=3D dataset; reshapes 3D (N,H,W) by scan_shape (or square)."""
    arr, _idx, _eels = _dm4_pick(path)
    return _dm4_to_4d(arr, scan_shape=scan_shape, path=path)


def open_lazy_cube(path, scan_shape=None,
                     apply_dectris_corrections: bool = False):
    """Universal lazy 4D-cube loader.  Returns a numpy memmap
    `(Nx, Ny, H, W)` for `.prz / .npz / .npy` and an `_H5Cube4D` wrapper
    for `.h5 / .hdf5`.  Stitches Dectris external links on masters.

    Always passes allow_pickle=True so np.load won't trip on pickled
    object arrays (some legacy .prz / .npy files).
    """
    if path.lower().endswith((".h5", ".hdf5")):
        import h5py
        f = h5py.File(path, "r")
        corr = (_h5_load_dectris_corrections(f)
                  if apply_dectris_corrections else {})
        try:
            dset_path, ndim = _h5_find_data_path(f)
            ds = f[dset_path]
        except ValueError:
            pairs = _h5_dectris_external_data(f, path)
            if not pairs:
                f.close()
                raise ValueError(
                    f"{os.path.basename(path)}: no 3D/4D dataset and "
                    f"no resolvable Dectris external links.")
            ds = _H5MasterFlat(pairs)
            ndim = 3
        if ndim == 4:
            return _H5Cube4D(ds, file_handle=f, corrections=corr)
        if ndim == 3:
            if scan_shape is None:
                inferred = _h5_infer_scan_shape(f, int(ds.shape[0]))
                if inferred is not None:
                    scan_shape = inferred
            if scan_shape is None:
                f.close()
                raise ValueError(
                    f"3D HDF5 dataset of length {ds.shape[0]}: "
                    f"scan_shape (Ny, Nx) is required and could not "
                    f"be inferred from the master file metadata.")
            return _H5Cube4D(ds, file_handle=f,
                                scan_shape=scan_shape, corrections=corr)
        f.close()
        raise ValueError(f"unexpected dataset ndim={ndim}")
    if path.lower().endswith((".dm4", ".dm3")):
        return _open_dm4(path, scan_shape=scan_shape)
    if path.lower().endswith((".prz", ".npz")):
        base, _ = os.path.splitext(path)
        cand = base + ".cube.npy"
        if os.path.exists(cand):
            return np.load(cand, mmap_mode="r", allow_pickle=True)
        # Uncompressed members can be memory-mapped in place (~0 RAM).
        try:
            return _npz_member_memmap(path, member="data")
        except Exception:
            # Compressed / odd layout → must materialize (caller guards RAM).
            arr = np.load(path, allow_pickle=True)
            return arr["data"]
    return np.load(path, mmap_mode="r", allow_pickle=True)


def _h5_dectris_external_data(h5_file, master_path: str
                                 ) -> list[tuple[str, str]]:
    """Resolve external links under /entry/data on a Dectris master.

    Returns a list of (file_path, dataset_path_inside_that_file) tuples,
    sorted by link name (which is the natural data_000001, data_000002…
    order). Empty if the master has no /entry/data group or no external
    links are reachable.

    Tries the link's stored filename first, then falls back to looking
    for the same basename in the master file's directory (handles cubes
    moved to a new server / network share).
    """
    import h5py
    out: list[tuple[str, str]] = []
    if "/entry/data" not in h5_file:
        return out
    grp = h5_file["/entry/data"]
    master_dir = os.path.dirname(os.path.abspath(master_path))
    for name in sorted(grp.keys()):
        try:
            link = grp.get(name, getlink=True)
        except Exception:
            continue
        if not isinstance(link, h5py.ExternalLink):
            continue
        f = link.filename
        if not os.path.isabs(f):
            f = os.path.join(master_dir, f)
        if not os.path.exists(f):
            cand = os.path.join(master_dir, os.path.basename(f))
            if os.path.exists(cand):
                f = cand
        if os.path.exists(f):
            out.append((f, link.path))
    return out


class _H5MasterFlat:
    """Flat (N_total, H, W) view stitched from Dectris external-linked
    data files. Lazy: each pattern read goes to the right file/dataset."""

    def __init__(self, file_dataset_pairs):
        import h5py, bisect
        self._handles = []
        self._datasets = []
        self._cumsum = [0]
        self._bisect = bisect
        H = W = None
        dtype = None
        for f, dpath in file_dataset_pairs:
            fh = h5py.File(f, "r")
            try:
                d = fh[dpath]
            except KeyError:
                fh.close()
                continue
            if d.ndim != 3:
                fh.close()
                continue
            self._handles.append(fh)
            self._datasets.append(d)
            self._cumsum.append(self._cumsum[-1] + d.shape[0])
            if H is None:
                H, W = int(d.shape[1]), int(d.shape[2])
                dtype = d.dtype
            elif (H, W) != (int(d.shape[1]), int(d.shape[2])):
                raise ValueError(
                    f"frame-shape mismatch across Dectris data files: "
                    f"{(H, W)} vs {tuple(d.shape[1:3])}")
        if not self._datasets:
            raise ValueError(
                "no readable 3D dataset in any external-linked data file")
        self.shape = (self._cumsum[-1], H, W)
        self.ndim = 3
        self.dtype = dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            i = int(idx)
            fi = self._bisect.bisect_right(self._cumsum, i) - 1
            local = i - self._cumsum[fi]
            return np.asarray(self._datasets[fi][local])
        if isinstance(idx, slice):
            r = range(*idx.indices(self.shape[0]))
            return np.stack([self[i] for i in r], axis=0)
        if isinstance(idx, (list, tuple, np.ndarray)):
            return np.stack([self[int(i)] for i in idx], axis=0)
        raise TypeError(f"unsupported index type {type(idx)}")

    def close(self):
        for h in self._handles:
            try: h.close()
            except Exception: pass


def _h5_infer_scan_shape(h5_file, n_frames: int
                            ) -> tuple[int, int] | None:
    """Infer (Ny, Nx) for a 3D Dectris master. Returns None if no path
    yields a factorisation Ny·Nx == n_frames.

    Convention (Dectris/Nóvina STEM step scan):
        ntrigger              → Ny  (rows / fast axis triggers)
        nimages_per_trigger   → Nx  (frames per trigger / slow axis)
    Falls back to a stored 2-vector or to nimages alone when ntrigger=1.
    """
    pairs = [
        ("/entry/instrument/detector/detectorSpecific/ntrigger",
         "/entry/instrument/detector/detectorSpecific/nimages_per_trigger"),
        ("/entry/instrument/detector/detectorSpecific/ntrigger",
         "/entry/instrument/detector/detectorSpecific/nimages"),
        ("/entry/instrument/scan/ny",
         "/entry/instrument/scan/nx"),
    ]
    for ny_p, nx_p in pairs:
        try:
            if ny_p in h5_file and nx_p in h5_file:
                ny = int(np.asarray(h5_file[ny_p][()]).item())
                nx = int(np.asarray(h5_file[nx_p][()]).item())
                # Reject ntrigger=1 / single-trigger acquisitions —
                # for STEM these almost always mean the scan grid is
                # NOT what the detector saw (one big trigger holding
                # the whole frame stream).  Fall through to factor
                # heuristic instead.
                if ny * nx == n_frames and ny > 1 and nx > 1:
                    return (ny, nx)
        except Exception:
            continue
    # Single 2-element dataset.
    for path in ("/entry/sample/scan_shape",
                  "/entry/instrument/scan/shape"):
        try:
            if path in h5_file:
                v = np.asarray(h5_file[path][()]).flatten().astype(int)
                if v.size == 2 and int(v[0]) * int(v[1]) == n_frames:
                    return (int(v[0]), int(v[1]))
        except Exception:
            continue
    return None


def _h5_load_dectris_corrections(h5_file) -> dict:
    """Pull Nóvina/Dectris detector corrections from a master file.

    Returns a dict with optional keys:
        pixel_mask : (H, W) bool — True at bad pixels (set to 0).
        flatfield  : (H, W) float — multiplicative gain map.
        saturation : float — saturation threshold (values >= are zeroed).

    Looks under common NeXus paths; missing entries are simply omitted.
    """
    corr: dict = {}
    candidates = {
        "pixel_mask": [
            "/entry/instrument/detector/pixel_mask",
            "/entry/instrument/detector/detectorSpecific/pixel_mask",
        ],
        "flatfield": [
            "/entry/instrument/detector/flatfield",
            "/entry/instrument/detector/detectorSpecific/flatfield",
        ],
        "saturation": [
            "/entry/instrument/detector/saturation_value",
            "/entry/instrument/detector/detectorSpecific/saturation_value",
        ],
    }
    for key, paths in candidates.items():
        for p in paths:
            try:
                if p in h5_file:
                    val = h5_file[p][()]
                    corr[key] = val
                    break
            except Exception:
                continue
    if "pixel_mask" in corr:
        # Dectris uses non-zero = bad. Convert to bool.
        corr["pixel_mask"] = np.asarray(corr["pixel_mask"]) != 0
    if "saturation" in corr:
        try:
            corr["saturation"] = float(np.asarray(corr["saturation"]))
        except Exception:
            corr.pop("saturation", None)
    if "flatfield" in corr:
        corr["flatfield"] = np.asarray(corr["flatfield"], dtype=np.float32)
    return corr


def _apply_dectris_corrections(frame: np.ndarray, corr: dict
                                 ) -> np.ndarray:
    """Apply pixel_mask / flatfield / saturation conservatively.

    Saturation is only zeroed when the stored value is large enough to
    plausibly be an overflow marker (>= 1e4). Some detectors store a
    small "max-counts-before-saturation" instead, which would otherwise
    wipe legitimate Bragg-peak pixels.
    """
    if not corr:
        return frame
    out = frame.astype(np.float32, copy=True)
    sat = corr.get("saturation")
    if sat is not None and float(sat) >= 1e4:
        out = np.where(out >= sat, 0.0, out)
    ff = corr.get("flatfield")
    if ff is not None and ff.shape == out.shape:
        out = out * ff
    pm = corr.get("pixel_mask")
    if pm is not None and pm.shape == out.shape:
        out = np.where(pm, 0.0, out)
    return out


def _h5_find_data_path(h5_file) -> tuple[str, int]:
    """Walk an HDF5 file. Return the path of the biggest 4D dataset if
    any, otherwise the biggest 3D dataset. Returns (path, ndim).

    Note: this does NOT resolve external links (Dectris masters point
    to data_000001.h5, data_000002.h5 …). For those, use
    `_h5_dectris_external_data` to enumerate the links and stitch.
    """
    import h5py
    cands_4d: list[tuple[str, tuple, int]] = []
    cands_3d: list[tuple[str, tuple, int]] = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            if obj.ndim == 4:
                cands_4d.append((name, tuple(obj.shape),
                                   int(np.prod(obj.shape))))
            elif obj.ndim == 3:
                cands_3d.append((name, tuple(obj.shape),
                                   int(np.prod(obj.shape))))
    h5_file.visititems(_visit)
    if cands_4d:
        cands_4d.sort(key=lambda t: -t[2])
        return cands_4d[0][0], 4
    if cands_3d:
        cands_3d.sort(key=lambda t: -t[2])
        return cands_3d[0][0], 3
    raise ValueError("no 3D or 4D dataset found in HDF5 file")


# Backwards-compat alias for any older code that imported this name.
def _h5_find_4d_path(h5_file) -> str:
    p, _ = _h5_find_data_path(h5_file)
    return p


class _H5FlatCube:
    """Wraps an h5py 4D `(Nx, Ny, H, W)` OR 3D `(N, H, W)` dataset so it
    indexes like a flat `(N, H, W)` numpy array.

    For 3D inputs (Eiger master + data, single-stream) the caller MUST
    pass `scan_shape=(Ny, Nx)` so we know how to map flat-i → (rx, ry).

    Reads are lazy — one pattern per __getitem__ call.
    """
    def __init__(self, h5_dataset, scan_shape: tuple | None = None,
                 corrections: dict | None = None):
        s = tuple(h5_dataset.shape)
        self._corr = corrections or {}
        if len(s) == 4:
            self.Nx, self.Ny, self.H, self.W = s
            self._mode = "4d"
        elif len(s) == 3:
            N, H, W = s
            if scan_shape is None:
                # Square fallback: 100×100 scans → N = 10000.
                root = int(round(N ** 0.5))
                if root * root == N:
                    Ny = Nx = root
                else:
                    raise ValueError(
                        f"3D dataset of length {N}: provide scan_shape "
                        f"(Ny, Nx) — the file has no scan grid info.")
            else:
                Ny, Nx = (int(scan_shape[0]), int(scan_shape[1]))
                if Ny * Nx != N:
                    raise ValueError(
                        f"scan_shape ({Ny}, {Nx}) gives "
                        f"Ny·Nx={Ny*Nx} but the dataset has {N} "
                        f"frames — pick the right shape.")
            self.Nx, self.Ny, self.H, self.W = Nx, Ny, H, W
            self._mode = "3d"
        else:
            raise ValueError(f"need 3D or 4D dataset, got shape {s}")
        self._d = h5_dataset
        self.shape = (self.Nx * self.Ny, self.H, self.W)
        self.dtype = h5_dataset.dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            i = int(idx)
            if self._mode == "3d":
                frame = np.asarray(self._d[i])
            else:
                rx, ry = divmod(i, self.Ny)
                frame = np.asarray(self._d[rx, ry])
            if self._corr:
                frame = _apply_dectris_corrections(frame, self._corr)
            return frame
        if isinstance(idx, slice):
            r = range(*idx.indices(self.shape[0]))
            return np.stack([self[i] for i in r], axis=0)
        if isinstance(idx, (list, tuple, np.ndarray)):
            return np.stack([self[int(i)] for i in idx], axis=0)
        raise TypeError(f"unsupported index type {type(idx)}")


class LoadPRZ:
    """In-memory wrapper around a .prz / .npy / .npz / .h5 cube, cropping-
    and-normalizing per diffraction pattern. Matches the behavior of
    eval_all.LoadPRZ exactly.

    Auto-detect: if a sidecar `<basename>.cube.npy` exists next to the .prz,
    load THAT instead (plain .npy files can be truly memory-mapped, while
    .prz/.npz archives are compressed and silently loaded into RAM).
    Use `prz_to_npy.py` to pre-create the sidecar.

    HDF5 (.h5 / .hdf5) is read lazily one pattern at a time — no sidecar
    needed; py4DSTEM EMD layouts auto-detect by picking the largest 4D
    dataset in the file.
    """

    def __init__(self, path, resize=192, vmax=10.0, vmin=None,
                 out_range=(0.0, 1.0),
                 scan_shape: tuple | None = None,
                 apply_dectris_corrections: bool = False,
                 blur_sigma: float | None = None,
                 log_stretch: bool | None = None,
                 ellipticity_ab: float | None = None,
                 ellipticity_theta_deg: float | None = None):
        # Prefer plain-.npy sidecar if available (true mmap, zero-RAM).
        sidecar = None
        if path.lower().endswith((".prz", ".npz")):
            base, _ = os.path.splitext(path)
            cand = base + ".cube.npy"
            if os.path.exists(cand):
                sidecar = cand
        used_path = sidecar if sidecar is not None else path
        self._h5_file = None
        if used_path.lower().endswith((".h5", ".hdf5")):
            import h5py
            self._h5_file = h5py.File(used_path, "r")
            corrections = (_h5_load_dectris_corrections(self._h5_file)
                              if apply_dectris_corrections else {})
            # Try direct: 4D / 3D dataset reachable from inside this file.
            try:
                dset_path, ndim = _h5_find_data_path(self._h5_file)
                ds = self._h5_file[dset_path]
            except ValueError:
                # No internal dataset — Dectris master? stitch external
                # links (`/entry/data/data_*` → data_NNNNNN.h5).
                pairs = _h5_dectris_external_data(self._h5_file,
                                                       used_path)
                if not pairs:
                    raise ValueError(
                        f"{os.path.basename(used_path)}: no 3D/4D "
                        f"dataset, and no resolvable Dectris external "
                        f"links under /entry/data. Check that "
                        f"data_NNNNNN.h5 files are next to the master.")
                ds = _H5MasterFlat(pairs)
                ndim = 3
            if scan_shape is None and ndim == 3:
                inferred = _h5_infer_scan_shape(self._h5_file,
                                                   int(ds.shape[0]))
                if inferred is not None:
                    scan_shape = inferred
            cube_view = _H5FlatCube(ds,
                                       scan_shape=scan_shape,
                                       corrections=corrections)
            self.Nx = cube_view.Nx; self.Ny = cube_view.Ny
            self.H  = cube_view.H;  self.W  = cube_view.W
            self.flat = cube_view
        else:
            if used_path.lower().endswith(".npy"):
                cube = np.load(used_path, mmap_mode='r',
                                  allow_pickle=True)  # true mmap
            else:
                # Memory-map the uncompressed member in place (~0 RAM);
                # only fall back to a full RAM load if it's compressed.
                try:
                    cube = _npz_member_memmap(used_path, member="data")
                except Exception:
                    arr = np.load(used_path, allow_pickle=True)
                    cube = arr['data']
            self.Nx, self.Ny, self.H, self.W = cube.shape
            self.flat = cube.reshape(-1, self.H, self.W)
        self.resize = int(resize); self.vmax = float(vmax)
        self.vmin = vmin; self.out_range = out_range
        # Auto-fill blur / log from the matching SAMPLES entry so every
        # caller automatically picks up the Pre-processing tab toggles
        # without code changes.  Explicit args win.
        if (blur_sigma is None or log_stretch is None
                or ellipticity_ab is None
                or ellipticity_theta_deg is None):
            try:
                _abs = os.path.abspath(path)
                for _k, _v in SAMPLES.items():
                    if os.path.abspath(_v.get("path", "")) == _abs:
                        if blur_sigma is None:
                            blur_sigma = float(
                                _v.get("blur_sigma", 0.0))
                        if log_stretch is None:
                            log_stretch = bool(
                                _v.get("log_stretch", False))
                        if ellipticity_ab is None:
                            ellipticity_ab = float(
                                _v.get("ellipticity_ab", 1.0))
                        if ellipticity_theta_deg is None:
                            ellipticity_theta_deg = float(
                                _v.get("ellipticity_theta_deg", 0.0))
                        break
            except Exception:
                pass
        self.blur_sigma = float(blur_sigma) if blur_sigma else 0.0
        self.log_stretch = bool(log_stretch) if log_stretch else False
        self.ellipticity_ab = (float(ellipticity_ab)
                                  if ellipticity_ab is not None else 1.0)
        self.ellipticity_theta_deg = (float(ellipticity_theta_deg)
                                          if ellipticity_theta_deg
                                          is not None else 0.0)
        self._interp = cv2.INTER_AREA if self.resize < self.H else cv2.INTER_LINEAR
        # Pre-compute the ellipticity affine matrix once.  Same matrix
        # for every frame, applied AFTER resize-to-self.resize so the
        # warp grid matches the model's input grid.  No-op for
        # identity (ab == 1).
        self._ellip_M = None
        if abs(self.ellipticity_ab - 1.0) > 1e-4:
            th = float(np.deg2rad(self.ellipticity_theta_deg))
            c, s = float(np.cos(th)), float(np.sin(th))
            R = np.array([[c, -s], [s, c]], dtype=np.float64)
            # See pre_panel._ellip_warp_matrix for the direction
            # derivation: scale dst major-axis coords by `ab` to
            # sample the source's elongated feature → corrected dst.
            S = np.diag([self.ellipticity_ab, 1.0])
            A = R @ S @ R.T
            cx, cy = (self.resize - 1) * 0.5, (self.resize - 1) * 0.5
            t = np.array([cx, cy]) - A @ np.array([cx, cy])
            M = np.zeros((2, 3), dtype=np.float32)
            M[:2, :2] = A
            M[:, 2] = t
            self._ellip_M = M

    def __len__(self):
        return self.flat.shape[0]

    def get_raw(self, idx):
        return self.flat[idx].astype(np.float32, copy=True)

    def __getitem__(self, idx):
        img = self.flat[idx].astype(np.float32, copy=False)
        if not np.isfinite(img).all():
            img = np.nan_to_num(img, copy=False)
        if img.shape[0] != self.resize or img.shape[1] != self.resize:
            img = cv2.resize(img, (self.resize, self.resize),
                             interpolation=self._interp)
        img = rescale_like_vmax(img, vmax=self.vmax, vmin=self.vmin,
                                 out_range=self.out_range)
        # Ellipticity correction — affine warp baked in so every
        # downstream tab (training / eval / post-hoc / ACOM) gets the
        # SAME corrected frame the Pre-processing tab is previewing.
        if self._ellip_M is not None:
            img = cv2.warpAffine(img, self._ellip_M,
                                    (self.resize, self.resize),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=0.0)
        # Pre-process filters baked into the on-the-fly load — Pre-
        # processing tab toggles propagate here via SAMPLES so the
        # model sees the same image the preview shows.
        if self.blur_sigma > 0:
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, sigma=self.blur_sigma)
        if self.log_stretch:
            # log1p(I × 50) compressed back into [0, 1] by dividing
            # by log1p(50) so downstream model receives the same
            # range it was trained on.
            img = np.log1p(np.clip(img, 0.0, None) * 50.0) \
                    / float(np.log1p(50.0))
        return torch.from_numpy(img).unsqueeze(0)


class LoadPRZMulti:
    """Concatenates multiple .prz cubes into a single flat dataset for
    training. Per-sample view is preserved (no shuffle across cubes at
    load time; that is left to the DataLoader).

    Each component cube is memory-mapped (not fully loaded). Access is
    dispatched per-pattern by a cumulative index table.

    The `scan_shape` concept doesn't apply to a multi-cube combined
    training set (there is no single 2D scan), so callers using
    LoadPRZMulti MUST NOT assume scan_shape*2 == len(dataset). Training
    (DINO pretext) doesn't need scan_shape; only per-SI evaluation
    does, and eval is always done with single-SI LoadPRZ.
    """

    def __init__(self, paths, resize=192, vmax=10.0, vmin=None,
                 out_range=(0.0, 1.0)):
        assert len(paths) >= 1
        self.components = [
            LoadPRZ(p, resize=resize, vmax=vmax, vmin=vmin, out_range=out_range)
            for p in paths
        ]
        self.paths = list(paths)
        self.lengths = np.array([len(c) for c in self.components], dtype=np.int64)
        self.cumsum = np.concatenate([[0], np.cumsum(self.lengths)])
        self.resize = self.components[0].resize
        self.vmax = self.components[0].vmax
        self.H = self.components[0].H
        self.W = self.components[0].W
        # Verify all components share H, W
        for c in self.components[1:]:
            assert (c.H, c.W) == (self.H, self.W), \
                f"pattern-shape mismatch across multi-PRZ: {(c.H, c.W)} vs {(self.H, self.W)}"

    def __len__(self):
        return int(self.cumsum[-1])

    def _locate(self, idx):
        comp = int(np.searchsorted(self.cumsum, idx, side='right') - 1)
        local = idx - int(self.cumsum[comp])
        return comp, local

    def get_raw(self, idx):
        comp, local = self._locate(idx)
        return self.components[comp].get_raw(local)

    def __getitem__(self, idx):
        comp, local = self._locate(idx)
        return self.components[comp][local]


# ── ROI-split experiment samples ──────────────────────────────────────────────
# 240312-NaPHI-Nadja-remeasure  (58mm CL, 115k mag, focused probe, 3.55nm step)
# README: "IN SI-005 TO SI-008 SCANNED THE WHOLE ROI WITH 4 QUARTERS"
# → SI-005/006/007/008 are 4 quarters of the SAME physical ROI.
_NA_REMEAS_ROOT = r"D:\DINOSR\data\240312-NaPHI-Nadja-remeasure\EF-4DSTEM"
for _si in range(1, 11):  # SI-001 through SI-010
    SAMPLES[f"NaPHI_Nadja_SI{_si:03d}"] = {
        "path": os.path.join(_NA_REMEAS_ROOT, f"SI-{_si:03d}", "DataCube_0.prz"),
        "vmax": 2, "scan_shape": (100, 100), "center_mask_radius": 15,
        "approved_label": None,
    }
# Combined training sets (paths stored as list; LoadPRZMulti handles)
SAMPLES["NaPHI_Nadja_4Q_SI5"] = {
    "paths": [os.path.join(_NA_REMEAS_ROOT, "SI-005", "DataCube_0.prz")],
    "vmax": 2, "scan_shape": (100, 100), "center_mask_radius": 15,
    "approved_label": None, "is_multi": True,
}
SAMPLES["NaPHI_Nadja_4Q_all4"] = {
    "paths": [os.path.join(_NA_REMEAS_ROOT, f"SI-{i:03d}", "DataCube_0.prz")
              for i in (5, 6, 7, 8)],
    "vmax": 2, "scan_shape": (200, 200),  # nominal; not used for multi
    "center_mask_radius": 15, "approved_label": None, "is_multi": True,
}

# 240312-MgNaPHI-remeasure  (58mm CL, 115k mag, focused probe, 3.55nm step)
# README: "SI-003 to SI-006 I SCANNED ACROSS THE FLAKE img 1217"
# The per-SI file name is irregular in this session:
#   SI-001: DataCube_0.prz
#   SI-003..SI-007, SI-010, SI-011: Survey_CH2_1.prz
#   SI-008: Survey_CH2_0_1.prz
#   SI-009: Survey_CH2_0_0_1.prz
#   SI-012: Survey_CH2_0_0_0_1.prz
# Each is a (100, 100, 512, 512) float32 cube.
_MG_REMEAS_ROOT = r"D:\DINOSR\data\240312-MgNaPHI-remeasure\EF-4DSTEM"
_MG_FILE_PER_SI = {
    1: "DataCube_0.prz",
    2: "DataCube_2_1.prz",
    3: "Survey_CH2_1.prz", 4: "Survey_CH2_1.prz",
    5: "Survey_CH2_1.prz", 6: "Survey_CH2_1.prz",
    7: "Survey_CH2_1.prz",
    8: "Survey_CH2_0_1.prz",
    9: "Survey_CH2_0_0_1.prz",
    10: "Survey_CH2_1.prz", 11: "Survey_CH2_1.prz",
    12: "Survey_CH2_0_0_0_1.prz",
}
for _si, _fn in _MG_FILE_PER_SI.items():
    SAMPLES[f"MgNaPHI_remeas_SI{_si:03d}"] = {
        "path": os.path.join(_MG_REMEAS_ROOT, f"SI-{_si:03d}", _fn),
        "vmax": 2, "scan_shape": (100, 100), "center_mask_radius": 15,
        "approved_label": None,
    }
SAMPLES["MgNaPHI_remeas_4Q_SI3"] = {
    "paths": [os.path.join(_MG_REMEAS_ROOT, "SI-003", _MG_FILE_PER_SI[3])],
    "vmax": 2, "scan_shape": (100, 100), "center_mask_radius": 15,
    "approved_label": None, "is_multi": True,
}
SAMPLES["MgNaPHI_remeas_4Q_all4"] = {
    "paths": [os.path.join(_MG_REMEAS_ROOT, f"SI-{i:03d}", _MG_FILE_PER_SI[i])
              for i in (3, 4, 5, 6)],
    "vmax": 2, "scan_shape": (200, 200),
    "center_mask_radius": 15, "approved_label": None, "is_multi": True,
}

# 240521-MgPhi1-remeasure-tilt-holeyC -- 58mm CL, 10k mag, 100x100 scan,
# tilt series. Each NBED-XXX{a,b,c} is the same particle at three different
# tilt angles. README key:
#   001a tilt=-35deg (out of plane)   001b tilt=+55deg (in-plane-ish)
#   001c tilt=  0deg (in-plane-ish)   002a tilt=-45deg
#   002b tilt=  0deg (lines)          002c tilt=+35deg
# Cube file in each subfolder is DataCube_0.prz (10 GB).
_TILT_HOLEYC_ROOT = r"D:\DINOSR\data\240521-MgPhi1-remeasure-tilt-holeyC"
for _grp in (1, 2):
    for _let in ("a", "b", "c"):
        SAMPLES[f"MgPhi_tilt_NBED{_grp:03d}{_let}"] = {
            "path": os.path.join(_TILT_HOLEYC_ROOT,
                                  f"NBED-{_grp:03d}{_let}",
                                  "DataCube_0.prz"),
            "vmax": 2, "scan_shape": (100, 100),
            "center_mask_radius": 15, "approved_label": None,
        }


# ── runtime-registered samples (loaded ad-hoc from the GUI) ───────────────────
# Keys created by `register_runtime_sample` are tagged with this prefix so
# downstream code can tell built-in vs user-loaded entries apart.
RUNTIME_SAMPLE_PREFIX = "loaded__"


def _scan_shape_from_prz(path):
    """Read just the shape of a .prz / .npz / .npy / .h5 cube without
    loading it into RAM. Returns (Ny, Nx, H, W)."""
    if path.lower().endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(path, "r") as f:
            try:
                dset_path, ndim = _h5_find_data_path(f)
                s = tuple(f[dset_path].shape)
            except ValueError:
                pairs = _h5_dectris_external_data(f, path)
                if not pairs:
                    raise
                # Stitched master: peek total N from all chunks.
                total = 0; H_ = W_ = None
                for fp, dp in pairs:
                    with h5py.File(fp, "r") as gf:
                        sh = tuple(gf[dp].shape)
                        if len(sh) != 3: continue
                        total += sh[0]
                        H_ = sh[1]; W_ = sh[2]
                if total == 0:
                    raise ValueError(
                        "Dectris master found but data files have no "
                        "3D dataset")
                s = (total, H_, W_)
                ndim = 3
            if ndim == 3:
                inferred = _h5_infer_scan_shape(f, s[0])
        if ndim == 3:
            N, H, W = s
            if inferred is not None:
                Ny, Nx = inferred
                return (Ny, Nx, H, W)
            root = int(round(N ** 0.5))
            if root * root == N:
                return (root, root, H, W)        # square fallback
            raise ValueError(
                f"3D HDF5 dataset of {N} frames: scan_shape (Ny, Nx) "
                f"required and could not be inferred from the master "
                f"file.  Pass scan_shape=(Ny, Nx) to "
                f"register_runtime_sample(...) or the GUI's load dialog.")
    elif path.lower().endswith((".prz", ".npz")):
        base, _ = os.path.splitext(path)
        cand = base + ".cube.npy"
        if os.path.exists(cand):
            arr = np.load(cand, mmap_mode="r", allow_pickle=True)
        else:
            arr = np.load(path, allow_pickle=True, mmap_mode="r")["data"]
        s = tuple(arr.shape)
    else:
        arr = np.load(path, mmap_mode="r", allow_pickle=True)
        s = tuple(arr.shape)
    if len(s) != 4:
        raise ValueError(f"expected 4D cube (Ny, Nx, H, W), got {s}")
    return s


def apply_sample_filters(img_2d: np.ndarray,
                            sample_key_or_cfg) -> np.ndarray:
    """Apply the SAMPLES-stored blur σ + log-stretch to a 2D image in
    [0, 1].  Mirrors the post-vmax half of LoadPRZ.__getitem__ so any
    panel that aggregates raw frames (class averages, grain averages,
    etc.) can render them with the SAME filters the model trained on.

    Accepts either a sample-key string or a SAMPLES cfg dict.
    """
    if isinstance(sample_key_or_cfg, str):
        cfg = SAMPLES.get(sample_key_or_cfg) or {}
    else:
        cfg = sample_key_or_cfg or {}
    out = img_2d.astype(np.float32, copy=False)
    sig = float(cfg.get("blur_sigma", 0.0) or 0.0)
    if sig > 0:
        from scipy.ndimage import gaussian_filter
        out = gaussian_filter(out, sigma=sig)
    if bool(cfg.get("log_stretch", False)):
        out = np.log1p(np.clip(out, 0.0, None) * 50.0) \
                / float(np.log1p(50.0))
    return out


def loaded_sample_keys():
    """Keys for datasets LOADED BY PATH this session (runtime-registered
    via register_runtime_sample / register_runtime_multi_sample), in
    load order.

    The GUI uses this instead of the full SAMPLES catalogue so users pick
    data by browsing to a file, not from the long built-in named list.
    Returns [] when nothing has been loaded yet.
    """
    return [k for k, v in SAMPLES.items()
            if isinstance(v, dict) and v.get("_runtime")]


def register_runtime_sample(path, *, scan_shape=None, vmax=2.0,
                              center_mask_radius=15, key=None,
                              blur_sigma: float = 0.0,
                              log_stretch: bool = False):
    """Inject an arbitrary .prz / .npy cube into SAMPLES at runtime so the
    rest of the pipeline (training, eval, post-hoc, ACOM) can address it
    by key.

    Idempotent: if the same absolute path is already registered, the
    existing key is returned (and metadata refreshed if you pass new
    values).

    Returns the key string.
    """
    abspath = os.path.abspath(path)
    # If we've already registered this exact file, reuse the key.
    for k, v in SAMPLES.items():
        if not k.startswith(RUNTIME_SAMPLE_PREFIX):
            continue
        # Never hijack a multi-cube entry (it exposes path=first-cube for
        # tab compatibility, but is owned by register_runtime_multi_sample).
        if v.get("is_multi"):
            continue
        if os.path.abspath(v.get("path", "")) == abspath:
            if scan_shape is not None:
                v["scan_shape"] = tuple(scan_shape)
            if vmax is not None:
                v["vmax"] = float(vmax)
            if center_mask_radius is not None:
                v["center_mask_radius"] = int(center_mask_radius)
            v["blur_sigma"] = float(blur_sigma)
            v["log_stretch"] = bool(log_stretch)
            return k
    if scan_shape is None:
        Ny, Nx, _H, _W = _scan_shape_from_prz(abspath)
        scan_shape = (Ny, Nx)
    if key is None:
        # synthesize a friendly key from the basename
        base = os.path.splitext(os.path.basename(abspath))[0]
        # avoid collisions with builtin entries
        candidate = f"{RUNTIME_SAMPLE_PREFIX}{base}"
        suffix = 0
        while candidate in SAMPLES:
            suffix += 1
            candidate = f"{RUNTIME_SAMPLE_PREFIX}{base}_{suffix}"
        key = candidate
    SAMPLES[key] = {
        "path": abspath,
        "vmax": float(vmax),
        "scan_shape": tuple(scan_shape),
        "center_mask_radius": int(center_mask_radius),
        "blur_sigma": float(blur_sigma),
        "log_stretch": bool(log_stretch),
        "approved_label": None,
        "_runtime": True,
    }
    return key


def register_runtime_multi_sample(paths, *, vmax=2.0,
                                     center_mask_radius=15, key=None):
    """Register a list of cubes as ONE multi-sample entry (for combined-
    training mode).  `LoadPRZMulti` handles the actual concatenation;
    `run_contrastive` routes through it when the SAMPLES entry has
    `is_multi=True` and a `paths` list.

    Returns the key string.  Existing single-cube `register_runtime_sample`
    is unchanged.
    """
    if not paths or len(paths) < 1:
        raise ValueError("multi-sample needs ≥ 1 path")
    abspaths = [os.path.abspath(p) for p in paths]
    # Idempotent: if already registered with the same exact paths, reuse.
    for k, v in SAMPLES.items():
        if not k.startswith(RUNTIME_SAMPLE_PREFIX):
            continue
        existing = v.get("paths")
        if existing and [os.path.abspath(p) for p in existing] == abspaths:
            v["vmax"] = float(vmax)
            v["center_mask_radius"] = int(center_mask_radius)
            return k
    # Probe per-cube shapes to detect mismatches early.
    shapes = []
    for p in abspaths:
        try:
            shapes.append(_scan_shape_from_prz(p))
        except Exception:
            shapes.append(None)
    # Use the first cube's H,W; total scan length = sum(Ny·Nx) — but we
    # store a synthetic scan_shape since multi-mode doesn't reshape onto
    # one grid.  Total frames as a 1-D scan grid for fallbacks.
    H = W = None
    total = 0
    for s in shapes:
        if s is None: continue
        ny, nx, h, w = s
        H = H or h; W = W or w
        total += ny * nx
    scan_shape = (total or 1, 1)
    if key is None:
        base = os.path.splitext(os.path.basename(abspaths[0]))[0]
        candidate = f"{RUNTIME_SAMPLE_PREFIX}multi__{base}"
        suffix = 0
        while candidate in SAMPLES:
            suffix += 1
            candidate = f"{RUNTIME_SAMPLE_PREFIX}multi__{base}_{suffix}"
        key = candidate
    SAMPLES[key] = {
        # `paths` (+ is_multi) is what Training/LoadPRZMulti consumes.
        # Also expose a single `path` (= first cube) so dataset-only
        # tabs that key on cfg["path"] (preview / ACOM dp_max / Blob)
        # don't KeyError on a multi entry — run_contrastive checks
        # is_multi FIRST, so this never short-circuits multi training.
        "path": abspaths[0],
        "paths": abspaths,
        "vmax": float(vmax),
        "scan_shape": scan_shape,
        "center_mask_radius": int(center_mask_radius),
        "approved_label": None,
        "is_multi": True,
        "_runtime": True,
    }
    return key
