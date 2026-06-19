"""
prz_to_npy.py — Convert .prz (compressed NpzFile) to .npy (plain array)
for true memory-mapped access.

The .prz format is an npz zip archive — it cannot be memory-mapped;
`np.load(path, mmap_mode='r')` silently loads into RAM. Converting
to raw .npy lets LoadPRZ (or any consumer) use real mmap and
zero-RAM lazy access.

Usage:
    python prz_to_npy.py <prz_path>
    python prz_to_npy.py <prz_path> <npy_path>

Conversion is ~1 min per 10GB cube (mostly disk-bound write).
"""
from __future__ import annotations
import os, sys, time
import numpy as np


def convert_one(prz_path: str, npy_path: str | None = None,
                 verify: bool = True) -> str:
    if npy_path is None:
        base, _ = os.path.splitext(prz_path)
        npy_path = base + ".cube.npy"
    if os.path.exists(npy_path):
        print(f"[skip]   {npy_path} (exists)")
        return npy_path
    t0 = time.perf_counter()
    print(f"[load]   {prz_path}", flush=True)
    arr = np.load(prz_path, allow_pickle=True)
    cube = arr["data"]
    print(f"[write]  {npy_path}  shape={cube.shape}  dtype={cube.dtype}  "
          f"size={cube.nbytes/1e9:.2f} GB", flush=True)
    np.save(npy_path, cube)
    dt = time.perf_counter() - t0
    print(f"[done]   in {dt:.1f}s  (rate={cube.nbytes/dt/1e9:.2f} GB/s)",
          flush=True)
    if verify:
        print(f"[verify] mmap reload...", flush=True)
        cube_mm = np.load(npy_path, mmap_mode='r')
        assert cube_mm.shape == cube.shape, f"shape mismatch: {cube_mm.shape} vs {cube.shape}"
        assert cube_mm.dtype == cube.dtype
        # spot-check a single pattern
        pat0 = cube[0, 0]
        pat0_mm = cube_mm[0, 0]
        assert np.allclose(pat0, pat0_mm), "content mismatch at [0,0]"
        print(f"[verify] OK", flush=True)
    return npy_path


def convert_many(prz_paths: list[str]) -> list[str]:
    out = []
    for p in prz_paths:
        out.append(convert_one(p))
    return out


if __name__ == "__main__":
    if len(sys.argv) == 2:
        convert_one(sys.argv[1])
    elif len(sys.argv) == 3:
        convert_one(sys.argv[1], sys.argv[2])
    else:
        # Default: convert all ROI-split SIs (4 NaPHI + 4 MgNaPHI).
        NA = [rf"D:\DINOSR\data\240312-NaPHI-Nadja-remeasure\EF-4DSTEM\SI-{i:03d}\DataCube_0.prz"
              for i in (5, 6, 7, 8)]
        MG_ROOT = r"D:\DINOSR\data\240312-MgNaPHI-remeasure\EF-4DSTEM"
        MG = [
            os.path.join(MG_ROOT, "SI-003", "Survey_CH2_1.prz"),
            os.path.join(MG_ROOT, "SI-004", "Survey_CH2_1.prz"),
            os.path.join(MG_ROOT, "SI-005", "Survey_CH2_1.prz"),
            os.path.join(MG_ROOT, "SI-006", "Survey_CH2_1.prz"),
        ]
        convert_many(NA + MG)
