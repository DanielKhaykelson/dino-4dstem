"""Headless BF-disk NBED centering for the 50nm IMC .prz files (mirrors the GUI
pre_panel NBED-center tab: search_radius=40, disk_radius=20, threshold=0.50).
Writes <base>_nbed.cube.npy (float32, same convention as the 150nm cubes) next to
the .prz, plus <base>_nbed_centers.npy (per-pattern BF centre, for QC).
  python src/make_nbed_50nm.py <path_to.prz>
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data import open_lazy_cube
from gui_app.nbed_center import nbed_center_cube

import os as _os
SR = int(_os.environ.get("NBED_SR", 70))   # 50nm beam is ~34px off-centre; widen coarse search from the GUI default 40
DR = int(_os.environ.get("NBED_DR", 20))
TF = float(_os.environ.get("NBED_TF", 0.50))


def main(prz):
    cube = open_lazy_cube(prz, scan_shape=(128, 128))
    print(f"[nbed] {os.path.basename(prz)} loaded shape={cube.shape} dtype={cube.dtype}", flush=True)
    base = prz[:-4] if prz.lower().endswith(".prz") else os.path.splitext(prz)[0]
    dest = base + "_nbed.cube.npy"
    if os.path.exists(dest):
        print(f"[nbed] {dest} already exists, skipping", flush=True); return dest
    t0 = time.time()

    def prog(d, t):
        if d % 16 == 0 or d == t:
            dt = time.time() - t0
            print(f"  {d}/{t} rows  {dt:.0f}s  eta {dt*(t-d)/max(d,1):.0f}s", flush=True)

    centers = nbed_center_cube(cube, dest, search_radius=SR, disk_radius=DR,
                               threshold_frac=TF, progress_cb=prog)
    np.save(base + "_nbed_centers.npy", centers)
    cyx = (cube.shape[2] - 1) / 2.0
    drift = np.sqrt(((centers[..., 0] - cyx) ** 2 + (centers[..., 1] - cyx) ** 2))
    print(f"[nbed] wrote {dest}  ({time.time()-t0:.0f}s)  "
          f"BF-centre drift from geometric: median {np.median(drift):.1f}px max {drift.max():.1f}px", flush=True)
    return dest


if __name__ == "__main__":
    main(sys.argv[1])
