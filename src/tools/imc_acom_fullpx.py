"""Per-PIXEL (stride=1) ACOM + CRYSTAL-TO-AMORPHOUS maps for IMC SI3/SI4/SI5.
Detection chain (identical to the per-grain analysis):
frame clip vmax=2 -> gaussian blur sigma=2 -> blob_log(threshold=0.05, sigma 2-8,
num_sigma=6, log_stretch) -> BraggVectors (0.00185 1/A/px) -> alpha CIF match
(k_max=0.35, min_peaks=4).
Saves per-pixel: corr, rmat, zone-axis, n_peaks, AND peak-to-halo metrics
(amp = median blob amplitude on the blurred frame; halo = max of the azimuthal-
MEDIAN radial profile in [beam, k_max]; ratio = amp / max(halo, 0.01)) so the
full-resolution crystallinity map can be compared against the DINO grains.
Streams the cube row-by-row (memory-safe); matching is one vectorised call.
Run one sample per invocation:  python imc_acom_fullpx.py SI3"""
import os, sys, json, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from data import open_lazy_cube
from gui_app.acom_core import (detect_peaks_2d, build_bragg_vectors, _match_safe,
                               load_crystal, prepare_crystal, zone_axis_from_matrix)

OUT = "docs/paper/draft_v2/figs"; VMAX = 2.0; INV_ANG = 0.00185; KMAX = 0.35
DET = dict(threshold=0.05, min_sigma=2.0, max_sigma=8.0, num_sigma=6, log_stretch=True)
CIF = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/cifs/alpha.cif"
BLUR = 2.0; MIN_PEAKS = 4
IMC = {
 "SI3": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
 "SI4": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
 "SI5": r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
}
name = sys.argv[1] if len(sys.argv) > 1 else "SI3"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1      # 2 = match GUI
MAXROWS = int(os.environ.get("PROBE_ROWS", "0"))           # >0 = timing probe
path = IMC[name]; Ny = Nx = 128; N = Ny * Nx

print(f"[{name}] loading alpha CIF + plan (k_max={KMAX})...", flush=True)
crystal = load_crystal(CIF); crystal = prepare_crystal(crystal, k_max=KMAX)
cube = open_lazy_cube(path, scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); rmax_px = KMAX / INV_ANG

READ_TIMEOUT = 90.0   # s; a single row-block read should take < 1s warm
def read_row(rx, retries=6):
    """Read cube[rx] with a hard timeout + retry so a transient disk stall
    (drive sleep / hiccup) is abandoned and retried instead of hanging forever.
    A stuck read thread is left as a leaked daemon; the retry issues a fresh read."""
    for attempt in range(retries):
        box = {}
        def _do():
            try: box["v"] = np.asarray(cube[rx], np.float32)
            except Exception as e: box["e"] = e
        th = threading.Thread(target=_do, daemon=True); th.start(); th.join(READ_TIMEOUT)
        if "v" in box: return box["v"]
        if "e" in box:
            print(f"   !! row {rx} read error {box['e']!r} (attempt {attempt+1}); retrying", flush=True)
        else:
            print(f"   !! row {rx} read STALLED >{READ_TIMEOUT:.0f}s (attempt {attempt+1}); retrying", flush=True)
        time.sleep(2.0)
    raise RuntimeError(f"row {rx} unreadable after {retries} attempts")

# precompute halo annulus ONCE (cheap vectorized halo = mean diffuse intensity in
# the diffraction band; spot contribution is area-negligible vs the halo)
yy, xx = np.indices((H, Wd)); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
halo_mask = (rr >= beam) & (rr <= rmax_px)
nrows = MAXROWS if MAXROWS > 0 else Ny
t0 = time.time(); peaks_all = []
n_pk = np.zeros(N, np.int16); amp = np.zeros(N, np.float32)
halo = np.zeros(N, np.float32); ratio = np.zeros(N, np.float32)
for rx in range(nrows):
    blk = np.clip(read_row(rx), 0, VMAX)   # whole row-block, timeout-guarded
    for ry in range(Nx):
        if (rx % STRIDE) or (ry % STRIDE):
            peaks_all.append(np.zeros((0, 3))); continue
        i = rx * Nx + ry
        sm = ndimage.gaussian_filter(blk[ry], BLUR)
        halo[i] = float(sm[halo_mask].mean())             # vectorized halo
        pk = detect_peaks_2d(sm, **DET)
        peaks_all.append(pk)
        if pk.shape[0]:
            pr = np.sqrt((pk[:, 0] - cyx) ** 2 + (pk[:, 1] - cyx) ** 2)
            keep = (pr >= beam) & (pr <= rmax_px)
            n_pk[i] = int(keep.sum())
            if keep.any():
                a = np.array([sm[int(round(x)), int(round(y))] for x, y in pk[keep][:, :2]])
                amp[i] = float(np.median(a))
        ratio[i] = amp[i] / max(halo[i], 0.01)
    el = time.time() - t0; eta = el / max(rx + 1, 1) * (nrows - rx - 1)
    print(f"   detect row {rx+1}/{nrows}  ({el:.0f}s, eta {eta/60:.1f}min)", flush=True)
if MAXROWS > 0:
    spp = (time.time() - t0) / max(nrows, 1)
    print(f"[PROBE] {nrows} rows in {time.time()-t0:.0f}s = {spp:.1f}s/row -> "
          f"full stride={STRIDE}: {spp*Ny/60:.1f}min/sample", flush=True)
    sys.exit(0)
print(f"[{name}] detection done ({time.time()-t0:.0f}s); median n_pk={np.median(n_pk):.0f}; matching {N} px...", flush=True)

bv = build_bragg_vectors(peaks_all, centers=[(cyx, cyx)] * N,
                         inv_ang_per_pixel=INV_ANG, Rshape=(N, 1))
t1 = time.time()
corr, rmat = _match_safe(crystal, bv, N, min_peaks=MIN_PEAKS, progress_bar=False)
print(f"[{name}] match done ({time.time()-t1:.0f}s); indexed(corr>0)={int((corr>0).sum())}/{N}", flush=True)

za_u = np.zeros(N, np.int8); za_v = np.zeros(N, np.int8); za_w = np.zeros(N, np.int8)
for i in range(N):
    if np.isfinite(rmat[i]).any():
        (u, v, w), _ = zone_axis_from_matrix(rmat[i])
        za_u[i], za_v[i], za_w[i] = u, v, w
np.savez(os.path.join(OUT, f"imc_acom_fullpx_{name}.npz"),
         corr=corr, rmat=rmat, n_peaks=n_pk, amp=amp, halo=halo, ratio=ratio,
         za_u=za_u, za_v=za_v, za_w=za_w,
         scan=np.array([Ny, Nx]), params=json.dumps(dict(VMAX=VMAX, BLUR=BLUR, **DET,
                                                         k_max=KMAX, min_peaks=MIN_PEAKS)))
print(f"[{name}] wrote imc_acom_fullpx_{name}.npz  (total {time.time()-t0:.0f}s)", flush=True)
