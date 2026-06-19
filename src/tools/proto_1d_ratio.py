"""Prototype: crystallinity = 1D peak/halo measured AT the radii of 2D blob peaks.
2D blob_log (thr=0.25, vmax=2, blur2) -> peak radii. 1D = azimuthal-mean radial
profile; SNIP halo baseline. For each found-peak radius r_k: peak(r_k)=max(m-halo)
in +-band, halo(r_k)=baseline. ratio = sum(peak)/sum(halo) over found radii (0 if
none). Noise blobs land at radii with no 1D peak -> contribute ~0 -> self-cancel.
Test on SI4 frames spanning scattered-intensity percentiles."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from skimage.feature import blob_log
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

OUT = "docs/paper/draft_v2/figs"; VMAX = 2.0; BLUR = 2.0; INV_ANG = 0.00185; KMAX = 0.35
DET = dict(threshold=0.25, min_sigma=2.0, max_sigma=8.0, num_sigma=6, overlap=0.4)
PATH = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy"
zg = np.load(os.path.join(OUT, "imc_glassorder_SI4.npz"))
scat = zg["scat"]; asg = zg["assigns"].astype(int); Ny, Nx = zg["scan"]
try:
    from skimage.filters import threshold_otsu
    GATE = float(np.exp(threshold_otsu(np.log(np.clip(scat, 1, None)))))
except Exception:
    GATE = float(np.percentile(scat, 25))
print(f"frame intensity gate (scat floor) = {GATE:.3e}", flush=True)
order = np.argsort(scat); pcts = [1, 8, 18, 30, 42, 55, 68, 80, 90, 96, 99]
picks = [int(order[int(p/100*(len(order)-1))]) for p in pcts]
cube = open_lazy_cube(PATH, scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape; cyx = (H-1)/2.0
beam = max(8, round(0.11*H)); lo = beam+1; hi = int(KMAX/INV_ANG); BAND = 3

def crystallinity_1d(raw):
    sm = ndimage.gaussian_filter(raw, BLUR)
    p = np.clip(sm - np.median(sm), 0, None); p = np.log1p(p); p = p/(p.max()+1e-9)
    blobs = blob_log(p, **DET)
    if not len(blobs):
        return 0, 0.0
    pr = np.sqrt((blobs[:, 0]-cyx)**2 + (blobs[:, 1]-cyx)**2)
    pr = pr[(pr >= beam) & (pr <= hi)]
    m, _, _ = _radial_mean_var(raw, (cyx, cyx), beam_px=beam); seg = m[lo:hi]
    if seg.size == 0 or seg.sum() <= 0 or not len(pr): return len(pr), 0.0
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); pk = np.clip(seg-halo, 0, None)
    psum = hsum = 0.0
    for r in pr:
        ri = int(round(r)) - lo
        if 0 <= ri < len(seg):
            a = max(0, ri-BAND); b = min(len(seg), ri+BAND+1)
            psum += pk[a:b].max(); hsum += halo[ri]
    return len(pr), float(psum / (hsum + 1e-9))

print(f"{'pct':>4} {'cls':>4} {'scat':>10} {'gate':>5} {'npk':>4} {'ratio_1d':>9}")
for j, i in enumerate(picks):
    rx, ry = divmod(i, Nx); raw = np.clip(np.asarray(cube[rx][ry], np.float32), 0, VMAX)
    gated = scat[i] < GATE
    if gated:
        print(f"p{pcts[j]:>3} c{asg[i]:>2} {scat[i]:>10.3e} {'VAC':>5} {'-':>4} {'-':>9}", flush=True); continue
    npk, r1 = crystallinity_1d(raw)
    print(f"p{pcts[j]:>3} c{asg[i]:>2} {scat[i]:>10.3e} {'ok':>5} {npk:>4} {r1:>9.3f}", flush=True)
