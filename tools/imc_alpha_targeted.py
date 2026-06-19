"""Shot-noise-aware, alpha-TARGETED crystallinity detector for IMC.
Per probe: from the azimuthal mean m(r) and azimuthal variance v(r) of the
diffraction pattern, at the KNOWN alpha ring radii only:
  - radial-peak signal  : m(r_alpha) above the snip halo baseline (Bragg ring present)
  - excess azim variance: v(r_alpha) - g*m(r_alpha), g = per-pixel shot-noise gain
    estimated as the 10th-percentile of v/m over the profile (Poisson floor).
    -> real Bragg spots give POSITIVE excess; pure shot noise gives ~0.
  - control radii (off the alpha rings) -> specificity (alpha must exceed control).
Compares the alpha-targeted crystalline fraction to the old radial p/h>0.5, to test
whether peak/halo missed real (sparse-Bragg) crystallinity or just shot noise."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
OUT = "docs/paper/draft_v2/figs"; INV_ANG = 0.00185
IMC = {
 "SI3": (r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy", (128, 128)),
 "SI4": (r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy", (128, 128)),
 "SI5": (r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy", (128, 128)),
}
ALPHA_D = [7.4, 6.0, 4.75, 3.9]      # strong alpha reflections within range
for name, (path, scan) in IMC.items():
    t0 = time.time(); Ny, Nx = scan; N = Ny * Nx
    cube = open_lazy_cube(path, scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape; cyx = (H - 1) / 2.0
    nb = H // 2; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = int(0.85 * nb)
    ra = [int(round(1.0 / (d * INV_ANG))) for d in ALPHA_D]; ra = [r for r in ra if lo < r < hi]
    # control radii: midpoints between alpha rings + a far-q quiet band, avoiding +-3px of any alpha ring
    cand = list(range(lo + 5, hi - 5, 6)); rc = [r for r in cand if all(abs(r - a) > 6 for a in ra)][:6]
    ex_a = np.zeros(N); ex_c = np.zeros(N); pk_a = np.zeros(N); g_arr = np.zeros(N); ph = np.full(N, np.nan)
    print(f"[{name}] alpha radii(px)={ra} control={rc} ...", flush=True)
    for rx in range(Ny):
        blk = np.asarray(cube[rx], np.float32)
        for ry in range(Nx):
            i = rx * Nx + ry; m, v, _ = _radial_mean_var(blk[ry], (cyx, cyx), beam_px=beam)
            seg = m[lo:hi]
            if not (seg.size and seg.sum() > 0): continue
            rr = np.arange(lo, hi); mm = m[lo:hi]; vv = v[lo:hi]; ok = mm > 0
            ratio = (vv[ok] / mm[ok]); g = np.percentile(ratio, 10) if ratio.size else 0.0; g_arr[i] = g
            # old p/h
            halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); ph[i] = np.clip(seg - halo, 0, None).sum() / (seg.sum() + 1e-9)
            # alpha-targeted (sum over a small band +-2px around each alpha radius)
            def band_excess(rad):
                e = 0.0; p = 0.0
                for r in rad:
                    sl = slice(r - 2, r + 3); mb = m[sl]; vb = v[sl]; good = mb > 0
                    if good.any():
                        e += np.clip(vb[good] - g * mb[good], 0, None).sum() / (mb[good].mean() ** 2 + 1e-9)
                        hb = halo[max(r - 2 - lo, 0):r + 3 - lo]
                        if hb.size: p += np.clip(mb[good][:hb.size] - hb[:good.sum()] if False else mb[good] - np.median(mb), 0, None).sum() / (mb[good].mean() + 1e-9)
                return e / max(len(rad), 1), p / max(len(rad), 1)
            ex_a[i], pk_a[i] = band_excess(ra); ex_c[i], _ = band_excess(rc)
        if rx % 40 == 0: print(f"   row {rx} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"{OUT}/imc_alpha_targeted_{name}.npz", ex_a=ex_a, ex_c=ex_c, pk_a=pk_a, g=g_arr, ph=ph, scan=np.array([Ny, Nx]))
    # alpha-crystalline = alpha excess significantly above control (noise-corrected, alpha-specific)
    spec = ex_a - ex_c
    thr = np.nanpercentile(spec, 50) + 0  # placeholder; report distribution
    f_old = float(np.nanmean(ph > 0.5))
    f_specpos = float((spec > 0).mean()); f_spec2 = float((spec > 2 * np.nanstd(ex_c)).mean())
    print(f"[{name}] DONE p/h>0.5={f_old*100:.0f}%  alpha-excess>control={f_specpos*100:.0f}%  "
          f"alpha-excess>2σ(control)={f_spec2*100:.0f}%  median g(shot-noise)={np.median(g_arr):.2f} ({time.time()-t0:.0f}s)", flush=True)
print("wrote imc_alpha_targeted_{SI3,SI4,SI5}.npz", flush=True)
