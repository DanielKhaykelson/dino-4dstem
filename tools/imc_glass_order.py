"""Material-physics maps for IMC crystallization (Ediger/Voyles/Ophus lens).
Per probe position, from the diffraction pattern, compute:
  - crystallinity  = peak/halo (sharp Bragg over diffuse background)
  - halo radius r_h = radius of the dominant scattering maximum (amorphous halo or
    first crystalline ring), -> d-spacing via IMC calib (0.00185 A^-1 / raw-px)
  - halo anisotropy = azimuthal std/mean of intensity in the r_h band (orientational
    order); LOW crystallinity + HIGH anisotropy = ORIENTED (pre-ordered) glass.
  - scattered     = total post-beam intensity (mass-thickness / scattering power).
Saves per-pixel metric maps + per-DINO-class mean patterns for SI3/SI4/SI5
(the 3=overview / 5=interface / 4=needles multi-scale set). DINO not retrained."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
OUT = "docs/paper/draft_v2/figs"
INV_ANG = 0.00185   # A^-1 per raw-512 px (IMC q-calibration)
IMC = {
 "SI3": dict(run="runs/_gui/IMC_SI3_m097k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy", scan=(128, 128)),
 "SI4": dict(run="runs/_gui/IMC_SI4_m097_k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy", scan=(128, 128)),
 "SI5": dict(run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", path=r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy", scan=(128, 128)),
}
for name, c in IMC.items():
    t0 = time.time(); Ny, Nx = c["scan"]; N = Ny * Nx
    asg = np.load(os.path.join(c["run"], "eval", "inference.npz"))["assigns"].astype(int)
    K = int(asg.max()) + 1
    cube = open_lazy_cube(c["path"], scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
    cyx = (H - 1) / 2.0; yy, xx = np.indices((H, Wd)); rr = np.sqrt((yy - cyx)**2 + (xx - cyx)**2)
    th = np.arctan2(yy - cyx, xx - cyx) % (2 * np.pi); ri = rr.astype(int)
    beam = max(8, round(0.11 * H)); post = rr >= beam; nb = H // 2
    lo = max(int(0.10 * nb), beam + 1); hi = int(0.85 * nb); nbins = 90
    ph = np.full(N, np.nan); rh = np.full(N, np.nan); aniso = np.full(N, np.nan); scat = np.zeros(N); prom = np.full(N, np.nan)
    dsum = np.zeros((K, H, Wd)); dcnt = np.zeros(K)
    print(f"[{name}] cube pass N={N} K={K}...", flush=True)
    for rx in range(Ny):
        blk = np.asarray(cube[rx], np.float32)
        for ry in range(Nx):
            i = rx * Nx + ry; pat = blk[ry]; scat[i] = pat[post].sum()
            m, v, _ = _radial_mean_var(pat, (cyx, cyx), beam_px=beam); seg = m[lo:hi]
            if seg.size and seg.sum() > 0:
                base = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
                peak = np.clip(seg - base, 0, None)
                ph[i] = peak.sum() / (seg.sum() + 1e-9)
                rpk = lo + int(np.argmax(peak)); rh[i] = rpk
                prom[i] = peak.max() / (base[rpk - lo] + 1e-9)
                band = (rr > rpk - 4) & (rr < rpk + 4)
                tb = (th[band] / (2 * np.pi) * nbins).astype(int); vv = pat[band]
                az = np.array([vv[tb == k].mean() if (tb == k).any() else 0 for k in range(nbins)])
                aniso[i] = az.std() / (az.mean() + 1e-9)
            dsum[asg[i]] += pat; dcnt[asg[i]] += 1
        if rx % 40 == 0: print(f"   row {rx} ({time.time()-t0:.0f}s)", flush=True)
    dmean = np.array([dsum[k] / max(dcnt[k], 1) for k in range(K)])
    d_ang = np.where(np.isfinite(rh) & (rh > 0), 1.0 / (rh * INV_ANG + 1e-9), np.nan)  # d-spacing (A)
    np.savez(os.path.join(OUT, f"imc_glassorder_{name}.npz"),
             ph=ph, rh=rh, d_ang=d_ang, aniso=aniso, prom=prom, scat=scat,
             assigns=asg, dmean=dmean, scan=np.array([Ny, Nx]), H=H)
    print(f"[{name}] DONE ({time.time()-t0:.0f}s)  median p/h={np.nanmedian(ph):.3f} "
          f"halo r_h(px) med={np.nanmedian(rh):.1f} d~{np.nanmedian(d_ang):.1f}A aniso med={np.nanmedian(aniso):.2f}", flush=True)
print("wrote imc_glassorder_{SI3,SI4,SI5}.npz", flush=True)
