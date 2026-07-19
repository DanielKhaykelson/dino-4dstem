"""Per-GRAIN ACOM + 1D-supported crystallinity (validated recipe), all 3 IMC.
Per DINO grain (connected component, >=30 px): clip-vmax2 grain-average ->
gaussian blur 2 -> blob_log(thr=0.25, sigma2-8) = 2D peaks.
  ACOM: those peaks -> alpha CIF match (k_max=0.35) -> corr + zone axis.
  crystallinity ratio: 1D azimuthal-mean profile; SNIP halo; at each found-peak
    radius take peak(=max(prof-halo) in +-3px)/halo; ratio = sum(peak)/sum(halo).
  frame/grain vacuum gate: grain mean scat < Otsu(log scat) floor -> vacuum.
Maps painted per grain: DINO classes | n_peaks | crystallinity ratio | ACOM corr."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from skimage.feature import blob_log
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from gui_app.acom_core import build_bragg_vectors, _match_safe, load_crystal, prepare_crystal, zone_axis_from_matrix
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl
try:
    from skimage.filters import threshold_otsu
except Exception:
    def threshold_otsu(x, nbins=256):
        h, e = np.histogram(x, nbins); c = (e[:-1]+e[1:])/2; w1 = np.cumsum(h); w2 = np.cumsum(h[::-1])[::-1]
        m1 = np.cumsum(h*c)/np.clip(w1, 1, None); m2 = (np.cumsum((h*c)[::-1])/np.clip(w2[::-1], 1, None))[::-1]
        v = w1[:-1]*w2[1:]*(m1[:-1]-m2[1:])**2; return c[:-1][np.argmax(v)]

OUT = "docs/paper/draft_v2/figs"; VMAX = 50.0; BLUR = 2.0; INV_ANG = 0.00185; KMAX = 0.35  # was 2.0 -> over-clipped ring-band spots (reach ~12), under-measured crystallinity; central beam masked out of band, high clip safe
DET = dict(threshold=0.05, min_sigma=2.0, max_sigma=8.0, num_sigma=6, overlap=0.4)
CIF = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/cifs/alpha.cif"
MINPIX = 30; BAND = 3; MIN_PEAKS = 4
IMC = {
 "SI3": dict(run="runs/_gui/IMC_SI3_m097k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy"),
 "SI4": dict(run="runs/_gui/IMC_SI4_m097_k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy"),
 "SI5": dict(run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", path=r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"),
}
print("loading alpha CIF + plan (k_max=0.35)...", flush=True)
crystal = prepare_crystal(load_crystal(CIF), k_max=KMAX)

def analyze(avg, cyx, H, beam, lo, hi):
    sm = ndimage.gaussian_filter(avg, BLUR)
    p = np.clip(sm - np.median(sm), 0, None); p = np.log1p(p); p = p/(p.max()+1e-9)
    blobs = blob_log(p, **DET)                                  # (N,3) y,x,sigma
    peaks = np.zeros((0, 3))
    if len(blobs):
        pr = np.sqrt((blobs[:, 0]-cyx)**2 + (blobs[:, 1]-cyx)**2)
        keep = (pr >= beam) & (pr <= hi)
        bk = blobs[keep]; prk = pr[keep]
        peaks = np.column_stack([bk[:, 0], bk[:, 1], [sm[int(round(y)), int(round(x))] for y, x, s in bk]]) if len(bk) else np.zeros((0, 3))
    else:
        prk = np.array([])
    # 1D crystallinity at found radii
    m, _, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam); seg = m[lo:hi]
    ratio = 0.0
    if seg.size and seg.sum() > 0 and len(prk):
        halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); pk = np.clip(seg-halo, 0, None)
        ps = hs = 0.0
        for r in prk:
            ri = int(round(r))-lo
            if 0 <= ri < len(seg):
                a = max(0, ri-BAND); b = min(len(seg), ri+BAND+1)
                ps += pk[a:b].max(); hs += halo[ri]
        ratio = float(ps/(hs+1e-9))
    return peaks, int(len(prk)), ratio

summary = {}
fig = Figure(figsize=(17, 11), facecolor="white")
for ri, name in enumerate(IMC):
    t0 = time.time(); c = IMC[name]
    asg = np.load(os.path.join(c["run"], "eval", "inference.npz"))["assigns"].astype(int)
    Ny = Nx = 128; N = Ny*Nx; asgmap = asg.reshape(Ny, Nx)
    zg = np.load(os.path.join(OUT, f"imc_glassorder_{name}.npz")); scat = zg["scat"]
    gate = float(np.exp(threshold_otsu(np.log(np.clip(scat, 1, None)))))
    gid = np.full(N, -1, np.int32); grains = []
    for k in sorted(set(asg)):
        lab, n = ndimage.label(asgmap == k)
        for gi in range(1, n+1):
            m = (lab == gi).ravel()
            if m.sum() >= MINPIX:
                gid[m] = len(grains); grains.append(dict(cls=int(k), n=int(m.sum())))
    G = len(grains)
    cube = open_lazy_cube(c["path"], scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
    cyx = (H-1)/2.0; beam = max(8, round(0.11*H)); lo = beam+1; hi = int(KMAX/INV_ANG)
    GSUM_KEEP = {}   # stash grain-avg patterns for diagnostics (thick-amorphous)
    gsum = np.zeros((G, H, Wd), np.float32); gcnt = np.zeros(G)
    for rx in range(Ny):
        blk = np.clip(np.asarray(cube[rx], np.float32), 0, VMAX)
        for ry in range(Nx):
            g = gid[rx*Nx+ry]
            if g >= 0: gsum[g] += blk[ry]; gcnt[g] += 1
    peaks_all = []; npk = np.zeros(G); ratio = np.zeros(G); gscat = np.zeros(G); vac = np.zeros(G, bool)
    for g in range(G):
        avg = gsum[g]/max(gcnt[g], 1)
        gscat[g] = float(np.median(scat[gid == g]))
        if gscat[g] < gate:
            vac[g] = True; peaks_all.append(np.zeros((0, 3))); continue
        pk, n, r = analyze(avg, cyx, H, beam, lo, hi); peaks_all.append(pk); npk[g] = n; ratio[g] = r
    bv = build_bragg_vectors(peaks_all, centers=[(cyx, cyx)]*G, inv_ang_per_pixel=INV_ANG, Rshape=(G, 1))
    corr, rmat = _match_safe(crystal, bv, G, min_peaks=MIN_PEAKS)
    corr = np.where(vac, np.nan, corr)
    # in-plane orientation angle (fiber plan about [001]) per indexed grain, degrees
    orient = np.full(G, np.nan)
    for g in range(G):
        if np.isfinite(corr[g]) and corr[g] > 0 and np.isfinite(rmat[g]).all():
            orient[g] = np.degrees(np.arctan2(rmat[g][1, 0], rmat[g][0, 0])) % 360.0
    summary[name] = dict(G=G, gate=round(gate, 1), vacuum=int(vac.sum()),
                         indexed=int(np.nansum(corr > 0)),
                         grains=[dict(cls=grains[g]["cls"], n=grains[g]["n"], vac=bool(vac[g]),
                                      npk=int(npk[g]), ratio=round(float(ratio[g]), 2),
                                      corr=round(float(corr[g]), 1) if np.isfinite(corr[g]) else None) for g in range(G)])
    # diagnostic: do GLASS grains (ratio<0.1) stay clean on n_peaks at this threshold?
    matl = ~vac
    glass = matl & (ratio < 0.1); cryst = matl & (ratio >= 0.3)
    gpk = npk[glass]; cpk = npk[cryst]
    print(f"[{name}] G={G} vac={int(vac.sum())} indexed={int(np.nansum(corr>0))} thr={DET['threshold']} | "
          f"GLASS(ratio<.1) n={int(glass.sum())} npk med={np.median(gpk) if gpk.size else 0:.0f} max={int(gpk.max()) if gpk.size else 0} | "
          f"CRYST(ratio>=.3) n={int(cryst.sum())} npk med={np.median(cpk) if cpk.size else 0:.0f} ({time.time()-t0:.0f}s)", flush=True)
    # maps
    def paint(vals, valid):
        mm = np.full(N, np.nan)
        for g in range(G):
            if valid[g]: mm[gid == g] = vals[g]
        return mm.reshape(Ny, Nx)
    mat = ~vac
    idxd = mat & (corr > 0)
    cols = [("tab20", asg.reshape(Ny, Nx).astype(float), "DINO classes", False, None),
            ("inferno", paint(npk, mat), f"grain n_peaks (thr{DET['threshold']})", True, None),
            ("viridis", paint(ratio, mat), "1D peak/halo crystallinity", True, None),
            ("hsv", paint(orient, idxd), f"ACOM orientation (in-plane deg)\n{int(idxd.sum())} indexed", True, (0, 360))]
    for ci, (cm, mp, t, mask, vlim) in enumerate(cols):
        ax = fig.add_subplot(3, 4, ri*4+ci+1)
        cmap = mpl.cm.get_cmap(cm).copy()
        if mask: cmap.set_bad("#222")
        kw = dict(vmin=vlim[0], vmax=vlim[1]) if vlim else {}
        im = ax.imshow(mp, cmap=cmap, interpolation="nearest", **kw); ax.set_xticks([]); ax.set_yticks([])
        if ci == 0: ax.set_ylabel(name, fontsize=12, fontweight="bold")
        if ri == 0: ax.set_title(t, fontsize=9)
        if mask: fig.colorbar(im, ax=ax, fraction=0.046)
    np.savez(os.path.join(OUT, f"grain_acom_v2_{name}.npz"),
             gid=gid, npk=npk, ratio=ratio, gscat=gscat, vac=vac,
             corr=np.where(np.isfinite(corr), corr, -1), orient=orient,
             cls=np.array([grains[g]["cls"] for g in range(G)]),
             gate=gate, gsum=gsum, gcnt=gcnt, H=H)
fig.suptitle(f"Per-GRAIN: ACOM (alpha) + 1D-supported peak/halo crystallinity, blob thr={DET['threshold']}, vacuum-gated (Otsu scat), vmax=2.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "grain_acom_v2.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "grain_acom_v2.png"))
json.dump(summary, open(os.path.join(OUT, "grain_acom_v2.json"), "w"), indent=2)
print("wrote grain_acom_v2.png + .json", flush=True)
