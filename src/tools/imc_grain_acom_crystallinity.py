"""Per-GRAIN crystallinity + ACOM on identical peak detection (user-specified).

Convention (matches GUI ACOM tab + what the model sees):
  - every raw frame clipped at vmax=2 BEFORE averaging
  - grain = connected component of one DINO class (min 30 px)
  - grain average -> gaussian blur sigma=2 (kills single-px shot noise)
    -> detect_peaks_2d (blob_log): threshold=0.05, min_sigma=2, max_sigma=8,
       num_sigma=6, log_stretch=True
  - k_max = 0.35 1/A, calib 0.00185 1/A/px  (r <= 189 px), beam r>=0.11*H excluded
    from peak COUNTS (matching gets all peaks, as in the GUI)
  - metrics per grain: (1) n_peaks  (2) peak-to-halo ratio: blob amplitude vs the
    azimuthal-MEDIAN profile (spot-robust halo) at the blob radii; halo strength =
    max of halo profile in [beam, k_max]
  - ACOM: same peaks -> BraggVectors -> alpha-CIF match (k_max=0.35) -> corr + ZA
  - plus the FULL-SAMPLE footprint average treated identically.
Outputs: per-grain table (json/npz), grain-painted maps (n_peaks, ratio, ACOM corr),
DINO class map, per-class consistency, full-sample row."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from data import open_lazy_cube
from gui_app.acom_core import (detect_peaks_2d, build_bragg_vectors, _match_safe,
                               load_crystal, prepare_crystal, zone_axis_from_matrix)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
try:
    from skimage.filters import threshold_otsu
except Exception:
    def threshold_otsu(x, nbins=256):
        h, e = np.histogram(x, nbins); c = (e[:-1] + e[1:]) / 2; w1 = np.cumsum(h); w2 = np.cumsum(h[::-1])[::-1]
        m1 = np.cumsum(h * c) / np.clip(w1, 1, None); m2 = (np.cumsum((h * c)[::-1]) / np.clip(w2[::-1], 1, None))[::-1]
        v = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2; return c[:-1][np.argmax(v)]

OUT = "docs/paper/draft_v2/figs"; VMAX = 2.0; INV_ANG = 0.00185; KMAX = 0.35
DET = dict(threshold=0.05, min_sigma=2.0, max_sigma=8.0, num_sigma=6, log_stretch=True)
CIF = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/cifs/alpha.cif"
MINPIX = 30; BLUR = 2.0; MIN_PEAKS = 4
IMC = {
 "SI3": dict(run="runs/_gui/IMC_SI3_m097k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy"),
 "SI4": dict(run="runs/_gui/IMC_SI4_m097_k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy"),
 "SI5": dict(run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", path=r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"),
}

print("loading alpha CIF + orientation plan (k_max=0.35)...", flush=True)
crystal = load_crystal(CIF); crystal = prepare_crystal(crystal, k_max=KMAX)

def halo_profiles(pat, cyx, H):
    """azimuthal MEDIAN (spot-robust halo) + MEAN radial profiles."""
    yy, xx = np.indices((H, H)); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
    ri = rr.astype(int); rmax = int(rr.max())
    med = np.zeros(rmax + 1); mean = np.zeros(rmax + 1)
    flat_r = ri.ravel(); flat_v = pat.ravel(); order = np.argsort(flat_r)
    sr = flat_r[order]; sv = flat_v[order]; bounds = np.searchsorted(sr, np.arange(rmax + 2))
    for r in range(rmax + 1):
        seg = sv[bounds[r]:bounds[r + 1]]
        if seg.size: med[r] = np.median(seg); mean[r] = seg.mean()
    return med, mean

def analyze_pattern(avg, cyx, H, beam, rmax_px):
    """blur -> detect -> n_peaks, peak/halo ratio, peaks (raw px for ACOM)."""
    sm = ndimage.gaussian_filter(avg, BLUR)
    peaks = detect_peaks_2d(sm, **DET)                       # (N,3) qx,qy,I (raw px)
    halo_med, _ = halo_profiles(avg, cyx, H)
    if peaks.shape[0]:
        pr = np.sqrt((peaks[:, 0] - cyx) ** 2 + (peaks[:, 1] - cyx) ** 2)
        keep = (pr >= beam) & (pr <= rmax_px)                # count window
        pk = peaks[keep]; prk = pr[keep]
    else:
        pk = peaks; prk = np.array([])
    n_pk = int(pk.shape[0])
    halo_max = float(halo_med[beam:int(rmax_px)].max()) if H else 0.0
    if n_pk:
        amp = np.array([sm[int(round(x)), int(round(y))] for x, y in pk[:, :2]])
        hal = np.array([halo_med[int(round(r))] for r in prk])
        ratio = float(np.median((amp - hal) / (hal + 1e-6)))
        amp_med = float(np.median(amp))
    else:
        ratio = 0.0; amp_med = 0.0
    return peaks, n_pk, ratio, amp_med, halo_max, halo_med

summary = {}
for name, c in IMC.items():
    t0 = time.time()
    asg = np.load(os.path.join(c["run"], "eval", "inference.npz"))["assigns"].astype(int)
    Ny = Nx = 128; N = Ny * Nx; asgmap = asg.reshape(Ny, Nx)
    # --- connected-component grains over all classes ---
    gid = np.full(N, -1, np.int32); grains = []
    for k in sorted(set(asg)):
        lab, n = ndimage.label(asgmap == k)
        for gi in range(1, n + 1):
            m = (lab == gi).ravel()
            if m.sum() >= MINPIX:
                gid[m] = len(grains); grains.append(dict(cls=int(k), n=int(m.sum())))
    G = len(grains)
    cube = open_lazy_cube(c["path"], scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
    cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); rmax_px = KMAX / INV_ANG
    # --- one streaming pass: clipped accumulation per grain + footprint avg ---
    gsum = np.zeros((G, H, Wd), np.float32); gcnt = np.zeros(G, np.int64)
    scat = np.zeros(N); yy, xx = np.indices((H, Wd))
    post = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2) >= beam
    fsum = np.zeros((H, Wd), np.float64); fcnt = 0
    print(f"[{name}] G={G} grains; streaming cube (clip at {VMAX})...", flush=True)
    for rx in range(Ny):
        blk = np.clip(np.asarray(cube[rx], np.float32), 0, VMAX)
        raw = np.asarray(cube[rx], np.float32)
        for ry in range(Nx):
            i = rx * Nx + ry; scat[i] = raw[ry][post].sum()
            g = gid[i]
            if g >= 0: gsum[g] += blk[ry]; gcnt[g] += 1
        if rx % 40 == 0: print(f"   row {rx} ({time.time()-t0:.0f}s)", flush=True)
    ls = np.log(np.clip(scat, 1, None)); foot = ls > threshold_otsu(ls)
    # footprint average (clipped frames): accumulate from grain sums where possible
    # (grains cover most px); for exactness use all footprint px via second light pass
    for rx in range(Ny):
        blk = np.clip(np.asarray(cube[rx], np.float32), 0, VMAX)
        sel = foot[rx * Nx:(rx + 1) * Nx]
        if sel.any(): fsum += blk[sel].sum(0); fcnt += int(sel.sum())
    favg = (fsum / max(fcnt, 1)).astype(np.float32)
    # --- per-grain detection + metrics ---
    peaks_all = []; tab = []
    for g in range(G):
        avg = gsum[g] / max(gcnt[g], 1)
        peaks, n_pk, ratio, amp, halo_max, _ = analyze_pattern(avg, cyx, H, beam, rmax_px)
        peaks_all.append(peaks)
        fr = float(foot[gid == g].mean())
        tab.append(dict(grain=g, cls=grains[g]["cls"], n=grains[g]["n"], foot=round(fr, 2),
                        n_peaks=n_pk, ratio=round(ratio, 3), amp=round(amp, 4),
                        halo=round(halo_max, 4)))
    # --- ACOM on the same peaks ---
    print(f"[{name}] matching {G} grains vs alpha (min_peaks={MIN_PEAKS})...", flush=True)
    bv = build_bragg_vectors(peaks_all, centers=[(cyx, cyx)] * G,
                             inv_ang_per_pixel=INV_ANG, Rshape=(G, 1))
    corr, rmat = _match_safe(crystal, bv, G, min_peaks=MIN_PEAKS)
    for g in range(G):
        za, mis = zone_axis_from_matrix(rmat[g])
        tab[g]["acom_corr"] = round(float(corr[g]), 3)
        tab[g]["za"] = list(za); tab[g]["za_mis"] = round(mis, 1) if np.isfinite(mis) else None
    # --- full-sample row ---
    fpeaks, fn, fratio, famp, fhalo, fhalo_med = analyze_pattern(favg, cyx, H, beam, rmax_px)
    fbv = build_bragg_vectors([fpeaks], centers=[(cyx, cyx)], inv_ang_per_pixel=INV_ANG)
    fcorr, frmat = _match_safe(crystal, fbv, 1, min_peaks=MIN_PEAKS)
    full_row = dict(n_peaks=fn, ratio=round(fratio, 3), amp=round(famp, 4),
                    halo=round(fhalo, 4), acom_corr=round(float(fcorr[0]), 3))
    summary[name] = dict(G=G, full_sample=full_row, grains=tab)
    np.savez(os.path.join(OUT, f"imc_grain_acom_{name}.npz"),
             gid=gid, corr=corr, rmat=rmat, foot=foot,
             n_peaks=np.array([t["n_peaks"] for t in tab]),
             ratio=np.array([t["ratio"] for t in tab]),
             cls=np.array([t["cls"] for t in tab]), favg=favg)
    npk = np.array([t["n_peaks"] for t in tab])
    print(f"[{name}] DONE ({time.time()-t0:.0f}s)  full-sample: {full_row}", flush=True)
    print(f"   grain n_peaks: med={np.median(npk):.0f} max={npk.max()} "
          f"| indexed(corr>0)={int((corr > 0).sum())}/{G}", flush=True)

json.dump(summary, open(os.path.join(OUT, "imc_grain_acom_crystallinity.json"), "w"), indent=2)

# ---------------- figure: maps + comparisons ----------------
fig = Figure(figsize=(17, 11), facecolor="white")
for ri, name in enumerate(IMC):
    z = np.load(os.path.join(OUT, f"imc_grain_acom_{name}.npz"))
    gid = z["gid"]; npk = z["n_peaks"]; ratio = z["ratio"]; corr = z["corr"]; cls = z["cls"]
    Ny = Nx = 128
    def paint(vals, default=np.nan):
        m = np.full(Ny * Nx, default)
        for g, v in enumerate(vals): m[gid == g] = v
        return m.reshape(Ny, Nx)
    asg = np.load(os.path.join(IMC[name]["run"], "eval", "inference.npz"))["assigns"].astype(int)
    ax = fig.add_subplot(3, 5, ri * 5 + 1); ax.imshow(asg.reshape(Ny, Nx), cmap="tab20", interpolation="nearest")
    ax.set_ylabel(f"IMC {name}", fontsize=11, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    if ri == 0: ax.set_title("DINO classes", fontsize=10)
    ax = fig.add_subplot(3, 5, ri * 5 + 2); im = ax.imshow(paint(npk), cmap="inferno"); ax.set_xticks([]); ax.set_yticks([])
    if ri == 0: ax.set_title("grain n_peaks (blob_log)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(3, 5, ri * 5 + 3); im = ax.imshow(paint(ratio), cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
    if ri == 0: ax.set_title("grain peak/halo ratio", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(3, 5, ri * 5 + 4); im = ax.imshow(paint(np.where(corr > 0, corr, np.nan)), cmap="magma"); ax.set_xticks([]); ax.set_yticks([])
    if ri == 0: ax.set_title("grain ACOM corr (alpha)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(3, 5, ri * 5 + 5)
    ok = corr > 0
    ax.scatter(npk[~ok], ratio[~ok], c="#999", s=22, label="unindexed")
    sc = ax.scatter(npk[ok], ratio[ok], c=corr[ok], cmap="magma", s=26, edgecolor="k", lw=0.3, label="indexed")
    ax.set_xlabel("n_peaks"); ax.set_ylabel("peak/halo ratio")
    if ri == 0: ax.set_title("crystallinity vs ACOM\n(color = corr)", fontsize=9)
    ax.legend(fontsize=6); fig.colorbar(sc, ax=ax, fraction=0.046)
fig.suptitle("IMC per-GRAIN crystallinity (blob_log thr=0.05, sigma 2-8, log-stretch, vmax=2, blur sigma=2) + ACOM (alpha, k_max=0.35) on the SAME peaks", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_grain_acom_crystallinity.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "imc_grain_acom_crystallinity.png"))
print("wrote imc_grain_acom_crystallinity.png + .json + per-sample npz", flush=True)
