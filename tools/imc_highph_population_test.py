"""Decisive test of the HIGH-p/h, NO-spot population (old '67% mature crystal').
Sum ~300 pixels from a contiguous patch of (per-pixel p/h>0.5, not spot-detected,
in footprint) and compare its radial profile to (a) a spot-crystal patch and
(b) a true-glass patch (p/h<0.3, no spots):
  sharp peaks AT the alpha radii -> fine-grained alpha polycrystal (powder ring)
  broad single bump ~4.5 A      -> amorphous halo (old p/h fraction = artifact)
Radial profile of a sum = sum of radial profiles, so patch size/orientation mixing
does not blur the radial axis. Also report azimuthal contrast at the main peak."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
try:
    from skimage.filters import threshold_otsu
except Exception:
    def threshold_otsu(x, nbins=256):
        h, e = np.histogram(x, nbins); c = (e[:-1] + e[1:]) / 2; w1 = np.cumsum(h); w2 = np.cumsum(h[::-1])[::-1]
        m1 = np.cumsum(h * c) / np.clip(w1, 1, None); m2 = (np.cumsum((h * c)[::-1]) / np.clip(w2[::-1], 1, None))[::-1]
        v = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2; return c[:-1][np.argmax(v)]

OUT = "docs/paper/draft_v2/figs"
IMC = {
 "SI3": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
 "SI4": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
}
INV_ANG = 0.00185; ALPHA_D = [7.4, 6.0, 4.75, 3.9]; NPIX = 300

def patch_from(maskmap, npix):
    lab, n = ndimage.label(maskmap)
    if n == 0: return None
    sizes = ndimage.sum(maskmap, lab, range(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    idx = np.where((lab == big).ravel())[0]
    return idx[:npix]

def psum(cube, idxs, Nx, H):
    acc = np.zeros((H, H), np.float64); rows = {}
    for i in idxs: rows.setdefault(int(i) // Nx, []).append(int(i) % Nx)
    for rx in sorted(rows):
        blk = np.asarray(cube[rx], np.float32)
        for ry in rows[rx]: acc += blk[ry]
    return acc / max(len(idxs), 1)

def azim_contrast(pat, cyx, r, dr=3):
    H, W = pat.shape; yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2); th = (np.arctan2(yy - cyx, xx - cyx) % (2*np.pi))
    band = (rr > r - dr) & (rr < r + dr); tb = (th[band] / (2*np.pi) * 72).astype(int); vv = pat[band]
    az = np.array([vv[tb == k].mean() if (tb == k).any() else 0 for k in range(72)])
    return float(az.max() / (az.mean() + 1e-9))

summary = {}
fig = Figure(figsize=(13, 8.5), facecolor="white")
for ri, name in enumerate(["SI3", "SI4"]):
    t0 = time.time()
    zg = np.load(os.path.join(OUT, f"imc_glassorder_{name}.npz"))
    ph = zg["ph"]; scat = zg["scat"]; Ny, Nx = zg["scan"]; H = int(zg["H"])
    za = np.load(os.path.join(OUT, f"imc_alpha_targeted_{name}.npz")); ex_a = za["ex_a"]; ex_c = za["ex_c"]
    ls = np.log(np.clip(scat, 1, None)); foot = ls > threshold_otsu(ls)
    spots = ex_a > (np.nanmean(ex_c) + 2 * np.nanstd(ex_c))
    pops = {"high-p/h NO-spot": (ph > 0.5) & ~spots & foot,
            "spot-crystal": spots & foot,
            "glass (low p/h)": (ph < 0.3) & ~spots & foot}
    cube = open_lazy_cube(IMC[name], scan_shape=(Ny, Nx)); cyx = (H - 1) / 2.0
    beam = max(8, round(0.11 * H)); lo = beam + 1; hi = int(0.85 * (H // 2))
    ra = [1.0 / (d * INV_ANG) for d in ALPHA_D]
    cols = {"high-p/h NO-spot": "#B8860B", "spot-crystal": "#C0392B", "glass (low p/h)": "#2471A3"}
    axp = fig.add_subplot(2, 2, ri * 2 + 2); rec = {}
    for ci, (lbl, mask) in enumerate(pops.items()):
        idxs = patch_from(mask.reshape(Ny, Nx), NPIX)
        if idxs is None or len(idxs) < 30:
            print(f"[{name}] {lbl}: too few pixels ({0 if idxs is None else len(idxs)})", flush=True); continue
        s = psum(cube, idxs, Nx, H)
        m, _, _ = _radial_mean_var(s, (cyx, cyx), beam_px=beam); seg = m[lo:hi]
        halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); pk = np.clip(seg - halo, 0, None)
        rpk = lo + int(np.argmax(pk)); d_pk = 1.0 / (rpk * INV_ANG)
        pmax = pk.max(); ab = np.where(pk > pmax / 2)[0]; fwhm = int(ab.max() - ab.min() + 1) if len(ab) else 0
        gph = float(pk.sum() / (seg.sum() + 1e-9)); ac = azim_contrast(s, cyx, rpk)
        rec[lbl] = dict(n=len(idxs), d_peak=round(d_pk, 2), fwhm_px=fwhm, ph_sum=round(gph, 2), azim_contrast=round(ac, 1))
        axp.plot(np.arange(lo, hi), seg / seg.max(), color=cols[lbl], lw=1.6,
                 label=f"{lbl}: d={d_pk:.1f}Å fwhm={fwhm}px ac={ac:.0f}")
        if ci == 0:
            ax = fig.add_subplot(2, 2, ri * 2 + 1)
            cr = slice(int(cyx) - 150, int(cyx) + 150)
            ax.imshow(np.log1p(np.clip(s[cr, cr], 0, None)), cmap="inferno")
            ax.set_title(f"{name}: high-p/h NO-spot patch sum (n={len(idxs)})\n"
                         f"d@peak={d_pk:.2f}Å fwhm={fwhm}px p/h(sum)={gph:.2f} azim-contrast={ac:.0f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    for r0 in ra:
        if lo < r0 < hi: axp.axvline(r0, color="r", ls=":", lw=0.8)
    axp.set_title(f"{name}: radial profiles (red dots = α rings)", fontsize=10)
    axp.legend(fontsize=7); axp.set_yticks([])
    axp.set_xlabel("radius (px)")
    summary[name] = rec
    print(f"[{name}] " + " | ".join(f"{k}: d={v['d_peak']}Å fwhm={v['fwhm_px']} ph={v['ph_sum']} ac={v['azim_contrast']}" for k, v in rec.items()) + f" ({time.time()-t0:.0f}s)", flush=True)

fig.suptitle("Decisive test: is the high-p/h NO-spot population (old '67%') fine α polycrystal (sharp rings @α) or glass halo (broad)?", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_highph_population_test.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "imc_highph_population_test.png"))
json.dump(summary, open(os.path.join(OUT, "imc_highph_population_test.json"), "w"), indent=2)
print("wrote imc_highph_population_test.png + .json", flush=True)
