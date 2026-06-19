"""Is the amorphous halo IMC or the carbon support? For each IMC film, compare the
average diffraction of the LOWEST-scatter frames (off-sample / vacuum-or-film only)
to the SAMPLE frames (high scatter). If the d~4.5A halo appears only on the sample
and the off-sample region is flat (or carries only a different, higher-q feature),
the halo is IMC, not the support. Radial profiles (beam-masked, log) overlaid with
the alpha-IMC ring positions; amorphous-carbon's main halo (d~2.1A) maps to ~257 px
(off-frame here) for reference.

  python tools/halo_origin_check.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")
INV = 0.00185; NY = NX = 128; VMAX = 2.0
IMC = {
 "SI3": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
 "SI4": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
 "SI5": r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
}
ALPHA = {"7.4": 73, "6.0": 90, "4.75": 114, "3.9": 139}

fig = Figure(figsize=(14, 3.4 * 3), facecolor="white")
for ri, (name, path) in enumerate(IMC.items()):
    t0 = time.time()
    z = np.load(os.path.join(FIGS, f"imc_glassorder_{name}.npz")); scat = z["scat"]
    lo_thr = np.percentile(scat, 8)        # off-sample (least scattering)
    hi_thr = np.percentile(scat, 70)       # sample (high scattering)
    low = scat <= lo_thr; high = scat >= hi_thr
    cube = open_lazy_cube(path, scan_shape=(NY, NX)); _, _, H, W = cube.shape
    accL = np.zeros((H, W)); accH = np.zeros((H, W)); nL = nH = 0
    for rx in range(NY):
        blk = np.clip(np.asarray(cube[rx], np.float32), 0, VMAX)
        for ry in range(NX):
            i = rx * NX + ry
            if low[i]: accL += blk[ry]; nL += 1
            elif high[i]: accH += blk[ry]; nH += 1
    mL = accL / max(nL, 1); mH = accH / max(nH, 1)
    cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H))
    rmL, _, _ = _radial_mean_var(mL, (cyx, cyx), beam_px=beam)
    rmH, _, _ = _radial_mean_var(mH, (cyx, cyx), beam_px=beam)
    hi_px = min(int(0.49 * H), 200); rr = np.arange(beam + 1, hi_px)
    cr = slice(int(cyx) - 150, int(cyx) + 150)

    def disp(m):
        p = m[cr, cr]; pc = (p.shape[0] - 1) / 2.0
        yy, xx = np.indices(p.shape); mask = np.sqrt((yy - pc) ** 2 + (xx - pc) ** 2) > beam
        lo, hh = np.percentile(p[mask], 2), np.percentile(p[mask], 99.5)
        return np.log1p(np.clip(p, lo, hh) * mask - lo)
    axA = fig.add_subplot(3, 3, ri * 3 + 1); axA.imshow(disp(mL), cmap="inferno"); axA.set_xticks([]); axA.set_yticks([])
    axA.set_title(f"{name}: off-sample avg (lowest 8% scatter, n={nL})", fontsize=9)
    axA.set_ylabel(name, fontsize=12, fontweight="bold")
    axB = fig.add_subplot(3, 3, ri * 3 + 2); axB.imshow(disp(mH), cmap="inferno"); axB.set_xticks([]); axB.set_yticks([])
    axB.set_title(f"{name}: sample avg (top 30% scatter, n={nH})", fontsize=9)
    axP = fig.add_subplot(3, 3, ri * 3 + 3)
    # normalize each profile by its own mean over the band for shape comparison
    sL = rmL[rr] / (rmL[rr].mean() + 1e-9); sH = rmH[rr] / (rmH[rr].mean() + 1e-9)
    axP.plot(rr, sH, color="#C0392B", lw=1.6, label="sample")
    axP.plot(rr, sL, color="#2E86C1", lw=1.6, label="off-sample")
    for d, px in ALPHA.items():
        if rr[0] <= px <= rr[-1]:
            axP.axvline(px, color="#888", ls=":", lw=0.8); axP.text(px, axP.get_ylim()[1]*0.95, d+"Å", fontsize=6, rotation=90, va="top", ha="right", color="#555")
    axP.set_xlabel("radius (px)"); axP.set_ylabel("norm. radial intensity"); axP.legend(fontsize=8)
    axP.set_title("radial profile (a-C halo d~2.1Å = ~257px, off-frame)", fontsize=8.5)
    # quantify: peak/halo contrast in the 4.5A band for each
    def contrast(rm):
        seg = rm[100:130]  # ~4.75-4.0 A band around the principal ring
        return float(seg.max() / (np.median(rm[rr]) + 1e-9))
    print(f"[{name}] off-sample n={nL} sample n={nH} | 4.5A-band peak/median: off={contrast(rmL):.2f} sample={contrast(rmH):.2f} ({time.time()-t0:.0f}s)", flush=True)
fig.suptitle("Halo origin check: off-sample (low-scatter) vs sample (high-scatter) average diffraction, per IMC film. "
             "If the d~4.5Å ring appears only on the sample, the amorphous halo is IMC, not the carbon support.", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "halo_origin_check.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "halo_origin_check.png"))
print("wrote halo_origin_check.png", flush=True)
