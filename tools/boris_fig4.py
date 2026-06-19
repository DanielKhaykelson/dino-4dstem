"""Boris Fig 4: crystallinity heatmaps, DINO vs NMF, on ONE shared brown->yellow
scale across all three IMC samples (same colour = same crystallinity everywhere).
For every DINO class and every NMF (k-means) class we average its diffraction over
the cube, measure the 2D Bragg excess B (counts spots + sharp rings) on the
class average, and paint each pixel by its class's B. DINO gives a smooth
crystallinity gradient; NMF gives a few discrete B levels (it separates peak vs
halo, not a continuum). Also caches class-average patterns for the per-class
meaning SI figure.  vmax=2, FOV-clipped.
  python tools/boris_fig4.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits")
INV = 0.00185; KMAX = 0.35; VMAX = 2.0; NY = NX = 128
FOV = {"SI3": 187, "SI4": 160, "SI5": 160}
ROLE = {"SI3": "overview", "SI4": "needles", "SI5": "interface"}
PATH = {"SI3": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
        "SI4": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
        "SI5": r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"}
NAMES = ["SI3", "SI4", "SI5"]


def bragg(avg, cyx, beam, lo, hi):
    m, _, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam); seg = m[lo:hi]
    if seg.size < 5 or seg.sum() <= 0: return 0.0
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
    yy, xx = np.indices(avg.shape); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
    hf = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1]); band = (rr >= lo) & (rr <= hi)
    return float(np.clip(avg[band] - hf[band], 0, None).sum() / (hf[band].sum() + 1e-9))


def class_B(cube, lab2d, H, beam, lo, hi):
    cyx = (H - 1) / 2.0; ids = sorted(np.unique(lab2d).tolist())
    csum = {c: np.zeros((H, H), np.float32) for c in ids}; ccnt = {c: 0 for c in ids}
    flat = lab2d.ravel()
    for rx in range(NY):
        blk = np.clip(np.asarray(cube[rx], np.float32), 0, VMAX)
        for ry in range(NX):
            c = int(flat[rx * NX + ry]); csum[c] += blk[ry]; ccnt[c] += 1
    Bval = {c: bragg(csum[c] / max(ccnt[c], 1), cyx, beam, lo, hi) for c in ids}
    avg = {c: csum[c] / max(ccnt[c], 1) for c in ids}
    return Bval, avg


DATA = {}; allB = []
for n in NAMES:
    t0 = time.time(); z = np.load(os.path.join(FIGS, f"boris_nmf_cache_{n}.npz"))
    cube = open_lazy_cube(PATH[n], scan_shape=(NY, NX)); _, _, H, _ = cube.shape
    beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV[n])
    Bd, avgd = class_B(cube, z["dino"], H, beam, lo, hi)
    Bn, avgn = class_B(cube, z["kmeans"], H, beam, lo, hi)
    DATA[n] = dict(dino=z["dino"], kmeans=z["kmeans"], Bd=Bd, Bn=Bn, avgd=avgd, avgn=avgn, H=H)
    allB += list(Bd.values()) + list(Bn.values())
    # cache class averages + B for the per-class-meaning SI figure
    np.savez(os.path.join(FIGS, f"boris_classavg_{n}.npz"),
             dino=z["dino"], kmeans=z["kmeans"], H=H,
             **{f"dino_avg_{c}": avgd[c] for c in avgd}, **{f"dino_B_{c}": Bd[c] for c in Bd},
             **{f"nmf_avg_{c}": avgn[c] for c in avgn}, **{f"nmf_B_{c}": Bn[c] for c in Bn})
    print(f"[{n}] DINO B range {min(Bd.values()):.2f}-{max(Bd.values()):.2f} | NMF B range {min(Bn.values()):.2f}-{max(Bn.values()):.2f} ({time.time()-t0:.0f}s)", flush=True)

vmin, vmax = np.percentile(allB, [5, 95])
cmap = mpl.cm.get_cmap("YlOrBr_r").copy()  # dark brown = low (less ordered), pale yellow = high? invert below


def paint(lab2d, Bmap):
    return np.vectorize(lambda c: Bmap[int(c)])(lab2d).astype(float)


fig = Figure(figsize=(8.6, 3.1 * 3 + 0.5), facecolor="white")
cmap = mpl.cm.get_cmap("YlOrBr").copy()   # YlOrBr: light=low, brown=high; we want brown=less-ordered -> reverse
cmap = mpl.cm.get_cmap("YlOrBr_r").copy()
for ri, n in enumerate(NAMES):
    d = DATA[n]
    for ci, (lab, Bm, title) in enumerate([(d["dino"], d["Bd"], "DINO"), (d["kmeans"], d["Bn"], "NMF (k-means)")]):
        ax = fig.add_subplot(3, 2, ri * 2 + ci + 1); ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(paint(lab, Bm), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if ri == 0: ax.set_title(title, fontsize=11, fontweight="bold")
        if ci == 0: ax.set_ylabel(f"{n}\n{ROLE[n]}", fontsize=11, fontweight="bold", rotation=0, labelpad=20, va="center")
fig.subplots_adjust(right=0.88)
cax = fig.add_axes([0.90, 0.15, 0.02, 0.7]); fig.colorbar(im, cax=cax, label="crystallinity (2D Bragg excess B)")
fig.suptitle("Crystallinity on one shared scale (brown = less ordered, yellow = more crystalline), DINO vs NMF.\n"
             "Each class is coloured by the Bragg excess of its average diffraction. DINO classes form a smooth crystallinity gradient; "
             "NMF lumps into a few discrete levels (it separates peak-vs-halo, not a continuum).", fontsize=9.5)
fig.tight_layout(rect=[0, 0, 0.88, 0.93]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "fig4_crystallinity_dino_vs_nmf.png"); fig.savefig(p, dpi=170, facecolor="white")
print(f"wrote {p}  (B scale {vmin:.2f}-{vmax:.2f})", flush=True)
