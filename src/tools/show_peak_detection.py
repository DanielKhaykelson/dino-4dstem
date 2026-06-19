"""Show what blob_log actually detects on single SI4 frames (the agreed chain:
clip vmax=2 -> gaussian blur sigma=2 -> median-sub -> clip -> log1p -> normalize
-> blob_log thr=0.05, sigma 2-8). Pick representative pixels from the saved npz:
high-ratio (crystal/needle), low-ratio glass, vacuum. Show the processed image
the detector sees with detected blobs circled + metrics. Light I/O (few rows)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from skimage.feature import blob_log
from data import open_lazy_cube
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle

OUT = "docs/paper/draft_v2/figs"; VMAX = 2.0; BLUR = 2.0
DET = dict(threshold=0.2, min_sigma=2.0, max_sigma=8.0, num_sigma=6)
PATH = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy"
z = np.load(os.path.join(OUT, "imc_acom_fullpx_SI4.npz"))
Ny, Nx = z["scan"]; ratio = z["ratio"]; halo = z["halo"]; npk = z["n_peaks"]; corr = z["corr"]
samp = halo >= 0.02
idx = np.arange(Ny * Nx)
hi = idx[samp][np.argsort(-ratio[samp])][:2]                       # crystal (needle)
glass = idx[samp & (ratio < np.nanpercentile(ratio[samp], 25))]
glass = glass[np.argsort(halo[glass])][-1:]                        # thick glass (high halo, low ratio)
vac = idx[~samp][np.argsort(halo[~samp])][-1:]                     # vacuum
picks = list(hi) + list(glass) + list(vac)
labels = ["CRYSTAL (top ratio)", "CRYSTAL (top ratio)", "GLASS (low ratio)", "VACUUM"]

cube = open_lazy_cube(PATH, scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
cyx = (H - 1) / 2.0
fig = Figure(figsize=(16, 8), facecolor="white")
for ci, (i, lbl) in enumerate(zip(picks, labels)):
    rx, ry = divmod(int(i), Nx)
    raw = np.clip(np.asarray(cube[rx][ry], np.float32), 0, VMAX)
    sm = ndimage.gaussian_filter(raw, BLUR)
    p = np.clip(sm - np.median(sm), 0, None); p = np.log1p(p); p = p / (p.max() + 1e-9)
    blobs = blob_log(p, min_sigma=DET["min_sigma"], max_sigma=DET["max_sigma"],
                     num_sigma=DET["num_sigma"], threshold=DET["threshold"], overlap=0.4)
    # absolute intensity of each blob on the blurred vmax=2 frame (counts, in [0,2])
    amps = np.array([sm[int(round(y)), int(round(x))] for y, x, s in blobs]) if len(blobs) else np.array([])
    FLOOR = 0.5                       # trial intensity floor (counts on vmax=2 scale)
    keep = amps >= FLOOR
    cr = slice(int(cyx) - 150, int(cyx) + 150)
    ax = fig.add_subplot(2, 4, ci + 1)
    ax.imshow(raw[cr, cr], cmap="inferno", vmin=0, vmax=VMAX)
    ax.set_title(f"{lbl}\nraw clip vmax=2  ({rx},{ry})", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(2, 4, 4 + ci + 1)
    ax.imshow(sm[cr, cr], cmap="gray", vmin=0, vmax=VMAX)
    for b, a, k in zip(blobs, amps, keep):
        y, x, s = b
        if cr.start <= y < cr.stop and cr.start <= x < cr.stop:
            ax.add_patch(Circle((x - cr.start, y - cr.start), s * 1.41 + 2, fill=False,
                                color=("lime" if k else "red"), lw=0.7))
    pct = np.percentile(amps, [50, 90]) if amps.size else [0, 0]
    ax.set_title(f"{int(keep.sum())} kept / {len(blobs)} (floor {FLOOR})\nblob I: med={pct[0]:.2f} p90={pct[1]:.2f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"{lbl} ({rx},{ry}): blobs={len(blobs)} kept@{FLOOR}={int(keep.sum())} "
          f"blobI med={pct[0]:.2f} p90={pct[1]:.2f} halo={halo[i]:.3f}", flush=True)
fig.suptitle("SI4 blob_log thr=0.1 + intensity screen (green kept I>=0.5, red rejected) on blurred vmax=2 frame", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
p2 = os.path.join(OUT, "peak_detection_SI4_iscreen.png"); fig.savefig(p2, dpi=150, facecolor="white")
import shutil; shutil.copy(p2, os.path.join(OUT, "latest_review", "peak_detection_SI4_iscreen.png"))
print("wrote peak_detection_SI4_iscreen.png", flush=True)
