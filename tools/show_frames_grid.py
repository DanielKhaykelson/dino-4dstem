"""Grid of SI4 single frames spanning real scattered-intensity (mass-thickness)
percentiles + DINO class, with blob_log thr=0.2 detections overlaid (raw vmax=2)."""
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
DET = dict(threshold=0.25, min_sigma=2.0, max_sigma=8.0, num_sigma=6, overlap=0.4)
PATH = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy"
zg = np.load(os.path.join(OUT, "imc_glassorder_SI4.npz"))
scat = zg["scat"]; asg = zg["assigns"].astype(int); Ny, Nx = zg["scan"]
order = np.argsort(scat)
pcts = [1, 8, 18, 30, 42, 55, 68, 80, 90, 96, 99]            # scattered-intensity percentiles
picks = [int(order[int(p/100*(len(order)-1))]) for p in pcts]
cube = open_lazy_cube(PATH, scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape; cyx = (H-1)/2.0
cr = slice(int(cyx)-150, int(cyx)+150)
fig = Figure(figsize=(17, 7.2), facecolor="white")
for j, i in enumerate(picks):
    rx, ry = divmod(i, Nx)
    raw = np.clip(np.asarray(cube[rx][ry], np.float32), 0, VMAX)
    sm = ndimage.gaussian_filter(raw, BLUR)
    p = np.clip(sm - np.median(sm), 0, None); p = np.log1p(p); p = p/(p.max()+1e-9)
    blobs = blob_log(p, **DET)
    ax = fig.add_subplot(2, 6, j+1)
    ax.imshow(raw[cr, cr], cmap="inferno", vmin=0, vmax=VMAX)
    for b in blobs:
        y, x, s = b
        if cr.start <= y < cr.stop and cr.start <= x < cr.stop:
            ax.add_patch(Circle((x-cr.start, y-cr.start), s*1.41+2, fill=False, color="lime", lw=0.7))
    ax.set_title(f"scat p{pcts[j]} c{asg[i]}\nI={scat[i]:.2e}  {len(blobs)} blobs", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"p{pcts[j]:2d} ({rx},{ry}) c{asg[i]} scat={scat[i]:.3e} blobs={len(blobs)}", flush=True)
fig.suptitle("SI4 frames across scattered-intensity percentiles (raw vmax=2) + blob_log thr=0.25 — lowest p = true vacuum", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(fig)
pp = os.path.join(OUT, "frames_grid_SI4.png"); fig.savefig(pp, dpi=150, facecolor="white")
import shutil; shutil.copy(pp, os.path.join(OUT, "latest_review", "frames_grid_SI4.png"))
print("wrote frames_grid_SI4.png", flush=True)
