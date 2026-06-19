"""Show raw diffraction from SI5 regions claimed AMORPHOUS (glass classes, low
spot signal): 4 locations x [single frame | 3x3-neighborhood sum], log scale.
Locations drawn from glass classes (0,6) inside footprint, lowest ex_a, far from
any spot detection."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from data import open_lazy_cube
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

OUT = "docs/paper/draft_v2/figs"
zg = np.load(os.path.join(OUT, "imc_glassorder_SI5.npz"))
ph = zg["ph"]; scat = zg["scat"]; asg = zg["assigns"].astype(int); Ny, Nx = zg["scan"]; H = int(zg["H"])
za = np.load(os.path.join(OUT, "imc_alpha_targeted_SI5.npz")); ex_a = za["ex_a"]; ex_c = za["ex_c"]
ls = np.log(np.clip(scat, 1, None)); foot = ls > np.percentile(ls, 40)
spots = ex_a > (np.nanmean(ex_c) + 2 * np.nanstd(ex_c))
far = ~ndimage.binary_dilation(spots.reshape(Ny, Nx), iterations=3).ravel()   # >=3 px from any spot detection
cand = np.where(np.isin(asg, [0, 6]) & foot & far)[0]
cand = cand[np.argsort(ex_a[cand])]
# spread picks across the map
picks = []
for i in cand:
    if all(abs(i // Nx - j // Nx) + abs(i % Nx - j % Nx) > 25 for j in picks): picks.append(int(i))
    if len(picks) == 4: break
cube = open_lazy_cube(r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy", scan_shape=(Ny, Nx))
cyx = (H - 1) / 2.0; cr = slice(int(cyx) - 150, int(cyx) + 150)
fig = Figure(figsize=(14, 7.6), facecolor="white")
for ci, i in enumerate(picks):
    rx, ry = divmod(i, Nx)
    blk = np.asarray(cube[rx], np.float32); single = blk[ry]
    acc = np.zeros((H, H), np.float64); n = 0
    for dx in (-1, 0, 1):
        if not (0 <= rx + dx < Ny): continue
        b = np.asarray(cube[rx + dx], np.float32)
        for dy in (-1, 0, 1):
            if 0 <= ry + dy < Nx: acc += b[ry + dy]; n += 1
    ax = fig.add_subplot(2, 4, ci + 1)
    ax.imshow(np.log1p(np.clip(single[cr, cr], 0, None)), cmap="inferno")
    ax.set_title(f"({rx},{ry}) class {asg[i]} SINGLE frame\np/h={ph[i]:.2f} exA={ex_a[i]:.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(2, 4, 4 + ci + 1)
    ax.imshow(np.log1p(np.clip((acc / n)[cr, cr], 0, None)), cmap="inferno")
    ax.set_title("3x3 neighborhood sum", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    print(f"pick ({rx},{ry}) class={asg[i]} ph={ph[i]:.2f} exA={ex_a[i]:.2f}", flush=True)
fig.suptitle("SI5 — diffraction from regions claimed AMORPHOUS (glass classes 0/6, no spot detection nearby)\n"
             "top: single frames · bottom: 3x3 sums — expect smooth broad halo, no discrete Bragg spots", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "si5_amorphous_frames.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "si5_amorphous_frames.png"))
print("wrote si5_amorphous_frames.png", flush=True)
