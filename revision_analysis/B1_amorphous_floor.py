import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.chdir("D:/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive"); sys.path.insert(0, "src")
import numpy as np, h5py
from scipy.ndimage import gaussian_filter
from gui_app.crystallinity_panel import _radial_mean_var
import matplotlib; matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.colors import PowerNorm

HDF = r"X:/ThinFilms/IMC/4DSTEM/Themis/Presentations/images/IMC50nm_acq2_AwayFromCrystal.hdf5"
with h5py.File(HDF, "r") as h:
    P = np.array(h["DATA"], float)
P = P[:128]                       # EMPAD: strip the 2 appended metadata rows (128x128 diffraction)
P = P - P.min()                   # shift background-subtracted data to non-negative for display/analysis
cy, cx = np.unravel_index(np.argmax(gaussian_filter(P, 2)), P.shape)
beam = max(6, round(0.11 * min(P.shape))); lo = beam + 1
hi = min(cy, cx, P.shape[0] - cy, P.shape[1] - cx) - 2
m, v, _ = _radial_mean_var(P, (cy, cx), beam_px=beam)
cv = np.sqrt(np.clip(v[lo:hi], 0, None)) / np.clip(m[lo:hi], 1e-9, None)
floor = np.percentile(cv, 90)
print(f"clean amorphous pattern {P.shape} center ({cy},{cx}) window r=[{lo},{hi}]")
print(f"amorphous spottiness FLOOR (p90 azimuthal CV) = {floor:.3f}  (V floor {floor**2:.3f})")

PREC = {"SI3": 0.24, "SI4": 0.53, "SI5": 0.10}
for k, val in PREC.items():
    print(f"  precursor {k}: {val:.2f}  = {val/floor:.0f}x floor")

fig = Figure(figsize=(13.5, 4.6), facecolor="white")
gs = fig.add_gridspec(1, 3, left=0.045, right=0.985, top=0.84, bottom=0.16, wspace=0.32)
# panel 1: amorphous pattern, sqrt stretch, halo visible
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(P, cmap="inferno", norm=PowerNorm(0.5, vmin=0, vmax=np.percentile(P, 99.5)))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("as-deposited IMC (away from crystal)\namorphous mean diffraction", fontsize=10, fontweight="bold")
# panel 2: azimuthal CV profile
ax = fig.add_subplot(gs[0, 1])
r = np.arange(lo, hi)
ax.plot(r, cv, "-", color="#6a51a3", lw=2)
ax.axhline(floor, color="k", ls="--", lw=1, label=f"p90 = {floor:.3f} (floor)")
ax.set_xlabel("radius (px)", fontsize=10); ax.set_ylabel("azimuthal CV per ring", fontsize=10)
ax.set_ylim(0, max(0.3, floor*2))
ax.set_title("azimuthally isotropic halo\n(CV low; rise at large r = outer-ring shot noise)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
# panel 3: floor vs precursor
ax = fig.add_subplot(gs[0, 2])
labs = ["amorphous\nfloor", "SI3", "SI4", "SI5"]; vals = [floor] + list(PREC.values())
ax.bar(range(4), vals, color=["#9e9ac8", "#7b3294", "#008837", "#e66101"], edgecolor="k")
ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=9); ax.set_ylabel("class-median spottiness", fontsize=10)
ax.axhline(floor, color="k", ls="--", lw=1)
ax.set_title("SI5 precursor at the floor;\nSI3/SI4 measurably above", fontsize=10, fontweight="bold")
fig.suptitle("B1  As-deposited amorphous IMC: a smooth isotropic halo (no Bragg spots)", fontsize=12, fontweight="bold")
p = "docs/paper/draft_v2/figs/Review/B1_amorphous_floor.png"; fig.savefig(p, dpi=150, facecolor="white"); print("wrote", p)
