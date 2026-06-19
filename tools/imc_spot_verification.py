"""Verify 'discrete spots = crystal, smooth ring = amorphous' on REAL single frames.
Per-pixel (single-frame) signals already computed:
  ex_a (imc_alpha_targeted) = shot-noise-corrected azimuthal-variance excess at the
        alpha radii  = SPOTTINESS  -> high = discrete Bragg spots = CRYSTAL.
  ph   (imc_glassorder)     = radial peak/halo (smooth ring strength) -> high+low-spot
        = smooth ring = AMORPHOUS halo (NOT crystal).
Pick representative single frames for: (A) high-spot (crystal), (B) high-ph/low-spot
(smooth ring = amorphous), (C) thin, and show the patterns with detected spots marked.
Reads only the rows that contain the chosen frames (light I/O)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

OUT = "docs/paper/draft_v2/figs"; NAMES = ["SI3", "SI4", "SI5"]
IMC = {
 "SI3": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
 "SI4": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
 "SI5": r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
}
INV_ANG = 0.00185; ALPHA_D = [7.4, 6.0, 4.75, 3.9]

def detect_spots(frame, cyx, lo, hi):
    """subtract azimuthal-median background; return spot mask (residual > 5*MAD) in annulus."""
    H, W = frame.shape; yy, xx = np.indices((H, W)); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
    ri = rr.astype(int); bg = np.zeros_like(frame)
    prof = np.array([frame[ri == r].mean() if (ri == r).any() else 0 for r in range(int(rr.max()) + 1)])
    bg = prof[np.clip(ri, 0, len(prof) - 1)]
    res = frame - bg; ann = (rr > lo) & (rr < hi)
    mad = np.median(np.abs(res[ann] - np.median(res[ann]))) + 1e-6
    spots = (res > np.median(res[ann]) + 6 * mad) & ann
    return spots, res

summary = {}
fig = Figure(figsize=(15, 11), facecolor="white")
ncol = 5
for ri, name in enumerate(NAMES):
    zg = np.load(os.path.join(OUT, f"imc_glassorder_{name}.npz")); ph = zg["ph"]; scat = zg["scat"]; Ny, Nx = zg["scan"]
    za = np.load(os.path.join(OUT, f"imc_alpha_targeted_{name}.npz")); ex_a = za["ex_a"]; ex_c = za["ex_c"]
    foot = np.log(np.clip(scat, 1, None)) > np.percentile(np.log(np.clip(scat, 1, None)), 35)
    cthr = np.nanmean(ex_c) + 2 * np.nanstd(ex_c)
    spotty = ex_a > cthr
    # categories (within footprint)
    idx = np.arange(Ny * Nx)
    A = idx[foot & spotty]; A = A[np.argsort(-ex_a[A])][:2]                       # crystal: top spottiness
    sm = foot & ~spotty & (ph > np.nanpercentile(ph[foot], 80))
    B = idx[sm]; B = B[np.argsort(-ph[B])][:2]                                    # smooth ring (high ph, low spot)
    C = idx[foot & ~spotty & (ph < np.nanpercentile(ph[foot], 40))]
    C = C[np.argsort(scat[C])][-1:] if len(C) else np.array([], int)             # thin/featureless sample
    picks = [("CRYSTAL\n(high spot)", A, "#C0392B"), ("AMORPHOUS\n(smooth ring, high p/h)", B, "#2471A3"),
             ("thin sample", C, "#7F8C8D")]
    rows = [(lbl, i, col) for lbl, arr, col in picks for i in arr][:ncol]
    cube = open_lazy_cube(IMC[name], scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
    cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = int(0.55 * H)
    summary[name] = dict(spot_frac_footprint=round(float((spotty & foot).sum() / foot.sum()), 3),
                         picks={lbl.split(chr(10))[0]: [int(i) for i in arr] for lbl, arr, _ in picks})
    for ci, (lbl, i, col) in enumerate(rows):
        rx, ry = divmod(int(i), Nx); frame = np.asarray(cube[rx][ry], np.float32)
        spots, res = detect_spots(frame, cyx, lo, hi)
        ax = fig.add_subplot(len(NAMES), ncol, ri * ncol + ci + 1)
        cr = slice(int(cyx) - 150, int(cyx) + 150)
        ax.imshow(np.log1p(np.clip(frame[cr, cr], 0, None)), cmap="inferno")
        ys, xs = np.where(spots[cr, cr]); ax.scatter(xs, ys, s=60, facecolors="none", edgecolors="lime", lw=1.0)
        ax.set_title(f"{name} {lbl}\nspots={int(spots.sum())} p/h={ph[i]:.2f} exA={ex_a[i]:.2f}", fontsize=8, color=col)
        ax.set_xticks([]); ax.set_yticks([])
    print(f"[{name}] spot(crystal) frac of footprint = {summary[name]['spot_frac_footprint']*100:.0f}% "
          f"| picks A(crystal)={list(A)} B(smooth/amorph)={list(B)}", flush=True)

fig.suptitle("Single-frame verification: discrete spots (green circles) = CRYSTAL  vs  smooth ring = AMORPHOUS halo\n"
             "spot detection must be per single frame (grain averages smear orientations into rings)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_spot_verification.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "imc_spot_verification.png"))
json.dump(summary, open(os.path.join(OUT, "imc_spot_verification.json"), "w"), indent=2)
print("wrote imc_spot_verification.png + .json", flush=True)
