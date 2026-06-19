"""See spots vs rings by summing frames within a single CONNECTED real-space grain.
A connected component of one DINO class is (mostly) one crystallite -> constant
orientation -> summing its low-dose frames adds Bragg spots COHERENTLY (SNR up),
unlike a DINO class average which mixes orientations into a ring.
Pick crystalline grains (high mean ex_a = spottiness) and amorphous grains
(low ex_a, high p/h) per sample; sum frames; show spots (crystal) vs smooth ring
(amorphous), with an azimuthal-contrast number. Reads only the rows each grain uses."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from data import open_lazy_cube
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle

OUT = "docs/paper/draft_v2/figs"
IMC = {
 "SI3": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
 "SI5": r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
}
NAMES = ["SI3", "SI5"]
INV_ANG = 0.00185; ALPHA_D = [7.4, 6.0, 4.75, 3.9]

def grain_sum(cube, idxs, Nx, H):
    acc = np.zeros((H, H), np.float64); rows = {}
    for i in idxs: rows.setdefault(i // Nx, []).append(i % Nx)
    for rx in sorted(rows):
        blk = np.asarray(cube[rx], np.float32)
        for ry in rows[rx]: acc += blk[ry]
    return acc / len(idxs)

def azim_contrast(pat, cyx, r, dr=3):
    H, W = pat.shape; yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2); th = (np.arctan2(yy - cyx, xx - cyx) % (2*np.pi))
    band = (rr > r - dr) & (rr < r + dr); tb = (th[band] / (2*np.pi) * 72).astype(int); vv = pat[band]
    az = np.array([vv[tb == k].mean() if (tb == k).any() else 0 for k in range(72)])
    return az.max() / (az.mean() + 1e-9), az

summary = {}
fig = Figure(figsize=(15, 9), facecolor="white"); ncol = 5
for ri, name in enumerate(NAMES):
    t0 = time.time(); zg = np.load(os.path.join(OUT, f"imc_glassorder_{name}.npz"))
    ph = zg["ph"]; asg = zg["assigns"].astype(int); Ny, Nx = zg["scan"]
    za = np.load(os.path.join(OUT, f"imc_alpha_targeted_{name}.npz")); ex_a = za["ex_a"]
    asgmap = asg.reshape(Ny, Nx)
    # connected components across all classes; per-grain mean ex_a, ph, size
    grains = []
    for k in sorted(set(asg)):
        lab, n = ndimage.label(asgmap == k)
        for gi in range(1, n + 1):
            mask = (lab == gi).ravel(); sz = int(mask.sum())
            if 50 <= sz <= 400:
                grains.append(dict(k=int(k), idx=np.where(mask)[0], n=sz,
                                   exa=float(np.nanmean(ex_a[mask])), ph=float(np.nanmean(ph[mask]))))
    grains.sort(key=lambda g: -g["exa"])
    cryst = grains[:3]                                   # most spotty
    amorph = sorted([g for g in grains if g["ph"] > np.nanmedian([x["ph"] for x in grains])],
                    key=lambda g: g["exa"])[:2]          # low spot, high p/h
    cube = open_lazy_cube(IMC[name], scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape; cyx = (H - 1) / 2.0
    ra = [int(round(1.0 / (d * INV_ANG))) for d in ALPHA_D]
    picks = [("CRYSTAL", g, "#C0392B") for g in cryst] + [("AMORPH", g, "#2471A3") for g in amorph]
    rec = []
    for ci, (lbl, g, col) in enumerate(picks[:ncol]):
        s = grain_sum(cube, g["idx"], Nx, H)
        # contrast at the strongest alpha ring
        cs = [(r, azim_contrast(s, cyx, r)[0]) for r in ra]
        rbest, cbest = max(cs, key=lambda x: x[1])
        rec.append(dict(label=lbl, k=g["k"], n=g["n"], exa=round(g["exa"], 2), ph=round(g["ph"], 2),
                        azim_contrast=round(float(cbest), 2), ring_px=int(rbest)))
        ax = fig.add_subplot(len(NAMES), ncol, ri * ncol + ci + 1)
        cr = slice(int(cyx) - 150, int(cyx) + 150)
        ax.imshow(np.log1p(np.clip(s[cr, cr], 0, None)), cmap="inferno")
        ax.add_patch(Circle((150, 150), rbest, fill=False, color="cyan", lw=0.6, ls=":"))
        ax.set_title(f"{name} {lbl} c{g['k']} (n={g['n']})\nazim-contrast={cbest:.1f}  p/h={g['ph']:.2f} exA={g['exa']:.2f}", fontsize=8, color=col)
        ax.set_xticks([]); ax.set_yticks([])
    summary[name] = rec
    print(f"[{name}] ({time.time()-t0:.0f}s) " + " | ".join(f"{r['label']} c{r['k']}: contrast={r['azim_contrast']} ph={r['ph']} exA={r['exa']}" for r in rec), flush=True)

fig.suptitle("Grain-sum patterns (one connected crystallite, constant orientation): CRYSTAL = discrete spots (high azimuthal contrast)\n"
             "AMORPHOUS = smooth ring (contrast ~1). cyan = strongest α ring radius", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_grain_sum_spots.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "imc_grain_sum_spots.png"))
json.dump(summary, open(os.path.join(OUT, "imc_grain_sum_spots.json"), "w"), indent=2)
print("wrote imc_grain_sum_spots.png + .json", flush=True)
