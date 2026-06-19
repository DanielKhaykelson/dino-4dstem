"""Regenerate ALL diffraction displays with LINEAR scale, vmin=0 vmax=2
(the GUI / earlier-averages convention), replacing the log1p versions:
  1. si5_amorphous_frames_vmax2 : SI5 glass-class single frames + 3x3 sums
                                  + the two large glass grain-sums (halo rings)
  2. imc_grain_sum_spots_vmax2  : SI3/SI5 crystal vs amorphous grain-sums
  3. imc_highph_pop_vmax2       : SI3/SI4 high-p/h-no-spot vs spot vs glass patch sums
Same selection logic as the originals (deterministic picks)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from data import open_lazy_cube
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
try:
    from skimage.filters import threshold_otsu
except Exception:
    def threshold_otsu(x, nbins=256):
        h, e = np.histogram(x, nbins); c = (e[:-1] + e[1:]) / 2; w1 = np.cumsum(h); w2 = np.cumsum(h[::-1])[::-1]
        m1 = np.cumsum(h * c) / np.clip(w1, 1, None); m2 = (np.cumsum((h * c)[::-1]) / np.clip(w2[::-1], 1, None))[::-1]
        v = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2; return c[:-1][np.argmax(v)]

OUT = "docs/paper/draft_v2/figs"; VMAX = 2.0
PATHS = {
 "SI3": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
 "SI4": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
 "SI5": r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
}
def load(name):
    zg = np.load(os.path.join(OUT, f"imc_glassorder_{name}.npz"))
    za = np.load(os.path.join(OUT, f"imc_alpha_targeted_{name}.npz"))
    return zg, za
def psum(cube, idxs, Nx, H):
    acc = np.zeros((H, H), np.float64); rows = {}
    for i in idxs: rows.setdefault(int(i) // Nx, []).append(int(i) % Nx)
    for rx in sorted(rows):
        blk = np.asarray(cube[rx], np.float32)
        for ry in rows[rx]: acc += blk[ry]
    return acc / max(len(idxs), 1)
def show(ax, img, cyx, title, col="k"):
    cr = slice(int(cyx) - 150, int(cyx) + 150)
    ax.imshow(np.clip(img[cr, cr], 0, None), cmap="inferno", vmin=0, vmax=VMAX)
    ax.set_title(title, fontsize=8, color=col); ax.set_xticks([]); ax.set_yticks([])

# ---------- 1. SI5 amorphous frames ----------
zg, za = load("SI5"); ph = zg["ph"]; scat = zg["scat"]; asg = zg["assigns"].astype(int)
Ny, Nx = zg["scan"]; H = int(zg["H"]); ex_a = za["ex_a"]; ex_c = za["ex_c"]
ls = np.log(np.clip(scat, 1, None)); foot = ls > np.percentile(ls, 40)
spots = ex_a > (np.nanmean(ex_c) + 2 * np.nanstd(ex_c))
far = ~ndimage.binary_dilation(spots.reshape(Ny, Nx), iterations=3).ravel()
cand = np.where(np.isin(asg, [0, 6]) & foot & far)[0]; cand = cand[np.argsort(ex_a[cand])]
picks = []
for i in cand:
    if all(abs(i // Nx - j // Nx) + abs(i % Nx - j % Nx) > 25 for j in picks): picks.append(int(i))
    if len(picks) == 4: break
cube5 = open_lazy_cube(PATHS["SI5"], scan_shape=(Ny, Nx)); cyx = (H - 1) / 2.0
fig = Figure(figsize=(14, 11), facecolor="white")
for ci, i in enumerate(picks):
    rx, ry = divmod(i, Nx); blk = np.asarray(cube5[rx], np.float32)
    show(fig.add_subplot(3, 4, ci + 1), blk[ry], cyx, f"({rx},{ry}) c{asg[i]} SINGLE\np/h={ph[i]:.2f} exA={ex_a[i]:.2f}")
    acc = np.zeros((H, H), np.float64); n = 0
    for dx in (-1, 0, 1):
        if not (0 <= rx + dx < Ny): continue
        b = np.asarray(cube5[rx + dx], np.float32)
        for dy in (-1, 0, 1):
            if 0 <= ry + dy < Nx: acc += b[ry + dy]; n += 1
    show(fig.add_subplot(3, 4, 4 + ci + 1), acc / n, cyx, "3x3 sum")
# bottom row: large glass grain-sums (halo ring visible) + crystal for contrast
asgmap = asg.reshape(Ny, Nx); g_glass, g_cry = [], []
for k in sorted(set(asg)):
    lab, n = ndimage.label(asgmap == k)
    for gi in range(1, n + 1):
        m = (lab == gi).ravel(); sz = int(m.sum())
        if 50 <= sz <= 400:
            d = dict(k=k, idx=np.where(m)[0], n=sz, exa=float(np.nanmean(ex_a[m])))
            (g_glass if k in (0, 6) else g_cry).append(d)
g_glass.sort(key=lambda g: g["exa"]); g_cry.sort(key=lambda g: -g["exa"])
for ci, (g, lbl, col) in enumerate([(g_glass[0], "GLASS grain-sum", "#2471A3"), (g_glass[1], "GLASS grain-sum", "#2471A3"),
                                     (g_cry[0], "CRYSTAL grain-sum", "#C0392B"), (g_cry[1], "CRYSTAL grain-sum", "#C0392B")]):
    s = psum(cube5, g["idx"], Nx, H)
    show(fig.add_subplot(3, 4, 8 + ci + 1), s, cyx, f"{lbl} c{g['k']} n={g['n']} exA={g['exa']:.0f}", col)
fig.suptitle(f"SI5 amorphous (glass classes 0/6) — LINEAR scale vmin=0 vmax={VMAX}\nrows: single frames | 3x3 sums | grain-sums (glass halo ring vs crystal spots)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(fig)
fig.savefig(os.path.join(OUT, "si5_amorphous_frames_vmax2.png"), dpi=150, facecolor="white")
print("wrote si5_amorphous_frames_vmax2.png", flush=True)

# ---------- 2. grain-sum spots SI3/SI5 ----------
fig = Figure(figsize=(15, 7), facecolor="white"); ncol = 5
for ri, name in enumerate(["SI3", "SI5"]):
    zg, za = load(name); ph = zg["ph"]; asg = zg["assigns"].astype(int); Ny, Nx = zg["scan"]; H = int(zg["H"])
    ex_a = za["ex_a"]; asgmap = asg.reshape(Ny, Nx)
    grains = []
    for k in sorted(set(asg)):
        lab, n = ndimage.label(asgmap == k)
        for gi in range(1, n + 1):
            m = (lab == gi).ravel(); sz = int(m.sum())
            if 50 <= sz <= 400:
                grains.append(dict(k=int(k), idx=np.where(m)[0], n=sz,
                                   exa=float(np.nanmean(ex_a[m])), ph=float(np.nanmean(ph[m]))))
    grains.sort(key=lambda g: -g["exa"]); cry = grains[:3]
    amorph = sorted([g for g in grains if g["ph"] > np.nanmedian([x["ph"] for x in grains])], key=lambda g: g["exa"])[:2]
    cube = open_lazy_cube(PATHS[name], scan_shape=(Ny, Nx)); cyx = (H - 1) / 2.0
    for ci, (lbl, g, col) in enumerate([("CRYSTAL", g, "#C0392B") for g in cry] + [("AMORPH", g, "#2471A3") for g in amorph]):
        s = psum(cube, g["idx"], Nx, H)
        show(fig.add_subplot(2, ncol, ri * ncol + ci + 1), s, cyx, f"{name} {lbl} c{g['k']} n={g['n']}\nexA={g['exa']:.0f} p/h={g['ph']:.2f}", col)
fig.suptitle(f"Grain-sums — LINEAR vmin=0 vmax={VMAX}: crystal = discrete spots, amorphous = smooth halo ring", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92]); FigureCanvasAgg(fig)
fig.savefig(os.path.join(OUT, "imc_grain_sum_spots_vmax2.png"), dpi=150, facecolor="white")
print("wrote imc_grain_sum_spots_vmax2.png", flush=True)

# ---------- 3. high-p/h population patches SI3/SI4 ----------
fig = Figure(figsize=(12, 7.6), facecolor="white")
for ri, name in enumerate(["SI3", "SI4"]):
    zg, za = load(name); ph = zg["ph"]; scat = zg["scat"]; Ny, Nx = zg["scan"]; H = int(zg["H"])
    ex_a = za["ex_a"]; ex_c = za["ex_c"]
    ls = np.log(np.clip(scat, 1, None)); foot = ls > threshold_otsu(ls)
    spots = ex_a > (np.nanmean(ex_c) + 2 * np.nanstd(ex_c))
    pops = {"high-p/h NO-spot": (ph > 0.5) & ~spots & foot, "spot-crystal": spots & foot,
            "glass (low p/h)": (ph < 0.3) & ~spots & foot}
    cube = open_lazy_cube(PATHS[name], scan_shape=(Ny, Nx)); cyx = (H - 1) / 2.0
    for ci, (lbl, mask) in enumerate(pops.items()):
        lab, n = ndimage.label(mask.reshape(Ny, Nx))
        if n == 0: continue
        sizes = ndimage.sum(mask.reshape(Ny, Nx), lab, range(1, n + 1)); big = int(np.argmax(sizes)) + 1
        idxs = np.where((lab == big).ravel())[0][:300]
        s = psum(cube, idxs, Nx, H)
        show(fig.add_subplot(2, 3, ri * 3 + ci + 1), s, cyx, f"{name} {lbl}\npatch sum n={len(idxs)}")
fig.suptitle(f"Population patch-sums — LINEAR vmin=0 vmax={VMAX}\nhigh-p/h = fine polycrystal rings | spot-crystal = discrete spots | glass = broad halo", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.9]); FigureCanvasAgg(fig)
fig.savefig(os.path.join(OUT, "imc_highph_pop_vmax2.png"), dpi=150, facecolor="white")
print("wrote imc_highph_pop_vmax2.png", flush=True)

import shutil
for f in ["si5_amorphous_frames_vmax2.png", "imc_grain_sum_spots_vmax2.png", "imc_highph_pop_vmax2.png"]:
    shutil.copy(os.path.join(OUT, f), os.path.join(OUT, "latest_review", f))
print("copied to latest_review", flush=True)
