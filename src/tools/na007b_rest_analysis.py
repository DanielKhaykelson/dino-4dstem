"""Na007b NON-LINE (rest) analysis: does DINO over/under-cluster vs NMF in the
rest of the flake, and does DINO add real structural info there?
- HAADF computed GUI-style (annulus 0.18-0.45 of frame, linear gray).
- DINO rest classes (excl. bg=0 and Line=1,8) vs polar+theta-shift NMF rest
  clusters (K=6, Line cluster = max overlap with DINO Line).
- Per-cluster: real avg diffraction, median crystallinity (peak/halo), mean
  HAADF (thickness proxy), spatial coherence (4-neighbour agreement).
- Distinctness (beam-masked ring r) within each method + cross-tab.
- Figure: 2 rows (DINO / NMF) of remaining clusters overlaid on HAADF."""
import os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.cm as cmx
from data import open_lazy_cube
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

F = r"docs/explainer/figs"; CUBE = r"D:/DINOSR/data/Na007b_nbed.cube.npy"
RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
Ny, Nx = 126, 100; N = Ny * Nx

asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int).reshape(Ny, Nx)
flake = (asg != 0); dino_line_mask = np.isin(asg, [1, 8]) & flake
ph = np.load(f"{F}/na007b_cryst.npz.npy")[0].reshape(Ny, Nx)
W = np.load(f"{F}/na007b_polar_W.npy")
KNMF = 6
nmf = KMeans(KNMF, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(W)).reshape(Ny, Nx)

# ---- cube pass: HAADF (GUI) + per-class / per-cluster real-diffraction sums ----
cube = open_lazy_cube(CUBE, scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
yy, xx = np.indices((H, Wd)); cy = (H - 1) / 2; rr = np.sqrt((yy - cy) ** 2 + (xx - cy) ** 2)
ham = ((rr >= 0.18 * H) & (rr <= 0.45 * H)).astype(np.float32)        # GUI HAADF annulus
HA = np.zeros((Ny, Nx))
dcls = sorted(int(c) for c in np.unique(asg[flake]))
dsum = {c: np.zeros((H, Wd)) for c in dcls}; dcnt = {c: 0 for c in dcls}
nsum = {c: np.zeros((H, Wd)) for c in range(KNMF)}; ncnt = {c: 0 for c in range(KNMF)}
print("cube pass...", flush=True); t0 = time.time()
for rx in range(Ny):
    blk = np.asarray(cube[rx], np.float32)
    HA[rx] = (blk * ham).sum(axis=(1, 2))
    for ry in range(Nx):
        if not flake[rx, ry]: continue
        pat = blk[ry]; c = asg[rx, ry]; dsum[c] += pat; dcnt[c] += 1
        nc = nmf[rx, ry]; nsum[nc] += pat; ncnt[nc] += 1
    if rx % 30 == 0: print(f"  row {rx} ({time.time()-t0:.0f}s)", flush=True)
np.save(f"{F}/na007b_HAADF_gui.npy", HA)

# ---- identify NMF Line cluster, define rest sets ----
ov = [((nmf == c) & dino_line_mask).sum() / max((nmf == c).sum(), 1) for c in range(KNMF)]
nmf_line_c = int(np.argmax(ov))
dino_rest = [c for c in dcls if c not in (0, 1, 8) and dcnt[c] > 30]
nmf_rest = [c for c in range(KNMF) if c != nmf_line_c and ncnt[c] > 30]
davg = {c: dsum[c] / max(dcnt[c], 1) for c in dcls}
navg = {c: nsum[c] / max(ncnt[c], 1) for c in range(KNMF)}
print(f"DINO rest classes={dino_rest} ({len(dino_rest)}); NMF rest clusters={nmf_rest} "
      f"({len(nmf_rest)}); NMF Line cluster={nmf_line_c}", flush=True)

# ---- distinctness (beam-masked ring corr) ----
nb = H // 2; ring = (rr > 0.18 * nb) & (rr < 0.95 * nb)
def disp(x): x = np.log1p(np.clip(x, 0, None)); v = x[ring]; return (x - v.min()) / (v.ptp() + 1e-9)
def rc(a, b): return float(np.corrcoef(disp(a)[ring], disp(b)[ring])[0, 1])
def minpair(avgd, keys):
    ps = [rc(avgd[a], avgd[b]) for a, b in itertools.combinations(keys, 2)]
    return (min(ps), np.median(ps), max(ps)) if ps else (1, 1, 1)
dmin = minpair(davg, dino_rest); nmin = minpair(navg, nmf_rest)
print(f"[Test1] DINO rest mutual r (min/med/max) = {dmin[0]:.3f}/{dmin[1]:.3f}/{dmin[2]:.3f}", flush=True)
print(f"        NMF  rest mutual r (min/med/max) = {nmin[0]:.3f}/{nmin[1]:.3f}/{nmin[2]:.3f}", flush=True)

# ---- spatial coherence (4-neighbour same-label fraction) + stats ----
def coherence(m):
    nbf = np.zeros_like(m, float); tot = 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nbf += np.roll(m, (dy, dx), (0, 1)).astype(float); tot += 1
    return float((nbf[m] / tot).mean()) if m.any() else 0.0
def stats(masks):
    return [(c, int(m.sum()), round(float(np.nanmedian(ph[m])), 3), round(float(np.nanmean(HA[m])), 1), round(coherence(m), 2)) for c, m in masks]
dmasks = [(c, (asg == c) & flake) for c in dino_rest]
nmasks = [(c, (nmf == c) & flake & ~dino_line_mask) for c in nmf_rest]
print("[stats] DINO rest (cls, n, p/h, HAADF, coher):", stats(dmasks), flush=True)
print("[stats] NMF  rest (cls, n, p/h, HAADF, coher):", stats(nmasks), flush=True)
mean_coh_d = np.mean([coherence(m) for _, m in dmasks]); mean_coh_n = np.mean([coherence(m) for _, m in nmasks])
print(f"[coherence] mean DINO rest = {mean_coh_d:.2f}; mean NMF rest = {mean_coh_n:.2f}", flush=True)

# ---- cross-tab: which DINO rest classes fall in each NMF rest cluster ----
print("[Test2] DINO rest classes inside each NMF rest cluster:", flush=True)
rmat = {}
for a, b in itertools.combinations(dino_rest, 2): rmat[(a, b)] = rmat[(b, a)] = rc(davg[a], davg[b])
for c, m in nmasks:
    tot = m.sum(); pres = [d for d in dino_rest if ((asg == d) & m).sum() > 0.05 * tot]
    if len(pres) >= 2:
        mr = min(rmat[(a, b)] for a, b in itertools.combinations(pres, 2))
        print(f"  NMF rest {c} (n={tot}) -> DINO {pres} min r={mr:.3f} {'MERGES distinct' if mr < 0.9 else 'similar'}", flush=True)
    else:
        print(f"  NMF rest {c} (n={tot}) -> DINO {pres}", flush=True)

# ---- figure: 2 rows (DINO / NMF) of remaining clusters on HAADF ----
lo, hi = np.percentile(HA, 1), np.percentile(HA, 99); bg = np.clip(HA, lo, hi)
HL = np.array([0.95, 0.15, 0.15])    # single-cluster highlight = red
def draw_one(ax, mask, title):
    ax.imshow(bg, cmap="gray", aspect="equal", interpolation="nearest")
    o = np.zeros((Ny, Nx, 4), np.float32); o[mask, :3] = HL; o[mask, 3] = 0.6
    ax.imshow(o, aspect="equal", interpolation="nearest"); ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
def draw_combined(ax, masks, title):
    ax.imshow(bg, cmap="gray", aspect="equal", interpolation="nearest")
    o = np.zeros((Ny, Nx, 4), np.float32); cols = cmx.tab20(np.linspace(0, 1, 20))
    for i, (c, m) in enumerate(masks): o[m, :3] = cols[i % 20][:3]; o[m, 3] = 0.6
    ax.imshow(o, aspect="equal", interpolation="nearest"); ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
ncol = 1 + max(len(dmasks), len(nmasks))
fig = Figure(figsize=(2.05 * ncol, 4.5), facecolor="white")
def row(ri, masks, label):
    ax = fig.add_subplot(2, ncol, ri * ncol + 1); draw_combined(ax, masks, f"{label}: all rest ({len(masks)})")
    ax.set_ylabel(label, fontsize=11, fontweight="bold")
    for j, (c, m) in enumerate(masks):
        ax = fig.add_subplot(2, ncol, ri * ncol + 2 + j)
        draw_one(ax, m, f"{label[0]}{c}  n={int(m.sum())}\np/h={np.nanmedian(ph[m]):.2f}")
row(0, dmasks, "DINO")
row(1, nmasks, "NMF")
fig.suptitle(f"Na007b non-Line (rest) clusters on HAADF — DINO {len(dmasks)} vs NMF {len(nmasks)} clusters", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
fig.savefig(f"{F}/na007b_rest_clusters_haadf.png", dpi=150, facecolor="white")
print("wrote na007b_rest_clusters_haadf.png + na007b_HAADF_gui.npy  DONE", flush=True)
