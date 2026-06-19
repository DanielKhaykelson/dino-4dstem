"""Why does DINO class 3 bleed into the Line (classes 1,8)? Test the boundary/
overlap hypothesis: (a) is class 3 enriched at the Line edge vs other rest
classes? (b) do class-3 pixels ADJACENT to the Line look more Line-like (oriented)
than interior class-3 pixels? Also centroid-cosine of class 3 to the Line."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import distance_transform_edt
from data import open_lazy_cube
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
F = r"docs/explainer/figs"; CUBE = r"D:/DINOSR/data/Na007b_nbed.cube.npy"; RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
Ny, Nx = 126, 100
inf = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)
asg = inf["assigns"].astype(int).reshape(Ny, Nx); embeds = inf["embeds"].astype(np.float32)
line = np.isin(asg, [1, 8]); flake = (asg != 0)
dist = distance_transform_edt(~line)                       # px-distance to nearest Line pixel
rest = [c for c in range(int(asg.max()) + 1) if c not in (0, 1, 8) and (asg == c).sum() > 30]
print("fraction of each rest class within 1.5 px of the Line (edge enrichment):", flush=True)
for c in rest:
    m = asg == c
    print(f"  class {c}: adj-frac={ (dist[m] <= 1.5).mean():.2f}  median-dist={np.median(dist[m]):.1f}px  n={m.sum()}", flush=True)
c3 = asg == 3; c3_adj = c3 & (dist <= 1.5); c3_int = c3 & (dist >= 4)
print(f"class 3: {c3.sum()} px; edge(<=1.5px)={c3_adj.sum()}; interior(>=4px)={c3_int.sum()}", flush=True)

# centroid cosine of class 3 to Line classes vs 2/5
cen = {c: embeds[asg.ravel() == c].mean(0) for c in [1, 8, 2, 3, 5]}
cn = {c: cen[c] / (np.linalg.norm(cen[c]) + 1e-9) for c in cen}
print("centroid cosine to Line classes:", flush=True)
for c in [2, 3, 5]:
    print(f"  class {c}: cos(.,1)={cn[c]@cn[1]:+.2f}  cos(.,8)={cn[c]@cn[8]:+.2f}", flush=True)

# one cube pass: avg diffraction for c3_adj, c3_int, Line
cube = open_lazy_cube(CUBE, scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
acc = {k: np.zeros((H, Wd)) for k in ["c3_adj", "c3_int", "line"]}; cnt = {k: 0 for k in acc}
masks = {"c3_adj": c3_adj, "c3_int": c3_int, "line": line}
for rx in range(Ny):
    blk = np.asarray(cube[rx], np.float32)
    for ry in range(Nx):
        for k, mk in masks.items():
            if mk[rx, ry]: acc[k] += blk[ry]; cnt[k] += 1
mean = {k: acc[k] / max(cnt[k], 1) for k in acc}
cy = (H - 1) / 2; yy, xx = np.indices((H, H)); rr = np.sqrt((yy - cy) ** 2 + (xx - cy) ** 2)
ri = rr.astype(int); th = (np.arctan2(yy - cy, xx - cy) % (2 * np.pi)); beam = int(0.11 * H)
def radial(m): tb = np.bincount(ri.ravel(), m.ravel()); n = np.bincount(ri.ravel()); return tb / np.clip(n, 1, None)
r0 = beam + int(np.argmax(radial(mean["line"])[beam:H//2]))
band = (rr > r0 - 4) & (rr < r0 + 4); nb = 72; tb = (th[band] / (2*np.pi) * nb).astype(int)
def aniso(m):
    v = m[band]; a = np.array([v[tb == k].mean() if (tb == k).any() else 0 for k in range(nb)]); return a.std()/(a.mean()+1e-9)
print(f"azimuthal anisotropy @r0={r0}: c3_edge={aniso(mean['c3_adj']):.2f}  c3_interior={aniso(mean['c3_int']):.2f}  Line={aniso(mean['line']):.2f}", flush=True)
ring = (rr > 0.18 * (H//2)) & (rr < 0.95 * (H//2))
def disp(x): x = np.log1p(np.clip(x, 0, None)); v = x[ring]; return (x - v.min())/(v.ptp()+1e-9)
def rc(a, b): return float(np.corrcoef(disp(a)[ring], disp(b)[ring])[0, 1])
print(f"ring-corr to Line: c3_edge={rc(mean['c3_adj'],mean['line']):.2f}  c3_interior={rc(mean['c3_int'],mean['line']):.2f}", flush=True)

def cr(m): return m[H//2-130:H//2+130, H//2-130:H//2+130]
fig = Figure(figsize=(13, 3.6), facecolor="white")
for i, (k, t) in enumerate([("c3_adj", "class 3 @ Line edge (<=1.5px)"), ("c3_int", "class 3 interior (>=4px)"), ("line", "Line (1,8)")]):
    ax = fig.add_subplot(1, 4, i + 1); ax.imshow(cr(mean[k]), cmap="inferno", vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(f"{t}\nn={cnt[k]} aniso={aniso(mean[k]):.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(1, 4, 4); d = np.clip(dist, 0, 8).astype(float); d[~c3] = np.nan
im = ax.imshow(d, cmap="viridis"); ax.imshow(np.dstack([line, np.zeros_like(line), np.zeros_like(line), line*0.5]).astype(float))
ax.set_title("class-3 px coloured by dist-to-Line\n(Line overlaid red)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Na007b: why class 3 bleeds into the Line — edge-overlap test", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92]); FigureCanvasAgg(fig); fig.savefig(f"{F}/na007b_class3_bleed.png", dpi=150, facecolor="white")
print("wrote na007b_class3_bleed.png", flush=True)
