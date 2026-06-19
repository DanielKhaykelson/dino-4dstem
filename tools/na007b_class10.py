"""Is DINO class 10 an interface class? Interface score = mean number of DISTINCT
neighbouring classes (8-nbr) over a class's pixels (high = sits on domain
boundaries). Plus class-10 spatial map on HAADF + avg diffraction (vmax=2)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import distance_transform_edt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
F = r"docs/explainer/figs"; RUN = r"runs/_gui/Na007b_k60_m097_vmax2"; Ny, Nx = 126, 100
asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int).reshape(Ny, Nx)
dmean = np.load(f"{F}/na007b_dino_classmeans.npy"); HA = np.load(f"{F}/na007b_HAADF_gui.npy")
flake = asg != 0; line = np.isin(asg, [1, 8]); dist = distance_transform_edt(~line)
# neighbour-diversity interface score
def ndiv():
    div = np.zeros((Ny, Nx))
    for y in range(Ny):
        for x in range(Nx):
            if not flake[y, x]: continue
            s = set()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < Ny and 0 <= xx < Nx and flake[yy, xx]: s.add(asg[yy, xx])
            div[y, x] = len(s)
    return div
D = ndiv()
rest = [c for c in range(int(asg.max()) + 1) if c not in (0,) and (asg == c).sum() > 30]
print("interface score (mean distinct 8-nbr classes) + Line-adjacency per class:", flush=True)
scores = {}
for c in rest:
    m = asg == c; scores[c] = float(D[m].mean())
    print(f"  class {c}: nbr-diversity={scores[c]:.2f}  line-adj(<=1.5px)={ (dist[m]<=1.5).mean():.2f}  n={m.sum()}", flush=True)
hi = sorted(scores, key=lambda c: -scores[c])[:3]
print(f"most interfacial classes: {hi}", flush=True)
# figure: class-10 location + avg
fig = Figure(figsize=(9, 3.4), facecolor="white")
ax = fig.add_subplot(1, 3, 1); lo, h = np.percentile(HA, 1), np.percentile(HA, 99)
ax.imshow(np.clip(HA, lo, h), cmap="gray"); o = np.zeros((Ny, Nx, 4), np.float32); o[asg == 10] = [1, 0.55, 0, .85]
ax.imshow(o); ax.set_title(f"class 10 on HAADF (n={(asg==10).sum()})", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(1, 3, 2)
im = dmean[10][dmean.shape[1]//2-130:dmean.shape[1]//2+130, dmean.shape[1]//2-130:dmean.shape[1]//2+130]
ax.imshow(im, cmap="inferno", vmin=0, vmax=2); ax.set_title("class 10 avg (vmax=2)", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(1, 3, 3); ax.bar([f"c{c}" for c in rest], [scores[c] for c in rest], color=["#ff8c00" if c == 10 else "#1f77b4" for c in rest])
ax.set_ylabel("nbr-class diversity"); ax.set_title("interface score per class (10 highlighted)", fontsize=10); ax.tick_params(axis='x', labelsize=7)
fig.suptitle(f"Na007b DINO class 10 — interface character (score={scores.get(10,0):.2f}, rank {sorted(scores,key=lambda c:-scores[c]).index(10)+1}/{len(rest)})", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92]); FigureCanvasAgg(fig); fig.savefig(f"{F}/na007b_class10_interface.png", dpi=150, facecolor="white")
print("wrote na007b_class10_interface.png", flush=True)
