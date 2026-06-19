"""Compare DINO classes 2, 3, 5 (most of the non-Line area) on Na007b — expert
4DSTEM/materials read: avg diffraction (vmax=2), pairwise differences, radial
profiles (per-frame mean), azimuthal modulation at the main ring. No cube pass."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
F = r"docs/explainer/figs"; RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
Ny, Nx = 126, 100; CLS = [2, 3, 5]; COL = {2: [0.12, 0.47, 1], 3: [0.18, 0.7, 0.2], 5: [1, 0.3, 0.1]}
asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int).reshape(Ny, Nx)
dmean = np.load(f"{F}/na007b_dino_classmeans.npy"); HA = np.load(f"{F}/na007b_HAADF_gui.npy")
ph = np.load(f"{F}/na007b_cryst.npz.npy")[0].reshape(Ny, Nx)
H = dmean.shape[1]; cy = (H - 1) / 2; yy, xx = np.indices((H, H))
rr = np.sqrt((yy - cy) ** 2 + (xx - cy) ** 2); ri = rr.astype(int); th = (np.arctan2(yy - cy, xx - cy) % (2 * np.pi))
def radial(m): tb = np.bincount(ri.ravel(), m.ravel()); n = np.bincount(ri.ravel()); return tb / np.clip(n, 1, None)
beam = int(0.11 * H); rmax = H // 2
prof = {c: radial(dmean[c]) for c in CLS}
r0 = beam + int(np.argmax(prof[CLS[0]][beam:rmax]))           # main ring
band = (rr > r0 - 4) & (rr < r0 + 4)
nbins = 72; tb = (th[band] / (2 * np.pi) * nbins).astype(int)
def azim(c):
    v = dmean[c][band]; out = np.array([v[tb == k].mean() if (tb == k).any() else 0 for k in range(nbins)])
    return out
azp = {c: azim(c) for c in CLS}
aniso = {c: float(azp[c].std() / (azp[c].mean() + 1e-9)) for c in CLS}
def cr(m): return m[H//2-130:H//2+130, H//2-130:H//2+130]

fig = Figure(figsize=(15, 7), facecolor="white")
for i, c in enumerate(CLS):
    ax = fig.add_subplot(2, 4, i + 1); ax.imshow(cr(dmean[c]), cmap="inferno", vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(f"class {c} avg (vmax=2)\nn={(asg==c).sum()} HAADF={HA[asg==c].mean():.0f} p/h={np.nanmedian(ph[asg==c]):.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
# spatial map
ax = fig.add_subplot(2, 4, 4); lo, hi = np.percentile(HA, 1), np.percentile(HA, 99)
ax.imshow(np.clip(HA, lo, hi), cmap="gray"); o = np.zeros((Ny, Nx, 4), np.float32)
for c in CLS: o[asg == c, :3] = COL[c]; o[asg == c, 3] = 0.8
ax.imshow(o); ax.set_title("location  2=blue 3=green 5=red", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
# pairwise differences (vmax=2 normalised)
for i, (a, b) in enumerate(itertools.combinations(CLS, 2)):
    d = np.clip(cr(dmean[a]), 0, 2) / 2 - np.clip(cr(dmean[b]), 0, 2) / 2
    ax = fig.add_subplot(2, 4, 5 + i); v = np.percentile(np.abs(d), 99)
    ax.imshow(d, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title(f"{a} - {b}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
# radial + azimuthal
ax = fig.add_subplot(2, 4, 8)
for c in CLS: ax.plot(np.log1p(prof[c][beam:rmax]), color=COL[c], lw=1.5, label=f"cls {c}")
ax.axvline(r0 - beam, color="gray", ls=":", lw=1); ax.set_title("radial log I(r); ring r0 dotted", fontsize=9)
ax.set_xlabel("r (px, beam-trimmed)"); ax.legend(fontsize=8)
fig.suptitle(f"Na007b DINO classes 2/3/5 (non-Line body).  azimuthal anisotropy @r0={r0}px: "
             f"2={aniso[2]:.2f} 3={aniso[3]:.2f} 5={aniso[5]:.2f}", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig); fig.savefig(f"{F}/na007b_class235.png", dpi=150, facecolor="white")
# numbers
for c in CLS: print(f"class {c}: n={(asg==c).sum()} HAADF={HA[asg==c].mean():.0f} p/h={np.nanmedian(ph[asg==c]):.3f} azim-aniso={aniso[c]:.3f}", flush=True)
for a, b in itertools.combinations(CLS, 2):
    rel = (prof[a][beam:rmax] - prof[b][beam:rmax]) / (0.5 * (prof[a][beam:rmax] + prof[b][beam:rmax]) + 1e-6)
    pk = beam + int(np.argmax(np.abs(rel)))
    print(f"  {a}-{b}: max radial diff at r={pk}px ({100*rel[pk-beam]:+.0f}%)", flush=True)
print("wrote na007b_class235.png", flush=True)
