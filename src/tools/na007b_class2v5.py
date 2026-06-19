"""Compare DINO class 2 vs class 5 on Na007b: avg diffraction, difference,
radial profiles, spatial maps. No cube pass (uses saved class means)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
F = r"docs/explainer/figs"; RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
Ny, Nx = 126, 100; A, B = 2, 5
asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int).reshape(Ny, Nx)
dmean = np.load(f"{F}/na007b_dino_classmeans.npy"); HA = np.load(f"{F}/na007b_HAADF_gui.npy")
mA, mB = dmean[A], dmean[B]; H = mA.shape[0]; cy = (H - 1) / 2
yy, xx = np.indices((H, H)); rr = np.sqrt((yy - cy) ** 2 + (xx - cy) ** 2).astype(int)
def radial(m): tb = np.bincount(rr.ravel(), m.ravel()); n = np.bincount(rr.ravel()); return tb / np.clip(n, 1, None)
rA, rB = radial(mA), radial(mB); rmax = H // 2
def cr(m): return m[H//2-130:H//2+130, H//2-130:H//2+130]   # raw (vmax=2 scale)
ca, cb = cr(mA), cr(mB); d = np.clip(ca, 0, 2) / 2 - np.clip(cb, 0, 2) / 2
fig = Figure(figsize=(16, 3.4), facecolor="white")
vd = np.percentile(np.abs(d), 99)
for i, (img, t, cm, kw) in enumerate([(ca, f"class {A} avg (vmax=2)", "inferno", dict(vmin=0, vmax=2)),
                                       (cb, f"class {B} avg (vmax=2)", "inferno", dict(vmin=0, vmax=2)),
                                       (d, f"{A} - {B} (vmax=2 norm)", "RdBu_r", dict(vmin=-vd, vmax=vd))]):
    ax = fig.add_subplot(1, 5, i + 1)
    ax.imshow(img, cmap=cm, interpolation="nearest", **kw); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(1, 5, 4); beam = int(0.11 * H)
ax.plot(np.log1p(rA[beam:rmax]), label=f"class {A}", lw=1.6); ax.plot(np.log1p(rB[beam:rmax]), label=f"class {B}", lw=1.6)
ax.set_title("radial profile log I(r)", fontsize=10); ax.set_xlabel("r (px, beam-trimmed)"); ax.legend(fontsize=8)
ax = fig.add_subplot(1, 5, 5); lo, hi = np.percentile(HA, 1), np.percentile(HA, 99)
ax.imshow(np.clip(HA, lo, hi), cmap="gray"); ov = np.zeros((Ny, Nx, 4), np.float32)
ov[asg == A] = [0.12, 0.47, 1, .8]; ov[asg == B] = [1, 0.3, 0.1, .8]; ax.imshow(ov)
ax.set_title(f"location: {A}=blue {B}=red", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Na007b DINO class {A} vs {B}", fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.92])
FigureCanvasAgg(fig); fig.savefig(f"{F}/na007b_class{A}v{B}.png", dpi=150, facecolor="white")
# numeric: where radial profiles differ most
rel = (rA[beam:rmax] - rB[beam:rmax]) / (0.5 * (rA[beam:rmax] + rB[beam:rmax]) + 1e-6)
peak = beam + int(np.argmax(np.abs(rel)))
print(f"class {A}: n={(asg==A).sum()} HAADF={HA[asg==A].mean():.0f}", flush=True)
print(f"class {B}: n={(asg==B).sum()} HAADF={HA[asg==B].mean():.0f}", flush=True)
print(f"largest radial difference at r={peak}px ({100*rel[peak-beam]:+.0f}% rel)", flush=True)
print("wrote na007b_class2v5.png", flush=True)
