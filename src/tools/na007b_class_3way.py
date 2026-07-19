"""Na007b DINO class comparison including the interface domain: class 2, class 5,
and class 3 (the yellowish domain at the Line/rest interface; ~69% of its pixels
border the blue Line region 1/8). Avg diffraction per class, radial profiles, and
spatial location. No cube pass (uses saved class means).
  python src/tools/na007b_class_3way.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

F = r"docs/explainer/figs"; REVIEW = r"docs/paper/draft_v2/figs/latest_review"
RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
Ny, Nx = 126, 100
LINE_IDS = [1, 8]                       # the large "Line" diffraction region (blue)
CLASSES = [2, 5, 10]                     # 10 = yellowish interface domain (Line/rest boundary)
NAMES = {2: "class 2", 5: "class 5", 10: "class 10 (interface)"}
# colours exactly as in the main figure (na007b_3way_panel_polar palette)
CMAIN = {2: [0.90, 0.49, 0.13], 5: [0.58, 0.40, 0.74], 10: [0.74, 0.74, 0.13]}  # orange / purple / olive
LINE_C = [0.13, 0.45, 0.95]             # main-figure Line blue
COLORS = {c: CMAIN[c] + [0.9] for c in CLASSES}

asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int).reshape(Ny, Nx)
dmean = np.load(f"{F}/na007b_dino_classmeans.npy"); HA = np.load(f"{F}/na007b_HAADF_gui.npy")
H = dmean.shape[1]; cy = (H - 1) / 2
yy, xx = np.indices((H, H)); rr = np.sqrt((yy - cy) ** 2 + (xx - cy) ** 2).astype(int)
beam = int(0.11 * H); rmax = H // 2


def radial(m):
    tb = np.bincount(rr.ravel(), m.ravel()); n = np.bincount(rr.ravel()); return tb / np.clip(n, 1, None)


def cr(m):
    return m[H // 2 - 130:H // 2 + 130, H // 2 - 130:H // 2 + 130]


ncol = len(CLASSES) + 2
fig = Figure(figsize=(3.2 * ncol, 3.4), facecolor="white")
# class-average diffraction panels
for i, c in enumerate(CLASSES):
    ax = fig.add_subplot(1, ncol, i + 1)
    ax.imshow(cr(dmean[c]), cmap="inferno", interpolation="nearest", vmin=0, vmax=2)
    ax.set_title(f"{NAMES[c]} avg (vmax=2)", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
# radial profiles
ax = fig.add_subplot(1, ncol, ncol - 1)
for c in CLASSES:
    rc = radial(dmean[c]); ax.plot(np.log1p(rc[beam:rmax]), lw=1.6, color=CMAIN[c], label=NAMES[c])
ax.set_title("radial profile log I(r)", fontsize=10); ax.set_xlabel("r (px, beam-trimmed)"); ax.legend(fontsize=7.5)
# location overlay (same colours as the main figure)
ax = fig.add_subplot(1, ncol, ncol)
lo, hi = np.percentile(HA, 1), np.percentile(HA, 99)
ax.imshow(np.clip(HA, lo, hi), cmap="gray")
ov = np.zeros((Ny, Nx, 4), np.float32)
ov[np.isin(asg, LINE_IDS)] = LINE_C + [0.30]            # faint Line blue (context)
for c in CLASSES:
    ov[asg == c] = COLORS[c]
ax.imshow(ov)
ax.set_title("location: 2=orange 5=purple 10=olive (Line=faint blue)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Na007b DINO class 2 vs 5 vs 10 (yellowish Line/rest interface domain)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92]); FigureCanvasAgg(fig)
p = f"{F}/na007b_class2v5v10.png"; fig.savefig(p, dpi=150, facecolor="white")
import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "na007b_class2v5v10.png"))
for c in CLASSES:
    print(f"{NAMES[c]}: n={(asg==c).sum()} HAADF={HA[asg==c].mean():.0f}", flush=True)
print("wrote na007b_class2v5v10.png", flush=True)
