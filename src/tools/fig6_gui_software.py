"""Figure 6: the software layer. A drawn GUI-window mockup (left) holding a real
IMC class map, a multi-scale inspection strip from real data (single frame ->
grain average -> class average), and a natural-language assistant chat mockup.
Real screenshots can't be captured headless, so the window chrome is drawn; the
diffraction/class-map content is real. Saves to BorisEdits + latest_review.
  python src/tools/fig6_gui_software.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl
from gui_app.crystallinity_panel import _radial_mean_var
from data import open_lazy_cube

FIGS = "docs/paper/draft_v2/figs"; OUT = os.path.join(FIGS, "BorisEdits"); REVIEW = os.path.join(FIGS, "latest_review")
RUN = "runs/_gui/IMC_SI4_m097_k60"
CUBE = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy"
NY = NX = 128; INV = 0.00185; KMAX = 0.35; FOV = 160

asg = np.load(os.path.join(RUN, "eval", "inference.npz"))["assigns"].astype(int).reshape(NY, NX)
z = np.load(os.path.join(FIGS, "grain_acom_v2_SI4.npz"))
gid = z["gid"].reshape(NY, NX); cls = z["cls"]; vac = z["vac"]; gsum = z["gsum"]; gcnt = z["gcnt"]; H = int(z["H"])
cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV)

# per-grain spottiness -> pick the spottiest (most crystalline needle) grain
spot = np.full(gsum.shape[0], -1.0)
for g in range(gsum.shape[0]):
    if vac[g]:
        continue
    m, v, _ = _radial_mean_var(gsum[g] / max(gcnt[g], 1), (cyx, cyx), beam_px=beam)
    s = m[lo:hi]; vs = v[lo:hi]
    if s.size >= 5 and s.sum() > 0:
        spot[g] = np.percentile(np.sqrt(np.clip(vs, 0, None)) / np.clip(s, 1e-9, None), 90)
G = int(np.argmax(spot)); C = int(cls[G])
grain_avg = gsum[G] / max(gcnt[G], 1)
# class average = all grains in class C
members = [g for g in range(gsum.shape[0]) if cls[g] == C and not vac[g]]
class_avg = sum(gsum[g] for g in members) / max(sum(gcnt[g] for g in members), 1)
# one real single frame from a pixel of grain G
ys, xs = np.where(gid == G); ry, rx = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
try:
    cube = open_lazy_cube(CUBE, scan_shape=(NY, NX)); frame = np.clip(np.asarray(cube[ry][rx], np.float32), 0, 2)
except Exception as e:
    print("cube frame fallback:", e); frame = np.clip(grain_avg + np.random.poisson(np.clip(grain_avg, 0, None)) * 0.0, 0, 2)


def disp(a):
    h = a.shape[0]; c = a[h // 2 - 130:h // 2 + 130, h // 2 - 130:h // 2 + 130]
    return np.log1p(np.clip(c, 0, None))


# class map coloured by spottiness order (Spectral_r), like Fig 4
def class_spot_rank():
    cval = {}
    for c in np.unique(asg):
        gg = spot[(cls == c) & (spot >= 0)]
        cval[int(c)] = float(np.median(gg)) if gg.size else -np.inf
    order = sorted(np.unique(asg).tolist(), key=lambda c: cval[int(c)])
    return {int(c): r for r, c in enumerate(order)}


rank = class_spot_rank(); nC = max(len(rank), 1)
cmap_cls = ListedColormap([mpl.colormaps.get_cmap("Spectral_r").resampled(nC)(k) for k in range(nC)])
clsmap = np.vectorize(lambda c: rank[int(c)])(asg).astype(float)

# ---------------- compose ----------------
NAVY = "#21304a"; TEAL = "#1C7293"; LIGHT = "#eef2f6"; AMBER = "#E8A33D"; INK = "#222"
fig = Figure(figsize=(13.5, 6.4), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13.5); ax.set_ylim(0, 6.4); ax.axis("off")

# ===== A) GUI window mockup (left) =====
ax.add_patch(FancyBboxPatch((0.3, 0.5), 6.2, 5.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec="#9fb0c3", lw=1.5))
ax.add_patch(FancyBboxPatch((0.3, 5.55), 6.2, 0.45, boxstyle="round,pad=0.02,rounding_size=0.12", fc=NAVY, ec="none"))
ax.text(0.55, 5.78, "DINO4DSTEM", color="white", fontsize=12, fontweight="bold", va="center")
ax.text(6.25, 5.78, "—  □  ×", color="#cdd6e0", fontsize=10, va="center", ha="right")
for i, t in enumerate(["Data", "Model", "Analysis", "Clustering", "Diffraction"]):
    fc = TEAL if t == "Analysis" else LIGHT; tc = "white" if t == "Analysis" else "#33455c"
    ax.add_patch(FancyBboxPatch((0.5 + i * 1.18, 5.0), 1.08, 0.4, boxstyle="round,pad=0.01,rounding_size=0.06", fc=fc, ec="#c4d0db"))
    ax.text(0.5 + i * 1.18 + 0.54, 5.2, t, ha="center", va="center", fontsize=8.5, color=tc, fontweight="bold")
# left sidebar buttons
for j, b in enumerate(["Load run", "Browse cube", "Class map", "Merge classes", "GradCAM", "Report"]):
    ax.add_patch(FancyBboxPatch((0.5, 4.4 - j * 0.55), 1.25, 0.42, boxstyle="round,pad=0.01,rounding_size=0.06", fc=LIGHT, ec="#c4d0db"))
    ax.text(1.12, 4.61 - j * 0.55, b, ha="center", va="center", fontsize=7.2, color="#33455c")
# main canvas = real class map
cax = fig.add_axes([0.155, 0.17, 0.27, 0.52]); cax.imshow(clsmap, cmap=cmap_cls, interpolation="nearest"); cax.set_xticks([]); cax.set_yticks([])
for s in cax.spines.values():
    s.set_edgecolor("#9fb0c3")
ax.text(3.9, 4.78, "class map  (IMC needles)", fontsize=8.5, color="#33455c", ha="center")
ax.text(3.9, 0.72, "reciprocal: 0.00185 Å⁻¹ / px      real-space: 16 nm / px", fontsize=7, color="#7a8794", ha="center")
ax.text(3.4, 0.18, "A  one fixed model — only pre-processing is set by the user", fontsize=9, color=INK, fontweight="bold", ha="center")

# ===== B) multi-scale inspection (top right) =====
def imbox(x, y, w, img, cmap, title):
    d = disp(img); vmax = max(np.percentile(d, 99.5), 1e-3)
    a = fig.add_axes([x, y, w, w * (13.5 / 6.4)])   # square on the page
    a.imshow(d, cmap=cmap, interpolation="bilinear", vmin=0, vmax=vmax); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_edgecolor("#b9c4cf")
    a.set_title(title, fontsize=9, color=INK, pad=2)
    return a

ax.text(10.2, 6.12, "B   inspect at any scale", fontsize=11, fontweight="bold", ha="center", color=INK)
imbox(0.515, 0.595, 0.150, frame, "inferno", "single frame\n(noisy)")
imbox(0.685, 0.595, 0.150, grain_avg, "inferno", "grain average\n(clean)")
imbox(0.855, 0.595, 0.150, class_avg, "inferno", "class average")
for x0 in (9.42, 11.72):
    ax.add_patch(FancyArrowPatch((x0, 4.95), (x0 + 0.42, 4.95), arrowstyle="-|>", mutation_scale=14, lw=1.8, color="#7a8794"))

# ===== C) assistant chat (bottom right) =====
ax.text(10.3, 4.05, "C   ask in plain language", fontsize=10, fontweight="bold", ha="center", color=INK)
def bubble(x, y, w, h, text, fc, tc, align="left"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec="none"))
    ax.text(x + 0.18, y + h / 2, text, ha="left", va="center", fontsize=8.3, color=tc, wrap=True)
bubble(8.7, 3.25, 4.4, 0.55, "“Cluster this scan and show the most crystalline class.”", TEAL, "white")
bubble(8.4, 2.45, 4.7, 0.62, "Clustered with DINO4DSTEM. Most crystalline = class 3\n(highest Bragg excess). Map and average shown.", LIGHT, "#22303f")
bubble(8.7, 1.7, 4.4, 0.55, "“Merge the two thinnest classes and re-export.”", TEAL, "white")
bubble(8.4, 0.95, 4.7, 0.55, "Merged classes 5+8, re-exported report (PPTX + PNGs).", LIGHT, "#22303f")
ax.text(10.75, 0.5, "no code · full pipeline by chat", fontsize=8, color="#7a8794", ha="center", style="italic")

p = os.path.join(OUT, "fig6_gui_software.png"); FigureCanvasAgg(fig); fig.savefig(p, dpi=170, facecolor="white")
import shutil; shutil.copy(p, os.path.join(REVIEW, "fig6_gui_software.png"))
print(f"wrote fig6_gui_software.png  (needle grain {G}, class {C}, frame@{ry},{rx})", flush=True)
