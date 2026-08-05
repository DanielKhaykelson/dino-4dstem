"""Na007b 3x3 comparison panel: rows = SAM / NMF(polar+theta-shift) / DINO,
cols = region map on BF (Line vs rest) / Line avg diffraction / non-Line avg.
NMF uses the polar+theta-shift loadings (na007b_polar_W.npy) clustered at a
coarse K=6 (its natural granularity); the Line cluster is picked by diffraction
match to the SAM Line. Reports IoU/Dice/containment vs SAM + beam-masked Pearson r."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

F = r"docs/explainer/figs"
CUBE = r"D:/DINOSR/data/Na007b_nbed.cube.npy"
RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
ORIENT = r"D:/DINOSR/NanoLetters/Figure4/ComapreToKmap/Na007b_orient_img.npy"
Ny, Nx = 126, 100; N = Ny * Nx; PXNM = 16.0
KNMF = 6

asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int).reshape(Ny, Nx)
flake = (asg != 0)
dino_line = np.isin(asg, [1, 8]) & flake
orient = np.load(ORIENT).reshape(Ny, Nx)
sam_line = np.isfinite(orient) & (orient != 0)
BF = np.load(f"{F}/na007b_HAADF_gui.npy")   # HAADF contrast (GUI style, sample bright)
W = np.load(f"{F}/na007b_polar_W.npy")
nmf = KMeans(KNMF, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(W)).reshape(Ny, Nx)

# ---- one cube pass: real-diffraction averages ----
cube = open_lazy_cube(CUBE, scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
acc = {k: np.zeros((H, Wd)) for k in ["SAM_line", "SAM_rest", "DINO_line", "DINO_rest"]}
cnt = {k: 0 for k in acc}
nsum = np.zeros((KNMF, H, Wd)); ncnt = np.zeros(KNMF)
print("cube pass...", flush=True); t0 = time.time()
for rx in range(Ny):
    blk = np.asarray(cube[rx], np.float32)
    for ry in range(Nx):
        pat = blk[ry]
        if sam_line[rx, ry]: acc["SAM_line"] += pat; cnt["SAM_line"] += 1
        elif flake[rx, ry]:  acc["SAM_rest"] += pat; cnt["SAM_rest"] += 1
        if dino_line[rx, ry]: acc["DINO_line"] += pat; cnt["DINO_line"] += 1
        elif flake[rx, ry]:   acc["DINO_rest"] += pat; cnt["DINO_rest"] += 1
        if flake[rx, ry]: nsum[nmf[rx, ry]] += pat; ncnt[nmf[rx, ry]] += 1
    if rx % 30 == 0: print(f"  row {rx} ({time.time()-t0:.0f}s)", flush=True)
means = {k: acc[k] / max(cnt[k], 1) for k in acc}
navg = np.array([nsum[c] / max(ncnt[c], 1) for c in range(KNMF)])

# ---- pick NMF Line cluster by ring-masked corr with SAM Line ----
cy = (H - 1) / 2; yy, xx = np.indices((H, Wd)); rr = np.sqrt((yy - cy) ** 2 + (xx - cy) ** 2); nb = H // 2
ring = (rr > 0.18 * nb) & (rr < 0.95 * nb)
def disp(x): x = np.log1p(np.clip(x, 0, None)); v = x[ring]; return (x - v.min()) / (v.ptp() + 1e-9)
def rcorr(a, b): return float(np.corrcoef(disp(a)[ring], disp(b)[ring])[0, 1])
corr_line = [rcorr(navg[c], means["SAM_line"]) for c in range(KNMF)]
nmf_line_c = int(np.argmax(corr_line))
print(f"NMF cluster ring-corr vs SAM Line: {[round(x,3) for x in corr_line]} -> Line cluster = {nmf_line_c}", flush=True)
nmf_line = (nmf == nmf_line_c) & flake
nmf_rest = flake & ~nmf_line
# NMF Line/rest avgs
means["NMF_line"] = navg[nmf_line_c]
rest_sum = sum(nsum[c] for c in range(KNMF) if c != nmf_line_c); rest_cnt = sum(ncnt[c] for c in range(KNMF) if c != nmf_line_c)
means["NMF_rest"] = rest_sum / max(rest_cnt, 1)

# ---- metrics ----
def iou(a, b): i = (a & b).sum(); u = (a | b).sum(); return i / u if u else 0.0
def dice(a, b): return 2 * (a & b).sum() / (a.sum() + b.sum()) if (a.sum() + b.sum()) else 0.0
def contain(a, b): return (a & b).sum() / a.sum() if a.sum() else 0.0   # frac of a inside b
M = {}
for nm, ln in [("DINO", dino_line), ("NMF", nmf_line)]:
    M[nm] = dict(IoU=iou(ln, sam_line), Dice=dice(ln, sam_line),
                 contain=contain(sam_line, ln), n=int(ln.sum()))
M["SAM"] = dict(n=int(sam_line.sum()))
r_line_vs_sam = {nm: rcorr(means[f"{nm}_line"], means["SAM_line"]) for nm in ["NMF", "DINO"]}
r_line_vs_rest = {nm: rcorr(means[f"{nm}_line"], means[f"{nm}_rest"]) for nm in ["SAM", "NMF", "DINO"]}
print("metrics:", M, flush=True)
print("r(Line, SAM Line):", {k: round(v, 3) for k, v in r_line_vs_sam.items()}, flush=True)
print("r(Line, rest):", {k: round(v, 3) for k, v in r_line_vs_rest.items()}, flush=True)

# ---- 3x3 panel ----
# Match the GUI exactly (posthoc_panel._render_overlay): virtual HAADF as LINEAR
# clip(1,99) grayscale; overlay ONLY the Line region (alpha 0.6). The rest of
# the flake stays plain gray HAADF (the GUI never colours unselected classes).
_lo, _hi = np.percentile(BF, 1), np.percentile(BF, 99); _bg = np.clip(BF, _lo, _hi)
LINEC = np.array([0.13, 0.45, 0.95])   # Line = blue (same in all three panels)
# palette for non-Line classes (deliberately avoids the Line blue)
PALETTE = np.array([[0.90, 0.49, 0.13], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
                    [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76],
                    [0.50, 0.50, 0.50], [0.74, 0.74, 0.13], [0.09, 0.75, 0.81]])
def _scalebar(ax, dark=False):
    sb = 500.0 / PXNM; c = "black" if dark else "white"
    ax.add_patch(Rectangle((5, Ny - 8), sb, 2.4, color=c, ec="black", lw=0.4))
    ax.text(5 + sb / 2, Ny - 11, "500 nm", color=c, ha="center", va="bottom", fontsize=7)
def overlay(ax, line_m, rest_m=None, a=0.60):
    ax.imshow(_bg, cmap="gray", aspect="equal", interpolation="nearest")
    ov = np.zeros((Ny, Nx, 4), np.float32); ov[line_m, :3] = LINEC; ov[line_m, 3] = a
    ax.imshow(ov, aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); _scalebar(ax, dark=False)
def classmap(ax, lab2d, line_ids):
    """Full class map: every flake cluster a colour, the Line cluster(s) forced to
    the shared blue; background (off-flake) light grey."""
    img = np.ones((Ny, Nx, 3), np.float32) * 0.93
    others = [c for c in sorted(set(lab2d[flake].tolist())) if c not in line_ids]
    for i, c in enumerate(others):
        img[(lab2d == c) & flake] = PALETTE[i % len(PALETTE)]
    img[np.isin(lab2d, list(line_ids)) & flake] = LINEC
    ax.imshow(img, aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); _scalebar(ax, dark=True)
def diff(ax, m):
    a = np.log1p(np.clip(m, 0, None)); Hh = a.shape[0]; a = a[Hh // 2 - 130:Hh // 2 + 130, Hh // 2 - 130:Hh // 2 + 130]
    o = a[a > 0]
    ax.imshow(a, cmap="inferno", vmax=(np.percentile(o, 99.5) if o.size else 1), interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    n = a.shape[0]; sb = 2.0 / 0.0185   # 2 nm^-1 bar at 0.0185 nm^-1 per detector pixel
    ax.add_patch(Rectangle((12, n - 16), sb, max(3.0, n * 0.014), color="white", ec="black", lw=0.4, zorder=12))
    ax.text(12, n - 20, "2 nm$^{-1}$", color="white", ha="left", va="bottom", fontsize=9, fontweight="bold",
            zorder=12, path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

rows = [("SAM", sam_line, flake & ~sam_line, "SAM_line", "SAM_rest"),
        ("NMF (polar+θ-shift)", nmf_line, nmf_rest, "NMF_line", "NMF_rest"),
        ("DINO", dino_line, flake & ~dino_line, "DINO_line", "DINO_rest")]
fig = Figure(figsize=(9.5, 9.6), facecolor="white")
col1_title = ["SAM Line overlaid on HAADF", "NMF class map (Line = blue)", "DINO class map (Line = blue)"]
_LET = "abcdefghi"
def _pl(ax, idx):
    ax.text(0.05, 0.96, _LET[idx], transform=ax.transAxes, fontsize=16, fontweight="bold",
            va="top", ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.16", fc="black", alpha=0.55, ec="none"), zorder=20)
for ri, (name, lm, rm, lk, rk) in enumerate(rows):
    ax = fig.add_subplot(3, 3, 3 * ri + 1)
    if ri == 0:   overlay(ax, lm, rm)                       # SAM = overlay on HAADF
    elif ri == 1: classmap(ax, nmf, {nmf_line_c})           # NMF = full class map
    else:         classmap(ax, asg, {1, 8})                 # DINO = full class map
    ax.set_ylabel(name, fontsize=12, fontweight="bold"); _pl(ax, 3 * ri + 0)
    ax.set_title(col1_title[ri], fontsize=9)
    ax = fig.add_subplot(3, 3, 3 * ri + 2); diff(ax, means[lk]); _pl(ax, 3 * ri + 1)
    if ri == 0: ax.set_title("Line: average diffraction", fontsize=10)
    ax = fig.add_subplot(3, 3, 3 * ri + 3); diff(ax, means[rk]); _pl(ax, 3 * ri + 2)
    if ri == 0: ax.set_title("non-Line: average diffraction", fontsize=10)
cap = (f"Na007b — SAM vs NMF (polar+θ-shift, K={KNMF}) vs DINO.  Line px: "
       f"SAM={M['SAM']['n']}, NMF={M['NMF']['n']}, DINO={M['DINO']['n']}.  "
       f"vs SAM Line:  DINO IoU={M['DINO']['IoU']:.2f}/Dice={M['DINO']['Dice']:.2f}  |  "
       f"NMF IoU={M['NMF']['IoU']:.2f}/Dice={M['NMF']['Dice']:.2f}.  "
       f"Beam-masked Pearson r(Line avg, SAM Line avg): DINO={r_line_vs_sam['DINO']:.3f}, "
       f"NMF={r_line_vs_sam['NMF']:.3f};  r(Line,rest): SAM={r_line_vs_rest['SAM']:.2f}, "
       f"NMF={r_line_vs_rest['NMF']:.2f}, DINO={r_line_vs_rest['DINO']:.2f}.")
print("FIG2 metrics:", cap, flush=True)   # numbers go in the manuscript caption, not on the figure
fig.tight_layout(rect=[0, 0, 1, 1]); FigureCanvasAgg(fig)
fig.savefig(f"{F}/na007b_3way_panel_polar.png", dpi=160, facecolor="white")
np.save(f"{F}/na007b_avg_NMFpolar_line.npy", means["NMF_line"])
np.save(f"{F}/na007b_avg_NMFpolar_rest.npy", means["NMF_rest"])
print("wrote na007b_3way_panel_polar.png  DONE", flush=True)
