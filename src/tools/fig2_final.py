"""Figure 2 (final): NaPHI Na007b — SAM / NMF / DINO.
col0 = class maps; col1 = same maps recoloured by per-class crystallinity
(peak/halo) -> shows DINO resolves a finer body gradient (the 'what we do better');
col2 = non-Line region-average diffraction (vmax=2). Line IoU/Dice/r annotated
(the 'what we do the same')."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib.cm as cmx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
F = "docs/explainer/figs"; OUT = "docs/paper/draft_v2/figs"; RUN = "runs/_gui/Na007b_k60_m097_vmax2"
ORIENT = r"D:/DINOSR/NanoLetters/Figure4/ComapreToKmap/Na007b_orient_img.npy"; Ny, Nx = 126, 100; KNMF = 6
asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int).reshape(Ny, Nx)
flake = asg != 0; dino_line = np.isin(asg, [1, 8]) & flake
orient = np.load(ORIENT).reshape(Ny, Nx); sam_line = np.isfinite(orient) & (orient != 0)
HA = np.load(f"{F}/na007b_HAADF_gui.npy"); ph = np.load(f"{F}/na007b_cryst.npz.npy")[0].reshape(Ny, Nx)
W = np.load(f"{F}/na007b_polar_W.npy"); nmf = KMeans(KNMF, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(W)).reshape(Ny, Nx)
nmf_line = int(np.argmax([((nmf == c) & dino_line).sum()/max((nmf == c).sum(), 1) for c in range(KNMF)]))
sR = np.load(f"{F}/na007b_avg_SAM_rest.npy"); nR = np.load(f"{F}/na007b_avg_NMFpolar_rest.npy"); dR = np.load(f"{F}/na007b_avg_DINO_rest.npy")
g = np.clip(HA, np.percentile(HA, 1), np.percentile(HA, 99)); g = (g - g.min())/(g.ptp()+1e-9)
def med(mask): return float(np.nanmedian(ph[mask])) if mask.any() else np.nan
phD = {c: med((asg == c)) for c in range(1, int(asg.max())+1)}
phN = {c: med((nmf == c) & flake) for c in range(KNMF)}
phS_line, phS_rest = med(sam_line), med(flake & ~sam_line)
norm = Normalize(0.22, 0.48); cmap = cmx.get_cmap("viridis"); cols = cmx.get_cmap("tab20").colors
def base(): r = np.zeros((Ny, Nx, 4), np.float32); r[..., :3] = (g[..., None]*0.5); r[..., 3] = 1; return r
def classmap(meth):
    im = base()
    if meth == "SAM": im[sam_line] = (0.95, 0.55, 0.1, 1)            # Line vs unindexed
    elif meth == "NMF":
        for c in range(KNMF): im[(nmf == c) & flake] = (*cols[c % 20][:3], 1)
    else:
        for c in range(1, int(asg.max())+1): im[asg == c] = (*cols[c % 20][:3], 1)
    return im
def phmap(meth):
    im = base()
    if meth == "SAM": im[sam_line] = cmap(norm(phS_line)); im[flake & ~sam_line] = cmap(norm(phS_rest))
    elif meth == "NMF":
        for c in range(KNMF): im[(nmf == c) & flake] = cmap(norm(phN[c]))
    else:
        for c in range(1, int(asg.max())+1): im[asg == c] = cmap(norm(phD[c]))
    return im
def cr(m): a = m; H = a.shape[0]; return a[H//2-130:H//2+130, H//2-130:H//2+130]
rows = [("SAM", sR), ("NMF", nR), ("DINO", dR)]
fig = Figure(figsize=(11, 9.8), facecolor="white")
for r, (name, rep) in enumerate(rows):
    a = fig.add_subplot(3, 3, 3*r+1); a.imshow(classmap(name)); a.set_ylabel(name, fontsize=13, fontweight="bold")
    if r == 0: a.set_title("class map", fontsize=11)
    a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(3, 3, 3*r+2); a.imshow(phmap(name))
    if r == 0: a.set_title("crystallinity (peak/halo) per class", fontsize=11)
    a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(3, 3, 3*r+3); a.imshow(cr(rep), cmap="inferno", vmin=0, vmax=2)
    if r == 0: a.set_title("non-Line avg diffraction (vmax=2)", fontsize=11)
    a.set_xticks([]); a.set_yticks([])
sm = cmx.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cax = fig.add_axes([0.37, 0.055, 0.27, 0.014]); fig.colorbar(sm, cax=cax, orientation="horizontal", label="peak/halo (amorphous → crystalline)")
fig.text(0.5, 0.013, "Line domain: DINO IoU 0.74 / Dice 0.85 ; polar-NMF IoU 0.77 / Dice 0.87 vs SAM ; "
         "beam-masked Pearson r(Line avg, SAM)=0.999 (same). DINO resolves a finer body crystallinity gradient (better).",
         ha="center", fontsize=8.5)
fig.suptitle("Figure 2 — NaPHI (Na007b): SAM / NMF / DINO — same on the Line, DINO finer in the body", fontsize=12)
fig.tight_layout(rect=[0, 0.08, 1, 0.95]); FigureCanvasAgg(fig); fig.savefig(f"{OUT}/fig2_naphi.png", dpi=160, facecolor="white")
print("DINO per-class p/h:", {c: round(phD[c], 2) for c in phD})
print("NMF per-cluster p/h:", {c: round(phN[c], 2) for c in phN})
print("wrote fig2_naphi.png", flush=True)
