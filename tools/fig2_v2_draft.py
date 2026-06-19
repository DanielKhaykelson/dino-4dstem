"""Figure 2 redesign DRAFT: 3 rows (SAM / NMF / DINO).
 col0 = class maps themselves (distinct colours).
 col1 = SAME maps recoloured by azimuthal anisotropy (texture/crystallite-size,
        shared scale) -> embeds the DINO sub-domain finding physically: SAM has it
        only on the Line, NMF a few blocks, DINO a smooth gradient.
 col2 = representative average diffraction (vmax=2)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib.cm as cmx
from data import open_lazy_cube
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
F = r"docs/explainer/figs"; CUBE = r"D:/DINOSR/data/Na007b_nbed.cube.npy"; RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
ORIENT = r"D:/DINOSR/NanoLetters/Figure4/ComapreToKmap/Na007b_orient_img.npy"; Ny, Nx = 126, 100; KNMF = 6
asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int).reshape(Ny, Nx)
flake = asg != 0; dino_line = np.isin(asg, [1, 8]) & flake
orient = np.load(ORIENT).reshape(Ny, Nx); sam_line = np.isfinite(orient) & (orient != 0)
HA = np.load(f"{F}/na007b_HAADF_gui.npy"); dmean = np.load(f"{F}/na007b_dino_classmeans.npy")
samL = np.load(f"{F}/na007b_avg_SAM_line.npy"); samR = np.load(f"{F}/na007b_avg_SAM_rest.npy")
W = np.load(f"{F}/na007b_polar_W.npy"); nmf = KMeans(KNMF, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(W)).reshape(Ny, Nx)
# cube pass: NMF cluster avg real patterns
cube = open_lazy_cube(CUBE, scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
nsum = np.zeros((KNMF, H, Wd)); ncnt = np.zeros(KNMF)
for rx in range(Ny):
    blk = np.asarray(cube[rx], np.float32)
    for ry in range(Nx):
        if flake[rx, ry]: nsum[nmf[rx, ry]] += blk[ry]; ncnt[nmf[rx, ry]] += 1
navg = np.array([nsum[c] / max(ncnt[c], 1) for c in range(KNMF)])
nmf_line_c = int(np.argmax([((nmf == c) & dino_line).sum() / max((nmf == c).sum(), 1) for c in range(KNMF)]))
# anisotropy of an average pattern at the main ring
cy = (H - 1) / 2; yy, xx = np.indices((H, H)); rr = np.sqrt((yy - cy)**2 + (xx - cy)**2)
th = np.arctan2(yy - cy, xx - cy) % (2*np.pi); beam = int(0.11*H)
def radial(m): tb = np.bincount(rr.astype(int).ravel(), m.ravel()); n = np.bincount(rr.astype(int).ravel()); return tb/np.clip(n,1,None)
r0 = beam + int(np.argmax(radial(dmean[8])[beam:H//2])); band = (rr > r0-4) & (rr < r0+4)
nb = 72; tbk = (th[band]/(2*np.pi)*nb).astype(int)
def aniso(m):
    v = m[band]; a = np.array([v[tbk==k].mean() if (tbk==k).any() else 0 for k in range(nb)]); return float(a.std()/(a.mean()+1e-9))
aD = {c: aniso(dmean[c]) for c in range(dmean.shape[0])}
aN = {c: aniso(navg[c]) for c in range(KNMF)}
aSline, aSrest = aniso(samL), aniso(samR)
amin, amax = 0.0, 1.1; norm = Normalize(amin, amax); pmap = cmx.get_cmap("plasma")
# build property-recolored maps
def recolor(method):
    img = np.zeros((Ny, Nx, 4), np.float32)
    g = np.clip(HA, np.percentile(HA, 1), np.percentile(HA, 99)); g = (g - g.min())/(g.ptp()+1e-9)
    img[..., :3] = g[..., None]; img[..., 3] = 1
    if method == "SAM":
        img[sam_line] = pmap(norm(aSline)); img[flake & ~sam_line] = pmap(norm(aSrest))
    elif method == "NMF":
        for c in range(KNMF):
            if c == nmf_line_c: continue
            img[(nmf == c) & flake] = pmap(norm(aN[c]))
        img[(nmf == nmf_line_c) & flake] = pmap(norm(aN[nmf_line_c]))
    else:
        for c in range(dmean.shape[0]):
            if c == 0: continue
            img[asg == c] = pmap(norm(aD[c]))
    return img
def classmap(method):
    img = np.zeros((Ny, Nx, 4), np.float32); cols = cmx.get_cmap("tab20").colors
    g = np.clip(HA, np.percentile(HA, 1), np.percentile(HA, 99)); g = (g-g.min())/(g.ptp()+1e-9)
    img[..., :3] = g[..., None]*0.5; img[..., 3] = 1
    if method == "SAM":
        hsv = cmx.get_cmap("hsv"); o = orient.copy()
        img[sam_line] = hsv((o[sam_line] % np.pi)/np.pi)
    elif method == "NMF":
        for c in range(KNMF): img[(nmf == c) & flake] = (*cols[c % 20][:3], 1)
    else:
        for c in range(1, dmean.shape[0]): img[asg == c] = (*cols[c % 20][:3], 1)
    return img
def crop(m): return m[H//2-130:H//2+130, H//2-130:H//2+130]
rows = [("SAM", samR), ("NMF", navg[[c for c in range(KNMF) if c != nmf_line_c][0]]), ("DINO", dmean[3])]
fig = Figure(figsize=(11, 9.6), facecolor="white")
for r, (name, rep) in enumerate(rows):
    ax = fig.add_subplot(3, 3, 3*r+1); ax.imshow(classmap(name)); ax.set_ylabel(name, fontsize=13, fontweight="bold")
    if r == 0: ax.set_title("class map", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(3, 3, 3*r+2); im = ax.imshow(recolor(name));
    if r == 0: ax.set_title("texture (azimuthal anisotropy)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(3, 3, 3*r+3); ax.imshow(crop(rep), cmap="inferno", vmin=0, vmax=2)
    if r == 0: ax.set_title("representative diffraction", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
sm = cmx.ScalarMappable(norm=norm, cmap=pmap); sm.set_array([])
cax = fig.add_axes([0.365, 0.06, 0.27, 0.015]); fig.colorbar(sm, cax=cax, orientation="horizontal", label="azimuthal anisotropy (low=fine polycrystalline, high=coarse/textured)")
fig.suptitle("Figure 2 DRAFT — class maps (col1), texture recolour (col2 = DINO sub-domain gradient), diffraction (col3)", fontsize=12)
fig.tight_layout(rect=[0, 0.09, 1, 0.95]); FigureCanvasAgg(fig); fig.savefig(f"{F}/fig2_v2_draft.png", dpi=160, facecolor="white")
print(f"DINO per-class anisotropy: {[round(aD[c],2) for c in range(dmean.shape[0])]}", flush=True)
print(f"NMF per-cluster anisotropy: {[round(aN[c],2) for c in range(KNMF)]} (Line cluster={nmf_line_c})", flush=True)
print(f"SAM Line/rest anisotropy: {aSline:.2f}/{aSrest:.2f}; wrote fig2_v2_draft.png", flush=True)
