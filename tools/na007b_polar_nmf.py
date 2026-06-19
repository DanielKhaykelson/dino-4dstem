"""Na007b NMF analysis with the GUI POLAR + theta-shift (no-log) variant
(Uesugi 2020 + Krajnak 2020). Uses gui_app.nmf_panel.build_nmf_input so it is
identical to the GUI tool. Emits: model selection (cophenetic/dispersion/
silhouette/EV), components, spatial loadings, reconstruction, and the over/under-
clustering answer (tests 1/2 + crystallinity), reusing unchanged DINO arrays."""
import os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import register_runtime_sample
from gui_app.nmf_panel import build_nmf_input
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

F = r"docs/explainer/figs"
RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
Ny, Nx = 126, 100
N = Ny * Nx

# DINO arrays (unchanged) -------------------------------------------------
asg = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)["assigns"].astype(int)
Kd = int(asg.max()) + 1
flake = (asg != 0)
dmean = np.load(f"{F}/na007b_dino_classmeans.npy")          # (Kd,512,512) real diffraction
ph = np.load(f"{F}/na007b_cryst.npz.npy")[0]                 # peak/halo per position

# Polar + theta-shift (no log) input via the GUI pipeline -----------------
key = register_runtime_sample(r"D:/DINOSR/data/Na007b_nbed.cube.npy",
                              scan_shape=(Ny, Nx), vmax=2.0, center_mask_radius=22)
cfg = dict(input="polar", log=False, sparse=False, theta_shift=False)
print("build_nmf_input (polar, no-log)...", flush=True); t0 = time.time()
X, _, comp_shape, info = build_nmf_input(key, cfg, vmax_override=2.0)   # X:(N,192*192)
P = comp_shape[0]; X = X.reshape(N, P, P).astype(np.float32)
print(f"  polar matrix {X.shape} comp_shape={comp_shape} {time.time()-t0:.0f}s", flush=True)

def bmean3(a, f):
    n, H, W = a.shape; s = H // f
    return a.reshape(n, s, f, s, f).mean((2, 4))

def aug(Xs):   # theta-shift = roll along theta (axis=1), 4 shifts
    s = Xs.shape[1]
    return np.concatenate([np.roll(Xs, sh, axis=1) for sh in (0, s // 4, s // 2, 3 * s // 4)], 0).reshape(-1, s * s).astype(np.float32)

X48 = bmean3(X, 4); X96 = bmean3(X, 2)   # 48^2 selection, 96^2 final

# ---- model selection on 48^2 (theta-aug, position subsample) ----
RANKS = [4, 6, 8, 10, 12, 14, 16, 18, 20]; NRUN = 10; SUB = 1500
rng = np.random.RandomState(0); pos = rng.choice(N, 4000, replace=False); sub = rng.choice(len(pos), SUB, replace=False)
Xsel = X48[pos]; Xsel_aug = aug(Xsel); npos = len(pos)
Xsel_flat = np.clip(Xsel.reshape(npos, -1), 0, None); normsel = np.linalg.norm(Xsel_flat)
coph = []; disp = []; sil = []; evar = []
print("model selection (cophenetic/dispersion/silhouette/EV)...", flush=True)
for r in RANKS:
    Cm = np.zeros((SUB, SUB)); base = None
    for run in range(NRUN):
        m = NMF(n_components=r, init="random", max_iter=200, random_state=run, tol=1e-3)
        m.fit(np.clip(Xsel_aug, 0, None)); Wp = m.transform(Xsel_flat)
        if run == 0: base = (m, Wp)
        lab = Wp[sub].argmax(1); Cm += (lab[:, None] == lab[None, :])
    Cm /= NRUN; d = 1 - Cm; np.fill_diagonal(d, 0); cd = squareform(d, checks=False)
    coph.append(float(cophenet(linkage(cd, "average"), cd)[0])); disp.append(float(np.mean(4 * (Cm - .5) ** 2)))
    m0, W0 = base; evar.append(float(1 - (np.linalg.norm(Xsel_flat - W0 @ m0.components_) ** 2) / normsel ** 2))
    Ws = StandardScaler().fit_transform(W0); lab = KMeans(r, n_init=8, random_state=0).fit_predict(Ws)
    sil.append(float(silhouette_score(Ws, lab, sample_size=min(3000, npos), random_state=0)))
    print(f"  rank {r}: coph={coph[-1]:.3f} disp={disp[-1]:.3f} EV={evar[-1]:.3f} sil={sil[-1]:.3f}", flush=True)
coph = np.array(coph); knee = RANKS[int(np.argmax(coph[:-1] - coph[1:]))]; silK = RANKS[int(np.argmax(sil))]
print(f"\n[rank] cophenetic-knee={knee}  silhouette-max={silK}", flush=True)
fig = Figure(figsize=(11, 8), facecolor="white")
def pan(i, y, t, yl, mk=None):
    ax = fig.add_subplot(2, 2, i); ax.plot(RANKS, y, "o-", color="#1C7293", lw=2)
    if mk: ax.axvline(mk, color="#C0392B", ls="--", label=f"choice={mk}"); ax.legend(fontsize=9)
    ax.set_xlabel("NMF rank"); ax.set_ylabel(yl); ax.set_title(t, fontsize=11); ax.grid(alpha=.3)
pan(1, sil, "(a) Silhouette (weak/ambiguous)", "silhouette", silK)
pan(2, evar, "(b) Explained variance (elbow)", "EV")
pan(3, coph, "(c) Cophenetic correlation (stability)", "cophenetic r", knee)
pan(4, disp, "(d) Dispersion (assignment ambiguity)", "dispersion")
fig.suptitle(f"Na007b Polar+theta-shift NMF model selection (silhouette max {max(sil):.2f}; cophenetic knee = rank {knee})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96]); FigureCanvasAgg(fig); fig.savefig(f"{F}/na007b_polar_modelselection.png", dpi=150, facecolor="white")

# ---- final NMF on 96^2 at knee rank ----
print(f"\nfinal NMF rank={knee} on 96^2 (theta-aug fit)...", flush=True); t0 = time.time()
Xf = np.clip(X96.reshape(N, -1), 0, None); Xfa = np.clip(aug(X96), 0, None)
mf = NMF(n_components=knee, init="nndsvda", max_iter=400, random_state=0, tol=1e-4); mf.fit(Xfa)
Wf = mf.transform(Xf); Hf = mf.components_; Xr = Wf @ Hf
EV = float(1 - (np.linalg.norm(Xf - Xr) ** 2) / np.linalg.norm(Xf) ** 2)
print(f"  fit {time.time()-t0:.0f}s EV={EV:.3f}", flush=True)
np.save(f"{F}/na007b_polar_W.npy", Wf); np.save(f"{F}/na007b_polar_H.npy", Hf)
s = 96; nc = 5; nr = int(np.ceil(knee / nc))
def grid(mats, ttl, fn, scan=False, cmap="inferno"):
    g = Figure(figsize=(2.2 * nc, 2.2 * nr), facecolor="white")
    for k in range(knee):
        ax = g.add_subplot(nr, nc, k + 1); im = mats[k].reshape(Ny, Nx) if scan else mats[k].reshape(s, s)
        ax.imshow(im, cmap=cmap, interpolation="nearest", aspect="auto"); ax.set_title(f"{k}", fontsize=9)
        if not scan: ax.set_xlabel("r"); ax.set_ylabel("theta")
        ax.set_xticks([]); ax.set_yticks([])
    g.suptitle(ttl, fontsize=12); g.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(g); g.savefig(f"{F}/{fn}", dpi=140, facecolor="white")
grid(Hf, f"Na007b Polar+theta-shift NMF components (rank {knee}, EV={EV:.2f}) - (theta,r)", "na007b_polar_components.png")
grid(Wf.T, f"Na007b Polar+theta-shift NMF spatial loadings (rank {knee})", "na007b_polar_loadings.png", scan=True, cmap="viridis")
ex = [int(0.3 * N), int(0.5 * N), int(0.7 * N), int(0.85 * N)]; figr = Figure(figsize=(9, 3 * len(ex)), facecolor="white")
for j, i in enumerate(ex):
    o = Xf[i].reshape(s, s); rc = Xr[i].reshape(s, s); res = o - rc; vm = np.percentile(o, 99.5)
    for c, (img, tt, cm) in enumerate([(o, "original", ("inferno", (0, vm))), (rc, "reconstruction", ("inferno", (0, vm))), (res, "residual", ("RdBu_r", (-vm / 2, vm / 2)))]):
        ax = figr.add_subplot(len(ex), 3, 3 * j + c + 1); ax.imshow(img, cmap=cm[0], vmin=cm[1][0], vmax=cm[1][1], interpolation="nearest", aspect="auto")
        ax.set_title((f"pos {i} " + tt) if c == 0 else tt, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
figr.suptitle(f"Na007b Polar+theta-shift NMF reconstruction (rank {knee}, EV={EV:.3f}) - polar (theta,r)", fontsize=12)
figr.tight_layout(rect=[0, 0, 1, 0.96]); FigureCanvasAgg(figr); figr.savefig(f"{F}/na007b_polar_reconstruction.png", dpi=140, facecolor="white")

# ---- over/under-clustering (auto-K KMeans on loadings) ----
Wsf = StandardScaler().fit_transform(Wf); best = (-1, None, None)
for k in range(3, 19):
    lab = KMeans(k, n_init=8, random_state=0).fit_predict(Wsf); sc = silhouette_score(Wsf, lab, sample_size=4000, random_state=0)
    if sc > best[0]: best = (sc, k, lab)
nsil, Kn, nmf_lab = best
print(f"\nNMF auto cluster-K = {Kn} (silhouette {nsil:.3f}); DINO active K = {Kd}", flush=True)
np.save(f"{F}/na007b_polar_nmflabels.npy", nmf_lab.reshape(Ny, Nx))
# test1: DINO mutual distinctness (ring-masked corr on real class means)
H2 = dmean.shape[1]; cy = (H2 - 1) / 2; yy, xx = np.indices((H2, H2)); rr = np.sqrt((yy - cy) ** 2 + (xx - cy) ** 2); nb = H2 // 2
ring = (rr > 0.18 * nb) & (rr < 0.95 * nb)
def disp_(x):
    x = np.log1p(np.clip(x, 0, None)); v = x[ring]; return (x - v.min()) / (v.ptp() + 1e-9)
Dd = [disp_(m) for m in dmean]
rmat = np.eye(Kd)
for a, b in itertools.combinations(range(Kd), 2):
    rmat[a, b] = rmat[b, a] = float(np.corrcoef(Dd[a][ring], Dd[b][ring])[0, 1])
od = rmat[~np.eye(Kd, dtype=bool)]
print(f"[Test1] DINO mutual distinctness r: min/med/max={od.min():.3f}/{np.median(od):.3f}/{od.max():.3f}; "
      f"pairs r>0.97={[(a, b) for a, b in itertools.combinations(range(Kd), 2) if rmat[a, b] > 0.97]}", flush=True)
# test2: DINO classes inside each NMF cluster
print("[Test2] DINO classes inside each Polar+theta-shift NMF cluster:", flush=True)
ct = np.zeros((Kn, Kd), int)
for i in range(N):
    if flake[i]: ct[nmf_lab[i], asg[i]] += 1
under = 0
for c in range(Kn):
    tot = ct[c].sum()
    if tot == 0: continue
    pres = [d for d in range(Kd) if ct[c, d] > 0.04 * tot]
    if len(pres) >= 2:
        mr = min(rmat[a, b] for a, b in itertools.combinations(pres, 2)); u = mr < 0.9; under += u
        print(f"  cluster {c} (n={tot}) -> DINO {pres} min r={mr:.3f} {'UNDER' if u else 'sim'}", flush=True)
    else:
        print(f"  cluster {c} (n={tot}) -> DINO {pres} (1:1)", flush=True)
print(f"  => {under}/{Kn} NMF clusters merge diffraction-distinct DINO classes", flush=True)
# test3: crystallinity per class
def med(a, lab, K):
    return [round(float(np.nanmedian(a[lab == c])), 3) if (lab == c).any() else None for c in range(K)]
print(f"[Test3] peak/halo per DINO class: {med(ph, asg, Kd)}", flush=True)
print(f"        per NMF cluster: {med(ph, nmf_lab, Kn)}", flush=True)
print("\nDONE", flush=True)
