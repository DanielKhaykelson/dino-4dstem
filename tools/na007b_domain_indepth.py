"""Na007b in-depth domain analysis — is DINO's rest partition real (and better
than NMF)? Combines:
  (cos) Centroid cosine matrix of DINO classes (GUI method: per-class mean of the
        128-d embeds, L2-normalised, cn @ cn.T) -> which classes DINO sees as close.
  (A)   Pairwise single-frame SEPARABILITY in a NEUTRAL space (polar patterns ->
        PCA-50, 5-fold CV balanced accuracy, linear). ~0.5 = redundant (merge),
        >>0.5 = a real learnable difference. Done for DINO rest AND NMF rest.
  (B)   Difference patterns for the most-similar pairs (is the split physical?).
  (C)   Scaffold/flake conflation: HAADF-Otsu largest-CC flake mask; per-cluster
        fraction inside the flake -> a cluster straddling flake+off-flake conflates.
DINO uses existing inference (no retraining). NMF = polar+theta-shift K=6."""
import os, sys, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import register_runtime_sample
from gui_app.nmf_panel import build_nmf_input
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from skimage.filters import threshold_otsu
from skimage.measure import label
from scipy.ndimage import binary_fill_holes, binary_closing
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

F = r"docs/explainer/figs"; RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
Ny, Nx = 126, 100; N = Ny * Nx
inf = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)
asg = inf["assigns"].astype(int); embeds = inf["embeds"].astype(np.float32)
asg2 = asg.reshape(Ny, Nx); flake = (asg2 != 0); dino_line = np.isin(asg2, [1, 8]) & flake
Kd = int(asg.max()) + 1
dmean = np.load(f"{F}/na007b_dino_classmeans.npy")
HA = np.load(f"{F}/na007b_HAADF_gui.npy")
Wl = np.load(f"{F}/na007b_polar_W.npy"); KNMF = 6
nmf = KMeans(KNMF, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(Wl)).reshape(Ny, Nx)
ovl = [((nmf == c) & dino_line).sum() / max((nmf == c).sum(), 1) for c in range(KNMF)]
nmf_line_c = int(np.argmax(ovl))
dino_rest = [c for c in range(Kd) if c not in (0, 1, 8) and ((asg2 == c) & flake).sum() > 30]
nmf_rest = [c for c in range(KNMF) if c != nmf_line_c and ((nmf == c) & flake).sum() > 30]

# (cos) DINO centroid cosine matrix
cen = np.zeros((Kd, embeds.shape[1]), np.float32)
for c in range(Kd):
    m = asg == c
    if m.any(): cen[c] = embeds[m].mean(0)
cn = cen / np.linalg.norm(cen, axis=1, keepdims=True).clip(1e-9)
COS = cn @ cn.T
print("[cos] DINO rest centroid cosine (high = DINO sees them close):", flush=True)
for a, b in itertools.combinations(dino_rest, 2):
    if COS[a, b] > 0.85: print(f"   D{a}-D{b}: cos={COS[a,b]:.3f}", flush=True)

# polar single-frame features (neutral) -> PCA-50
key = register_runtime_sample(r"D:/DINOSR/data/Na007b_nbed.cube.npy", scan_shape=(Ny, Nx), vmax=2.0, center_mask_radius=22)
print("build_nmf_input (polar, neutral features)...", flush=True); t0 = time.time()
X, _, cs, _ = build_nmf_input(key, dict(input="polar", log=False, sparse=False, theta_shift=False), vmax_override=2.0)
P = cs[0]; X = X.reshape(N, P, P).astype(np.float32)
def bmean3(a, f): n, H, W = a.shape; s = H // f; return a.reshape(n, s, f, s, f).mean((2, 4))
Xp = bmean3(X, 4).reshape(N, -1)                       # 48^2 neutral features
pca = PCA(50, random_state=0).fit(Xp[flake.ravel()]); Pf = pca.transform(Xp)
print(f"  features ready {time.time()-t0:.0f}s; PCA-50 EV={pca.explained_variance_ratio_.sum():.2f}", flush=True)

# (A) pairwise separability
def sep_matrix(labmap, classes, cap=500):
    lab = labmap.ravel(); idxall = {c: np.where((lab == c) & flake.ravel())[0] for c in classes}
    rng = np.random.RandomState(0)
    M = np.full((len(classes), len(classes)), np.nan)
    for ia, a in enumerate(classes):
        for ib in range(ia + 1, len(classes)):
            b = classes[ib]
            ia_, ib_ = idxall[a], idxall[b]
            if len(ia_) > cap: ia_ = rng.choice(ia_, cap, False)
            if len(ib_) > cap: ib_ = rng.choice(ib_, cap, False)
            idx = np.concatenate([ia_, ib_]); y = np.concatenate([np.zeros(len(ia_)), np.ones(len(ib_))])
            clf = LogisticRegression(max_iter=500, class_weight="balanced")
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            acc = cross_val_score(clf, Pf[idx], y, cv=cv, scoring="balanced_accuracy").mean()
            M[ia, ib] = M[ib, ia] = acc
    np.fill_diagonal(M, 1.0); return M
print("separability (DINO rest)...", flush=True); Ad = sep_matrix(asg2, dino_rest)
print("separability (NMF rest)...", flush=True); An = sep_matrix(nmf, nmf_rest)
def offdiag(M): return M[~np.eye(M.shape[0], dtype=bool)]
print(f"[A] DINO rest separability balanced-acc: min/med/max = {offdiag(Ad).min():.2f}/{np.median(offdiag(Ad)):.2f}/{offdiag(Ad).max():.2f}", flush=True)
print(f"    NMF  rest separability balanced-acc: min/med/max = {offdiag(An).min():.2f}/{np.median(offdiag(An)):.2f}/{offdiag(An).max():.2f}", flush=True)
merge = [(dino_rest[i], dino_rest[j], round(Ad[i, j], 2), round(COS[dino_rest[i], dino_rest[j]], 2))
         for i in range(len(dino_rest)) for j in range(i + 1, len(dino_rest)) if Ad[i, j] < 0.65]
print(f"[A] DINO merge candidates (sep<0.65) [a,b,sep,cos]: {merge}", flush=True)
real = sum(1 for i in range(len(dino_rest)) for j in range(i + 1, len(dino_rest)) if Ad[i, j] >= 0.75)
tot = len(dino_rest) * (len(dino_rest) - 1) // 2
print(f"[A] DINO rest pairs clearly distinct (sep>=0.75): {real}/{tot}", flush=True)

# (C) scaffold/flake mask
thr = threshold_otsu(HA); fl = binary_fill_holes(binary_closing(HA > thr, iterations=1))
lb = label(fl); flake_hi = lb == (np.bincount(lb.ravel())[1:].argmax() + 1) if lb.max() else fl
def fr(mask): return (mask & flake_hi).sum() / max(mask.sum(), 1)
print("[C] per-cluster fraction inside HAADF flake (0=off-flake/scaffold, 1=on-flake):", flush=True)
dfr = {c: round(fr((asg2 == c) & flake), 2) for c in dino_rest}
nfr = {c: round(fr((nmf == c) & flake), 2) for c in nmf_rest}
print(f"    DINO: {dfr}", flush=True)
print(f"    NMF : {nfr}  (values near 0.5 straddle flake+scaffold = conflation)", flush=True)

# ---- figures ----
def heat(ax, M, classes, ttl, prefix, vmin, vmax, cmap):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([f"{prefix}{c}" for c in classes], fontsize=8, rotation=90)
    ax.set_yticklabels([f"{prefix}{c}" for c in classes], fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6,
                                              color="white" if (M[i, j] < (vmin+vmax)/2) == (cmap != "viridis") else "black")
    ax.set_title(ttl, fontsize=10)
fig = Figure(figsize=(15, 4.6), facecolor="white")
COSr = np.array([[COS[a, b] for b in dino_rest] for a in dino_rest])
heat(fig.add_subplot(1, 3, 1), COSr, dino_rest, "DINO centroid cosine (embed space)\nhigh=close", "D", 0.5, 1.0, "magma")
heat(fig.add_subplot(1, 3, 2), Ad, dino_rest, "DINO rest single-frame separability\n(0.5=redundant, 1=distinct)", "D", 0.5, 1.0, "viridis")
heat(fig.add_subplot(1, 3, 3), An, nmf_rest, "NMF rest single-frame separability\n(control)", "N", 0.5, 1.0, "viridis")
fig.suptitle("Na007b: are DINO rest splits real? centroid cosine vs neutral-space separability", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(fig); fig.savefig(f"{F}/na007b_domain_separability.png", dpi=160, facecolor="white")

# (B) difference patterns for the most-similar DINO pairs (lowest sep)
pairs = sorted([(Ad[i, j], dino_rest[i], dino_rest[j]) for i in range(len(dino_rest)) for j in range(i + 1, len(dino_rest))])[:6]
def crop(m): a = np.log1p(np.clip(m, 0, None)); H = a.shape[0]; return a[H//2-130:H//2+130, H//2-130:H//2+130]
figd = Figure(figsize=(3 * len(pairs), 3.2), facecolor="white")
for k, (acc, a, b) in enumerate(pairs):
    ca, cb = crop(dmean[a]), crop(dmean[b]); d = ca / (ca.max()+1e-9) - cb / (cb.max()+1e-9)
    ax = figd.add_subplot(1, len(pairs), k + 1); v = np.percentile(np.abs(d), 99)
    ax.imshow(d, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title(f"D{a} - D{b}\nsep={acc:.2f} cos={COS[a,b]:.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
figd.suptitle("Na007b: difference of average patterns for the most-similar DINO rest pairs (is the split physical?)", fontsize=11)
figd.tight_layout(rect=[0, 0, 1, 0.9]); FigureCanvasAgg(figd); figd.savefig(f"{F}/na007b_domain_diffpatterns.png", dpi=150, facecolor="white")

# (C) flake mask + per-cluster fraction bars
figc = Figure(figsize=(11, 4.2), facecolor="white")
ax = figc.add_subplot(1, 3, 1); lo, hi = np.percentile(HA, 1), np.percentile(HA, 99)
ax.imshow(np.clip(HA, lo, hi), cmap="gray"); ax.contour(flake_hi, levels=[0.5], colors="cyan", linewidths=1.2)
ax.set_title("HAADF + flake mask (cyan)", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
ax = figc.add_subplot(1, 3, 2); ax.bar(range(len(dino_rest)), [dfr[c] for c in dino_rest], color="#1f77b4")
ax.set_xticks(range(len(dino_rest))); ax.set_xticklabels([f"D{c}" for c in dino_rest], fontsize=8); ax.axhline(0.5, color="r", ls="--")
ax.set_ylim(0, 1); ax.set_ylabel("frac on flake"); ax.set_title("DINO rest: fraction on flake", fontsize=10)
ax = figc.add_subplot(1, 3, 3); ax.bar(range(len(nmf_rest)), [nfr[c] for c in nmf_rest], color="#2ca02c")
ax.set_xticks(range(len(nmf_rest))); ax.set_xticklabels([f"N{c}" for c in nmf_rest], fontsize=8); ax.axhline(0.5, color="r", ls="--")
ax.set_ylim(0, 1); ax.set_title("NMF rest: fraction on flake (≈0.5 = scaffold+flake conflation)", fontsize=10)
figc.suptitle("Na007b: scaffold/flake conflation test (independent HAADF flake mask)", fontsize=12)
figc.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(figc); figc.savefig(f"{F}/na007b_domain_scaffold.png", dpi=150, facecolor="white")
print("wrote na007b_domain_{separability,diffpatterns,scaffold}.png  DONE", flush=True)
