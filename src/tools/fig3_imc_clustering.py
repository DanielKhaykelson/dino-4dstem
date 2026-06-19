"""Figure 3: IMC SI3/SI4/SI5 — DINO vs NMF under MULTIPLE clustering methods.
Polar+theta-shift NMF features -> {KMeans, Agglomerative, GaussianMixture,
HDBSCAN} -> ARI/AMI vs DINO. Shows NMF (a) fails to reproduce DINO and (b) is
highly sensitive to the clustering choice, while DINO is a single stable map."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib.cm as cmx
from data import register_runtime_sample
from gui_app.nmf_panel import build_nmf_input, fit_nmf
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ARI, adjusted_mutual_info_score as AMI
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
OUT = "docs/paper/draft_v2/figs"
IMC = {
 "SI3": dict(run="runs/_gui/IMC_SI3_m097k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy", vmax=5.0, scan=(128, 128)),
 "SI4": dict(run="runs/_gui/IMC_SI4_m097_k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy", vmax=5.0, scan=(128, 128)),
 "SI5": dict(run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", path=r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy", vmax=5.0, scan=(128, 128)),
}
try:
    import hdbscan; HAVE_HDB = True
except Exception:
    HAVE_HDB = False
def bmean3(a, f): n, H, W = a.shape; s = H // f; return a.reshape(n, s, f, s, f).mean((2, 4))
def aug(Xs): s = Xs.shape[1]; return np.concatenate([np.roll(Xs, sh, 1) for sh in (0, s//4, s//2, 3*s//4)], 0).reshape(-1, s*s).astype(np.float32)
res = {}; maps = {}
for name, c in IMC.items():
    t0 = time.time(); Ny, Nx = c["scan"]; N = Ny * Nx
    asg = np.load(os.path.join(c["run"], "eval", "inference.npz"))["assigns"].astype(int)
    K = int(len(np.unique(asg)))
    key = register_runtime_sample(c["path"], scan_shape=(Ny, Nx), vmax=c["vmax"])
    X, _, cs, _ = build_nmf_input(key, dict(input="polar", log=False, sparse=False, theta_shift=False), vmax_override=c["vmax"])
    P = cs[0]; Xd = bmean3(np.clip(X.reshape(N, P, P), 0, None).astype(np.float32), 4); del X
    W, _, _ = fit_nmf(Xd.reshape(N, -1), aug(Xd), n_components=min(2*K, 30), max_iter=300)
    Ws = StandardScaler().fit_transform(W)
    methods = {}
    methods["KMeans"] = KMeans(K, n_init=8, random_state=0).fit_predict(Ws)
    methods["Agglomerative"] = AgglomerativeClustering(K).fit_predict(Ws)
    methods["GaussianMixture"] = GaussianMixture(K, random_state=0, max_iter=200).fit(Ws).predict(Ws)
    if HAVE_HDB:
        methods["HDBSCAN"] = hdbscan.HDBSCAN(min_cluster_size=max(20, N // 60)).fit_predict(Ws)
    res[name] = {m: dict(ARI=round(float(ARI(asg, lab)), 3), AMI=round(float(AMI(asg, lab)), 3),
                          K_found=int(len(set(lab[lab >= 0])))) for m, lab in methods.items()}
    maps[name] = dict(dino=asg.reshape(Ny, Nx), **{m: lab.reshape(Ny, Nx) for m, lab in methods.items()}, scan=(Ny, Nx))
    print(f"[{name}] K={K} " + " ".join(f"{m}:ARI={res[name][m]['ARI']}" for m in methods) + f"  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(os.path.join(OUT, "fig3_clustering_sensitivity.json"), "w"), indent=2)

# ---- figure: (a) ARI bars per sample/method  (b) SI5 maps under each method + DINO ----
mlist = ["KMeans", "Agglomerative", "GaussianMixture"] + (["HDBSCAN"] if HAVE_HDB else [])
fig = Figure(figsize=(13, 7.5), facecolor="white")
ax = fig.add_subplot(2, 1, 1); x = np.arange(len(IMC)); w = 0.8 / len(mlist)
for i, m in enumerate(mlist):
    ax.bar(x + i*w, [res[s][m]["ARI"] for s in IMC], w, label=m)
ax.axhspan(0, 0.3, color="#FDEDEC", zorder=0)
ax.set_xticks(x + 0.4 - w/2); ax.set_xticklabels(list(IMC.keys())); ax.set_ylabel("ARI vs DINO")
ax.set_title("(a) NMF reproduction of DINO across clustering methods — all low (fails) and method-dependent (spread)", fontsize=11)
ax.legend(fontsize=9, ncol=len(mlist)); ax.set_ylim(0, max(0.3, max(res[s][m]["ARI"] for s in IMC for m in mlist) + 0.05))
S = "SI5"; Ny, Nx = maps[S]["scan"]; panels = ["dino"] + mlist
for j, p in enumerate(panels):
    axm = fig.add_subplot(2, len(panels), len(panels) + j + 1)
    lab = maps[S][p]; axm.imshow(lab, cmap="tab20", interpolation="nearest")
    ttl = "DINO (stable)" if p == "dino" else f"NMF + {p}\nARI={res[S][p]['ARI']}"
    axm.set_title(ttl, fontsize=9); axm.set_xticks([]); axm.set_yticks([])
fig.suptitle("Figure 3 — IMC: DINO vs NMF under multiple clustering methods (maps shown for SI5)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig); fig.savefig(os.path.join(OUT, "fig3_imc_clustering.png"), dpi=150, facecolor="white")

# ---- main-text grid: rows = samples, cols = DINO + each NMF clustering variant ----
from matplotlib.colors import ListedColormap
import matplotlib as mpl
ROLE = {"SI3": "overview (matrix + needles)", "SI4": "needles", "SI5": "needle / matrix interface"}
def show(ax, lab):
    uni = sorted(np.unique(lab).tolist()); has_noise = (-1 in uni)
    cols = []
    for k, u in enumerate(uni):
        cols.append((0.7, 0.7, 0.7) if u == -1 else mpl.colormaps.get_cmap("tab20").resampled(max(len([x for x in uni if x != -1]), 1))(k - (1 if has_noise else 0)))
    lut = {u: k for k, u in enumerate(uni)}
    ax.imshow(np.vectorize(lut.get)(lab).astype(float), cmap=ListedColormap(cols),
              interpolation="nearest", vmin=0, vmax=max(len(uni) - 1, 1))
    ax.set_xticks([]); ax.set_yticks([])
panels = ["dino"] + mlist
g = Figure(figsize=(1.95 * len(panels), 2.15 * 3 + 0.7), facecolor="white")
for ri, name in enumerate(IMC):
    for ci, p in enumerate(panels):
        ax = g.add_subplot(3, len(panels), ri * len(panels) + ci + 1)
        show(ax, maps[name][p])
        if ri == 0:
            ax.set_title("DINO" if p == "dino" else f"NMF + {p}", fontsize=9.5)
        if p != "dino":
            ax.text(0.5, -0.07, f"ARI = {res[name][p]['ARI']:.2f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8.5, color="black")
        if ci == 0:
            ax.set_ylabel(f"{name}\n{ROLE[name]}", fontsize=10, fontweight="bold")
g.suptitle("IMC: self-supervised (DINO) vs NMF class maps for the three fields of view. DINO is one stable map per sample; NMF is shown clustered by "
           "four standard algorithms.\nARI (agreement with DINO, 1=identical, 0=chance) stays low and varies with the algorithm — NMF does not reproduce the DINO map.",
           fontsize=9.5)
g.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(g)
gp = os.path.join("docs/explainer/figs", "imc_dino_nmf_maps.png"); g.savefig(gp, dpi=170, facecolor="white")
import shutil; shutil.copy(gp, "docs/paper/draft_v2/figs/latest_review/imc_dino_nmf_maps.png")
print("wrote imc_dino_nmf_maps.png (variant grid)", flush=True)
print("\nSUMMARY:", json.dumps(res, indent=1), flush=True)
print(f"HDBSCAN available: {HAVE_HDB}; wrote fig3_imc_clustering.png", flush=True)
