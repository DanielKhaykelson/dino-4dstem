"""Compute NMF + all clustering variants (k-means, agglomerative, GMM, HDBSCAN,
fuzzy c-means) once per IMC sample and cache the label maps + DINO assigns, so the
Boris-edit figures can be built without re-running NMF each time.
  python tools/boris_nmf_cache.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import register_runtime_sample
from gui_app.nmf_panel import build_nmf_input, fit_nmf, fuzzy_cmeans
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
try:
    import hdbscan; HAVE_HDB = True
except Exception:
    HAVE_HDB = False

OUT = "docs/paper/draft_v2/figs"
IMC = {
 "SI3": dict(run="runs/_gui/IMC_SI3_m097k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy", vmax=5.0, scan=(128, 128)),
 "SI4": dict(run="runs/_gui/IMC_SI4_m097_k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy", vmax=5.0, scan=(128, 128)),
 "SI5": dict(run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", path=r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy", vmax=5.0, scan=(128, 128)),
}
def bmean3(a, f): n, H, W = a.shape; s = H // f; return a.reshape(n, s, f, s, f).mean((2, 4))
def aug(Xs): s = Xs.shape[1]; return np.concatenate([np.roll(Xs, sh, 1) for sh in (0, s//4, s//2, 3*s//4)], 0).reshape(-1, s*s).astype(np.float32)

for name, c in IMC.items():
    t0 = time.time(); Ny, Nx = c["scan"]; N = Ny * Nx
    asg = np.load(os.path.join(c["run"], "eval", "inference.npz"))["assigns"].astype(int)
    K = int(len(np.unique(asg)))
    key = register_runtime_sample(c["path"], scan_shape=(Ny, Nx), vmax=c["vmax"])
    X, _, cs, _ = build_nmf_input(key, dict(input="polar", log=False, sparse=False, theta_shift=False), vmax_override=c["vmax"])
    P = cs[0]; Xd = bmean3(np.clip(X.reshape(N, P, P), 0, None).astype(np.float32), 4); del X
    W, _, _ = fit_nmf(Xd.reshape(N, -1), aug(Xd), n_components=min(2 * K, 30), max_iter=300)
    Ws = StandardScaler().fit_transform(W)
    lab = {}
    lab["kmeans"] = KMeans(K, n_init=10, random_state=0).fit_predict(Ws)
    lab["aglo"] = AgglomerativeClustering(K).fit_predict(Ws)
    lab["gmm"] = GaussianMixture(K, random_state=0, max_iter=200).fit(Ws).predict(Ws)
    lab["fcm"] = fuzzy_cmeans(Ws.astype(np.float32), K)[0].astype(int)
    if HAVE_HDB:
        lab["hdbscan"] = hdbscan.HDBSCAN(min_cluster_size=max(20, N // 60)).fit_predict(Ws)
    save = dict(dino=asg.reshape(Ny, Nx), scan=np.array([Ny, Nx]), K=K)
    for m, v in lab.items():
        save[m] = v.reshape(Ny, Nx)
    np.savez(os.path.join(OUT, f"boris_nmf_cache_{name}.npz"), **save)
    print(f"[{name}] K={K} methods={list(lab)} ({time.time()-t0:.0f}s)", flush=True)
print("wrote boris_nmf_cache_{SI3,SI4,SI5}.npz", flush=True)
