"""BATCH: redo the classical NMF arm with the correct Polar+theta-shift (no-log)
variant and compare to DINO, for all 5 samples (IMC SI3/SI4/SI5, NaPHI, EuInAs).
Protocol matches the old classical baseline: NMF(n_comp=min(2K,30)) on polar
features -> KMeans(K_active) -> ARI/AMI vs DINO assigns (whole map). theta-roll
done at downsampled resolution to bound memory. Writes results JSON + per-sample
NMF label maps. (DINO assigns unchanged — existing inference.)"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import SAMPLES, register_runtime_sample
from gui_app.nmf_panel import build_nmf_input, fit_nmf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

OUT = r"docs/explainer/figs"
RUNS = {
    "IMC_SI3": "runs/_gui/IMC_SI3_m097k60",
    "IMC_SI4": "runs/_gui/IMC_SI4_m097_k60",
    "IMC_SI5": "runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60",
    "NaPHI_Na007b": "runs/_gui/Na007b_k60_m097_vmax2",
    "EuInAs": "runs/_sweep_m_K_20260525_213539/EuInAs_B100/stage2/m0.9700_seed42_K60",
}

def resolve(rundir):
    tk = os.path.join(rundir, "_train_kwargs.json")
    if os.path.exists(tk):
        d = json.load(open(tk)); sc = d["_sample_config"]
        return sc["path"], float(sc.get("vmax", 5.0)), tuple(sc["scan_shape"])
    rs = json.load(open(os.path.join(rundir, "run_summary.json")))
    key = rs.get("sample"); c = rs.get("cfg") or {}; s = SAMPLES.get(key, {})
    path = c.get("path") or s.get("path")
    vmax = float(c.get("vmax") or s.get("vmax") or 5.0)
    scan = c.get("scan_shape") or s.get("scan_shape") or s.get("scan_size")
    return path, vmax, tuple(scan)

def bmean3(a, f): n, H, W = a.shape; s = H // f; return a.reshape(n, s, f, s, f).mean((2, 4))
def aug(Xs):
    s = Xs.shape[1]
    return np.concatenate([np.roll(Xs, sh, axis=1) for sh in (0, s // 4, s // 2, 3 * s // 4)], 0).reshape(-1, s * s).astype(np.float32)

results = {}
for name, rundir in RUNS.items():
    t0 = time.time()
    try:
        path, vmax, scan = resolve(rundir)
        Ny, Nx = int(scan[0]), int(scan[1]); N = Ny * Nx
        inf = np.load(os.path.join(rundir, "eval", "inference.npz"), allow_pickle=True)
        asg = inf["assigns"].astype(int)
        K = int(len(np.unique(asg)))
        key = register_runtime_sample(path, scan_shape=(Ny, Nx), vmax=vmax)
        print(f"[{name}] path={os.path.basename(path)} N={N} K_active={K} vmax={vmax}", flush=True)
        X, _, cs, _ = build_nmf_input(key, dict(input="polar", log=False, sparse=False, theta_shift=False), vmax_override=vmax)
        P = cs[0]; Xd = bmean3(np.clip(X.reshape(N, P, P), 0, None).astype(np.float32), 4); del X
        Xflat = Xd.reshape(N, -1); Xaug = aug(Xd)
        ncomp = min(2 * K, 30)
        W, H, err = fit_nmf(Xflat, Xaug, n_components=ncomp, max_iter=300)
        lab = KMeans(K, n_init=8, random_state=0).fit_predict(StandardScaler().fit_transform(W))
        ari = float(adjusted_rand_score(asg, lab)); ami = float(adjusted_mutual_info_score(asg, lab))
        results[name] = dict(K=K, N=N, ncomp=ncomp, ARI_new=round(ari, 3), AMI_new=round(ami, 3), secs=round(time.time() - t0))
        np.save(os.path.join(OUT, f"polarnmf_labels_{name}.npy"), lab.reshape(Ny, Nx))
        print(f"[{name}] DONE  polar+theta-shift NMF vs DINO: ARI={ari:.3f} AMI={ami:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        results[name] = dict(error=repr(e)); print(f"[{name}] FAILED: {e!r}", flush=True)

json.dump(results, open(os.path.join(OUT, "polar_nmf_vs_dino.json"), "w"), indent=2)
print("\n=== SUMMARY (polar+theta-shift NMF vs DINO) ===", flush=True)
for k, v in results.items(): print(f"  {k}: {v}", flush=True)
print("wrote polar_nmf_vs_dino.json", flush=True)
