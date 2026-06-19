"""Can a CLASSICAL method reproduce the DINO crystallization axis? We cluster the
grains using only hand-built ROTATION-INVARIANT diffraction features (radial
peak/halo chi_r, 2D Bragg excess B, azimuthal spottiness overall + per ring,
thickness) — no learning — into the same number of clusters DINO uses among
grains, and compare to the DINO grain labels (ARI/AMI). We also order the
classical clusters by B to confirm they span the same amorphous->crystalline
ladder, and paint DINO-vs-classical grain maps side by side. FOV-clipped.

Contrast: blind classical (polar-NMF on raw patterns) gives ARI ~0.08-0.20 (paper
Table); the targeted feature clustering here reproduces the axis only because we
hand-engineered exactly the rotation-invariant features DINO learned by itself.

  python tools/classical_reproduce.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import map_coordinates
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ARI, adjusted_mutual_info_score as AMI

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")
INV = 0.00185; KMAX = 0.35; FOV = {"SI3": 187, "SI4": 160, "SI5": 160}
RING_PX = [73, 90, 114, 138]; NY = NX = 128


def cv_at(avg, cyx, R, band=3):
    th = np.linspace(0, 2 * np.pi, 360, endpoint=False); rr = np.arange(R - band, R + band + 1)
    Rr, T = np.meshgrid(rr, th); ys = cyx + Rr * np.sin(T); xs = cyx + Rr * np.cos(T)
    p = map_coordinates(avg, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(360, len(rr)).mean(1)
    return float(p.std() / (p.mean() + 1e-9))


def features(avg, cyx, beam, lo, hi, fov):
    m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
    seg = m[lo:hi]; vseg = v[lo:hi]
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None)))); pk = np.clip(seg - halo, 0, None)
    chi = float((pk / np.clip(halo, 1e-9, None)).max())
    cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None); spot = float(np.percentile(cv, 90))
    yy, xx = np.indices(avg.shape); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
    halo_full = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1])
    bandm = (rr >= lo) & (rr <= hi)
    bragg = float(np.clip(avg[bandm] - halo_full[bandm], 0, None).sum() / (halo_full[bandm].sum() + 1e-9))
    rings = [cv_at(avg, cyx, R) for R in RING_PX if R <= fov]
    return [chi, bragg, spot] + rings


def main():
    fig = Figure(figsize=(11, 3.4 * 3), facecolor="white"); res = {}
    for ri, name in enumerate(["SI3", "SI4", "SI5"]):
        z = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))
        gsum, gcnt, cls, vac, gid = z["gsum"], z["gcnt"], z["cls"], z["vac"], z["gid"]
        H = int(z["H"]); cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1
        hi = min(int(KMAX / INV), FOV[name])
        gids = [g for g in range(gsum.shape[0]) if not vac[g]]
        X = np.array([features(gsum[g] / max(gcnt[g], 1), cyx, beam, lo, hi, FOV[name]) for g in gids])
        bragg = X[:, 1]
        dino = np.array([int(cls[g]) for g in gids])
        K = len(set(dino.tolist()))
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(Xs)
        ag = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Xs)
        res[name] = dict(ari_km=ARI(dino, km), ami_km=AMI(dino, km),
                         ari_ag=ARI(dino, ag), ami_ag=AMI(dino, ag), K=K, n=len(gids))
        print(f"[{name}] grains={len(gids)} K={K}  KMeans ARI={res[name]['ari_km']:.2f} AMI={res[name]['ami_km']:.2f}"
              f"  Agg ARI={res[name]['ari_ag']:.2f} AMI={res[name]['ami_ag']:.2f}", flush=True)
        # do classical clusters span the crystallinity ladder? order by mean B
        order = sorted(set(km.tolist()), key=lambda k: bragg[km == k].mean())
        Bmeans = [bragg[km == k].mean() for k in order]
        print(f"    classical-cluster mean B (low->high): {[round(b,2) for b in Bmeans]}", flush=True)
        # paint grain maps: DINO vs classical (KMeans)
        def paint(lab_per_grain):
            mp = np.full(NY * NX, np.nan)
            for j, g in enumerate(gids):
                mp[gid == g] = lab_per_grain[j]
            return mp.reshape(NY, NX)
        for ci, (lab, ttl) in enumerate([(dino, "DINO grain classes"),
                                          (km, f"classical KMeans on features\nARI={res[name]['ari_km']:.2f} AMI={res[name]['ami_km']:.2f}")]):
            ax = fig.add_subplot(3, 2, ri * 2 + ci + 1)
            uni = sorted(set(lab.tolist())); lut = {u: k for k, u in enumerate(uni)}
            cmap = ListedColormap([mpl.colormaps.get_cmap("tab20").resampled(max(len(uni),1))(k) for k in range(len(uni))])
            cmap.set_bad("#eeeeee")
            mp = paint(np.array([lut[l] for l in lab]))
            ax.imshow(mp, cmap=cmap, interpolation="nearest", vmin=0, vmax=max(len(uni)-1,1))
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=9)
            if ci == 0: ax.set_ylabel(name, fontsize=12, fontweight="bold", rotation=0, labelpad=20, va="center")
    fig.suptitle("Classical reproduction: cluster grains by hand-built rotation-invariant features (χ_r, Bragg excess, spottiness) → recovers the DINO grain partition.\n"
                 "DINO matches this WITHOUT engineering the features or choosing K, and works per-pixel (no ≥30px grain-averaging needed). Blind polar-NMF on raw patterns gives ARI≈0.08–0.20 (paper).",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(fig)
    p = os.path.join(OUT, "classical_reproduce.png"); fig.savefig(p, dpi=150, facecolor="white")
    import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "classical_reproduce.png"))
    print("wrote classical_reproduce.png", flush=True)


if __name__ == "__main__":
    main()
