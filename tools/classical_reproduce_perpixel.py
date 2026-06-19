"""Fully-independent classical baseline at PER-PIXEL resolution (no DINO grains).
On every single frame we compute the same rotation-invariant features used on the
grains (radial peak/halo chi_r, 2D Bragg excess, azimuthal spottiness overall +
per ring, total scattered intensity), denoised exactly like the analysis (vmax=2
clip + gaussian sigma=2), then KMeans into the DINO active-K and compare the class
map to DINO (ARI/AMI over the full scan and over grain pixels). This is the honest
per-pixel test: single low-dose frames are shot-noise-limited, so we expect the
classical map to be much noisier than DINO's.

  python tools/classical_reproduce_perpixel.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import gaussian_filter
from data import open_lazy_cube
from gui_app.crystallinity_panel import _snip_baseline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ARI, adjusted_mutual_info_score as AMI
try:
    from skimage.filters import threshold_otsu
except Exception:
    threshold_otsu = None

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"; REVIEW = os.path.join(FIGS, "latest_review")
INV = 0.00185; KMAX = 0.35; VMAX = 2.0; BLUR = 2.0; NY = NX = 128
RING_PX = [73, 90, 114, 138]; FOV = {"SI3": 187, "SI4": 160, "SI5": 160}
IMC = {
 "SI3": dict(run="runs/_gui/IMC_SI3_m097k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy"),
 "SI4": dict(run="runs/_gui/IMC_SI4_m097_k60", path=r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy"),
 "SI5": dict(run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", path=r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"),
}


def feat_frame(fr, ridx, counts, beam, lo, hi, ring_bins, fov):
    f = gaussian_filter(np.clip(fr, 0, VMAX).astype(np.float32), BLUR)
    flat = f.ravel()
    s = np.bincount(ridx, weights=flat, minlength=counts.size)
    s2 = np.bincount(ridx, weights=flat * flat, minlength=counts.size)
    m = s / counts; var = np.clip(s2 / counts - m * m, 0, None)
    seg = m[lo:hi]; tot = float(seg.sum())
    if tot <= 0:
        return None
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
    pk = np.clip(seg - halo, 0, None)
    chi = float((pk / np.clip(halo, 1e-9, None)).max())
    cv = np.sqrt(var[lo:hi]) / np.clip(seg, 1e-9, None)
    spot = float(np.percentile(cv, 90))
    # 2D Bragg excess over the band
    band = (ridx >= lo) & (ridx <= hi)
    halo_full = np.interp(ridx[band], np.arange(lo, hi), halo, left=halo[0], right=halo[-1])
    bragg = float(np.clip(flat[band] - halo_full, 0, None).sum() / (halo_full.sum() + 1e-9))
    rings = [float(np.sqrt(var[R]) / (m[R] + 1e-9)) for R in ring_bins if R <= fov]
    return [chi, bragg, spot, np.log1p(tot)] + rings


def run_sample(name, c):
    t0 = time.time()
    asg = np.load(os.path.join(c["run"], "eval", "inference.npz"))["assigns"].astype(int)
    K = len(np.unique(asg))
    gid = np.load(os.path.join(FIGS, f"grain_acom_v2_{name}.npz"))["gid"]
    cube = open_lazy_cube(c["path"], scan_shape=(NY, NX)); _, _, H, W = cube.shape
    cy = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV[name])
    yy, xx = np.indices((H, W)); ridx = np.round(np.sqrt((yy - cy) ** 2 + (xx - cy) ** 2)).astype(int).ravel()
    nb = ridx.max() + 1; counts = np.clip(np.bincount(ridx, minlength=nb).astype(float), 1, None)
    feats = np.zeros((NY * NX, 4 + sum(R <= FOV[name] for R in RING_PX)), np.float32)
    valid = np.zeros(NY * NX, bool)
    for rx in range(NY):
        blk = np.asarray(cube[rx], np.float32)
        for ry in range(NX):
            fv = feat_frame(blk[ry], ridx, counts, beam, lo, hi, RING_PX, FOV[name])
            i = rx * NX + ry
            if fv is not None:
                feats[i] = fv; valid[i] = True
    Xs = StandardScaler().fit_transform(feats[valid])
    z = np.full((NY * NX, feats.shape[1]), np.nan, np.float32); z[valid] = Xs
    lab = np.full(NY * NX, -1)
    lab[valid] = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(Xs)
    gm = gid >= 0
    ari_all = ARI(asg[valid], lab[valid]); ami_all = AMI(asg[valid], lab[valid])
    gv = gm & valid
    ari_gr = ARI(asg[gv], lab[gv]); ami_gr = AMI(asg[gv], lab[gv])
    print(f"[{name}] K={K} px={int(valid.sum())}  ALL ARI={ari_all:.2f} AMI={ami_all:.2f} | "
          f"GRAINS ARI={ari_gr:.2f} AMI={ami_gr:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    fnames = ["chi_r (peak/halo)", "Bragg excess", "spottiness (90pct CV)", "log total intensity"] + \
             [f"CV {d}Å" for d, R in zip([7.4, 6.0, 4.75, 3.9], RING_PX) if R <= FOV[name]]
    return dict(asg=asg.reshape(NY, NX), lab=lab.reshape(NY, NX), K=K,
                ari_all=ari_all, ami_all=ami_all, ari_gr=ari_gr, ami_gr=ami_gr,
                z=z.reshape(NY, NX, -1), fnames=fnames)


def main():
    res = {n: run_sample(n, IMC[n]) for n in IMC}
    fig = Figure(figsize=(11, 3.4 * 3), facecolor="white")
    for ri, n in enumerate(IMC):
        r = res[n]
        for ci, (mp, ttl) in enumerate([(r["asg"], "DINO (per-pixel)"),
                (r["lab"], f"classical KMeans per-pixel\nALL ARI={r['ari_all']:.2f} | grains ARI={r['ari_gr']:.2f}")]):
            ax = fig.add_subplot(3, 2, ri * 2 + ci + 1); ax.set_xticks([]); ax.set_yticks([])
            uni = sorted(set(mp[mp >= 0].tolist())); lut = {u: k for k, u in enumerate(uni)}
            cmap = ListedColormap([mpl.colormaps.get_cmap("tab20").resampled(max(len(uni), 1))(k) for k in range(len(uni))])
            cmap.set_bad("#dddddd")
            disp = np.vectorize(lambda v: lut.get(v, -1))(mp).astype(float); disp[disp < 0] = np.nan
            ax.imshow(disp, cmap=cmap, interpolation="nearest", vmin=0, vmax=max(len(uni) - 1, 1))
            ax.set_title(ttl, fontsize=9)
            if ci == 0: ax.set_ylabel(n, fontsize=12, fontweight="bold", rotation=0, labelpad=20, va="center")
    fig.suptitle("Fully-independent PER-PIXEL classical baseline (no DINO grains): KMeans on single-frame rotation-invariant features (vmax=2, σ=2 blur) vs DINO.\n"
                 "Single low-dose frames are shot-noise-limited → the classical per-pixel map is salt-and-pepper; DINO produces coherent domains from the same frames.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(fig)
    p = os.path.join(OUT, "classical_reproduce_perpixel.png"); fig.savefig(p, dpi=150, facecolor="white")
    import shutil; os.makedirs(REVIEW, exist_ok=True); shutil.copy(p, os.path.join(REVIEW, "classical_reproduce_perpixel.png"))
    print("wrote classical_reproduce_perpixel.png", flush=True)

    def grid_fig(feat_idx, fname, suptitle):
        nc = len(feat_idx)
        f = Figure(figsize=(2.2 * nc, 2.35 * 3 + 0.6), facecolor="white")
        for ri, n in enumerate(IMC):
            zz = res[n]["z"]; nm = res[n]["fnames"]
            for ci, fi in enumerate(feat_idx):
                ax = f.add_subplot(3, nc, ri * nc + ci + 1); ax.set_xticks([]); ax.set_yticks([])
                cmap = mpl.cm.get_cmap("RdBu_r").copy(); cmap.set_bad("#dddddd")
                im = ax.imshow(zz[:, :, fi], cmap=cmap, vmin=-2.5, vmax=2.5, interpolation="nearest")
                if ri == 0: ax.set_title(nm[fi], fontsize=9)
                if ci == 0: ax.set_ylabel(n, fontsize=12, fontweight="bold", rotation=0, labelpad=20, va="center")
        f.colorbar(im, ax=f.axes, fraction=0.012, pad=0.01, label="z-score")
        f.suptitle(suptitle, fontsize=10)
        FigureCanvasAgg(f)
        pp = os.path.join(OUT, fname); f.savefig(pp, dpi=150, facecolor="white")
        shutil.copy(pp, os.path.join(REVIEW, fname)); print(f"wrote {fname}", flush=True)

    grid_fig([0, 1, 2, 3],
             "classical_perpixel_zmaps.png",
             "Per-pixel z-scored classical features (the inputs to KMeans), vmax=2 + σ=2 blur per frame. "
             "RdBu: red = above-mean (more crystalline/spotty/thick), blue = below. Grey = vacuum/zero-signal frames.")
    grid_fig([4, 5, 6, 7],
             "classical_perpixel_ringcv.png",
             "Per-pixel azimuthal spottiness (coefficient of variation) at each α ring, z-scored. "
             "High (red) = discrete spots on that ring; low (blue) = smooth ring / halo.")


if __name__ == "__main__":
    main()
