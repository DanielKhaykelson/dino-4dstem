"""Footprint-normalized IMC crystallinity, sample located by virtual BF/HAADF
(GUI convention): BF = central disk (r<=0.11H), HAADF = annulus 0.18-0.45 H.
HAADF = mass-thickness -> bright on sample, dark in vacuum -> Otsu = footprint.
Crystallinity per pixel: MATURE = radial peak/halo>0.5 (smooth ring, from
imc_glassorder), SPARSE = alpha-targeted azim-variance>2sigma (from
imc_alpha_targeted). Report mature/sparse/combined normalized to the BF/HAADF
sample footprint (not the FOV)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import open_lazy_cube
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
try:
    from skimage.filters import threshold_otsu
except Exception:
    def threshold_otsu(x, nbins=256):
        h, e = np.histogram(x, nbins); c = (e[:-1] + e[1:]) / 2; w1 = np.cumsum(h); w2 = np.cumsum(h[::-1])[::-1]
        m1 = np.cumsum(h * c) / np.clip(w1, 1, None); m2 = (np.cumsum((h * c)[::-1]) / np.clip(w2[::-1], 1, None))[::-1]
        v = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2; return c[:-1][np.argmax(v)]

OUT = "docs/paper/draft_v2/figs"; NAMES = ["SI3", "SI4", "SI5"]
IMC = {
 "SI3": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
 "SI4": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
 "SI5": r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
}
summary = {}
fig = Figure(figsize=(15, 10), facecolor="white")
for ri, name in enumerate(NAMES):
    t0 = time.time(); Ny = Nx = 128; N = Ny * Nx
    cube = open_lazy_cube(IMC[name], scan_shape=(Ny, Nx)); _, _, H, Wd = cube.shape
    cyx = (H - 1) / 2.0; yy, xx = np.indices((H, Wd)); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
    bf_mask = rr <= 0.11 * H                              # virtual BF (direct disk)
    haadf_mask = (rr >= 0.18 * H) & (rr <= 0.45 * H)      # virtual HAADF (annulus)
    BF = np.zeros(N); HA = np.zeros(N)
    for rx in range(Ny):
        blk = np.asarray(cube[rx], np.float32)
        BF[rx*Nx:(rx+1)*Nx] = blk[:, bf_mask].sum(1)
        HA[rx*Nx:(rx+1)*Nx] = blk[:, haadf_mask].sum(1)
    # footprint from HAADF (mass-thickness): bright = sample
    lh = np.log(np.clip(HA, 1, None)); thr = threshold_otsu(lh); foot = lh > thr
    # crystallinity (precomputed per-pixel)
    zg = np.load(os.path.join(OUT, f"imc_glassorder_{name}.npz")); ph = zg["ph"]
    za = np.load(os.path.join(OUT, f"imc_alpha_targeted_{name}.npz")); ex_a = za["ex_a"]; ex_c = za["ex_c"]
    mature = ph > 0.5
    sparse = ex_a > (np.nanmean(ex_c) + 2 * np.nanstd(ex_c))
    cryst = mature | sparse
    cov = float(foot.mean())
    def ff(m): return float((m & foot).sum() / max(foot.sum(), 1))
    summary[name] = dict(footprint_coverage=round(cov, 3), mature=round(ff(mature), 3),
                         sparse=round(ff(sparse), 3), combined=round(ff(cryst), 3),
                         glass=round(1 - ff(cryst), 3), n_sample_px=int(foot.sum()))
    print(f"[{name}] HAADF footprint={cov*100:.0f}% of FOV | of SAMPLE: mature={ff(mature)*100:.0f}% "
          f"sparse={ff(sparse)*100:.0f}% combined={ff(cryst)*100:.0f}%  ({time.time()-t0:.0f}s)", flush=True)
    # ---- plots: BF | HAADF+footprint | crystalline overlay | bars ----
    def lin(x): x = x.reshape(Ny, Nx); lo, hi = np.percentile(x, [1, 99]); return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    ax = fig.add_subplot(3, 4, ri*4 + 1); ax.imshow(lin(BF), cmap="gray"); ax.set_ylabel(name, fontsize=11, fontweight="bold")
    ax.set_title("virtual BF" if ri == 0 else "", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(3, 4, ri*4 + 2); ax.imshow(lin(HA), cmap="gray"); ax.contour(foot.reshape(Ny, Nx), [0.5], colors="cyan", linewidths=1.2)
    ax.set_title(f"virtual HAADF + footprint\n(sample={cov*100:.0f}% of FOV)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(3, 4, ri*4 + 3)
    base = lin(HA) * 0.5; rgb = np.dstack([base, base, base])
    rgb[mature.reshape(Ny, Nx) & foot.reshape(Ny, Nx)] = [1, 0.85, 0.1]
    rgb[(sparse & ~mature).reshape(Ny, Nx) & foot.reshape(Ny, Nx)] = [1, 0.2, 0.1]
    rgb[~foot.reshape(Ny, Nx)] = [0.05, 0.05, 0.15]
    ax.imshow(rgb); ax.set_title("mature ring (yellow)\nsparse spot (red); vacuum dark", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(3, 4, ri*4 + 4)
    vals = [summary[name]["mature"]*100, summary[name]["sparse"]*100, summary[name]["combined"]*100]
    ax.bar(["mature", "sparse", "combined"], vals, color=["#F1C40F", "#C0392B", "#7D3C98"])
    ax.set_ylim(0, max(40, max(vals)*1.25)); ax.set_ylabel("% of SAMPLE")
    for xi, vv in enumerate(vals): ax.text(xi, vv+0.6, f"{vv:.0f}%", ha="center", fontsize=10)
    ax.set_title("crystalline of sample\n(BF/HAADF footprint)", fontsize=9)

fig.suptitle("IMC crystallinity normalized to BF/HAADF sample footprint — mature (smooth ring) vs sparse (Bragg spots)\n"
             "SI3 mature spherulite -> SI4 mixed -> SI5 pure early sparse nucleation (NOT glass)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_bfhaadf_footprint.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "imc_bfhaadf_footprint.png"))
json.dump(summary, open(os.path.join(OUT, "imc_bfhaadf_footprint.json"), "w"), indent=2)
print("\n==== BF/HAADF footprint-normalized (% of SAMPLE) ====")
for n in NAMES:
    s = summary[n]; print(f"{n}: sample={s['footprint_coverage']*100:.0f}% FOV | mature={s['mature']*100:.0f}% sparse={s['sparse']*100:.0f}% combined={s['combined']*100:.0f}%")
print("wrote imc_bfhaadf_footprint.png + .json", flush=True)
