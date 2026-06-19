"""Re-render the per-grain crystallinity/ACOM figure with a PHYSICS-based
vacuum mask: vacuum = grain halo_max < 0.02 (no diffuse scattering = no
material). ADF-Otsu footprint is NOT used — it mislabels thin needles/edges
(SI4 needles foot=0.38, SI5 edge classes foot=0 but halo=0.07 = real thin
sample). Ratio redefined robustly: amp / max(halo_max, 0.01)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl

OUT = "docs/paper/draft_v2/figs"
RUNS = {
 "SI3": "runs/_gui/IMC_SI3_m097k60",
 "SI4": "runs/_gui/IMC_SI4_m097_k60",
 "SI5": "runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60",
}
HALO_VAC = 0.02
fig = Figure(figsize=(17, 11), facecolor="white")
stats = {}
gj = json.load(open(os.path.join(OUT, "imc_grain_acom_crystallinity.json")))
for ri, name in enumerate(RUNS):
    z = np.load(os.path.join(OUT, f"imc_grain_acom_{name}.npz"))
    gid = z["gid"]; npk = z["n_peaks"].astype(float)
    corr = z["corr"].astype(float); cls = z["cls"]
    Ny = Nx = 128; G = len(npk)
    tab = gj[name]["grains"]
    halo = np.array([t["halo"] for t in tab]); amp = np.array([t["amp"] for t in tab])
    ratio = amp / np.maximum(halo, 0.01)          # robust peak/halo
    ok = halo >= HALO_VAC
    def paint(vals, valid):
        m = np.full(Ny * Nx, np.nan)
        for g in range(G):
            if valid[g]: m[gid == g] = vals[g]
        return m.reshape(Ny, Nx)
    asg = np.load(os.path.join(RUNS[name], "eval", "inference.npz"))["assigns"].astype(int)
    cmaps = [("inferno", paint(npk, ok), "grain n_peaks"),
             ("viridis", paint(ratio, ok), "peak/halo ratio"),
             ("magma", paint(np.where(corr > 0, corr, np.nan), ok), "ACOM corr (alpha)")]
    ax = fig.add_subplot(3, 5, ri * 5 + 1); ax.imshow(asg.reshape(Ny, Nx), cmap="tab20", interpolation="nearest")
    ax.set_ylabel(f"IMC {name}", fontsize=11, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    if ri == 0: ax.set_title("DINO classes", fontsize=10)
    for ci, (cm, m, ttl) in enumerate(cmaps):
        ax = fig.add_subplot(3, 5, ri * 5 + 2 + ci)
        cmap = mpl.cm.get_cmap(cm).copy(); cmap.set_bad("#222222")
        im = ax.imshow(m, cmap=cmap, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        if ri == 0: ax.set_title(ttl + "\n(dark = off-sample/unindexed)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(3, 5, ri * 5 + 5)
    sel = ok; idx = sel & (corr > 0); un = sel & (corr <= 0)
    ax.scatter(npk[un], ratio[un], c="#999", s=24, label="unindexed")
    sc = ax.scatter(npk[idx], ratio[idx], c=corr[idx], cmap="magma", s=28, edgecolor="k", lw=0.3)
    ax.set_xlabel("n_peaks"); ax.set_ylabel("peak/halo ratio")
    if ri == 0: ax.set_title("on-sample grains\n(color = ACOM corr)", fontsize=9)
    ax.legend(fontsize=6); fig.colorbar(sc, ax=ax, fraction=0.046)
    stats[name] = dict(grains_total=G, on_sample=int(ok.sum()),
                       indexed_on_sample=int((ok & (corr > 0)).sum()),
                       glass_on_sample=int((ok & (npk <= 2)).sum()))
fig.suptitle("IMC per-GRAIN crystallinity + ACOM (alpha, k_max=0.35) — identical blob_log peaks (thr=0.05, sigma 2-8, log-stretch, vmax=2, blur 2)\n"
             "vacuum mask = halo<0.02 (no diffuse scattering); ratio = amp/max(halo,0.01); full-sample averages give 0 peaks (rings != blobs) -> grain level is mandatory", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "imc_grain_acom_crystallinity.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "imc_grain_acom_crystallinity.png"))
print(json.dumps(stats, indent=1)); print("re-rendered imc_grain_acom_crystallinity.png", flush=True)
