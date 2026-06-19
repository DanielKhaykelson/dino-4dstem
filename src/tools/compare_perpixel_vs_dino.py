"""Per-PIXEL ACOM + crystallinity vs DINO grains — the head-to-head.
Loads imc_acom_fullpx_{name}.npz (per-pixel corr, n_peaks, ratio, zone-axis)
and the DINO grain results. Answers:
 (1) does per-pixel crystal-to-amorphous (peak/halo) MATCH the DINO grains?
     -> eta^2 of per-pixel ratio explained by DINO class; per-class medians.
 (2) does per-pixel ACOM orientation respect DINO grain boundaries?
     -> within-grain orientation consistency vs across.
 (3) what do DINO classes mean: crystalline / amorphous / orientation?
Figure rows = SI3/SI4/SI5; cols:
  DINO classes | per-px n_peaks | per-px peak/halo (crystal-amorph) |
  per-px ACOM corr | per-px zone-axis (indexed) | agreement scatter
Plus a vacuum mask from per-pixel halo (<0.02)."""
import os, sys, json, collections
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

def eta2(vals, labs, valid):
    v = vals[valid]; l = labs[valid]
    if v.size < 5: return float("nan")
    gm = v.mean(); sst = ((v - gm) ** 2).sum()
    ssb = sum(((v[l == c].mean() - gm) ** 2) * (l == c).sum() for c in set(l.tolist()))
    return float(ssb / (sst + 1e-12))

NAMES = list(RUNS)
fig = Figure(figsize=(19, 11), facecolor="white")
report = {}
for ri, name in enumerate(NAMES):
    z = np.load(os.path.join(OUT, f"imc_acom_fullpx_{name}.npz"))
    Ny, Nx = z["scan"]; N = Ny * Nx
    corr = z["corr"]; npk = z["n_peaks"].astype(float); ratio = z["ratio"]; halo = z["halo"]
    zau, zav, zaw = z["za_u"], z["za_v"], z["za_w"]
    asg = np.load(os.path.join(RUNS[name], "eval", "inference.npz"))["assigns"].astype(int)
    sample = halo >= HALO_VAC
    # (1) crystal-amorphous (ratio) explained by DINO class
    e2_ratio = eta2(ratio, asg, sample)
    e2_npk = eta2(npk, asg, sample)
    e2_corr = eta2(np.where(corr > 0, corr, 0.0), asg, sample)
    # per-class medians
    byc = {}
    for c in sorted(set(asg.tolist())):
        m = sample & (asg == c)
        if m.sum() >= 20:
            byc[c] = dict(n=int(m.sum()), ratio=round(float(np.median(ratio[m])), 2),
                          npk=round(float(np.median(npk[m])), 1),
                          corr=round(float(np.median(corr[m])), 2),
                          idx_frac=round(float((corr[m] > 0).mean()), 2))
    report[name] = dict(eta2_ratio=round(e2_ratio, 3), eta2_npeaks=round(e2_npk, 3),
                        eta2_corr=round(e2_corr, 3),
                        indexed_frac=round(float((corr[sample] > 0).mean()), 3),
                        per_class=byc)
    # ---- maps ----
    def M(v, valid=sample, default=np.nan):
        m = np.full(N, default); m[valid] = v[valid]; return m.reshape(Ny, Nx)
    cols = [("tab20", asg.reshape(Ny, Nx).astype(float), "DINO classes", False),
            ("inferno", M(npk), "per-px n_peaks", True),
            ("viridis", M(ratio), "per-px peak/halo\n(crystal-amorph)", True),
            ("magma", M(np.where(corr > 0, corr, np.nan)), "per-px ACOM corr", True)]
    for ci, (cm, mp, ttl, mask) in enumerate(cols):
        ax = fig.add_subplot(3, 6, ri * 6 + ci + 1)
        cmap = mpl.cm.get_cmap(cm).copy()
        if mask: cmap.set_bad("#222222")
        im = ax.imshow(mp, cmap=cmap, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0: ax.set_ylabel(f"IMC {name}", fontsize=11, fontweight="bold")
        if ri == 0: ax.set_title(ttl, fontsize=9)
        if mask: fig.colorbar(im, ax=ax, fraction=0.046)
    # zone-axis map (color by integer ZA family, only indexed)
    ax = fig.add_subplot(3, 6, ri * 6 + 5)
    za_id = np.full(N, np.nan)
    fams = {}
    for i in np.where(sample & (corr > 0))[0]:
        key = (abs(int(zau[i])), abs(int(zav[i])), abs(int(zaw[i])))
        za_id[i] = fams.setdefault(key, len(fams))
    im = ax.imshow(za_id.reshape(Ny, Nx), cmap="tab10", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if ri == 0: ax.set_title("per-px zone axis\n(indexed only)", fontsize=9)
    # agreement: per-pixel ratio distribution by DINO class
    ax = fig.add_subplot(3, 6, ri * 6 + 6)
    order = sorted(byc, key=lambda c: byc[c]["ratio"])
    data = [ratio[sample & (asg == c)] for c in order]
    ax.boxplot(data, showfliers=False, widths=0.6)
    ax.set_xticklabels([f"c{c}" for c in order], fontsize=6, rotation=90)
    ax.set_ylabel("per-px peak/halo", fontsize=8)
    if ri == 0: ax.set_title(f"ratio by DINO class\nη²={e2_ratio:.2f}", fontsize=9)
    else: ax.set_title(f"η²={e2_ratio:.2f}", fontsize=9)
    print(f"[{name}] eta2 ratio={e2_ratio:.3f} npk={e2_npk:.3f} corr={e2_corr:.3f} "
          f"indexed={report[name]['indexed_frac']:.2f}", flush=True)

fig.suptitle("Full per-PIXEL (stride=1) ACOM + crystal-to-amorphous (peak/halo) vs DINO grains — identical blob_log (thr=0.05, σ2-8, vmax=2, blur2, α k_max=0.35)\n"
             "η² = fraction of per-pixel crystallinity variance explained by DINO class (high → DINO grains ARE the crystallinity domains, with no CIF)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "compare_perpixel_vs_dino.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "compare_perpixel_vs_dino.png"))
json.dump(report, open(os.path.join(OUT, "compare_perpixel_vs_dino.json"), "w"), indent=2)
print("\n", json.dumps(report, indent=1)[:1500], flush=True)
print("wrote compare_perpixel_vs_dino.png + .json", flush=True)
