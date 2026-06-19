"""SI5, concretely: what diffraction feature makes two grains of one DINO class
alike, and what separates classes? From grain_acom_v2_SI5.npz (per-grain average
patterns). For the classes with enough grains we (a) show individual grain
patterns side by side, (b) overlay their rotation-invariant radial profiles
m(r), and (c) quantify within- vs between-class similarity of the radial profile,
the azimuthal spottiness, and the ACOM orientation.

  python tools/si5_grain_feature.py
"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

FIGS = "docs/paper/draft_v2/figs"; OUT = "docs/explainer/figs"
REVIEW = os.path.join(FIGS, "latest_review")
INV_ANG = 0.00185; KMAX = 0.35
z = np.load(os.path.join(FIGS, "grain_acom_v2_SI5.npz"))
gsum, gcnt, cls, vac = z["gsum"], z["gcnt"], z["cls"], z["vac"]
orient, H = z["orient"], int(z["H"])
cyx = (H - 1) / 2.0; beam = max(8, round(0.11 * H)); lo = beam + 1; hi = int(KMAX / INV_ANG)

prof = {}; spot = {}; chi = {}
for g in range(gsum.shape[0]):
    if vac[g]:
        continue
    avg = gsum[g] / max(gcnt[g], 1)
    m, v, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
    seg = m[lo:hi]; vseg = v[lo:hi]
    if seg.size < 5 or seg.sum() <= 0:
        continue
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
    pk = np.clip(seg - halo, 0, None)
    prof[g] = seg / (np.linalg.norm(seg) + 1e-9)           # rotation-invariant radial signature
    chi[g] = float((pk / np.clip(halo, 1e-9, None)).max())
    cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None)
    spot[g] = float(np.percentile(cv, 90))

gids = sorted(prof)
gcls = {g: int(cls[g]) for g in gids}
classes = sorted(set(gcls.values()), key=lambda c: np.median([chi[g] for g in gids if gcls[g] == c]))
big = [c for c in classes if sum(gcls[g] == c for g in gids) >= 4]   # enough grains

# within- vs between-class radial-profile correlation
def corr(a, b):
    return float(np.corrcoef(prof[a], prof[b])[0, 1])
win = [corr(a, b) for c in big for a, b in itertools.combinations([g for g in gids if gcls[g] == c], 2)]
btw = [corr(a, b) for a, b in itertools.combinations(gids, 2) if gcls[a] != gcls[b] and gcls[a] in big and gcls[b] in big]
print(f"radial-profile corr: WITHIN class {np.mean(win):.3f}+-{np.std(win):.3f}  BETWEEN {np.mean(btw):.3f}+-{np.std(btw):.3f}")
for c in big:
    gg = [g for g in gids if gcls[g] == c]
    sp = [spot[g] for g in gg]; ch = [chi[g] for g in gg]; orr = orient[gg]; orr = orr[np.isfinite(orr)]
    print(f" class {c}: n={len(gg)} chi={np.median(ch):.2f} spot={np.median(sp):.2f}"
          f" orient(deg)={'/'.join(f'{o:.0f}' for o in np.sort(orr)) if orr.size else 'none indexed'}")

# ---------------- figure ----------------
NCOL = 5
fig = Figure(figsize=(3 + NCOL * 1.7 + 4, 2.3 * len(big)), facecolor="white")
gs = fig.add_gridspec(len(big), NCOL + 2, width_ratios=[1] * NCOL + [1.3, 1.3])
cmap = cm.get_cmap("tab10")
cr = slice(int(cyx) - 150, int(cyx) + 150)
for ri, c in enumerate(big):
    gg = [g for g in gids if gcls[g] == c]
    gg = sorted(gg, key=lambda g: -gcnt[g])[:NCOL]
    for ci, g in enumerate(gg):
        ax = fig.add_subplot(gs[ri, ci])
        avg = gsum[g] / max(gcnt[g], 1)
        ax.imshow(np.log1p(np.clip(avg[cr, cr], 0, None)), cmap="inferno")
        ax.set_xticks([]); ax.set_yticks([])
        o = orient[g]
        ax.set_title(f"grain {g}\n{gcnt[g]:.0f}px"
                     + (f"  θ={o:.0f}°" if np.isfinite(o) else ""), fontsize=7)
        if ci == 0:
            ax.set_ylabel(f"class {c}", fontsize=11, fontweight="bold", rotation=0,
                          labelpad=26, va="center", color=cmap(ri))
    # overlaid radial profiles of this class
    ax = fig.add_subplot(gs[ri, NCOL])
    rr = np.arange(lo, hi)
    for g in gg:
        ax.plot(rr, prof[g], color=cmap(ri), lw=0.8, alpha=0.8)
    ax.set_title(f"class {c}: m(r) (rot-inv.)", fontsize=7)
    ax.set_xlabel("radius (px)", fontsize=7); ax.tick_params(labelsize=6); ax.set_yticks([])

# summary panels in the last column, top rows
axc = fig.add_subplot(gs[0, NCOL + 1])
axc.bar(["within\nclass", "between\nclass"], [np.mean(win), np.mean(btw)],
        yerr=[np.std(win), np.std(btw)], color=["#1F618D", "#C0392B"], capsize=4)
axc.set_ylim(0, 1); axc.set_ylabel("radial-profile\nPearson r", fontsize=8)
axc.set_title("m(r) similarity", fontsize=8); axc.tick_params(labelsize=7)

axs = fig.add_subplot(gs[1, NCOL + 1]) if len(big) > 1 else fig.add_subplot(gs[0, NCOL + 1])
axs.boxplot([[spot[g] for g in gids if gcls[g] == c] for c in big],
            showfliers=False, widths=0.6)
axs.set_xticklabels([f"c{c}" for c in big], fontsize=7)
axs.set_ylabel("azimuthal\nspottiness", fontsize=8); axs.set_title("spottiness by class", fontsize=8)
axs.tick_params(labelsize=7)

if len(big) > 2:
    axo = fig.add_subplot(gs[2, NCOL + 1])
    for ri, c in enumerate(big):
        oo = orient[[g for g in gids if gcls[g] == c]]; oo = oo[np.isfinite(oo)]
        if oo.size:
            axo.scatter(oo, [ri] * oo.size, color=cmap(ri), s=20)
    axo.set_yticks(range(len(big))); axo.set_yticklabels([f"c{c}" for c in big], fontsize=7)
    axo.set_xlim(0, 360); axo.set_xlabel("ACOM orientation (deg)", fontsize=7)
    axo.set_title("orientation: free within class", fontsize=8); axo.tick_params(labelsize=6)

fig.suptitle("SI5 grains: two grains of one DINO class share their rotation-invariant radial profile m(r) (ring positions + peak/halo sharpness) and azimuthal spottiness;\n"
             "classes differ in those. Orientation varies freely within a class. (Individual grain-average patterns, log scale, vmax=2.)", fontsize=10)
fig.tight_layout(rect=[0.02, 0, 1, 0.94]); FigureCanvasAgg(fig)
p = os.path.join(OUT, "si5_grain_feature.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(REVIEW, "si5_grain_feature.png"))
print("wrote si5_grain_feature.png", flush=True)
