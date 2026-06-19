"""Footprint-normalized crystallinity for IMC (fair cross-FOV comparison).

% crystalline over the whole scan is unfair: SI3/SI4/SI5 are different fields of
view with different amounts of vacuum / thin background. So:
  1. sample FOOTPRINT = mass-thickness (post-beam scattering 'scat') above a
     vacuum threshold (Otsu on log-scat) -> the actual material.
  2. crystalline pixel = alpha-targeted, shot-noise-corrected detection
     (ex_a > control mean + 2*sigma) from imc_alpha_targeted_{name}.npz.
  3. report crystalline fraction (a) over whole scan vs (b) NORMALIZED to the
     footprint, plus footprint coverage of the FOV.
Inputs: imc_glassorder_{name}.npz (scat, assigns) + imc_alpha_targeted_{name}.npz."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
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
summary = {}
fig = Figure(figsize=(14, 10), facecolor="white")
for ri, name in enumerate(NAMES):
    zg = np.load(os.path.join(OUT, f"imc_glassorder_{name}.npz"))
    za = np.load(os.path.join(OUT, f"imc_alpha_targeted_{name}.npz"))
    scat = zg["scat"]; asg = zg["assigns"].astype(int); ph = zg["ph"]; Ny, Nx = zg["scan"]
    ex_a = za["ex_a"]; ex_c = za["ex_c"]
    N = Ny * Nx
    # 1. footprint from mass-thickness (log-scat Otsu)
    ls = np.log(np.clip(scat, 1, None)); thr = threshold_otsu(ls)
    foot = ls > thr                              # sample (non-vacuum)
    # 2a. SPARSE crystal: alpha-targeted azim-variance excess (few Bragg spots), 2 sigma > control
    cthr = np.nanmean(ex_c) + 2 * np.nanstd(ex_c)
    sparse = ex_a > cthr
    # 2b. MATURE crystal: strong smooth radial ring (peak/halo high) -> LOW azim variance
    mature = ph > 0.5
    cryst = sparse | mature                      # any crystalline (combined, complementary)
    # 3. footprint-normalized fractions
    cov = float(foot.mean())
    def ff(m): return float((m & foot).sum() / max(foot.sum(), 1))
    summary[name] = dict(footprint_coverage=round(cov, 3),
                         cryst_whole=round(float(cryst.mean()), 3),
                         mature_footprint=round(ff(mature), 3),
                         sparse_footprint=round(ff(sparse), 3),
                         cryst_footprint=round(ff(cryst), 3),
                         glass_footprint=round(1 - ff(cryst), 3), n_sample_px=int(foot.sum()))
    f_whole = float(cryst.mean()); f_foot = ff(cryst)
    print(f"[{name}] foot cover={cov*100:.0f}% | MATURE(p/h>.5)={ff(mature)*100:.0f}% "
          f"SPARSE(α-var)={ff(sparse)*100:.0f}% COMBINED={ff(cryst)*100:.0f}%  (all footprint-normalized)", flush=True)
    # ---- plots ----
    ax = fig.add_subplot(3, 3, ri*3 + 1)
    im = ax.imshow(ls.reshape(Ny, Nx), cmap="magma"); ax.contour(foot.reshape(Ny, Nx), [0.5], colors="cyan", linewidths=1.2)
    ax.set_title(f"{name}: log mass-thickness + footprint\ncoverage={cov*100:.0f}% of FOV", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel(name, fontsize=11, fontweight="bold")
    ax = fig.add_subplot(3, 3, ri*3 + 2)
    rgb = np.zeros((N, 3)); base = (ls - ls.min()) / (ls.max() - ls.min() + 1e-9) * 0.5
    rgb[:, 0] = base; rgb[:, 1] = base; rgb[:, 2] = base
    rgb[(mature) & foot] = [1, 0.85, 0.1]; rgb[(sparse & ~mature) & foot] = [1, 0.2, 0.1]; rgb[(~foot)] = [0.05, 0.05, 0.15]
    ax.imshow(rgb.reshape(Ny, Nx, 3)); ax.set_title(f"{name}: mature ring (yellow) / sparse spot (red)\non sample; vacuum=dark", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(3, 3, ri*3 + 3)
    vals = [summary[name]["mature_footprint"]*100, summary[name]["sparse_footprint"]*100, summary[name]["cryst_footprint"]*100]
    ax.bar(["mature\n(p/h>.5)", "sparse\n(α-var)", "combined"], vals, color=["#F1C40F", "#C0392B", "#7D3C98"])
    ax.set_ylabel("% of SAMPLE crystalline"); ax.set_ylim(0, max(40, max(vals)*1.25))
    for xi, vv in enumerate(vals): ax.text(xi, vv+0.6, f"{vv:.0f}%", ha="center", fontsize=10)
    ax.set_title(f"{name}: footprint-normalized\nmature vs sparse crystallinity", fontsize=9)

fig.suptitle("IMC crystallinity normalized to the SAMPLE FOOTPRINT (mass-thickness), not the field of view\n"
             "fair cross-FOV comparison: SI3/SI4/SI5 have different vacuum/background coverage", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
fig.savefig(os.path.join(OUT, "imc_footprint_normalized_cryst.png"), dpi=150, facecolor="white")
import shutil; shutil.copy(os.path.join(OUT, "imc_footprint_normalized_cryst.png"), os.path.join(OUT, "latest_review", "imc_footprint_normalized_cryst.png"))
json.dump(summary, open(os.path.join(OUT, "imc_footprint_normalized_cryst.json"), "w"), indent=2)
print("\n==== footprint-normalized (% of SAMPLE) ====")
for n in NAMES:
    s = summary[n]; print(f"{n}: sample={s['footprint_coverage']*100:.0f}% of FOV | mature={s['mature_footprint']*100:.0f}% sparse={s['sparse_footprint']*100:.0f}% combined={s['cryst_footprint']*100:.0f}%")
print("wrote imc_footprint_normalized_cryst.png + .json", flush=True)
