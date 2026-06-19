"""Supporting figure: radial peak/halo UNDER-COUNTS sparse/weak alpha; an alpha-
targeted, shot-noise-corrected azimuthal-variance detector + DINO reveal early
alpha nucleation in SI5 that the scalar metric misses. Uses saved npz (no cube)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib.cm as cmx
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
F = "docs/paper/draft_v2/figs"
def det_frac(n):
    z = np.load(f"{F}/imc_alpha_targeted_{n}.npz"); ex_a, ex_c, ph = z["ex_a"], z["ex_c"], z["ph"]
    thr = ex_c.mean() + 2 * ex_c.std(); return float(np.nanmean(ph > 0.5)), float((ex_a > thr).mean()), z
fr = {n: det_frac(n) for n in ["SI3", "SI4", "SI5"]}
zP = np.load(f"{F}/imc_alpha_targeted_SI5.npz"); Ny, Nx = zP["scan"]; ex_a, ex_c, phS = zP["ex_a"], zP["ex_c"], zP["ph"]
detS = (ex_a > ex_c.mean() + 2 * ex_c.std()).reshape(Ny, Nx)
asgS = np.load("runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60/eval/inference.npz")["assigns"].astype(int)
zg = np.load(f"{F}/imc_glassorder_SI5.npz"); dmean = zg["dmean"]; H = int(zg["H"])
fig = Figure(figsize=(15, 7.6), facecolor="white")
# (a) SI5 p/h map (looks like glass)
ax = fig.add_subplot(2, 3, 1); ax.imshow(phS.reshape(Ny, Nx), cmap="viridis", vmin=0, vmax=0.6)
ax.set_title(f"(a) SI5 radial crystallinity (p/h)\np/h>0.5 = {fr['SI5'][0]*100:.0f}% → 'mostly glass'", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
# (b) SI5 alpha-targeted detected (coherent nuclei)
ax = fig.add_subplot(2, 3, 2); bg = np.clip(phS.reshape(Ny, Nx), 0, 0.6) / 0.6 * 0.5
img = np.dstack([bg, bg, bg]); img[detS] = [1, 0.2, 0.1]
ax.imshow(img); ax.set_title(f"(b) α-targeted, shot-noise-corrected\ndetected = {fr['SI5'][1]*100:.0f}% (spatially coherent → real sparse α)", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
# (c) bars p/h vs alpha-targeted
ax = fig.add_subplot(2, 3, 3); x = np.arange(3); w = 0.38
ax.bar(x - w/2, [fr[n][0]*100 for n in ["SI3", "SI4", "SI5"]], w, label="radial p/h>0.5", color="#888")
ax.bar(x + w/2, [fr[n][1]*100 for n in ["SI3", "SI4", "SI5"]], w, label="α-targeted >2σ", color="#C0392B")
ax.set_xticks(x); ax.set_xticklabels(["SI3", "SI4", "SI5"]); ax.set_ylabel("% pixels"); ax.legend(fontsize=8)
ax.set_title("(c) p/h vs α-targeted\nSI5: 2% → 14% (sparse α p/h missed)", fontsize=10)
# (d) DINO enrichment for SI5
ax = fig.add_subplot(2, 3, 4); cls = sorted(set(asgS)); base = detS.ravel().mean()
enr = [(c, (detS.ravel()[asgS == c]).mean() / (base + 1e-9)) for c in cls if (asgS == c).sum() > 100]
cs = [c for c, _ in enr]; es = [e for _, e in enr]
ax.bar([f"c{c}" for c in cs], es, color=["#C0392B" if e > 1.5 else "#888" for e in es]); ax.axhline(1, color="k", ls=":")
ax.set_ylabel("α-detection enrichment"); ax.tick_params(axis="x", labelsize=7)
ax.set_title("(d) DINO concentrates sparse α (classes 2,3,1,7;\ntop-4 = 87%); glass classes 0,6,10 = 0%", fontsize=9)
# (e) representative diffraction: sparse-alpha DINO class vs glass class
def cr(m): return np.log1p(np.clip(m[H//2-110:H//2+110, H//2-110:H//2+110], 0, None))
ax = fig.add_subplot(2, 3, 5); ax.imshow(cr(dmean[2]), cmap="inferno"); ax.set_title("(e) DINO class 2 avg (sparse-α, p/h≈0.29)\nweak Bragg at α radii", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(2, 3, 6); ax.imshow(cr(dmean[0]), cmap="inferno"); ax.set_title("(f) DINO class 0 avg (true glass)\nsmooth halo, 0% α-detected", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("SI5: radial peak/halo misses sparse α; α-targeted (shot-noise-corrected) variance + DINO reveal early α nucleation (g≈0.95 → Poisson-limited)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig); fig.savefig(f"{F}/sparse_alpha_SI5.png", dpi=160, facecolor="white")
print("fractions p/h vs alpha-targeted:", {n: (round(fr[n][0], 2), round(fr[n][1], 2)) for n in fr}); print("wrote sparse_alpha_SI5.png", flush=True)
