"""Material analysis + figure for IMC crystallization from the per-pixel metrics
(tools/imc_glass_order.py outputs). For a sample: classify each DINO class by its
mean-pattern d-spacings into alpha / gamma / amorphous; map regimes; test the
oriented-glass precursor (low crystallinity + high halo anisotropy) ahead of the
crystal front; test outward growth (anisotropy/crystallinity vs distance-to-crystal).
Usage: python tools/imc_glass_order_fig.py SI3   (or SI4/SI5)
DINO is the segmentation lens; physics (d-spacing, anisotropy) layered on top."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.ndimage import distance_transform_edt
from gui_app.crystallinity_panel import _snip_baseline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
F = "docs/paper/draft_v2/figs"; INV_ANG = 0.00185
ALPHA_D = [10.4, 7.4, 6.0, 4.75, 3.9]; GAMMA_D = [8.6, 7.5, 5.2, 4.5, 4.0]
name = sys.argv[1] if len(sys.argv) > 1 else "SI3"
z = np.load(f"{F}/imc_glassorder_{name}.npz")
ph, aniso, d_ang, scat, asg, dmean = z["ph"], z["aniso"], z["d_ang"], z["scat"], z["assigns"], z["dmean"]
Ny, Nx = z["scan"]; H = int(z["H"]); N = Ny * Nx
cy = (H - 1) / 2; yy, xx = np.indices((H, H)); rr = np.sqrt((yy - cy)**2 + (xx - cy)**2).astype(int)
beam = max(8, round(0.11 * H)); nb = H // 2; lo = max(int(0.10 * nb), beam + 1); hi = int(0.85 * nb)
def radial(m): tb = np.bincount(rr.ravel(), m.ravel()); n = np.bincount(rr.ravel()); return tb / np.clip(n, 1, None)
def ring_ds(m, ntop=5):
    prof = radial(m)[lo:hi]; base = np.exp(_snip_baseline(np.log(np.clip(prof, 1e-6, None)))); pk = np.clip(prof - base, 0, None)
    # local maxima
    idx = [i for i in range(1, len(pk)-1) if pk[i] > pk[i-1] and pk[i] >= pk[i+1] and pk[i] > 0.05*pk.max()]
    idx = sorted(idx, key=lambda i: -pk[i])[:ntop]
    return sorted([round(1.0/(((lo+i)*INV_ANG)+1e-9), 1) for i in idx], reverse=True)
def classify(ds):
    if not ds: return "amorph"
    a = sum(min(abs(np.array(ALPHA_D)-d)) < 0.6 for d in ds); g = sum(min(abs(np.array(GAMMA_D)-d)) < 0.6 for d in ds)
    return "alpha" if a > g else ("gamma" if g > a else "mixed")
# per-class summary
K = dmean.shape[0]; cls = {}
for c in range(K):
    msk = asg == c
    if msk.sum() < 20: continue
    ds = ring_ds(dmean[c]); cls[c] = dict(n=int(msk.sum()), ph=round(float(np.nanmedian(ph[msk])), 3),
        aniso=round(float(np.nanmedian(aniso[msk])), 2), d=round(float(np.nanmedian(d_ang[msk])), 1), rings=ds, poly=classify(ds))
print(f"=== {name}: per-DINO-class diffraction fingerprint ===", flush=True)
for c, v in sorted(cls.items(), key=lambda kv: -kv[1]["ph"]):
    print(f"  c{c}: n={v['n']} p/h={v['ph']} aniso={v['aniso']} halo_d={v['d']}A rings={v['rings']} -> {v['poly']}", flush=True)
# regime thresholds (data-driven)
phv = ph[np.isfinite(ph)]; tc = float(np.nanpercentile(phv, 70))   # crystalline cut (tunable)
cryst = (ph > tc); glass = np.isfinite(ph) & ~cryst
ta = float(np.nanpercentile(aniso[glass & np.isfinite(aniso)], 75))  # oriented-glass cut among glass
phm = ph.reshape(Ny, Nx); anm = aniso.reshape(Ny, Nx); cm = cryst.reshape(Ny, Nx)
oriented_glass = glass.reshape(Ny, Nx) & (anm > ta); iso_glass = glass.reshape(Ny, Nx) & ~(anm > ta)
print(f"\nregime fractions: crystal={cryst.mean():.2f} oriented-glass={oriented_glass.mean():.2f} iso-glass={iso_glass.mean():.2f} (tc={tc:.2f} ta={ta:.2f})", flush=True)
# precursor test: anisotropy vs distance-to-crystal (glass only)
dist = distance_transform_edt(~cm)
gl = glass.reshape(Ny, Nx) & (dist > 0) & (dist < 25)
# STRICT amorphous control: exclude any partially-crystalline pixels (p/h < 0.40)
T_AMORPH = 0.40
strict = (phm < T_AMORPH) & np.isfinite(phm) & (dist > 0) & (dist < 25)
bins = np.arange(1, 22, 2); prof_an = []; prof_ph = []; prof_an_s = []; prof_d_s = []; prof_n_s = []
for b in bins:
    sel = gl & (dist >= b) & (dist < b + 2)
    prof_an.append(np.nanmedian(anm[sel]) if sel.any() else np.nan)
    prof_ph.append(np.nanmedian(phm[sel]) if sel.any() else np.nan)
    ss = strict & (dist >= b) & (dist < b + 2)
    prof_an_s.append(np.nanmedian(anm[ss]) if ss.any() else np.nan)
    prof_d_s.append(np.nanmedian(d_ang.reshape(Ny, Nx)[ss]) if ss.any() else np.nan)
    prof_n_s.append(int(ss.sum()))
print("precursor (ALL glass)   dist->aniso:", [f"{int(b)}:{a:.2f}" for b, a in zip(bins, prof_an)], flush=True)
print("precursor (STRICT amorph p/h<0.4) dist->aniso:", [f"{int(b)}:{a:.2f}(n{n})" for b, a, n in zip(bins, prof_an_s, prof_n_s)], flush=True)
print("strict amorph halo d-spacing vs dist:", [f"{int(b)}:{d:.1f}" for b, d in zip(bins, prof_d_s)], flush=True)
far = strict & (dist > 12)
print(f"STRICT-amorph anisotropy: near-crystal(1-3px) median={np.nanmedian(anm[strict&(dist<3)]):.2f} "
      f"vs far(>12px) median={np.nanmedian(anm[far]):.2f}", flush=True)
# ---- figure ----
fig = Figure(figsize=(15, 8.5), facecolor="white")
def im(ax, M, t, cmap, **kw): ax.imshow(M, cmap=cmap, interpolation="nearest", **kw); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
im(fig.add_subplot(2, 4, 1), asg.reshape(Ny, Nx), f"{name} DINO classes (lens)", "tab20")
im(fig.add_subplot(2, 4, 2), phm, "crystallinity (peak/halo)", "viridis", vmin=0, vmax=np.nanpercentile(ph, 98))
im(fig.add_subplot(2, 4, 3), anm, "halo azimuthal anisotropy", "magma", vmin=0, vmax=np.nanpercentile(aniso, 98))
# regime map
reg = np.zeros((Ny, Nx, 3)); reg[iso_glass] = [0.2, 0.2, 0.35]; reg[oriented_glass] = [0.95, 0.75, 0.1]; reg[cm] = [0.8, 0.1, 0.1]
ax = fig.add_subplot(2, 4, 4); ax.imshow(reg, interpolation="nearest"); ax.set_title("regimes: crystal(red)\noriented-glass(gold) iso-glass(blue)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
# joint ph-aniso
ax = fig.add_subplot(2, 4, 5); good = np.isfinite(ph) & np.isfinite(aniso)
ax.hexbin(ph[good], aniso[good], gridsize=40, cmap="Greys", bins="log")
ax.axvline(tc, color="r", ls="--", lw=1); ax.axhline(ta, color="orange", ls="--", lw=1)
ax.set_xlabel("crystallinity p/h"); ax.set_ylabel("halo anisotropy"); ax.set_title("joint distribution", fontsize=10)
# precursor profile
ax = fig.add_subplot(2, 4, 6); ax.plot(bins, prof_an, "o-", color="#C0392B", label="all glass")
ax.plot(bins, prof_an_s, "^-", color="#E67E22", label="strict amorph (p/h<0.4)")
ax2 = ax.twinx(); ax2.plot(bins, prof_ph, "s--", color="#1C7293", alpha=0.6, label="crystallinity")
ax.set_xlabel("distance into glass from crystal (px)"); ax.set_ylabel("halo anisotropy"); ax2.set_ylabel("p/h", color="#1C7293")
ax.set_title("precursor test (strict control)", fontsize=10); ax.invert_xaxis(); ax.legend(fontsize=7, loc="upper left")
# per-class d-spacing bars vs alpha/gamma refs
ax = fig.add_subplot(2, 4, 7)
cc = sorted(cls.items(), key=lambda kv: -kv[1]["ph"])[:6]
for i, (c, v) in enumerate(cc):
    for d in v["rings"]: ax.scatter(d, i, s=30, color={"alpha":"#1f77b4","gamma":"#2ca02c","amorph":"#888","mixed":"#9467bd"}[v["poly"]])
for d in ALPHA_D: ax.axvline(d, color="#1f77b4", ls=":", alpha=0.5)
for d in GAMMA_D: ax.axvline(d, color="#2ca02c", ls=":", alpha=0.5)
ax.set_yticks(range(len(cc))); ax.set_yticklabels([f"c{c}({v['poly'][:1]})" for c, v in cc], fontsize=8)
ax.set_xlabel("d-spacing (A)  [blue=α refs, green=γ refs]"); ax.set_title("class ring d-spacings vs α/γ", fontsize=10); ax.invert_xaxis()
# crystalline vs glass mean radial
ax = fig.add_subplot(2, 4, 8)
cmean = dmean[[c for c in cls if cls[c]["ph"] > tc]].mean(0) if any(cls[c]["ph"] > tc for c in cls) else dmean[0]
gmean = dmean[[c for c in cls if cls[c]["ph"] <= tc]].mean(0) if any(cls[c]["ph"] <= tc for c in cls) else dmean[0]
ax.plot(np.log1p(radial(cmean)[lo:hi]), color="#C0392B", label="crystalline avg")
ax.plot(np.log1p(radial(gmean)[lo:hi]), color="#1C7293", label="glass avg")
ax.set_xlabel("radial bin (beam-trimmed)"); ax.set_title("crystalline vs glass radial", fontsize=10); ax.legend(fontsize=8)
fig.suptitle(f"IMC {name} — DINO-guided crystallization physics (crystal=blob+needles, glass=matrix)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96]); FigureCanvasAgg(fig); fig.savefig(f"{F}/imc_material_{name}.png", dpi=150, facecolor="white")
print(f"\nwrote imc_material_{name}.png", flush=True)
