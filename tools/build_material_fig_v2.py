"""Material insight v2 (corrected): spots = crystal, ring/halo classification.
Per sample (SI3/SI4/SI5):
  - footprint (ADF mass-thickness, Otsu) -> crystalline % OF SAMPLE
  - crystal = discrete Bragg spots = alpha-targeted shot-noise-corrected azim
    variance (ex_a > control+2sigma). NOT radial p/h (that scores ring/halo
    smoothness, mislabeled as 'crystallinity' before).
  - 3 exemplar grain-sums (connected components, constant orientation):
      A: top ex_a            -> expect discrete spots (coarse crystallite)
      B: high p/h, low ex_a  -> sharp SMOOTH ring: fine polycrystal OR halo?
      C: low p/h, low ex_a   -> broad halo (true glass)
    + their radial profiles vs alpha ring positions -> decides what B is.
Figure: rows=samples, cols=[map | A | B | C | radial profiles]; bottom strip =
footprint-normalized fractions + SI5 DINO-class enrichment."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from data import open_lazy_cube
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
try:
    from skimage.filters import threshold_otsu
except Exception:
    def threshold_otsu(x, nbins=256):
        h, e = np.histogram(x, nbins); c = (e[:-1] + e[1:]) / 2; w1 = np.cumsum(h); w2 = np.cumsum(h[::-1])[::-1]
        m1 = np.cumsum(h * c) / np.clip(w1, 1, None); m2 = (np.cumsum((h * c)[::-1]) / np.clip(w2[::-1], 1, None))[::-1]
        v = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2; return c[:-1][np.argmax(v)]

OUT = "docs/paper/draft_v2/figs"
IMC = {
 "SI3": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-003/Survey_CH2_1_nbed.cube.npy",
 "SI4": r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM/SI-004/Survey_CH2_0_1_nbed.cube.npy",
 "SI5": r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy",
}
NAMES = ["SI3", "SI4", "SI5"]
INV_ANG = 0.00185; ALPHA_D = [7.4, 6.0, 4.75, 3.9]

def grain_sum(cube, idxs, Nx, H):
    acc = np.zeros((H, H), np.float64); rows = {}
    for i in idxs: rows.setdefault(int(i) // Nx, []).append(int(i) % Nx)
    for rx in sorted(rows):
        blk = np.asarray(cube[rx], np.float32)
        for ry in rows[rx]: acc += blk[ry]
    return acc / max(len(idxs), 1)

summary = {}
fig = Figure(figsize=(17, 13.5), facecolor="white")
gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 0.85], hspace=0.32, wspace=0.18)
for ri, name in enumerate(NAMES):
    t0 = time.time()
    zg = np.load(os.path.join(OUT, f"imc_glassorder_{name}.npz"))
    ph = zg["ph"]; scat = zg["scat"]; asg = zg["assigns"].astype(int); Ny, Nx = zg["scan"]; H = int(zg["H"])
    za = np.load(os.path.join(OUT, f"imc_alpha_targeted_{name}.npz")); ex_a = za["ex_a"]; ex_c = za["ex_c"]
    ls = np.log(np.clip(scat, 1, None)); foot = ls > threshold_otsu(ls)
    spots = ex_a > (np.nanmean(ex_c) + 2 * np.nanstd(ex_c))
    f_spot = float((spots & foot).sum() / max(foot.sum(), 1))
    f_ring = float(((ph > 0.5) & ~spots & foot).sum() / max(foot.sum(), 1))
    # ---- exemplar grains (connected components, 50-400 px) ----
    grains = []
    asgmap = asg.reshape(Ny, Nx)
    for k in sorted(set(asg)):
        lab, n = ndimage.label(asgmap == k)
        for gi in range(1, n + 1):
            m = (lab == gi).ravel(); sz = int(m.sum())
            if 50 <= sz <= 400 and foot[m].mean() > 0.7:
                grains.append(dict(k=int(k), idx=np.where(m)[0], n=sz,
                                   exa=float(np.nanmean(ex_a[m])), ph=float(np.nanmean(ph[m]))))
    phmed = np.nanmedian([g["ph"] for g in grains])
    A = max(grains, key=lambda g: g["exa"])
    loexa = sorted(grains, key=lambda g: g["exa"])[:max(3, len(grains)//4)]
    B = max(loexa, key=lambda g: g["ph"])
    C = min(loexa, key=lambda g: g["ph"])
    cube = open_lazy_cube(IMC[name], scan_shape=(Ny, Nx)); cyx = (H - 1) / 2.0
    beam = max(8, round(0.11 * H)); lo = beam + 1; hi = int(0.85 * (H // 2))
    ra = [1.0 / (d * INV_ANG) for d in ALPHA_D]
    # ---- col 0: map ----
    ax = fig.add_subplot(gs[ri, 0])
    base = np.clip((ls - np.percentile(ls, 1)) / (np.percentile(ls, 99) - np.percentile(ls, 1) + 1e-9), 0, 1) * 0.55
    rgb = np.dstack([base, base, base]).reshape(Ny, Nx, 3)
    rgb[(spots & foot).reshape(Ny, Nx)] = [1, 0.25, 0.1]
    rgb[~foot.reshape(Ny, Nx)] = [0.04, 0.04, 0.13]
    ax.imshow(rgb); ax.set_ylabel(f"IMC {name}", fontsize=12, fontweight="bold")
    ax.set_title(f"Bragg-spot crystal (red) on ADF\n{f_spot*100:.0f}% of sample", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    # ---- cols 1-3: grain sums; col 4: radial profiles ----
    prof_ax = fig.add_subplot(gs[ri, 4])
    labels = [("A: spotty (top exA)", A, "#C0392B"), ("B: smooth sharp ring", B, "#B8860B"), ("C: low p/h (glass?)", C, "#2471A3")]
    rec = {}
    for ci, (lbl, g, col) in enumerate(labels):
        s = grain_sum(cube, g["idx"], Nx, H)
        m, _, _ = _radial_mean_var(s, (cyx, cyx), beam_px=beam); seg = m[lo:hi]
        halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
        pk = np.clip(seg - halo, 0, None); rpk = lo + int(np.argmax(pk))
        d_pk = 1.0 / (rpk * INV_ANG); gph = pk.sum() / (seg.sum() + 1e-9)
        # sharpness: FWHM of dominant peak in px
        pmax = pk.max(); above = np.where(pk > pmax / 2)[0]
        fwhm = int(above.max() - above.min() + 1) if len(above) else 0
        rec[lbl[0]] = dict(cls=g["k"], n=g["n"], exa=round(g["exa"], 1), ph_grain=round(gph, 2),
                           d_peak=round(d_pk, 2), fwhm_px=fwhm)
        ax = fig.add_subplot(gs[ri, 1 + ci])
        cr = slice(int(cyx) - 150, int(cyx) + 150)
        ax.imshow(np.log1p(np.clip(s[cr, cr], 0, None)), cmap="inferno")
        for r0 in ra: ax.add_patch(Circle((150, 150), r0, fill=False, color="cyan", lw=0.5, ls=":"))
        ax.set_title(f"{lbl}  c{g['k']} n={g['n']}\nexA={g['exa']:.0f} d@peak={d_pk:.2f}Å fwhm={fwhm}px", fontsize=8, color=col)
        ax.set_xticks([]); ax.set_yticks([])
        prof_ax.plot(np.arange(lo, hi), seg / seg.max(), color=col, lw=1.4, label=lbl[0])
    for r0, d in zip(ra, ALPHA_D):
        if lo < r0 < hi: prof_ax.axvline(r0, color="r", ls=":", lw=0.8)
    prof_ax.set_title("radial profiles (red dots = α rings)\nsharp@α = crystal · broad = glass halo", fontsize=8)
    prof_ax.legend(fontsize=7); prof_ax.set_yticks([])
    if ri == 2: prof_ax.set_xlabel("radius (px)")
    summary[name] = dict(spot_frac=round(f_spot, 3), smoothring_frac=round(f_ring, 3),
                         footprint=round(float(foot.mean()), 3), exemplars=rec)
    print(f"[{name}] spot={f_spot*100:.0f}% smooth-ring={f_ring*100:.0f}% of sample | "
          + " | ".join(f"{k}: d={v['d_peak']}Å fwhm={v['fwhm_px']}px exA={v['exa']}" for k, v in rec.items())
          + f" ({time.time()-t0:.0f}s)", flush=True)

# ---- bottom strip: fractions + SI5 enrichment ----
ax = fig.add_subplot(gs[3, 0:2])
x = np.arange(3); w = 0.38
ax.bar(x - w/2, [summary[n]["spot_frac"]*100 for n in NAMES], w, color="#C0392B", label="Bragg-spot crystal")
ax.bar(x + w/2, [summary[n]["smoothring_frac"]*100 for n in NAMES], w, color="#B8860B", label="smooth sharp ring (B-type)")
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("% of SAMPLE (footprint-norm.)")
ax.legend(fontsize=8); ax.set_title("corrected, footprint-normalized composition", fontsize=10)
ax = fig.add_subplot(gs[3, 2:4])
zal = np.load(os.path.join(OUT, "imc_alpha_targeted_SI5.npz"))
sp5 = zal["ex_a"] > (np.nanmean(zal["ex_c"]) + 2*np.nanstd(zal["ex_c"]))
asg5 = np.load(os.path.join(OUT, "imc_glassorder_SI5.npz"))["assigns"].astype(int)
basef = sp5.mean(); cls = [c for c in sorted(set(asg5)) if (asg5 == c).sum() > 100]
enr = [(sp5[asg5 == c]).mean() / (basef + 1e-9) for c in cls]
ax.bar([f"c{c}" for c in cls], enr, color=["#C0392B" if e > 1.5 else "#999" for e in enr])
ax.axhline(1, color="k", ls=":"); ax.set_ylabel("spot-crystal enrichment"); ax.tick_params(labelsize=7)
ax.set_title("SI5: DINO classes concentrate the spot-crystal\n(no crystallographic input)", fontsize=10)
ax = fig.add_subplot(gs[3, 4]); ax.axis("off")
ax.text(0, 0.95, "Reading:\nspots = coarse crystallites\nsharp smooth ring = fine\n  polycrystal (if @α)\nbroad bump = glass halo\n\np/h alone mislabels:\nit scores ring smoothness,\nnot crystallinity.", fontsize=9, va="top")
fig.suptitle("IMC material insight v2 — crystal = discrete Bragg spots (shot-noise-corrected azimuthal variance), footprint-normalized;\n"
             "exemplar grain-sums + radial profiles classify spotty crystal vs sharp-ring vs glass halo", fontsize=12)
FigureCanvasAgg(fig)
p = os.path.join(OUT, "material_insight_IMC_v2.png"); fig.savefig(p, dpi=150, facecolor="white")
import shutil; shutil.copy(p, os.path.join(OUT, "latest_review", "material_insight_IMC_v2.png"))
json.dump(summary, open(os.path.join(OUT, "material_insight_IMC_v2.json"), "w"), indent=2)
print("\nSUMMARY:", json.dumps(summary, indent=1), flush=True)
print("wrote material_insight_IMC_v2.png + .json", flush=True)
