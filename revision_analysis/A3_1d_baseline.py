"""
A3  -  1D-radial-profile clustering baseline (v2: grain-level maps + automatic K).

Referee: "why not just cluster the azimuthally-averaged 1D radial profiles?"
Quantifies what the 2D rotation-invariant embedding adds over 1D.

Per IMC field we azimuthally integrate every diffraction pattern to a 1D radial profile and
k-means cluster it two ways:
  matched K = DINO active-class count   (like-for-like)
  auto    K = chosen by silhouette      (let 1D pick its own resolution)
then compare the per-grain descriptor eta^2 of each 1D clustering to DINO.

Maps are shown CONSOLIDATED TO GRAINS for all methods (each grain painted by its majority
label), so any fragmentation is real grain-level structure, not pixel speckle. The point to
look for: 1D clustering splits needles of equal crystallinity into different classes (it keys
on intensity/thickness), whereas DINO groups them by order.

Expectation: 1D recovers the RADIAL part of the axis (chi) but not azimuthal spottiness.

Output: figs/Review/A3_1d_baseline_{field}.png + printed eta^2 table + chosen auto-K.
Machine rules: sklearn KMeans (single process); entry guarded; profiles cached to scratch.
"""
import os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CUBES = {
    "SI3": r"D:\DINOSR\data\231228-IMC150nm-0p2apersec-anneal-70c-60min\EF-4DSTEM\SI-003\Survey_CH2_1_nbed.cube.npy",
    "SI4": r"D:\DINOSR\data\231228-IMC150nm-0p2apersec-anneal-70c-60min\EF-4DSTEM\SI-004\Survey_CH2_0_1_nbed.cube.npy",
    "SI5": r"D:\DINOSR\data\231228-IMC150nm-0p2apersec-anneal-70c-60min\EF-4DSTEM\SI-005\Survey_CH2_1_nbed.cube.npy",
}
RUN = {"SI3": "runs/_gui/IMC_SI3_m097k60", "SI4": "runs/_gui/IMC_SI4_m097_k60",
       "SI5": "runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60"}
FOV = {"SI3": 187, "SI4": 160, "SI5": 160}
DISP = {"SI3": "interface", "SI4": "needles", "SI5": "interface (mag.)"}
INV, KMAX = 0.00185, 0.35
NY = NX = 128
KEYS = ["spot", "B", "chi"]
LBL = {"spot": "azimuthal\nspottiness", "B": "Bragg\nexcess B", "chi": "radial\npeak/halo chi"}
CACHE = "D:/temp/claude"


def main(fields):
    os.chdir("D:/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive")
    sys.path.insert(0, "src")
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib as mpl

    FIGS = "docs/paper/draft_v2/figs"; OUT = f"{FIGS}/Review"

    def eta2(v, lab):
        v = np.asarray(v, float); lab = np.asarray(lab)
        ok = np.isfinite(v) & (np.asarray(lab) >= 0); v, lab = v[ok], np.asarray(lab)[ok]
        if v.size < 5 or len(set(lab.tolist())) < 2:
            return np.nan
        gm = v.mean(); sst = ((v - gm) ** 2).sum()
        ssb = sum(((v[lab == c].mean() - gm) ** 2) * (lab == c).sum() for c in set(lab.tolist()))
        return float(ssb / (sst + 1e-12))

    def per_grain_desc(name):
        z = np.load(f"{FIGS}/grain_acom_v2_{name}.npz")
        cls, vac, gsum, gcnt, gid = z["cls"], z["vac"], z["gsum"], z["gcnt"], z["gid"]
        H = int(z["H"]); c = (H - 1) / 2.0
        beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV[name])
        yy, xx = np.indices((H, H)); rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2); band = (rr >= lo) & (rr <= hi)
        G = gsum.shape[0]; spot = np.full(G, np.nan); B = np.full(G, np.nan); chi = np.full(G, np.nan)
        for g in range(G):
            if vac[g]:
                continue
            avg = gsum[g] / max(gcnt[g], 1)
            m, v, _ = _radial_mean_var(avg, (c, c), beam_px=beam); seg, vseg = m[lo:hi], v[lo:hi]
            if seg.size < 5 or seg.sum() <= 0:
                continue
            halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
            chi[g] = (np.clip(seg - halo, 0, None) / np.clip(halo, 1e-9, None)).max()
            spot[g] = np.percentile(np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None), 90)
            hf = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1])
            B[g] = np.clip(avg[band] - hf[band], 0, None).sum() / (hf[band].sum() + 1e-9)
        return gid, cls, dict(spot=spot, B=B, chi=chi), lo, hi

    def radial_profiles(name, lo, hi):
        cf = f"{CACHE}/A3_prof_{name}_{lo}_{hi}.npy"
        if os.path.exists(cf):
            print(f"      (cached profiles {name})"); return np.load(cf)
        cube = np.load(CUBES[name], mmap_mode="r")
        assert cube.shape[0] * cube.shape[1] == NY * NX, cube.shape
        H = cube.shape[-1]; cube = cube.reshape(-1, H, H); c = (H - 1) / 2.0
        yy, xx = np.indices((H, H)); rb = np.hypot(yy - c, xx - c).astype(int).ravel()
        nb = rb.max() + 1; cnt = np.bincount(rb, minlength=nb).astype(np.float64)
        N = cube.shape[0]; prof = np.zeros((N, hi - lo), np.float64); t0 = time.time()
        for i in range(N):
            m = np.bincount(rb, np.asarray(cube[i], np.float64).ravel(), minlength=nb) / cnt
            prof[i] = m[lo:hi]
            if i % 4000 == 0:
                print(f"      radial {i}/{N}  {time.time()-t0:.0f}s", flush=True)
        np.save(cf, prof); return prof

    def grain_labels(pixlab, gid, G):
        out = np.full(G, -1)
        for g in range(G):
            px = pixlab[gid == g]
            if px.size:
                u, ct = np.unique(px, return_counts=True); out[g] = u[np.argmax(ct)]
        return out

    def grain_map(glab, gid):
        out = np.full(NY * NX, -1)
        v = gid >= 0
        out[v] = glab[gid[v]]
        return out.reshape(NY, NX)

    def draw(ax, mp, title):
        u = sorted(set(int(x) for x in np.unique(mp) if x >= 0))
        rmap = {c: i for i, c in enumerate(u)}
        pal = mpl.colormaps.get_cmap("tab20")(np.linspace(0, 1, 20))
        rgb = np.ones((*mp.shape, 3)) * 0.91           # background gray
        for c, i in rmap.items():
            rgb[mp == c] = pal[i % 20][:3]
        ax.imshow(rgb, interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])

    for name in fields:
        print(f"\n===== A3  {name} ({DISP[name]}) =====")
        gid, cls, vals, lo, hi = per_grain_desc(name)
        G = cls.shape[0]
        dino_pix = np.load(f"{RUN[name]}/eval/inference.npz")["assigns"].astype(int)
        Km = len(np.unique(dino_pix))
        print(f"      matched K = {Km} (DINO active);  radial window r=[{lo},{hi}]")
        prof = radial_profiles(name, lo, hi)
        shape = np.log(np.clip(prof, 1e-6, None)); shape = shape - shape.mean(1, keepdims=True)
        shape = shape / (np.linalg.norm(shape, axis=1, keepdims=True) + 1e-9)

        # matched-K and auto-K clustering on shape profiles
        lab_m = KMeans(Km, n_init=8, random_state=0).fit_predict(shape)
        best = None
        for k in range(3, 19):
            lab = KMeans(k, n_init=6, random_state=0).fit_predict(shape)
            s = silhouette_score(shape, lab, sample_size=3000, random_state=0)
            if best is None or s > best[1]:
                best = (k, s, lab)
        Ka, sil, lab_a = best
        print(f"      auto K (silhouette) = {Ka} (score {sil:.3f})")

        # grain-level labels + eta^2
        gl_dino = cls
        gl_m = grain_labels(lab_m, gid, G)
        gl_a = grain_labels(lab_a, gid, G)
        ok = np.isfinite(vals["spot"]) & (cls >= 0)
        print(f"      {'descriptor':16s} {'DINO':>7s} {'1D matched':>11s} {'1D auto':>9s}")
        res = {}
        for k in KEYS:
            e_d = eta2(vals[k][ok], cls[ok])
            e_m = eta2(vals[k][ok], gl_m[ok])
            e_a = eta2(vals[k][ok], gl_a[ok])
            res[k] = (e_d, e_m, e_a)
            print(f"      {k:16s} {e_d:7.2f} {e_m:11.2f} {e_a:9.2f}")
        ari_m = adjusted_rand_score(dino_pix, lab_m)
        print(f"      ARI(DINO, 1D-matched) = {ari_m:.2f};  1D auto-K used {Ka} classes vs DINO {Km}")

        # figure: grain-level maps (DINO | 1D matched | 1D auto) + eta^2 bars
        fig = Figure(figsize=(16.5, 4.7), facecolor="white")
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.1], left=0.01, right=0.985,
                              top=0.84, bottom=0.14, wspace=0.12)
        draw(fig.add_subplot(gs[0, 0]), grain_map(gl_dino, gid), f"DINO (grain-level, K={Km})")
        draw(fig.add_subplot(gs[0, 1]), grain_map(gl_m, gid), f"1D radial k-means (matched K={Km})")
        draw(fig.add_subplot(gs[0, 2]), grain_map(gl_a, gid), f"1D radial k-means (auto K={Ka})")
        ax = fig.add_subplot(gs[0, 3])
        x = np.arange(len(KEYS)); w = 0.27
        ax.bar(x - w, [res[k][0] for k in KEYS], w, label="DINO (2D)", color="#2c7fb8")
        ax.bar(x, [res[k][1] for k in KEYS], w, label=f"1D matched (K={Km})", color="#d95f0e")
        ax.bar(x + w, [res[k][2] for k in KEYS], w, label=f"1D auto (K={Ka})", color="#fdae6b")
        ax.set_xticks(x); ax.set_xticklabels([LBL[k] for k in KEYS], fontsize=10)
        ax.set_ylabel("per-grain $\\eta^2$", fontsize=11); ax.set_ylim(0, 1)
        ax.legend(fontsize=8.5, loc="upper right"); ax.grid(axis="y", alpha=0.3)
        ax.set_title("what the 2D embedding adds", fontsize=11, fontweight="bold")
        fig.suptitle(f"A3  {name} ({DISP[name]}): DINO vs 1D radial-profile clustering "
                     f"(maps consolidated to grains)", fontsize=13, fontweight="bold", y=0.97)
        p = f"{OUT}/A3_1d_baseline_{name}.png"; fig.savefig(p, dpi=150, facecolor="white")
        print(f"      wrote {p}")


if __name__ == "__main__":
    flds = sys.argv[1:] or ["SI3", "SI4", "SI5"]
    main(flds)
