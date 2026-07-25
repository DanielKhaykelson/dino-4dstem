"""
A3  -  1D-radial-profile clustering baseline.

Referee: "why not just cluster the azimuthally-averaged 1D radial profiles?"
This quantifies what the 2D rotation-invariant embedding adds over 1D.

For each IMC field we azimuthally integrate every diffraction pattern to a 1D radial
profile, k-means at K matched to the DINO active-class count, map it, and compare the
per-grain descriptor eta^2 of the 1D clustering to that of DINO.

Two 1D baselines (we report the stronger, to be fair, not a strawman):
  raw   = cluster the raw radial profiles (carries intensity/thickness)
  shape = cluster L2-normalised log profiles (shape only)

Expectation / point of the test: 1D recovers the RADIAL part of the axis (chi, some B)
but not azimuthal spottiness, because azimuthal averaging discards the ring-breakup signal
that separates the crystalline classes. DINO's 2D embedding recovers all three.

Outputs: figs/revision/A3_1d_baseline_{field}.png + printed eta^2 comparison.
Machine rules: no DataLoader (sklearn KMeans, single process); entry point guarded.
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


def main(fields):
    os.chdir("D:/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive")
    sys.path.insert(0, "src")
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib as mpl

    FIGS = "docs/paper/draft_v2/figs"; OUT = f"{FIGS}/revision"

    def eta2(v, lab):
        v = np.asarray(v, float); lab = np.asarray(lab)
        ok = np.isfinite(v); v, lab = v[ok], lab[ok]
        if v.size < 5 or len(set(lab.tolist())) < 2:
            return np.nan
        gm = v.mean(); sst = ((v - gm) ** 2).sum()
        ssb = sum(((v[lab == c].mean() - gm) ** 2) * (lab == c).sum() for c in set(lab.tolist()))
        return float(ssb / (sst + 1e-12))

    def per_grain_desc(name):
        """descriptors + grain-id map, recomputed from source (same as A1/Fig5)."""
        z = np.load(f"{FIGS}/grain_acom_v2_{name}.npz")
        cls, vac, gsum, gcnt, gid = z["cls"], z["vac"], z["gsum"], z["gcnt"], z["gid"]
        H = int(z["H"]); c = (H - 1) / 2.0
        beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV[name])
        yy, xx = np.indices((H, H)); rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
        band = (rr >= lo) & (rr <= hi)
        G = gsum.shape[0]
        spot = np.full(G, np.nan); B = np.full(G, np.nan); chi = np.full(G, np.nan)
        for g in range(G):
            if vac[g]:
                continue
            avg = gsum[g] / max(gcnt[g], 1)
            m, v, _ = _radial_mean_var(avg, (c, c), beam_px=beam)
            seg, vseg = m[lo:hi], v[lo:hi]
            if seg.size < 5 or seg.sum() <= 0:
                continue
            halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
            chi[g] = (np.clip(seg - halo, 0, None) / np.clip(halo, 1e-9, None)).max()
            spot[g] = np.percentile(np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None), 90)
            hf = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1])
            B[g] = np.clip(avg[band] - hf[band], 0, None).sum() / (hf[band].sum() + 1e-9)
        return gid, cls, dict(spot=spot, B=B, chi=chi), beam, lo, hi

    def radial_profiles(path, lo, hi):
        cube = np.load(path, mmap_mode="r")
        assert cube.shape[0] * cube.shape[1] == NY * NX, cube.shape
        H = cube.shape[-1]; cube = cube.reshape(-1, H, H)
        c = (H - 1) / 2.0
        yy, xx = np.indices((H, H)); rb = np.hypot(yy - c, xx - c).astype(int).ravel()
        nb = rb.max() + 1; cnt = np.bincount(rb, minlength=nb).astype(np.float64)
        N = cube.shape[0]; prof = np.zeros((N, hi - lo), np.float64)
        t0 = time.time()
        for i in range(N):
            m = np.bincount(rb, np.asarray(cube[i], np.float64).ravel(), minlength=nb) / cnt
            prof[i] = m[lo:hi]
            if i % 4000 == 0:
                print(f"      radial {i}/{N}  {time.time()-t0:.0f}s", flush=True)
        return prof

    for name in fields:
        print(f"\n===== A3  {name} ({DISP[name]}) =====")
        gid, cls, vals, beam, lo, hi = per_grain_desc(name)
        dino_pix = np.load(f"{RUN[name]}/eval/inference.npz")["assigns"].astype(int)
        K = len(np.unique(dino_pix))
        print(f"      matched K = {K} (DINO active classes);  radial window r=[{lo},{hi}]")

        prof = radial_profiles(CUBES[name], lo, hi)
        # naive baseline: raw profiles ; fair baseline: L2-normalised log profiles (shape)
        raw = prof
        shape = np.log(np.clip(prof, 1e-6, None))
        shape = shape - shape.mean(1, keepdims=True)
        shape = shape / (np.linalg.norm(shape, axis=1, keepdims=True) + 1e-9)
        lab_raw = KMeans(K, n_init=8, random_state=0).fit_predict(raw)
        lab_shape = KMeans(K, n_init=8, random_state=0).fit_predict(shape)

        # per-grain majority label for each clustering, and DINO grain label = cls[grain]
        def grain_labels(pixlab):
            G = cls.shape[0]; out = np.full(G, -1)
            for g in range(G):
                px = pixlab[gid == g]
                if px.size:
                    u, ct = np.unique(px, return_counts=True); out[g] = u[np.argmax(ct)]
            return out
        gl_raw = grain_labels(lab_raw); gl_shape = grain_labels(lab_shape)
        ok = np.isfinite(vals["spot"]) & (cls >= 0)

        print(f"      {'descriptor':16s} {'DINO eta2':>10s} {'1D-shape eta2':>14s} {'1D-raw eta2':>12s} {'2D adds':>9s}")
        res = {}
        for k in KEYS:
            e_dino = eta2(vals[k][ok], cls[ok])
            e_shape = eta2(vals[k][ok], gl_shape[ok])
            e_raw = eta2(vals[k][ok], gl_raw[ok])
            res[k] = (e_dino, e_shape, e_raw)
            best1d = np.nanmax([e_shape, e_raw])
            print(f"      {k:16s} {e_dino:10.2f} {e_shape:14.2f} {e_raw:12.2f} {e_dino-best1d:+9.2f}")
        ari = adjusted_rand_score(dino_pix, lab_shape)
        print(f"      partition similarity DINO vs 1D-shape map: ARI = {ari:.2f}")

        # figure: DINO map | 1D-shape map | eta2 bars
        fig = Figure(figsize=(14, 4.6), facecolor="white")
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], left=0.02, right=0.985,
                              top=0.86, bottom=0.14, wspace=0.18)
        for ci, (lab, ttl) in enumerate([(dino_pix, f"DINO class map (K={K})"),
                                          (lab_shape, f"1D radial-profile k-means (K={K})")]):
            ax = fig.add_subplot(gs[0, ci]); mp = lab.reshape(NY, NX)
            u = np.unique(mp); rmap = {c: i for i, c in enumerate(u)}
            disp = np.vectorize(rmap.get)(mp)
            cols = mpl.colormaps.get_cmap("tab20")(np.linspace(0, 1, 20))[:len(u)]
            ax.imshow(disp, cmap=ListedColormap(cols), norm=BoundaryNorm(np.arange(-0.5, len(u)), len(u)),
                      interpolation="nearest", aspect="equal")
            ax.set_title(ttl, fontsize=12, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[0, 2])
        x = np.arange(len(KEYS)); w = 0.38
        ax.bar(x - w/2, [res[k][0] for k in KEYS], w, label="DINO (2D)", color="#2c7fb8")
        ax.bar(x + w/2, [np.nanmax([res[k][1], res[k][2]]) for k in KEYS], w, label="1D profiles (best)", color="#d95f0e")
        ax.set_xticks(x); ax.set_xticklabels([LBL[k] for k in KEYS], fontsize=10)
        ax.set_ylabel("per-grain eta$^2$ (class effect)", fontsize=11); ax.set_ylim(0, 1)
        ax.legend(fontsize=10, loc="upper right"); ax.grid(axis="y", alpha=0.3)
        ax.set_title("what the 2D embedding adds", fontsize=12, fontweight="bold")
        fig.suptitle(f"A3  {name} ({DISP[name]}): DINO vs 1D-radial-profile clustering",
                     fontsize=13, fontweight="bold")
        p = f"{OUT}/A3_1d_baseline_{name}.png"
        fig.savefig(p, dpi=150, facecolor="white"); print(f"      wrote {p}")


if __name__ == "__main__":
    flds = sys.argv[1:] or ["SI3", "SI4", "SI5"]
    main(flds)
