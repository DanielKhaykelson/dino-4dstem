"""
A4  -  Non-circular statistics panel for the crystallinity axis.

Replaces the tautological "monotonic when sorted by median" framing with the real,
per-grain evidence, recomputed from source (grain_acom_v2_*.npz), NOT copied from notes:

  (1) per-grain eta^2 of the class label on each descriptor  (class predicts the descriptor)
  (2) Kruskal-Wallis p across classes                        (non-parametric significance)
  (3) inter-descriptor Spearman rank agreement, per field and pooled over all classes
      (three independent descriptors concordantly rank the classes)

Key framing point foregrounded: azimuthal spottiness is ORTHOGONAL to DINO's 1D radial
training loss (the loss uses the azimuthally-averaged profile, which discards azimuthal
structure), so the class ordering it produces is not something the model was trained to
optimise; chi and B partly overlap the radial loss.

Output: figs/Review/A4_stats_panel.png + printed table.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    os.chdir("D:/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive")
    sys.path.insert(0, "src")
    import numpy as np
    from scipy.stats import kruskal, spearmanr
    from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    FIGS = "docs/paper/draft_v2/figs"; OUT = f"{FIGS}/Review"
    INV, KMAX = 0.00185, 0.35
    FOV = {"SI3": 187, "SI4": 160, "SI5": 160}
    DISP = {"SI3": "interface", "SI4": "needles", "SI5": "interface (mag.)"}
    KEYS = ["spot", "B", "chi"]
    LBL = {"spot": "spottiness", "B": "Bragg excess B", "chi": "peak/halo chi"}
    ORTH = {"spot": "orthogonal to 1D loss", "B": "partial overlap", "chi": "partial overlap"}

    def eta2(v, lab):
        v = np.asarray(v, float); lab = np.asarray(lab); ok = np.isfinite(v); v, lab = v[ok], lab[ok]
        if v.size < 5 or len(set(lab.tolist())) < 2: return np.nan
        gm = v.mean(); sst = ((v - gm) ** 2).sum()
        ssb = sum(((v[lab == c].mean() - gm) ** 2) * (lab == c).sum() for c in set(lab.tolist()))
        return float(ssb / (sst + 1e-12))

    def per_grain(name):
        z = np.load(f"{FIGS}/grain_acom_v2_{name}.npz")
        cls, vac, gsum, gcnt = z["cls"], z["vac"], z["gsum"], z["gcnt"]
        H = int(z["H"]); c = (H - 1) / 2.0
        beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV[name])
        yy, xx = np.indices((H, H)); rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2); band = (rr >= lo) & (rr <= hi)
        G = gsum.shape[0]; spot = np.full(G, np.nan); B = np.full(G, np.nan); chi = np.full(G, np.nan)
        for g in range(G):
            if vac[g]: continue
            avg = gsum[g] / max(gcnt[g], 1)
            m, v, _ = _radial_mean_var(avg, (c, c), beam_px=beam); seg, vseg = m[lo:hi], v[lo:hi]
            if seg.size < 5 or seg.sum() <= 0: continue
            halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
            chi[g] = (np.clip(seg - halo, 0, None) / np.clip(halo, 1e-9, None)).max()
            spot[g] = np.percentile(np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None), 90)
            hf = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1])
            B[g] = np.clip(avg[band] - hf[band], 0, None).sum() / (hf[band].sum() + 1e-9)
        return cls, dict(spot=spot, B=B, chi=chi)

    NAMES = ["SI3", "SI4", "SI5"]
    DATA = {n: per_grain(n) for n in NAMES}
    stats = {}; pooled = {k: [] for k in KEYS}; pooled_meta = []
    print("=" * 92)
    print("A4  NON-CIRCULAR STATISTICS")
    print("=" * 92)
    for name in NAMES:
        cls, vals = DATA[name]; ok = np.isfinite(vals["spot"])
        cs = sorted(set(int(c) for c in cls[ok]))
        print(f"\n----- {name} ({DISP[name]}): {int(ok.sum())} grains, {len(cs)} classes")
        st = {}
        for k in KEYS:
            v = vals[k]; e = eta2(v[ok], cls[ok])
            groups = [v[(cls == c) & ok] for c in cs if np.isfinite(v[(cls == c) & ok]).sum() > 0]
            H, p = kruskal(*groups)
            med = {c: float(np.nanmedian(v[(cls == c) & ok])) for c in cs}
            st[k] = (e, p, med)
            for c in cs: pooled[k].append(med[c])
            print(f"      {LBL[k]:16s} eta2 = {e:.2f}   Kruskal-Wallis p = {p:.1e}   ({ORTH[k]})")
        # per-field inter-descriptor Spearman
        for i in range(3):
            for j in range(i + 1, 3):
                a = [st[KEYS[i]][2][c] for c in cs]; b = [st[KEYS[j]][2][c] for c in cs]
                rho, pp = spearmanr(a, b)
                print(f"      Spearman {KEYS[i]:4s}~{KEYS[j]:4s} = {rho:+.2f} (p={pp:.1e})")
        stats[name] = st
        for c in cs: pooled_meta.append(name)

    print("\n----- POOLED over all 25 classes (3 independently trained fields)")
    prho = {}
    for i in range(3):
        for j in range(i + 1, 3):
            rho, pp = spearmanr(pooled[KEYS[i]], pooled[KEYS[j]])
            prho[(KEYS[i], KEYS[j])] = rho
            print(f"      Spearman {KEYS[i]:4s}~{KEYS[j]:4s} = {rho:+.2f} (p={pp:.1e})")

    # ---- figure: eta2 bars (left) + pooled inter-descriptor concordance (right) ----
    fig = Figure(figsize=(14.5, 5.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], left=0.06, right=0.985, top=0.88, bottom=0.13, wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(NAMES)); w = 0.26
    colk = {"spot": "#2c7fb8", "B": "#41ab5d", "chi": "#d95f0e"}
    for ki, k in enumerate(KEYS):
        vals = [stats[n][k][0] for n in NAMES]
        bars = ax.bar(x + (ki - 1) * w, vals, w, label=LBL[k], color=colk[k])
        for xi, n in zip(x + (ki - 1) * w, NAMES):
            p = stats[n][k][1]; star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"
            ax.text(xi, stats[n][k][0] + 0.02, star, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({DISP[n]})" for n in NAMES], fontsize=10)
    ax.set_ylabel("per-grain $\\eta^2$  (class predicts descriptor)", fontsize=11); ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Class label predicts each per-grain descriptor\n(stars: Kruskal-Wallis significance)",
                 fontsize=11, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    pm = np.array(pooled_meta)
    cm = {"SI3": "#7b3294", "SI4": "#008837", "SI5": "#e66101"}
    sp = np.array(pooled["spot"]); ch = np.array(pooled["chi"])
    for n in NAMES:
        m = pm == n
        ax2.scatter(np.array(pooled["spot"])[m], np.array(pooled["chi"])[m], s=55, c=cm[n],
                    edgecolor="k", linewidth=0.4, label=n, alpha=0.9)
    ax2.set_xlabel("class-median spottiness", fontsize=11); ax2.set_ylabel("class-median peak/halo chi", fontsize=11)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_title(f"Three descriptors rank the classes concordantly\n"
                  f"pooled Spearman: spot~chi {prho[('spot','chi')]:+.2f}, "
                  f"spot~B {prho[('spot','B')]:+.2f}, B~chi {prho[('B','chi')]:+.2f}",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, title="field"); ax2.grid(alpha=0.3, which="both")
    fig.suptitle("A4  The crystallinity axis is statistically real and non-circular", fontsize=13, fontweight="bold")
    p = f"{OUT}/A4_stats_panel.png"; fig.savefig(p, dpi=150, facecolor="white"); print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
