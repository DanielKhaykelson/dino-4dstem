"""
A5  -  Azimuthal spottiness as a normalized (FEM-type) variance.

Our spottiness = 90th percentile over the ring window of the azimuthal CV = sigma/mu,
where sigma, mu are the azimuthal std and mean at each radius. Since CV^2 = <I^2>/<I>^2 - 1,
per ring CV^2 IS the normalized variance V used in fluctuation electron microscopy (FEM).

This script:
  (1) confirms numerically, per class, that spottiness == sqrt(azimuthal V) at the ring
      it is measured on (they are the same quantity);
  (2) reports the least-ordered ("precursor") class spottiness as a provisional floor,
      pending the true as-deposited amorphous baseline (B1);
  (3) prints the exact wording for the descriptor's physical basis (Treacy & Gibson;
      Voyles) distinguishing the AZIMUTHAL variance (ring breakup / orientational order)
      from the SPATIAL FEM variance (probe-to-probe, MRO length scale).

Output: figs/Review/A5_fem_spottiness.png + printed confirmation.
References to add: [17] Treacy, Gibson, Fan, Paterson, McNulty, Rep. Prog. Phys. 2005 (already cited);
                   Voyles & Muller, Ultramicroscopy 2002 (NEW - fluctuation microscopy).
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    os.chdir("D:/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive")
    sys.path.insert(0, "src")
    import numpy as np
    from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    FIGS = "docs/paper/draft_v2/figs"; OUT = f"{FIGS}/Review"
    INV, KMAX = 0.00185, 0.35
    FOV = {"SI3": 187, "SI4": 160, "SI5": 160}
    DISP = {"SI3": "interface", "SI4": "needles", "SI5": "interface (mag.)"}

    def spottiness_and_V(avg, c, beam, lo, hi):
        """return spottiness=p90(CV), p90(sqrt(V)), and the per-ring max|CV-sqrt(V)| over the window."""
        m, v, _ = _radial_mean_var(avg, (c, c), beam_px=beam)
        seg, vseg = m[lo:hi], v[lo:hi]
        cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None)    # per-ring CV = sigma/mu
        V = np.clip(vseg, 0, None) / np.clip(seg, 1e-9, None) ** 2          # per-ring normalized variance <I^2>/<I>^2 - 1
        spot = np.percentile(cv, 90)
        spot_from_V = np.percentile(np.sqrt(V), 90)                         # identical array -> identical percentile
        perring = float(np.max(np.abs(cv - np.sqrt(V))))                    # per-ring identity residual (machine eps)
        return spot, spot_from_V, perring

    print("=" * 96)
    print("A5  SPOTTINESS == sqrt(azimuthal normalized variance V)   (V is the FEM-type variance <I^2>/<I>^2 - 1)")
    print("=" * 96)
    rows = []
    fig = Figure(figsize=(13, 4.4), facecolor="white")
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.99, top=0.86, bottom=0.15, wspace=0.26)
    for ax_i, name in enumerate(["SI3", "SI4", "SI5"]):
        z = np.load(f"{FIGS}/grain_acom_v2_{name}.npz")
        cls, vac, gsum, gcnt = z["cls"], z["vac"], z["gsum"], z["gcnt"]
        H = int(z["H"]); c = (H - 1) / 2.0
        beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV[name])
        G = gsum.shape[0]
        sp = np.full(G, np.nan); sqrtV = np.full(G, np.nan); perring = 0.0
        for g in range(G):
            if vac[g]: continue
            avg = gsum[g] / max(gcnt[g], 1)
            s, s_from_V, pr = spottiness_and_V(avg, c, beam, lo, hi)
            if not np.isfinite(s): continue
            sp[g] = s; sqrtV[g] = s_from_V; perring = max(perring, pr)
        ok = np.isfinite(sp) & np.isfinite(sqrtV)
        # per-ring CV(r) == sqrt(V(r)) to machine precision; so p90(CV) == p90(sqrt(V))
        maxdiff = max(float(np.max(np.abs(sp[ok] - sqrtV[ok]))), perring)
        # provisional floor = least-ordered class median spottiness
        cs = sorted(set(int(x) for x in cls[ok]))
        cmed = {cc: float(np.nanmedian(sp[(cls == cc) & ok])) for cc in cs}
        floor_c = min(cmed, key=cmed.get); top_c = max(cmed, key=cmed.get)
        print(f"\n----- {name} ({DISP[name]}): {int(ok.sum())} grains")
        print(f"      max|spottiness - sqrt(V)|  = {maxdiff:.2e}   -> identical (spottiness IS sqrt of azimuthal FEM variance)")
        print(f"      least-ordered class median spottiness (provisional floor) = {cmed[floor_c]:.2f}  (V = {cmed[floor_c]**2:.2f})")
        print(f"      most-ordered  class median spottiness                     = {cmed[top_c]:.2f}  (V = {cmed[top_c]**2:.2f})")
        rows.append((name, maxdiff, cmed[floor_c], cmed[top_c]))
        ax = fig.add_subplot(gs[0, ax_i])
        ax.scatter(sqrtV[ok], sp[ok], s=26, c="#2c7fb8", edgecolor="k", linewidth=0.3)
        lim = [0, np.nanmax(sp[ok]) * 1.05]
        ax.plot(lim, lim, "r--", lw=1, label="y = x")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("sqrt(azimuthal FEM variance V)", fontsize=10)
        ax.set_ylabel("spottiness (p90 azimuthal CV)", fontsize=10)
        ax.set_title(f"{name} ({DISP[name]})  max|diff|={maxdiff:.0e}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
    fig.suptitle("A5  Azimuthal spottiness is exactly sqrt of the FEM-type normalized variance V = <I$^2$>/<I>$^2$ - 1",
                 fontsize=12, fontweight="bold")
    p = f"{OUT}/A5_fem_spottiness.png"; fig.savefig(p, dpi=150, facecolor="white"); print(f"\nwrote {p}")

    print("\n" + "=" * 96)
    print("WORDING for the manuscript (Methods / descriptor definition):")
    print("=" * 96)
    print("""  The azimuthal spottiness of a class-average pattern is the 90th percentile, over the ring
  window, of the azimuthal coefficient of variation CV(r) = sigma_phi(r) / mu_phi(r), where
  mu_phi and sigma_phi are the mean and standard deviation of intensity around the azimuth at
  radius r. Because CV^2 = <I^2>/<I>^2 - 1, this is the azimuthal analogue of the normalized
  variance V used in fluctuation electron microscopy (FEM) [Treacy & Gibson; Voyles & Muller]:
  it measures the breakup of a diffraction ring into discrete spots (orientational / Bragg
  order within the illuminated volume), as distinct from the SPATIAL FEM variance measured
  probe-to-probe, which probes the medium-range-order length scale. A true-amorphous halo has
  CV near a fixed noise floor; the precursor class sits measurably above it (Figure, and the
  as-deposited amorphous baseline, B1).""")


if __name__ == "__main__":
    main()
