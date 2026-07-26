"""
A1  -  Thickness-confound control for the 'single crystallinity axis'.

Referee worry: DINO may cluster by total scattered intensity, and the descriptors may
merely track thickness. We test this three ways, per co-located IMC field (SI3/SI4/SI5):

  1. raw eta^2 of the class label on each descriptor          (spottiness, B, chi)
  2. PARTIAL eta^2 controlling for a thickness proxy          (regress proxy out, redo eta^2)
  3. class separation WITHIN fixed thickness bands            (3 terciles)
  plus: how much the class label explains the thickness proxy itself
        (if classes were thickness bins this would be high).

Thickness proxy = per-grain integrated intensity of the grain-average diffraction pattern:
  I_scat  = mean over the scattered band (beam < r < window edge)   <- the referee's exact worry
  I_tot   = unmasked mean over the whole pattern                    <- exposure/thickness

Descriptors are recomputed from source (grain-average patterns in grain_acom_v2_*.npz),
identical recipe to Figure 5, NOT copied from the notes.

Outputs: figs/revision/A1_thickness_{field}.png  and a printed table.
Machine rules: no DataLoader here; entry point guarded; radial profiles are log-y elsewhere.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    os.chdir("D:/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive")
    sys.path.insert(0, "src")
    import numpy as np
    from scipy.stats import spearmanr
    from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import matplotlib as mpl

    FIGS = "docs/paper/draft_v2/figs"
    OUT = "docs/paper/draft_v2/figs/Review"
    INV, KMAX = 0.00185, 0.35
    FOV = {"SI3": 187, "SI4": 160, "SI5": 160}
    DISP = {"SI3": "interface", "SI4": "needles", "SI5": "interface (mag.)"}
    KEYS = ["spot", "B", "chi"]
    LBL = {"spot": "azimuthal spottiness", "B": "Bragg excess B", "chi": "radial peak/halo chi"}

    def eta2(v, lab):
        v = np.asarray(v, float); lab = np.asarray(lab)
        ok = np.isfinite(v); v, lab = v[ok], lab[ok]
        if v.size < 5 or len(set(lab.tolist())) < 2:
            return np.nan
        gm = v.mean(); sst = ((v - gm) ** 2).sum()
        ssb = sum(((v[lab == c].mean() - gm) ** 2) * (lab == c).sum() for c in set(lab.tolist()))
        return float(ssb / (sst + 1e-12))

    def partial_eta2(desc, lab, thick):
        """eta^2 of class on the residual of desc after regressing out thick (linear)."""
        desc = np.asarray(desc, float); thick = np.asarray(thick, float); lab = np.asarray(lab)
        ok = np.isfinite(desc) & np.isfinite(thick)
        desc, thick, lab = desc[ok], thick[ok], lab[ok]
        A = np.vstack([thick, np.ones_like(thick)]).T
        coef, *_ = np.linalg.lstsq(A, desc, rcond=None)
        resid = desc - A @ coef
        r2 = 1.0 - (resid ** 2).sum() / (((desc - desc.mean()) ** 2).sum() + 1e-12)  # var(desc) explained by thick
        return eta2(resid, lab), float(r2)

    def per_grain(name):
        z = np.load(f"{FIGS}/grain_acom_v2_{name}.npz")
        cls, vac, gsum, gcnt = z["cls"], z["vac"], z["gsum"], z["gcnt"]
        H = int(z["H"]); c = (H - 1) / 2.0
        beam = max(8, round(0.11 * H)); lo = beam + 1; hi = min(int(KMAX / INV), FOV[name])
        yy, xx = np.indices((H, H)); rr = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
        band = (rr >= lo) & (rr <= hi)
        G = gsum.shape[0]
        spot = np.full(G, np.nan); B = np.full(G, np.nan); chi = np.full(G, np.nan)
        I_scat = np.full(G, np.nan); I_tot = np.full(G, np.nan)
        for g in range(G):
            if vac[g]:
                continue
            avg = gsum[g] / max(gcnt[g], 1)
            m, v, _ = _radial_mean_var(avg, (c, c), beam_px=beam)
            seg, vseg = m[lo:hi], v[lo:hi]
            if seg.size < 5 or seg.sum() <= 0:
                continue
            halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
            pk = np.clip(seg - halo, 0, None)
            chi[g] = (pk / np.clip(halo, 1e-9, None)).max()
            cv = np.sqrt(np.clip(vseg, 0, None)) / np.clip(seg, 1e-9, None)
            spot[g] = np.percentile(cv, 90)
            hf = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1])
            B[g] = np.clip(avg[band] - hf[band], 0, None).sum() / (hf[band].sum() + 1e-9)
            I_scat[g] = avg[band].mean()
            I_tot[g] = avg.mean()
        return cls, dict(spot=spot, B=B, chi=chi), I_scat, I_tot

    print("=" * 104)
    print("A1  THICKNESS-CONFOUND CONTROL   (proxy = per-grain integrated intensity of the class/grain-average pattern)")
    print("    raw eta2  = class effect on the descriptor")
    print("    partial   = class effect after regressing out the thickness proxy (I_scat)")
    print("    R2(thick) = fraction of the descriptor's variance explained by thickness alone")
    print("    within-band eta2 = class effect inside each thickness tercile (T1<T2<T3)")
    print("=" * 104)

    rows = []
    for name in ["SI3", "SI4", "SI5"]:
        cls, vals, I_scat, I_tot = per_grain(name)
        ok = np.isfinite(vals["spot"]) & np.isfinite(I_scat)
        lab = cls[ok]
        thick = I_scat[ok]
        thick_tot = I_tot[ok]
        # class -> thickness: does the class label predict thickness?
        e_ct = eta2(thick, lab)
        rho_ct, _ = spearmanr(thick, lab)  # not meaningful (labels nominal); use class-median thickness vs desc below
        # thickness terciles
        q = np.quantile(thick, [1/3, 2/3])
        terc = np.digitize(thick, q)  # 0,1,2
        print(f"\n----- {name} ({DISP[name]}):  {int(ok.sum())} grains, {len(set(lab.tolist()))} classes")
        print(f"      class label explains eta2 = {e_ct:.2f} of the thickness proxy itself "
              f"({'classes ARE thickness-like' if e_ct > 0.6 else 'classes cut ACROSS thickness'})")
        print(f"      {'descriptor':22s} {'raw eta2':>9s} {'part|scat':>10s} {'part|total':>11s} "
              f"{'R2(scat)':>9s} {'corr d~thk':>11s} {'  within-band eta2 (T1/T2/T3)':>30s}")
        for k in KEYS:
            desc = vals[k][ok]
            e_raw = eta2(desc, lab)
            e_par, r2 = partial_eta2(desc, lab, thick)          # control for scattered intensity
            e_par_tot, r2_tot = partial_eta2(desc, lab, thick_tot)  # control for total/unmasked intensity
            rho, _ = spearmanr(desc, thick)
            wb = []
            for t in range(3):
                mm = terc == t
                wb.append(eta2(desc[mm], lab[mm]))
            wbtxt = "/".join(f"{x:.2f}" if np.isfinite(x) else " - " for x in wb)
            print(f"      {LBL[k]:22s} {e_raw:9.2f} {e_par:10.2f} {e_par_tot:11.2f} {r2:9.2f} {rho:11.2f} "
                  f"{wbtxt:>30s}")
            rows.append((name, k, e_raw, e_par, e_par_tot, r2, rho, e_ct, wb))

        # figure: descriptor vs thickness, coloured by class
        fig = Figure(figsize=(15, 4.4), facecolor="white")
        gs = fig.add_gridspec(1, 3, left=0.06, right=0.995, top=0.86, bottom=0.14, wspace=0.24)
        cmap = mpl.colormaps.get_cmap("tab20")
        classes = sorted(set(lab.tolist()))
        cidx = {c: i for i, c in enumerate(classes)}
        for ci, k in enumerate(KEYS):
            ax = fig.add_subplot(gs[0, ci])
            desc = vals[k][ok]
            cols = [cmap(cidx[c] % 20) for c in lab]
            ax.scatter(thick, desc, c=cols, s=34, edgecolor="k", linewidth=0.3, alpha=0.9)
            e_raw = eta2(desc, lab); e_par, r2 = partial_eta2(desc, lab, thick)
            ax.set_xlabel("thickness proxy  (scattered intensity, a.u.)", fontsize=10)
            ax.set_ylabel(LBL[k], fontsize=10)
            ax.set_title(f"{LBL[k]}\nraw eta2={e_raw:.2f}   partial|thickness={e_par:.2f}   "
                         f"R2(thick)={r2:.2f}", fontsize=10, fontweight="bold")
            ax.grid(alpha=0.25)
        fig.suptitle(f"A1  {name} ({DISP[name]}) - descriptors vs thickness proxy, coloured by DINO class",
                     fontsize=12, fontweight="bold")
        p = f"{OUT}/A1_thickness_{name}.png"
        fig.savefig(p, dpi=150, facecolor="white")
        print(f"      wrote {p}")

    print("\n" + "=" * 104)
    print("VERDICT (per descriptor, worst case across the three fields):")
    for k in KEYS:
        rr = [r for r in rows if r[1] == k]
        raw = np.nanmin([r[2] for r in rr]); par = np.nanmin([r[3] for r in rr])
        par_tot = np.nanmin([r[4] for r in rr]); r2 = np.nanmax([r[5] for r in rr])
        keep = min(par, par_tot) >= 0.30
        print(f"  {LBL[k]:22s} raw>= {raw:.2f}  partial|scat>= {par:.2f}  partial|total>= {par_tot:.2f}  "
              f"maxR2(thick)= {r2:.2f}  -> {'SURVIVES' if keep else 'CONTAMINATED'}")
    print("=" * 104)


if __name__ == "__main__":
    main()
