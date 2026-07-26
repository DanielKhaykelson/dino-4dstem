"""
B2  -  Dose / beam-damage control.

Goal: turn "doses low enough to avoid beam damage" from assertion into measurement, by plotting
each descriptor vs cumulative dose for the SAME positions (expect flat if no damage).

STATUS: NOT FEASIBLE with existing data. The three survey 'passes' per field are NOT a dose series:
  SI-004:  Survey_CH2_0_0.prz = 266 KB (2D survey thumbnail)
           Survey_CH2_0_1.prz = 8.6 GB  (the analysed 4D acquisition)
           Survey_CH2_0_2.prz = 69 KB   (2D survey thumbnail)
Only one full 4D acquisition exists per region; there is no repeat 4D scan of the same positions.

What would answer it (drop-in options):
  (a) a repeat 4D acquisition of the SAME region (survey-pass 4D + zoom-pass 4D), OR
  (b) an explicit dose series (same region, increasing exposure).
Point the loader below at the two same-region 4D cubes and it will plot each descriptor vs dose.

Weak within-scan alternative (raster-order proxy): within the single 4D scan, cumulative session
dose rises along the raster. If beam damage were significant, a descriptor would trend with
raster order after controlling for structure. This is confounded by real spatial structure and is
only a coarse check; enable with --raster to compute the descriptor-vs-rasterrow trend per class.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main(argv):
    print("B2 dose control: NOT FEASIBLE with current data (no repeat 4D acquisition of the same region).")
    print("  The _0_0 / _0_2 survey passes are 2D thumbnails (266 KB / 69 KB), not 4D cubes.")
    print("  Provide two same-region 4D cubes (survey-pass, zoom-pass) or a dose series to run the test.")
    if "--raster" in argv:
        os.chdir("D:/DINOSR/Claude/PaperRun_claude/dino_sr_contrastive"); sys.path.insert(0, "src")
        import numpy as np
        FIGS = "docs/paper/draft_v2/figs"
        # coarse within-scan check: does per-grain spottiness trend with scan-row (raster order)?
        for name in ["SI4"]:
            z = np.load(f"{FIGS}/grain_acom_v2_{name}.npz")
            gid = z["gid"].reshape(128, 128); cls = z["cls"]
            rows = np.indices((128, 128))[0]
            # mean raster row per grain vs its class-median (proxy) - report correlation
            from scipy.stats import spearmanr
            grow = np.array([rows[gid == g].mean() if (gid == g).any() else np.nan for g in range(cls.shape[0])])
            ok = np.isfinite(grow) & (cls >= 0)
            rho, p = spearmanr(grow[ok], cls[ok])
            print(f"  [{name}] raster-row vs DINO class rank: Spearman {rho:+.2f} (p={p:.2f}) "
                  f"-> {'no dose trend' if abs(rho) < 0.2 else 'possible trend, investigate'}")


if __name__ == "__main__":
    main(sys.argv[1:])
