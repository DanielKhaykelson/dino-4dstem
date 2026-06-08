"""Add the classical-baseline test + a comparable report_auto.md to IMC_SI5
WITHOUT overwriting its curated manuscript figures (test1/test5/class_mean
are left untouched; we reuse the offline probe/ablation/orientation numbers)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import SAMPLES
from gui_app import interpret_core as ic

RUN = r"runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60"
CUBE = r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"
SAMPLES["IMC_SI5"] = dict(path=CUBE, vmax=5.0, scan_shape=(128, 128),
                          center_mask_radius=22)
inf = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)
ctx = ic.Ctx(RUN, "IMC_SI5", CUBE, 5.0, (128, 128), inf["embeds"],
             inf["assigns"], (20, 40, 120, False))
out = ctx.out
print(f"[si5] K={ctx.K} N={ctx.assigns.size}", flush=True)

def prog(d, t, s):
    if d % 32 == 0 or d == t:
        print(f"  {s}: {d}/{t}", flush=True)

# classical needs radial + downsampled patterns; compute_factors_and_means
# writes NO figures (safe), classical_baselines writes only its own files.
fac = ic.compute_factors_and_means(ctx, collect_classical=True, progress=prog)
classical = ic.classical_baselines(ctx, fac, progress=prog)
print("  classical best:", classical["best"], classical["best_ARI"], flush=True)

# reuse offline probe + ablation + orientation numbers for the report
t1 = json.load(open(os.path.join(out, "test1_ranking.json")))
rows = [dict(factor=r["factor"], probe_R2=r["probe_R2"],
             eta2=r.get("eta2", r.get("eta2_separates")), MI=r["MI"])
        for r in t1["continuous"]]
cat = {}
if "phase" in t1:
    cat["phase"] = {"AMI": round(t1["phase"]["AMI"], 4),
                    "ARI": round(t1["phase"]["ARI"], 4)}
to_p = os.path.join(out, "test_orientation_auto.json")
if os.path.exists(to_p):
    to = json.load(open(to_p))
    cat["zone_axis"] = {"AMI": to["DINO_vs_ZA_AMI"], "ARI": to["DINO_vs_ZA_ARI"],
                        "n_indexed": to["n_indexed"],
                        "n_zone_axes": to["n_zone_axes"]}
probe = dict(rows=rows, categorical=cat)
ab = json.load(open(os.path.join(out, "test2_4_ablations.json")))
t4p = os.path.join(out, "test4_scattered_norm.json")
if os.path.exists(t4p):
    ab.update(json.load(open(t4p)))     # add scattered_norm
acom = ic.find_acom_arrays(RUN)
rep = ic.write_report(ctx, probe=probe, ablations=ab, acom=acom,
                      classical=classical, did_gradcam=True)
print("[si5] report ->", rep, flush=True)
