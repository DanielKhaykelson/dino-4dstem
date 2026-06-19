"""Per-prototype GradCAM/IG for the IMC runs via the GUI's viz_paper_attribution,
with SAMPLES entries injected to the CORRECT _nbed.cube + training vmax=5
(model preprocessing = what it was trained on; spatial preproc read from run cfg).
Run one sample:  python run_gradcam_imc.py SI4"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data
NB = r"D:\DINOSR\data\231228-IMC150nm-0p2apersec-anneal-70c-60min\EF-4DSTEM"
IMC = {
 "IMC_SI3": dict(path=os.path.join(NB, "SI-003", "Survey_CH2_1_nbed.cube.npy"),
                 run="runs/_gui/IMC_SI3_m097k60", cmr=15),
 "IMC_SI4": dict(path=os.path.join(NB, "SI-004", "Survey_CH2_0_1_nbed.cube.npy"),
                 run="runs/_gui/IMC_SI4_m097_k60", cmr=20),
 "IMC_SI5": dict(path=r"D:\DINOSR\data\IMC_150nm_SI5_nbed.cube.npy",
                 run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", cmr=20),
}
for k, c in IMC.items():
    data.SAMPLES[k] = {"path": c["path"], "vmax": 5, "scan_shape": (128, 128),
                       "center_mask_radius": c["cmr"], "approved_label": None}

import viz_paper_attribution as vpa
key = "IMC_" + (sys.argv[1] if len(sys.argv) > 1 else "SI4")
c = IMC[key]
print(f"[gradcam] {key}: run={c['run']} data={os.path.basename(c['path'])} vmax=5", flush=True)
out = vpa.run(c["run"], key, n_samples_per_proto=3)
print(f"[gradcam] wrote {out}", flush=True)
import shutil
dst = os.path.join("docs/paper/draft_v2/figs/latest_review", f"gradcam_{key}.png")
try:
    shutil.copy(out, dst); print(f"[gradcam] copied -> {dst}", flush=True)
except Exception as e:
    print(f"[gradcam] copy skipped: {e}", flush=True)
