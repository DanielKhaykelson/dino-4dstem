"""Multi-phase ACOM for EuInAs_B100 (EuInAs compound + In + As precipitates),
then fold the orientation/phase cross-check into its interpretation report.
Calibration 0.02184 A^-1/px derived from the .prz device.calib metadata
(2.739e-4 rad x binning 2, 200 kV lambda=0.02508 A)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import SAMPLES, open_lazy_cube
from gui_app import interpret_core as ic
from gui_app.acom_core import (load_crystal, prepare_crystal,
                               acom_multiphase_full_dataset)

RUN = r"runs/_sweep_m_K_20260525_213539/EuInAs_B100/stage2/m0.9700_seed42_K60"
CIFS = {
    "EuInAs": r"D:/DINOSR/data/EuInAs_cifs/1618106.cif",
    "In":     r"D:/DINOSR/data/EuInAs_cifs/mp-85_In.cif",
    "As":     r"D:/DINOSR/data/EuInAs_cifs/mp-11_As.cif",
}
INV_ANG = 0.02184          # A^-1 per raw (256) pixel, from .prz metadata
KMAX, CORR, MINSIG, THR, STRIDE, MINPK = 1.5, 0.1, 2.0, 0.1, 2, 4

print(f"[euacom] building {len(CIFS)} phases (kmax={KMAX}, auto, GPU) …",
      flush=True)
crystals = {}
for nm, cif in CIFS.items():
    cr = load_crystal(cif)
    prepare_crystal(cr, k_max=KMAX, plan_mode="auto", use_cuda=True)
    crystals[nm] = cr
    print(f"  built {nm}", flush=True)

sc = SAMPLES["EuInAs_B100"]
cube = open_lazy_cube(sc["path"], scan_shape=tuple(sc["scan_shape"]))
detect_kw = dict(min_sigma=MINSIG, max_sigma=6.0, num_sigma=6,
                 threshold=THR, log_stretch=True)

def prog(d, t, s):
    if s in ("match", "build_vectors") or d % 1024 == 0:
        print(f"    {s}: {d}/{t}", flush=True)

t0 = time.time()
res = acom_multiphase_full_dataset(
    crystals, cube, inv_ang_per_pixel=INV_ANG, detect_kw=detect_kw,
    subsample_stride=STRIDE, min_peaks=MINPK, threshold=CORR, margin=0.0,
    progress_cb=prog)
pid = res["phase_id"]; n_idx = int((pid >= 0).sum())
print(f"[euacom] done in {time.time()-t0:.0f}s  indexed={n_idx}/{pid.size}",
      flush=True)
names = res.get("phase_names", list(CIFS))
for i, nm in enumerate(names):
    print(f"    {nm}: {(pid == i).sum()} px", flush=True)

base = os.path.join(RUN, "acom", "maps"); os.makedirs(base, exist_ok=True)
np.save(os.path.join(base, "mpfull_phase_id.npy"), pid)
np.save(os.path.join(base, "mpfull_winning_corr.npy"), res["winning_corr"])
np.save(os.path.join(base, "mpfull_winning_rmat.npy"), res["winning_rmat"])
json.dump(dict(phases=names, k_max=KMAX, corr_threshold=CORR,
               min_sigma=MINSIG, det_threshold=THR, stride=STRIDE,
               min_peaks=MINPK, inv_ang_per_pixel=INV_ANG, plan="auto",
               indexed=n_idx),
          open(os.path.join(RUN, "acom", "acom_config_interp.json"), "w"),
          indent=2)

# ---- augment interpretation report with ACOM ----
rs = json.load(open(os.path.join(RUN, "run_summary.json")))
cfg = rs["cfg"]
polar = (int(cfg.get("center_mask_radius", 0)),
         int(cfg.get("polar_mask_cols", 0)),
         int(cfg.get("center_crop_size", 140)),
         bool(cfg.get("com_centering", False)))
inf = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)
ctx = ic.Ctx(RUN, "EuInAs_B100", sc["path"], sc.get("vmax", 30),
             tuple(sc["scan_shape"]), inf["embeds"], inf["assigns"], polar)
acom = ic.find_acom_arrays(RUN)
fac = ic.compute_factors_and_means(ctx, collect_classical=False)
probe = ic.probe_and_signatures(ctx, fac, acom=acom)

def _load(fn):
    p = os.path.join(ctx.out, fn)
    return json.load(open(p)) if os.path.exists(p) else None

rep = ic.write_report(ctx, probe=probe, ablations=_load("test2_4_ablations.json"),
                      acom=acom, classical=_load("classical_baselines.json"),
                      did_gradcam=False)
cat = probe.get("categorical", {})
corr = next((r for r in probe["rows"] if r["factor"] == "ACOM correlation"), {})
print(f"[euacom] ACOM corr probe_R2={corr.get('probe_R2')} eta2={corr.get('eta2')}",
      flush=True)
print(f"[euacom] phase={cat.get('phase')}  zone_axis={cat.get('zone_axis')}",
      flush=True)
print(f"[euacom] report -> {rep}", flush=True)
print("[euacom] DONE", flush=True)
