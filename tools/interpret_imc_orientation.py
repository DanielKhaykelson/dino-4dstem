"""Valid-library ACOM (auto plan) → correlate DINO classes with ACOM
zone axis, to confirm whether orientation drives the clustering.
Uses the user's detection params (threshold 0.1, min_sigma 2) + GPU.
Strided for tractability overnight.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score)

RUN = r"runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60"
CUBE = r"D:/DINOSR/data/IMC_150nm_SI5_nbed.cube.npy"
CIF_DIR = r"D:\DINOSR\data\231228-IMC150nm-0p2apersec-anneal-70c-60min\cifs"
INV_ANG = 0.00493
K_MAX = 0.5
STRIDE = 4
OUT = os.path.join(RUN, "_interpretability")


def main():
    from gui_app.acom_core import (load_crystal, prepare_crystal,
                                    acom_multiphase_full_dataset,
                                    zone_axis_from_matrix)
    detect_kw = dict(min_sigma=2.0, max_sigma=6.0, num_sigma=6,
                     threshold=0.1, log_stretch=True)
    crystals = {}
    for nm, fn in (("alpha", "alpha.cif"), ("gamma", "gamma.cif")):
        cr = load_crystal(os.path.join(CIF_DIR, fn))
        prepare_crystal(cr, k_max=K_MAX, plan_mode="auto", use_cuda=True)
        crystals[nm] = cr
        print(f"[orient] built {nm}: zones={getattr(cr,'orientation_num_zones',None)}",
              flush=True)
    cube = np.load(CUBE, mmap_mode="r")

    def _prog(d, t, s):
        if d % 256 == 0 or s.startswith("match"):
            print(f"  {s} {d}/{t}", flush=True)
    res = acom_multiphase_full_dataset(
        crystals, cube, inv_ang_per_pixel=INV_ANG, detect_kw=detect_kw,
        subsample_stride=STRIDE, min_peaks=4, threshold=0.03, margin=0.0,
        progress_cb=_prog)
    Ny, Nx = res["scan_shape"]
    phase_id = res["phase_id"]; win_rmat = res["winning_rmat"]
    win_corr = res["winning_corr"]
    np.save(os.path.join(OUT, "orient_phase_id.npy"), phase_id)
    np.save(os.path.join(OUT, "orient_winning_corr.npy"), win_corr)

    # zone-axis label per position
    za_lab = np.full(Ny * Nx, -1, int)
    keys = {}
    for rx in range(Ny):
        for ry in range(Nx):
            if phase_id[rx, ry] < 0:
                continue
            R = win_rmat[rx, ry]
            if not np.isfinite(R).all():
                continue
            za, _ = zone_axis_from_matrix(R)
            k = (int(phase_id[rx, ry]), za)
            za_lab[rx * Nx + ry] = keys.setdefault(k, len(keys))

    assigns = np.load(os.path.join(RUN, "eval", "inference.npz"),
                      allow_pickle=True)["assigns"].astype(int)
    m = za_lab >= 0
    n_idx = int(m.sum())
    if n_idx > 50:
        ami = float(adjusted_mutual_info_score(assigns[m], za_lab[m]))
        ari = float(adjusted_rand_score(assigns[m], za_lab[m]))
    else:
        ami = ari = float("nan")
    out = dict(plan="auto", k_max=K_MAX, stride=STRIDE, detect=detect_kw,
               n_indexed=n_idx, n_zone_axes=len(keys),
               DINO_vs_ZA_AMI=round(ami, 4), DINO_vs_ZA_ARI=round(ari, 4))
    json.dump(out, open(os.path.join(OUT, "test_orientation_auto.json"),
                        "w"), indent=2)
    print(f"[orient] indexed={n_idx} zone_axes={len(keys)}  "
          f"DINO~ZA AMI={ami:.3f} ARI={ari:.3f}", flush=True)


if __name__ == "__main__":
    main()
