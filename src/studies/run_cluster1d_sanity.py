"""run_cluster1d_sanity.py -- 3-epoch sanity run to verify the new
cluster1d loss runs end-to-end on Na007b. Writes runs/_cluster1d_sanity/.
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from data import SAMPLES
from run_contrastive import run_config

OUT = os.path.join("runs", "_cluster1d_sanity")
SAMPLE = "Na007b"
EPOCHS = 3
K = 6


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SAMPLES[SAMPLE]
    base = cfg["path"][:-4] if cfg["path"].endswith(".prz") else cfg["path"]
    rad_path = base + ".radial.npy"
    th_path = base + ".gate_thresholds.json"
    os.makedirs(OUT, exist_ok=True)
    print(f"[sanity] device={device}  epochs={EPOCHS}  sample={SAMPLE}", flush=True)
    t0 = time.perf_counter()
    run_config("c", sample=SAMPLE, outdir=OUT, device=device,
               epochs=EPOCHS, seed=42, batch_size=128,
               lr=3e-4, weight_decay=1e-6,
               num_prototypes=K,
               t0=0.04, tfin=0.07,
               warmup_epochs=2, ramp_epochs=1,
               entropy_gate=False,
               projection_dim=128, projection_hidden=256,
               theta_shift_range=None,
               theta_shift_range_student=192, theta_shift_range_teacher=16,
               center_mask_radius=15,
               center_crop_size=140,
               vmax=None,
               polar_size=192, polar_mask_cols=45,
               pipeline="polar",
               centroid_lambda=0.0, centroid_margin=0.3,
               conf_weight_gamma=0.0,
               entropy_gate_override=None,
               lam_spatial=0.0,
               architecture="resnet", n_layers=1,
               w_ent=0.0,
               com_centering=True,
               com_search_radius_factor=2.0,
               aug_disable=["hflip", "vflip", "colorjitter"],
               supcon_radials_path=rad_path,
               supcon_thresholds_path=th_path,
               supcon_lambda=0.0,            # off -- testing cluster1d alone
               supcon_temperature=0.3,
               contrastive_lambda_override=0.0,
               proto_repel_lambda=0.0,
               cluster1d_lambda=0.3,
               cluster1d_margin=0.4,
               cluster1d_warmup_frac=0.0,
               cluster1d_ramp_frac=0.0,
               )
    print(f"[sanity] done in {time.perf_counter()-t0:.0f}s", flush=True)
    # Print the CSV
    import csv
    with open(os.path.join(OUT, "training_log.csv")) as f:
        rows = list(csv.DictReader(f))
    print("\nTraining log:")
    for r in rows:
        print(f"  ep={r['epoch']}  L={r['avg_loss']}  Ldino={r['avg_loss_dino']}  "
              f"L1d_intra={r['avg_loss_cluster1d_intra']}  "
              f"L1d_inter={r['avg_loss_cluster1d_inter']}  "
              f"c1d_mean_off={r['cluster1d_mean_off']}  "
              f"c1d_max_off={r['cluster1d_max_off']}  "
              f"effK={r['effK']}  K_act={r['active_classes']}  "
              f"avg_conf={r['avg_conf']}")


if __name__ == "__main__":
    main()
