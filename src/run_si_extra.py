"""Train an extra IMC region with EXACTLY the SI6 production kwargs (n_layers=1,
fixed config), then evaluate. Reads runs/_gui/IMC_SI6_2_3/_train_kwargs.json as the
template so the configuration is byte-identical except the data path / outdir.
  python src/run_si_extra.py SI2
  python src/run_si_extra.py SI6_eval   # just (re)evaluate an existing run
"""
from __future__ import annotations
import os, sys, json, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import torch
from data import SAMPLES
from run_contrastive import run_config, evaluate_and_report

TEMPLATE = "runs/_gui/IMC_SI6_2_3/_train_kwargs.json"
D = r"D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/EF-4DSTEM"
D50 = r"D:/DINOSR/data/231228-IMC50nm-0p2apersec-anneal-70c-60min"
TARGETS = {
    "SI2": dict(cube=f"{D}/SI-002/Survey_CH2_1_nbed.cube.npy", outdir="runs/_gui/IMC_SI2"),
    "SI6": dict(cube=f"{D}/SI-006/Survey_CH2_1_nbed.cube.npy", outdir="runs/_gui/IMC_SI6_2_3"),
    "SI1": dict(cube=f"{D}/SI-001/Survey_CH2_0_1_nbed.cube.npy", outdir="runs/_gui/IMC_SI1"),
    # 50nm IMC (BF-centered per-dataset via make_nbed_50nm.py, sr=70); same recipe as the 150nm.
    "50nm_SI1": dict(cube=f"{D50}/SI-001/Survey_CH2_1_nbed.cube.npy",   outdir="runs/_gui/IMC50_SI1"),
    "50nm_SI2": dict(cube=f"{D50}/SI-002/Survey_CH2_1_nbed.cube.npy",   outdir="runs/_gui/IMC50_SI2"),
    "50nm_SI3": dict(cube=f"{D50}/SI-003/Survey_CH2_1_nbed.cube.npy",   outdir="runs/_gui/IMC50_SI3"),
    "50nm_SI4": dict(cube=f"{D50}/SI-004/Survey_CH2_0_1_nbed.cube.npy", outdir="runs/_gui/IMC50_SI4"),
    "50nm_SI5": dict(cube=f"{D50}/SI-005/Survey_CH2_1_nbed.cube.npy",   outdir="runs/_gui/IMC50_SI5"),
}
NON_RUNCONFIG = {"sample", "outdir", "_sample_config"}


def main(argv=None):
    argv = argv or sys.argv[1:]
    tgt = argv[0] if argv else "SI2"
    eval_only = tgt.endswith("_eval"); tgt = tgt.replace("_eval", "")
    spec = TARGETS[tgt]; cube = spec["cube"]; outdir = spec["outdir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[si_extra] target={tgt} cube={cube} outdir={outdir} device={device} eval_only={eval_only}", flush=True)

    tk = json.load(open(TEMPLATE))
    sc = dict(tk.get("_sample_config", {})); sc["path"] = cube
    sample = f"IMC_{tgt}_extra"
    SAMPLES[sample] = sc                                   # runtime registration (mirrors GUI)

    kw = {k: v for k, v in tk.items() if k not in NON_RUNCONFIG}
    kw["vmax"] = sc.get("vmax", 5.0)                       # template had null -> use sample-config vmax
    base = cube[:-4] if cube.endswith(".npy") else cube
    rp = cube + ".radial.npy"
    kw["supcon_radials_path"] = rp if os.path.exists(rp) else None
    kw["supcon_thresholds_path"] = None                    # supcon_lambda=0 -> defaults

    if not eval_only and not os.path.exists(os.path.join(outdir, "best.pth")):
        os.makedirs(outdir, exist_ok=True)
        t0 = time.perf_counter()
        print(f"[{datetime.now():%H:%M:%S}] TRAIN {tgt} (n_layers={kw.get('n_layers')}, K={kw.get('num_prototypes')}, ep={kw.get('epochs')})", flush=True)
        run_config("c", sample=sample, outdir=outdir, device=device, **kw)
        print(f"[{datetime.now():%H:%M:%S}] train done ({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        print(f"[si_extra] best.pth exists or eval_only -> skip training", flush=True)

    # ensure eval reads the correct preprocessing (it looks in run_summary.json["cfg"]).
    rs_path = os.path.join(outdir, "run_summary.json")
    if not os.path.exists(rs_path):
        cfg = {k: tk.get(k) for k in ("polar_mask_cols", "center_crop_size", "polar_size",
                                      "center_mask_radius", "com_centering", "com_search_radius_factor")}
        json.dump({"sample": sample, "config": "c", "outdir": outdir, "cfg": cfg},
                  open(rs_path, "w"), indent=2)
        print(f"[si_extra] wrote run_summary.json cfg={cfg}", flush=True)

    if not os.path.exists(os.path.join(outdir, "eval", "metrics.json")):
        t0 = time.perf_counter()
        print(f"[{datetime.now():%H:%M:%S}] EVAL {tgt}", flush=True)
        evaluate_and_report("c", sample=sample, outdir=outdir, device=device)
        print(f"[{datetime.now():%H:%M:%S}] eval done ({time.perf_counter()-t0:.0f}s)", flush=True)
    # Completion marker the watchdog relies on (run_config does not write one).
    if os.path.exists(os.path.join(outdir, "eval", "metrics.json")):
        open(os.path.join(outdir, "_done.flag"), "w").close()
    print(f"[si_extra] DONE {tgt} -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
