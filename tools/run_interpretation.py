"""run_interpretation.py — run the interpretability battery on a run dir,
headless (the CLI twin of the Analysis ▸ Interpretation GUI tab).

  python tools/run_interpretation.py <run_dir> [--no-ablate] [--no-classical]
                                     [--gradcam]

Reads the run's _train_kwargs.json for the dataset + polar pipeline, registers
the sample so LoadPRZ applies the right preprocessing (vmax / log-stretch),
auto-detects any prior ACOM output, runs whatever is possible, and writes
<run>/_interpretability/report_auto.md + figures.  ACOM-only pieces are
skipped (with a note) when no ACOM run is found.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data import SAMPLES
from gui_app import interpret_core as ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--no-ablate", action="store_true")
    ap.add_argument("--no-classical", action="store_true")
    ap.add_argument("--gradcam", action="store_true")
    a = ap.parse_args()
    RUN = a.run

    tk = json.load(open(os.path.join(RUN, "_train_kwargs.json")))
    sc = tk["_sample_config"]; key = tk["sample"]
    SAMPLES[key] = sc
    inf = np.load(os.path.join(RUN, "eval", "inference.npz"), allow_pickle=True)
    polar = (int(tk["center_mask_radius"]), int(tk["polar_mask_cols"]),
             int(tk["center_crop_size"]), bool(tk["com_centering"]))
    ctx = ic.Ctx(RUN, key, sc["path"], sc.get("vmax", 5.0),
                 tuple(sc["scan_shape"]), inf["embeds"], inf["assigns"], polar)
    acom = ic.find_acom_arrays(RUN)
    print(f"[interp] {key}  K={ctx.K}  N={ctx.assigns.size}  "
          f"ACOM={'found' if acom else 'none'}", flush=True)

    def prog(d, t, s):
        if d % 32 == 0 or d == t:
            print(f"  {s}: {d}/{t}", flush=True)

    want_classical = not a.no_classical
    fac = ic.compute_factors_and_means(ctx, collect_classical=want_classical,
                                       progress=prog)
    probe = ic.probe_and_signatures(ctx, fac, acom=acom)
    ic.figures_class_means(ctx, fac)
    classical = (ic.classical_baselines(ctx, fac, progress=prog)
                 if want_classical else None)

    abl = None
    if not a.no_ablate:
        import torch
        from dino_sr_contrastive_model import load_contrastive_checkpoint
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _, _ = load_contrastive_checkpoint(
            os.path.join(RUN, "best.pth"), device=dev)
        model.eval()
        abl = ic.run_ablations(ctx, model, dev, progress=prog)

    did_g = False
    if a.gradcam:
        try:
            ic.run_gradcam(ctx); did_g = True
        except Exception as e:
            print(f"  gradcam skipped: {e}", flush=True)

    rep = ic.write_report(ctx, probe=probe, ablations=abl, acom=acom,
                          classical=classical, did_gradcam=did_g)
    print(f"[interp] report -> {rep}", flush=True)


if __name__ == "__main__":
    main()
