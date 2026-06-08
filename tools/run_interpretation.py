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


def _make_inference(run, sc, polar, out_path):
    """Generate eval/inference.npz when a finished run lacks it (no auto-eval).
    Mirrors the Post-hoc tab's infer_scan call."""
    import torch
    from data import LoadPRZ
    from dino_sr_contrastive_model import load_contrastive_checkpoint
    from contrastive_eval import infer_scan
    mask_r, mask_cols, ccrop, com = polar
    ckpt = os.path.join(run, "best.pth")
    if not os.path.exists(ckpt):
        raise RuntimeError(f"no best.pth in {run}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = load_contrastive_checkpoint(ckpt, device=dev)
    model.eval()
    ds = LoadPRZ(sc["path"], resize=192, vmax=sc.get("vmax", 5.0))
    print(f"[interp] no inference.npz — running infer_scan ({len(ds)} patterns) …",
          flush=True)
    inf = infer_scan(model, ds, dev, dense_remap=True, polar_size=192,
                     polar_mask_cols=mask_cols, center_crop_size=ccrop,
                     com_centering=com, center_mask_radius=mask_r,
                     eval_temp=0.06, batch_size=128)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, soft_probs=inf["soft_probs"], assigns=inf["assigns"],
             embeds=inf["embeds"])
    print(f"[interp] wrote {out_path}", flush=True)
    return dict(embeds=inf["embeds"], assigns=inf["assigns"],
                soft_probs=inf["soft_probs"])


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
    polar = (int(tk["center_mask_radius"]), int(tk["polar_mask_cols"]),
             int(tk["center_crop_size"]), bool(tk["com_centering"]))
    inf_path = os.path.join(RUN, "eval", "inference.npz")
    if not os.path.exists(inf_path):
        inf = _make_inference(RUN, sc, polar, inf_path)
    else:
        inf = np.load(inf_path, allow_pickle=True)
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
