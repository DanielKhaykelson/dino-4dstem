"""_train_worker.py -- runs run_contrastive.run_config in its own process.

Reads JSON kwargs from <outdir>/_train_kwargs.json. The GUI's
TrainingJob spawns this as a subprocess so it can be killed cleanly
with terminate().
"""
from __future__ import annotations
import os, sys, json, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    if len(sys.argv) != 2:
        print("usage: _train_worker.py <kwargs.json>")
        sys.exit(1)
    kwargs_path = sys.argv[1]
    with open(kwargs_path) as f:
        spec = json.load(f)
    sample = spec.pop("sample")
    outdir = spec.pop("outdir")
    # Sample config injected by the GUI for runtime-registered samples
    # (loaded ad-hoc on the Pre-processing tab and not present in the
    # static data.SAMPLES dict that this subprocess imports fresh).
    sample_cfg = spec.pop("_sample_config", None)
    if sample_cfg is not None:
        try:
            import data
            data.SAMPLES[sample] = dict(sample_cfg)
        except Exception:
            # if we can't register, run_contrastive will raise KeyError —
            # but we want a clean message.
            pass
    try:
        import torch
        from run_contrastive import run_config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_config("c", sample=sample, outdir=outdir, device=device, **spec)
        with open(os.path.join(outdir, "_done.flag"), "w") as f:
            f.write("ok")
    except Exception:
        with open(os.path.join(outdir, "_error.txt"), "w") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
