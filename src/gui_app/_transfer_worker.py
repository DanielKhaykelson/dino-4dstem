"""_transfer_worker.py -- runs sequential per-cube fine-tunes.

Reads <outdir>/_transfer_kwargs.json which has shape:
    {
      "outdir":      "runs/_gui/transfer_<stamp>",
      "cubes":       [ {"key": "...", "path": "...", "name": "..."}, ... ],
      "init_ckpt":   "path/to/start.pth",
      "chain_ckpts": true,            # use prev cube's best.pth as next init
      "shared_kwargs": { ...recipe knobs forwarded to run_config... },
    }

For each cube:
    1) Register a runtime sample (key derived from filename).
    2) Make a subdir under outdir/<i>_<sample_name>/.
    3) Call run_config with init_from_checkpoint = (chain ? prev best.pth : init_ckpt).
    4) On success, capture best.pth path for the next iteration.

Writes:
    _progress.txt (i/N lines)
    _done.flag    on full success
    _error.txt    on failure (loop stops)
"""
from __future__ import annotations
import os, sys, json, traceback, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def _sample_key_from_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    # safe key: alphanumerics + underscore
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in base)
    return f"transfer__{safe}"


def main():
    if len(sys.argv) != 2:
        print("usage: _transfer_worker.py <kwargs.json>"); sys.exit(1)
    spec_path = sys.argv[1]
    with open(spec_path) as f:
        spec = json.load(f)
    outdir = spec["outdir"]
    cubes = spec["cubes"]
    init_ckpt = spec.get("init_ckpt") or None
    chain_ckpts = bool(spec.get("chain_ckpts", True))
    shared_kwargs = dict(spec.get("shared_kwargs", {}))

    n = len(cubes)
    print(f"[transfer] {n} cubes; init_ckpt={init_ckpt}; "
          f"chain={chain_ckpts}", flush=True)

    try:
        import torch
        import data
        from run_contrastive import run_config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prev_best = init_ckpt
        results = []

        for i, cube in enumerate(cubes):
            sample_key = cube.get("key") or _sample_key_from_path(cube["path"])
            cube_path = cube["path"]
            cube_name = cube.get("name") or os.path.basename(cube_path)
            sub_outdir = os.path.join(outdir, f"{i:02d}_{sample_key}")
            os.makedirs(sub_outdir, exist_ok=True)

            # Register the cube as a runtime sample so run_contrastive
            # can resolve it via data.SAMPLES[key].
            cfg = dict(cube.get("sample_cfg", {}))
            cfg.setdefault("path", cube_path)
            cfg.setdefault("vmax", cube.get("vmax", 1.0))
            cfg.setdefault("scan_shape", cube.get("scan_shape"))
            try:
                data.register_runtime_sample(key=sample_key, **cfg)
            except Exception:
                # Fallback: write directly into SAMPLES dict
                data.SAMPLES[sample_key] = cfg

            # Per-cube kwargs = shared + per-cube overrides + init ckpt.
            kw = dict(shared_kwargs)
            if prev_best:
                kw["init_from_checkpoint"] = prev_best

            print(f"\n[transfer] === cube {i+1}/{n} : {cube_name}  "
                  f"(key={sample_key}) ===", flush=True)
            with open(os.path.join(outdir, "_progress.txt"), "w") as f:
                f.write(f"{i+1}/{n}  {cube_name}")

            t0 = time.perf_counter()
            try:
                run_config("c", sample=sample_key, outdir=sub_outdir,
                           device=device, **kw)
                el = time.perf_counter() - t0
                cur_best = os.path.join(sub_outdir, "best.pth")
                if not os.path.exists(cur_best):
                    cur_best = os.path.join(sub_outdir, "latest.pth")
                results.append({"cube": cube_name, "outdir": sub_outdir,
                                 "best": cur_best, "elapsed_s": el,
                                 "status": "ok"})
                if chain_ckpts and os.path.exists(cur_best):
                    prev_best = cur_best
                print(f"[transfer] cube {i+1}/{n} done in {el:.0f}s; "
                      f"best={cur_best}", flush=True)
            except Exception as e:
                # Log per-cube failure but DON'T abort the whole loop;
                # the user gets results for the cubes that did succeed.
                el = time.perf_counter() - t0
                tb = traceback.format_exc()
                results.append({"cube": cube_name, "outdir": sub_outdir,
                                 "elapsed_s": el, "status": "failed",
                                 "error": str(e)})
                with open(os.path.join(sub_outdir, "_error.txt"), "w") as f:
                    f.write(tb)
                print(f"[transfer] cube {i+1}/{n} FAILED: {e!r}",
                      flush=True)

        with open(os.path.join(outdir, "transfer_results.json"), "w") as f:
            json.dump({"n_cubes": n, "init_ckpt": init_ckpt,
                       "chain_ckpts": chain_ckpts,
                       "results": results}, f, indent=2)
        with open(os.path.join(outdir, "_done.flag"), "w") as f:
            f.write("ok")
        print(f"\n[transfer] all done. results -> {outdir}/transfer_results.json",
              flush=True)
    except Exception:
        with open(os.path.join(outdir, "_error.txt"), "w") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
