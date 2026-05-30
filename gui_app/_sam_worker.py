"""_sam_worker.py -- runs SAM mask generation over a class's patterns
in its own process.

Reads JSON kwargs from <outdir>/_sam_kwargs.json.  Outputs (under
<outdir>):
    angle.npy       — per-pattern min-mask-angle (NaN where no mask)
    masks_rle.npz   — compact RLE-encoded masks (object array)
    angle_map.png   — angle map at scan resolution (HSV cmap)
    _done.flag / _error.txt

Spawned from the GUI's SamRunJob so the long SAM pass doesn't block
the UI and can be terminated cleanly.
"""
from __future__ import annotations
import os, sys, json, traceback, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    if len(sys.argv) != 2:
        print("usage: _sam_worker.py <kwargs.json>")
        sys.exit(1)
    kwargs_path = sys.argv[1]
    with open(kwargs_path) as f:
        spec = json.load(f)
    outdir = spec["outdir"]
    cube_path = spec["cube_path"]
    pattern_indices = spec["pattern_indices"]   # list[int] of scan-flat indices
    scan_shape = spec.get("scan_shape")          # [Ny, Nx]
    ckpt_path = spec["sam_checkpoint"]
    model_type = spec.get("sam_model_type", "vit_h")
    device = spec.get("device", "cuda")
    pre_kwargs = spec.get("preprocess", {})
    filter_kwargs = spec.get("filter", {})
    amg_kwargs = spec.get("amg", {})
    save_masks = bool(spec.get("save_masks", True))
    progress_every = int(spec.get("progress_every", 50))

    try:
        import numpy as np
        import torch
        from sam_utils import (SamMaskProcessor, extract_min_angle,
                                masks_to_rle_list)

        # --- load patterns from the cube via a lazy mmap-like loader ---
        # The pre_panel uses _open_lazy(); we duplicate that minimally
        # here to avoid importing tk on the worker.
        if cube_path.endswith(".prz"):
            sig = np.load(cube_path, allow_pickle=True)
            data = sig['data']
            # data shape can be (Ny, Nx, H, W) or (N, H, W)
            if data.ndim == 4:
                Ny, Nx, H, W = data.shape
                data = data.reshape(Ny * Nx, H, W)
            elif data.ndim == 3:
                pass
            else:
                raise ValueError(f"unexpected cube ndim={data.ndim}")
        else:
            arr = np.load(cube_path, mmap_mode="r", allow_pickle=True)
            if arr.ndim == 4:
                Ny, Nx, H, W = arr.shape
                data = arr.reshape(Ny * Nx, H, W)
            elif arr.ndim == 3:
                data = arr
            else:
                raise ValueError(f"unexpected cube ndim={arr.ndim}")

        # Build the per-class image stack (just the requested indices)
        idx = np.asarray(pattern_indices, dtype=np.int64)
        n = int(idx.size)
        print(f"[sam-worker] running SAM on {n} patterns "
              f"(model={model_type}, device={device}); "
              f"output -> {outdir}", flush=True)

        # --- SAM ---
        device_eff = "cuda" if (device == "cuda"
                                 and torch.cuda.is_available()) else "cpu"
        proc = SamMaskProcessor(
            checkpoint_path=ckpt_path,
            model_type=model_type,
            device=device_eff,
            amg_kwargs=amg_kwargs,
        )

        t0 = time.perf_counter()
        all_filtered = []
        for i, scan_flat_idx in enumerate(idx.tolist()):
            try:
                img2d = np.asarray(data[scan_flat_idx], dtype=np.float32)
                _, _, flt = proc.run_one(
                    img2d,
                    blur_sigma=pre_kwargs.get("blur_sigma", 4.0),
                    rescale_lo=pre_kwargs.get("rescale_lo", 0.0),
                    rescale_hi=pre_kwargs.get("rescale_hi", 0.6),
                    downsample=pre_kwargs.get("downsample", 0.5),
                    filter_kwargs=filter_kwargs,
                )
                all_filtered.append(flt)
            except Exception as e:
                print(f"[sam-worker] WARN pattern {scan_flat_idx}: "
                      f"{e!r}", flush=True)
                all_filtered.append([])
            if (i + 1) % progress_every == 0:
                el = time.perf_counter() - t0
                rate = (i + 1) / max(el, 1e-6)
                eta = (n - i - 1) / max(rate, 1e-6)
                print(f"[sam-worker] {i+1}/{n}  "
                      f"({rate:.2f} pat/s, eta {eta/60:.1f} min)",
                      flush=True)
                # Heartbeat file so the GUI poll can show progress.
                with open(os.path.join(outdir, "_progress.txt"), "w") as f:
                    f.write(f"{i+1}/{n}")

        # --- outputs ---
        angles = extract_min_angle(all_filtered)
        np.save(os.path.join(outdir, "angle.npy"), angles)

        # Per-class scan-shape angle map (only if scan_shape provided
        # AND the requested indices densely cover the whole scan)
        if scan_shape and len(scan_shape) == 2 \
                and n == scan_shape[0] * scan_shape[1]:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                Ny, Nx = scan_shape
                ang2d = angles.reshape(Ny, Nx)
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(ang2d, cmap="hsv", interpolation="nearest")
                ax.set_title(
                    f"SAM angle map  ({n} patterns)\n"
                    f"NaN = no streaks detected")
                fig.colorbar(im, ax=ax, label="streak angle (deg, [0,180))")
                fig.tight_layout()
                fig.savefig(os.path.join(outdir, "angle_map.png"), dpi=120)
                plt.close(fig)
            except Exception as e:
                print(f"[sam-worker] angle map render failed: {e!r}",
                      flush=True)

        if save_masks:
            try:
                rle = masks_to_rle_list(all_filtered)
                np.savez_compressed(
                    os.path.join(outdir, "masks_rle.npz"),
                    masks=np.array(rle, dtype=object))
            except Exception as e:
                print(f"[sam-worker] mask RLE save failed: {e!r}",
                      flush=True)

        # Sidecar JSON with the run config (so we can reload tuning).
        with open(os.path.join(outdir, "_sam_run_summary.json"), "w") as f:
            json.dump({
                "n_patterns": n,
                "model_type": model_type,
                "device_effective": device_eff,
                "preprocess": pre_kwargs,
                "filter": filter_kwargs,
                "amg": amg_kwargs,
                "elapsed_s": time.perf_counter() - t0,
                "n_with_masks": int(np.sum(~np.isnan(angles))),
            }, f, indent=2)

        with open(os.path.join(outdir, "_done.flag"), "w") as f:
            f.write("ok")
        print(f"[sam-worker] done in "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    except Exception:
        with open(os.path.join(outdir, "_error.txt"), "w") as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
