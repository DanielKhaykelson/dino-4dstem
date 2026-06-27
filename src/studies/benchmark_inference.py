"""
benchmark_inference.py — measure DINO4DSTEM inference throughput for the
in-situ-use claim in the paper. Runs the full eval pipeline (polar
transform -> CNN trunk -> projection -> prototype softmax) on the GPU and
reports patterns-per-second end-to-end.

Writes `runs/inference_benchmark.json`:
    {
      "gpu": "...",
      "architecture": "L1 ResNet-18",
      "input_shape": [1, 1, 192, 192],
      "patterns_per_second": ...,
      "ms_per_pattern": ...,
      "batch_size_used": ...,
      "warmup_iter": 10,
      "timed_iter": 100
    }
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import numpy as np

from dino_sr_contrastive_model import load_contrastive_checkpoint
from contrastive_eval import infer_scan
from data import SAMPLES, LoadPRZ


def benchmark(ckpt_path: str, sample: str, batch_size: int = 128,
                warmup: int = 10, timed: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device}", flush=True)
    gpu_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"

    model, eval_temp, _, _ = load_contrastive_checkpoint(ckpt_path, device=device)
    model.eval()
    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])

    # Preload a batch to memory (skip disk I/O in the measurement loop).
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)
    it = iter(loader)
    batch = next(it)
    x_raw = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device).float()

    # Preprocessing must be part of the timing to reflect real in-situ usage.
    from torchvision.transforms import v2 as T
    from torchvision.transforms import InterpolationMode
    from dino_sr_contrastive_model import PolarTransform, PolarMaskLeft
    preproc = T.Compose([
        T.CenterCrop(140),
        T.Resize(192, interpolation=InterpolationMode.BILINEAR, antialias=True),
        PolarTransform(output_size=192),
        PolarMaskLeft(k_cols=30),
    ])

    @torch.no_grad()
    def one_step():
        x = preproc(x_raw)
        feat = model.teacher_encoder(x)
        proj = model.teacher_projector(feat)
        logits = model.prototypes(proj)
        probs = torch.softmax((logits - model.center) / eval_temp, dim=-1)
        assigns = probs.argmax(dim=-1)
        return assigns

    # Warmup.
    for _ in range(warmup):
        _ = one_step()
    torch.cuda.synchronize() if device.type == "cuda" else None

    t0 = time.perf_counter()
    for _ in range(timed):
        _ = one_step()
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t0

    total_patterns = timed * batch_size
    pps = total_patterns / elapsed
    ms_per = 1000.0 / pps

    result = {
        "gpu": gpu_name,
        "architecture": f"L{model.n_layers} ResNet-18 + polar + circular-theta",
        "input_shape": [1, 1, 192, 192],
        "patterns_per_second": float(pps),
        "ms_per_pattern": float(ms_per),
        "elapsed_total_s": float(elapsed),
        "batch_size": batch_size,
        "warmup_iter": warmup,
        "timed_iter": timed,
        "total_patterns_timed": int(total_patterns),
    }
    print(f"[bench] {pps:,.0f} patterns/s   ({ms_per:.2f} ms/pattern)  "
          f"batch={batch_size} on {gpu_name}", flush=True)
    # Also: full-scan latency.
    scan_n = len(dataset)
    full_scan_s = scan_n / pps
    result["example_full_scan_patterns"] = int(scan_n)
    result["example_full_scan_seconds"] = float(full_scan_s)
    print(f"[bench] full scan ({scan_n} patterns): {full_scan_s:.2f} s", flush=True)

    out = os.path.join("runs", "inference_benchmark.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bench] wrote {out}", flush=True)
    return result


if __name__ == "__main__":
    # Use the current Na007b winner.
    benchmark(
        ckpt_path="runs/Na007b/sweep_polar_centroid/best.pth",
        sample="Na007b",
        batch_size=128,
        warmup=10, timed=50,
    )
