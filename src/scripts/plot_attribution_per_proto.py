"""plot_attribution_per_proto.py -- per (test_sample, prototype) two-panel
GradCAM + IG figure on the class average. One PNG per prototype per
test sample, all using the source-family model.

Output: <source_model>/transfer/<test_sample>/eval/attribution_per_proto/p{c}.png

Run with one or both families, e.g.:
    python plot_attribution_per_proto.py --family NaPHI
    python plot_attribution_per_proto.py --family MgNaPHI
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from data import SAMPLES, LoadPRZ
from dino_sr_contrastive_model import load_contrastive_checkpoint
from viz_gradcam import GradCAM, integrated_gradients, polar_cam_to_cartesian
from viz_paper_attribution import (build_polar_preproc, build_cart_preproc,
                                     _norm01, _log_disp, _beam_mask)

# Defaults for the v2/K6 pipeline. Override via CLI for v5/K8.
DEFAULT_ROOT = os.path.join("runs", "_per_family")
DEFAULT_MODEL_DIR = {
    "NaPHI":   "NaPHI_combined_K6_30ep",
    "MgNaPHI": "MgNaPHI_combined_K6_30ep",
}
FAMILY_TEST = {
    "NaPHI":   ["NaPHI_Nadja_SI003", "NaPHI_Nadja_SI004",
                 "NaPHI_Nadja_SI009", "NaPHI_Nadja_SI010",
                 "NaPHI_Nadja_SI005", "NaPHI_Nadja_SI006",
                 "NaPHI_Nadja_SI007", "NaPHI_Nadja_SI008"],
    "MgNaPHI": ["MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI011",
                 "MgNaPHI_remeas_SI001", "MgNaPHI_remeas_SI003",
                 "MgNaPHI_remeas_SI005", "MgNaPHI_remeas_SI006",
                 "MgNaPHI_remeas_SI010", "MgNaPHI_remeas_SI007",
                 "MgNaPHI_remeas_SI008", "MgNaPHI_remeas_SI009"],
}
SIGMA = 2.0
IG_STEPS = 50
N_TOP_FOR_AVG = 200
H = 192
MASK_R = 15
CENTER_CROP = 140


def _per_proto_two_panel(avg_cart, cam_cart, ig_cart, bm,
                          sample, c, count, out_png):
    """One PNG, two panels: avg+GradCAM (left), avg+IG (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    avg_disp = _log_disp(avg_cart, bm)
    avg_norm = _norm01(avg_disp) * bm
    for ax, attr_cart, label in [
        (axes[0], cam_cart, "GradCAM"),
        (axes[1], ig_cart,  "IG"),
    ]:
        attr_n = _norm01(attr_cart) * bm
        ax.imshow(avg_norm, cmap="gray", aspect="equal", interpolation="nearest")
        ax.imshow(attr_n, cmap="jet", alpha=0.55,
                   aspect="equal", interpolation="nearest")
        ax.set_title(f"{label}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{sample}  -  p{c}  (N={int(count)})", fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _process_one_sample(model, sample, src_dir, device):
    """Compute per-proto class-avg GradCAM + IG, save 2-panel PNGs."""
    transfer_root = os.path.join(src_dir, "transfer", sample)
    eval_dir = os.path.join(transfer_root, "eval")
    inf_path = os.path.join(eval_dir, "inference.npz")
    if not os.path.exists(inf_path):
        print(f"  [skip] {sample}: no inference.npz", flush=True); return
    inf = np.load(inf_path)
    soft_probs = inf["soft_probs"]; assigns = inf["assigns"]
    K = soft_probs.shape[1]
    counts = np.bincount(assigns, minlength=K)

    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=H, vmax=cfg["vmax"])

    polar_pre = build_polar_preproc(polar_size=H, polar_mask_cols=45,
                                      center_crop_size=CENTER_CROP)
    cart_pre = build_cart_preproc(polar_size=H, center_crop_size=CENTER_CROP)

    SCALE = CENTER_CROP / H
    eff_polar_r = (45 / H) * (H / 2.0) * SCALE
    eff_cart_r = MASK_R * SCALE
    bm_r = max(eff_polar_r, eff_cart_r)
    bm = _beam_mask(H, H, bm_r)

    last_mod = list(model.student_encoder.children())[-1]
    for p in model.student_encoder.parameters():
        p.requires_grad_(True)
    for p in model.student_projector.parameters():
        p.requires_grad_(True)
    for p in model.prototypes.parameters():
        p.requires_grad_(True)
    cam_tool = GradCAM(model, last_mod)

    out_dir = os.path.join(eval_dir, "attribution_per_proto")
    os.makedirs(out_dir, exist_ok=True)

    print(f"  {sample}", flush=True)
    for c in range(K):
        idx = np.where(assigns == c)[0]
        if idx.size == 0:
            continue
        scores = soft_probs[idx, c]
        top = idx[np.argsort(-scores)[:min(N_TOP_FOR_AVG, len(idx))]]
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0).astype(np.float32)
        w = soft_probs[top, c].astype(np.float32)
        wavg = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
        wavg_norm = np.clip(wavg / max(float(cfg["vmax"]), 1e-6), 0.0, 1.0)
        x_full = (torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0)
                   .to(device).float())
        x_full = F.interpolate(x_full, size=(H, H), mode="bilinear",
                                align_corners=False)
        x_cart = cart_pre(x_full); x_polar = polar_pre(x_full)
        with torch.enable_grad():
            xp = x_polar.detach().requires_grad_(True)
            cam_p = cam_tool(xp, target_class=c)
            ig_p = integrated_gradients(model, x_polar.detach(),
                                         target_class=c, n_steps=IG_STEPS)
        avg_cart_arr = x_cart[0, 0].detach().cpu().numpy()
        cam_cart_arr = polar_cam_to_cartesian(cam_p).detach().cpu().numpy()
        ig_cart_arr = polar_cam_to_cartesian(ig_p).detach().cpu().numpy()

        cam_cart_arr = gaussian_filter(cam_cart_arr, sigma=SIGMA)
        ig_cart_arr = gaussian_filter(np.abs(ig_cart_arr), sigma=SIGMA)

        out_png = os.path.join(out_dir, f"p{c}.png")
        _per_proto_two_panel(avg_cart_arr, cam_cart_arr, ig_cart_arr, bm,
                              sample, c, counts[c], out_png)
        print(f"    p{c} -> {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(DEFAULT_MODEL_DIR))
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"sweep root, e.g. runs/_per_family_v5  "
                         f"(default {DEFAULT_ROOT})")
    ap.add_argument("--model-dir", default=None,
                    help="model dir name under root, e.g. "
                         "NaPHI_combined_K8_30ep. Default: "
                         "NaPHI_combined_K6_30ep / MgNaPHI_combined_K6_30ep")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = args.model_dir or DEFAULT_MODEL_DIR[args.family]
    src_dir = os.path.join(args.root, model_dir)
    print(f"[attr-per-proto] family={args.family}  src={src_dir}", flush=True)
    ckpt = os.path.join(src_dir, "best.pth")
    model, _, _, _ = load_contrastive_checkpoint(ckpt, device=device)
    model.eval()

    for sample in FAMILY_TEST[args.family]:
        try:
            _process_one_sample(model, sample, src_dir, device)
        except Exception as e:
            print(f"  [FAIL] {sample}: {e!r}", flush=True)
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
