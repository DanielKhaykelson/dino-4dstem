"""Compact SI figure: per-prototype learned attention (Grad-CAM) for the three
IMC films. One row per sample (SI3/SI4/SI5); one cell per DINO prototype =
confidence-weighted class-average diffraction (inferno) with the Grad-CAM
attention overlaid (turbo, alpha 0.5). Prototypes are ordered amorphous->
crystalline by the per-class median 2D BRAGG EXCESS (imc_class_order.json from
imc_grain_descriptors.py) -- a rotation-invariant index that counts BOTH discrete
spots and sharp powder rings, so spotty single-crystal and fine-polycrystalline
classes are no longer under-ranked the way the old azimuth-averaged 1D peak/halo
did. Reuses the exact GradCAM / preprocessing of viz_paper_attribution
(training vmax=5, run cfg).

  python tools/imc_grain_descriptors.py   # first: writes imc_class_order.json
  python tools/imc_gradcam_si.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import data
import viz_paper_attribution as vpa
from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline

NB = r"D:\DINOSR\data\231228-IMC150nm-0p2apersec-anneal-70c-60min\EF-4DSTEM"
IMC = {
 "SI3": dict(path=os.path.join(NB, "SI-003", "Survey_CH2_1_nbed.cube.npy"),
             run="runs/_gui/IMC_SI3_m097k60", cmr=15),
 "SI4": dict(path=os.path.join(NB, "SI-004", "Survey_CH2_0_1_nbed.cube.npy"),
             run="runs/_gui/IMC_SI4_m097_k60", cmr=20),
 "SI5": dict(path=r"D:\DINOSR\data\IMC_150nm_SI5_nbed.cube.npy",
             run="runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60", cmr=20),
}
for k, c in IMC.items():
    data.SAMPLES["IMC_" + k] = {"path": c["path"], "vmax": 5, "scan_shape": (128, 128),
                                "center_mask_radius": c["cmr"], "approved_label": None}
OUT = "docs/explainer/figs"; REVIEW = "docs/paper/draft_v2/figs/latest_review"
ATTR_SIGMA = 2.0


def bragg2d(avg, beam):
    """2D Bragg excess on a class-average pattern: integrate intensity ABOVE the
    radially-symmetric halo (SNIP of the azimuthal-mean profile), normalised by
    the halo, over the ring band. Counts BOTH discrete spots and sharp powder
    rings -> rotation-invariant crystalline-order index (fixes the azimuth-
    averaged 1D peak/halo, which under-ranked spotty / sharp-ring classes)."""
    H = avg.shape[-1]; cyx = (H - 1) / 2.0
    beam = max(6, int(round(beam)))
    lo = beam + 1; hi = int(0.48 * H)
    m, _, _ = _radial_mean_var(avg, (cyx, cyx), beam_px=beam)
    seg = m[lo:hi]
    if seg.size < 5 or seg.sum() <= 0:
        return 0.0
    halo = np.exp(_snip_baseline(np.log(np.clip(seg, 1e-6, None))))
    yy, xx = np.indices((H, H)); rr = np.sqrt((yy - cyx) ** 2 + (xx - cyx) ** 2)
    halo_full = np.interp(rr, np.arange(lo, hi), halo, left=halo[0], right=halo[-1])
    band = (rr >= lo) & (rr <= hi)
    return float(np.clip(avg[band] - halo_full[band], 0, None).sum() / (halo_full[band].sum() + 1e-9))


def collect(sample, run_dir, device, ig_steps=50):
    """Return list of (count, bragg, disp_avg, cam, ig) per prototype, ordered
    amorphous->crystalline by class-average 2D Bragg excess. cam = Grad-CAM,
    ig = Integrated Gradients (both polar->cartesian, smoothed, 0-1)."""
    cfg = data.SAMPLES[sample]
    dataset = vpa.LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    inf = np.load(os.path.join(run_dir, "eval", "inference.npz"))
    soft_probs = inf["soft_probs"]; assigns = inf["assigns"]; K = soft_probs.shape[1]
    tcfg = vpa._read_train_cfg(run_dir)
    polar_pre = vpa.build_polar_preproc(polar_size=tcfg["polar_size"],
                                        polar_mask_cols=tcfg["polar_mask_cols"],
                                        center_crop_size=tcfg["center_crop"])
    cart_pre = vpa.build_cart_preproc(polar_size=tcfg["polar_size"],
                                      center_crop_size=tcfg["center_crop"])
    SCALE = tcfg["center_crop"] / tcfg["polar_size"]
    eff_polar_r = (tcfg["polar_mask_cols"] / tcfg["polar_size"]) * (tcfg["polar_size"] / 2.0) * SCALE
    bm_r = max(eff_polar_r, tcfg["mask_r"] * SCALE)
    model, _, _, _ = vpa.load_contrastive_checkpoint(os.path.join(run_dir, "best.pth"), device=device)
    for grp in (model.student_encoder, model.student_projector, model.prototypes):
        for p in grp.parameters():
            p.requires_grad_(True)
    model.eval()
    cam_tool = vpa.GradCAM(model, list(model.student_encoder.children())[-1])
    counts = np.bincount(assigns, minlength=K)
    out = []
    print(f"[si-attr] {sample}: K={K}", flush=True)
    for c in range(K):
        idx = np.where(assigns == c)[0]
        if idx.size == 0:
            continue
        top = idx[np.argsort(-soft_probs[idx, c])[:max(200, 3)]]
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0).astype(np.float32)
        w = soft_probs[top, c].astype(np.float32)
        wavg = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
        wavg_norm = np.clip(wavg / max(float(cfg["vmax"]), 1e-6), 0.0, 1.0)
        x_full = torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0).to(device).float()
        x_full = F.interpolate(x_full, size=(192, 192), mode="bilinear", align_corners=False)
        x_cart = cart_pre(x_full); x_polar = polar_pre(x_full)
        with torch.enable_grad():
            xp = x_polar.detach().requires_grad_(True)
            cam_p = cam_tool(xp, target_class=c)
            ig_p = vpa.integrated_gradients(model, x_polar.detach(),
                                            target_class=c, n_steps=ig_steps)
        avg = x_cart[0, 0].detach().cpu().numpy()
        cam = vpa._gaussian_blur(vpa.polar_cam_to_cartesian(cam_p).detach().cpu().numpy(), ATTR_SIGMA)
        ig = vpa._gaussian_blur(vpa.polar_cam_to_cartesian(ig_p).detach().cpu().numpy(), ATTR_SIGMA)
        H = avg.shape[-1]; beam_disp = bm_r * H / 192.0
        bm = vpa._beam_mask(H, H, beam_disp)
        disp = vpa._log_disp(avg, bm)
        out.append((int(counts[c]), bragg2d(avg, beam_disp), disp,
                    vpa._norm01(cam), vpa._norm01(ig)))
    cam_tool.close()
    out.sort(key=lambda t: t[1])           # amorphous -> crystalline (by 2D Bragg excess)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {n: collect("IMC_" + n, IMC[n]["run"], device) for n in IMC}
    maxK = max(len(v) for v in rows.values())
    import shutil
    os.makedirs(REVIEW, exist_ok=True)
    P = 1.7
    # attribution type (Grad-CAM idx 3 / IG idx 4, cmap) x render mode
    ATTR = [("Grad-CAM", 3, "turbo"), ("IG", 4, "magma")]
    ORDER_TXT = ("Each row = one sample; prototypes ordered amorphous→crystalline "
                 "by class-average 2D Bragg excess $B$ (counts spots AND sharp rings).")
    for aname, aidx, acmap in ATTR:
        tag = "gradcam" if aname == "Grad-CAM" else "ig"
        for mode in ("overlay", "only"):
            if mode == "overlay":
                fname = f"imc_{tag}_si.png"
                sup = (f"Per-prototype learned attention ({aname}, {acmap}) on the class-average diffraction (inferno) — IMC films.\n"
                       f"{ORDER_TXT} Attention concentrates on the halo / first-ring annulus across all classes.")
            else:
                fname = f"imc_{tag}_si_only.png"
                sup = (f"Per-prototype {aname} attribution ONLY ({acmap}, no diffraction underlay) — IMC films.\n"
                       f"{ORDER_TXT} The bright ring is the halo / first-ring annulus the model keys on.")
            fig, axes = plt.subplots(3, maxK, figsize=(maxK * P, 3 * P + 0.6), squeeze=False)
            for ri, n in enumerate(["SI3", "SI4", "SI5"]):
                protos = rows[n]
                for ci in range(maxK):
                    ax = axes[ri][ci]; ax.set_xticks([]); ax.set_yticks([])
                    if ci >= len(protos):
                        ax.set_axis_off(); continue
                    cnt, brg, disp = protos[ci][0], protos[ci][1], protos[ci][2]
                    attr = protos[ci][aidx]
                    if mode == "overlay":
                        ax.imshow(disp, cmap="inferno"); ax.imshow(attr, cmap=acmap, alpha=0.5)
                    else:
                        ax.imshow(attr, cmap=acmap)
                    ax.set_title(f"N={cnt}  B={brg:.2f}", fontsize=7, pad=2)
                    for s in ax.spines.values():
                        s.set_edgecolor("#888"); s.set_linewidth(0.5)
                axes[ri][0].set_ylabel(n, fontsize=13, fontweight="bold", rotation=0,
                                       labelpad=22, va="center")
            fig.suptitle(sup, fontsize=11, y=0.995)
            fig.tight_layout(rect=[0.01, 0, 1, 0.95])
            p = os.path.join(OUT, fname)
            fig.savefig(p, dpi=160, facecolor="white"); plt.close(fig)
            shutil.copy(p, os.path.join(REVIEW, fname))
            print(f"[si-attr] wrote {p} (+ latest_review copy)", flush=True)


if __name__ == "__main__":
    main()
