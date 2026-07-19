"""IG difference map: for Line-edge frames, integrated-gradients toward the
class-3 logit vs the Line logit, and their DIFFERENCE (IG3 - IG_Line) to localize
the reflection that tips the decision to class 3."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, gaussian_filter
from data import register_runtime_sample, LoadPRZ, SAMPLES
from dino_sr_contrastive_model import load_contrastive_checkpoint
from viz_gradcam import integrated_gradients, polar_cam_to_cartesian, build_polar_preproc, build_cart_preproc, resolve_prototype_ids, dense_target
from viz_paper_attribution import _read_train_cfg
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
F_ = r"docs/explainer/figs"; RUN = r"runs/_gui/Na007b_k60_m097_vmax2"; Ny, Nx = 126, 100; LINE_T = 8
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inf = np.load(os.path.join(RUN, "eval", "inference.npz")); asg = inf["assigns"].astype(int); sp = inf["soft_probs"]
line = np.isin(asg.reshape(Ny, Nx), [1, 8]); dist = distance_transform_edt(~line).ravel()
key = register_runtime_sample(r"D:/DINOSR/data/Na007b_nbed.cube.npy", scan_shape=(Ny, Nx), vmax=2.0, center_mask_radius=22)
cfg = SAMPLES[key]; vmax = float(cfg["vmax"]); ds = LoadPRZ(cfg["path"], resize=192, vmax=vmax); tc = _read_train_cfg(RUN)
polar_pre = build_polar_preproc(polar_size=tc["polar_size"], polar_mask_cols=tc["polar_mask_cols"], center_crop_size=tc["center_crop"])
cart_pre = build_cart_preproc(polar_size=tc["polar_size"], center_crop_size=tc["center_crop"])
model, _, _, _ = load_contrastive_checkpoint(os.path.join(RUN, "best.pth"), device=dev)
for p in model.parameters(): p.requires_grad_(True)
model.eval()
orig_ids = resolve_prototype_ids(RUN, model, ds, dev, polar_pre=polar_pre)  # dense id -> real prototype
c3 = np.where(asg == 3)[0]; edge = c3[dist[c3] <= 1.5]; edge = edge[np.argsort(-sp[edge, 3])[:2]]
ln = np.where(asg == LINE_T)[0]; ln = ln[np.argsort(-sp[ln, LINE_T])[:1]]
frames = [("edge", int(i)) for i in edge] + [("Line", int(i)) for i in ln]
def prep(i):
    xn = np.clip(ds.get_raw(i) / max(vmax, 1e-6), 0, 1)
    xf = F.interpolate(torch.from_numpy(xn)[None, None].to(dev).float(), size=(192, 192), mode="bilinear", align_corners=False)
    return cart_pre(xf), polar_pre(xf)
def ig(xp, c): return gaussian_filter(polar_cam_to_cartesian(integrated_gradients(model, xp.detach(), target_class=c, n_steps=50)).detach().cpu().numpy(), 2.0)
rows = []
for tag, i in frames:
    xc, xp = prep(i); g3, gL = ig(xp, dense_target(orig_ids, 3)), ig(xp, dense_target(orig_ids, LINE_T)); rows.append((tag, i, xc[0, 0].detach().cpu().numpy(), g3, gL, g3 - gL))
    print(f"{tag} {i}: prob3={sp[i,3]:.2f} prob8={sp[i,8]:.2f} dist={dist[i]:.1f}", flush=True)
fig = Figure(figsize=(13, 3.0 * len(rows)), facecolor="white")
cols = ["pattern", "IG -> class 3", f"IG -> Line(c{LINE_T})", "IG3 - IG_Line (red=class-3 cue)"]
for r, (tag, i, raw, g3, gL, d) in enumerate(rows):
    for c, (img, base, cm, kw) in enumerate([(raw, None, "inferno", dict(vmin=0, vmax=0.5)), (g3, raw, "jet", {}), (gL, raw, "jet", {}), (d, raw, "RdBu_r", {})]):
        ax = fig.add_subplot(len(rows), 4, 4 * r + c + 1)
        if base is not None:
            ax.imshow(base, cmap="gray", vmin=0, vmax=1)
            v = np.percentile(np.abs(img), 99) if c == 3 else None
            ax.imshow(img, cmap=cm, alpha=0.55, **(dict(vmin=-v, vmax=v) if v else {}))
        else: ax.imshow(np.clip(img, 0, 1), cmap=cm, **kw)
        if r == 0: ax.set_title(cols[c], fontsize=9)
        if c == 0: ax.set_ylabel(f"{tag}\np3={sp[i,3]:.2f} p8={sp[i,8]:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Na007b: IG difference (class-3 minus Line) on Line-edge frames", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig); fig.savefig(f"{F_}/na007b_class3_ig_diff.png", dpi=150, facecolor="white")
print("wrote na007b_class3_ig_diff.png", flush=True)
