"""Why does DINO send Line-edge pixels to class 3 instead of the Line (1/8)?
For the SAME edge frames, compute GradCAM toward the class-3 logit vs the Line
logit, and read the soft-probs. Hypothesis: the Line decision needs strong
oriented spots (weak in partial-volume edge frames), while class 3 keys on the
ring that IS present -> the model scores class 3 higher."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, gaussian_filter
from data import register_runtime_sample, LoadPRZ, SAMPLES
from dino_sr_contrastive_model import load_contrastive_checkpoint
from viz_gradcam import GradCAM, integrated_gradients, polar_cam_to_cartesian, build_polar_preproc, build_cart_preproc
from viz_paper_attribution import _read_train_cfg
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

F_ = r"docs/explainer/figs"; RUN = r"runs/_gui/Na007b_k60_m097_vmax2"
Ny, Nx = 126, 100; LINE_T = 8           # nearest Line class to class 3 (cos 0.86)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inf = np.load(os.path.join(RUN, "eval", "inference.npz"))
asg = inf["assigns"].astype(int); sp = inf["soft_probs"]; asg2 = asg.reshape(Ny, Nx)
line = np.isin(asg2, [1, 8]); dist = distance_transform_edt(~line).ravel()
key = register_runtime_sample(r"D:/DINOSR/data/Na007b_nbed.cube.npy", scan_shape=(Ny, Nx), vmax=2.0, center_mask_radius=22)
cfg = SAMPLES[key]; vmax = float(cfg["vmax"]); ds = LoadPRZ(cfg["path"], resize=192, vmax=vmax)
tc = _read_train_cfg(RUN)
polar_pre = build_polar_preproc(polar_size=tc["polar_size"], polar_mask_cols=tc["polar_mask_cols"], center_crop_size=tc["center_crop"])
cart_pre = build_cart_preproc(polar_size=tc["polar_size"], center_crop_size=tc["center_crop"])
model, _, _, _ = load_contrastive_checkpoint(os.path.join(RUN, "best.pth"), device=dev)
for p in model.parameters(): p.requires_grad_(True)
model.eval(); cam_tool = GradCAM(model, list(model.student_encoder.children())[-1])

# pick frames: 2 class-3 edge (dist<=1.5, high prob3), 1 class-3 interior (dist>=4), 1 Line
c3 = np.where(asg == 3)[0]
edge = c3[(dist[c3] <= 1.5)]; edge = edge[np.argsort(-sp[edge, 3])[:2]]
int_ = c3[(dist[c3] >= 4)]; int_ = int_[np.argsort(-sp[int_, 3])[:1]]
ln = np.where(asg == LINE_T)[0]; ln = ln[np.argsort(-sp[ln, LINE_T])[:1]]
frames = [("edge", int(i)) for i in edge] + [("interior", int(i)) for i in int_] + [("Line", int(i)) for i in ln]

def prep(i):
    raw = ds.get_raw(i); xn = np.clip(raw / max(vmax, 1e-6), 0, 1)
    xf = F.interpolate(torch.from_numpy(xn)[None, None].to(dev).float(), size=(192, 192), mode="bilinear", align_corners=False)
    return cart_pre(xf), polar_pre(xf)
def cam(xp, c):
    x = xp.detach().requires_grad_(True)
    return gaussian_filter(polar_cam_to_cartesian(cam_tool(x, target_class=c)).detach().cpu().numpy(), 2.0)

rows = []
for tag, i in frames:
    xc, xp = prep(i)
    rows.append((tag, i, xc[0, 0].detach().cpu().numpy(), cam(xp, 3), cam(xp, LINE_T), sp[i, 3], sp[i, 1], sp[i, 8]))
    print(f"{tag} frame {i}: dist={dist[i]:.1f}px  prob3={sp[i,3]:.2f} prob1={sp[i,1]:.2f} prob8={sp[i,8]:.2f}  assigned={asg[i]}", flush=True)

fig = Figure(figsize=(10, 3.0 * len(rows)), facecolor="white")
cols = ["pattern (cart)", "GradCAM -> class 3", f"GradCAM -> Line (c{LINE_T})"]
for r, (tag, i, raw, cm3, cmL, p3, p1, p8) in enumerate(rows):
    for cidx, (img, base) in enumerate([(raw, None), (cm3, raw), (cmL, raw)]):
        ax = fig.add_subplot(len(rows), 3, 3 * r + cidx + 1)
        if base is not None:
            ax.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.imshow(img, cmap="jet", alpha=0.55, interpolation="bilinear")
        else:
            ax.imshow(np.clip(img, 0, 1), cmap="inferno", vmin=0, vmax=0.5, interpolation="nearest")
        if r == 0: ax.set_title(cols[cidx], fontsize=10)
        if cidx == 0: ax.set_ylabel(f"{tag}\np3={p3:.2f} p8={p8:.2f}\ndist={dist[i]:.1f}px", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Na007b: why Line-edge pixels -> class 3, not Line — GradCAM toward class-3 vs Line logit", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig); fig.savefig(f"{F_}/na007b_class3_gradcam.png", dpi=150, facecolor="white")
print("wrote na007b_class3_gradcam.png", flush=True)
