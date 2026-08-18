"""
ws2_utils.py — shared helpers for the DINO-4DSTEM tutorial notebooks.
================================================================================

These are thin, well-documented wrappers around the real library functions in
``src/`` so the notebooks stay readable.  Nothing here is GUI-specific except
:func:`make_phantom`, which drives the (validated) synthetic engine headlessly.

Import it from a notebook with::

    import ws2_utils as wu
    wu.add_src_to_path()

Everything assumes the repo layout ``<repo>/notebooks/`` next to ``<repo>/src/``.
"""
from __future__ import annotations
import os
import sys
import glob
import json
import numpy as np


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
def repo_root() -> str:
    """Absolute path to the repository root (the folder that holds ``src/``)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)          # notebooks/ -> repo root


def add_src_to_path() -> str:
    """Put ``<repo>/src`` on ``sys.path`` so ``import data`` etc. work. Also
    sets a UTF-8 stdout on Windows.  Returns the src path."""
    src = os.path.join(repo_root(), "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    return src


def preimport():
    """Import the heavy analysis modules up front.  Several of them switch
    matplotlib's backend to Agg *on import*, which would break inline plotting
    if it happened mid-notebook.  Call this, THEN ``%matplotlib inline``."""
    add_src_to_path()
    import importlib
    for m in ("data", "contrastive_eval", "dino_sr_contrastive_model"):
        try: importlib.import_module(m)
        except Exception: pass


def find_ws2_example() -> str | None:
    """Newest pre-built WS2 example directory under ``runs/synth/``, or None."""
    pat = os.path.join(repo_root(), "runs", "synth", "WS2_example_*")
    dirs = [d for d in glob.glob(pat) if os.path.isdir(d)
            and os.path.exists(os.path.join(d, "phantom.cube.npy"))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def find_ws2_checkpoint() -> str | None:
    """A trained WS2 checkpoint directory (has best.pth + eval/inference.npz)."""
    for name in ("synth_finite2", "synth_finite1", "synth_5"):
        d = os.path.join(repo_root(), "runs", "_gui", name)
        if os.path.exists(os.path.join(d, "best.pth")):
            return d
    hits = glob.glob(os.path.join(repo_root(), "runs", "_gui", "*", "best.pth"))
    return os.path.dirname(hits[0]) if hits else None


# --------------------------------------------------------------------------
# synthetic phantom generation (headless — reuses the validated GUI engine)
# --------------------------------------------------------------------------
# The WS2 "example" preset, exactly as the GUI's one-click button sets it.
WS2_EXAMPLE = dict(
    zone_axes=[(0, 0, 1), (1, 1, 0), (1, 0, 0)],   # 3 domains = 3 orientations
    crystallinity=["crystalline", "crystalline", "crystalline"],
    tile_xyz=(14, 14, 6), scan=(96, 96), scan_step_A=3.0,
    beam_kV=200.0, conv_mrad=1.5, det_size=256, det_max_mrad=28.0,
    dose_e_A2=1e4, engine="finite", n_voronoi_seeds=64, pot_gpts=512,
    vacuum_pad_A=6.0, rng_seed=0,
)

# A fast, low-fidelity version for demos / CI (seconds, not ~40 min).
WS2_QUICK = dict(
    zone_axes=[(0, 0, 1), (1, 1, 0)], crystallinity=["crystalline", "crystalline"],
    tile_xyz=(6, 6, 3), scan=(16, 16), scan_step_A=3.0,
    beam_kV=200.0, conv_mrad=1.5, det_size=96, det_max_mrad=28.0,
    dose_e_A2=1e4, engine="finite", n_voronoi_seeds=8, pot_gpts=128,
    vacuum_pad_A=6.0, rng_seed=0,
)


def make_phantom(out_dir: str, spec: dict | None = None, save: bool = True):
    """Generate a WS2 phantom headlessly and (optionally) save it, returning
    ``(cube, classmaps, meta, out_dir)``.

    ``spec`` is a dict like :data:`WS2_QUICK` (default) or :data:`WS2_EXAMPLE`.
    This drives the SAME simulation code the GUI uses, with no window shown.
    NOTE: it needs ``customtkinter`` + ``abtem`` + ``ase`` installed and a
    local display server (Tk) — fine on a desktop; on a headless box use the
    pre-built example via :func:`find_ws2_example` instead.
    """
    add_src_to_path()
    import copy
    import matplotlib
    _prev_backend = matplotlib.get_backend()      # importing the panel below
    import customtkinter as ctk                    # flips the mpl backend;
    from gui_app.synth_panel import (SynthPanel,   # we restore it afterwards.
                                     PREDEFINED_CRYSTALS, _Structure)
    sp = spec or WS2_QUICK
    os.makedirs(out_dir, exist_ok=True)

    root = ctk.CTk(); root.withdraw()
    try:
        p = SynthPanel(root, app=None)
        ws2 = PREDEFINED_CRYSTALS["WS2 (2H layered)"]
        p._structures = [
            _Structure(kind="ase", ase_builder=copy.deepcopy(ws2),
                       label=f"WS2 {za}", zone_axes=[za],
                       tile_xyz=tuple(sp["tile_xyz"]),
                       in_plane_range_deg=(0.0, 360.0), crystallinity=cry)
            for za, cry in zip(sp["zone_axes"], sp["crystallinity"])]
        ny, nx = sp["scan"]
        p.scan_ny.set(ny); p.scan_nx.set(nx)
        p.scan_step_A.set(sp["scan_step_A"]); p.beam_kV.set(sp["beam_kV"])
        p.conv_mrad.set(sp["conv_mrad"]); p.det_size.set(sp["det_size"])
        p.det_max_mrad.set(sp["det_max_mrad"]); p.dose_e_A2.set(sp["dose_e_A2"])
        p.engine.set(sp["engine"]); p.layout.set("voronoi")
        p.n_voronoi_seeds.set(sp["n_voronoi_seeds"]); p.pot_gpts.set(sp["pot_gpts"])
        p.rng_seed.set(sp["rng_seed"]); p.vacuum_pad_A.set(sp["vacuum_pad_A"])
        p.out_basename.set("WS2_phantom")
        try: p._sim_stop_requested = False
        except Exception: pass
        cube, classmaps, meta = p._run_simulation(out_dir)
    finally:
        # Drain the panel's queued Tk `after(...)` status callbacks so none
        # dangle past destroy() (avoids "invalid command name" noise), then
        # restore the notebook's matplotlib backend (e.g. inline).
        for _ in range(3):
            try: root.update()
            except Exception: break
        try: root.destroy()
        except Exception: pass
        try: matplotlib.use(_prev_backend, force=True)
        except Exception: pass

    if save:
        np.save(os.path.join(out_dir, "phantom.cube.npy"), cube.astype(np.float32))
        for k, m in classmaps.items():
            np.save(os.path.join(out_dir, f"phantom.classmap_{k}.npy"),
                    m.astype(np.int32))
        with open(os.path.join(out_dir, "phantom.sim_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=lambda o: repr(o))
    return cube, classmaps, meta, out_dir


# --------------------------------------------------------------------------
# virtual images + ROI mean diffraction
# --------------------------------------------------------------------------
def radial_mask(H, W, r_in, r_out):
    """Boolean annulus mask (r_in <= r < r_out) about the frame centre."""
    yy, xx = np.ogrid[:H, :W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (rr >= r_in) & (rr < r_out)


def virtual_bf_haadf(cube):
    """Virtual bright-field and HAADF images from a 4-D cube ``(Ny,Nx,H,W)``.

    BF = counts inside the central disk (r <= 0.06·H); HAADF = counts in the
    0.18–0.45·H annulus.  Returns ``(bf, haadf)`` each ``(Ny, Nx)`` float32.
    """
    Ny, Nx, H, W = cube.shape
    bm = radial_mask(H, W, 0.0, 0.06 * H).ravel()
    ham = radial_mask(H, W, 0.18 * H, 0.45 * H).ravel()
    bf = np.zeros((Ny, Nx), np.float32); ha = np.zeros((Ny, Nx), np.float32)
    for y in range(Ny):
        flat = np.asarray(cube[y]).astype(np.float32).reshape(Nx, H * W)
        bf[y] = flat[:, bm].sum(1)
        ha[y] = flat[:, ham].sum(1)
    return bf, ha


def roi_mean_diffraction(cube, y0, y1, x0, x1):
    """Mean diffraction pattern over a rectangular scan ROI → ``(H, W)``."""
    blk = np.asarray(cube[int(y0):int(y1), int(x0):int(x1)], dtype=np.float32)
    return blk.reshape(-1, blk.shape[-2], blk.shape[-1]).mean(0)


def logshow(ax, img, cmap="inferno", p=99.7, gain=60):
    """Log-stretched imshow of a diffraction pattern (reveals rings + spots)."""
    x = np.asarray(img, np.float32)
    o = x[x > 0]
    vmax = np.percentile(o, p) if o.size else 1.0
    ax.imshow(np.log1p(np.clip(x, 0, vmax) / max(vmax, 1e-6) * gain), cmap=cmap,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------------------
# interactive ROI (live-updating mean diffraction) — needs %matplotlib widget
# --------------------------------------------------------------------------
def interactive_roi(cube, virtual_img=None, roi=(8, 8), cmap="inferno"):
    """A live ROI: drag a rectangle on the virtual image → the mean
    diffraction of that ROI updates on the right.  Requires the ``ipympl``
    backend (``%matplotlib widget``).  Returns the matplotlib figure.

    (hyperspy offers a similar ``RectangularROI().interactive`` widget; we use
    ipympl here since it's already installed and self-contained.)
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    Ny, Nx, H, W = cube.shape
    if virtual_img is None:
        virtual_img, _ = virtual_bf_haadf(cube)
    n, m = roi
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4.2))
    axL.imshow(virtual_img, cmap="gray"); axL.set_title("virtual image — drag a box")
    axL.set_xticks([]); axL.set_yticks([])
    dp = roi_mean_diffraction(cube, 0, n, 0, m)
    imR = axR.imshow(np.log1p(dp), cmap=cmap, interpolation="nearest")
    axR.set_title("mean diffraction of ROI"); axR.set_xticks([]); axR.set_yticks([])

    def _on_select(eclick, erelease):
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        y0 = int(max(0, y0)); y1 = int(min(Ny, max(y1, y0 + 1)))
        x0 = int(max(0, x0)); x1 = int(min(Nx, max(x1, x0 + 1)))
        d = roi_mean_diffraction(cube, y0, y1, x0, x1)
        o = d[d > 0]; vmax = np.percentile(o, 99.7) if o.size else 1.0
        imR.set_data(np.log1p(np.clip(d, 0, vmax) / max(vmax, 1e-6) * 60))
        imR.set_clim(0, np.log1p(60))
        axR.set_title(f"mean diffraction — ROI [{y0}:{y1}, {x0}:{x1}]")
        fig.canvas.draw_idle()

    fig._roi_selector = RectangleSelector(
        axL, _on_select, useblit=True, interactive=True,
        button=[1], minspanx=1, minspany=1, spancoords="data")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# metrics vs ground truth
# --------------------------------------------------------------------------
def align_and_score(gt2d, pred2d):
    """Hungarian-align a predicted class map to a ground-truth map and score.

    Returns a dict with: ``accuracy``, ``ARI``, ``NMI``, ``mean_IoU``,
    ``per_class`` (list of {gt, N, IoU, dice, precision, recall, f1}),
    ``confusion`` (gt × pred, aligned), and ``pred_aligned`` (2-D, relabelled
    to match GT ids).  Extra predicted classes beyond K_gt map to -1.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    gt = np.asarray(gt2d).ravel().astype(int)
    pr = np.asarray(pred2d).ravel().astype(int)
    Kg = int(gt.max()) + 1
    Kp = int(pr.max()) + 1
    C = np.zeros((Kg, Kp), dtype=np.int64)          # confusion gt × pred
    np.add.at(C, (gt, pr), 1)
    # Hungarian on the padded cost so every gt gets a (best) pred column.
    n = max(Kg, Kp)
    cost = np.zeros((n, n)); cost[:Kg, :Kp] = C.max() - C
    rows, cols = linear_sum_assignment(cost)
    pred2gt = {int(c): int(r) for r, c in zip(rows, cols) if c < Kp and r < Kg}
    aligned = np.array([pred2gt.get(int(v), -1) for v in pr])
    acc = float((aligned == gt).mean())
    Ca = C[:, [next((c for c, r in pred2gt.items() if r == g), -1) for g in range(Kg)]] \
        if False else None                          # (kept explicit below)
    # aligned confusion gt × gt-id
    conf = np.zeros((Kg, Kg + 1), dtype=np.int64)   # last col = unassigned (-1)
    a2 = np.where(aligned < 0, Kg, aligned)
    np.add.at(conf, (gt, a2), 1)
    per = []
    ious = []
    for g in range(Kg):
        tp = int(conf[g, g])
        fp = int((aligned == g).sum() - tp)
        fn = int((gt == g).sum() - tp)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per.append(dict(gt=g, N=int((gt == g).sum()), IoU=iou, dice=dice,
                        precision=prec, recall=rec, f1=f1))
        ious.append(iou)
    return dict(accuracy=acc,
                ARI=float(adjusted_rand_score(gt, pr)),
                NMI=float(normalized_mutual_info_score(gt, pr)),
                mean_IoU=float(np.mean(ious)),
                per_class=per, confusion=conf,
                pred_aligned=aligned.reshape(np.asarray(gt2d).shape),
                K_gt=Kg, K_pred=Kp)


def class_mean_diffraction(cube, labels2d, max_per_class=300, seed=0):
    """Per-class mean diffraction pattern → dict ``{class_id: (H,W)}``.
    Subsamples up to ``max_per_class`` frames per class for speed."""
    Ny, Nx, H, W = cube.shape
    lab = np.asarray(labels2d).ravel()
    rng = np.random.RandomState(seed)
    out = {}
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        if idx.size > max_per_class:
            idx = rng.permutation(idx)[:max_per_class]
        ys, xs = np.divmod(idx, Nx)
        out[int(c)] = np.mean(
            [np.asarray(cube[y, x], np.float32) for y, x in zip(ys, xs)], axis=0)
    return out


def class_palette(K):
    """A ``ListedColormap`` of K distinct categorical colours (tab10/tab20)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    base = plt.get_cmap("tab20" if K > 10 else "tab10")
    return ListedColormap([base(i % base.N) for i in range(max(K, 2))])


# --------------------------------------------------------------------------
# run the trained DINO model on a cube (inference) + comparison plots
# --------------------------------------------------------------------------
def _run_params(ckpt_dir):
    """Read the polar/crop/mask/COM/vmax params the run was trained with.
    (``run_summary.json`` nests them under ``cfg`` / ``pre_pipeline``.)"""
    kw = {}
    for fn in ("_train_kwargs.json", "run_summary.json"):
        p = os.path.join(ckpt_dir, fn)
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                for sub in ("cfg", "pre_pipeline"):     # run_summary nesting
                    if isinstance(d.get(sub), dict):
                        kw.update({k: v for k, v in d[sub].items()})
                kw.update({k: v for k, v in d.items()
                           if not isinstance(v, dict)})
            except Exception:
                pass
    if kw.get("vmax") in (None, "None"):
        kw["vmax"] = None
    return dict(polar_size=int(kw.get("polar_size", 192)),
                polar_mask_cols=int(kw.get("polar_mask_cols", 30)),
                center_crop_size=int(kw.get("center_crop_size", 140)),
                center_mask_radius=int(kw.get("center_mask_radius", 15)),
                com_centering=bool(kw.get("com_centering", False)),
                vmax=kw.get("vmax"))


def run_dino_inference(cube_path, ckpt_dir, scan_shape, vmax=None,
                       device=None, use_cache=True):
    """Run the trained DINO checkpoint on a cube and return a 2-D class map.

    Returns ``dict(class_map (Ny,Nx), assigns (N,), soft (N,K), embeds (N,D),
    K, params)``.  If ``use_cache`` and a ``&lt;ckpt_dir&gt;/eval/inference.npz``
    exists, it's loaded instead of recomputing (fast).
    """
    add_src_to_path()
    import numpy as np
    Ny, Nx = int(scan_shape[0]), int(scan_shape[1])
    prm = _run_params(ckpt_dir)
    cache = os.path.join(ckpt_dir, "eval", "inference.npz")
    if use_cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        a = np.asarray(d["assigns"])
        return dict(class_map=a.reshape(Ny, Nx), assigns=a,
                    soft=np.asarray(d["soft_probs"]),
                    embeds=np.asarray(d["embeds"]),
                    K=int(a.max()) + 1, params=prm, cached=True)
    import torch
    from data import LoadPRZ
    from dino_sr_contrastive_model import load_contrastive_checkpoint
    import contrastive_eval as CE
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = LoadPRZ(cube_path, resize=prm["polar_size"],
                 vmax=float(vmax if vmax is not None else (prm["vmax"] or 2.0)))
    model, eval_temp, _, _ = load_contrastive_checkpoint(
        os.path.join(ckpt_dir, "best.pth"), device=device)
    inf = CE.infer_scan(
        model, ds, device, eval_temp=eval_temp, polar_size=prm["polar_size"],
        polar_mask_cols=prm["polar_mask_cols"],
        center_crop_size=prm["center_crop_size"],
        center_mask_radius=prm["center_mask_radius"],
        com_centering=prm["com_centering"])
    a = np.asarray(inf["assigns"])
    return dict(class_map=a.reshape(Ny, Nx), assigns=a,
                soft=np.asarray(inf["soft_probs"]),
                embeds=np.asarray(inf["embeds"]),
                K=int(a.max()) + 1, params=prm, cached=False)   # active K


def cosine_matrix(embeds, assigns, K=None):
    """K×K cosine-similarity matrix between per-class mean embeddings
    (via ``contrastive_eval.centroid_cosine_matrix``)."""
    add_src_to_path()
    import contrastive_eval as CE
    K = K or int(np.asarray(assigns).max()) + 1
    cos, cen = CE.centroid_cosine_matrix(np.asarray(embeds),
                                         np.asarray(assigns), K)
    return cos


def plot_gt_vs_pred(gt2d, score, title="ground truth vs prediction"):
    """4-panel comparison: GT | prediction (aligned) | agreement | per-class
    IoU, using the dict from :func:`align_and_score`.  Returns the figure."""
    import matplotlib.pyplot as plt
    gt = np.asarray(gt2d)
    pred = score["pred_aligned"]
    K = int(gt.max()) + 1
    pal = class_palette(K)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    ax[0].imshow(gt, cmap=pal, vmin=-.5, vmax=K - .5, interpolation="nearest")
    ax[0].set_title("Ground truth")
    ax[1].imshow(pred, cmap=pal, vmin=-.5, vmax=K - .5, interpolation="nearest")
    ax[1].set_title("Prediction (aligned)")
    ax[2].imshow(pred == gt, cmap="RdYlGn", vmin=0, vmax=1,
                 interpolation="nearest")
    ax[2].set_title("Agreement (green = correct)")
    ious = [p["IoU"] for p in score["per_class"]]
    ax[3].bar(range(K), ious, color=[pal(i) for i in range(K)])
    ax[3].set_ylim(0, 1.02); ax[3].set_title("per-class IoU")
    ax[3].set_xticks(range(K)); ax[3].set_xlabel("class")
    for a in ax[:3]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{title}   |   accuracy={score['accuracy']*100:.1f}%   "
                 f"ARI={score['ARI']:.3f}   NMI={score['NMI']:.3f}   "
                 f"mean IoU={score['mean_IoU']:.3f}", fontsize=12)
    fig.tight_layout()
    return fig
