"""
eval_all.py — Standalone evaluation script for all DINO-SR paper samples.

For each sample:
  1. Virtual HAADF, BF, COM magnitude + COM X/Y (DPC) maps
  2. Class map (from approved checkpoint)
  3. Uncertainty map (entropy) + confidence map
  4. Mean diffraction patterns per class (mean / median / conf-weighted)
  5. Similar-class detection (cosine similarity matrix + pair panels)
  6. Explainability (GradCAM + Blur-IG + Class Contrast)
  7. Summary figure (class map + HAADF + BF + COM X/Y + uncertainty)
  8. HDF5 export ({sample}_dino_results.h5) for py4DSTEM / MATLAB interop

Usage:
  conda run -n py4DSTEM_SAM python eval_all.py [sample_name]
  (omit sample_name to run all)
"""

import os, sys, math, traceback, gc
import numpy as np
try:
    import h5py as _h5py
except ImportError:
    _h5py = None
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import colorcet as cc
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import cv2

# ── local imports ──────────────────────────────────────────────────────────────
from dino_sr_fixed import (
    DINOModelSR, get_augmentation_transforms, load_checkpoint,
)
from dino_sr_ablation import (
    AblationDINOModelSR, load_ablation_checkpoint,
)
from explainability import ExplainabilityEngine

# ── constants ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "DINO_PAPER_V2")

RESIZE = 192
CENTER_CROP_SIZE = 135
N_EXPLAIN_EXAMPLES = 5

CM_GC  = 'inferno'
CM_IG  = 'hot'
CM_CON = 'RdBu_r'

# v4: local-copy paths. Data was copied from X:\ to D:\DINOSR\data\ to
# eliminate NAS I/O stalls that caused training to hang mid-epoch. Keep
# short, consistent filenames (one per sample key).
LOCAL_DATA_ROOT = r"D:\DINOSR\data"

SAMPLES = {
    "Na007a": {
        "path": os.path.join(LOCAL_DATA_ROOT, "Na007a.prz"),
        # shape (126, 100) confirmed via prz header; vmax=2 assumed from
        # Na007b (same NaPhi session, near-identical raw dynamic range).
        # Adjust vmax if the class map looks over/under-saturated.
        "vmax": 2, "scan_shape": (126, 100), "center_mask_radius": 15,
        "approved_label": None,   # no approved run yet for this sample
    },
    "Na007b": {
        "path": os.path.join(LOCAL_DATA_ROOT, "Na007b.prz"),
        "vmax": 2, "scan_shape": (126, 100), "center_mask_radius": 15,
        "approved_label": "heat_040_050",
    },
    "Na006a": {
        "path": os.path.join(LOCAL_DATA_ROOT, "Na006a.prz"),
        "vmax": 4, "scan_shape": (100, 100), "center_mask_radius": 15,
        "approved_label": "heat_040_070",
    },
    "EuInAs_B100": {
        "path": os.path.join(LOCAL_DATA_ROOT, "EuInAs_B100.prz"),
        "vmax": 30, "scan_shape": (66, 396), "center_mask_radius": 10,
        "approved_label": "heat_040_070",
    },
    "IMC_50nm_SI2": {
        "path": os.path.join(LOCAL_DATA_ROOT, "IMC_50nm_SI2.prz"),
        "vmax": 3, "scan_shape": (128, 128), "center_mask_radius": 15,
        "approved_label": "const_065",
    },
    "IMC_150nm_SI5": {
        "path": os.path.join(LOCAL_DATA_ROOT, "IMC_150nm_SI5.prz"),
        "vmax": 5, "scan_shape": (128, 128), "center_mask_radius": 15,
        "approved_label": "const_040",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading (copied from dino_sr_paper_figures_updated.py for standalone use)
# ═══════════════════════════════════════════════════════════════════════════════

def rescale_like_vmax(x, vmax, vmin=None, out_range=(0.0, 1.0), dtype=np.float32):
    x = np.asarray(x); lo, hi = map(float, out_range)
    if x.ndim == 2:
        vm = x.min() if vmin is None else float(vmin)
        d = (float(vmax) - vm) or 1e-12; y = (x - vm) / d
    elif x.ndim == 3:
        vm = x.min(axis=(1, 2), keepdims=True) if vmin is None else float(vmin)
        d = float(vmax) - vm
        if not np.isscalar(d): d[d == 0] = 1e-12
        else: d = d or 1e-12
        y = (x - vm) / d
    else:
        raise ValueError("Expected 2D or 3D")
    return (np.clip(y, 0, 1) * (hi - lo) + lo).astype(dtype, copy=False)


class LoadPRZ:
    def __init__(self, path, resize=192, vmax=10.0, vmin=None, out_range=(0.0, 1.0)):
        arr = np.load(path, allow_pickle=True, mmap_mode='r')
        cube = arr['data']
        self.Nx, self.Ny, self.H, self.W = cube.shape
        self.flat = cube.reshape(-1, self.H, self.W)
        self.resize = int(resize); self.vmax = float(vmax)
        self.vmin = vmin; self.out_range = out_range
        self._interp = cv2.INTER_AREA if self.resize < self.H else cv2.INTER_LINEAR

    def __len__(self): return self.flat.shape[0]

    def get_raw(self, idx):
        return self.flat[idx].astype(np.float32, copy=True)

    def __getitem__(self, idx):
        img = self.flat[idx].astype(np.float32, copy=False)
        if not np.isfinite(img).all(): img = np.nan_to_num(img, copy=False)
        if img.shape[0] != self.resize or img.shape[1] != self.resize:
            img = cv2.resize(img, (self.resize, self.resize), interpolation=self._interp)
        img = rescale_like_vmax(img, vmax=self.vmax, vmin=self.vmin,
                                out_range=self.out_range)
        return torch.from_numpy(img).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Eval transform + inference
# ═══════════════════════════════════════════════════════════════════════════════

def build_eval_transform(mask_radius, crop_size):
    _, ta = get_augmentation_transforms(center_mask_radius=mask_radius,
                                        center_crop_size=crop_size)
    det_ops = [t for t in ta.transforms
               if not any(x in type(t).__name__
                          for x in ["Random", "Jitter", "Blur", "Gaussian"])]
    return T.Compose(det_ops)


def run_inference(model, dataloader, transform, eval_temp, device, arch="resnet"):
    aa, ap = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device).float()
            x = transform(x)
            if arch.lower() == "vit":
                feat = model.teacher_head(model.teacher_encoder(x))
            else:
                feat = model.teacher_projector(model.teacher_encoder(x))
            feat_n = F.normalize(feat, dim=-1)
            logits = model.prototypes(feat_n)
            probs = torch.softmax((logits - model.center) / eval_temp, dim=-1)
            aa.append(probs.argmax(dim=-1).cpu())
            ap.append(probs.cpu())
    aa = torch.cat(aa).numpy()
    ap = torch.cat(ap, dim=0)
    pb = ap.mean(dim=0).clamp_min(1e-12)
    ek = math.exp(-(pb * pb.log()).sum().item())
    ac = ap.max(dim=-1).values.mean().item()
    ur, ct = np.unique(aa, return_counts=True)
    sr = ur[np.argsort(-ct)]
    remap = {r: i for i, r in enumerate(sr)}
    assigns = np.array([remap[x] for x in aa])
    return assigns, ek, ac, len(ur), ap, remap


# ═══════════════════════════════════════════════════════════════════════════════
# Virtual imaging: HAADF, BF, COM
# ═══════════════════════════════════════════════════════════════════════════════

def compute_virtual_images(dataset, scan_shape, center_mask_radius):
    """Compute virtual HAADF, BF, and COM maps from raw diffraction data."""
    N = len(dataset)
    Nx, Ny = scan_shape
    assert N == Nx * Ny, f"Dataset size {N} != scan_shape product {Nx*Ny}"

    H, W = dataset.H, dataset.W
    cy, cx = H / 2.0, W / 2.0

    # Create BF mask (central disk)
    yy, xx = np.ogrid[:H, :W]
    bf_mask = ((yy - cy)**2 + (xx - cx)**2) <= center_mask_radius**2

    # Coordinate grids for COM
    y_coords = np.arange(H, dtype=np.float64)
    x_coords = np.arange(W, dtype=np.float64)

    haadf_flat = np.zeros(N, dtype=np.float64)
    bf_flat = np.zeros(N, dtype=np.float64)
    com_x_flat = np.zeros(N, dtype=np.float64)
    com_y_flat = np.zeros(N, dtype=np.float64)

    for i in range(N):
        dp = dataset.get_raw(i).astype(np.float64)
        haadf_flat[i] = dp[~bf_mask].sum()   # HAADF = outside central beam disk
        bf_flat[i] = dp[bf_mask].sum()        # BF   = inside central beam disk

        total = dp.sum()
        if total > 1e-12:
            com_y_flat[i] = (dp.sum(axis=1) * y_coords).sum() / total - cy
            com_x_flat[i] = (dp.sum(axis=0) * x_coords).sum() / total - cx
        if i % 2000 == 0:
            print(f"  Virtual images: {i}/{N} ...", flush=True)

    haadf = haadf_flat.reshape(Nx, Ny)
    bf = bf_flat.reshape(Nx, Ny)
    com_x = com_x_flat.reshape(Nx, Ny)
    com_y = com_y_flat.reshape(Nx, Ny)
    com_mag = np.sqrt(com_x**2 + com_y**2)

    return haadf, bf, com_mag, com_x, com_y


def save_virtual_images(haadf, bf, com_mag, save_dir, com_x=None, com_y=None):
    """Save virtual HAADF, BF, COM magnitude (+ optional COM X/Y) as PNGs."""
    os.makedirs(save_dir, exist_ok=True)

    layers = [
        (haadf,   "virtual_haadf", "gray",    "Virtual HAADF"),
        (bf,      "virtual_bf",    "gray",    "Virtual BF"),
        (com_mag, "virtual_com",   "inferno", "COM Magnitude"),
    ]
    if com_x is not None:
        layers.append((com_x, "com_x", "RdBu_r", "COM X (DPC)"))
    if com_y is not None:
        layers.append((com_y, "com_y", "RdBu_r", "COM Y (DPC)"))

    for arr, name, cmap, title in layers:
        fig, ax = plt.subplots(figsize=(6, 6))
        if cmap == "RdBu_r":
            v = np.percentile(np.abs(arr), 99)
            im = ax.imshow(arr, cmap=cmap, vmin=-v, vmax=v, interpolation='nearest')
        else:
            v1, v99 = np.percentile(arr, [1, 99])
            im = ax.imshow(arr, cmap=cmap, vmin=v1, vmax=v99, interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join(save_dir, f"{name}.png"),
                    dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    print(f"  Saved virtual images to {save_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Class map
# ═══════════════════════════════════════════════════════════════════════════════

def save_class_map(assignments, scan_shape, save_dir, tag="best_auto"):
    os.makedirs(save_dir, exist_ok=True)
    n_active = len(np.unique(assignments))
    cmap = ListedColormap(cc.glasbey[:max(n_active, 2)])
    class_map = assignments.reshape(scan_shape)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(class_map, cmap=cmap, interpolation='nearest')
    ax.set_title(f"Class Map ({tag})  K={n_active}", fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.savefig(os.path.join(save_dir, f"map_{tag}.png"),
                dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Uncertainty / confidence maps (Piece 1)
# ═══════════════════════════════════════════════════════════════════════════════

def save_uncertainty_maps(all_probs_np, scan_shape, save_dir):
    """
    Compute and save per-position entropy (uncertainty) and max-confidence maps.

    Parameters
    ----------
    all_probs_np : np.ndarray (N, K) float32 — softmax probabilities
    scan_shape   : (Nx, Ny)
    save_dir     : directory to write PNGs

    Returns
    -------
    entropy_map : np.ndarray (Nx, Ny) — high = uncertain (phase boundary / amorphous)
    conf_map    : np.ndarray (Nx, Ny) — max(prob) per position
    """
    os.makedirs(save_dir, exist_ok=True)
    p = all_probs_np.astype(np.float64)
    eps = 1e-12
    entropy  = -(p * np.log(p + eps)).sum(axis=1)   # (N,)
    max_conf = p.max(axis=1)                          # (N,)

    entropy_map = entropy.reshape(scan_shape)
    conf_map    = max_conf.reshape(scan_shape)

    for arr, name, cmap, title in [
        (entropy_map, "uncertainty_map", "hot",     "Entropy (Uncertainty)"),
        (conf_map,    "confidence_map",  "viridis",  "Max Confidence"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(arr, cmap=cmap, interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(os.path.join(save_dir, f"{name}.png"),
                    dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    return entropy_map, conf_map


# ═══════════════════════════════════════════════════════════════════════════════
# HDF5 export (Piece 5)
# ═══════════════════════════════════════════════════════════════════════════════

def save_hdf5_results(assignments, all_probs_np, class_means, entropy_map,
                      com_x, com_y, scan_shape, sample_name, save_dir,
                      ckpt_path, eval_temp, ek, vmax):
    """
    Export all per-sample results to a self-contained HDF5 file.
    Loadable in py4DSTEM, MATLAB, and any h5py-compatible tool.
    """
    if _h5py is None:
        print("  h5py not installed -- skipping HDF5 export")
        return
    os.makedirs(save_dir, exist_ok=True)
    h5_path = os.path.join(save_dir, f"{sample_name}_dino_results.h5")
    with _h5py.File(h5_path, "w") as f:
        f.create_dataset("assignments",  data=assignments.reshape(scan_shape),
                         compression="gzip")
        f.create_dataset("all_probs",    data=all_probs_np,  compression="gzip")
        if entropy_map is not None:
            f.create_dataset("uncertainty", data=entropy_map, compression="gzip")
        if com_x is not None:
            f.create_dataset("com_x", data=com_x, compression="gzip")
        if com_y is not None:
            f.create_dataset("com_y", data=com_y, compression="gzip")
        if class_means:
            grp = f.create_group("mean_diffraction")
            for c, mdp in class_means.items():
                grp.create_dataset(f"class_{c}", data=mdp, compression="gzip")
        meta = f.create_group("metadata")
        meta.attrs["sample_name"]  = sample_name
        meta.attrs["ckpt_path"]    = str(ckpt_path)
        meta.attrs["eval_temp"]    = float(eval_temp)
        meta.attrs["effK"]         = float(ek)
        meta.attrs["vmax"]         = float(vmax)
        meta.attrs["scan_shape"]   = list(scan_shape)
    print(f"  Saved HDF5 -> {h5_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Mean diffraction patterns (3 variants: mean, median, conf-weighted)
# ═══════════════════════════════════════════════════════════════════════════════

def save_mean_diffraction_patterns(assignments, all_probs, dataset, n_active,
                                   save_dir, vmax_display=None):
    os.makedirs(save_dir, exist_ok=True)
    class_indices = {}
    for i, c in enumerate(assignments):
        class_indices.setdefault(c, []).append(i)

    means, medians, conf_means = {}, {}, {}

    for c, idxs in class_indices.items():
        print(f"  MDP class {c}: {len(idxs)} patterns ...", flush=True)
        raws = np.stack([dataset.get_raw(i).astype(np.float64) for i in idxs])

        means[c] = raws.mean(axis=0).astype(np.float32)
        medians[c] = np.median(raws, axis=0).astype(np.float32)

        weights = all_probs[idxs, c].astype(np.float64)
        w_sum = weights.sum()
        if w_sum > 1e-8:
            conf_means[c] = ((raws * weights[:, None, None]).sum(axis=0) / w_sum
                             ).astype(np.float32)
        else:
            conf_means[c] = means[c].copy()

    def _save_grid(dp_dict, tag, title):
        for c, dp in dp_dict.items():
            fig, ax = plt.subplots(figsize=(5, 5))
            vm = np.percentile(dp, 99.5) if vmax_display is None else vmax_display
            ax.imshow(dp, cmap='gray', vmin=0, vmax=vm, interpolation='nearest')
            ax.set_title(f"Class {c}  (N={len(class_indices[c])})\n{tag}",
                         fontsize=10, fontweight='bold')
            ax.axis('off')
            fig.savefig(os.path.join(save_dir, f"class_{c}_{tag}.png"),
                        dpi=200, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        nc = len(dp_dict)
        if nc == 0: return
        cols = min(5, nc); rows = math.ceil(nc / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
        axes_flat = np.array(axes).flatten()
        for i, (c, dp) in enumerate(sorted(dp_dict.items())):
            vm = np.percentile(dp, 99.5) if vmax_display is None else vmax_display
            axes_flat[i].imshow(dp, cmap='gray', vmin=0, vmax=vm, interpolation='nearest')
            axes_flat[i].set_title(f"C{c}  N={len(class_indices[c])}", fontsize=9)
            axes_flat[i].axis('off')
        for j in range(i + 1, len(axes_flat)): axes_flat[j].set_visible(False)
        fig.suptitle(f"{title}\n({tag})", fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, hspace=0.35)
        fig.savefig(os.path.join(save_dir, f"all_classes_grid_{tag}.png"),
                    dpi=200, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)

    _save_grid(means, "mean", "Mean Diffraction Patterns")
    _save_grid(medians, "median", "Median Diffraction Patterns")
    _save_grid(conf_means, "conf_mean", "Confidence-Weighted Mean Diffraction Patterns")
    print(f"  Saved MDP to {save_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Similar-class detection pipeline (ISSUE 2)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_means(assignments, dataset):
    """
    Return (means, class_indices) where means: class_id -> float32 MDP array.
    Uses get_raw() — unmasked raw patterns.
    """
    class_indices = {}
    for i, c in enumerate(assignments):
        class_indices.setdefault(c, []).append(i)
    means = {}
    for c, idxs in class_indices.items():
        raws = np.stack([dataset.get_raw(i).astype(np.float64) for i in idxs])
        means[c] = raws.mean(axis=0).astype(np.float32)
    return means, class_indices


def _radial_profile(image):
    """Mean intensity as a function of radius from image center."""
    H, W = image.shape
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.mgrid[:H, :W]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2).ravel().astype(int)
    vals = image.ravel().astype(np.float64)
    r_max = r.max()
    radial = np.zeros(r_max + 1)
    counts = np.zeros(r_max + 1)
    np.add.at(radial, r, vals)
    np.add.at(counts, r, 1)
    counts = np.maximum(counts, 1)
    return np.arange(r_max + 1), radial / counts


def _azimuthal_profile(image, n_bins=72):
    """Mean intensity as a function of azimuthal angle (degrees, 0-360)."""
    H, W = image.shape
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.mgrid[:H, :W]
    theta = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0
    bin_idx = np.floor(theta / (360.0 / n_bins)).astype(int).ravel()
    vals = image.ravel().astype(np.float64)
    az = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    np.add.at(az, bin_idx, vals)
    np.add.at(counts, bin_idx, 1)
    counts = np.maximum(counts, 1)
    angles = np.arange(n_bins) * (360.0 / n_bins)
    return angles, az / counts


def detect_similar_classes(means, class_indices, assigns, scan_shape, eval_dir,
                            sample_name, sim_thresh=0.97):
    """
    Cosine similarity matrix between all class MDPs. Flag pairs > sim_thresh.
    For flagged pairs: difference map, radial/azimuthal profiles, Welch t-test,
    spatial separation, and a summary panel with DISTINCT / POSSIBLY REDUNDANT verdict.
    Saves:
      eval/class_similarity_matrix.png
      eval/similar_pair_{i}_{j}.png  (one per flagged pair)
    Returns list of finding dicts.
    """
    from scipy import stats as sp_stats

    os.makedirs(eval_dir, exist_ok=True)
    classes = sorted(means.keys())
    n = len(classes)
    findings = []

    if n < 2:
        return findings

    # ── Build MDP vector matrix ──────────────────────────────────────────────
    vecs = np.stack([means[c].ravel().astype(np.float64) for c in classes])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    vecs_n = vecs / norms
    sim_mat = vecs_n @ vecs_n.T  # (n, n)

    # ── Save similarity matrix heatmap ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
    im = ax.imshow(sim_mat, cmap='RdYlGn', vmin=0.0, vmax=1.0, interpolation='nearest')
    ax.set_xticks(range(n)); ax.set_xticklabels([f"C{c}" for c in classes], fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"C{c}" for c in classes], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_mat[i,j]:.2f}", ha='center', va='center',
                    fontsize=7, color='black' if sim_mat[i, j] < 0.8 else 'white')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{sample_name} — MDP Cosine Similarity Matrix\n"
                 f"(threshold={sim_thresh})", fontsize=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(eval_dir, "class_similarity_matrix.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved class_similarity_matrix.png")

    # ── Scan for flagged pairs ───────────────────────────────────────────────
    for ii in range(n):
        for jj in range(ii + 1, n):
            sim = float(sim_mat[ii, jj])
            ca, cb = classes[ii], classes[jj]
            if sim <= sim_thresh:
                continue

            print(f"  Flagged similar pair: C{ca} vs C{cb}  sim={sim:.4f}")
            dp_a = means[ca].astype(np.float64)
            dp_b = means[cb].astype(np.float64)
            diff = dp_a - dp_b

            # Difference map normalised to [-1,1] by 99th abs percentile
            abspct = np.percentile(np.abs(diff), 99)
            if abspct < 1e-12: abspct = 1.0
            diff_norm = np.clip(diff / abspct, -1, 1)

            # Radial profiles
            r_ax, rad_a = _radial_profile(dp_a)
            _,    rad_b = _radial_profile(dp_b)

            # Azimuthal profiles
            az_ax, az_a = _azimuthal_profile(dp_a)
            _,     az_b = _azimuthal_profile(dp_b)

            # Welch t-test pixel-wise (flattened) and Cohen's d
            flat_a = dp_a.ravel()
            flat_b = dp_b.ravel()
            t_stat, p_val = sp_stats.ttest_ind(flat_a, flat_b, equal_var=False)
            pooled_std = np.sqrt((flat_a.std()**2 + flat_b.std()**2) / 2.0)
            cohens_d = (flat_a.mean() - flat_b.mean()) / max(pooled_std, 1e-12)

            # Spatial separation: where does each class appear?
            class_map = assigns.reshape(scan_shape)
            mask_a = (class_map == ca).astype(np.float32)
            mask_b = (class_map == cb).astype(np.float32)
            # Overlap fraction = IoU
            intersection = (mask_a * mask_b).sum()
            union = np.clip(mask_a + mask_b, 0, 1).sum()
            spatial_iou = float(intersection / max(union, 1))

            # Verdict
            if abs(cohens_d) > 0.5 or p_val < 0.001:
                verdict = "DISTINCT (statistically separable)"
            elif abs(cohens_d) < 0.2 and p_val > 0.05:
                verdict = "POSSIBLY REDUNDANT"
            else:
                verdict = "BORDERLINE"

            findings.append({
                "ca": ca, "cb": cb, "sim": sim,
                "p_val": p_val, "cohens_d": cohens_d,
                "spatial_iou": spatial_iou, "verdict": verdict,
            })

            # ── Summary panel ─────────────────────────────────────────────
            fig = plt.figure(figsize=(22, 10))
            gs = fig.add_gridspec(2, 5, hspace=0.45, wspace=0.35)

            # Row 0: MDP A, MDP B, difference map, radial, azimuthal
            vm_a = np.percentile(dp_a, 99.5)
            vm_b = np.percentile(dp_b, 99.5)
            vm   = max(vm_a, vm_b, 1e-6)

            ax00 = fig.add_subplot(gs[0, 0])
            ax00.imshow(dp_a, cmap='gray', vmin=0, vmax=vm, interpolation='nearest')
            ax00.set_title(f"Class {ca} MDP\n(N={len(class_indices[ca])})", fontsize=9)
            ax00.axis('off')

            ax01 = fig.add_subplot(gs[0, 1])
            ax01.imshow(dp_b, cmap='gray', vmin=0, vmax=vm, interpolation='nearest')
            ax01.set_title(f"Class {cb} MDP\n(N={len(class_indices[cb])})", fontsize=9)
            ax01.axis('off')

            ax02 = fig.add_subplot(gs[0, 2])
            ax02.imshow(diff_norm, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
            ax02.set_title(f"Difference\n(A − B, norm to 99th)", fontsize=9)
            ax02.axis('off')

            ax03 = fig.add_subplot(gs[0, 3])
            ax03.plot(r_ax, rad_a, label=f"C{ca}", linewidth=1.2)
            ax03.plot(r_ax, rad_b, label=f"C{cb}", linewidth=1.2, linestyle='--')
            ax03.set_xlabel("Radius (px)", fontsize=8)
            ax03.set_ylabel("Mean intensity", fontsize=8)
            ax03.set_title("Radial profile", fontsize=9)
            ax03.legend(fontsize=7)
            ax03.tick_params(labelsize=7)

            ax04 = fig.add_subplot(gs[0, 4])
            ax04.plot(az_ax, az_a, label=f"C{ca}", linewidth=1.2)
            ax04.plot(az_ax, az_b, label=f"C{cb}", linewidth=1.2, linestyle='--')
            ax04.set_xlabel("Angle (°)", fontsize=8)
            ax04.set_ylabel("Mean intensity", fontsize=8)
            ax04.set_title("Azimuthal profile", fontsize=9)
            ax04.legend(fontsize=7)
            ax04.tick_params(labelsize=7)

            # Row 1: class maps (A, B, overlap) + text stats
            ax10 = fig.add_subplot(gs[1, 0])
            ax10.imshow(mask_a, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
            ax10.set_title(f"Class {ca} spatial", fontsize=9); ax10.axis('off')

            ax11 = fig.add_subplot(gs[1, 1])
            ax11.imshow(mask_b, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
            ax11.set_title(f"Class {cb} spatial", fontsize=9); ax11.axis('off')

            overlap_vis = mask_a * 0.5 + mask_b * 1.0
            ax12 = fig.add_subplot(gs[1, 2])
            ax12.imshow(overlap_vis, cmap='RdBu_r', vmin=0, vmax=1.5,
                        interpolation='nearest')
            ax12.set_title(f"Spatial overlap\n(IoU={spatial_iou:.3f})", fontsize=9)
            ax12.axis('off')

            ax13 = fig.add_subplot(gs[1, 3:])
            ax13.axis('off')
            stats_txt = (
                f"Cosine similarity: {sim:.4f}  (thresh={sim_thresh})\n"
                f"Welch t-test p-value: {p_val:.4e}\n"
                f"Cohen's d: {cohens_d:.4f}\n"
                f"Spatial IoU: {spatial_iou:.4f}\n\n"
                f"Verdict: {verdict}"
            )
            ax13.text(0.05, 0.95, stats_txt, transform=ax13.transAxes,
                      fontsize=10, va='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            fig.suptitle(
                f"{sample_name}  —  Similar-class analysis: C{ca} vs C{cb}\n"
                f"cosine sim = {sim:.4f}",
                fontsize=12, fontweight='bold'
            )
            pair_path = os.path.join(eval_dir, f"similar_pair_{ca}_{cb}.png")
            fig.savefig(pair_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved {pair_path}")

    if not findings:
        print(f"  No similar pairs found above threshold {sim_thresh}")

    return findings


# ═══════════════════════════════════════════════════════════════════════════════
# Explainability
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_map(x, mode="percentile"):
    if x is None: return np.zeros((10, 10))
    x = np.asarray(x, dtype=np.float64)
    if x.max() == x.min(): return np.zeros_like(x)
    if mode == "abs_robust":
        ax = np.abs(x); vm = max(np.percentile(ax, 98), 1e-8)
        return np.clip(ax / vm, 0, 1)
    elif mode == "diverging":
        ax = np.abs(x); vm = max(np.percentile(ax, 98), 1e-8)
        return np.clip(x / (2 * vm) + 0.5, 0, 1)
    elif mode == "soft":
        v0, v1 = np.percentile(x, 0.5), np.percentile(x, 99.5)
        rng = v1 - v0 if v1 > v0 else 1e-8
        return np.clip((x - v0) / rng, 0, 1)
    else:
        return np.clip((x - x.min()) / (x.max() - x.min() + 1e-12), 0, 1)


def save_explainability(model, assignments, dataset, eval_tf, arch,
                        eval_temp, save_dir, n_examples=5):
    os.makedirs(save_dir, exist_ok=True)
    engine = ExplainabilityEngine(
        model, architecture="vit" if arch.lower() == "vit" else "resnet",
        device=str(DEVICE))

    for k in np.unique(assignments):
        idxs = np.where(assignments == k)[0]; nf = len(idxs)
        cd = os.path.join(save_dir, f"class_{k}_examples")
        os.makedirs(cd, exist_ok=True)
        rng = np.random.RandomState(42)
        sel = rng.choice(idxs, min(nf, 50), replace=False)
        sm1 = sm2 = scon = simg = None; cnt = 0; pairs = []

        for ie, i in enumerate(sel):
            try:
                item = dataset[i]
                xr = item.to(DEVICE).float()
                if xr.ndim == 2: xr = xr.unsqueeze(0).unsqueeze(0)
                elif xr.ndim == 3: xr = xr.unsqueeze(0)
                xi = eval_tf(xr)
                maps = engine.explain(xi, target_class=int(k),
                                      temp=eval_temp, ig_steps=20)
                m1 = maps['gradcam']; m2 = maps['blur_ig']; con = maps['class_contrast']
                inp = xi.detach().cpu().numpy().squeeze()

                if ie < n_examples:
                    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
                    ax[0].imshow(normalize_map(inp), cmap='gray')
                    ax[0].set_title(f"Input idx {i}", fontsize=9); ax[0].axis('off')
                    ax[1].imshow(normalize_map(inp), cmap='gray')
                    im1 = ax[1].imshow(normalize_map(m1), cmap=CM_GC, alpha=0.6)
                    ax[1].set_title("GradCAM", fontsize=9); ax[1].axis('off')
                    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
                    im2 = ax[2].imshow(normalize_map(m2, "abs_robust"), cmap=CM_IG)
                    ax[2].set_title("Blur IG", fontsize=9); ax[2].axis('off')
                    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
                    ax[3].imshow(normalize_map(inp), cmap='gray')
                    if con is not None:
                        im3 = ax[3].imshow(normalize_map(con, "diverging"),
                                           cmap=CM_CON, alpha=0.7, vmin=0, vmax=1)
                        fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
                    ax[3].set_title("Contrast", fontsize=9); ax[3].axis('off')
                    fig.subplots_adjust(wspace=0.35)
                    fig.savefig(os.path.join(cd, f"ex_{ie:03d}_idx_{i}.png"),
                                dpi=150, bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)

                if cnt < 30:
                    if len(pairs) < 5: pairs.append((inp, m1, m2, con))
                    if sm1 is None:
                        sm1 = m1.astype(np.float64); sm2 = m2.astype(np.float64)
                        simg = inp.astype(np.float64)
                        scon = (con.astype(np.float64) if con is not None
                                else np.zeros_like(inp, dtype=np.float64))
                    else:
                        sm1 += m1; sm2 += m2; simg += inp
                        if con is not None: scon += con
                    cnt += 1
            except Exception:
                traceback.print_exc()
                continue

        if cnt == 0: continue
        ai = simg / cnt; am1 = sm1 / cnt; am2 = sm2 / cnt; acon = scon / cnt

        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        axes[0].imshow(normalize_map(ai, "soft"), cmap='gray')
        axes[0].set_title(f"Class {k} (N={nf})", fontsize=9); axes[0].axis('off')
        axes[1].imshow(normalize_map(ai, "soft"), cmap='gray')
        im1 = axes[1].imshow(normalize_map(am1, "soft"), cmap=CM_GC, alpha=0.6)
        axes[1].set_title("Avg GradCAM", fontsize=9); axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        im2 = axes[2].imshow(normalize_map(am2, "abs_robust"), cmap=CM_IG)
        axes[2].set_title("Avg Blur IG", fontsize=9); axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        axes[3].imshow(normalize_map(ai, "soft"), cmap='gray')
        im3 = axes[3].imshow(normalize_map(acon, "diverging"),
                             cmap=CM_CON, alpha=0.7, vmin=0, vmax=1)
        axes[3].set_title("Avg Contrast", fontsize=9); axes[3].axis('off')
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        fig.subplots_adjust(wspace=0.35)
        fig.savefig(os.path.join(save_dir, f"Class_{k}_AVG.png"),
                    dpi=200, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"    Class {k}: {cnt} examples for averages, {len(pairs)} for summary")

    del engine; torch.cuda.empty_cache()
    print(f"  Saved explainability to {save_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary figure
# ═══════════════════════════════════════════════════════════════════════════════

def save_summary_figure(assignments, scan_shape, haadf, bf, com_mag, save_dir,
                        sample_name, com_x=None, com_y=None, entropy_map=None):
    os.makedirs(save_dir, exist_ok=True)
    n_active = len(np.unique(assignments))
    cmap = ListedColormap(cc.glasbey[:max(n_active, 2)])
    class_map = assignments.reshape(scan_shape)

    # Build panel list — always have class + HAADF + BF + COM mag; add COM X/Y + entropy if available
    panels = [
        (class_map, f"Class Map (K={n_active})", "class", None),
        (haadf,     "Virtual HAADF",              "gray",    None),
        (bf,        "Virtual BF",                 "gray",    None),
        (com_mag,   "COM Magnitude",               "inferno", None),
    ]
    if com_x is not None:
        v = np.percentile(np.abs(com_x), 99)
        panels.append((com_x, "COM X (DPC)", "RdBu_r", (-v, v)))
    if com_y is not None:
        v = np.percentile(np.abs(com_y), 99)
        panels.append((com_y, "COM Y (DPC)", "RdBu_r", (-v, v)))
    if entropy_map is not None:
        panels.append((entropy_map, "Uncertainty (Entropy)", "hot", None))

    n = len(panels)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6.5 * rows))
    axes_flat = np.array(axes).flatten()

    for i, (arr, title, cm, clim) in enumerate(panels):
        ax = axes_flat[i]
        if cm == "class":
            ax.imshow(arr, cmap=cmap, interpolation='nearest')
        elif clim is not None:
            ax.imshow(arr, cmap=cm, vmin=clim[0], vmax=clim[1], interpolation='nearest')
        else:
            v1, v99 = np.percentile(arr, [1, 99])
            ax.imshow(arr, cmap=cm, vmin=v1, vmax=v99, interpolation='nearest')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{sample_name} — Summary", fontsize=14, fontweight='bold')
    fig.tight_layout()
    # Use summary_v2.png when extended panels (COM / entropy) are present
    fname = "summary_v2.png" if (com_x is not None or entropy_map is not None) else "summary.png"
    fig.savefig(os.path.join(save_dir, fname),
                dpi=200, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_sample(sample_name, cfg):
    print(f"\n{'='*70}")
    print(f"  Evaluating: {sample_name}")
    print(f"{'='*70}")

    label = cfg["approved_label"]
    ckpt_dir = os.path.join(BASE_OUTPUT_DIR, sample_name, "L1", label)
    ckpt_path = os.path.join(ckpt_dir, "best_auto.pth")
    eval_dir = os.path.join(ckpt_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"  SKIP: checkpoint not found: {ckpt_path}")
        return

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"  Loading dataset: {cfg['path']}")
    dataset = LoadPRZ(cfg["path"], resize=RESIZE, vmax=cfg["vmax"])
    scan_shape = cfg["scan_shape"]
    print(f"  Dataset: {len(dataset)} patterns, scan_shape={scan_shape}, "
          f"dp_size=({dataset.H},{dataset.W})")

    # ── Virtual images ────────────────────────────────────────────────────────
    vi_path = os.path.join(eval_dir, "virtual_haadf.png")
    if os.path.exists(vi_path) and os.path.exists(os.path.join(eval_dir, "com_x.png")):
        print("  Virtual images already exist, skipping ...")
        haadf = bf = com_mag = com_x = com_y = None
    else:
        print("  Computing virtual images (HAADF, BF, COM X/Y) ...")
        haadf, bf, com_mag, com_x, com_y = compute_virtual_images(
            dataset, scan_shape, cfg["center_mask_radius"])
        save_virtual_images(haadf, bf, com_mag, eval_dir, com_x=com_x, com_y=com_y)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"  Loading checkpoint: {ckpt_path}")
    # All approved runs are L1 resnet (ablation model)
    model, eval_temp, metrics = load_ablation_checkpoint(ckpt_path, device=DEVICE)
    model.eval()
    arch = "resnet"
    print(f"  eval_temp={eval_temp:.4f}, metrics keys: {list(metrics.keys())}")

    # ── Build eval transform ──────────────────────────────────────────────────
    eval_tf = build_eval_transform(cfg["center_mask_radius"], CENTER_CROP_SIZE)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    # ── Run inference ─────────────────────────────────────────────────────────
    print("  Running inference ...")
    assigns, ek, ac, n_active, all_probs, remap = run_inference(
        model, dataloader, eval_tf, eval_temp, DEVICE, arch)
    print(f"  effK={ek:.2f}, n_active={n_active}, avg_conf={ac:.3f}")

    # Remap all_probs columns to match assignments
    all_probs_np = all_probs.numpy()

    # ── Class map ─────────────────────────────────────────────────────────────
    save_class_map(assigns, scan_shape, eval_dir, tag="best_auto")
    print("  Saved class map")

    # ── Eval summary JSON (for dashboard) ─────────────────────────────────────
    import json as _json
    _summary = {"eff_k": float(ek), "n_active": int(n_active),
                "avg_conf": float(ac), "eval_temp": float(eval_temp)}
    with open(os.path.join(eval_dir, "eval_summary.json"), "w", encoding="utf-8") as _f:
        _json.dump(_summary, _f, indent=2)
    print(f"  Saved eval_summary.json (EffK={ek:.2f} n_active={n_active})")

    # ── Uncertainty / confidence maps (Piece 1) ───────────────────────────────
    entropy_map = conf_map = None
    unc_path = os.path.join(eval_dir, "uncertainty_map.png")
    if not os.path.exists(unc_path):
        print("  Computing uncertainty maps ...")
        try:
            entropy_map, conf_map = save_uncertainty_maps(
                all_probs_np, scan_shape, eval_dir)
            print("  Saved uncertainty_map.png and confidence_map.png")
        except Exception:
            traceback.print_exc()
            print("  ERROR in uncertainty maps!")
    else:
        print("  Uncertainty maps already exist, skipping ...")

    # ── Mean diffraction patterns ─────────────────────────────────────────────
    mdp_dir = os.path.join(eval_dir, "mdp_best_auto")
    if not os.path.exists(os.path.join(mdp_dir, "all_classes_grid_mean.png")):
        print("  Computing mean diffraction patterns ...")
        try:
            save_mean_diffraction_patterns(assigns, all_probs_np, dataset,
                                           n_active, mdp_dir)
        except Exception:
            traceback.print_exc()
            print("  ERROR in mean diffraction patterns!")
    else:
        print("  MDP already exist, skipping ...")

    # ── Similar-class detection ───────────────────────────────────────────────
    sim_mat_path = os.path.join(eval_dir, "class_similarity_matrix.png")
    if not os.path.exists(sim_mat_path):
        print("  Running similar-class detection ...")
        try:
            class_means, class_indices = compute_class_means(assigns, dataset)
            sim_findings = detect_similar_classes(
                class_means, class_indices, assigns, scan_shape, eval_dir, sample_name)
            if sim_findings:
                print(f"  Similar-class findings ({len(sim_findings)}):")
                for f in sim_findings:
                    print(f"    C{f['ca']} vs C{f['cb']}: sim={f['sim']:.4f} "
                          f"p={f['p_val']:.2e} d={f['cohens_d']:.3f}  -> {f['verdict']}")
        except Exception:
            traceback.print_exc()
            print("  ERROR in similar-class detection!")
    else:
        print("  Similarity matrix already exists, skipping ...")

    # ── Explainability ────────────────────────────────────────────────────────
    explain_dir = os.path.join(eval_dir, "explainability_best_auto")
    if not os.path.exists(os.path.join(explain_dir, "Class_0_AVG.png")):
        print("  Running explainability (GradCAM + Blur-IG + Contrast) ...")
        try:
            save_explainability(model, assigns, dataset, eval_tf, arch,
                                eval_temp, explain_dir, n_examples=N_EXPLAIN_EXAMPLES)
        except Exception:
            traceback.print_exc()
            print("  ERROR in explainability!")
    else:
        print("  Explainability already exists, skipping ...")

    # ── HDF5 export (Piece 5) ─────────────────────────────────────────────────
    h5_path = os.path.join(eval_dir, f"{sample_name}_dino_results.h5")
    if not os.path.exists(h5_path):
        print("  Saving HDF5 results ...")
        try:
            class_means_for_h5, _ = compute_class_means(assigns, dataset)
            save_hdf5_results(
                assigns, all_probs_np, class_means_for_h5, entropy_map,
                com_x, com_y, scan_shape, sample_name, eval_dir,
                ckpt_path, eval_temp, ek, cfg["vmax"])
        except Exception:
            traceback.print_exc()
            print("  ERROR in HDF5 export!")
    else:
        print("  HDF5 already exists, skipping ...")

    # ── Summary figure ────────────────────────────────────────────────────────
    # summary_v2.png = extended (COM + entropy panels); summary.png = legacy 4-panel.
    # Skip if either already exists to avoid redundant regeneration.
    _has_new_data = (entropy_map is not None) or (com_x is not None)
    summary_path = os.path.join(eval_dir,
                                "summary_v2.png" if _has_new_data else "summary.png")
    _summary_exists = (os.path.exists(os.path.join(eval_dir, "summary_v2.png")) or
                       os.path.exists(os.path.join(eval_dir, "summary.png")))
    if not _summary_exists:
        # Need virtual images for summary
        if haadf is None:
            print("  Recomputing virtual images for summary ...")
            haadf, bf, com_mag, com_x, com_y = compute_virtual_images(
                dataset, scan_shape, cfg["center_mask_radius"])
        save_summary_figure(assigns, scan_shape, haadf, bf, com_mag, eval_dir,
                            sample_name, com_x=com_x, com_y=com_y,
                            entropy_map=entropy_map)
        print(f"  Saved {os.path.basename(summary_path)}")
    else:
        print("  Summary already exists, skipping ...")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    del model, dataset, dataloader
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  DONE: {sample_name}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Output dir: {BASE_OUTPUT_DIR}")

    # Filter samples if a name was given
    if len(sys.argv) > 1:
        names = [n for n in sys.argv[1:] if n in SAMPLES]
        if not names:
            print(f"Unknown sample(s): {sys.argv[1:]}. Available: {list(SAMPLES.keys())}")
            sys.exit(1)
    else:
        names = list(SAMPLES.keys())

    for name in names:
        try:
            evaluate_sample(name, SAMPLES[name])
        except Exception:
            traceback.print_exc()
            print(f"  FAILED: {name}")


if __name__ == "__main__":
    main()
