"""
Evaluation helpers for the DINO-SR + contrastive runs.

Spec 14 required metrics (no t-SNE, UMAP only):

  Quantitative
    - NMI (normalized mutual information) against a reference labeling
      (the pure-DINO baseline assignments, since we have no ground truth).
    - KNN purity (k=10)
    - intra-class cosine, inter-class cosine, ratio
    - radial profile consistency
    - prototype usage entropy + effective K

  Qualitative
    - UMAP of student contrastive embeddings, colored by prototype assignment
    - Representative patterns per prototype (nearest to centroid)
    - Highest-teacher-softmax-entropy samples
    - Theta-roll invariance sanity: cos(embed(x), embed(roll(x, k)))

Also contains a scanwise inference helper that returns (N, K) soft labels,
(N, D) contrastive embeddings, and argmax assignments.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Force non-interactive matplotlib backend BEFORE any pyplot import anywhere
# downstream. Avoids Tcl/tkinter "main thread is not in main loop" crashes
# when plots are created from background scripts or threads.
import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import torch
import torch.nn.functional as F

from dino_sr_contrastive_model import (
    PolarThetaRoll,
    PolarTransform,
    PolarMaskLeft,
)


# =========================================================================
# 1. INFERENCE — student path, deterministic eval transform
# =========================================================================

@torch.no_grad()
def infer_scan(model, dataset, device,
               polar_size: int = 192,
               polar_mask_cols: int = 30,
               center_crop_size: int = 140,
               eval_temp: float = 0.06,
               batch_size: int = 128,
               dense_remap: bool = True,
               com_centering: bool = False,
               com_search_radius_factor: float = 2.0,
               center_mask_radius: int = 15):
    """Run model over the full dataset with a deterministic eval transform.

    When `dense_remap=True` (default) the returned assigns are renumbered to
    0..K_active-1 sorted by count descending, and the corresponding columns
    of soft_probs / teacher_probs are reordered to match. The `K_original_ids`
    entry in the returned dict records the mapping new_id -> original_id so
    downstream code can recover prototype identity if needed.

    Returns:
        dict with:
            soft_probs     : (N, K_active) softmax over (remapped) prototypes
            assigns        : (N,) argmax cluster id in [0, K_active)
            embeds         : (N, D) L2-normed contrastive embeddings
            features       : (N, in_dim) backbone features pre-projection
            teacher_probs  : (N, K_active) teacher softmax reordered to match
            K_original_ids : list[int]  new_id -> original prototype id
                             (empty list if dense_remap=False)
            K_original     : int  original K before any dead-pruning
    """
    from torchvision.transforms import v2 as T
    from torchvision.transforms import InterpolationMode

    ops = [T.CenterCrop(center_crop_size)]
    if com_centering:
        from dino_sr_ablation import CenterOnCOM
        ops.append(CenterOnCOM(
            search_radius=int(com_search_radius_factor * center_mask_radius)))
    ops += [
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        PolarTransform(output_size=polar_size),
        PolarMaskLeft(k_cols=polar_mask_cols),
    ]
    eval_tf = T.Compose(ops)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    model.eval()
    all_probs, all_t_probs, all_embeds, all_feat = [], [], [], []
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch)
        x = x.to(device, non_blocking=True).float()
        x = eval_tf(x)
        feat = model.teacher_encoder(x)
        proj = model.teacher_projector(feat)
        logits = model.prototypes(proj)
        t_probs = torch.softmax((logits - model.center) / eval_temp, dim=-1)
        # Student contrastive head for the embedding (uses BN; make sure we
        # run it in eval mode -- handled by model.eval() above).
        embed = model.student_contrastive(model.student_encoder(x))
        s_logits = model.prototypes(model.student_projector(model.student_encoder(x)))
        s_probs = torch.softmax((s_logits - model.center) / eval_temp, dim=-1)
        all_probs.append(s_probs.cpu())
        all_t_probs.append(t_probs.cpu())
        all_embeds.append(embed.cpu())
        all_feat.append(feat.cpu())

    soft_probs = torch.cat(all_probs, 0).numpy()
    teacher_probs = torch.cat(all_t_probs, 0).numpy()
    embeds = torch.cat(all_embeds, 0).numpy()
    features = torch.cat(all_feat, 0).numpy()
    assigns = soft_probs.argmax(axis=1)
    K_original = int(soft_probs.shape[1])
    original_ids: list = []
    if dense_remap:
        assigns, soft_probs, teacher_probs, original_ids = dense_remap_by_usage(
            assigns, soft_probs, teacher_probs)
    return dict(soft_probs=soft_probs, teacher_probs=teacher_probs,
                embeds=embeds, features=features, assigns=assigns,
                K_original_ids=original_ids, K_original=K_original)


# =========================================================================
# 2. QUANTITATIVE METRICS
# =========================================================================

def nmi(a, b):
    """Normalized mutual information between two discrete labelings."""
    a = np.asarray(a).astype(int); b = np.asarray(b).astype(int)
    n = len(a)
    if n == 0:
        return 0.0
    ua = np.unique(a); ub = np.unique(b)
    mi = 0.0
    for va in ua:
        for vb in ub:
            both = ((a == va) & (b == vb)).sum()
            if both == 0:
                continue
            pab = both / n
            pa  = (a == va).sum() / n
            pb  = (b == vb).sum() / n
            mi += pab * math.log(pab / (pa * pb + 1e-12) + 1e-12)
    def H(x):
        _, c = np.unique(x, return_counts=True)
        p = c / c.sum()
        return -(p * np.log(p + 1e-12)).sum()
    Ha = H(a); Hb = H(b)
    denom = 0.5 * (Ha + Hb)
    return float(mi / denom) if denom > 0 else 0.0


def knn_purity(embeds: np.ndarray, labels: np.ndarray, k: int = 10,
               max_query: int = 2000, seed: int = 0) -> float:
    """KNN purity (k=10 default).

    For each query, fraction of its top-k nearest (cosine) neighbors with
    the same label as the query. Averaged across `max_query` random
    queries. Purity is computed against all-other points in the set.
    """
    N = embeds.shape[0]
    if N < k + 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    qidx = rng.choice(N, size=min(max_query, N), replace=False)
    # Normalize (should already be L2-normed for contrastive; normalize
    # again to be safe).
    E = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12)
    purs = []
    # Compute in chunks to avoid an (N, N) matrix.
    chunk = 256
    for i in range(0, len(qidx), chunk):
        q = qidx[i: i + chunk]
        sims = E[q] @ E.T          # (c, N)
        # Exclude self — set own sim to -inf.
        for ci, gi in enumerate(q):
            sims[ci, gi] = -np.inf
        top = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        for ci, gi in enumerate(q):
            same = (labels[top[ci]] == labels[gi]).mean()
            purs.append(same)
    return float(np.mean(purs))


def intra_inter_cosine(embeds: np.ndarray, labels: np.ndarray,
                        max_per_class: int = 200, seed: int = 0):
    """Intra- and inter-class mean cosine similarity on the provided embeddings."""
    rng = np.random.default_rng(seed)
    E = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12)
    classes = np.unique(labels)
    if len(classes) < 2:
        return float("nan"), float("nan")
    by_class = {}
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        by_class[c] = idx
    intra = []
    for c, idx in by_class.items():
        if len(idx) < 2:
            continue
        Ec = E[idx]
        s = Ec @ Ec.T
        iu = np.triu_indices_from(s, k=1)
        intra.extend(s[iu].tolist())
    inter = []
    cls = list(by_class.keys())
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            Ei, Ej = E[by_class[cls[i]]], E[by_class[cls[j]]]
            inter.extend((Ei @ Ej.T).ravel().tolist())
    return float(np.mean(intra)) if intra else float("nan"), \
           float(np.mean(inter)) if inter else float("nan")


def radial_profile_consistency(raw_patterns: np.ndarray, labels: np.ndarray,
                                n_rings: int = 40,
                                max_per_class: int = 100,
                                seed: int = 0) -> float:
    """Mean within-class vs between-class cosine on radial profiles.

    For each sample compute its 1-D radial profile (azimuthal average vs
    radius), averaged per class. Return mean intra-class correlation
    minus inter-class correlation — higher is better.
    """
    if raw_patterns.ndim != 3:
        raise ValueError("raw_patterns must be (N, H, W)")
    N, H, W = raw_patterns.shape
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = min(cy, cx)
    bins = np.linspace(0, r_max, n_rings + 1)
    bin_idx = np.clip(np.digitize(r, bins) - 1, 0, n_rings - 1)

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    profs = {}
    for c in classes:
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        ps = []
        for i in idx:
            p = np.zeros(n_rings)
            pat = raw_patterns[i]
            for k in range(n_rings):
                m = (bin_idx == k)
                if m.any():
                    p[k] = pat[m].mean()
            ps.append(p)
        profs[c] = np.stack(ps, 0)

    def corr(a, b):
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        return float((a * b).mean())

    intras, inters = [], []
    cls = list(profs.keys())
    for c in cls:
        M = profs[c]
        mean = M.mean(0)
        for row in M:
            intras.append(corr(row, mean))
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            inters.append(corr(profs[cls[i]].mean(0), profs[cls[j]].mean(0)))
    intra = float(np.mean(intras)) if intras else float("nan")
    inter = float(np.mean(inters)) if inters else float("nan")
    return intra - inter


def dense_remap_by_usage(assigns: np.ndarray, soft_probs: np.ndarray,
                          teacher_probs: np.ndarray | None = None):
    """Renumber active prototypes densely 0..K_active-1, sorted by count
    descending (largest cluster becomes class 0).

    Dead prototypes (zero samples assigned) are dropped from the label space
    but their columns remain in soft_probs / teacher_probs via the remap,
    so the caller can still access "did the network ever vote for this?"
    for dead columns if needed.

    Returns:
        new_assigns       : (N,) dense int labels
        new_soft_probs    : (N, K_active) columns reordered to match new labels
        new_teacher_probs : (N, K_active) same reorder, or None
        old_ids_new_order : list[int] mapping new id -> original prototype id
    """
    K = soft_probs.shape[1]
    counts = np.bincount(assigns, minlength=K)
    active = np.where(counts > 0)[0]                      # original ids with votes
    if len(active) == 0:
        return assigns.copy(), soft_probs.copy(), \
               (teacher_probs.copy() if teacher_probs is not None else None), []
    # Sort active ids by count descending.
    order = active[np.argsort(-counts[active])]
    old_to_new = {int(old): i for i, old in enumerate(order)}
    new_assigns = np.array([old_to_new[int(a)] for a in assigns], dtype=np.int64)
    new_soft = soft_probs[:, order]
    new_teacher = teacher_probs[:, order] if teacher_probs is not None else None
    return new_assigns, new_soft, new_teacher, [int(o) for o in order]


def stripe_metric(assigns: np.ndarray, scan_shape: tuple, K: int,
                   min_members: int = 8) -> list:
    """For each prototype, ratio of max/min eigenvalue of the 2-D spatial
    covariance of assigned scan positions. Values near 1 are blob-like
    (physically coherent region); values >> 1 are line-like (likely a
    scan-drift stripe or other 1-D artifact).

    Returns a list of length K. Prototypes with fewer than `min_members`
    samples get score 1.0 (not informative).
    """
    assigns_2d = np.asarray(assigns).reshape(scan_shape)
    scores = []
    for c in range(K):
        ys, xs = np.where(assigns_2d == c)
        if len(ys) < min_members:
            scores.append(1.0)
            continue
        pos = np.stack([ys, xs], axis=1).astype(np.float64)
        pos = pos - pos.mean(axis=0, keepdims=True)
        cov = (pos.T @ pos) / max(len(pos) - 1, 1)
        eigs = np.linalg.eigvalsh(cov)
        eigs = eigs[eigs > 1e-6]
        if len(eigs) < 2:
            scores.append(1.0)
        else:
            scores.append(float(eigs.max() / eigs.min()))
    return scores


def prototype_usage_entropy(soft_probs: np.ndarray):
    """(entropy, effective_K, per_class_count)."""
    p = soft_probs.mean(0)
    p = np.clip(p, 1e-12, 1.0)
    ent = float(-(p * np.log(p)).sum())
    eff_k = math.exp(ent)
    counts = np.bincount(soft_probs.argmax(axis=1), minlength=soft_probs.shape[1])
    return ent, eff_k, counts


def centroid_cosine_matrix(embeds: np.ndarray, labels: np.ndarray, K: int):
    """Per-class centroid, cosine similarity matrix.

    Returns (K, K) cosine matrix and (K, D) centroid matrix. Dead classes
    (no members) produce zero centroids; their cos-sim is zero against others.
    """
    D = embeds.shape[1]
    centroids = np.zeros((K, D), dtype=np.float32)
    for c in range(K):
        mask = (labels == c)
        if mask.any():
            centroids[c] = embeds[mask].mean(0)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    cn = centroids / (norms + 1e-12)
    cos = cn @ cn.T
    return cos, centroids


# =========================================================================
# 3. ROTATION INVARIANCE SANITY
# =========================================================================

@torch.no_grad()
def theta_roll_cosine_distribution(model, dataset, device,
                                    polar_size: int = 192,
                                    polar_mask_cols: int = 30,
                                    center_crop_size: int = 140,
                                    batch_size: int = 64,
                                    n_batches: int = 4,
                                    shift_range: int = 192,
                                    seed: int = 0):
    """For `batch_size * n_batches` held-out samples, compute cos(z(x),
    z(roll(x, k))) where k is a random theta shift.

    Returns list of cosines. High values indicate learned theta-invariance.
    """
    from torchvision.transforms import v2 as T
    from torchvision.transforms import InterpolationMode
    eval_tf = T.Compose([
        T.CenterCrop(center_crop_size),
        T.Resize(polar_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
        PolarTransform(output_size=polar_size),
        PolarMaskLeft(k_cols=polar_mask_cols),
    ])
    rng = np.random.default_rng(seed)
    N = len(dataset)
    sample_idx = rng.choice(N, size=min(batch_size * n_batches, N), replace=False)
    model.eval()
    cosines = []
    for b_start in range(0, len(sample_idx), batch_size):
        idx = sample_idx[b_start: b_start + batch_size]
        patts = []
        for i in idx:
            t = dataset[i]
            if isinstance(t, (list, tuple)):
                t = t[0]
            patts.append(t)
        x = torch.stack(patts, 0).to(device).float()
        x = eval_tf(x)
        # random per-sample k
        k = torch.randint(-shift_range // 2, shift_range // 2 + 1, (x.size(0),),
                          device=x.device)
        H = x.size(-2)
        row_idx = (torch.arange(H, device=x.device)[None, :] - k[:, None]) % H
        row_idx = row_idx[:, None, :, None].expand(-1, x.size(1), -1, x.size(-1))
        x_rolled = torch.gather(x, 2, row_idx)
        z1 = model.student_contrastive(model.student_encoder(x))
        z2 = model.student_contrastive(model.student_encoder(x_rolled))
        c = (z1 * z2).sum(-1)
        cosines.extend(c.cpu().numpy().tolist())
    return cosines


# =========================================================================
# 4. UMAP (falls back to PCA if umap-learn missing)
# =========================================================================

def umap_or_pca(embeds: np.ndarray, n_neighbors: int = 30,
                min_dist: float = 0.1, random_state: int = 0):
    """Try UMAP; if unavailable, 2-D PCA. Returns (coords, method_name)."""
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=random_state, n_components=2)
        coords = reducer.fit_transform(embeds)
        return coords, "UMAP"
    except Exception as exc:  # noqa: BLE001
        print(f"[umap_or_pca] UMAP failed ({exc!r}); falling back to PCA.")
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=random_state).fit_transform(embeds)
        return coords, "PCA"


# =========================================================================
# 5. FIGURES — matplotlib, all saved with bbox_inches='tight'
# =========================================================================

def _save_fig(fig, path: str):
    fig.savefig(path, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_loss_curves(training_log_csv: str, outpath: str):
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(training_log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    ax = axes[0]
    ax.plot(df.epoch, df.avg_loss_dino, label="L_DINO", lw=2)
    ax.plot(df.epoch, df.avg_loss_contrastive, label="L_contrastive", lw=2)
    ax.plot(df.epoch, df.avg_loss, label="L_total", lw=1.4, ls="--", color="k")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("training losses"); ax.legend(); ax.grid(alpha=0.3)
    ax2 = axes[1]
    ax2.plot(df.epoch, df.lambda_eff, lw=2, color="tab:orange")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("lambda_eff")
    ax2.set_title("contrastive lambda schedule"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, outpath)


def plot_training_dynamics(training_log_csv: str, outpath: str):
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(training_log_csv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df.epoch, df.effK, lw=2)
    axes[0].set_title("effective K (exp of prototype-usage entropy)")
    axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)
    axes[1].plot(df.epoch, df.teacher_ent_mean, lw=2, label="mean")
    axes[1].fill_between(df.epoch, df.teacher_ent_p10, df.teacher_ent_p90,
                          alpha=0.25, label="p10-p90")
    axes[1].set_title("teacher softmax entropy")
    axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(df.epoch, df.student_embed_norm_mean, lw=2)
    axes[2].fill_between(df.epoch,
                         df.student_embed_norm_mean - df.student_embed_norm_std,
                         df.student_embed_norm_mean + df.student_embed_norm_std,
                         alpha=0.25)
    axes[2].axhline(1.0, color="k", ls=":", lw=1, label="L2=1")
    axes[2].set_title("student embedding norm (expect 1.0)")
    axes[2].set_xlabel("epoch"); axes[2].legend(); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, outpath)


def plot_centroid_cosine(cos: np.ndarray, outpath: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title("prototype centroid cosine")
    ax.set_xticks(range(cos.shape[0])); ax.set_yticks(range(cos.shape[0]))
    for i in range(cos.shape[0]):
        for j in range(cos.shape[0]):
            ax.text(j, i, f"{cos[i,j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(cos[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_fig(fig, outpath)


def plot_prototype_usage(counts: np.ndarray, outpath: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    order = np.argsort(-counts)
    ax.bar(range(len(counts)), counts[order])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(o) for o in order])
    ax.set_xlabel("prototype (sorted by usage)")
    ax.set_ylabel("N samples")
    ax.set_title("prototype usage histogram")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save_fig(fig, outpath)


def plot_umap(embeds: np.ndarray, labels: np.ndarray, outpath: str):
    import matplotlib.pyplot as plt
    coords, method = umap_or_pca(embeds)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    K = int(labels.max()) + 1
    cmap = plt.get_cmap("tab10")
    for c in range(K):
        m = (labels == c)
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.55,
                   color=cmap(c % 10), label=f"p{c} (n={m.sum()})")
    ax.set_title(f"{method} of student contrastive embeddings")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, loc="best", framealpha=0.85)
    fig.tight_layout()
    _save_fig(fig, outpath)
    return method


def plot_representative_patterns(dataset, assigns: np.ndarray,
                                  soft_probs: np.ndarray,
                                  outpath: str, per_proto: int = 6):
    """For each prototype, display the top-N patterns by that prototype's
    softmax probability (not distance-to-centroid -- softmax prob is
    what the loss actually trained). Dead prototypes get blank rows."""
    import matplotlib.pyplot as plt
    K = soft_probs.shape[1]
    fig, axes = plt.subplots(K, per_proto, figsize=(per_proto * 1.6, K * 1.6))
    if K == 1:
        axes = np.array([axes])
    for c in range(K):
        mask = (assigns == c)
        if not mask.any():
            for j in range(per_proto):
                axes[c, j].set_axis_off()
            axes[c, 0].set_title(f"p{c} DEAD", fontsize=8, loc="left")
            continue
        idx = np.where(mask)[0]
        scores = soft_probs[idx, c]
        top = idx[np.argsort(-scores)[:per_proto]]
        for j in range(per_proto):
            axes[c, j].set_axis_off()
            if j >= len(top):
                continue
            t = dataset[int(top[j])]
            if isinstance(t, (list, tuple)):
                t = t[0]
            arr = t.cpu().numpy() if torch.is_tensor(t) else t
            if arr.ndim == 3:
                arr = arr[0]
            axes[c, j].imshow(arr, cmap="gray")
            if j == 0:
                axes[c, j].set_title(f"p{c} (n={int(mask.sum())})",
                                     fontsize=8, loc="left")
    fig.tight_layout()
    _save_fig(fig, outpath)


def plot_boundary_samples(dataset, teacher_probs: np.ndarray,
                           outpath: str, n: int = 16):
    """Highest-teacher-softmax-entropy samples. Entropy above the p98
    threshold — ambiguous, boundary, or artifact."""
    import matplotlib.pyplot as plt
    ent = -(teacher_probs * np.log(np.clip(teacher_probs, 1e-12, 1))).sum(1)
    top = np.argsort(-ent)[:n]
    cols = 4; rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.set_axis_off()
        if i >= len(top):
            continue
        idx = int(top[i])
        t = dataset[idx]
        if isinstance(t, (list, tuple)):
            t = t[0]
        arr = t.cpu().numpy() if torch.is_tensor(t) else t
        if arr.ndim == 3:
            arr = arr[0]
        ax.imshow(arr, cmap="gray")
        ax.set_title(f"i={idx} H={ent[idx]:.2f}", fontsize=7)
    fig.suptitle("highest teacher-softmax-entropy samples", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, outpath)


def plot_rotation_sanity(cosines: list, outpath: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(cosines, bins=30, color="tab:blue", alpha=0.8)
    med = float(np.median(cosines))
    ax.axvline(med, color="k", ls="--", lw=1.2, label=f"median={med:.3f}")
    ax.set_xlabel("cos(z(x), z(roll(x, k)))")
    ax.set_ylabel("count")
    ax.set_title("theta-roll invariance: embedding cosine distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, outpath)


def plot_class_map(assigns: np.ndarray, scan_shape, outpath: str,
                    max_dim_inches: float = 10.0, min_dim_inches: float = 3.5,
                    original_ids: list | None = None):
    """2-D prototype map on the scan grid.

    Class IDs are expected to be dense 0..K_active-1 (use `dense_remap_by_usage`
    upstream). The colormap is picked so the number of colors matches
    `K_active` without leaving holes when the source run had dead prototypes.

    `original_ids` (optional) is the new_id -> original_id list from
    `dense_remap_by_usage`. When supplied, the colorbar labels show
    `new_id (orig_id)` so identity can be cross-referenced back to the
    trained model without guessing.
    """
    import matplotlib.pyplot as plt
    m = assigns.reshape(scan_shape)
    K_active = int(assigns.max()) + 1
    Ny, Nx = scan_shape
    aspect = Nx / max(Ny, 1)                  # width / height
    if aspect >= 1:
        fig_w = max_dim_inches
        fig_h = max(min_dim_inches, max_dim_inches / aspect)
    else:
        fig_h = max_dim_inches
        fig_w = max(min_dim_inches, max_dim_inches * aspect)
    # Pick a cmap that has at least K_active distinct colors.
    if K_active <= 10:
        cmap, vmax = "tab10", 9
    elif K_active <= 20:
        cmap, vmax = "tab20", 19
    else:
        cmap, vmax = "turbo", max(K_active - 1, 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"prototype assignment on scan  (K_active={K_active}, "
                  f"shape={Ny}x{Nx})")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                         ticks=list(range(K_active)), shrink=0.85)
    if original_ids and len(original_ids) == K_active:
        cbar.ax.set_yticklabels(
            [f"{i} (orig p{o})" for i, o in enumerate(original_ids)], fontsize=7)
    fig.tight_layout()
    _save_fig(fig, outpath)


def plot_class_averages(dataset, soft_probs: np.ndarray, assigns: np.ndarray,
                          outpath: str, N_top: int = 200,
                          beam_mask_radius: int = 40,
                          pct_lo: float = 2.0, pct_hi: float = 99.5,
                          original_ids: list | None = None):
    """Per-prototype class average: mean / median / confidence-weighted mean
    raw Cartesian pattern of the top-N_top highest-confidence samples for
    each dense class. Beam-masked + percentile-clipped + log1p for display.

    Expects `assigns` already dense-remapped (0..K_active-1) and `soft_probs`
    column-aligned to match.
    """
    import matplotlib.pyplot as plt
    K_active = int(soft_probs.shape[1])
    counts = np.bincount(assigns, minlength=K_active)

    H0, W0 = dataset.H, dataset.W
    yy, xx = np.ogrid[:H0, :W0]
    cy, cx = H0 / 2.0, W0 / 2.0
    bmask = ((yy - cy) ** 2 + (xx - cx) ** 2) > beam_mask_radius ** 2

    def _disp(arr):
        masked = arr * bmask
        vals = masked[bmask]
        if vals.size == 0:
            return masked
        lo = np.percentile(vals, pct_lo)
        hi = np.percentile(vals, pct_hi)
        clipped = np.clip(masked, lo, hi)
        return np.log1p(clipped - lo)

    fig, axes = plt.subplots(K_active, 3, figsize=(6.5, K_active * 2.2))
    if K_active == 1:
        axes = axes[None, :]
    for j, t in enumerate(
            (f"mean (top {N_top})",
             f"median (top {N_top})",
             f"conf-weighted mean (top {N_top})")):
        axes[0, j].set_title(t, fontsize=9)

    for c in range(K_active):
        mask = (assigns == c)
        if not mask.any():
            for j in range(3):
                axes[c, j].set_axis_off()
            continue
        idx = np.where(mask)[0]
        scores = soft_probs[idx, c]
        top = idx[np.argsort(-scores)[:min(N_top, len(idx))]]
        patterns = np.stack([dataset.get_raw(int(i)) for i in top], 0)
        weights = soft_probs[top, c]
        mean = patterns.mean(axis=0)
        median = np.median(patterns, axis=0)
        wmean = (patterns * weights[:, None, None]).sum(axis=0) / (weights.sum() + 1e-12)
        for j, arr in enumerate((mean, median, wmean)):
            axes[c, j].imshow(_disp(arr), cmap="inferno")
            axes[c, j].set_xticks([]); axes[c, j].set_yticks([])
        orig_tag = (f" (orig p{original_ids[c]})"
                     if original_ids and c < len(original_ids) else "")
        axes[c, 0].set_ylabel(f"p{c}{orig_tag}\n(N={int(counts[c])})",
                               fontsize=7, rotation=0,
                               labelpad=26, ha="right", va="center")
    fig.suptitle(f"per-prototype class averages (raw Cartesian, beam-masked "
                  f"r<{beam_mask_radius}px, {pct_lo:.0f}–{pct_hi:.0f} pct clip, "
                  "log1p)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, outpath)


def plot_embedding_norms(embeds: np.ndarray, outpath: str):
    import matplotlib.pyplot as plt
    norms = np.linalg.norm(embeds, axis=1)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(norms, bins=40, color="tab:purple", alpha=0.8)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_title(f"student embedding norm (expect 1.0; mean={norms.mean():.3f})")
    ax.set_xlabel("||z||_2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, outpath)


def plot_pairwise_similarity(embeds: np.ndarray, outpath: str,
                              n: int = 2000, seed: int = 0):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    E = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12)
    idx = rng.choice(E.shape[0], size=min(n, E.shape[0]), replace=False)
    sub = E[idx]
    s = sub @ sub.T
    iu = np.triu_indices_from(s, k=1)
    vals = s[iu]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(vals, bins=60, color="tab:green", alpha=0.8)
    ax.set_title(f"held-out pairwise cos-sim (mean={vals.mean():.3f})")
    ax.set_xlabel("cos-sim")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, outpath)
