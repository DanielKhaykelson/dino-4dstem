"""
analyze_class.py — focused per-class diagnostic.

Given (sample, config, class_id), answer:

  1. Where do its samples sit on the scan? Is the distribution spatially
     coherent, or scattered across multiple layers / regions?
  2. How confident was the teacher about each of them? (Softmax peakedness
     + second-best prototype.) High-entropy → boundary / ambiguous.
  3. Where are they in the contrastive embedding? Do they form a coherent
     cluster or are they sprinkled into the neighbor classes?
  4. What do their mean / representative patterns look like, compared to
     the neighbor classes?

Output: one figure `fig_classX_diagnostic.png` in the run's eval/ dir.

Usage:
    python analyze_class.py --sample EuInAs_B100 --config winner_polar_centroid --class-id 3
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import SAMPLES, LoadPRZ


def _beam_mask(H, W, radius):
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H / 2.0, W / 2.0
    return ((yy - cy) ** 2 + (xx - cx) ** 2) > radius ** 2


def _clip_log1p(arr, mask=None, lo_p=2.0, hi_p=99.5):
    ref = arr[mask] if mask is not None else arr.ravel()
    if ref.size == 0:
        return arr
    lo = np.percentile(ref, lo_p)
    hi = np.percentile(ref, hi_p)
    clipped = np.clip(arr, lo, hi)
    if mask is not None:
        clipped = clipped * mask
    return np.log1p(clipped - lo)


def _class_mean(dataset, idx_list):
    if len(idx_list) == 0:
        return None
    patterns = np.stack([dataset.get_raw(int(i)) for i in idx_list], 0)
    return patterns.mean(0)


def analyze(sample: str, config: str, class_id: int, top_n: int = 12,
             outpath: str | None = None, use_umap: bool = True):
    cfg = SAMPLES[sample]
    dataset = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    scan_shape = cfg["scan_shape"]
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "runs", sample, config)
    eval_dir = os.path.join(run_dir, "eval")
    inf = np.load(os.path.join(eval_dir, "inference.npz"))
    soft_probs = inf["soft_probs"]           # (N, K_active)
    t_probs = inf["teacher_probs"]           # (N, K_active)
    embeds = inf["embeds"]                   # (N, D)
    assigns = inf["assigns"]                 # (N,) dense labels
    # Reverse-lookup original prototype id for this dense class.
    K_original_ids = list(inf.get("K_original_ids", [])) or []
    orig_p = K_original_ids[class_id] if class_id < len(K_original_ids) else None

    K_active = soft_probs.shape[1]
    members = np.where(assigns == class_id)[0]
    others = np.where(assigns != class_id)[0]
    print(f"[analyze] sample={sample} config={config}  class={class_id}"
          + (f" (orig p{orig_p})" if orig_p is not None else ""))
    print(f"[analyze] {len(members)}/{len(assigns)} samples = "
          f"{100*len(members)/len(assigns):.1f}% of scan")
    if len(members) == 0:
        print("[analyze] class is empty; nothing to do.")
        return

    # ---------- Diagnostics ----------
    # 1. Spatial structure.
    Ny, Nx = scan_shape
    cls_map = (assigns == class_id).reshape(scan_shape)
    # Fraction in each scan row band (top/middle/bottom for layered samples).
    ys = np.where(cls_map)
    row_hist, _ = np.histogram(ys[0], bins=Ny, range=(0, Ny))
    # 2. Teacher confidence & entropy on class members.
    t_cls = t_probs[members]                               # (M, K)
    peaks = t_cls.max(axis=1)                              # top prob
    second = np.partition(t_cls, -2, axis=1)[:, -2]        # 2nd-best prob
    ent = -(t_cls * np.log(np.clip(t_cls, 1e-12, 1))).sum(1)
    # Top-2 alternative class distribution.
    top_alt_ids = t_cls.argsort(axis=1)[:, -2]             # second best dense id
    alt_hist = np.bincount(top_alt_ids, minlength=K_active)
    # 3. Embedding coherence.
    E = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12)
    E_cls = E[members]
    intra_cos = float((E_cls @ E_cls.T)[np.triu_indices(len(E_cls), k=1)].mean()
                       if len(E_cls) > 1 else float("nan"))
    # Cosine of each class-c sample to centroid of ALL classes
    centroids = np.stack([
        E[assigns == c].mean(0) if (assigns == c).any()
        else np.zeros(E.shape[1])
        for c in range(K_active)], 0)
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    cos_to_all = E_cls @ centroids.T                       # (M, K_active)
    # For each member, which centroid is NEAREST in embedding space?
    nearest_centroid = cos_to_all.argmax(axis=1)
    nearest_hist = np.bincount(nearest_centroid, minlength=K_active)
    # 4. Patterns: class mean + top-N high-confidence + high-entropy members
    scores = soft_probs[members, class_id]
    conf_top = members[np.argsort(-scores)[:top_n]]
    ent_top = members[np.argsort(-ent)[:top_n]]
    cls_mean = _class_mean(dataset, members)

    # Neighbor classes (top-2 alternatives by frequency) for comparison.
    alt_ranked = np.argsort(-alt_hist)
    alt_ranked = [int(a) for a in alt_ranked if a != class_id][:2]
    neighbor_means = {
        c: _class_mean(dataset, np.where(assigns == c)[0])
        for c in alt_ranked
    }

    # ---------- Figure ----------
    H0, W0 = dataset.H, dataset.W
    bm = _beam_mask(H0, W0, radius=40)

    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(5, 6, hspace=0.55, wspace=0.40)

    # Row 0 — class scan map + row histogram + nearest-centroid histogram
    ax = fig.add_subplot(gs[0, :3])
    ax.imshow(cls_map.astype(int), cmap="Greys_r", aspect="auto",
               vmin=0, vmax=1)
    ax.set_title(f"scan map: class {class_id}"
                 + (f" (orig p{orig_p})" if orig_p is not None else "")
                 + f"  — {len(members)} samples "
                 f"({100*len(members)/len(assigns):.1f}% of scan)",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    ax.barh(np.arange(Ny), row_hist, color="crimson")
    ax.invert_yaxis()
    ax.set_title("count per scan row\n(layered → bands)", fontsize=9)
    ax.set_xlabel("count"); ax.set_ylabel("row y")

    ax = fig.add_subplot(gs[0, 4:])
    bars = ax.bar(np.arange(K_active), nearest_hist,
                    color=["crimson" if c == class_id else "tab:blue"
                           for c in range(K_active)])
    ax.set_title(f"nearest centroid of each class-{class_id} sample "
                 f"in contrastive embedding", fontsize=9)
    ax.set_xlabel("dense class id"); ax.set_ylabel("count")
    ax.set_xticks(range(K_active))

    # Row 1 — teacher confidence histograms + entropy hist + alt-class hist
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(peaks, bins=30, color="tab:green", alpha=0.85)
    ax.set_title(f"teacher peak prob\nmedian={np.median(peaks):.2f}",
                  fontsize=9)
    ax.set_xlim(0, 1); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(second, bins=30, color="tab:orange", alpha=0.85)
    ax.set_title(f"teacher 2nd-best prob\nmedian={np.median(second):.2f}",
                  fontsize=9)
    ax.set_xlim(0, 1); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.hist(ent, bins=30, color="tab:purple", alpha=0.85)
    ax.set_title(f"teacher softmax entropy\nmedian={np.median(ent):.2f}",
                  fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 3])
    bars = ax.bar(np.arange(K_active), alt_hist,
                    color=["crimson" if c == class_id else "tab:gray"
                           for c in range(K_active)])
    ax.set_title("2nd-best class of each member", fontsize=9)
    ax.set_xlabel("dense class id"); ax.set_ylabel("count")
    ax.set_xticks(range(K_active))

    # Embedding coherence summary.
    ax = fig.add_subplot(gs[1, 4:])
    ax.axis("off")
    coherence_note = (
        f"intra-class cosine (members only, mean of pairwise):  {intra_cos:.3f}\n"
        f"fraction of members whose nearest centroid IS class {class_id}:  "
        f"{nearest_hist[class_id] / len(members):.2%}\n"
        f"most-common 2nd-best class:  "
        f"{alt_ranked[0] if alt_ranked else 'n/a'}"
        + (f" (orig p{K_original_ids[alt_ranked[0]]})"
           if alt_ranked and alt_ranked[0] < len(K_original_ids) else "")
        + "\n"
        + ("\nReading:\n"
           f"- intra-class cos near 1.0 → coherent phase\n"
           f"- intra-class cos well below neighbors' → misclassification / "
           f"high-entropy boundary cluster\n"
           f"- nearest-centroid usually != class_id → the contrastive head "
           f"disagrees with the DINO head assignment\n"
           f"- alt-class histogram concentrated on one neighbor → "
           f"2-way ambiguity (likely real boundary)\n"
           f"  spread across many → generic ambiguity (likely artifact)")
    )
    ax.text(0.0, 1.0, coherence_note, fontsize=9, va="top",
             family="monospace")

    # Row 2 — class mean | two neighbor class means, side by side
    def _show(ax_, arr, title):
        if arr is None:
            ax_.set_axis_off(); return
        ax_.imshow(_clip_log1p(arr, mask=bm), cmap="inferno")
        ax_.set_xticks([]); ax_.set_yticks([])
        ax_.set_title(title, fontsize=9)

    ax = fig.add_subplot(gs[2, 0:2])
    _show(ax, cls_mean, f"class {class_id} mean (N={len(members)})")
    for k, (alt_c, mean) in enumerate(neighbor_means.items()):
        ax = fig.add_subplot(gs[2, 2 + 2*k: 4 + 2*k])
        orig_alt = (K_original_ids[alt_c] if alt_c < len(K_original_ids) else None)
        tag = f" (orig p{orig_alt})" if orig_alt is not None else ""
        _show(ax, mean, f"class {alt_c}{tag} mean (neighbor)")

    # Row 3 — top-N highest-confidence members (the "prototypical" ones)
    for j, idx in enumerate(conf_top[:6]):
        ax = fig.add_subplot(gs[3, j])
        arr = dataset.get_raw(int(idx))
        ax.imshow(_clip_log1p(arr, mask=bm), cmap="inferno")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"hi-conf\ni={idx} p={soft_probs[idx, class_id]:.2f}",
                     fontsize=7)

    # Row 4 — top-N highest-entropy members (the "ambiguous" ones)
    for j, idx in enumerate(ent_top[:6]):
        ax = fig.add_subplot(gs[4, j])
        arr = dataset.get_raw(int(idx))
        ax.imshow(_clip_log1p(arr, mask=bm), cmap="inferno")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"hi-entropy\ni={idx} H={ent[np.where(members==idx)[0][0]]:.2f}",
                     fontsize=7)

    if outpath is None:
        outpath = os.path.join(eval_dir,
                                f"fig_class{class_id}_diagnostic.png")
    fig.suptitle(f"{sample} / {config}  —  class {class_id} "
                 + (f"(orig p{orig_p})" if orig_p is not None else "")
                 + " deep-dive diagnostic",
                 fontsize=11)
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] wrote {outpath}")
    return dict(
        class_id=class_id, orig_prototype_id=orig_p,
        n_members=int(len(members)),
        fraction_of_scan=float(len(members) / len(assigns)),
        intra_cos=intra_cos,
        frac_nearest_centroid_is_own_class=float(
            nearest_hist[class_id] / max(len(members), 1)),
        nearest_centroid_hist=[int(x) for x in nearest_hist],
        alt_class_hist=[int(x) for x in alt_hist],
        teacher_peak_median=float(np.median(peaks)),
        teacher_second_median=float(np.median(second)),
        teacher_entropy_median=float(np.median(ent)),
        row_occupancy=row_hist.tolist(),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--class-id", type=int, required=True)
    ap.add_argument("--top-n", type=int, default=12)
    args = ap.parse_args(argv)
    res = analyze(args.sample, args.config, args.class_id, top_n=args.top_n)
    import json
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
