"""
run_roi_split.py — Within-sample ROI-split transfer-learning validation.

Per initial_plan.txt #1: "Can we train on a small n of measurements (n>=1)
and then use them as transfer learning for the rest?"

Test within-sample ROI-split data:
  NaPHI family    (240312-NaPHI-Nadja-remeasure, 58mm CL, 115k mag, focused):
      SI-005, SI-006, SI-007, SI-008 = 4 quarters of the SAME ROI
  MgNaPHI family  (240312-MgNaPHI-remeasure,     58mm CL, 115k mag, focused):
      SI-003, SI-004, SI-005, SI-006 = 4 positions across SAME flake (img 1217)

NaPHI and MgNaPHI are treated SEPARATELY (no cross-sample training).

Approach (single-SI training, avoids multi-PRZ OOM):
  - Train winner config (config "c", 50ep) on ONE SI per family.
  - Evaluate that model on all 4 SIs of its family (same-SI + 3 transfer targets).
  - Measure cross-SI class-map consistency by Hungarian-matching:
      • self-pair  = eval-on-same-SI → reference
      • transfer   = eval-on-other-SI-j for j != train-SI
    For j != train-SI, Hungarian match the transfer map to the self map and
    report IoU / NMI / ARI / agreement. High numbers (e.g. >0.5 mean IoU,
    >0.4 NMI) across all 3 transfer targets → transfer works within-sample.

Output layout:
  runs/roi_split/NaPHI_Nadja/train1_SI5/best.pth
  runs/roi_split/NaPHI_Nadja/train1_SI5/eval_NaPHI_Nadja_SI005/eval/...
  runs/roi_split/NaPHI_Nadja/train1_SI5/eval_NaPHI_Nadja_SI006/eval/...
  runs/roi_split/NaPHI_Nadja/compare_self_vs_transfer.json
"""
from __future__ import annotations
import os, sys, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from run_contrastive import run_config, evaluate_and_report
from compare_maps import _iou_matrix, _hungarian_match, _nmi, _ari


# ── Winner config (matches Na007b sweep_polar_centroid/best.pth) ──────────────
WINNER_KWARGS = dict(
    epochs=50, seed=42, batch_size=128,
    lr=3e-4, weight_decay=1e-6,
    num_prototypes=6,
    t0=0.04, tfin=0.07,
    warmup_epochs=20, ramp_epochs=10,
    entropy_gate=False,
    projection_dim=128, projection_hidden=256,
    theta_shift_range=None,
    theta_shift_range_student=192, theta_shift_range_teacher=16,
    center_mask_radius=15, center_crop_size=140,
    vmax=2, polar_size=192, polar_mask_cols=30,
    pipeline="polar",
    centroid_lambda=0.05, centroid_margin=0.3,
    conf_weight_gamma=0.0,
    entropy_gate_override=None,
    lam_spatial=0.0,
    architecture="resnet", n_layers=1,
)


def train(train_sample: str, outdir: str, device) -> None:
    sentinel = os.path.join(outdir, "best.pth")
    if os.path.exists(sentinel):
        print(f"[skip] {outdir}  (best.pth exists)", flush=True)
        return
    os.makedirs(outdir, exist_ok=True)
    t0 = time.perf_counter()
    print(f"\n[{datetime.now():%H:%M:%S}] TRAIN {train_sample} -> {outdir}",
          flush=True)
    run_config("c", sample=train_sample, outdir=outdir, device=device, **WINNER_KWARGS)
    print(f"[{datetime.now():%H:%M:%S}] train done ({time.perf_counter()-t0:.0f}s)",
          flush=True)


def eval_on(eval_sample: str, ckpt_outdir: str, eval_outdir: str, device) -> None:
    sentinel = os.path.join(eval_outdir, "eval", "metrics.json")
    if os.path.exists(sentinel):
        print(f"[skip] {eval_outdir}  (metrics.json exists)", flush=True)
        return
    os.makedirs(eval_outdir, exist_ok=True)
    ckpt = os.path.join(ckpt_outdir, "best.pth")
    t0 = time.perf_counter()
    print(f"\n[{datetime.now():%H:%M:%S}] EVAL {eval_sample} ckpt={ckpt_outdir}",
          flush=True)
    evaluate_and_report("c", sample=eval_sample, outdir=eval_outdir,
                        device=device, ckpt_path=ckpt)
    print(f"[{datetime.now():%H:%M:%S}] eval done ({time.perf_counter()-t0:.0f}s)",
          flush=True)


def compare_inference(ref_npz: str, tgt_npz: str,
                       label_ref: str, label_tgt: str,
                       outpath: str,
                       scan_shape=(100, 100)) -> dict:
    """Hungarian-match `tgt` against `ref` and return metrics dict.

    Use-case: ref = model's class map on the training-SI (self),
              tgt = same model's class map on a different ROI SI (transfer).
    """
    import matplotlib.pyplot as plt
    ref_assigns = np.load(ref_npz)["assigns"]
    tgt_assigns = np.load(tgt_npz)["assigns"]
    Ka = int(ref_assigns.max()) + 1
    Kb = int(tgt_assigns.max()) + 1

    iou = _iou_matrix(ref_assigns, tgt_assigns, Ka, Kb)
    r, c = _hungarian_match(iou)
    matched = [(int(i), int(j), float(iou[i, j])) for i, j in zip(r, c)]

    relabel = np.full(Kb, -1, dtype=np.int64)
    for i, j, _ in matched:
        relabel[j] = i
    unmatched = [j for j in range(Kb) if relabel[j] < 0]
    for k_off, j in enumerate(unmatched):
        relabel[j] = Ka + k_off
    tgt_on_ref = relabel[tgt_assigns]
    # Agreement only makes sense when shapes match; both have same len here since
    # same scan dims. But agreement is pixel-identity — interpret as "if the
    # same vocabulary is used, do the SAME pixel locations get the same label?"
    # For a TRANSFER test across DIFFERENT scans this pixel-identity doesn't
    # carry physical meaning — only NMI and histogram-of-classes matter.
    if ref_assigns.shape == tgt_assigns.shape:
        agreement = float((ref_assigns == tgt_on_ref).mean())
    else:
        agreement = None
    nmi = _nmi(ref_assigns, tgt_assigns)
    ari = _ari(ref_assigns, tgt_assigns)

    # Per-class histogram comparison (does the model see similar class
    # distribution on same-flake different-ROI?)
    ref_counts = np.bincount(ref_assigns, minlength=max(Ka, Kb))
    tgt_counts_on_ref = np.bincount(tgt_on_ref, minlength=max(Ka, Kb))
    ref_frac = ref_counts / ref_counts.sum()
    tgt_frac = tgt_counts_on_ref / tgt_counts_on_ref.sum()
    l1_hist = float(np.abs(ref_frac - tgt_frac).sum())

    result = dict(
        label_ref=label_ref, label_tgt=label_tgt,
        K_ref=Ka, K_tgt=Kb,
        agreement_fraction=agreement, NMI=nmi, ARI=ari,
        mean_matched_iou=float(np.mean([p[2] for p in matched])) if matched else 0.0,
        matched_pairs=[dict(ref=p[0], tgt=p[1], iou=p[2]) for p in matched],
        unmatched_tgt_classes=unmatched,
        ref_class_fractions=[float(x) for x in ref_frac.tolist()],
        tgt_class_fractions_relabeled=[float(x) for x in tgt_frac.tolist()],
        hist_L1_distance=l1_hist,  # 0 = identical histogram, 2 = max
    )
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2, default=float)

    # Figure
    try:
        Ny, Nx = scan_shape
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        ref_map = ref_assigns.reshape(scan_shape)
        tgt_map_rel = tgt_on_ref.reshape(scan_shape)
        K_any = max(int(ref_map.max()), int(tgt_map_rel.max())) + 1
        cmap = "tab10" if K_any <= 10 else ("tab20" if K_any <= 20 else "turbo")
        axes[0].imshow(ref_map, cmap=cmap, vmin=0, vmax=max(9, K_any - 1),
                       interpolation="nearest")
        axes[0].set_title(f"{label_ref}  (K={Ka})", fontsize=9)
        axes[1].imshow(tgt_map_rel, cmap=cmap, vmin=0, vmax=max(9, K_any - 1),
                       interpolation="nearest")
        axes[1].set_title(f"{label_tgt}  Hung-relabeled  (K={Kb})", fontsize=9)
        axes[2].bar(range(max(Ka, Kb)),
                    ref_frac[:max(Ka, Kb)], alpha=0.55, label="ref", color="tab:blue")
        axes[2].bar(range(max(Ka, Kb)),
                    tgt_frac[:max(Ka, Kb)], alpha=0.55, label="tgt", color="tab:orange")
        axes[2].set_title(f"class-histogram  L1={l1_hist:.3f}  NMI={nmi:.3f}  "
                          f"meanIoU={result['mean_matched_iou']:.3f}", fontsize=9)
        axes[2].set_xlabel("class id (ref-aligned)")
        axes[2].set_ylabel("fraction")
        axes[2].legend(fontsize=8)
        for a in axes[:2]: a.set_axis_off()
        fig.suptitle(os.path.basename(outpath).replace(".json", ""), fontsize=10)
        fig.savefig(outpath.replace(".json", ".png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"  [compare_fig] {exc!r}")
    return result


def run_family(family_name: str,
               train_sample: str,
               test_samples: list[str],
               ref_test_sample: str,
               train_dirname: str,
               device) -> dict:
    """Train single-SI, eval on all test SIs, compare each transfer-SI
    against the reference (self) SI map."""
    base = os.path.join("runs", "roi_split", family_name)
    os.makedirs(base, exist_ok=True)

    train_dir = os.path.join(base, train_dirname)
    train(train_sample, train_dir, device)

    for tsi in test_samples:
        eval_on(tsi, train_dir, os.path.join(train_dir, f"eval_{tsi}"), device)

    # Compare: ref-SI map vs each transfer-SI map (same model, different data).
    comparisons = {}
    ref_npz = os.path.join(train_dir, f"eval_{ref_test_sample}", "eval", "inference.npz")
    for tsi in test_samples:
        if tsi == ref_test_sample:
            continue
        tgt_npz = os.path.join(train_dir, f"eval_{tsi}", "eval", "inference.npz")
        if not (os.path.exists(ref_npz) and os.path.exists(tgt_npz)):
            print(f"[cmp] missing npz: ref={os.path.exists(ref_npz)} "
                  f"tgt={os.path.exists(tgt_npz)} for tsi={tsi}")
            continue
        out = os.path.join(base, f"compare_{ref_test_sample}_vs_{tsi}.json")
        r = compare_inference(ref_npz, tgt_npz,
                              label_ref=f"ref={ref_test_sample}",
                              label_tgt=f"transfer={tsi}",
                              outpath=out)
        comparisons[tsi] = r
        mi = r["mean_matched_iou"]; nmi = r["NMI"]; l1 = r["hist_L1_distance"]
        print(f"  [{family_name}] {ref_test_sample} -> {tsi}: "
              f"meanIoU={mi:.3f}  NMI={nmi:.3f}  histL1={l1:.3f}",
              flush=True)

    # Also collect per-SI metrics (K_active, proto_counts) from each eval's
    # metrics.json to show cross-SI stability.
    per_si_metrics = {}
    for tsi in test_samples:
        mj = os.path.join(train_dir, f"eval_{tsi}", "eval", "metrics.json")
        if os.path.exists(mj):
            with open(mj) as f:
                content = f.read().replace("NaN", "null")
            m = json.loads(content)
            per_si_metrics[tsi] = {
                "K_active": m.get("K_active", m.get("active_prototypes")),
                "proto_counts": m.get("proto_counts"),
                "KNN_purity_k10": m.get("KNN_purity_k10"),
                "intra_over_inter": m.get("intra_over_inter"),
                "effective_K": m.get("effective_K"),
            }

    summary = {
        "family": family_name,
        "train_sample": train_sample,
        "ref_test_sample": ref_test_sample,
        "test_samples": test_samples,
        "per_si_metrics": per_si_metrics,
        "transfer_comparisons": comparisons,
    }
    out = os.path.join(base, "transfer_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[summary] {family_name} -> {out}")
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[roi_split] device={device}", flush=True)

    # NaPHI family (240312-NaPHI-Nadja-remeasure, 58mm CL, 115k mag, focused)
    # SI-005..008 = 4 quarters of SAME ROI.
    naphi = run_family(
        family_name="NaPHI_Nadja",
        train_sample="NaPHI_Nadja_SI005",
        test_samples=[f"NaPHI_Nadja_SI{i:03d}" for i in (5, 6, 7, 8)],
        ref_test_sample="NaPHI_Nadja_SI005",
        train_dirname="train1_SI5",
        device=device,
    )

    # MgNaPHI family (240312-MgNaPHI-remeasure, 58mm CL, 115k mag, focused)
    # SI-003..006 = 4 positions across SAME flake (img 1217).
    mgnaphi = run_family(
        family_name="MgNaPHI_remeas",
        train_sample="MgNaPHI_remeas_SI003",
        test_samples=[f"MgNaPHI_remeas_SI{i:03d}" for i in (3, 4, 5, 6)],
        ref_test_sample="MgNaPHI_remeas_SI003",
        train_dirname="train1_SI3",
        device=device,
    )

    # ── global summary ──
    print("\n" + "=" * 72)
    print("ROI-SPLIT TRANSFER SUMMARY (train-on-1 SI, eval on 4 ROIs of same sample)")
    print("=" * 72)
    for fam in (naphi, mgnaphi):
        print(f"\n[{fam['family']}]  train={fam['train_sample']}")
        print(f"  per-SI metrics:")
        for tsi, m in fam["per_si_metrics"].items():
            print(f"    {tsi}: K_active={m['K_active']}  "
                  f"effK={m['effective_K']}  "
                  f"KNNp={m['KNN_purity_k10']}")
        print(f"  transfer comparisons (ref={fam['ref_test_sample']}):")
        for tsi, c in fam["transfer_comparisons"].items():
            print(f"    -> {tsi}: meanIoU={c['mean_matched_iou']:.3f}  "
                  f"NMI={c['NMI']:.3f}  histL1={c['hist_L1_distance']:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
