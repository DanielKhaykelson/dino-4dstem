"""
run_roi_split_crosstrain.py — Correct "is transfer good?" experiment.

User's definition of good transfer (verbatim):
    "if when transferred result was similar a lot to the result of
     trained+evaled on the same sample, hence the IoU of eval+train vs
     transferred model."

Concretely for same-ROI quarters SI-i (i in 5,6,7,8 for NaPHI or 3,4,5,6
for MgNaPHI):
    native_map(j)       = map when model is TRAINED on SI-j and EVAL on SI-j
    transfer_map(i → j) = map when model is TRAINED on SI-i (i != j)
                          and EVAL on SI-j
    transfer_quality(i → j) = IoU(native_map(j), transfer_map(i → j))

Both maps cover the SAME pixels (SI-j), so pixel-wise Hungarian-matched IoU
is physically meaningful.  High IoU across all (i, j) pairs with i != j
means training on one SI alone is sufficient to predict other SIs.

Outputs a 4x4 IoU matrix per family, plus NMI/agreement per cell.

Prereq: NaPHI train1_SI5 and MgNaPHI train1_SI3 already exist.  This
script trains the MISSING per-SI models and runs the full 4x4 evaluation.
"""
from __future__ import annotations
# --- repo path bootstrap (added by reorg; keeps imports working) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, 'studies'), _os.path.join(_ROOT, 'scripts'), _os.path.join(_ROOT, 'tools')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end bootstrap ---
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
from run_roi_split import WINNER_KWARGS, compare_inference


def train(train_sample: str, outdir: str, device) -> None:
    sentinel = os.path.join(outdir, "best.pth")
    if os.path.exists(sentinel):
        print(f"[skip train] {outdir}", flush=True)
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
        print(f"[skip eval] {eval_outdir}", flush=True)
        return
    os.makedirs(eval_outdir, exist_ok=True)
    ckpt = os.path.join(ckpt_outdir, "best.pth")
    t0 = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] EVAL ckpt={os.path.basename(os.path.dirname(ckpt))} "
          f"-> {eval_sample}", flush=True)
    evaluate_and_report("c", sample=eval_sample, outdir=eval_outdir,
                        device=device, ckpt_path=ckpt)
    print(f"  done ({time.perf_counter()-t0:.0f}s)", flush=True)


def run_family(family_name: str,
               train_samples_by_si: dict,   # {si_number: sample_key}
               train_dirname_by_si: dict,   # {si_number: outdir_name}
               test_samples_by_si: dict,    # {si_number: sample_key_for_eval}
               device) -> dict:
    """For each SI, train a model; then evaluate EVERY model on EVERY SI,
    giving an N×N matrix of inference.npz files.  Compute IoU between
    native(j) and transfer(i→j) for every j and every i != j."""
    base = os.path.join("runs", "roi_split", family_name)
    os.makedirs(base, exist_ok=True)
    sis = sorted(train_samples_by_si.keys())

    # 1. Train.
    train_dirs = {}
    for si in sis:
        d = os.path.join(base, train_dirname_by_si[si])
        train_dirs[si] = d
        train(train_samples_by_si[si], d, device)

    # 2. Eval: each model on each SI.
    # eval_dirs[(trained_on_i, eval_on_j)] = dir.
    eval_dirs = {}
    for i in sis:
        for j in sis:
            d = os.path.join(train_dirs[i], f"eval_{test_samples_by_si[j]}")
            eval_dirs[(i, j)] = d
            eval_on(test_samples_by_si[j], train_dirs[i], d, device)

    # 3. Compare: native(j) vs transfer(i->j) for all j, i != j.
    # Also compute "native vs native" on another SI for sanity.
    matrix = {}  # (i, j) -> dict of metrics
    for j in sis:
        native_npz = os.path.join(eval_dirs[(j, j)], "eval", "inference.npz")
        if not os.path.exists(native_npz):
            print(f"[cmp] missing native npz for j={j}: {native_npz}")
            continue
        for i in sis:
            if i == j:
                matrix[(i, j)] = {"self": True}
                continue
            trans_npz = os.path.join(eval_dirs[(i, j)], "eval", "inference.npz")
            if not os.path.exists(trans_npz):
                print(f"[cmp] missing transfer npz for {i}->{j}: {trans_npz}")
                continue
            out = os.path.join(base,
                               f"compare_native_j{j:02d}_vs_transfer_i{i:02d}_j{j:02d}.json")
            r = compare_inference(native_npz, trans_npz,
                                  label_ref=f"native_j{j:02d}",
                                  label_tgt=f"transfer_i{i:02d}_j{j:02d}",
                                  outpath=out)
            matrix[(i, j)] = {
                "mean_matched_iou": r["mean_matched_iou"],
                "NMI": r["NMI"], "ARI": r["ARI"],
                "agreement_fraction": r.get("agreement_fraction"),
                "hist_L1_distance": r["hist_L1_distance"],
                "K_native": r["K_ref"], "K_transfer": r["K_tgt"],
            }
            mi = r["mean_matched_iou"]; nmi = r["NMI"]
            ag = r.get("agreement_fraction", 0.0)
            print(f"  [{family_name}] train_on_SI{i:02d} -> eval_on_SI{j:02d}: "
                  f"IoU={mi:.3f}  NMI={nmi:.3f}  agree={ag:.3f}",
                  flush=True)

    # 4. Summary.
    summary = {
        "family": family_name,
        "sis": sis,
        "train_samples_by_si": train_samples_by_si,
        "test_samples_by_si": test_samples_by_si,
        "matrix": {f"train{i:02d}_eval{j:02d}": v for (i, j), v in matrix.items()},
        "iou_matrix_flat": [
            [matrix.get((i, j), {}).get("mean_matched_iou", None) for j in sis]
            for i in sis
        ],
        "nmi_matrix_flat": [
            [matrix.get((i, j), {}).get("NMI", None) for j in sis]
            for i in sis
        ],
        "agreement_matrix_flat": [
            [matrix.get((i, j), {}).get("agreement_fraction", None) for j in sis]
            for i in sis
        ],
    }
    out = os.path.join(base, "crosstrain_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[summary] {out}", flush=True)

    # 5. IoU matrix figure.
    try:
        import matplotlib.pyplot as plt
        N = len(sis)
        iou_arr = np.full((N, N), np.nan, dtype=np.float32)
        for (i, j), r in matrix.items():
            if "self" in r:
                iou_arr[sis.index(i), sis.index(j)] = 1.0
            elif "mean_matched_iou" in r:
                iou_arr[sis.index(i), sis.index(j)] = r["mean_matched_iou"]
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        im = ax.imshow(iou_arr, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels([f"SI-{si:02d}" for si in sis])
        ax.set_yticklabels([f"SI-{si:02d}" for si in sis])
        ax.set_xlabel("eval on SI-j")
        ax.set_ylabel("trained on SI-i")
        ax.set_title(f"{family_name}: native(j) vs transfer(i→j) IoU\n"
                      f"(diagonal = 1.0 by definition)",
                      fontsize=10)
        for p in range(N):
            for q in range(N):
                v = iou_arr[p, q]
                if not np.isnan(v):
                    ax.text(q, p, f"{v:.2f}",
                            ha="center", va="center",
                            fontsize=9, color="white" if v < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Hungarian-matched IoU")
        fig_path = os.path.join(base, "fig_crosstrain_iou_matrix.png")
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[figure] {fig_path}", flush=True)
    except Exception as exc:
        print(f"  [iou_fig] {exc!r}")

    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[crosstrain] device={device}", flush=True)

    # NaPHI family (SI-005..008 = 4 quarters of same ROI)
    naphi = run_family(
        family_name="NaPHI_Nadja",
        train_samples_by_si={5: "NaPHI_Nadja_SI005",
                              6: "NaPHI_Nadja_SI006",
                              7: "NaPHI_Nadja_SI007",
                              8: "NaPHI_Nadja_SI008"},
        train_dirname_by_si={5: "train1_SI5", 6: "train1_SI6",
                              7: "train1_SI7", 8: "train1_SI8"},
        test_samples_by_si={5: "NaPHI_Nadja_SI005",
                             6: "NaPHI_Nadja_SI006",
                             7: "NaPHI_Nadja_SI007",
                             8: "NaPHI_Nadja_SI008"},
        device=device,
    )

    # MgNaPHI family (SI-003..006 = scanned across same flake img 1217)
    mgnaphi = run_family(
        family_name="MgNaPHI_remeas",
        train_samples_by_si={3: "MgNaPHI_remeas_SI003",
                              4: "MgNaPHI_remeas_SI004",
                              5: "MgNaPHI_remeas_SI005",
                              6: "MgNaPHI_remeas_SI006"},
        train_dirname_by_si={3: "train1_SI3", 4: "train1_SI4",
                              5: "train1_SI5", 6: "train1_SI6"},
        test_samples_by_si={3: "MgNaPHI_remeas_SI003",
                             4: "MgNaPHI_remeas_SI004",
                             5: "MgNaPHI_remeas_SI005",
                             6: "MgNaPHI_remeas_SI006"},
        device=device,
    )

    print("\n" + "=" * 72)
    print("CROSS-TRAIN IoU SUMMARY  (native[j] vs transfer[i->j])")
    print("=" * 72)
    for fam in (naphi, mgnaphi):
        sis = fam["sis"]
        print(f"\n[{fam['family']}] SIs: {sis}")
        print("      eval-> " + " ".join(f"SI-{j:02d}".rjust(7) for j in sis))
        for ri, i in enumerate(sis):
            row = fam["iou_matrix_flat"][ri]
            cells = []
            for v in row:
                if v is None:
                    cells.append("   nan")
                else:
                    cells.append(f"{v:7.3f}")
            print(f"train SI-{i:02d}: " + " ".join(cells))
    print("=" * 72)


if __name__ == "__main__":
    main()
