"""compute_line_coverage_within_family.py

Each family has its own model, evaluates only its own test samples.

  Model N (NaPHI):    train on SI-003+004,    eval on SI-005..010
  Model M (MgNaPHI):  train on SI-004+011,    eval on SI-001,003,005,006,
                                                       007,008,009,010

Coverage = fraction of test patterns that land in the user-picked LINE
prototype(s) of that family's model.

Outlier story: within MgNaPHI, the bulk (SI-001/003/005/006/010) should
have low line coverage, while SI-007/008/009 (same physical area, three
remeasures) should be conspicuously higher and cluster tightly.

Usage:
    1. After per-family pipeline finishes, view
        runs/_per_family/train_NaPHI/eval/class_averages/p*.png
        runs/_per_family/train_MgNaPHI/eval/class_averages/p*.png
    2. Run with --naphi-line and --mgnaphi-line specifying the picked
        prototype indices, e.g.
        python compute_line_coverage_within_family.py \
            --naphi-line 0 2 --mgnaphi-line 4
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
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from data import SAMPLES, LoadPRZ
from dino_sr_contrastive_model import load_contrastive_checkpoint
from contrastive_eval import infer_scan

OUT_ROOT = os.path.join("runs", "_per_family")
SRC_NAPHI_DIR = os.path.join(OUT_ROOT, "NaPHI_combined_K6_30ep")
SRC_MGNAPHI_DIR = os.path.join(OUT_ROOT, "MgNaPHI_combined_K6_30ep")
# Per-test-sample eval results land at:
#   runs/_per_family/NaPHI_combined_K6_30ep/transfer/<test_sample>/inference.npz
#   runs/_per_family/MgNaPHI_combined_K6_30ep/transfer/<test_sample>/inference.npz

# Test sets include training samples (unsupervised -- no leakage), marked
# with "(train)" so they're visible on the plot but not mistaken for held-out.
NAPHI_TEST = [
    ("NaPHI_Nadja_SI003", "NaPHI bulk (train)"),
    ("NaPHI_Nadja_SI004", "NaPHI bulk (train)"),
    ("NaPHI_Nadja_SI009", "NaPHI bulk"),
    ("NaPHI_Nadja_SI010", "NaPHI bulk"),
    ("NaPHI_Nadja_SI005", "NaPHI 4Q"),
    ("NaPHI_Nadja_SI006", "NaPHI 4Q"),
    ("NaPHI_Nadja_SI007", "NaPHI 4Q"),
    ("NaPHI_Nadja_SI008", "NaPHI 4Q"),
]

MGNAPHI_TEST = [
    ("MgNaPHI_remeas_SI004", "MgNaPHI bulk (train)"),
    ("MgNaPHI_remeas_SI011", "MgNaPHI bulk (train)"),
    ("MgNaPHI_remeas_SI001", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI003", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI005", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI006", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI010", "MgNaPHI bulk"),
    ("MgNaPHI_remeas_SI007", "MgNaPHI SI-007 (outlier)"),
    ("MgNaPHI_remeas_SI008", "MgNaPHI SI-008 (=SI-007 area)"),
    ("MgNaPHI_remeas_SI009", "MgNaPHI SI-009 (=SI-007 area)"),
]


def _eval_assigns(model, sample, device, src_dir):
    """Run inference of `model` on `sample`, save inference.npz +
    class_averages + 200 examples per class to
    src_dir/transfer/<sample>/eval/.
    Returns (assigns, soft_probs)."""
    cfg = SAMPLES[sample]
    ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    inf = infer_scan(model, ds, device, dense_remap=False,
                      polar_size=192, polar_mask_cols=45,
                      center_crop_size=140,
                      com_centering=True, center_mask_radius=15,
                      eval_temp=0.06, batch_size=128)
    transfer_root = os.path.join(src_dir, "transfer", sample)
    eval_dir = os.path.join(transfer_root, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    np.savez(os.path.join(eval_dir, "inference.npz"),
              soft_probs=inf["soft_probs"], assigns=inf["assigns"],
              embeds=inf["embeds"])
    # minimal run_summary so viz_paper_outputs picks up the right mask/crop
    with open(os.path.join(transfer_root, "run_summary.json"), "w") as f:
        json.dump({"cfg": {
            "center_mask_radius": 15, "polar_mask_cols": 45,
            "polar_size": 192, "center_crop_size": 140,
        }}, f)
    # class averages + 200 examples per class
    try:
        import viz_paper_outputs
        viz_paper_outputs.render_class_map(transfer_root, sample)
        viz_paper_outputs.render_class_averages_and_examples(
            transfer_root, sample, n_examples=200)
    except Exception as e:
        print(f"  [warn] paper-outputs failed for {sample}: {e!r}",
              flush=True)
    return inf["assigns"], inf["soft_probs"]


def _coverage(model, samples, line_protos, device, src_dir, family_name):
    rows = []
    for sample, family in samples:
        if sample not in SAMPLES or not os.path.exists(SAMPLES[sample]["path"]):
            print(f"  [SKIP] {sample}", flush=True); continue
        try:
            assigns, _ = _eval_assigns(model, sample, device, src_dir)
            total = int(len(assigns))
            line_count = int(np.isin(assigns, line_protos).sum())
            cov = line_count / max(total, 1)
            counts = np.bincount(assigns, minlength=6).tolist()
            rows.append({
                "sample": sample, "family": family,
                "source_model": os.path.basename(src_dir),
                "total": total, "line_count": line_count,
                "coverage": float(cov), "counts": counts,
            })
            print(f"  {sample:<32} {family:<32}  cov={cov:.4f}  "
                  f"line/total={line_count}/{total}  counts={counts}",
                  flush=True)
        except Exception as e:
            print(f"  [FAIL] {sample}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
    return rows


def _plot_family(rows, line_protos, model_label, out_png):
    fams = list({r["family"] for r in rows})
    fam_color = {f: plt.get_cmap("tab10").colors[i % 10] for i, f in enumerate(fams)}
    rows = sorted(rows, key=lambda r: r["coverage"])
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 0.8 * len(fams) + 2))
    fam_y = {f: i for i, f in enumerate(fams)}
    for r in rows:
        y = fam_y[r["family"]] + (rng.random() - 0.5) * 0.2
        ax.scatter([r["coverage"]], [y], s=160,
                    color=fam_color[r["family"]],
                    edgecolors="black", linewidths=0.6, zorder=3)
        short = r["sample"].replace("_remeas_", "_").replace("_Nadja_", "_")
        ax.annotate(short, (r["coverage"], y), xytext=(6, 0),
                     textcoords="offset points", fontsize=9, va="center")
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(f"line coverage = (# patterns in line protos {line_protos}) / total")
    ax.set_title(f"{model_label} test samples — line prototype membership "
                  f"(line protos: {line_protos})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--naphi-line", type=int, nargs="+", required=True,
                    help="prototype indices in Model N that are line-like, "
                         "e.g. --naphi-line 0 2")
    ap.add_argument("--mgnaphi-line", type=int, nargs="+", required=True,
                    help="prototype indices in Model M that are line-like, "
                         "e.g. --mgnaphi-line 4")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[line-coverage within-family] device={device}", flush=True)
    print(f"  Model N line protos: {args.naphi_line}", flush=True)
    print(f"  Model M line protos: {args.mgnaphi_line}", flush=True)

    # Load source models
    print(f"\n[load] {SRC_NAPHI_DIR}", flush=True)
    model_N, _, _, _ = load_contrastive_checkpoint(
        os.path.join(SRC_NAPHI_DIR, "best.pth"), device=device)
    model_N.eval()
    print(f"[load] {SRC_MGNAPHI_DIR}", flush=True)
    model_M, _, _, _ = load_contrastive_checkpoint(
        os.path.join(SRC_MGNAPHI_DIR, "best.pth"), device=device)
    model_M.eval()

    print(f"\n[NaPHI test set, source = NaPHI_combined_K6_30ep]", flush=True)
    rows_N = _coverage(model_N, NAPHI_TEST, np.array(args.naphi_line),
                        device, SRC_NAPHI_DIR, "NaPHI")
    print(f"\n[MgNaPHI test set, source = MgNaPHI_combined_K6_30ep]", flush=True)
    rows_M = _coverage(model_M, MGNAPHI_TEST, np.array(args.mgnaphi_line),
                        device, SRC_MGNAPHI_DIR, "MgNaPHI")

    # Plot per family -- saved alongside their source model
    _plot_family(rows_N, args.naphi_line,
                  "NaPHI_combined_K6_30ep -> NaPHI test set",
                  os.path.join(SRC_NAPHI_DIR, "fig_line_coverage.png"))
    _plot_family(rows_M, args.mgnaphi_line,
                  "MgNaPHI_combined_K6_30ep -> MgNaPHI test set",
                  os.path.join(SRC_MGNAPHI_DIR, "fig_line_coverage.png"))

    summary = {
        "source_NaPHI": "NaPHI_combined_K6_30ep",
        "source_MgNaPHI": "MgNaPHI_combined_K6_30ep",
        "naphi_line_protos": args.naphi_line,
        "mgnaphi_line_protos": args.mgnaphi_line,
        "naphi_test_results": rows_N,
        "mgnaphi_test_results": rows_M,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT_ROOT, "line_coverage_within_family.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[done] wrote {OUT_ROOT}/line_coverage_within_family.json", flush=True)
    print(f"  per-test inference at:", flush=True)
    print(f"    {SRC_NAPHI_DIR}/transfer/<test_sample>/eval/inference.npz",
          flush=True)
    print(f"    {SRC_MGNAPHI_DIR}/transfer/<test_sample>/eval/inference.npz",
          flush=True)


if __name__ == "__main__":
    main()
