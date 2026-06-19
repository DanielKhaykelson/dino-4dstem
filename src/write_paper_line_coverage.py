"""write_paper_line_coverage.py -- consolidate manual line-coverage
labels into two CSVs ready for the paper:

    paper_line_coverage_per_sample.csv  (one row per sample)
    paper_line_coverage_summary.csv     (group-level stats + outlier gap)

Inputs:
    runs/_per_family_v5/NaPHI_combined_K8_30ep/line_labels_summary.csv
    runs/_per_family_v5/MgNaPHI_combined_K8_30ep/line_labels_summary.csv
"""
from __future__ import annotations
import csv, os, statistics

ROOT = os.path.join("runs", "_per_family_v5")
NAPHI_CSV   = os.path.join(ROOT, "NaPHI_combined_K8_30ep",   "line_labels_summary.csv")
MGNAPHI_CSV = os.path.join(ROOT, "MgNaPHI_combined_K8_30ep", "line_labels_summary.csv")

NAPHI_TRAIN  = {"NaPHI_Nadja_SI003", "NaPHI_Nadja_SI004"}
MGNAPHI_TRAIN = {"MgNaPHI_remeas_SI004", "MgNaPHI_remeas_SI011"}
OUTLIER_TRIO  = {"MgNaPHI_remeas_SI007",
                 "MgNaPHI_remeas_SI008",
                 "MgNaPHI_remeas_SI009"}


def _short(s):
    return s.replace("MgNaPHI_remeas_", "").replace("NaPHI_Nadja_", "")


# README-derived duplicate / repeat-group annotations.
SAMPLE_NOTES = {
    # NaPHI 4-quarter set (SI-005..SI-008 = 4 quarters of same ROI)
    "NaPHI_Nadja_SI005": "4-quarter set (same ROI as SI-006/007/008)",
    "NaPHI_Nadja_SI006": "4-quarter set (same ROI as SI-005/007/008)",
    "NaPHI_Nadja_SI007": "4-quarter set (same ROI as SI-005/006/008)",
    "NaPHI_Nadja_SI008": "4-quarter set (same ROI as SI-005/006/007)",
    # NaPHI same-flake-different-position pair
    "NaPHI_Nadja_SI009": "same flake as SI-010 (different scan position)",
    "NaPHI_Nadja_SI010": "same flake as SI-009 (different scan position; thinner area)",
    # MgNaPHI trio (predicted-low-Mg outlier + beam-damage remeasures)
    "MgNaPHI_remeas_SI007": "predicted-low-Mg outlier (EDS 1603)",
    "MgNaPHI_remeas_SI008": "remeasure of SI-007 area (beam-damage check)",
    "MgNaPHI_remeas_SI009": "remeasure of SI-008 area (beam-damage check)",
}


def _read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("sample"): continue
            rows.append({"sample": r["sample"],
                          "total": int(r["total"]),
                          "line_frames": float(r["line_frames"]),
                          "coverage": float(r["coverage"])})
    return rows


def _group_label(s):
    if s in OUTLIER_TRIO:                           return "MgNaPHI outlier trio"
    if s.startswith("MgNaPHI_remeas_"):              return "MgNaPHI bulk"
    if s.startswith("NaPHI_Nadja_"):                  return "NaPHI"
    return "?"


def _stats(values):
    if not values:
        return dict(n=0, mean=None, std=None, min=None, max=None)
    return dict(
        n=len(values),
        mean=statistics.fmean(values),
        std=(statistics.pstdev(values) if len(values) > 1 else 0.0),
        min=min(values), max=max(values),
    )


def main():
    naphi = _read(NAPHI_CSV)
    mgnaphi = _read(MGNAPHI_CSV)
    all_rows = naphi + mgnaphi

    # ---- per-sample CSV ----
    per_sample_path = os.path.join(ROOT, "paper_line_coverage_per_sample.csv")
    with open(per_sample_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "group", "sample", "is_train",
                     "total_patterns", "line_frames", "line_coverage",
                     "notes"])
        for r in all_rows:
            s = r["sample"]
            family = "NaPHI" if "Nadja" in s else "MgNaPHI"
            grp = _group_label(s)
            is_train = (s in NAPHI_TRAIN or s in MGNAPHI_TRAIN)
            note = SAMPLE_NOTES.get(s, "")
            w.writerow([family, grp, _short(s), "yes" if is_train else "no",
                         r["total"], f"{r['line_frames']:.0f}",
                         f"{r['coverage']:.4f}", note])
    print(f"wrote {per_sample_path}")

    # ---- group-level summary ----
    groups = {}
    for r in all_rows:
        groups.setdefault(_group_label(r["sample"]), []).append(r["coverage"])

    summary_path = os.path.join(ROOT, "paper_line_coverage_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "n", "mean_coverage", "std_coverage",
                     "min", "max"])
        # fixed order so paper table is consistent
        for g in ["NaPHI", "MgNaPHI bulk", "MgNaPHI outlier trio"]:
            s = _stats(groups.get(g, []))
            if s["n"] == 0:
                continue
            w.writerow([g, s["n"], f"{s['mean']:.4f}", f"{s['std']:.4f}",
                         f"{s['min']:.4f}", f"{s['max']:.4f}"])
        # outlier gap
        bulk = sorted(groups.get("MgNaPHI bulk", []))
        trio = sorted(groups.get("MgNaPHI outlier trio", []))
        if bulk and trio:
            w.writerow([])
            w.writerow(["outlier_gap_min", "highest_bulk", "lowest_trio",
                         "gap"])
            w.writerow(["", f"{bulk[-1]:.4f}", f"{trio[0]:.4f}",
                         f"{trio[0] - bulk[-1]:.4f}"])
    print(f"wrote {summary_path}")

    # ---- console summary ----
    print()
    print("==== summary ====")
    for g in ["NaPHI", "MgNaPHI bulk", "MgNaPHI outlier trio"]:
        s = _stats(groups.get(g, []))
        if s["n"] == 0: continue
        print(f"  {g:<25} n={s['n']:>2}  mean={s['mean']:.3f}  "
              f"std={s['std']:.3f}  range=[{s['min']:.3f}, {s['max']:.3f}]")
    bulk = sorted(groups.get("MgNaPHI bulk", []))
    trio = sorted(groups.get("MgNaPHI outlier trio", []))
    if bulk and trio:
        print(f"\n  outlier gap = {trio[0] - bulk[-1]:.3f} "
              f"(lowest trio {trio[0]:.3f} - highest bulk {bulk[-1]:.3f})")


if __name__ == "__main__":
    main()
