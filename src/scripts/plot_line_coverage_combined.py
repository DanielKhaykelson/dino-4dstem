"""plot_line_coverage_combined.py -- single combined figure showing the
NaPHI test set (under Model N) and MgNaPHI test set (under Model M) on
one coverage axis. Highlights the SI-007/SI-008/SI-009 outlier trio.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

ROOT = os.path.join("runs", "_per_family")
J = json.load(open(os.path.join(ROOT, "line_coverage_within_family.json")))

rows = []
for r in J["naphi_test_results"] + J["mgnaphi_test_results"]:
    rows.append(r)

# Two rows only: NaPHI on top, MgNaPHI on bottom. No "bulk"/"4Q"/etc.
# subgroup labels -- just plot every sample by its name.
rng = np.random.default_rng(0)

NAPHI_COLOR = "#1f77b4"
MGNAPHI_COLOR = "#ff7f0e"
OUTLIER_COLOR = "#d62728"

fig, ax = plt.subplots(figsize=(13, 4))
for r in rows:
    sample = r["sample"]
    if sample.startswith("NaPHI_"):
        y = 1
        color = NAPHI_COLOR
    else:
        y = 0
        color = OUTLIER_COLOR if sample.endswith(("SI007", "SI008", "SI009")) \
                else MGNAPHI_COLOR
    y_jit = y + (rng.random() - 0.5) * 0.18
    ax.scatter([r["coverage"]], [y_jit], s=200, color=color,
                edgecolors="black", linewidths=0.7, zorder=3)
    short = sample.replace("_remeas_", " ").replace("_Nadja_", " ")
    ax.annotate(short, (r["coverage"], y_jit), xytext=(8, 0),
                 textcoords="offset points", fontsize=9, va="center")

# gap annotation between MgNaPHI bulk max and outlier trio min
mg_outlier_cov = [r["coverage"] for r in rows
                   if r["sample"].endswith(("SI007", "SI008", "SI009"))
                   and r["sample"].startswith("MgNaPHI_")]
mg_other_cov = [r["coverage"] for r in rows
                 if not r["sample"].endswith(("SI007", "SI008", "SI009"))
                 and r["sample"].startswith("MgNaPHI_")]
if mg_outlier_cov and mg_other_cov:
    gap_lo = max(mg_other_cov); gap_hi = min(mg_outlier_cov)
    ax.axvspan(gap_lo, gap_hi, color="grey", alpha=0.10, zorder=0)
    ax.text((gap_lo + gap_hi) / 2, -0.6,
             f"gap {gap_lo:.3f} -> {gap_hi:.3f}",
             ha="center", va="bottom", fontsize=9, color="dimgrey",
             style="italic")

ax.set_yticks([0, 1])
ax.set_yticklabels(["MgNaPHI", "NaPHI"], fontsize=11)
ax.set_xlabel("line-phase coverage = (# patterns in line prototypes) / total",
              fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.7, 1.5)
ax.set_title(
    "Line-phase coverage  (Model N -> NaPHI test, Model M -> MgNaPHI test, K=6 each)",
    fontsize=11)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
out = os.path.join(ROOT, "fig_line_coverage_combined.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {out}")
