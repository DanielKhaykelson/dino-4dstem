"""Build the cross-sample comparison figure (cross_all.png) from each run's
result JSONs. Re-run after adding/updating samples."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(__file__), "figs", "cross_all.png")
SAMPLES = [
    ("IMC SI4",    "runs/_gui/IMC_SI4_m097_k60"),
    ("IMC SI3",    "runs/_gui/IMC_SI3_m097k60"),
    ("IMC SI3·log","runs/_gui/IMC_SI3_m097k60_logstretch"),
    ("IMC SI5",    "runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60"),
    ("NaPHI",      "runs/_gui/Na007b_k60_m097_vmax2"),
    ("EuInAs",     "runs/_sweep_m_K_20260525_213539/EuInAs_B100/stage2/m0.9700_seed42_K60"),
]


def L(run, fn):
    p = os.path.join(ROOT, run, "_interpretability", fn)
    return json.load(open(p)) if os.path.exists(p) else None


labels, best_ari, best_ami, sn, ql, qh = [], [], [], [], [], []
for name, run in SAMPLES:
    cl = L(run, "classical_baselines.json")
    ab = L(run, "test2_4_ablations.json") or {}
    if cl is None:
        continue
    labels.append(name)
    best_ari.append(cl["best_ARI"])
    best_ami.append(max(v["AMI"] for v in cl["methods"].values()))
    s = ab.get("scattered_norm")
    if s is None:                                   # IMC_SI5: separate file
        t4 = L(run, "test4_scattered_norm.json") or {}
        s = t4.get("scattered_norm", {"ARI_vs_orig": np.nan})
    sn.append(s["ARI_vs_orig"])
    ql.append(ab.get("qmask_low", {}).get("ARI_vs_orig", np.nan))
    qh.append(ab.get("qmask_high", {}).get("ARI_vs_orig", np.nan))

x = np.arange(len(labels)); w = 0.38
fig = Figure(figsize=(14, 3.8))
ax = fig.add_subplot(1, 3, 1)
ax.bar(x - w / 2, best_ari, w, label="best ARI", color="#065A82")
ax.bar(x + w / 2, best_ami, w, label="best AMI", color="#E8A33D")
ax.axhline(0.5, ls="--", color="grey", lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("agreement with DINO")
ax.set_title("(a) Can classical methods reproduce it?")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
ax2 = fig.add_subplot(1, 3, 2)
ax2.bar(x, sn, 0.6, color="#1C7293")
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
ax2.set_ylim(0, 1); ax2.set_ylabel("ARI after removing scattered intensity")
ax2.set_title("(b) Intensity dependence\n(low = strongly depends)")
ax2.grid(axis="y", alpha=0.3)
ax3 = fig.add_subplot(1, 3, 3)
ax3.bar(x - w / 2, ql, w, label="mask low-q", color="#21295C")
ax3.bar(x + w / 2, qh, w, label="mask high-q", color="#97BC62")
ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
ax3.set_ylim(0, 1); ax3.set_ylabel("ARI after masking band")
ax3.set_title("(c) Low-q vs high-q\n(low = that band matters)")
ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)
fig.tight_layout()
FigureCanvasAgg(fig)
fig.savefig(OUT, dpi=190, facecolor="white", bbox_inches="tight")
print("wrote", OUT, "with", len(labels), "samples:", labels)
