"""analyze_model_comparisons.py — collect metrics + class maps from
runs/ModelComparisons/ and build the 3 SI comparison figures + a summary table.
Run AFTER run_model_comparisons.py finishes.
  python src/analyze_model_comparisons.py
"""
from __future__ import annotations
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import image as mpimg

ROOT = os.path.join("runs", "ModelComparisons")
METRICS = ["effective_K", "K_active", "KNN_purity_k10", "intra_class_cosine",
           "inter_class_cosine", "intra_over_inter", "rotation_invariance_cos_median",
           "prototype_usage_entropy"]


def load(label, exp):
    f = os.path.join(ROOT, exp, label, "eval", "metrics.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); d["_classmap"] = os.path.join(ROOT, exp, label, "eval", "fig_class_map.png")
    return d


def montage(items, title, fname, ncol):
    n = len(items); nrow = int(np.ceil(n / ncol))
    fig = Figure(figsize=(3.1 * ncol, 3.3 * nrow + 0.5), facecolor="white")
    for i, (cap, path) in enumerate(items):
        ax = fig.add_subplot(nrow, ncol, i + 1); ax.set_xticks([]); ax.set_yticks([])
        try:
            ax.imshow(mpimg.imread(path))
        except Exception:
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center")
        ax.set_title(cap, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); FigureCanvasAgg(fig)
    p = os.path.join(ROOT, fname); fig.savefig(p, dpi=150, facecolor="white"); print("wrote", p)


rows = []
# ---- collect everything ----
ALL = {
    "01_vit_vs_resnet": ["vit_Na007b", "vit_IMC_SI5"],
    "02_resnet_depth": ["Na007b_L1", "Na007b_L2", "Na007b_L3", "Na007b_L4",
                         "IMC_SI5_L1", "IMC_SI5_L2", "IMC_SI5_L3", "IMC_SI5_L4"],
    "03_cluster1d": ["EuInAs_c1d_off", "EuInAs_c1d_on", "IMC_SI5_c1d_off"],
}
data = {}
for exp, labels in ALL.items():
    for lab in labels:
        d = load(lab, exp)
        data[lab] = d
        if d:
            rows.append({"exp": exp, "label": lab, **{m: d.get(m) for m in METRICS}})

# ---- CSV ----
with open(os.path.join(ROOT, "SUMMARY.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["exp", "label"] + METRICS); w.writeheader(); w.writerows(rows)
print("wrote", os.path.join(ROOT, "SUMMARY.csv"))

# ---- Fig 1: ViT vs ResNet(L2) class maps ----
m1 = []
for s, vit, res in [("Na007b", "vit_Na007b", "Na007b_L2"), ("IMC_SI5", "vit_IMC_SI5", "IMC_SI5_L2")]:
    if data.get(vit): m1.append((f"{s}  ViT", data[vit]["_classmap"]))
    if data.get(res): m1.append((f"{s}  ResNet-L2", data[res]["_classmap"]))
if m1: montage(m1, "ViT vs ResNet (n_layers=2) — DINO class maps", "fig_01_vit_vs_resnet.png", ncol=2)

# ---- Fig 2: depth L1-L4 ----
m2 = []
for s in ("Na007b", "IMC_SI5"):
    for L in (1, 2, 3, 4):
        lab = f"{s}_L{L}"
        if data.get(lab): m2.append((f"{s}  L{L}", data[lab]["_classmap"]))
if m2: montage(m2, "ResNet encoder depth L1–L4 — DINO class maps", "fig_02_resnet_depth.png", ncol=4)

# ---- Fig 3: cluster1d on/off ----
m3 = []
for s, off, on in [("EuInAs", "EuInAs_c1d_off", "EuInAs_c1d_on"),
                   ("IMC_SI5", "IMC_SI5_c1d_off", "IMC_SI5_L1")]:  # IMC on == depth L1
    if data.get(off): m3.append((f"{s}  cluster1d OFF", data[off]["_classmap"]))
    if data.get(on):  m3.append((f"{s}  cluster1d ON", data[on]["_classmap"]))
if m3: montage(m3, "1-D radial clustering loss OFF vs ON (EuInAs leakage)", "fig_03_cluster1d.png", ncol=2)

# ---- REPORT.md ----
def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)
with open(os.path.join(ROOT, "REPORT.md"), "w") as fh:
    fh.write("# Model comparisons (SI ablations)\n\nLatest-default recipe: DINO + theta-roll + cluster1d(0.1), K=60, 30 ep, m=0.97.\n\n")
    fh.write("| label | " + " | ".join(METRICS) + " |\n")
    fh.write("|" + "---|" * (len(METRICS) + 1) + "\n")
    for r in rows:
        fh.write(f"| {r['label']} | " + " | ".join(fmt(r.get(m)) for m in METRICS) + " |\n")
print("wrote", os.path.join(ROOT, "REPORT.md"))
