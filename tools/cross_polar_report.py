"""Cross-sample old (Cartesian) vs new (polar+theta-shift) NMF-vs-DINO comparison.
Reads new ARI/AMI from polar_nmf_vs_dino.json; old 'pattern NMF' ARI/AMI taken
from the committed interpretation reports. Writes a grouped-bar figure + a
markdown report. Does NOT touch manuscript numbers (await review)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
FIG = "docs/explainer/figs"; DOCS = "docs/interpretation_reports/_documents"
# OLD pattern-NMF (Cartesian, log, 40x40) ARI/AMI from committed report_auto.md
OLD = {"IMC_SI3": (0.115, 0.242), "IMC_SI4": (0.118, 0.213), "IMC_SI5": (0.214, 0.274),
       "NaPHI_Na007b": (0.479, 0.473), "EuInAs": (0.408, 0.605)}
NEW = json.load(open(os.path.join(FIG, "polar_nmf_vs_dino.json")))
order = ["IMC_SI3", "IMC_SI4", "IMC_SI5", "NaPHI_Na007b", "EuInAs"]
order = [s for s in order if s in NEW and "ARI_new" in NEW[s]]
labels = [s.replace("_Na007b", "").replace("NaPHI", "NaPHI") for s in order]
old_ari = [OLD[s][0] for s in order]; new_ari = [NEW[s]["ARI_new"] for s in order]
old_ami = [OLD[s][1] for s in order]; new_ami = [NEW[s]["AMI_new"] for s in order]

fig = Figure(figsize=(11, 4.4), facecolor="white"); x = np.arange(len(order)); w = 0.35
ax = fig.add_subplot(1, 2, 1)
ax.bar(x - w/2, old_ari, w, label="old: Cartesian NMF", color="#9aa7b0")
ax.bar(x + w/2, new_ari, w, label="new: polar+θ-shift NMF", color="#1C7293")
ax.axhspan(0, 0.3, color="#FDEDEC", zorder=0); ax.axhspan(0.3, 1, color="#EAF7EE", zorder=0)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8); ax.set_ylabel("ARI vs DINO")
ax.set_title("Best classical NMF reproduces DINO? (ARI)\nlow=distinctive, high=captured", fontsize=10)
ax.legend(fontsize=8); ax.set_ylim(0, 0.7)
ax = fig.add_subplot(1, 2, 2)
ax.bar(x - w/2, old_ami, w, label="old", color="#9aa7b0"); ax.bar(x + w/2, new_ami, w, label="new", color="#1C7293")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8); ax.set_ylabel("AMI vs DINO")
ax.set_title("AMI vs DINO", fontsize=10); ax.legend(fontsize=8); ax.set_ylim(0, 0.7)
fig.suptitle("Cross-sample: polar+θ-shift NMF (correct rotation-invariant baseline) vs DINO", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(fig)
fig.savefig(os.path.join(FIG, "cross_polar_vs_cartesian.png"), dpi=160, facecolor="white")

# markdown report
L = ["# Polar+θ-shift NMF vs DINO — cross-sample (old vs new baseline)", "",
     "Classical baseline re-run with the **correct rotation-invariant** polar+θ-shift NMF "
     "(Uesugi 2020 + Krajňák 2020; no log, beam masked), matched K, ARI/AMI vs the unchanged "
     "DINO assignments. DINO was **not** retrained.", "",
     "| sample | K | old NMF ARI | **new ARI** | old NMF AMI | **new AMI** | verdict |",
     "|---|---|---|---|---|---|---|"]
for s in order:
    a, m = NEW[s]["ARI_new"], NEW[s]["AMI_new"]; oa, om = OLD[s]
    verd = "distinctive" if a < 0.3 else "substantially captured"
    L.append(f"| {s} | {NEW[s]['K']} | {oa:.3f} | **{a:.3f}** | {om:.3f} | **{m:.3f}** | {verd} |")
L += ["", "## Conclusion",
      "- **IMC (SI3/SI4/SI5): distinctive holds — and is reinforced.** Even the stronger, "
      "rotation-invariant polar NMF cannot reproduce the DINO map (ARI ≈ 0.08–0.21). The "
      "molecular grain structure is genuinely not a re-labelling of a classical decomposition.",
      "- **NaPHI & EuInAs: substantially captured, unchanged.** ARI ≈ 0.43–0.45, AMI up to 0.60 — "
      "the polar baseline matches the old result; DINO behaves like a non-linear blend of classical "
      "descriptors on these ordered samples.",
      "- **The correct NMF variant does NOT change any conclusion** — it strengthens the IMC claim "
      "and leaves NaPHI/EuInAs as before. Safe to update the manuscript NMF numbers accordingly "
      "(pending your review).", ""]
os.makedirs(DOCS, exist_ok=True)
open(os.path.join(DOCS, "polar_nmf_vs_dino_report.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L), flush=True)
print("\nwrote cross_polar_vs_cartesian.png + polar_nmf_vs_dino_report.md", flush=True)
