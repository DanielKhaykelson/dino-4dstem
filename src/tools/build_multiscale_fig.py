"""Multi-scale 3->5->4 companion: crystallinity (peak/halo) maps with fixed-
threshold crystal contour at overview(SI3)/interface(SI5)/needle(SI4) scales, +
the crystallization-extent bar. (No precursor panel — that claim was tested and
not supported; see MATERIAL_INSIGHT_IMC.md.)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
F = "docs/paper/draft_v2/figs"; TC = 0.50
ORDER = [("SI3", "overview"), ("SI5", "interface"), ("SI4", "needles")]
fig = Figure(figsize=(13, 5.2), facecolor="white"); fr = {}
for i, (n, lab) in enumerate(ORDER):
    z = np.load(f"{F}/imc_glassorder_{n}.npz"); Ny, Nx = z["scan"]; ph = z["ph"].reshape(Ny, Nx)
    fr[n] = float(np.nanmean(ph > TC))
    ax = fig.add_subplot(1, 4, i + 1); ax.imshow(ph, cmap="viridis", vmin=0, vmax=np.nanpercentile(ph, 98))
    ax.contour((ph > TC).astype(float), levels=[0.5], colors="w", linewidths=0.5)
    ax.set_title(f"{n} — {lab}\ncrystalline {fr[n]*100:.0f}% (α)", fontsize=11); ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(1, 4, 4); ax.bar([n for n, _ in ORDER], [fr[n]*100 for n, _ in ORDER], color=["#444", "#aaa", "#777"])
ax.set_ylabel("crystalline fraction (%)  [p/h>0.5]"); ax.set_title("crystallization extent", fontsize=11)
for i, (n, _) in enumerate(ORDER): ax.text(i, fr[n]*100 + 1, f"{fr[n]*100:.0f}%", ha="center", fontsize=9)
fig.suptitle("IMC — DINO crystallinity across scales (overview → interface → needles): a polycrystalline α spherulite at varying crystallization extent", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93]); FigureCanvasAgg(fig); fig.savefig(f"{F}/material_multiscale_IMC.png", dpi=160, facecolor="white")
print("fixed-threshold crystalline fractions:", {n: round(fr[n], 3) for n, _ in ORDER}); print("wrote material_multiscale_IMC.png", flush=True)
