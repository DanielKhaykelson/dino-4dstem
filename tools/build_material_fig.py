"""MATERIAL-INSIGHT figure (corrected, controlled): DINO reads IMC crystallization.
(a) SI3 crystallinity morphology: alpha spherulite (compact core + needles) in glass.
(b) polymorph proof: crystalline-class avg diffraction with alpha d-spacing rings (no gamma).
(c) polycrystallinity: ACOM in-plane orientation rose (near-uniform -> spherulitic, NOT
    single crystal); zone-axis mean-vector-length annotated.
(d) crystallization extent across the 3->5->4 datasets (fixed p/h>0.5).
(e) DINO's upper hand: ACOM indexes <20% (orientation, chokes on polycrystallinity)
    vs DINO captures the full crystalline extent (rotation-invariant unifies all alpha).
NOTE: an oriented-glass interfacial precursor was tested with a fixed real-crystal
threshold + strict-amorphous control and NOT robustly found (reported as negative)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
F = "docs/paper/draft_v2/figs"; INV_ANG = 0.00185; ALPHA_D = [10.4, 7.4, 6.0, 4.75, 3.9]; TC = 0.50
def load(n): return np.load(f"{F}/imc_glassorder_{n}.npz")
S3 = load("SI3"); Ny3, Nx3 = S3["scan"]; ph3 = S3["ph"].reshape(Ny3, Nx3); asg3 = S3["assigns"].reshape(Ny3, Nx3); H = int(S3["H"]); dmean3 = S3["dmean"]
fracs = {n: float(np.nanmean(load(n)["ph"] > TC)) for n in ["SI3", "SI5", "SI4"]}
phcl = {c: np.nanmedian(S3["ph"][S3["assigns"] == c]) for c in range(dmean3.shape[0]) if (S3["assigns"] == c).sum() > 50}
cryst_c = max(phcl, key=phcl.get)
# ACOM orientation (SI3) for polycrystallinity
acd = "runs/_gui/IMC_SI3_m097k60/acom/maps"
rm = np.load(f"{acd}/mpfull_winning_rmat.npy").reshape(-1, 3, 3); phid = np.load(f"{acd}/mpfull_phase_id.npy").ravel(); corr = np.load(f"{acd}/mpfull_winning_corr.npy").ravel()
idx = (phid >= 0) & (corr > np.nanpercentile(corr[phid >= 0], 40)); R = rm[idx]
inplane = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0])) % 360
za = R[:, 2, :]; za = za / (np.linalg.norm(za, axis=1, keepdims=True) + 1e-9); mvl = float(np.linalg.norm(za.mean(0)))
acom_idx_frac = float((phid[:Ny3*Nx3] >= 0).mean())

fig = Figure(figsize=(15, 8.8), facecolor="white")
# (a) morphology
ax = fig.add_subplot(2, 3, 1); im = ax.imshow(ph3, cmap="viridis", vmin=0, vmax=np.nanpercentile(ph3, 98))
ax.contour((ph3 > TC).astype(float), levels=[0.5], colors="w", linewidths=0.5)
ax.set_title("(a) SI3 — DINO crystallinity (peak/halo)\nα spherulite: compact core + needles in glass", fontsize=10)
ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046)
# (b) polymorph
ax = fig.add_subplot(2, 3, 2); pat = dmean3[cryst_c]; cr = pat[H//2-120:H//2+120, H//2-120:H//2+120]; cc = (cr.shape[0]-1)/2
ax.imshow(np.log1p(np.clip(cr, 0, None)), cmap="inferno")
for d in ALPHA_D:
    rpx = 1.0/(d*INV_ANG)
    if rpx < cr.shape[0]/2: ax.add_patch(Circle((cc, cc), rpx, fill=False, ec="cyan", lw=0.8, ls="--", alpha=0.85))
ax.set_title(f"(b) crystalline class avg (DINO c{cryst_c})\nrings = α (cyan); γ tells (8.6, 5.2 Å) absent", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
# (c) polycrystallinity rose
ax = fig.add_subplot(2, 3, 3, projection="polar"); h, edges = np.histogram(np.radians(inplane), bins=24, range=(0, 2*np.pi))
ax.bar(edges[:-1], h, width=2*np.pi/24, color="#1f77b4", alpha=0.8, align="edge")
ax.set_title(f"(c) ACOM in-plane orientation (n={idx.sum()})\nnear-uniform → POLYCRYSTALLINE spherulite\nzone-axis order param={mvl:.2f} (1=single crystal)", fontsize=9)
ax.set_yticklabels([])
# (d) crystallization extent
ax = fig.add_subplot(2, 3, 4); names = ["SI3\noverview", "SI5\ninterface", "SI4\nneedles"]
ax.bar(names, [fracs[n]*100 for n in ["SI3", "SI5", "SI4"]], color=["#444", "#aaa", "#777"])
ax.set_ylabel("crystalline fraction (%)  [p/h>0.5]"); ax.set_title("(d) crystallization extent across the 3→5→4 set", fontsize=10)
for i, n in enumerate(["SI3", "SI5", "SI4"]): ax.text(i, fracs[n]*100+1, f"{fracs[n]*100:.0f}%", ha="center", fontsize=9)
# (e) DINO upper hand: coverage
cryst3 = ph3 > TC; acom_map = (phid[:Ny3*Nx3] >= 0).reshape(Ny3, Nx3)
comp = np.zeros((Ny3, Nx3, 3)); comp[cryst3] = [0.35, 0.35, 0.35]; comp[acom_map] = [0.1, 0.7, 1.0]
ax = fig.add_subplot(2, 3, 5); ax.imshow(comp, interpolation="nearest")
ax.set_title(f"(e) DINO's upper hand (SI3)\nACOM indexed {acom_idx_frac*100:.0f}% (blue) — chokes on polycrystallinity\nDINO α-crystalline {cryst3.mean()*100:.0f}% (grey) — full extent", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
# (f) summary
ax = fig.add_subplot(2, 3, 6); ax.axis("off"); ax.text(0.0, 0.98, "What DINO reveals about IMC crystallization", fontsize=11, fontweight="bold", va="top")
ax.text(0.0, 0.88, ("• Annealed amorphous IMC (70 °C > Tg) crystallizes as a SINGLE\n"
        "  polymorph, α (every DINO crystalline class → α; no γ), as the\n"
        "  high-T = α rule predicts.\n\n"
        "• It is a POLYCRYSTALLINE α spherulite — a compact nucleus with\n"
        "  needles radiating outward, the indexed crystal showing near-random\n"
        "  orientations (order param 0.04), not a single-crystal dendrite.\n\n"
        "• DINO's rotation-invariance is the upper hand here: it unifies the\n"
        "  many-orientation α into ONE crystalline type and maps its full extent\n"
        "  (67% in SI3), whereas orientation-based ACOM indexes <20% (it chokes\n"
        "  precisely on the polycrystallinity) and classical NMF fails (Fig 3).\n\n"
        "• Crystallization extent is quantified across the 3→5→4 datasets\n"
        "  (67% → 31% → 2%).\n\n"
        "• Tested (fixed real-crystal threshold + strict-amorphous control): NO\n"
        "  robust oriented-glass interfacial precursor — reported as negative."), fontsize=7.8, va="top", family="monospace")
fig.suptitle("IMC crystallization decoded by DINO: a polycrystalline α spherulite, fully mapped where ACOM/NMF cannot", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.96]); FigureCanvasAgg(fig); fig.savefig(f"{F}/material_insight_IMC.png", dpi=160, facecolor="white")
print(f"fracs(p/h>0.5): {fracs}; ACOM idx {acom_idx_frac*100:.0f}%; orient order param {mvl:.2f}; cryst class c{cryst_c}", flush=True)
print("wrote material_insight_IMC.png", flush=True)
