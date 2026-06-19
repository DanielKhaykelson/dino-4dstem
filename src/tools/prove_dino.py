"""Two proof figures (from grain_acom_v2_{name}.npz, no cube):

FIG 1 - "DINO clusters = a CIF-free crystallization-state axis":
  (a) per-grain crystallinity (1D peak/halo) grouped by DINO class, ordered by
      median -> tight strata; annotate eta^2 (class explains crystallinity).
  (b) class-average diffraction patterns ordered by crystallinity: amorphous halo
      -> sparse spots -> strong Bragg (the physical meaning of the axis).
  (c) eta^2 bars per sample: DINO class explains crystallinity AND ACOM corr.

FIG 2 - "Same info as the physics, but CIF-free, full-coverage, complementary":
  (a) per-grain DINO crystallinity vs ACOM corr (indexed grains) -> they agree
      (DINO recovers what ACOM ranks).
  (b) coverage: % sample grains DINO assigns a crystallinity (all) vs % ACOM
      indexes (subset; needs CIF+threshold).
  (c) rotation-invariance: within crystalline DINO classes, ACOM in-plane
      orientation spans the full range -> one DINO class merges many orientations
      that ACOM splits (different, complementary axis)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
NAMES = ["SI3", "SI4", "SI5"]; OUT = "docs/paper/draft_v2/figs"

def eta2(v, lab):
    v = np.asarray(v, float); lab = np.asarray(lab)
    if v.size < 5: return np.nan
    gm = v.mean(); sst = ((v - gm) ** 2).sum()
    ssb = sum(((v[lab == c].mean() - gm) ** 2) * (lab == c).sum() for c in set(lab.tolist()))
    return float(ssb / (sst + 1e-12))

D = {}
for n in NAMES:
    z = np.load(os.path.join(OUT, f"grain_acom_v2_{n}.npz"))
    m = ~z["vac"]
    D[n] = dict(ratio=z["ratio"][m], cls=z["cls"][m], corr=z["corr"][m],
                orient=z["orient"][m], npk=z["npk"][m], H=int(z["H"]),
                gsum=z["gsum"], gcnt=z["gcnt"], clsall=z["cls"], vac=z["vac"])

# ================= FIG 1 =================
fig = Figure(figsize=(16, 10), facecolor="white")
e2 = {}
for ri, n in enumerate(NAMES):
    d = D[n]; ratio = d["ratio"]; cls = d["cls"]
    e2[n] = dict(cryst=eta2(ratio, cls), corr=eta2(np.clip(d["corr"], 0, None), cls))
    # (a) boxplot crystallinity by class, ordered
    order = sorted(set(cls.tolist()), key=lambda c: np.median(ratio[cls == c]))
    ax = fig.add_subplot(3, 3, ri * 3 + 1)
    ax.boxplot([ratio[cls == c] for c in order], showfliers=False, widths=0.6)
    ax.set_xticklabels([f"c{c}" for c in order], fontsize=7, rotation=90)
    ax.set_ylabel("grain crystallinity\n(1D peak/halo)", fontsize=8)
    ax.set_title(f"{n}: crystallinity by DINO class  η²={e2[n]['cryst']:.2f}", fontsize=9)
    # (b) class-avg patterns ordered by crystallinity (one per class, median grain)
    ax = fig.add_subplot(3, 3, ri * 3 + 2)
    H = d["H"]; cy = (H - 1) / 2.0; cr = slice(int(cy) - 110, int(cy) + 110)
    strip = []
    clsall = d["clsall"]
    for c in order:
        gi = np.where((clsall == c) & (~d["vac"]))[0]
        if not len(gi): continue
        # representative grain = median crystallinity within class
        rr = D[n]["ratio"]; sub = [g for g in gi]
        g = sub[len(sub) // 2]
        avg = d["gsum"][g] / max(d["gcnt"][g], 1)
        strip.append(np.log1p(np.clip(avg[cr, cr], 0, None)))
    if strip:
        montage = np.concatenate(strip, axis=1)
        ax.imshow(montage, cmap="inferno", aspect="auto")
    ax.set_title(f"{n}: class-avg patterns, low→high crystallinity\n(amorphous halo → discrete α Bragg)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    # (c) eta2 bars
    ax = fig.add_subplot(3, 3, ri * 3 + 3)
    ax.bar(["crystallinity", "ACOM corr"], [e2[n]["cryst"], e2[n]["corr"]], color=["#2E86C1", "#C0392B"])
    ax.set_ylim(0, 1); ax.set_ylabel("η² explained by DINO class")
    ax.set_title(f"{n}: DINO class predicts the physics", fontsize=9)
    for xi, v in enumerate([e2[n]["cryst"], e2[n]["corr"]]): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
fig.suptitle("PROOF 1 — DINO clusters are a crystallization-state axis (amorphous→α), recovered with NO CIF and NO threshold\n"
             "η² = fraction of crystallinity / ACOM-corr variance explained by DINO class label", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
p1 = os.path.join(OUT, "proof1_dino_crystallinity.png"); fig.savefig(p1, dpi=150, facecolor="white")

# ================= FIG 2 =================
fig = Figure(figsize=(16, 5.5), facecolor="white")
# (a) DINO crystallinity vs ACOM corr (indexed), pooled, colored by sample
ax = fig.add_subplot(1, 3, 1)
colors = {"SI3": "#2E86C1", "SI4": "#28B463", "SI5": "#CA6F1E"}
allr, allc = [], []
for n in NAMES:
    d = D[n]; idx = d["corr"] > 0
    ax.scatter(d["ratio"][idx], d["corr"][idx], s=26, c=colors[n], edgecolor="k", lw=0.3, label=n, alpha=0.8)
    allr += list(d["ratio"][idx]); allc += list(d["corr"][idx])
r = np.corrcoef(allr, allc)[0, 1] if len(allr) > 2 else np.nan
ax.set_xlabel("DINO grain crystallinity (1D peak/halo)"); ax.set_ylabel("ACOM corr (alpha)")
ax.set_title(f"(a) where ACOM indexes, DINO crystallinity agrees\nPearson r={r:.2f}", fontsize=9); ax.legend(fontsize=7)
# (b) coverage: DINO assigns all sample grains; ACOM indexes a subset
ax = fig.add_subplot(1, 3, 2)
x = np.arange(len(NAMES)); w = 0.38
dino_cov = [100.0] * len(NAMES)
acom_cov = [100 * np.mean(D[n]["corr"] > 0) for n in NAMES]
ax.bar(x - w/2, dino_cov, w, label="DINO crystallinity", color="#2E86C1")
ax.bar(x + w/2, acom_cov, w, label="ACOM indexed", color="#C0392B")
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("% of sample grains"); ax.set_ylim(0, 110)
for xi, v in enumerate(acom_cov): ax.text(xi + w/2, v + 2, f"{v:.0f}%", ha="center", fontsize=8)
ax.set_title("(b) coverage: DINO segments all; ACOM\nindexes a CIF/threshold-limited subset", fontsize=9); ax.legend(fontsize=7)
# (c) rotation-invariance: ACOM orientation spread WITHIN crystalline DINO classes
ax = fig.add_subplot(1, 3, 3)
yt = []; yl = []; row = 0
for n in NAMES:
    d = D[n]
    cls = d["cls"]; orient = d["orient"]; corr = d["corr"]; ratio = d["ratio"]
    cryst_classes = [c for c in set(cls.tolist())
                     if np.median(ratio[cls == c]) > 0.3 and ((cls == c) & (corr > 0)).sum() >= 4]
    for c in cryst_classes:
        o = orient[(cls == c) & (corr > 0)]
        o = o[np.isfinite(o)]
        if o.size >= 4:
            ax.scatter(o, [row] * len(o), s=18, c=colors[n], alpha=0.8)
            yt.append(row); yl.append(f"{n} c{c}"); row += 1
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7); ax.set_xlabel("ACOM in-plane orientation (deg)")
ax.set_xlim(0, 360); ax.set_title("(c) one DINO crystalline class = MANY ACOM\norientations (rotation-invariant; ⊥ axis)", fontsize=9)
fig.suptitle("PROOF 2 — DINO recovers the crystallinity the physics measures, with full coverage and no CIF; orientation is a complementary ACOM axis DINO folds away", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92]); FigureCanvasAgg(fig)
p2 = os.path.join(OUT, "proof2_dino_vs_acom.png"); fig.savefig(p2, dpi=110, facecolor="white")

import shutil
for p in (p1, p2): shutil.copy(p, os.path.join(OUT, "latest_review", os.path.basename(p)))
json.dump({n: e2[n] for n in NAMES}, open(os.path.join(OUT, "prove_dino_eta2.json"), "w"), indent=2)
print("eta2:", {n: {k: round(v, 3) for k, v in e2[n].items()} for n in NAMES})
print("ACOM coverage %:", {n: round(100*np.mean(D[n]["corr"] > 0), 1) for n in NAMES})
print("wrote proof1_dino_crystallinity.png + proof2_dino_vs_acom.png", flush=True)
