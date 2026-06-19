"""Within-class vs between-class crystallinity consistency for IMC.

A DINO 'class' (label) appears as many disconnected 'grains' (connected components)
in real space. Test: do the separate grains of the SAME class share a crystallinity
(p/h, ACOM) value that DISTINGUISHES them from another class?
  -> label connected components per class (>=MINPIX), per-grain mean p/h + ACOM corr.
  -> eta^2 = between-class variance / total variance of per-grain p/h
     (high -> class strongly predicts a grain's crystallinity = crystallinity strata).
Uses imc_glassorder_{name}.npz (per-pixel ph, assigns) + ACOM corr (SI3/SI4)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy import ndimage
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

OUT = "docs/paper/draft_v2/figs"; MINPIX = 40
ACOM = {"SI3": "runs/_gui/IMC_SI3_m097k60/acom/maps",
        "SI4": "runs/_gui/IMC_SI4_m097_k60/acom/maps"}
NAMES = ["SI3", "SI4", "SI5"]

def eta2(vals, labs):
    vals = np.asarray(vals); labs = np.asarray(labs); gm = vals.mean()
    sst = ((vals - gm) ** 2).sum()
    ssb = sum(((vals[labs == c].mean() - gm) ** 2) * (labs == c).sum() for c in set(labs))
    return float(ssb / (sst + 1e-12))

summary = {}
fig = Figure(figsize=(14, 10), facecolor="white")
for ri, name in enumerate(NAMES):
    z = np.load(os.path.join(OUT, f"imc_glassorder_{name}.npz"))
    ph = z["ph"]; asg = z["assigns"].astype(int); Ny, Nx = z["scan"]
    phmap = ph.reshape(Ny, Nx); asgmap = asg.reshape(Ny, Nx)
    corr = None
    if name in ACOM:
        c = np.load(os.path.join(ACOM[name], "mpfull_winning_corr.npy")).ravel()
        corr = c[:Ny * Nx].reshape(Ny, Nx)
    g_cls, g_ph, g_corr, g_n = [], [], [], []
    for k in sorted(set(asg)):
        lab, nlab = ndimage.label(asgmap == k)
        for gi in range(1, nlab + 1):
            mask = lab == gi
            if mask.sum() < MINPIX: continue
            vals = phmap[mask]
            g_cls.append(int(k)); g_ph.append(float(np.nanmean(vals))); g_n.append(int(mask.sum()))
            g_corr.append(float(np.nanmean(corr[mask])) if corr is not None else np.nan)
    g_cls = np.array(g_cls); g_ph = np.array(g_ph); g_corr = np.array(g_corr); g_n = np.array(g_n)
    e2 = eta2(g_ph, g_cls)
    e2c = eta2(g_corr[~np.isnan(g_corr)], g_cls[~np.isnan(g_corr)]) if corr is not None and (~np.isnan(g_corr)).sum() > 3 else None
    order = sorted(set(g_cls), key=lambda c: np.median(g_ph[g_cls == c]))
    summary[name] = dict(n_grains=int(len(g_cls)), n_classes=len(set(g_cls)),
                         eta2_ph=round(e2, 3), eta2_acom=None if e2c is None else round(e2c, 3),
                         per_class={int(c): dict(n_grains=int((g_cls == c).sum()),
                                                 ph_med=round(float(np.median(g_ph[g_cls == c])), 3),
                                                 ph_std=round(float(np.std(g_ph[g_cls == c])), 3)) for c in order})
    # --- plot: strip of per-grain p/h by class (col1), and ACOM (col2) ---
    ax = fig.add_subplot(3, 2, ri * 2 + 1)
    for xi, c in enumerate(order):
        v = g_ph[g_cls == c]; jit = (np.arange(len(v)) - len(v) / 2) * 0.0 + np.linspace(-0.15, 0.15, len(v))
        ax.scatter(xi + jit, v, s=np.clip(g_n[g_cls == c] / 8, 6, 90), alpha=0.7,
                   color="#C0392B" if np.median(v) > np.median(g_ph) else "#888", edgecolor="k", lw=0.3)
        ax.plot([xi - 0.25, xi + 0.25], [np.median(v)] * 2, "k-", lw=2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"c{c}" for c in order], fontsize=8)
    ax.set_ylabel("per-grain p/h"); ax.set_title(f"{name}: per-grain crystallinity by class  (η²={e2:.2f})\n"
                                                 f"each dot=one real-space grain; bar=class median", fontsize=10)
    ax = fig.add_subplot(3, 2, ri * 2 + 2)
    if corr is not None:
        for xi, c in enumerate(order):
            v = g_corr[g_cls == c]
            ax.scatter([xi] * len(v) + np.linspace(-0.15, 0.15, len(v)), v, s=30, alpha=0.7,
                       color="#2471A3", edgecolor="k", lw=0.3)
            ax.plot([xi - 0.25, xi + 0.25], [np.nanmedian(v)] * 2, "k-", lw=2)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([f"c{c}" for c in order], fontsize=8)
        ax.set_ylabel("per-grain ACOM corr"); ax.set_title(f"{name}: per-grain ACOM by class  (η²={e2c:.2f})", fontsize=10)
    else:
        ax.scatter(g_ph, g_n, c=g_cls, cmap="tab20", s=30, edgecolor="k", lw=0.3)
        ax.set_xlabel("per-grain p/h"); ax.set_ylabel("grain size (px)")
        ax.set_title(f"{name}: no ACOM — grain p/h vs size (color=class)", fontsize=10)
    print(f"[{name}] grains={len(g_cls)} classes={len(set(g_cls))} "
          f"eta2(p/h)={e2:.3f} eta2(ACOM)={e2c}", flush=True)

fig.suptitle("Are grains of the SAME DINO class consistent in crystallinity? (η² = between-class / total variance)\n"
             "high η² => class predicts a grain's crystallinity = DINO classes are crystallinity strata", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); FigureCanvasAgg(fig)
fig.savefig(os.path.join(OUT, "imc_class_grain_consistency.png"), dpi=150, facecolor="white")
json.dump(summary, open(os.path.join(OUT, "imc_class_grain_consistency.json"), "w"), indent=2)
print("\n==== eta^2 (fraction of per-grain crystallinity variance explained by DINO class) ====")
for n in NAMES: print(f"{n}: p/h η²={summary[n]['eta2_ph']}  ACOM η²={summary[n]['eta2_acom']}  ({summary[n]['n_grains']} grains)")
print("wrote imc_class_grain_consistency.png + .json", flush=True)
