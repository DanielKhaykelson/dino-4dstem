"""interpret_imc.py — what physical factors does the DINO IMC_SI5 model
cluster on?  Tests 1 (probe/MI ranking) + 5 (class signatures).

Reuses factor maps already on disk:
  * DINO   : <run>/eval/inference.npz  (embeds, assigns)
  * ACOM   : <run>/_imc_report/05_full_{phase_id,winning_corr}.npy,
             05_n_peaks_per_position.npy
  * crystal: newest <run>/crystallinity/report_all_*/  (peakhalo/azimvar
             per window → max-over-windows "strength"; _scattered_intensity)

Outputs → <run>/_interpretability/.
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (adjusted_rand_score,
                              normalized_mutual_info_score,
                              adjusted_mutual_info_score, mutual_info_score)

RUN = sys.argv[1] if len(sys.argv) > 1 else (
    r"runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/m0.9700_seed42_K60")
OUT = os.path.join(RUN, "_interpretability")
os.makedirs(OUT, exist_ok=True)
NY = NX = 128


def _flat(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(-1)


def eta_squared(labels, x):
    """Correlation ratio η² (0..1): fraction of x's variance explained by
    the categorical partition `labels`.  High = the clustering separates x."""
    m = np.isfinite(x)
    x = x[m]; lab = labels[m]
    if x.size < 10:
        return np.nan
    gm = x.mean()
    ss_tot = ((x - gm) ** 2).sum()
    if ss_tot <= 0:
        return np.nan
    ss_between = 0.0
    for c in np.unique(lab):
        xc = x[lab == c]
        if xc.size:
            ss_between += xc.size * (xc.mean() - gm) ** 2
    return float(ss_between / ss_tot)


def mi_cont(labels, x, bins=24):
    m = np.isfinite(x)
    x = x[m]; lab = labels[m]
    if x.size < 10:
        return np.nan
    xb = np.digitize(x, np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    return float(mutual_info_score(lab, xb))


def probe_continuous(emb, x):
    """5-fold CV R² of Ridge(embedding → x).  How linearly decodable x is."""
    m = np.isfinite(x)
    if m.sum() < 200:
        return np.nan
    X = StandardScaler().fit_transform(emb[m]); y = x[m]
    sc = cross_val_score(Ridge(alpha=1.0), X, y,
                          cv=KFold(5, shuffle=True, random_state=0),
                          scoring="r2")
    return float(np.mean(sc))


def probe_categorical(emb, y):
    m = y >= 0
    if m.sum() < 200 or len(np.unique(y[m])) < 2:
        return np.nan
    X = StandardScaler().fit_transform(emb[m])
    sc = cross_val_score(
        LogisticRegression(max_iter=2000, C=1.0,
                            class_weight="balanced"),
        X, y[m], cv=KFold(5, shuffle=True, random_state=0),
        scoring="balanced_accuracy")
    return float(np.mean(sc))


def main():
    inf = np.load(os.path.join(RUN, "eval", "inference.npz"),
                  allow_pickle=True)
    emb = inf["embeds"].astype(np.float32)            # (N,128)
    assigns = inf["assigns"].astype(int)              # (N,)
    N = assigns.size
    print(f"[interp] N={N}  K={assigns.max()+1}  emb={emb.shape}", flush=True)

    # ---- ACOM factors (corners-based) ----
    rep = os.path.join(RUN, "_imc_report")
    phase_id = _flat(np.load(os.path.join(rep, "05_full_phase_id.npy")))
    acom_corr = _flat(np.load(os.path.join(rep, "05_full_winning_corr.npy")))
    n_peaks = _flat(np.load(os.path.join(rep,
                    "05_n_peaks_per_position.npy")))

    # ---- crystallinity factors (max over q-windows) ----
    cdir = sorted(glob.glob(os.path.join(
        RUN, "crystallinity", "report_all_*")))[-1]
    print(f"[interp] crystallinity from {os.path.basename(cdir)}", flush=True)

    def _stack_max(prefix):
        fs = sorted(glob.glob(os.path.join(cdir, f"{prefix}_w*.npy")))
        if not fs:
            return np.full(N, np.nan)
        st = np.stack([np.load(f).astype(float).reshape(-1) for f in fs], 0)
        return np.nanmax(st, axis=0)
    peakhalo = _stack_max("peakhalo")
    azimvar = _stack_max("azimvar")
    try:
        scattered = _flat(np.load(os.path.join(
            cdir, "_scattered_intensity.npy")))
    except Exception:
        scattered = np.full(N, np.nan)

    cont = {
        "ACOM corr":            acom_corr,
        "crystallinity (peak/halo)": peakhalo,
        "spottiness (azim-var)": azimvar,
        "scattered intensity":  scattered,
        "n Bragg peaks":        n_peaks,
    }

    # ---- Test 1: ranking ----
    rows = []
    for name, x in cont.items():
        rows.append(dict(
            factor=name,
            eta2_separates=round(eta_squared(assigns, x), 4),
            MI=round(mi_cont(assigns, x), 4),
            probe_R2=round(probe_continuous(emb, x), 4)))
        print(f"  {name:26s} eta2={rows[-1]['eta2_separates']}  "
              f"MI={rows[-1]['MI']}  probeR2={rows[-1]['probe_R2']}",
              flush=True)
    # categorical phase
    ph_ami = adjusted_mutual_info_score(assigns, phase_id.astype(int))
    ph_ari = adjusted_rand_score(assigns, phase_id.astype(int))
    ph_probe = probe_categorical(emb, phase_id.astype(int))
    print(f"  ACOM phase(alpha/gamma/neither): AMI={ph_ami:.3f} "
          f"ARI={ph_ari:.3f} probe_bal_acc={ph_probe}", flush=True)

    with open(os.path.join(OUT, "test1_ranking.json"), "w") as f:
        json.dump(dict(continuous=rows,
                       phase=dict(AMI=ph_ami, ARI=ph_ari,
                                  probe_bal_acc=ph_probe)), f, indent=2)

    # ranking bar chart (probe R² = "is it encoded")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    nm = [r["factor"] for r in rows]
    r2 = [r["probe_R2"] if r["probe_R2"] == r["probe_R2"] else 0 for r in rows]
    et = [r["eta2_separates"] if r["eta2_separates"] == r["eta2_separates"]
          else 0 for r in rows]
    x = np.arange(len(nm))
    ax.bar(x - 0.2, r2, 0.4, label="probe R² (encoded in embedding)")
    ax.bar(x + 0.2, et, 0.4, label="η² (separated by DINO classes)")
    ax.set_xticks(x); ax.set_xticklabels(nm, rotation=25, ha="right",
                                          fontsize=8)
    ax.set_ylabel("score (0–1)")
    ax.set_title("What the DINO IMC_SI5 embedding encodes / separates")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "test1_factor_ranking.png"), dpi=150)
    plt.close(fig)

    # ---- Test 5: per-class signature heatmap ----
    K = assigns.max() + 1
    facs = list(cont.items())
    sig = np.full((K, len(facs)), np.nan)
    for ci in range(K):
        msk = assigns == ci
        for fi, (nme, x) in enumerate(facs):
            v = x[msk]; v = v[np.isfinite(v)]
            if v.size:
                sig[ci, fi] = np.median(v)
    # z-score each factor column for the heatmap
    z = sig.copy()
    for fi in range(len(facs)):
        col = z[:, fi]; mu = np.nanmean(col); sd = np.nanstd(col) or 1
        z[:, fi] = (col - mu) / sd
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(facs)))
    ax.set_xticklabels([f[0] for f in facs], rotation=25, ha="right",
                       fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"class {c}" for c in
                                                 range(K)], fontsize=8)
    ax.set_title("per-class physical signature (z-scored median)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "test5_class_signatures.png"), dpi=150)
    plt.close(fig)
    np.save(os.path.join(OUT, "test5_class_signature_medians.npy"), sig)

    print(f"[interp] wrote test1 + test5 to {OUT}", flush=True)


if __name__ == "__main__":
    main()
