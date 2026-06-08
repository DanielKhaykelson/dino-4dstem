"""interpret_core.py — compute backend for the Analysis ▸ Interpretation tab.

A GUI-agnostic port of the tools/interpret_imc_*.py battery, parameterised by
a linked run + dataset (no hard-coded paths).  Answers: *what physical factor
does the trained DINO model base its clustering on?*

Pieces (each independently runnable):
  • factors + probing  — scattered intensity / crystallinity / spottiness
                         (+ ACOM corr / n-peaks if an ACOM run exists);
                         probe R² (ridge) + η² + MI ranking.
  • causal ablations   — re-infer on radial-only / blur / per-frame-norm /
                         low-q / high-q / scattered-norm inputs; ARI vs orig.
  • class averages     — per-class mean pattern + radial profile + signature.
  • orientation x-check — DINO classes vs ACOM zone-axis (needs ACOM).
  • GradCAM / IG       — via viz_paper_attribution (per-prototype attention).

All figures are written with an explicit Agg canvas (never pyplot) so the
heavy work is safe to run inside a worker thread alongside the Tk app.
"""
import os
import glob
import json
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, mutual_info_score)

from gui_app.crystallinity_panel import _radial_mean_var, _snip_baseline


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _get_cmap(name):
    try:
        import matplotlib
        return matplotlib.colormaps[name]
    except Exception:
        from matplotlib import cm
        return cm.get_cmap(name)


def _save(fig, path):
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")


def _flat(a):
    return np.asarray(a, dtype=float).reshape(-1)


# --------------------------------------------------------------------------
# statistics (ported verbatim from tools/interpret_imc.py)
# --------------------------------------------------------------------------
def eta_squared(labels, x):
    """Correlation ratio η² — fraction of x's variance the partition explains."""
    m = np.isfinite(x)
    x = x[m]; lab = labels[m]
    if x.size < 10:
        return np.nan
    gm = x.mean()
    ss_tot = ((x - gm) ** 2).sum()
    if ss_tot <= 0:
        return np.nan
    ssb = 0.0
    for c in np.unique(lab):
        xc = x[lab == c]
        if xc.size:
            ssb += xc.size * (xc.mean() - gm) ** 2
    return float(ssb / ss_tot)


def mi_cont(labels, x, bins=24):
    m = np.isfinite(x)
    x = x[m]; lab = labels[m]
    if x.size < 10:
        return np.nan
    xb = np.digitize(x, np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    return float(mutual_info_score(lab, xb))


def probe_continuous(emb, x):
    """5-fold CV R² of Ridge(embedding → x): how linearly decodable x is."""
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
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        X, y[m], cv=KFold(5, shuffle=True, random_state=0),
        scoring="balanced_accuracy")
    return float(np.mean(sc))


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------
class Ctx:
    """Everything the battery needs, gathered from the linked Post-hoc run."""
    def __init__(self, run, sample, cube_path, vmax, scan_shape,
                 embeds, assigns, polar_cfg):
        self.run = run
        self.sample = sample
        self.cube_path = cube_path
        self.vmax = float(vmax)
        self.scan_shape = tuple(scan_shape)         # (Ny, Nx)
        self.embeds = np.asarray(embeds, np.float32)
        self.assigns = np.asarray(assigns, int)
        self.mask_r, self.mask_cols, self.ccrop, self.com = polar_cfg
        self.out = os.path.join(run, "_interpretability")
        os.makedirs(self.out, exist_ok=True)

    @property
    def K(self):
        return int(self.assigns.max()) + 1


# --------------------------------------------------------------------------
# ACOM discovery (reuse outputs from a prior ACOM full-dataset run)
# --------------------------------------------------------------------------
def find_acom_arrays(run):
    """Locate phase_id / winning_corr (+ optional rmat / n_peaks) from any
    prior ACOM run.  Returns a dict or None if no ACOM output is on disk.

    Search order covers the GUI ACOM tab (acom/maps/mpfull_*), the
    interpretability re-run (orient_*), and the imc_report tool (05_full_*).
    """
    def _first(*names):
        for n in names:
            if n and os.path.exists(n):
                return n
        return None

    j = os.path.join
    pid = _first(j(run, "acom", "maps", "mpfull_phase_id.npy"),
                 j(run, "_interpretability", "orient_phase_id.npy"),
                 j(run, "_imc_report", "05_full_phase_id.npy"))
    corr = _first(j(run, "acom", "maps", "mpfull_winning_corr.npy"),
                  j(run, "_interpretability", "orient_winning_corr.npy"),
                  j(run, "_imc_report", "05_full_winning_corr.npy"))
    if pid is None or corr is None:
        return None
    out = {"phase_id": np.load(pid), "winning_corr": np.load(corr),
           "src": os.path.dirname(pid)}
    rmat = _first(j(run, "acom", "maps", "mpfull_winning_rmat.npy"),
                  j(run, "_imc_report", "05_full_winning_rmat.npy"))
    if rmat is not None:
        try:
            out["winning_rmat"] = np.load(rmat)
        except Exception:
            pass
    npk = _first(j(run, "_imc_report", "05_n_peaks_per_position.npy"),
                 j(run, "acom", "maps", "mpfull_n_peaks.npy"))
    if npk is not None:
        try:
            out["n_peaks"] = np.load(npk)
        except Exception:
            pass
    return out


def _zone_axis_labels(acom):
    """Per-position integer label = (phase, zone-axis) from rotation matrices.
    Returns (labels[N], n_zone_axes) or (None, 0) if rmat unavailable."""
    if "winning_rmat" not in acom:
        return None, 0
    try:
        from gui_app.acom_core import zone_axis_from_matrix
    except Exception:
        return None, 0
    pid = _flat(acom["phase_id"]).astype(int)
    R = acom["winning_rmat"].reshape(-1, 3, 3)
    lab = np.full(pid.size, -1, int)
    keys = {}
    for i in range(pid.size):
        if pid[i] < 0 or not np.isfinite(R[i]).all():
            continue
        try:
            za, _ = zone_axis_from_matrix(R[i])
        except Exception:
            continue
        lab[i] = keys.setdefault((int(pid[i]), tuple(za)), len(keys))
    return lab, len(keys)


# --------------------------------------------------------------------------
# one cube pass: physical factors + class means
# --------------------------------------------------------------------------
def compute_factors_and_means(ctx, beam_px=None, qlo_frac=0.10, qhi_frac=0.90,
                              collect_classical=False, ds=40, progress=None):
    """Single pass over the cube → per-position factors + per-class means.

    Returns dict(scattered, peakhalo, azimvar, means, profiles, counts,
    beam_px, nb).  With ``collect_classical`` also returns per-position
    ``radial`` (azimuthal profile) and ``patt_ds`` (downsampled log pattern)
    for the classical-baseline comparison.
    """
    from gui_app.posthoc_panel import _open_lazy
    import cv2
    Ny, Nx = ctx.scan_shape
    cube = _open_lazy(ctx.cube_path, scan_shape=(Ny, Nx))
    # probe one frame for geometry
    f0 = np.asarray(cube[0][0], np.float32)
    H, W = f0.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    if beam_px is None:
        beam_px = max(8.0, round(0.11 * H))         # ≈ post-beam radius
    yy, xx = np.indices((H, W))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    postbeam = rr >= beam_px
    nb = int(min(H, W) // 2)
    lo = max(int(qlo_frac * nb), int(beam_px) + 1)
    hi = min(int(qhi_frac * nb), nb)
    eps = 1e-6

    N = Ny * Nx
    scattered = np.zeros(N, np.float64)
    peakhalo = np.full(N, np.nan)
    azimvar = np.full(N, np.nan)
    K = ctx.K
    sums = np.zeros((K, H, W), np.float64)
    cnts = np.zeros(K, np.int64)
    assn = ctx.assigns.reshape(Ny, Nx)
    radial = np.zeros((N, nb), np.float32) if collect_classical else None
    patt_ds = (np.zeros((N, ds * ds), np.float32)
               if collect_classical else None)

    for rx in range(Ny):
        row = np.asarray(cube[rx], np.float32)      # (Nx,H,W)
        for ry in range(Nx):
            i = rx * Nx + ry
            pat = row[ry]
            scattered[i] = float(pat[postbeam].sum())
            m, v, _ = _radial_mean_var(pat, (cy, cx), beam_px=beam_px)
            seg = m[lo:hi]
            if seg.size and np.nansum(seg) > 0:
                logI = np.log(np.clip(seg, eps, None))
                halo = np.exp(_snip_baseline(logI))
                peakhalo[i] = float(np.clip(seg - halo, 0, None).sum()
                                    / (seg.sum() + eps))
                vv = v[lo:hi]
                azimvar[i] = float(np.mean(vv / (seg * seg + eps)))
            if collect_classical:
                radial[i] = m[:nb]
                patt_ds[i] = cv2.resize(np.log1p(np.clip(pat, 0, None)),
                                        (ds, ds),
                                        interpolation=cv2.INTER_AREA).ravel()
            c = int(assn[rx, ry])
            sums[c] += pat; cnts[c] += 1
        if progress:
            progress(rx + 1, Ny, "factors+means")

    means = np.array([sums[c] / max(cnts[c], 1) for c in range(K)])
    rb = np.clip(rr.astype(int), 0, nb - 1)
    npix = np.bincount(rb.ravel(), None, nb)
    profs = np.array([np.bincount(rb.ravel(), means[c].ravel(), nb)
                      / np.maximum(npix, 1) for c in range(K)])
    return dict(scattered=scattered, peakhalo=peakhalo, azimvar=azimvar,
                means=means, profiles=profs, counts=cnts,
                beam_px=float(beam_px), nb=nb, rb=rb,
                radial=radial, patt_ds=patt_ds, ds=ds)


# --------------------------------------------------------------------------
# Test 1 — probing / ranking  (+ Test 5 signatures)
# --------------------------------------------------------------------------
def probe_and_signatures(ctx, factors, acom=None):
    emb = ctx.embeds; assigns = ctx.assigns; N = assigns.size
    cont = {
        "scattered intensity": factors["scattered"],
        "crystallinity (peak/halo)": factors["peakhalo"],
        "spottiness (azim-var)": factors["azimvar"],
    }
    if acom is not None:
        cont["ACOM correlation"] = _flat(acom["winning_corr"])
        if "n_peaks" in acom:
            cont["n Bragg peaks"] = _flat(acom["n_peaks"])

    rows = []
    for name, x in cont.items():
        x = np.asarray(x, float).reshape(-1)
        if x.size != N:
            x = np.full(N, np.nan)
        rows.append(dict(factor=name,
                         eta2=round(eta_squared(assigns, x), 4),
                         MI=round(mi_cont(assigns, x), 4),
                         probe_R2=round(probe_continuous(emb, x), 4)))

    cat = {}
    if acom is not None:
        pid = _flat(acom["phase_id"]).astype(int)
        if pid.size == N and len(np.unique(pid[pid >= 0])) >= 2:
            cat["phase"] = dict(
                AMI=round(float(adjusted_mutual_info_score(assigns, pid)), 4),
                ARI=round(float(adjusted_rand_score(assigns, pid)), 4))
        za, nz = _zone_axis_labels(acom)
        if za is not None and za.size == N:
            m = za >= 0
            if m.sum() > 50:
                cat["zone_axis"] = dict(
                    AMI=round(float(adjusted_mutual_info_score(
                        assigns[m], za[m])), 4),
                    ARI=round(float(adjusted_rand_score(
                        assigns[m], za[m])), 4),
                    n_indexed=int(m.sum()), n_zone_axes=int(nz))

    # signature medians (z-scored per factor)
    K = ctx.K
    names = list(cont.keys())
    sig = np.full((K, len(names)), np.nan)
    for ci in range(K):
        msk = assigns == ci
        for fi, nm in enumerate(names):
            v = np.asarray(cont[nm], float).reshape(-1)
            v = v[msk]; v = v[np.isfinite(v)]
            if v.size:
                sig[ci, fi] = np.median(v)

    out = dict(rows=rows, categorical=cat, factor_names=names, signature=sig)
    json.dump(dict(continuous=rows, categorical=cat),
              open(os.path.join(ctx.out, "test1_ranking.json"), "w"), indent=2)
    np.save(os.path.join(ctx.out, "test5_class_signature_medians.npy"), sig)
    _fig_ranking(ctx, rows)
    _fig_signatures(ctx, sig, names, K)
    return out


def _fig_ranking(ctx, rows):
    fig = Figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    nm = [r["factor"] for r in rows]
    r2 = [r["probe_R2"] if r["probe_R2"] == r["probe_R2"] else 0 for r in rows]
    et = [r["eta2"] if r["eta2"] == r["eta2"] else 0 for r in rows]
    x = np.arange(len(nm))
    ax.bar(x - 0.2, r2, 0.4, label="probe R² (encoded in embedding)")
    ax.bar(x + 0.2, et, 0.4, label="η² (separated by classes)")
    ax.set_xticks(x); ax.set_xticklabels(nm, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("score (0–1)")
    ax.set_title(f"{ctx.sample}: what the embedding encodes / separates")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(ctx.out, "test1_factor_ranking.png"))


def _fig_signatures(ctx, sig, names, K):
    z = sig.copy()
    for fi in range(len(names)):
        col = z[:, fi]; mu = np.nanmean(col); sd = np.nanstd(col) or 1
        z[:, fi] = (col - mu) / sd
    fig = Figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"class {c}" for c in range(K)], fontsize=8)
    ax.set_title("per-class physical signature (z-scored median)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, os.path.join(ctx.out, "test5_class_signatures.png"))


# --------------------------------------------------------------------------
# class-mean figures (Test 5)
# --------------------------------------------------------------------------
def figures_class_means(ctx, factors):
    means = factors["means"]; profs = factors["profiles"]
    cnts = factors["counts"]; nb = factors["nb"]; rb = factors["rb"]
    K = means.shape[0]
    ncol = 5; nrow = int(np.ceil(K / ncol))
    fig = Figure(figsize=(2.4 * ncol, 2.4 * nrow))
    for c in range(K):
        ax = fig.add_subplot(nrow, ncol, c + 1)
        img = np.log1p(np.clip(means[c], 0, None))
        out = img[rb > 10]
        vmax = float(np.percentile(out, 99.5)) if out.size else img.max()
        ax.imshow(img, cmap="inferno", vmax=max(vmax, 1e-6),
                  interpolation="nearest", aspect="equal")
        ax.set_title(f"class {c} ({int(cnts[c])}px)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{ctx.sample} — per-class mean diffraction", fontsize=12)
    _save(fig, os.path.join(ctx.out, "class_mean_patterns.png"))

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    lo = max(int(0.10 * nb), 8)
    cmap = _get_cmap("tab20")
    for c in range(K):
        ax.semilogy(np.arange(lo, nb), np.clip(profs[c, lo:], 1e-3, None),
                    color=cmap(c % 20), lw=1.3, label=f"c{c}")
    ax.set_xlabel("radial bin (px, beam trimmed)")
    ax.set_ylabel("mean I(r) (log)")
    ax.set_title("per-class mean radial profile")
    ax.legend(fontsize=7, ncol=2, loc="upper right"); ax.grid(alpha=0.3)
    _save(fig, os.path.join(ctx.out, "class_radial_profiles.png"))
    np.save(os.path.join(ctx.out, "class_mean_radial_profiles.npy"), profs)


# --------------------------------------------------------------------------
# Classical-baseline comparison — could a non-DINO method reproduce the map?
# --------------------------------------------------------------------------
def _km_ari(feat, ref, K, npca=0):
    """Standardize → (optional PCA) → KMeans(K) → ARI/AMI vs ref labels."""
    X = np.asarray(feat, np.float64)
    if X.ndim == 1:
        X = X[:, None]
    X = np.nan_to_num(X)
    X = StandardScaler().fit_transform(X)
    if npca and X.shape[1] > npca:
        X = PCA(n_components=npca, random_state=0).fit_transform(X)
    lab = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(X)
    return (round(float(adjusted_rand_score(ref, lab)), 4),
            round(float(adjusted_mutual_info_score(ref, lab)), 4), lab)


def classical_baselines(ctx, factors, progress=None):
    """Cluster a series of *classical* 4D-STEM feature sets into K (= DINO K)
    and measure how well each reproduces the DINO partition (ARI / AMI).

    Answers: could a classical pipeline (virtual DF, azimuthal integration,
    PCA / NMF decomposition, or a combination) have produced the same grain
    structure, or is the DINO partition distinctive?
    """
    if factors.get("radial") is None or factors.get("patt_ds") is None:
        raise RuntimeError("classical baselines need collect_classical=True")
    ref = ctx.assigns
    K = ctx.K
    rad = factors["radial"]; patt = factors["patt_ds"]
    scattered = factors["scattered"]

    methods = {}
    steps = [
        ("virtual DF (total scattered I)", lambda: _km_ari(scattered, ref, K)),
        ("azimuthal profile (radial) → PCA",
         lambda: _km_ari(rad, ref, K, npca=20)),
        ("pattern PCA (linear decomp.)",
         lambda: _km_ari(patt, ref, K, npca=30)),
    ]
    # NMF on the (non-negative) downsampled log-patterns — the classical
    # decomposition the user already runs in the NMF tab.
    def _nmf():
        Xn = np.clip(np.nan_to_num(patt), 0, None)
        W = NMF(n_components=min(2 * K, 30), init="nndsvda", max_iter=300,
                random_state=0).fit_transform(Xn)
        return _km_ari(W, ref, K)
    steps.append(("pattern NMF (non-neg decomp.)", _nmf))

    maps = {}
    for i, (name, fn) in enumerate(steps):
        ari, ami, lab = fn()
        methods[name] = dict(ARI=ari, AMI=ami)
        maps[name] = lab.reshape(ctx.scan_shape)
        if progress:
            progress(i + 1, len(steps) + 1, f"classical: {name}")

    # combination of complementary classical features
    combo = np.concatenate([
        StandardScaler().fit_transform(np.nan_to_num(scattered)[:, None]),
        PCA(20, random_state=0).fit_transform(
            StandardScaler().fit_transform(np.nan_to_num(rad))),
        PCA(30, random_state=0).fit_transform(
            StandardScaler().fit_transform(np.nan_to_num(patt))),
    ], axis=1)
    ari, ami, lab = _km_ari(combo, ref, K)
    methods["combination (DF + radial + pattern)"] = dict(ARI=ari, AMI=ami)
    maps["combination (DF + radial + pattern)"] = lab.reshape(ctx.scan_shape)
    if progress:
        progress(len(steps) + 1, len(steps) + 1, "classical: combination")

    best = max(methods.items(), key=lambda kv: kv[1]["ARI"])
    out = dict(methods=methods, best=best[0], best_ARI=best[1]["ARI"],
               K=K)
    json.dump(out, open(os.path.join(ctx.out, "classical_baselines.json"),
                        "w"), indent=2)
    _fig_classical(ctx, methods, maps)
    return out


def _fig_classical(ctx, methods, maps):
    # bar chart
    fig = Figure(figsize=(8, 4.2))
    ax = fig.add_subplot(111)
    nm = list(methods.keys())
    ari = [methods[n]["ARI"] for n in nm]
    ami = [methods[n]["AMI"] for n in nm]
    x = np.arange(len(nm))
    ax.bar(x - 0.2, ari, 0.4, label="ARI vs DINO")
    ax.bar(x + 0.2, ami, 0.4, label="AMI vs DINO")
    ax.axhline(0.8, ls="--", color="grey", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(nm, rotation=20, ha="right",
                                         fontsize=7)
    ax.set_ylabel("agreement with DINO (0–1)"); ax.set_ylim(0, 1)
    ax.set_title(f"{ctx.sample}: can a classical method reproduce the DINO "
                 f"map? (1 = identical)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(ctx.out, "classical_baselines.png"))

    # DINO vs each classical map
    items = list(maps.items())
    n = len(items) + 1
    fig = Figure(figsize=(2.8 * n, 3.2))
    cmap = ListedColormap(_get_cmap("tab20")(np.linspace(0, 1, max(ctx.K, 2))))
    ax = fig.add_subplot(1, n, 1)
    ax.imshow(ctx.assigns.reshape(ctx.scan_shape), cmap=cmap,
              interpolation="nearest", aspect="equal")
    ax.set_title("DINO", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for j, (name, m) in enumerate(items):
        ax = fig.add_subplot(1, n, j + 2)
        ax.imshow(m, cmap=cmap, interpolation="nearest", aspect="equal")
        ax.set_title(f"{name}\nARI={methods[name]['ARI']}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("DINO vs classical clusterings (labels are arbitrary; "
                 "shape match is what matters)", fontsize=10)
    _save(fig, os.path.join(ctx.out, "classical_vs_dino_maps.png"))


# --------------------------------------------------------------------------
# Tests 2 & 4 — causal ablations
# --------------------------------------------------------------------------
def _radial_only(x):
    a = x[0].numpy(); H, W = a.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W))
    rb = np.clip(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int),
                 0, min(H, W) // 2 - 1)
    nb = rb.max() + 1
    prof = (np.bincount(rb.ravel(), a.ravel(), nb)
            / np.maximum(np.bincount(rb.ravel(), None, nb), 1))
    import torch
    return torch.from_numpy(prof[rb].astype(np.float32)).unsqueeze(0)


def _blur(sig):
    from scipy.ndimage import gaussian_filter
    import torch
    return lambda x: torch.from_numpy(
        gaussian_filter(x[0].numpy(), sigma=sig)).unsqueeze(0)


def _qmask(inner, outer):
    import torch
    def f(x):
        a = x[0].numpy().copy(); H, W = a.shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        yy, xx = np.indices((H, W))
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(H, W) / 2.0)
        a[(r >= inner) & (r < outer)] = 0.0
        return torch.from_numpy(a.astype(np.float32)).unsqueeze(0)
    return f


def _scattered_norm(beam_r):
    import torch
    def f(x):
        a = x[0].numpy().astype(np.float32); H, W = a.shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        yy, xx = np.indices((H, W))
        post = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) >= beam_r
        s = float(a[post].sum())
        if s > 1e-6:
            a = a / s * 100.0
        return torch.from_numpy(a).unsqueeze(0)
    return f


class _Wrap:
    def __init__(self, base, fn):
        self.base = base; self.fn = fn
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        return self.fn(self.base[i])


def run_ablations(ctx, model, dev, which=None, progress=None):
    """Re-infer the model on ablated inputs; ARI vs the original class map."""
    import torch  # noqa
    from data import LoadPRZ
    from contrastive_eval import infer_scan
    Ny, Nx = ctx.scan_shape
    base = LoadPRZ(ctx.cube_path, resize=192, vmax=ctx.vmax)
    orig = ctx.assigns
    beam_r = max(8.0, round(0.11 * 192))

    def _infer(ds):
        inf = infer_scan(model, ds, dev, dense_remap=True, polar_size=192,
                         polar_mask_cols=ctx.mask_cols,
                         center_crop_size=ctx.ccrop, com_centering=ctx.com,
                         center_mask_radius=ctx.mask_r, eval_temp=0.06,
                         batch_size=128)
        return np.asarray(inf["assigns"]).astype(int)

    catalog = {
        "baseline":      lambda x: x,
        "scattered_norm": _scattered_norm(beam_r),
        "radial_only":   _radial_only,
        "blur(s=2)":     _blur(2.0),
        "qmask_low":     _qmask(0.0, 0.45),
        "qmask_high":    _qmask(0.45, 1.01),
    }
    if which:
        catalog = {k: v for k, v in catalog.items()
                   if k in which or k == "baseline"}
    results = {}; maps = {}
    for i, (name, fn) in enumerate(catalog.items()):
        a = _infer(_Wrap(base, fn))
        results[name] = dict(
            ARI_vs_orig=round(float(adjusted_rand_score(orig, a)), 4),
            NMI_vs_orig=round(float(normalized_mutual_info_score(orig, a)), 4),
            K_active=int(len(np.unique(a))))
        maps[name] = a.reshape(Ny, Nx)
        if progress:
            progress(i + 1, len(catalog), f"ablation {name}")

    json.dump(results, open(os.path.join(ctx.out, "test2_4_ablations.json"),
                            "w"), indent=2)
    _fig_ablation_maps(ctx, maps, results)
    return results


def _fig_ablation_maps(ctx, maps, results):
    n = len(maps)
    fig = Figure(figsize=(3.0 * n, 3.4))
    K = ctx.K
    cmap = ListedColormap(_get_cmap("tab20")(np.linspace(0, 1, max(K, 2))))
    for j, (name, m) in enumerate(maps.items()):
        ax = fig.add_subplot(1, n, j + 1)
        ax.imshow(m, cmap=cmap, interpolation="nearest", aspect="equal")
        ax.set_title(f"{name}\nARI={results[name]['ARI_vs_orig']}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Ablation re-inference vs original map "
                 "(high ARI = info not needed)", fontsize=11)
    _save(fig, os.path.join(ctx.out, "test2_4_ablation_maps.png"))


# --------------------------------------------------------------------------
# GradCAM / IG
# --------------------------------------------------------------------------
def run_gradcam(ctx):
    """Per-prototype class-average Grad-CAM + IG via viz_paper_attribution.
    Returns the output figure path (or raises)."""
    import viz_paper_attribution as vpa
    vpa.run(ctx.run, ctx.sample, n_samples_per_proto=3)
    return os.path.join(ctx.run, "eval", "paper_attribution",
                        "fig_paper_attribution.png")


# --------------------------------------------------------------------------
# auto report
# --------------------------------------------------------------------------
def write_report(ctx, probe=None, ablations=None, acom=None, classical=None,
                 did_gradcam=False):
    L = []
    L.append(f"# Interpretation report — {ctx.sample}")
    L.append("")
    L.append(f"Run: `{os.path.basename(ctx.run)}`  ·  "
             f"K_active = {ctx.K}  ·  N = {ctx.assigns.size}")
    L.append("")

    drivers = []
    if probe:
        L.append("## What the embedding encodes / separates (probing)")
        L.append("")
        L.append("| factor | probe R² | η² | MI |")
        L.append("|---|---|---|---|")
        for r in sorted(probe["rows"], key=lambda r: -(r["probe_R2"] or 0)):
            L.append(f"| {r['factor']} | {r['probe_R2']} | {r['eta2']} "
                     f"| {r['MI']} |")
        L.append("")
        for r in probe["rows"]:
            if (r["probe_R2"] or 0) >= 0.2:
                drivers.append(r["factor"])
        cat = probe.get("categorical", {})
        if "phase" in cat:
            L.append(f"ACOM phase vs classes: AMI={cat['phase']['AMI']}, "
                     f"ARI={cat['phase']['ARI']}.")
        if "zone_axis" in cat:
            z = cat["zone_axis"]
            L.append(f"ACOM zone-axis vs classes: AMI={z['AMI']}, "
                     f"ARI={z['ARI']} ({z['n_indexed']} indexed, "
                     f"{z['n_zone_axes']} zone axes).")
        L.append("")

    if ablations:
        L.append("## Causal ablations (ARI vs original map)")
        L.append("")
        L.append("| ablation | ARI | K |")
        L.append("|---|---|---|")
        for k, v in ablations.items():
            L.append(f"| {k} | {v['ARI_vs_orig']} | {v['K_active']} |")
        L.append("")
        L.append("High ARI = that information was not needed; low ARI = the "
                 "model depends on it.")
        L.append("")

    uniq = None
    if classical:
        L.append("## Could a classical method reproduce the DINO map?")
        L.append("")
        L.append("| classical feature | ARI vs DINO | AMI vs DINO |")
        L.append("|---|---|---|")
        for k, v in sorted(classical["methods"].items(),
                           key=lambda kv: -kv[1]["ARI"]):
            L.append(f"| {k} | {v['ARI']} | {v['AMI']} |")
        L.append("")
        b = classical["best_ARI"]
        ba = max((v.get("AMI", 0.0) for v in classical["methods"].values()),
                 default=0.0)
        if b >= 0.8:
            uniq = (f"a classical pipeline (**{classical['best']}**) already "
                    f"reproduces the DINO partition (ARI={b}); DINO is **not** "
                    "adding distinctive structure here")
        elif b >= 0.45 or ba >= 0.55:
            uniq = (f"classical features **substantially capture** the "
                    f"partition (best ARI={b}, AMI={round(ba, 2)}); DINO "
                    "refines them but does not depart strongly — it behaves "
                    "like a non-linear blend of classical descriptors")
        elif b >= 0.3:
            uniq = (f"classical methods **partially** reproduce the map (best "
                    f"ARI={b}, AMI={round(ba, 2)}); DINO refines beyond them")
        else:
            uniq = (f"**no classical baseline reproduces the DINO partition** "
                    f"(best ARI={b}, AMI={round(ba, 2)}) — the clustering is "
                    "distinctive, not a re-labelling of a virtual-DF / radial "
                    "/ PCA / NMF map")
        L.append(f"**Uniqueness:** {uniq}.")
        L.append("")

    # nuanced verdict
    L.append("## Reading")
    L.append("")
    bits = []
    if drivers:
        bits.append("the embedding most strongly encodes **"
                    + ", ".join(drivers) + "**")
    if ablations:
        sn = ablations.get("scattered_norm", {}).get("ARI_vs_orig")
        ro = ablations.get("radial_only", {}).get("ARI_vs_orig")
        ql = ablations.get("qmask_low", {}).get("ARI_vs_orig")
        qh = ablations.get("qmask_high", {}).get("ARI_vs_orig")
        if sn is not None and sn < 0.2:
            bits.append("removing the overall **scattered intensity** "
                        f"collapses the map (ARI={sn}) — a major driver")
        if ro is not None and ro < 0.3:
            bits.append("the **1-D radial profile alone** is insufficient "
                        f"(ARI={ro}) — 2-D structure is needed")
        if ql is not None and qh is not None and ql < qh:
            bits.append(f"**low-q** matters more than high-q "
                        f"(ARI {ql} vs {qh})")
    if acom is not None:
        za = (probe or {}).get("categorical", {}).get("zone_axis", {})
        if za and za.get("AMI", 1) < 0.15:
            bits.append("it is **not** based on crystallographic orientation "
                        f"(zone-axis AMI={za['AMI']})")
    if uniq:
        bits.append(uniq)
    if not bits:
        bits.append("results written — inspect the figures for the dominant "
                    "axis")
    # capitalise the first clause for a clean sentence
    bits[0] = bits[0][:1].upper() + bits[0][1:]
    L.append("Summary: " + "; ".join(bits) + ".")
    L.append("")
    if acom is None:
        L.append("> **ACOM not available** for this run, so the "
                 "orientation/phase factors and the zone-axis cross-check "
                 "were skipped. Run ACOM (Diffraction ▸ ACOM, full-dataset "
                 "mode) first to include them.")
        L.append("")
    if did_gradcam:
        L.append("Grad-CAM / IG per-prototype attention written to "
                 "`eval/paper_attribution/`.")
        L.append("")

    # distinct filename so a hand-curated report.md is never clobbered
    path = os.path.join(ctx.out, "report_auto.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    return path
