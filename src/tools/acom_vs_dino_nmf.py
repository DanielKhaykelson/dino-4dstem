"""acom_vs_dino_nmf.py -- does the ACOM (crystallographic) label
field agree with DINO's learned class map more than with NMF
clustering?

Builds three per-pixel label fields over the same scan:
  DINO : assigns from eval/inference.npz
  NMF  : Wang-2024 polar NMF + KMeans (reuses gui_app.nmf_panel)
  ACOM : full-dataset orientation match → integer zone-axis label
          (one CIF; pixels with too few peaks / no match = -1)

Then computes ARI / NMI / V-measure for
  ACOM ↔ DINO    and    ACOM ↔ NMF
over the pixels where ACOM produced a valid match.  If
ACOM↔DINO > ACOM↔NMF, DINO's classes encode crystallographic
structure that NMF doesn't.

Output: a 3-panel map figure + a metrics JSON + console summary,
written to <run>/_acom_vs_dino_nmf/.

Usage:
  python tools/acom_vs_dino_nmf.py --run <run_dir> --cif <gamma.cif>
        [--stride 2] [--k-max 0.5] [--inv-ang-per-px 0.00493]
        [--corr-min 0.0] [--nmf-k 10]
"""
from __future__ import annotations
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def _log(m): print(f"[acom_vs] {m}", flush=True)


def _register(run_dir):
    """Resolve + register the sample (SAMPLE_LOCK / run_summary)."""
    import json as _j
    lock = None
    cur = os.path.abspath(run_dir)
    for _ in range(5):
        p = os.path.join(cur, "SAMPLE_LOCK.json")
        if os.path.exists(p): lock = p; break
        nxt = os.path.dirname(cur)
        if nxt == cur: break
        cur = nxt
    rs = _j.load(open(os.path.join(run_dir, "run_summary.json")))
    sample = rs["sample"]
    if lock:
        spec = _j.load(open(lock))
        from data import register_runtime_sample
        register_runtime_sample(
            spec["cube_path"], vmax=float(spec.get("vmax", 5.0)),
            center_mask_radius=int(spec.get("polar_mask_cols", 0)) // 2,
            key=sample)
    return sample


def _nmf_kmeans_labels(sample, k):
    from gui_app.nmf_panel import (NMF_VARIANTS, build_nmf_input,
                                       fit_nmf, cluster_W)
    cfg = NMF_VARIANTS["Polar  (Wang et al. 2024)"]
    X, X_aug, comp_shape, info = build_nmf_input(sample, cfg)
    W, H, err = fit_nmf(X, X_aug=None, n_components=int(k))
    lab = cluster_W(W, "K-means", k=int(k))
    return np.asarray(lab, dtype=np.int32)


def _acom_za_labels(sample, cif, stride, k_max, inv_a, corr_min,
                       plan_mode="corners"):
    """Full-dataset ACOM → per-pixel integer-ZA label map (flattened),
    plus a validity mask (corr >= corr_min and finite ZA)."""
    from gui_app.acom_core import (load_crystal, prepare_crystal,
                                       acom_full_dataset,
                                       zone_axis_from_matrix)
    from data import SAMPLES
    from gui_app.posthoc_panel import _open_lazy
    cfg = SAMPLES[sample]
    _log("building crystal …")
    cr = load_crystal(cif); prepare_crystal(cr, k_max=k_max,
                                                 plan_mode=plan_mode)
    cube = _open_lazy(cfg["path"], scan_shape=cfg["scan_shape"])
    t0 = time.time()
    def _prog(done, total, stage):
        if stage == "detect" and done % 512 == 0:
            _log(f"  detect {done}/{total} ({time.time()-t0:.0f}s)")
    omap, bv, scan_shape = acom_full_dataset(
        cr, cube, inv_ang_per_pixel=inv_a, subsample_stride=stride,
        progress_cb=_prog,
        detect_kw=dict(min_sigma=1.5, max_sigma=4.0, num_sigma=5,
                          threshold=0.05, log_stretch=True))
    Ny, Nx = scan_shape
    corr = np.full((Ny, Nx), -1.0, np.float32)
    cv = getattr(omap, "corr", None)
    if cv is not None:
        a = np.asarray(cv)
        corr[:] = a[..., 0] if a.ndim == 3 else a
    mv = np.asarray(getattr(omap, "matrix", None))
    za_label = np.full((Ny, Nx), -1, np.int32)
    za_lut = {}                       # (u,v,w) -> int id
    if mv is not None and mv.ndim >= 4:
        top = mv[..., 0, :, :] if mv.ndim == 5 else mv
        for rx in range(Ny):
            for ry in range(Nx):
                if corr[rx, ry] < corr_min: continue
                R = top[rx, ry]
                if not np.isfinite(R).all(): continue
                za, _ = zone_axis_from_matrix(R)
                if za not in za_lut: za_lut[za] = len(za_lut)
                za_label[rx, ry] = za_lut[za]
    valid = za_label.ravel() >= 0
    return za_label.ravel(), valid, scan_shape, len(za_lut)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cif", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--k-max", type=float, default=0.5)
    ap.add_argument("--inv-ang-per-px", type=float, default=0.00493)
    ap.add_argument("--corr-min", type=float, default=0.0)
    ap.add_argument("--nmf-k", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = os.path.abspath(args.run)
    out = args.out or os.path.join(run, "_acom_vs_dino_nmf")
    os.makedirs(out, exist_ok=True)

    from sklearn.metrics import (adjusted_rand_score,
                                       normalized_mutual_info_score,
                                       v_measure_score)
    sample = _register(run)
    _log(f"sample = {sample}")

    inf = np.load(os.path.join(run, "eval", "inference.npz"),
                     allow_pickle=True)
    dino = np.asarray(inf["assigns"], dtype=np.int32)
    from data import SAMPLES
    Ny, Nx = SAMPLES[sample]["scan_shape"]

    _log("NMF + KMeans …")
    nmf = _nmf_kmeans_labels(sample, args.nmf_k)

    _log(f"ACOM full-dataset (stride={args.stride}) …")
    acom, valid, scan_shape, n_za = _acom_za_labels(
        sample, args.cif, args.stride, args.k_max,
        args.inv_ang_per_px, args.corr_min)
    _log(f"ACOM produced {n_za} distinct zone-axis labels; "
            f"{int(valid.sum())}/{valid.size} pixels matched.")

    # Restrict all three to the ACOM-valid pixels.
    d = dino[valid]; n = nmf[valid]; a = acom[valid]
    def _trip(x, y):
        return dict(ARI=float(adjusted_rand_score(x, y)),
                      NMI=float(normalized_mutual_info_score(x, y)),
                      V=float(v_measure_score(x, y)))
    m_dino = _trip(a, d)
    m_nmf  = _trip(a, n)
    m_dn   = _trip(d, n)
    metrics = dict(
        sample=sample, n_valid=int(valid.sum()),
        n_total=int(valid.size), n_za_labels=int(n_za),
        acom_vs_dino=m_dino, acom_vs_nmf=m_nmf,
        dino_vs_nmf=m_dn,
        verdict=("ACOM aligns with DINO more than NMF"
                   if m_dino["ARI"] > m_nmf["ARI"]
                   else "ACOM aligns with NMF more than DINO"))
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"),
                indent=2)
    _log(f"ACOM↔DINO  ARI={m_dino['ARI']:.3f} NMI={m_dino['NMI']:.3f}")
    _log(f"ACOM↔NMF   ARI={m_nmf['ARI']:.3f} NMI={m_nmf['NMI']:.3f}")
    _log(f"VERDICT: {metrics['verdict']}")

    # Figure: DINO | NMF | ACOM-ZA, each masked to valid pixels.
    def _panel(ax, lab_flat, title, K):
        img = lab_flat.reshape(Ny, Nx).astype(float)
        img[~valid.reshape(Ny, Nx)] = np.nan
        cmap = (plt.get_cmap("tab20") if K > 10
                  else plt.get_cmap("tab10")).copy()
        cmap.set_bad("white")
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    _panel(axes[0], dino, f"DINO  (K={dino.max()+1})", dino.max()+1)
    _panel(axes[1], nmf,  f"NMF-KMeans  (K={nmf.max()+1})",
              nmf.max()+1)
    _panel(axes[2], acom, f"ACOM zone-axis  ({n_za} ZA)", n_za)
    fig.suptitle(
        f"{sample} — on ACOM-matched pixels  ·  "
        f"ACOM↔DINO ARI={m_dino['ARI']:.3f}  ·  "
        f"ACOM↔NMF ARI={m_nmf['ARI']:.3f}  ·  {metrics['verdict']}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, "acom_vs_dino_nmf.png"),
                  dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    _log(f"done → {out}")


if __name__ == "__main__":
    main()
