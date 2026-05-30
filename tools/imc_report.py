"""imc_report.py -- IMC_SI5 full-stack analysis report.

Runs end-to-end on the IMC_SI5 sweep run at m=0.97 seed=42 K=60
(K_eff_active=13).  Produces a markdown + DOCX report under
``<run_root>/_imc_report/``.

Pipeline
--------
  §1  Wang 2024 polar NMF  + 4 clustering methods (K-means / Aglo
        / HDBSCAN / FCM).  n_components picked via knee on the
        reconstruction-error curve (default candidates 6/8/10/12).
  §2  DINO  vs  NMF-KMeans cross-comparison (confusion + ARI/NMI/V).
  §3  Multi-phase ACOM (α + γ) on every active DINO class average.
  §4  Multi-phase ACOM on the top-5 largest grains per class.
  §5  Full-dataset multi-phase ACOM at stride=2.
  §6  Crystallinity metrics per class (peak count, P/B ratio,
        sharpest FWHM, IBF intensity).
  §7  Markdown report + DOCX export.

This script is launched with --root pointing at the sweep run dir
that holds eval/inference.npz, e.g.
    --root <sweep_root>/IMC_SI5/stage2/m0.9700_seed42_K60

The α + γ CIFs are taken from
    D:/DINOSR/data/231228-IMC150nm-0p2apersec-anneal-70c-60min/cifs/
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time, traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
CIF_DIR = r"D:\DINOSR\data\231228-IMC150nm-0p2apersec-anneal-70c-60min\cifs"
ALPHA_CIF = os.path.join(CIF_DIR, "alpha.cif")
GAMMA_CIF = os.path.join(CIF_DIR, "gamma.cif")
# Cube is 192×192 resampled from a 512×512 detector at 0.0185 nm⁻¹/px,
# so the cube pixel size is 0.0185 × 512/192 = 0.0493 nm⁻¹/px
# = 0.00493 1/Å/px.  Detector edge = 96 × 0.00493 ≈ 0.474 1/Å.
INV_ANG_PER_PX = 0.00493
K_NMF_CANDIDATES = (6, 8, 10, 12)     # NMF n_components grid for elbow
ACOM_K_MAX = 0.5                       # 1/Å — capped at detector edge
ACOM_PLAN_MODE = "corners"             # broader ZA coverage for triclinic
ACOM_DETECT_KW = dict(
    # Stricter than the GUI default: IMC class averages have many
    # subtle features and a loose blob_log fires on noise, inflating
    # the "corr" sum.  threshold=0.05 + max_sigma=4 keeps O(30-150)
    # real Bragg peaks per pattern.
    min_sigma=1.5, max_sigma=4.0, num_sigma=5,
    threshold=0.05, log_stretch=True,
)
# CrystalPhase / quantify_phase params
QUANT_KW = dict(
    corr_kernel_size=0.04,
    sigma_excitation_error=0.02,
    power_intensity=0.25,
    power_intensity_experiment=0.25,
    max_number_patterns=1,           # one ZA per phase per pattern
    allow_strain=False,              # speed
    include_false_positives=True,
    weight_false_positives=1.0,
    weight_unmatched_peaks=1.0,
)
FULL_STRIDE = 2


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _log(msg: str):
    print(f"[imc_report] {msg}", flush=True)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def _save_fig(fig, path: str, dpi: int = 140):
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    _log(f"saved {os.path.relpath(path, start=os.path.dirname(path))}")


def _find_sample_lock(start_dir: str, max_walk: int = 5):
    cur = os.path.abspath(start_dir)
    for _ in range(int(max_walk)):
        p = os.path.join(cur, "SAMPLE_LOCK.json")
        if os.path.exists(p):
            return p
        parent = os.path.dirname(cur)
        if parent == cur: break
        cur = parent
    return None


def _register_sample_from_lock(run_dir: str) -> str:
    """Walk up the run dir for SAMPLE_LOCK.json and register the cube
    so the GUI's NMF panel can address the sample by key."""
    lock = _find_sample_lock(run_dir)
    if lock is None:
        raise RuntimeError(f"no SAMPLE_LOCK.json found near {run_dir}")
    spec = json.load(open(lock, encoding="utf-8"))
    cube_p = spec["cube_path"]
    if not os.path.exists(cube_p):
        raise RuntimeError(f"cube not on disk: {cube_p}")
    vmax = float(spec.get("vmax", 5.0))
    pmc = int(spec.get("polar_mask_cols", 0))
    derived_cmr = pmc // 2
    # Pick a key matching the run's run_summary.json
    rs = json.load(open(os.path.join(run_dir, "run_summary.json")))
    sample_key = rs["sample"]
    from data import register_runtime_sample
    register_runtime_sample(cube_p, vmax=vmax,
                              center_mask_radius=derived_cmr,
                              key=sample_key)
    _log(f"sample registered: {sample_key}  ←  {os.path.basename(cube_p)}  "
            f"vmax={vmax}  cmr={derived_cmr}")
    return sample_key


def _load_dino_inf(run_dir: str):
    p = os.path.join(run_dir, "eval", "inference.npz")
    if not os.path.exists(p):
        raise RuntimeError(f"no inference.npz at {p}")
    d = np.load(p, allow_pickle=True)
    return dict(assigns=d["assigns"],
                  soft_probs=d["soft_probs"],
                  embeds=d["embeds"])


# ---------------------------------------------------------------------------
# §1 — NMF + 4 clustering methods
# ---------------------------------------------------------------------------
def run_nmf_and_clusters(sample_key: str, outdir: str,
                            n_components_grid=K_NMF_CANDIDATES):
    """Wang 2024 polar NMF.  Picks n_components by elbow on
    reconstruction-error curve, then clusters W with all 4 methods."""
    from gui_app.nmf_panel import (NMF_VARIANTS, build_nmf_input,
                                       fit_nmf, cluster_W,
                                       POLAR_SIZE, CENTER_CROP,
                                       POLAR_MASK_COLS)
    wang_cfg = NMF_VARIANTS["Polar  (Wang et al. 2024)"]
    _log("§1: building polar input matrix …")
    t0 = time.time()
    X, X_aug, comp_shape, info = build_nmf_input(sample_key, wang_cfg)
    _log(f"     X.shape={X.shape}  D={info['D']}  ({time.time()-t0:.0f}s)")

    # Elbow over n_components_grid
    errs = []
    Ws = {}
    Hs = {}
    for nc in n_components_grid:
        _log(f"     fitting NMF n={nc} …")
        t0 = time.time()
        W, H, err = fit_nmf(X, X_aug=None, n_components=int(nc))
        Ws[nc] = W; Hs[nc] = H; errs.append((int(nc), float(err)))
        _log(f"       err={err:.4g}  ({time.time()-t0:.0f}s)")

    # Pick knee: largest 2nd derivative on the err curve.
    ns = np.array([n for n, _ in errs], dtype=float)
    es = np.array([e for _, e in errs], dtype=float)
    if len(ns) >= 3:
        d2 = np.diff(es, 2)
        knee = int(ns[1 + int(np.argmax(np.abs(d2)))])
    else:
        knee = int(ns[len(ns) // 2])
    _log(f"     selected n_components = {knee}  "
            f"(err curve: {dict(zip(ns.astype(int).tolist(), es.tolist()))})")

    W_best, H_best = Ws[knee], Hs[knee]

    # Save error curve
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(ns, es, "o-", color="#1f77b4")
    ax.axvline(knee, color="#d62728", linestyle="--",
                label=f"selected n={knee}")
    ax.set_xlabel("n_components")
    ax.set_ylabel("reconstruction error")
    ax.set_title("NMF reconstruction-error vs n_components", fontsize=11)
    ax.legend(fontsize=8)
    _save_fig(fig, os.path.join(outdir, "01a_nmf_err_curve.png"))

    # Save NMF component panels (H reshaped to (theta, r) ).
    th, r = comp_shape
    cols = 5
    rows = (knee + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0,
                                                          rows * 2.4),
                                squeeze=False)
    H_imgs = H_best.reshape(knee, th, r)
    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        ax.set_xticks([]); ax.set_yticks([])
        if k >= knee:
            ax.set_axis_off(); continue
        img = H_imgs[k]
        # log-stretch for visibility (Wang 2024 components have wide
        # dynamic range — central peak dominates otherwise).
        img_d = np.log1p(np.clip(img, 0, None))
        ax.imshow(img_d, cmap="inferno", aspect="auto",
                    interpolation="nearest")
        ax.set_title(f"H{k}", fontsize=9)
    fig.suptitle(
        f"Wang 2024 polar NMF — {knee} components  "
        f"(reshaped to θ × r = {th} × {r})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, os.path.join(outdir, "01b_nmf_components.png"))

    # Run all 4 clustering methods on W.
    # Pick a target K per method: the elbow n_components is a sensible
    # default for K-means / Aglo / FCM; HDBSCAN ignores K.
    K_for_clusters = knee
    cluster_labels = {}
    for m in ("K-means", "Aglo", "HDBSCAN", "FCM"):
        _log(f"     clustering W with {m} …")
        t0 = time.time()
        if m == "Aglo":
            lab = cluster_W(W_best, m, k=K_for_clusters,
                              distance="euclidean")
        elif m == "HDBSCAN":
            lab = cluster_W(W_best, m, min_cluster_size=80)
        else:
            lab = cluster_W(W_best, m, k=K_for_clusters)
        cluster_labels[m] = np.asarray(lab, dtype=np.int32)
        _log(f"       {m} done in {time.time()-t0:.0f}s, "
                f"n_classes={int(cluster_labels[m].max()) + 1}")

    return dict(W=W_best, H=H_best, n_components=knee,
                  cluster_labels=cluster_labels,
                  comp_shape=comp_shape, errs=errs)


def render_cluster_maps(sample_key: str, nmf_result, outdir: str):
    """4-panel side-by-side of the cluster-method class maps."""
    from data import SAMPLES
    Ny, Nx = SAMPLES[sample_key]["scan_shape"]
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.2))
    for ax, m in zip(axes, ("K-means", "Aglo", "HDBSCAN", "FCM")):
        lab = nmf_result["cluster_labels"][m]
        K_act = int(lab.max()) + 1
        cmap = (plt.get_cmap("tab10") if K_act <= 10
                  else plt.get_cmap("tab20") if K_act <= 20
                  else plt.get_cmap("turbo"))
        palette = ListedColormap([cmap(i / max(K_act - 1, 1))
                                       if K_act > 20
                                       else cmap(i) for i in range(K_act)])
        ax.imshow(lab.reshape(Ny, Nx), cmap=palette,
                    vmin=-0.5, vmax=K_act - 0.5,
                    interpolation="nearest", aspect="auto")
        ax.set_title(f"{m}  (K={K_act})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"NMF cluster maps  —  Wang 2024 polar  "
        f"(n_components = {nmf_result['n_components']})",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, os.path.join(outdir, "01c_nmf_cluster_maps.png"))


# ---------------------------------------------------------------------------
# §2 — DINO vs NMF (KMeans) comparison
# ---------------------------------------------------------------------------
def compare_dino_vs_nmf(sample_key: str, nmf_result, dino_inf,
                              outdir: str):
    from data import SAMPLES
    from sklearn.metrics import (adjusted_rand_score,
                                       normalized_mutual_info_score,
                                       v_measure_score)
    Ny, Nx = SAMPLES[sample_key]["scan_shape"]
    dino_lab = np.asarray(dino_inf["assigns"], dtype=np.int32)
    nmf_lab  = nmf_result["cluster_labels"]["K-means"]

    ari = adjusted_rand_score(dino_lab, nmf_lab)
    nmi = normalized_mutual_info_score(dino_lab, nmf_lab)
    vm  = v_measure_score(dino_lab, nmf_lab)
    metrics = dict(ARI=float(ari), NMI=float(nmi),
                     V_measure=float(vm),
                     K_dino=int(dino_lab.max()) + 1,
                     K_nmf=int(nmf_lab.max()) + 1)
    with open(os.path.join(outdir, "02_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix (rows = DINO, cols = NMF), normalised by DINO row.
    Kd, Kn = metrics["K_dino"], metrics["K_nmf"]
    C = np.zeros((Kd, Kn), dtype=np.int64)
    for d, n in zip(dino_lab, nmf_lab):
        C[int(d), int(n)] += 1
    Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1)

    fig = plt.figure(figsize=(15.5, 5.4))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.20)
    # DINO map
    ax = fig.add_subplot(gs[0, 0])
    cmap_d = (plt.get_cmap("tab20") if Kd <= 20 else plt.get_cmap("turbo"))
    pd = ListedColormap([cmap_d(i / max(Kd - 1, 1)) if Kd > 20
                              else cmap_d(i) for i in range(Kd)])
    ax.imshow(dino_lab.reshape(Ny, Nx), cmap=pd,
               vmin=-0.5, vmax=Kd - 0.5,
               interpolation="nearest", aspect="auto")
    ax.set_title(f"DINO  K_active={Kd}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    # NMF map
    ax = fig.add_subplot(gs[0, 1])
    cmap_n = (plt.get_cmap("tab10") if Kn <= 10 else plt.get_cmap("tab20"))
    pn = ListedColormap([cmap_n(i) for i in range(Kn)])
    ax.imshow(nmf_lab.reshape(Ny, Nx), cmap=pn,
               vmin=-0.5, vmax=Kn - 0.5,
               interpolation="nearest", aspect="auto")
    ax.set_title(f"NMF-KMeans  K={Kn}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    # Confusion heatmap (row-normalised)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(Cn, cmap="viridis", aspect="auto",
                     vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel(f"NMF class (K={Kn})")
    ax.set_ylabel(f"DINO class (K={Kd})")
    ax.set_title(
        f"Confusion (row-norm)\n"
        f"ARI={ari:.3f}  NMI={nmi:.3f}  V={vm:.3f}",
        fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(
        f"§2 — DINO vs NMF (KMeans) classification of IMC_SI5",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, os.path.join(outdir, "02_nmf_vs_dino.png"))
    _log(f"§2 metrics: ARI={ari:.3f}  NMI={nmi:.3f}  V={vm:.3f}")
    return metrics, C


# ---------------------------------------------------------------------------
# §3, §4 — class-average + grain ACOM (multi-phase α + γ)
# ---------------------------------------------------------------------------
def _class_averages_raw(sample_key, dino_inf, top_n=256):
    """Top-N-by-confidence raw-detector class averages.  Returns
    list of (K_active,) 2D patterns at raw resolution + class sizes."""
    from data import SAMPLES, LoadPRZ, apply_sample_filters
    cfg = SAMPLES[sample_key]
    ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    K = int(dino_inf["soft_probs"].shape[1])
    soft = dino_inf["soft_probs"]; ass = dino_inf["assigns"]
    avgs = []
    sizes = []
    for c in range(K):
        idx = np.where(ass == c)[0]
        sizes.append(int(idx.size))
        if idx.size == 0:
            avgs.append(None); continue
        s = soft[idx, c]
        top = idx[np.argsort(-s)[:min(top_n, len(idx))]]
        patterns = np.stack([ds.get_raw(int(i)) for i in top],
                              0).astype(np.float32)
        avg = patterns.mean(axis=0)
        try:
            v = float(cfg.get("vmax", 5.0))
            tmp = np.clip(avg / max(v, 1e-6), 0.0, 1.0)
            tmp = apply_sample_filters(tmp, cfg)
            avg = tmp * v
        except Exception:
            pass
        avgs.append(avg.astype(np.float32))
    return avgs, sizes


def _grains_per_class(sample_key, dino_inf, top_n_grains=5):
    """For each class, return up to top_n_grains largest connected
    components.  Returns list of (cls, grain_id, grain_mask, n_pix)."""
    from scipy.ndimage import label
    from data import SAMPLES
    Ny, Nx = SAMPLES[sample_key]["scan_shape"]
    K = int(dino_inf["soft_probs"].shape[1])
    ass_grid = np.asarray(dino_inf["assigns"]).reshape(Ny, Nx)
    grains = []
    for c in range(K):
        mask = (ass_grid == c)
        if not mask.any(): continue
        lab, _n = label(mask)
        if _n == 0: continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        order = np.argsort(-sizes)
        for gid in [int(g) for g in order[:top_n_grains] if sizes[g] > 0]:
            gm = (lab == gid)
            grains.append((c, gid, gm, int(gm.sum())))
    return grains


def _grain_average(sample_key, grain_mask):
    """Raw-detector grain-averaged pattern.  Mirrors posthoc helper."""
    from data import SAMPLES, LoadPRZ, apply_sample_filters
    cfg = SAMPLES[sample_key]
    ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    Ny, Nx = cfg["scan_shape"]
    flat_idx = np.where(grain_mask.ravel())[0]
    pats = np.stack([ds.get_raw(int(i)) for i in flat_idx],
                       0).astype(np.float32)
    avg = pats.mean(axis=0)
    try:
        v = float(cfg.get("vmax", 5.0))
        tmp = np.clip(avg / max(v, 1e-6), 0.0, 1.0)
        tmp = apply_sample_filters(tmp, cfg)
        avg = tmp * v
    except Exception:
        pass
    return avg


def run_mp_acom_classes(sample_key, dino_inf, crystals, outdir):
    """§3 — Multi-phase ACOM on every DINO class average."""
    from gui_app.acom_core import (acom_multiphase_batch,
                                        zone_axis_from_matrix)
    _log("§3: ACOM on class averages …")
    K = int(dino_inf["soft_probs"].shape[1])
    avgs, sizes = _class_averages_raw(sample_key, dino_inf)
    patterns, labels, classes = [], [], []
    for c in range(K):
        if avgs[c] is None: continue
        patterns.append(avgs[c]); labels.append(f"N={sizes[c]}"); classes.append(c)
    if not patterns:
        return None
    mp = acom_multiphase_batch(crystals, patterns,
                                    inv_ang_per_pixel=INV_ANG_PER_PX,
                                    detect_kw=ACOM_DETECT_KW)
    # Card grid
    N = len(patterns); cols = 4; rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7,
                                                          rows * 3.0),
                                squeeze=False)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    names = mp["phase_names"]
    for k in range(N):
        ax = axes[k // cols][k % cols]
        ax.imshow(np.log1p(np.clip(patterns[k], 0, None)),
                    cmap="inferno", aspect="equal",
                    interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        pi = int(mp["phase_id"][k])
        if pi < 0:
            tag = "neither" if pi == -1 else "ambiguous"
            cs = "  ".join(f"{names[p]}={mp['corr_per_phase'][p, k]:.3f}"
                              for p in range(len(names)))
            ax.set_title(f"p{classes[k]}  {tag}  ({cs})\n{labels[k]}",
                            fontsize=8, color="#888")
        else:
            R = mp["rmat_per_phase"][pi, k]
            za, mis = zone_axis_from_matrix(R)
            ax.set_title(
                f"p{classes[k]}  → {names[pi]}  ZA=[{za[0]} {za[1]} {za[2]}]  "
                f"corr={mp['corr_per_phase'][pi, k]:.3f}\n{labels[k]}",
                fontsize=8, color=palette[pi % len(palette)])
            for spine in ax.spines.values():
                spine.set_edgecolor(palette[pi % len(palette)])
                spine.set_linewidth(2.0)
    for k in range(N, rows * cols):
        axes[k // cols][k % cols].set_axis_off()
    fig.suptitle(
        f"§3 — Multi-phase ACOM on DINO class averages  "
        f"(phases = {names},  thr = 0.0,  calib = "
        f"{INV_ANG_PER_PX:.5g} 1/Å/px)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_fig(fig, os.path.join(outdir, "03_class_avg_acom.png"))

    # CSV table
    rows_csv = [("class", "n_pixels", "winning_phase",
                   "winning_corr", "ZA",
                   *[f"corr_{n}" for n in names])]
    for k in range(N):
        pi = int(mp["phase_id"][k])
        if pi < 0:
            row = (classes[k], sizes[classes[k]],
                     "neither" if pi == -1 else "ambiguous",
                     float(np.nan), "")
        else:
            R = mp["rmat_per_phase"][pi, k]
            za, mis = zone_axis_from_matrix(R)
            row = (classes[k], sizes[classes[k]], names[pi],
                     float(mp["corr_per_phase"][pi, k]),
                     f"[{za[0]} {za[1]} {za[2]}]")
        row = list(row) + [float(mp["corr_per_phase"][p, k])
                              for p in range(len(names))]
        rows_csv.append(tuple(row))
    with open(os.path.join(outdir, "03_class_avg_table.csv"), "w",
                newline="") as f:
        csv.writer(f).writerows(rows_csv)
    return mp


def run_mp_acom_grains(sample_key, dino_inf, crystals, outdir,
                            top_n_grains=5):
    """§4 — Multi-phase ACOM on top-N largest grains per class."""
    from gui_app.acom_core import (acom_multiphase_batch,
                                        zone_axis_from_matrix)
    _log("§4: ACOM on top grains …")
    grains = _grains_per_class(sample_key, dino_inf,
                                    top_n_grains=top_n_grains)
    if not grains:
        return None
    patterns, classes, gids, npx = [], [], [], []
    for (c, gid, gm, n) in grains:
        patterns.append(_grain_average(sample_key, gm))
        classes.append(c); gids.append(gid); npx.append(n)
    mp = acom_multiphase_batch(crystals, patterns,
                                    inv_ang_per_pixel=INV_ANG_PER_PX,
                                    detect_kw=ACOM_DETECT_KW)
    N = len(patterns); cols = 5; rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4,
                                                          rows * 2.8),
                                squeeze=False)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    names = mp["phase_names"]
    for k in range(N):
        ax = axes[k // cols][k % cols]
        ax.imshow(np.log1p(np.clip(patterns[k], 0, None)),
                    cmap="inferno", aspect="equal",
                    interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        pi = int(mp["phase_id"][k])
        if pi < 0:
            tag = "neither" if pi == -1 else "ambig"
            ax.set_title(
                f"p{classes[k]} g{gids[k]} {npx[k]}px  {tag}",
                fontsize=7, color="#888")
        else:
            R = mp["rmat_per_phase"][pi, k]
            za, mis = zone_axis_from_matrix(R)
            ax.set_title(
                f"p{classes[k]} g{gids[k]} {npx[k]}px\n"
                f"{names[pi]} ZA=[{za[0]} {za[1]} {za[2]}] "
                f"c={mp['corr_per_phase'][pi, k]:.3f}",
                fontsize=7, color=palette[pi % len(palette)])
            for spine in ax.spines.values():
                spine.set_edgecolor(palette[pi % len(palette)])
                spine.set_linewidth(1.5)
    for k in range(N, rows * cols):
        axes[k // cols][k % cols].set_axis_off()
    fig.suptitle(
        f"§4 — Multi-phase ACOM on top-{top_n_grains} grains per class  "
        f"(phases = {names})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, os.path.join(outdir, "04_grain_acom.png"))

    # CSV
    rows_csv = [("class", "grain_id", "n_pixels", "winning_phase",
                   "winning_corr", "ZA",
                   *[f"corr_{n}" for n in names])]
    for k in range(N):
        pi = int(mp["phase_id"][k])
        if pi < 0:
            row = (classes[k], gids[k], npx[k],
                     "neither" if pi == -1 else "ambiguous",
                     float(np.nan), "")
        else:
            R = mp["rmat_per_phase"][pi, k]
            za, mis = zone_axis_from_matrix(R)
            row = (classes[k], gids[k], npx[k], names[pi],
                     float(mp["corr_per_phase"][pi, k]),
                     f"[{za[0]} {za[1]} {za[2]}]")
        row = list(row) + [float(mp["corr_per_phase"][p, k])
                              for p in range(len(names))]
        rows_csv.append(tuple(row))
    with open(os.path.join(outdir, "04_grain_table.csv"), "w",
                newline="") as f:
        csv.writer(f).writerows(rows_csv)
    return mp


# ---------------------------------------------------------------------------
# §5 — full-dataset α / γ / amorphous map
# ---------------------------------------------------------------------------
def _amorphous_gate_via_peakcount(cube, stride, min_peaks=6):
    """Per-position blob_log count, used to label 'amorphous' (NA)."""
    from gui_app.acom_core import detect_peaks_2d
    Ny, Nx, H, W = cube.shape
    n_peaks = np.zeros((Ny, Nx), dtype=np.int32)
    cy, cx = H / 2.0, W / 2.0
    t0 = time.time()
    for rx in range(Ny):
        for ry in range(Nx):
            if (rx % stride) or (ry % stride):
                continue
            pat = np.asarray(cube[rx, ry], dtype=np.float32)
            peaks = detect_peaks_2d(pat, **ACOM_DETECT_KW)
            if peaks.size:
                rad = np.sqrt((peaks[:, 0] - cy) ** 2
                                + (peaks[:, 1] - cx) ** 2)
                n_peaks[rx, ry] = int((rad > 4.0).sum())
        if (rx & 7) == 0:
            _log(f"     amorphous-gate row {rx+1}/{Ny}  "
                    f"({time.time()-t0:.0f}s)")
    return n_peaks


def run_mp_acom_full(sample_key, crystals, outdir, stride=FULL_STRIDE,
                          min_peaks_for_crystal=6):
    from gui_app.acom_core import (acom_multiphase_full_dataset,
                                        zone_axis_from_matrix)
    from data import SAMPLES
    cfg = SAMPLES[sample_key]
    _log(f"§5: α/γ/NA full-dataset ACOM, stride={stride} …")
    # Force RAM-load: mmap path on Windows for the 2.4 GB cube triggers
    # OSError(22) mid-match_orientations after random number of calls.
    # Loading fully into memory is reliable and only adds ~5 s startup.
    _log(f"  RAM-loading cube …")
    cube = np.array(np.load(cfg["path"], allow_pickle=True),
                       dtype=np.float32, copy=True)
    _log(f"  cube ready: shape={cube.shape}  "
            f"size={cube.nbytes/1e9:.2f} GB")
    t0 = time.time()
    def _prog(done, total, stage):
        if stage == "detect" and done % 256 == 0:
            dt = time.time() - t0
            eta = (dt / max(done, 1)) * (total - done)
            _log(f"     detect {done}/{total}  ({dt:.0f}s, ETA {eta:.0f}s)")
    mp = acom_multiphase_full_dataset(
        crystals, cube, inv_ang_per_pixel=INV_ANG_PER_PX,
        detect_kw=ACOM_DETECT_KW, subsample_stride=stride,
        progress_cb=_prog)
    # Amorphous gate: positions with < min_peaks Bragg-detected (excl
    # BF disk) get relabelled as -1 ("amorphous/NA") regardless of
    # which phase numerically won.
    _log(f"     amorphous-gate (min_peaks={min_peaks_for_crystal}) …")
    n_peaks = _amorphous_gate_via_peakcount(cube, stride,
                                                  min_peaks=min_peaks_for_crystal)
    amorphous_mask = (n_peaks < min_peaks_for_crystal)
    mp["phase_id"][amorphous_mask] = -1
    mp["n_peaks_per_position"] = n_peaks
    np.save(os.path.join(outdir, "05_n_peaks_per_position.npy"),
              n_peaks)
    Ny, Nx = mp["scan_shape"]
    names = mp["phase_names"]; n_ph = len(names)
    phase_id = mp["phase_id"]
    corr_win = mp["winning_corr"]
    rmat_win = mp["winning_rmat"]
    # α = blue, γ = red, NA / amorphous = mid-grey (distinguishable
    # from black 'unprocessed').
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    NA_COLOR = (0.55, 0.55, 0.55)
    phase_rgb = np.zeros((Ny, Nx, 3), dtype=np.float32)
    for pi in range(n_ph):
        m = (phase_id == pi)
        if m.any(): phase_rgb[m] = to_rgb(palette[pi])
    na_mask = (phase_id == -1)
    if na_mask.any():
        phase_rgb[na_mask] = NA_COLOR
    # ZA RGB
    za_rgb = np.zeros((Ny, Nx, 3), dtype=np.float32)
    za_map = {}
    for rx in range(Ny):
        for ry in range(Nx):
            pi = int(phase_id[rx, ry])
            if pi < 0: continue
            R = rmat_win[rx, ry]
            if not np.isfinite(R).all(): continue
            za, _ = zone_axis_from_matrix(R)
            key = (pi, za)
            if key not in za_map:
                za_map[key] = plt.get_cmap("tab20")(len(za_map) % 20)[:3]
            za_rgb[rx, ry] = za_map[key]
    # corr diff
    if n_ph >= 2:
        cp_sorted = np.sort(-mp["corr_per_phase"], axis=0)
        corr_diff = (-cp_sorted[0] - (-cp_sorted[1])).astype(np.float32)
    else:
        corr_diff = np.zeros_like(corr_win)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.0))
    a_mask, a_corr, a_za, a_diff = axes.ravel()
    a_mask.imshow(phase_rgb, interpolation="nearest", aspect="auto")
    lh = []
    for pi, n in enumerate(names):
        frac = float((phase_id == pi).sum()) / max(Ny * Nx, 1)
        lh.append(Patch(color=palette[pi],
                          label=f"{n} {frac*100:.1f}%"))
    f_n = float((phase_id == -1).sum()) / max(Ny * Nx, 1)
    if f_n > 0:
        lh.append(Patch(color=NA_COLOR,
                          label=f"amorphous / NA {f_n*100:.1f}%"))
    a_mask.legend(handles=lh, loc="lower right", fontsize=8,
                      framealpha=0.75)
    a_mask.set_title("phase mask", fontsize=11)
    a_mask.set_xticks([]); a_mask.set_yticks([])
    im = a_corr.imshow(corr_win, cmap="viridis",
                          interpolation="nearest", aspect="auto")
    a_corr.set_title("winning corr (top-1)", fontsize=11)
    a_corr.set_xticks([]); a_corr.set_yticks([])
    fig.colorbar(im, ax=a_corr, fraction=0.045, pad=0.02)
    a_za.imshow(za_rgb, interpolation="nearest", aspect="auto")
    a_za.set_title("winning ZA (phase × ZA hash)", fontsize=11)
    a_za.set_xticks([]); a_za.set_yticks([])
    im2 = a_diff.imshow(corr_diff, cmap="magma",
                            interpolation="nearest", aspect="auto")
    a_diff.set_title("corr(top1) − corr(top2)", fontsize=11)
    a_diff.set_xticks([]); a_diff.set_yticks([])
    fig.colorbar(im2, ax=a_diff, fraction=0.045, pad=0.02)
    fig.suptitle(
        f"§5 — Full-dataset multi-phase ACOM  stride={stride}  "
        f"phases={names}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, os.path.join(outdir, "05_full_dataset_mp_acom.png"))
    # Save phase_id grid for downstream use.
    np.save(os.path.join(outdir, "05_full_phase_id.npy"), phase_id)
    np.save(os.path.join(outdir, "05_full_winning_corr.npy"), corr_win)
    return mp


# ---------------------------------------------------------------------------
# §5_v2 — full-dataset CrystalPhase NNLS fit (the proper py4DSTEM
# multi-phase API).  Replaces the corr-argmax approach with NNLS
# fitting that produces per-phase weights + a residual + a "dominant
# phase" map that's directly comparable to the py4DSTEM colab.
# ---------------------------------------------------------------------------
def run_phase_quantify_full(sample_key, crystals_dict, outdir,
                                  stride=1):
    from py4DSTEM.process.diffraction.crystal_phase import CrystalPhase
    from gui_app.acom_core import build_bragg_vectors, detect_peaks_2d
    from data import SAMPLES
    cfg = SAMPLES[sample_key]
    _log(f"§5 (NNLS): full-dataset CrystalPhase fit, stride={stride} …")
    _log("  RAM-loading cube …")
    cube = np.array(np.load(cfg["path"], allow_pickle=True),
                       dtype=np.float32, copy=True)
    Ny, Nx, H, W = cube.shape
    center = (H / 2.0, W / 2.0)
    _log(f"  cube ready: shape={cube.shape}  size={cube.nbytes/1e9:.2f} GB")

    # 1) Per-position peak detection.
    peaks_all = []
    centers_all = []
    t0 = time.time()
    for rx in range(Ny):
        for ry in range(Nx):
            if (rx % stride) or (ry % stride):
                peaks_all.append(np.zeros((0, 3), dtype=float))
                centers_all.append(center); continue
            peaks_all.append(
                detect_peaks_2d(cube[rx, ry], **ACOM_DETECT_KW))
            centers_all.append(center)
        if (rx & 7) == 0:
            _log(f"     detect row {rx+1}/{Ny}  "
                    f"({time.time()-t0:.0f}s)")

    bv = build_bragg_vectors(peaks_all, centers=centers_all,
                                  inv_ang_per_pixel=INV_ANG_PER_PX,
                                  Rshape=(Ny, Nx))

    # 2) match_orientations per phase populates crystal.orientation_map.
    #    progress_bar disabled — tqdm streams thousands of lines to
    #    stderr which can EINVAL the PowerShell pipe on long runs.
    names = list(crystals_dict.keys())
    crystals = list(crystals_dict.values())
    for n, cr in zip(names, crystals):
        _log(f"  match_orientations [{n}] (no progress bar) …")
        t1 = time.time()
        cr.match_orientations(bv, progress_bar=False)
        _log(f"     done ({time.time()-t1:.0f}s)")

    # 3) CrystalPhase + quantify_phase NNLS fit.
    _log("  building CrystalPhase + quantify_phase NNLS fit …")
    cp = CrystalPhase(crystals, crystal_names=names,
                         name=f"IMC ({'+'.join(names)})")
    t1 = time.time()
    cp.quantify_phase(bv, k_max=ACOM_K_MAX,
                         progress_bar=False, **QUANT_KW)
    _log(f"  quantify_phase done ({time.time()-t1:.0f}s)")

    # 4) Save data arrays for re-analysis.
    np.save(os.path.join(outdir, "05_phase_weights.npy"),
              cp.phase_weights)
    np.save(os.path.join(outdir, "05_phase_residuals.npy"),
              cp.phase_residuals)
    np.save(os.path.join(outdir, "05_phase_reliability.npy"),
              cp.phase_reliability)

    # 5) Plot dominant phase + weight maps + residual map.
    try:
        fig, _ = cp.plot_dominant_phase(figsize=(7.5, 7.5),
                                              returnfig=True,
                                              print_fractions=True,
                                              ticks=False,
                                              legend_add=True)
        _save_fig(fig, os.path.join(outdir, "05_dominant_phase.png"))
    except Exception as e:
        _log(f"  plot_dominant_phase failed: {e!r}")

    try:
        fig, _ = cp.plot_phase_weights(figsize=(5 * len(names), 5),
                                             returnfig=True)
        _save_fig(fig, os.path.join(outdir, "05_phase_weights_panels.png"))
    except Exception as e:
        _log(f"  plot_phase_weights failed: {e!r}")

    # Residual map directly (low residual = good fit).
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    im = ax.imshow(cp.phase_residuals, cmap="magma",
                     interpolation="nearest", aspect="auto")
    ax.set_title("phase fit residual (lower = better)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    _save_fig(fig, os.path.join(outdir, "05_phase_residuals.png"))

    return cp, bv


# ---------------------------------------------------------------------------
# §8_v2 — per-class fit overlay using CrystalPhase.quantify_single_pattern
# (gray-circle experimental peaks + colored triangles per phase, matching
# the colab notebook the user referenced).
# ---------------------------------------------------------------------------
def run_class_avg_phase_fits(sample_key, dino_inf, crystals_dict,
                                  outdir):
    from py4DSTEM.process.diffraction.crystal_phase import CrystalPhase
    from gui_app.acom_core import build_bragg_vectors, detect_peaks_2d
    _log("§8 (NNLS): per-class CrystalPhase fit + overlay …")
    avgs, sizes = _class_averages_raw(sample_key, dino_inf)
    classes_active = [c for c in range(len(avgs)) if avgs[c] is not None]
    N = len(classes_active)
    H, W = avgs[classes_active[0]].shape
    center = (H / 2.0, W / 2.0)

    peaks_all = [detect_peaks_2d(avgs[c], **ACOM_DETECT_KW)
                  for c in classes_active]
    bv = build_bragg_vectors(peaks_all,
                                  centers=[center] * N,
                                  inv_ang_per_pixel=INV_ANG_PER_PX,
                                  Rshape=(N, 1))

    names = list(crystals_dict.keys())
    crystals = list(crystals_dict.values())
    for n, cr in zip(names, crystals):
        _log(f"  match_orientations [{n}] …")
        cr.match_orientations(bv, progress_bar=False)
    cp = CrystalPhase(crystals, crystal_names=names)

    # Per-class quantify_single_pattern with plot.
    overlays_paths = []
    fit_summary = [("class", "n_pixels",
                      *[f"weight_{n}" for n in names],
                      "residual", "reliability")]
    for i, c in enumerate(classes_active):
        try:
            fig, ax = cp.quantify_single_pattern(
                bv, xy_position=(i, 0),
                k_max=ACOM_K_MAX,
                plot_result=True, returnfig=True, verbose=False,
                scale_markers_experiment=0.02,
                scale_markers_calculated=100,
                figsize=(7.5, 5.5),
                **QUANT_KW)
            try:
                ax.set_title(f"DINO class p{c}  N={sizes[c]}",
                                fontsize=11)
            except Exception: pass
            out_png = os.path.join(outdir,
                                       f"08_class_p{c:02d}_fit.png")
            _save_fig(fig, out_png)
            overlays_paths.append(os.path.basename(out_png))
        except Exception as e:
            _log(f"  class p{c} fit failed: {e!r}")
            overlays_paths.append(None)
        # Tabulate the per-class weights from cp.phase_weights
        try:
            w = cp.phase_weights[i, 0]
            # collapse fits to per-phase totals
            per_phase = np.zeros(len(names))
            for fi, (pi, _) in enumerate(cp.crystal_identity):
                per_phase[pi] += float(w[fi])
            fit_summary.append((c, sizes[c],
                                  *[float(v) for v in per_phase],
                                  float(cp.phase_residuals[i, 0]),
                                  float(cp.phase_reliability[i, 0])))
        except Exception:
            fit_summary.append((c, sizes[c], *[np.nan]*len(names),
                                  np.nan, np.nan))
    # Stitch the per-class PNGs into a single grid panel for the report.
    if overlays_paths:
        cols = 3
        rows = (N + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                        figsize=(cols * 5.0,
                                                  rows * 4.0),
                                        squeeze=False)
        for k in range(rows * cols):
            ax = axes[k // cols][k % cols]
            ax.set_xticks([]); ax.set_yticks([])
            if k < len(overlays_paths) and overlays_paths[k]:
                try:
                    img = plt.imread(os.path.join(outdir,
                                                       overlays_paths[k]))
                    ax.imshow(img); ax.set_axis_off()
                except Exception:
                    ax.text(0.5, 0.5, "load err", ha="center",
                              va="center", transform=ax.transAxes)
            else:
                ax.set_axis_off()
        fig.suptitle(
            "§8 — per-class CrystalPhase NNLS fits  "
            "(gray = experimental peaks, ▽ red = α-IMC, ▲ blue = γ-IMC)",
            fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        _save_fig(fig, os.path.join(outdir, "08_class_fits_grid.png"))

    # CSV
    with open(os.path.join(outdir, "08_class_phase_fits.csv"),
                "w", newline="") as f:
        csv.writer(f).writerows(fit_summary)
    return cp


# ---------------------------------------------------------------------------
# §8 — per-class predicted-pattern overlay (py4DSTEM
# plot_diffraction_pattern style: predicted Bragg disks over the
# experimental peaks).  One panel per active DINO class, run
# against BOTH α and γ so the user can compare fit quality directly.
# ---------------------------------------------------------------------------
def run_class_avg_fit_overlays(sample_key, dino_inf, crystals, outdir):
    from gui_app.acom_core import (acom_single_pattern,
                                        zone_axis_from_matrix)
    from py4DSTEM.process.diffraction import plot_diffraction_pattern
    _log("§8: per-class fit overlays …")
    avgs, sizes = _class_averages_raw(sample_key, dino_inf)
    K = len(avgs)
    # Skip empty classes.
    active = [c for c in range(K) if avgs[c] is not None]
    N = len(active)
    n_ph = len(crystals)
    cols = 2 * n_ph + 1     # [raw + α-fit + γ-fit + ...]
    rows = N
    fig, axes = plt.subplots(rows, cols,
                                figsize=(cols * 2.6, rows * 2.6),
                                squeeze=False)
    for ri, c in enumerate(active):
        pat = avgs[c]
        # raw + peaks
        ax = axes[ri][0]
        ax.imshow(np.log1p(np.clip(pat, 0, None)),
                    cmap="inferno", aspect="equal",
                    interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"p{c}  N={sizes[c]}", fontsize=9)
        for pi, (name, cr) in enumerate(crystals.items()):
            try:
                res = acom_single_pattern(
                    cr, pat,
                    inv_ang_per_pixel=INV_ANG_PER_PX,
                    detect_kw=ACOM_DETECT_KW)
                R = None
                ort = res["orientation"]
                v = getattr(ort, "matrix", None)
                if v is not None:
                    arr = np.asarray(v)
                    R = arr[0] if arr.ndim == 3 else arr
                za, mis = zone_axis_from_matrix(R)
            except Exception as e:
                res = None
                za, mis = ((0, 0, 0), float("nan"))
            # Overlay axes
            ax_o = axes[ri][1 + 2 * pi]
            ax_o.set_xticks([]); ax_o.set_yticks([])
            if res is None or res["fit_pattern"] is None:
                ax_o.text(0.5, 0.5,
                            f"{name}: fit failed",
                            ha="center", va="center", fontsize=8,
                            color="#a33", transform=ax_o.transAxes)
            else:
                try:
                    plot_diffraction_pattern(
                        res["fit_pattern"],
                        bragg_peaks_compare=res["calibrated_pl"],
                        scale_markers=600,
                        scale_markers_compare=3e4,
                        min_marker_size=1, figsize=(4, 4),
                        input_fig_handle=(fig, ax_o))
                except Exception as e:
                    ax_o.text(0.5, 0.5, f"plot err",
                                 ha="center", va="center", fontsize=8,
                                 color="#a33", transform=ax_o.transAxes)
                ax_o.set_title(
                    f"{name}: ZA=[{za[0]} {za[1]} {za[2]}]  "
                    f"corr={res['corr']:.3f}",
                    fontsize=9)
            # On the right of each overlay, an "info" mini-axis.
            ax_i = axes[ri][2 + 2 * pi]
            ax_i.set_axis_off()
            if res is not None and res["fit_pattern"] is not None:
                # Tabulate: #experimental peaks, #predicted Bragg.
                n_exp = len(res["peaks"])
                try:
                    n_pred = len(np.asarray(res["fit_pattern"].data["qx"]))
                except Exception:
                    n_pred = 0
                ax_i.text(0.05, 0.5,
                            f"n_exp = {n_exp}\n"
                            f"n_pred = {n_pred}\n"
                            f"corr = {res['corr']:.3f}\n"
                            f"miso = {mis:.1f}°",
                            ha="left", va="center", fontsize=8,
                            family="monospace",
                            transform=ax_i.transAxes)
    fig.suptitle(
        "§8 — predicted-pattern overlay per DINO class  "
        "(experimental peaks ⊕ predicted Bragg disks per phase)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save_fig(fig, os.path.join(outdir, "08_class_fit_overlays.png"))


# ---------------------------------------------------------------------------
# §6 — crystallinity metrics per class
# ---------------------------------------------------------------------------
def run_crystallinity_per_class(sample_key, dino_inf, outdir):
    """For each class avg, compute:
    - n_peaks detected (blob_log)
    - peak/background ratio (top-10 peak amps over median floor)
    - sharpest peak FWHM proxy (≈ blob sigma)
    """
    from gui_app.acom_core import detect_peaks_2d
    _log("§6: crystallinity metrics …")
    avgs, sizes = _class_averages_raw(sample_key, dino_inf)
    K = len(avgs)
    rows = [("class", "n_pixels", "n_peaks", "peak_to_background",
              "median_intensity", "max_intensity")]
    for c in range(K):
        if avgs[c] is None:
            rows.append((c, sizes[c], 0, np.nan, np.nan, np.nan)); continue
        pat = avgs[c]
        peaks = detect_peaks_2d(pat, **ACOM_DETECT_KW)
        med = float(np.median(pat))
        mx  = float(pat.max())
        if peaks.size:
            cy, cx = pat.shape[0] / 2, pat.shape[1] / 2
            # exclude BF disk
            keep = np.sqrt((peaks[:, 0] - cy) ** 2
                            + (peaks[:, 1] - cx) ** 2) > 4.0
            peaks_n = int(keep.sum())
            if keep.any():
                amps = peaks[keep, 2]
                pb = float(np.sort(amps)[-min(10, len(amps)):].mean()
                            / max(med, 1e-9))
            else:
                pb = float("nan")
        else:
            peaks_n = 0; pb = float("nan")
        rows.append((c, sizes[c], peaks_n, pb, med, mx))
    with open(os.path.join(outdir, "06_crystallinity.csv"), "w",
                newline="") as f:
        csv.writer(f).writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# §7 — markdown + DOCX
# ---------------------------------------------------------------------------
def assemble_report(run_dir, sample_key, outdir, dino_inf,
                       nmf_result, dino_vs_nmf_metrics,
                       mp_classes, mp_grains, mp_full,
                       crystallinity_rows):
    _log("§7: writing markdown report …")
    K_active = int(dino_inf["soft_probs"].shape[1])
    pbar = dino_inf["soft_probs"].mean(axis=0)
    Heff = -np.sum(pbar * np.log(np.maximum(pbar, 1e-12)))
    K_eff = float(np.exp(Heff))
    # Occupancy
    occ = np.bincount(dino_inf["assigns"], minlength=K_active)
    occ_frac = occ / occ.sum()
    occ_sorted = sorted(zip(range(K_active), occ.tolist(),
                                occ_frac.tolist()),
                            key=lambda r: -r[1])
    md_lines = []
    md_lines += [
        f"# IMC_SI5 — full analysis report",
        f"",
        f"**Run** : `{run_dir}`  ",
        f"**Sample** : `{sample_key}`  ",
        f"**Generated** : {datetime.now().isoformat(timespec='seconds')}  ",
        f"",
        f"## Summary",
        f"",
        f"- DINO run config : m = 0.97, seed = 42, K_nominal = 60",
        f"- K_active (post-prune) = **{K_active}**",
        f"- K_eff (Shannon)        = **{K_eff:.2f}**",
        f"- Top-3 class occupancy  = "
            f"{occ_sorted[0][2]*100:.1f}% · "
            f"{occ_sorted[1][2]*100:.1f}% · "
            f"{occ_sorted[2][2]*100:.1f}% · …",
        f"- Calibration : 0.0185 nm⁻¹/px → 0.00185 1/Å/px",
        f"- Phases tested in ACOM : α-IMC + γ-IMC (`{os.path.basename(ALPHA_CIF)}`, `{os.path.basename(GAMMA_CIF)}`)",
        f"",
        f"## §1 — Wang 2024 polar NMF + clustering",
        f"",
        f"![nmf_err](01a_nmf_err_curve.png)",
        f"",
        (f"NMF selected **n = {nmf_result['n_components']}** components by knee."
            if nmf_result is not None
            else "NMF outputs (n components, components grid, cluster maps) from a prior run."),
        f"",
        f"![nmf_components](01b_nmf_components.png)",
        f"",
        f"![nmf_cluster_maps](01c_nmf_cluster_maps.png)",
        f"",
        f"## §2 — DINO vs NMF (KMeans)",
        f"",
        f"![nmf_vs_dino](02_nmf_vs_dino.png)",
        f"",
        f"| metric | value |",
        f"|---|---|",
        (f"| K_DINO        | {dino_vs_nmf_metrics['K_dino']} |"
            if dino_vs_nmf_metrics else "| _metrics unavailable_ | — |"),
        (f"| K_NMF         | {dino_vs_nmf_metrics['K_nmf']} |"
            if dino_vs_nmf_metrics else ""),
        (f"| ARI           | {dino_vs_nmf_metrics['ARI']:.4f} |"
            if dino_vs_nmf_metrics else ""),
        (f"| NMI           | {dino_vs_nmf_metrics['NMI']:.4f} |"
            if dino_vs_nmf_metrics else ""),
        (f"| V-measure     | {dino_vs_nmf_metrics['V_measure']:.4f} |"
            if dino_vs_nmf_metrics else ""),
        f"",
        f"## §3 — Multi-phase ACOM on DINO class averages",
        f"",
        f"![class_avg_acom](03_class_avg_acom.png)",
        f"",
        f"See `03_class_avg_table.csv` for the full table (per class: "
            f"winning phase, corr, ZA, per-phase corrs).",
        f"",
        f"## §4 — Multi-phase ACOM on top grains per class",
        f"",
        f"![grain_acom](04_grain_acom.png)",
        f"",
        f"See `04_grain_table.csv`.",
        f"",
        f"## §5 — CrystalPhase NNLS fit map (dense)",
        f"",
        f"Dominant phase at each scan position from py4DSTEM's "
            f"`CrystalPhase.quantify_phase` — NNLS fit of the "
            f"experimental Bragg vectors to (α + γ) simulated patterns. "
            f"Fraction text in the title; per-phase weight maps below.",
        f"",
        f"![dominant_phase](05_dominant_phase.png)",
        f"",
        f"![phase_weights_panels](05_phase_weights_panels.png)",
        f"",
        f"![phase_residuals](05_phase_residuals.png)",
        f"",
        f"Raw weights/residual/reliability saved to "
            f"`05_phase_weights.npy`, `05_phase_residuals.npy`, "
            f"`05_phase_reliability.npy`.",
        f"",
        f"## §8 — Per-class NNLS fit overlays",
        f"",
        f"For every DINO class, the **gray circles** are the "
            f"experimental peaks (size ∝ intensity), **▽ red triangles** "
            f"are α-IMC predictions, **▲ blue triangles** are γ-IMC.  "
            f"`phase_weight`, fit residual and reliability per class "
            f"are tabulated in `08_class_phase_fits.csv`.",
        f"",
        f"![class_fits_grid](08_class_fits_grid.png)",
        f"",
        f"## §6 — Crystallinity metrics per class",
        f"",
        f"| class | n_pixels | n_peaks | peak/bg | median I | max I |",
        f"|---:|---:|---:|---:|---:|---:|",
    ]
    for row in crystallinity_rows[1:]:
        c, n_px, n_peaks, pb, med, mx = row
        pb_s = f"{pb:.2f}" if isinstance(pb, float) and pb == pb else "—"
        med_s = f"{med:.3g}" if isinstance(med, float) and med == med else "—"
        mx_s  = f"{mx:.3g}"  if isinstance(mx, float)  and mx == mx  else "—"
        md_lines.append(
            f"| {c} | {n_px} | {n_peaks} | {pb_s} | {med_s} | {mx_s} |")
    md_lines += [
        f"",
        f"## §7 — Crystallographic narrative",
        f"",
        f"_See companion narrative — assembled after inspecting the figures._",
        f"",
    ]
    md_path = os.path.join(outdir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    _log(f"wrote {md_path}")
    # DOCX export via existing helper (best-effort).
    try:
        sys.path.insert(0, os.path.join(REPO, "paper"))
        from md_to_docx import md_to_docx
        docx_path = os.path.join(outdir, "report.docx")
        md_to_docx(md_path, docx_path)
        _log(f"wrote {docx_path}")
    except Exception as e:
        _log(f"DOCX export skipped: {e!r}")
    return md_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global INV_ANG_PER_PX, ACOM_K_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="run dir (the one with eval/inference.npz)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <run>/_imc_report/)")
    ap.add_argument("--stride", type=int, default=FULL_STRIDE,
                    help="full-dataset ACOM stride (default 2)")
    ap.add_argument("--skip-full", action="store_true",
                    help="skip §5 (slow)")
    ap.add_argument("--skip-nmf", action="store_true",
                    help="skip §1+§2 (NMF + DINO-vs-NMF) — useful on re-run")
    ap.add_argument("--k-max", type=float, default=ACOM_K_MAX,
                    help="CIF structure-factor cap, 1/Å (default 0.5)")
    ap.add_argument("--inv-ang-per-px", type=float,
                    default=INV_ANG_PER_PX,
                    help=f"1/Å per pixel for the cube "
                         f"(default {INV_ANG_PER_PX})")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    out_dir = os.path.abspath(args.out or
                                  os.path.join(run_dir, "_imc_report"))
    _ensure_dir(out_dir)
    # Override module-level calibration from CLI.
    INV_ANG_PER_PX = float(args.inv_ang_per_px)
    ACOM_K_MAX     = float(args.k_max)
    _log(f"run = {run_dir}")
    _log(f"out = {out_dir}")
    _log(f"INV_ANG_PER_PX = {INV_ANG_PER_PX}  k_max = {ACOM_K_MAX} 1/Å")

    sample_key = _register_sample_from_lock(run_dir)
    dino_inf = _load_dino_inf(run_dir)
    _log(f"DINO inference loaded: assigns={dino_inf['assigns'].shape}  "
            f"soft={dino_inf['soft_probs'].shape}  "
            f"embeds={dino_inf['embeds'].shape}")

    nmf_result = None; metrics = None
    if not args.skip_nmf:
        nmf_result = run_nmf_and_clusters(sample_key, out_dir)
        render_cluster_maps(sample_key, nmf_result, out_dir)
        metrics, _C = compare_dino_vs_nmf(sample_key, nmf_result,
                                                dino_inf, out_dir)
    else:
        # Re-use cached metrics if a previous run wrote them.
        mp_metrics = os.path.join(out_dir, "02_metrics.json")
        if os.path.exists(mp_metrics):
            metrics = json.load(open(mp_metrics))
        _log("§1+§2 skipped (--skip-nmf)")

    # Build α + γ crystals (heavy — ~60 s combined)
    from gui_app.acom_core import load_crystal, prepare_crystal
    _log("building α + γ crystals …")
    t0 = time.time()
    cr_alpha = load_crystal(ALPHA_CIF)
    prepare_crystal(cr_alpha, k_max=ACOM_K_MAX,
                       plan_mode=ACOM_PLAN_MODE)
    cr_gamma = load_crystal(GAMMA_CIF)
    prepare_crystal(cr_gamma, k_max=ACOM_K_MAX,
                       plan_mode=ACOM_PLAN_MODE)
    _log(f"  ready ({time.time()-t0:.0f}s)")
    crystals = {"alpha": cr_alpha, "gamma": cr_gamma}

    # §3 class-avg ACOM
    mp_classes = run_mp_acom_classes(sample_key, dino_inf, crystals,
                                            out_dir)
    # §4 grain ACOM
    mp_grains  = run_mp_acom_grains(sample_key, dino_inf, crystals,
                                          out_dir, top_n_grains=5)
    # §5 — CrystalPhase NNLS full-dataset fit at the chosen stride.
    mp_full = None
    if not args.skip_full:
        try:
            mp_full, _ = run_phase_quantify_full(
                sample_key, crystals, out_dir,
                stride=int(args.stride))
        except Exception as e:
            _log(f"§5 failed: {e!r}")
            traceback.print_exc()

    # §6 crystallinity
    cryst_rows = run_crystallinity_per_class(sample_key, dino_inf,
                                                    out_dir)
    # §8 — per-class CrystalPhase NNLS fits with overlay plots.
    try:
        run_class_avg_phase_fits(sample_key, dino_inf, crystals,
                                        out_dir)
    except Exception as e:
        _log(f"§8 failed: {e!r}")
        traceback.print_exc()

    # §7 report
    assemble_report(run_dir, sample_key, out_dir, dino_inf,
                       nmf_result, metrics, mp_classes, mp_grains,
                       mp_full, cryst_rows)
    _log("ALL DONE.")


if __name__ == "__main__":
    main()
