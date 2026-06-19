"""cross_sample_classmaps.py -- produce cross-sample class-map figures.

Two figures, both designed for at-a-glance comparison:

  fig_cross_stage1_classmaps.png
      Rows = samples, cols = m values (Stage 1 at K=30, seed=42).
      A single figure showing how each sample's segmentation changes
      with m, side-by-side across samples.

  fig_cross_stage2_classmaps_<sample>.png  (one per sample)
      Rows = the top-m candidates of that sample, cols = K values
      (10, 15, 30, 60, 120).  Shows the K plateau visually.

All class maps are rendered with a consistent matplotlib tab20 (K<=20)
or viridis (K>20) palette discretised to the run's K.  Class indices
are NOT comparable across panels (DINO class ids are arbitrary), but
the spatial structure is -- which is the whole point.
"""
from __future__ import annotations
import argparse, csv, html, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

SAMPLE_ORDER = ["Na007b", "EuInAs_B100", "IMC_SI5"]
CUBE_SHAPES = {
    "Na007b":      (126, 100),
    "EuInAs_B100": (66,  396),
    "IMC_SI5":     (128, 128),
}


def _scan_shape(sample: str, ny_nx_fallback) -> tuple:
    s = CUBE_SHAPES.get(sample)
    if s: return s
    return ny_nx_fallback


def _load_assigns(run_dir: str):
    inf = os.path.join(run_dir, "eval", "inference.npz")
    if not os.path.exists(inf):
        return None, None
    try:
        d = np.load(inf, allow_pickle=True)
        return (np.asarray(d["assigns"]),
                int(np.asarray(d["soft_probs"]).shape[1]))
    except Exception:
        return None, None


def _classmap_panel(ax, assigns, K, scan_shape, title=""):
    Ny, Nx = scan_shape
    if assigns is None or assigns.size != Ny * Nx:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                  transform=ax.transAxes, fontsize=10, color="#888")
        ax.set_axis_off()
        return
    cm = assigns.reshape(Ny, Nx)
    cmap = plt.get_cmap("tab20" if K <= 20 else "viridis", K)
    # aspect='equal' preserves the native scan aspect ratio.
    ax.imshow(cm, cmap=cmap, vmin=-0.5, vmax=K - 0.5,
                interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, pad=3)


def _run_dir(root, sample, stage, m, K, seed):
    name = f"m{m:.4f}_seed{seed}_K{K}"
    return os.path.join(root, sample, stage, name)


def _K_eff_from(run_dir: str) -> float:
    inf = os.path.join(run_dir, "eval", "inference.npz")
    if not os.path.exists(inf): return float("nan")
    try:
        d = np.load(inf, allow_pickle=True)
        soft = np.asarray(d["soft_probs"])
        p_bar = soft.mean(axis=0)
        pb = np.clip(p_bar, 1e-12, 1.0)
        return float(np.exp(-(pb * np.log(pb)).sum()))
    except Exception:
        return float("nan")


def _aspect_panel(scan_shape, base_height_inch=2.5,
                      min_w=2.0, max_w=8.0):
    """Pick a panel (w, h) that preserves the scan's native aspect
    ratio.  Long-thin scans get wide-short cells; square scans get
    square cells.  Bounded so EuInAs (6:1) doesn't blow up the figure.
    """
    Ny, Nx = scan_shape
    aspect = Nx / max(Ny, 1)
    # Cap aspect-driven width — for EuInAs (aspect=6) we'd want a
    # 15-inch panel; that's too big.  Cap at max_w and tolerate
    # whitespace in the cell.
    w = max(min_w, min(max_w, base_height_inch * aspect))
    h = base_height_inch
    return w, h


# ---------------------------------------------------------------- Stage 1
def _kpi_box(ax, keff):
    """Small K_eff badge in the top-left of each panel."""
    if not np.isfinite(keff): return
    ax.text(0.04, 0.96, f"K_eff={keff:.1f}",
              transform=ax.transAxes, fontsize=10, color="white",
              va="top", ha="left",
              bbox=dict(facecolor="black", alpha=0.62,
                          edgecolor="none", pad=2.5))


def render_stage1_per_sample(root: str, sample: str, out_png: str,
                                  m_grid: list, K: int = 30,
                                  seeds=(42, 7)):
    """ONE figure per sample with native scan aspect preserved.

    Layout chosen from the scan's aspect:
      square-ish (Na007b, IMC):  rows = seeds, cols = m  (8 wide)
      wide-strip (EuInAs):       rows = m,     cols = seeds  (8 tall,
                                  each row is a wide horizontal strip)
    """
    stage_dir = os.path.join(root, sample, "stage1")
    if not os.path.isdir(stage_dir):
        print(f"[stage1] {sample}: no stage1 dir"); return
    Ny, Nx = CUBE_SHAPES.get(sample, (1, 1))
    aspect = Nx / max(Ny, 1)
    present = [s for s in seeds
                  if any(os.path.exists(os.path.join(
                      _run_dir(root, sample, "stage1", m, K, s),
                      "eval", "inference.npz"))
                          for m in m_grid)]
    if not present:
        print(f"[stage1] {sample}: no completed runs"); return

    # Wide-strip samples (aspect > ~2.5): transpose the grid so each
    # row is a strip rather than 8 strips side-by-side.
    transpose = aspect > 2.5

    if transpose:
        rows, cols = m_grid, present                  # rows = m
        row_label = lambda r: f"m = {r:g}"
        col_label = lambda c: f"seed = {c}"
        cell_h = 1.6
        cell_w = cell_h * aspect
    else:
        rows, cols = present, m_grid                  # rows = seeds
        row_label = lambda r: f"seed = {r}"
        col_label = lambda c: f"m = {c:g}"
        cell_h = 2.5
        cell_w = cell_h * aspect

    n_rows, n_cols = len(rows), len(cols)
    fig_w = cell_w * n_cols + 1.5
    fig_h = cell_h * n_rows + 1.3
    fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(fig_w, fig_h), dpi=130,
                                squeeze=False)
    for i, r_val in enumerate(rows):
        for j, c_val in enumerate(cols):
            m, seed = (r_val, c_val) if transpose else (c_val, r_val)
            run_dir = _run_dir(root, sample, "stage1", m, K, seed)
            assigns, K_pred = _load_assigns(run_dir)
            keff = _K_eff_from(run_dir)
            _classmap_panel(axes[i][j], assigns,
                                K_pred if K_pred else K, (Ny, Nx))
            _kpi_box(axes[i][j], keff)
            if j == 0:
                axes[i][j].set_ylabel(row_label(r_val),
                                          fontsize=12, labelpad=10,
                                          fontweight="bold")
            if i == 0:
                axes[i][j].set_title(col_label(c_val),
                                          fontsize=12, pad=6,
                                          fontweight="bold")
    fig.suptitle(f"{sample}   Stage 1 class maps   (K={K})",
                   fontsize=14, y=0.995, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150, facecolor="white",
                  bbox_inches="tight")
    plt.close(fig)
    print(f"[stage1] {out_png}", flush=True)


# ---------------------------------------------------------------- Stage 2
def _top_m_for_sample(root: str, sample: str):
    """Read run dirs in stage2 for this sample, infer top-m candidates
    from what was actually run."""
    stage_dir = os.path.join(root, sample, "stage2")
    if not os.path.isdir(stage_dir): return []
    ms = set()
    for d in os.listdir(stage_dir):
        try:
            m_part = d.split("_")[0]
            ms.add(float(m_part.replace("m", "")))
        except Exception: continue
    return sorted(ms)


def render_stage2(root: str, sample: str, out_png: str,
                     K_grid: list, seeds=(42, 7)):
    """One figure per sample; native scan aspect preserved.

    Layout chosen from the scan's aspect:
      square-ish:  rows = (m, seed) pairs, cols = K     (typical)
      wide-strip:  rows = K, cols = (m, seed) pairs     (transpose)
    """
    ms = _top_m_for_sample(root, sample)
    if not ms:
        print(f"[stage2] {sample}: no stage2 dirs"); return
    rows_present = []
    for m in ms:
        for seed in seeds:
            any_done = any(os.path.exists(os.path.join(
                _run_dir(root, sample, "stage2", m, K, seed),
                "eval", "inference.npz")) for K in K_grid)
            if any_done:
                rows_present.append((m, seed))
    if not rows_present:
        print(f"[stage2] {sample}: no completed stage2 runs"); return

    Ny, Nx = CUBE_SHAPES.get(sample, (1, 1))
    aspect = Nx / max(Ny, 1)
    transpose = aspect > 2.5

    if transpose:
        # Each K gets its own row; columns are (m, seed) pairs.
        rows, cols = K_grid, rows_present
        row_label = lambda r: f"K = {r}"
        col_label = lambda c: f"m={c[0]:g}\nseed={c[1]}"
        cell_h = 1.4
        cell_w = cell_h * aspect
    else:
        rows, cols = rows_present, K_grid
        row_label = lambda r: f"m={r[0]:g}\nseed={r[1]}"
        col_label = lambda c: f"K = {c}"
        cell_h = 2.4
        cell_w = cell_h * aspect

    n_rows, n_cols = len(rows), len(cols)
    fig_w = cell_w * n_cols + 1.6
    fig_h = cell_h * n_rows + 1.3
    fig, axes = plt.subplots(n_rows, n_cols,
                                figsize=(fig_w, fig_h), dpi=130,
                                squeeze=False)
    for i, r_val in enumerate(rows):
        for j, c_val in enumerate(cols):
            if transpose:
                K, (m, seed) = r_val, c_val
            else:
                (m, seed), K = r_val, c_val
            run_dir = _run_dir(root, sample, "stage2", m, K, seed)
            assigns, K_pred = _load_assigns(run_dir)
            keff = _K_eff_from(run_dir)
            _classmap_panel(axes[i][j], assigns,
                                K_pred if K_pred else K, (Ny, Nx))
            _kpi_box(axes[i][j], keff)
            if j == 0:
                axes[i][j].set_ylabel(row_label(r_val),
                                          fontsize=11, labelpad=10,
                                          fontweight="bold")
            if i == 0:
                axes[i][j].set_title(col_label(c_val),
                                          fontsize=11, pad=6,
                                          fontweight="bold")
    fig.suptitle(f"{sample}   Stage 2 K-plateau class maps",
                   fontsize=14, y=0.995, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150, facecolor="white",
                  bbox_inches="tight")
    plt.close(fig)
    print(f"[stage2] {out_png}", flush=True)


# ---------------------------------------------------------------- Patch report
def _append_to_report(report_path: str, new_figs: dict):
    """Append a section linking the cross-sample class-map figures
    to the existing cross_sample_report.html (if present)."""
    if not os.path.exists(report_path):
        return
    txt = open(report_path, encoding="utf-8").read()
    if "<!-- CROSS_CLASSMAPS_INSERTED -->" in txt:
        # Already appended; replace section.
        before = txt.split("<!-- CROSS_CLASSMAPS_INSERTED -->")[0]
        txt = before
    section = ["<!-- CROSS_CLASSMAPS_INSERTED -->"]
    section.append("<h2>7. Cross-sample class maps</h2>")
    section.append("<p>For each (sample, m, K, seed=42) cell with a "
                       "completed run, the class map is rendered with a "
                       "discretised tab20 / viridis palette.  Class "
                       "<i>indices</i> are not comparable across panels "
                       "(DINO class ids are arbitrary) but spatial "
                       "structure is.  Use these to spot partial-"
                       "collapse signatures and over-/under-clustering "
                       "by eye.</p>")
    for s, rel in new_figs.get("stage1", {}).items():
        section.append(f"<h3>Stage 1 -- {html.escape(s)}</h3>")
        section.append(f"<img src='{html.escape(rel)}' />")
        section.append(f"<div class=caption>{html.escape(s)} Stage 1. "
                          "Rows = seeds, columns = m values, K=30.  "
                          "Each panel is rendered at the sample's "
                          "native scan aspect ratio.</div>")
    for s, rel in new_figs.get("stage2", {}).items():
        section.append(f"<h3>Stage 2 -- {html.escape(s)}</h3>")
        section.append(f"<img src='{html.escape(rel)}' />")
        section.append(f"<div class=caption>{html.escape(s)} Stage 2. "
                          "Rows = (m, seed) pairs, columns = K values. "
                          "A clean data-driven-K plateau looks visually "
                          "similar across columns within a row.</div>")
    section.append("</body></html>")
    if "</body></html>" in txt:
        txt = txt.replace("</body></html>", "")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n".join(section))
    print(f"[report] cross-classmaps appended to {report_path}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    fig_dir = os.path.join(root, "_cross_figs")
    os.makedirs(fig_dir, exist_ok=True)

    # Fixed Stage 1 m grid (matches sweep spec).
    m_grid = [0.5, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
    K_grid = [10, 15, 30, 60, 120]

    rel_paths = {"stage1": {}, "stage2": {}}

    # Per-sample Stage 1 grids -- 2 seeds × m, native scan aspect.
    for s in SAMPLE_ORDER:
        out_s1 = os.path.join(fig_dir,
                                  f"fig_stage1_classmaps_{s}.png")
        render_stage1_per_sample(root, s, out_s1, m_grid,
                                       K=30, seeds=(42, 7))
        if os.path.exists(out_s1):
            rel_paths["stage1"][s] = os.path.join(
                "_cross_figs", f"fig_stage1_classmaps_{s}.png")

    # Per-sample Stage 2 grids -- (m, seed) rows × K, native aspect.
    for s in SAMPLE_ORDER:
        out_s2 = os.path.join(fig_dir,
                                  f"fig_stage2_classmaps_{s}.png")
        render_stage2(root, s, out_s2, K_grid, seeds=(42, 7))
        if os.path.exists(out_s2):
            rel_paths["stage2"][s] = os.path.join(
                "_cross_figs", f"fig_stage2_classmaps_{s}.png")

    # Append section to cross_sample_report.html if it exists
    report = os.path.join(root, "cross_sample_report.html")
    _append_to_report(report, rel_paths)


if __name__ == "__main__":
    main()
