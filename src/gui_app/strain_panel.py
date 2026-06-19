"""strain_panel.py -- Strain analysis between two class averages.

Workflow
--------
1. Pick a reference class (A) and a sample class (B).
2. Re-detect blobs on each class average using the current Detect-tab
   knobs (DoG / LoG / DoH).
3. Match A-blobs to B-blobs by nearest-neighbour after a global
   centering shift.
4. Fit an affine transform A·p + t mapping reference → sample, robustly
   via RANSAC.
5. Decompose the affine into rotation + symmetric strain matrix.
   Display ε_xx, ε_yy, ε_xy, rotation ω, plus the matched-pair overlay.

The user said: blob detection should be run on every pattern, not just
class averages.  This panel re-detects on the two class avgs, but it
also CHECKS whether a previous "Apply to whole class" run exists for
the picked classes.  If only avg-level detections are present, the
status warns the user that ACOM / per-pixel strain need a whole-class
run from the Detect tab.
"""
from __future__ import annotations
import os, json, time

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                 NavigationToolbar2Tk)
from matplotlib.patches import Circle


# ---------------------------------------------------------------------------
def _section(parent, title):
    f = ctk.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x", padx=2, pady=(8, 0))
    ctk.CTkLabel(f, text=title, font=("Segoe UI", 11, "bold")).pack(
        anchor="w")
    sep = ctk.CTkFrame(parent, height=2, fg_color=("#cccccc", "#444444"))
    sep.pack(fill="x", padx=2, pady=(0, 4))
    return parent


# ---------------------------------------------------------------------------
# Strain math
# ---------------------------------------------------------------------------
def match_nearest(pa: np.ndarray, pb: np.ndarray,
                   max_d: float = 12.0) -> np.ndarray:
    """Greedy 1-to-1 nearest-neighbour matching after subtracting per-set
    centroids (so a global shift doesn't ruin matches). Returns array of
    shape (M, 2) of index pairs (i_a, i_b)."""
    if pa.shape[0] == 0 or pb.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    a = pa - pa.mean(axis=0, keepdims=True)
    b = pb - pb.mean(axis=0, keepdims=True)
    # Pairwise distance.
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    pairs = []
    used_b = set()
    # Greedy: iterate by ascending distance.
    flat_idx = np.argsort(d, axis=None)
    for k in flat_idx:
        i, j = int(k // d.shape[1]), int(k % d.shape[1])
        if i in {p[0] for p in pairs} or j in used_b:
            continue
        if d[i, j] > max_d:
            break
        pairs.append((i, j))
        used_b.add(j)
    return np.asarray(pairs, dtype=np.int64) if pairs else np.zeros(
        (0, 2), dtype=np.int64)


def fit_affine_lsq(pa: np.ndarray, pb: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares fit: pb = A · pa + t. Returns (A, t, mean_residual)."""
    n = pa.shape[0]
    X = np.hstack([pa, np.ones((n, 1))])     # (n, 3)
    sol, *_ = np.linalg.lstsq(X, pb, rcond=None)   # (3, 2)
    A = sol[:2, :].T                          # (2, 2)
    t = sol[2, :]
    pred = pa @ A.T + t
    resid = float(np.linalg.norm(pb - pred, axis=1).mean())
    return A, t, resid


def ransac_affine(pa: np.ndarray, pb: np.ndarray,
                   n_iter: int = 200, inlier_thresh: float = 1.5
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """RANSAC affine fit. Picks 3 random pairs per trial, fits, counts
    inliers (residual < thresh px). Returns (A, t, inlier_mask, resid)."""
    n = pa.shape[0]
    if n < 3:
        if n == 0:
            return np.eye(2), np.zeros(2), np.zeros(0, bool), 0.0
        # under-determined — fall back to LSQ
        A, t, r = fit_affine_lsq(pa, pb)
        return A, t, np.ones(n, bool), r
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.choice(n, size=3, replace=False)
        try:
            A, t, _ = fit_affine_lsq(pa[idx], pb[idx])
        except np.linalg.LinAlgError:
            continue
        pred = pa @ A.T + t
        d = np.linalg.norm(pb - pred, axis=1)
        inliers = d < inlier_thresh
        c = int(inliers.sum())
        if c > best_count:
            best_count = c
            best_inliers = inliers
    if best_count < 3:
        # Couldn't get 3 inliers anywhere; just LSQ on all.
        A, t, r = fit_affine_lsq(pa, pb)
        return A, t, np.ones(n, bool), r
    # Refit on all inliers.
    A, t, resid = fit_affine_lsq(pa[best_inliers], pb[best_inliers])
    return A, t, best_inliers, resid


def affine_to_strain(A: np.ndarray) -> dict:
    """Decompose 2x2 affine A into rotation + symmetric strain.

    Polar decomposition: A = R · U, where R is rotation and U is the
    symmetric stretch tensor. ε = U − I is the strain.
    """
    U_, S_, Vt = np.linalg.svd(A, full_matrices=False)
    R = U_ @ Vt
    if np.linalg.det(R) < 0:
        # Reflection — swap one axis.
        Vt = Vt.copy()
        Vt[-1] *= -1
        S_ = S_.copy()
        S_[-1] *= -1
        R = U_ @ Vt
    S_diag = np.diag(S_)
    U = Vt.T @ S_diag @ Vt           # symmetric stretch
    eps = U - np.eye(2)
    omega = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return dict(
        eps_xx=float(eps[0, 0]),
        eps_yy=float(eps[1, 1]),
        eps_xy=float(0.5 * (eps[0, 1] + eps[1, 0])),
        rotation_deg=omega,
        det_A=float(np.linalg.det(A)),
        principal_strains=tuple(map(float, S_ - 1.0)),
        eps=eps, R=R, U=U,
    )


# ---------------------------------------------------------------------------
class StrainPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.outdir = None
        self.sample = None
        self._scan_shape = None
        self._class_avgs = None
        self._K = 0
        self._last = None       # dict with results of last fit
        self._build()

    # ----- run linkage --------------------------------------------------
    def link_run(self, outdir, sample):
        self.outdir = outdir
        self.sample = sample
        self._info_lbl.configure(
            text=f"linked: {os.path.basename(outdir)}  (sample={sample})")
        self._class_avgs = None
        self._last = None
        # Inherit K and the scan grid from the post-hoc panel if it's
        # already done inference, so the A/B dropdowns are usable
        # without first running compute.
        self._try_inherit_from_posthoc()
        self._refresh_class_menus()
        self._update_per_pattern_warning()
        self._render_idle()

    def _try_inherit_from_posthoc(self):
        """Pull K + assigns + scan_shape from the Post-hoc panel if it
        already has inference cached.  No-op if posthoc isn't loaded
        yet — a 'Reload classes' button covers that case."""
        try:
            ph = getattr(self.app, "posthoc", None) if self.app else None
            if ph is None or getattr(ph, "_inf", None) is None:
                return
            soft = ph._inf.get("soft_probs")
            if soft is None:
                return
            self._K = int(soft.shape[1])
            self._scan_shape = ph._scan_shape
            self._assigns = ph._inf["assigns"]
        except Exception:
            pass

    def on_runtime_sample_added(self, key):
        pass

    # ----- UI -----------------------------------------------------------
    def _build(self):
        self._vars = {
            "class_a": ctk.StringVar(value="0"),
            "class_b": ctk.StringVar(value="1"),
            "max_d":   ctk.DoubleVar(value=12.0),
            "ransac_thresh": ctk.DoubleVar(value=1.5),
            "ransac_iters":  ctk.IntVar(value=200),
            "use_ransac":    ctk.BooleanVar(value=True),
        }

        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)
        self._info_lbl = ctk.CTkLabel(top, text="(no run linked)",
                                        font=("Consolas", 10))
        self._info_lbl.pack(side="left", padx=8)
        self._compute_btn = ctk.CTkButton(top, text="Compute strain",
                                            width=160,
                                            fg_color=("#2D7A2D", "#1F7A1F"),
                                            command=self._compute_strain)
        self._compute_btn.pack(side="right", padx=4)

        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        sidebar = ctk.CTkScrollableFrame(body, width=300)
        sidebar.pack(side="left", fill="y")

        _section(sidebar, "Class pair  (A → B)")
        ab = ctk.CTkFrame(sidebar, fg_color="transparent")
        ab.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(ab, text="A (ref):", width=70).pack(side="left")
        self._a_menu = ctk.CTkOptionMenu(ab,
            variable=self._vars["class_a"], values=["0"], width=80)
        self._a_menu.pack(side="left", padx=4)
        ctk.CTkLabel(ab, text="B:", width=30).pack(side="left",
                                                       padx=(8, 0))
        self._b_menu = ctk.CTkOptionMenu(ab,
            variable=self._vars["class_b"], values=["0"], width=80)
        self._b_menu.pack(side="left", padx=4)
        ctk.CTkButton(sidebar, text="Reload classes  (refresh K)",
                       width=240,
                       command=self._reload_classes
                       ).pack(anchor="w", padx=8, pady=2)

        _section(sidebar, "Match + RANSAC")
        for k, lbl in [("max_d", "max match dist (px)"),
                          ("ransac_thresh", "RANSAC inlier (px)"),
                          ("ransac_iters", "RANSAC iters")]:
            row = ctk.CTkFrame(sidebar, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=1)
            ctk.CTkLabel(row, text=lbl, width=160, anchor="w"
                           ).pack(side="left")
            ctk.CTkEntry(row, textvariable=self._vars[k], width=80
                           ).pack(side="left")
        ctk.CTkCheckBox(sidebar, text="use RANSAC",
                          variable=self._vars["use_ransac"]
                          ).pack(anchor="w", padx=8, pady=2)

        _section(sidebar, "Strain values")
        self._values_lbl = ctk.CTkLabel(sidebar,
            text="(run compute first)", font=("Consolas", 10),
            justify="left", anchor="w", wraplength=280)
        self._values_lbl.pack(anchor="w", padx=8, pady=2)

        ctk.CTkButton(sidebar, text="Save snapshot",
                       command=self._save_snapshot
                       ).pack(fill="x", padx=8, pady=(8, 4))

        self._status_lbl = ctk.CTkLabel(sidebar,
            text="", font=("Consolas", 9), justify="left",
            text_color=("#444", "#aaa"), wraplength=280)
        self._status_lbl.pack(anchor="w", padx=8, pady=(8, 4))
        self._warn_lbl = ctk.CTkLabel(sidebar,
            text="", font=("Consolas", 9, "bold"), justify="left",
            text_color=("#aa6e2a", "#e2a05f"), wraplength=280)
        self._warn_lbl.pack(anchor="w", padx=8, pady=(2, 4))

        # canvas
        canv = ctk.CTkFrame(body)
        canv.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._fig = Figure(figsize=(11, 5))
        self._canvas = FigureCanvasTkAgg(self._fig, master=canv)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, canv)
        self._render_idle()

    def _reload_classes(self):
        """Force-load class assignments via the Post-hoc panel (runs
        inference if it hasn't already), then repopulate A/B dropdowns."""
        if not self.outdir:
            messagebox.showinfo("Strain", "Load a run dir first.")
            return
        self._status_lbl.configure(text="loading class assigns …")
        self.update_idletasks()
        ok = self._ensure_class_avgs()
        if ok:
            self._refresh_class_menus()
            self._status_lbl.configure(
                text=f"K = {self._K} classes available.")
        else:
            self._status_lbl.configure(
                text="failed — see Post-hoc tab")

    def _refresh_class_menus(self):
        K = max(1, self._K)
        vals = [str(c) for c in range(K)]
        self._a_menu.configure(values=vals)
        self._b_menu.configure(values=vals)
        if self._vars["class_a"].get() not in vals:
            self._vars["class_a"].set("0")
        if self._vars["class_b"].get() not in vals:
            self._vars["class_b"].set(vals[1] if K > 1 else "0")

    # ----- per-pattern blob warning ------------------------------------
    def _update_per_pattern_warning(self):
        """Check whether 'Apply to whole class' has been run for all
        classes (i.e., per-pattern blob_coords.npy exist). If not, warn."""
        if not self.outdir:
            self._warn_lbl.configure(text=""); return
        method_dir = self._blob_method_dir()
        if not method_dir or not os.path.isdir(method_dir):
            self._warn_lbl.configure(text=(
                "[warning] No saved blob output found.  Run blob "
                "detection in the Detect sub-tab first."))
            return
        # Count classes with per-pattern coords.
        present = []
        missing = []
        for entry in sorted(os.listdir(method_dir)):
            if not entry.startswith("p"):
                continue
            p_path = os.path.join(method_dir, entry, "blob_coords.npy")
            if os.path.exists(p_path):
                present.append(entry)
            else:
                missing.append(entry)
        if not present:
            self._warn_lbl.configure(text=(
                "[warning] Blob detection has not been saved per-"
                "pattern.  Strain on class avgs will still work, but "
                "ACOM / per-pixel strain require running 'Apply to "
                "ALL classes' in the Detect tab."))
        elif missing:
            self._warn_lbl.configure(text=(
                f"[note] per-pattern blobs found for {len(present)} "
                f"classes, missing for {len(missing)} "
                f"({', '.join(missing[:5])}…).  ACOM needs all classes."))
        else:
            self._warn_lbl.configure(text=(
                f"[ok] per-pattern blob outputs cover {len(present)} "
                f"classes."))

    def _blob_method_dir(self) -> str | None:
        if not self.outdir:
            return None
        # Use whichever method the Detect tab last wrote.
        cfg_path = os.path.join(self.outdir, "blob", "blob_config.json")
        if not os.path.exists(cfg_path):
            return None
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            method = cfg.get("method", "DoG")
        except Exception:
            method = "DoG"
        return os.path.join(self.outdir, "blob", method)

    # ----- compute ------------------------------------------------------
    def _ensure_class_avgs(self):
        if self._class_avgs is not None:
            return True
        try:
            ph = getattr(self.app, "posthoc", None) if self.app else None
            if ph is None:
                return False
            if getattr(ph, "outdir", None) != self.outdir:
                ph.link_run(self.outdir, self.sample)
            if not ph._ensure_inference():
                return False
            self._class_avgs = np.asarray(ph._compute_class_averages())
            self._K = int(ph._inf["soft_probs"].shape[1])
            self._scan_shape = ph._scan_shape
            self._refresh_class_menus()
            return True
        except Exception as e:
            self._status_lbl.configure(text=f"class-avg compute failed: {e!r}")
            return False

    def _detect_on(self, img: np.ndarray) -> np.ndarray:
        """Apply current Detect-tab tuning on a 2D image, return blob
        coords array shape (N, 3) [y, x, sigma]. Reads tuning from the
        Detect panel directly (no disk I/O)."""
        try:
            from gui_app.blob_panel import _preprocess, _detect
        except ImportError:
            return np.zeros((0, 3), np.float32)
        bp = getattr(self.app, "blob", None)
        if bp is None:
            return np.zeros((0, 3), np.float32)
        kw = bp._gather_current_knobs()
        prep = _preprocess(img, kw["blur_sigma"], kw["rescale_lo"],
                            kw["rescale_hi"], kw["log_stretch"])
        return _detect(prep, kw["method"], kw)

    def _compute_strain(self):
        if not self._ensure_class_avgs():
            self._status_lbl.configure(text="need class averages first")
            return
        try:
            a = int(self._vars["class_a"].get())
            b = int(self._vars["class_b"].get())
        except Exception:
            self._status_lbl.configure(text="bad class ids"); return
        if a == b:
            self._status_lbl.configure(
                text="A and B are the same class — pick different ones.")
            return
        if not (0 <= a < self._K and 0 <= b < self._K):
            self._status_lbl.configure(text="class id out of range"); return
        self._status_lbl.configure(text="detecting blobs on A and B …")
        self.update_idletasks()
        avg_a = self._class_avgs[a]
        avg_b = self._class_avgs[b]
        coords_a = self._detect_on(avg_a)     # (Na, 3) y, x, sigma
        coords_b = self._detect_on(avg_b)
        if coords_a.shape[0] < 3 or coords_b.shape[0] < 3:
            self._status_lbl.configure(text=(
                f"too few blobs detected (A={coords_a.shape[0]}, "
                f"B={coords_b.shape[0]}); tune the Detect knobs."))
            return

        pa = coords_a[:, :2].astype(np.float32)   # y, x in image coords
        pb = coords_b[:, :2].astype(np.float32)
        pairs = match_nearest(pa, pb,
                                max_d=float(self._vars["max_d"].get()))
        if pairs.shape[0] < 3:
            self._status_lbl.configure(text=(
                f"not enough matched pairs ({pairs.shape[0]}). "
                f"Increase max-d, or pick more similar classes."))
            self._render_match(avg_a, avg_b, pa, pb, pairs, None, None)
            return
        pa_m = pa[pairs[:, 0]]
        pb_m = pb[pairs[:, 1]]

        if self._vars["use_ransac"].get():
            A, t, inliers, resid = ransac_affine(
                pa_m, pb_m,
                n_iter=int(self._vars["ransac_iters"].get()),
                inlier_thresh=float(self._vars["ransac_thresh"].get()))
        else:
            A, t, resid = fit_affine_lsq(pa_m, pb_m)
            inliers = np.ones(pairs.shape[0], dtype=bool)

        strain = affine_to_strain(A)
        self._last = dict(
            a=a, b=b, avg_a=avg_a, avg_b=avg_b,
            coords_a=pa, coords_b=pb, pairs=pairs,
            inliers=inliers, A=A, t=t, residual=resid,
            strain=strain,
        )
        # Display values.
        e1, e2 = strain["principal_strains"]
        text = (
            f"class A=p{a}  B=p{b}\n"
            f"matched pairs: {pairs.shape[0]}  "
            f"inliers: {int(inliers.sum())}\n"
            f"residual:      {resid:.3f} px\n"
            f"\n"
            f"ε_xx     = {strain['eps_xx']:+.4f}\n"
            f"ε_yy     = {strain['eps_yy']:+.4f}\n"
            f"ε_xy     = {strain['eps_xy']:+.4f}\n"
            f"rotation = {strain['rotation_deg']:+.3f}°\n"
            f"det(A)   = {strain['det_A']:+.4f}\n"
            f"principal ε: ({e1:+.4f}, {e2:+.4f})\n"
        )
        self._values_lbl.configure(text=text)
        self._status_lbl.configure(text="strain computed.")
        self._render_match(avg_a, avg_b, pa, pb, pairs, inliers, A)
        self._update_per_pattern_warning()

    # ----- rendering ----------------------------------------------------
    def _render_idle(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5,
                "1. Make sure the Detect tab has tuning set.\n"
                "2. Pick class A (reference) and class B.\n"
                "3. Click 'Compute strain'.",
                ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        self._canvas.draw_idle()

    def _disp(self, a):
        ref = a.flatten()
        if not ref.size or ref.max() <= 0:
            return a
        lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
        return np.log1p(np.clip(a, lo, hi) - lo)

    def _render_match(self, avg_a, avg_b, pa, pb, pairs, inliers, A):
        self._fig.clear()
        ax_a = self._fig.add_subplot(1, 2, 1)
        ax_b = self._fig.add_subplot(1, 2, 2)
        ax_a.imshow(self._disp(avg_a), cmap="inferno",
                      aspect="equal", interpolation="nearest")
        ax_b.imshow(self._disp(avg_b), cmap="inferno",
                      aspect="equal", interpolation="nearest")
        # All blobs in cyan
        for (y, x, s) in zip(pa[:, 0], pa[:, 1], np.full(pa.shape[0], 4)):
            ax_a.add_patch(Circle((x, y), 4, edgecolor="cyan",
                                      facecolor="none", lw=0.8))
        for (y, x, s) in zip(pb[:, 0], pb[:, 1], np.full(pb.shape[0], 4)):
            ax_b.add_patch(Circle((x, y), 4, edgecolor="cyan",
                                      facecolor="none", lw=0.8))
        # Matched pairs: green for inlier, red for outlier; same-coloured
        # circle on each side (linking is implicit by index).
        if pairs.size:
            for k, (i, j) in enumerate(pairs):
                inlier = (inliers is not None
                          and bool(inliers[k]))
                col = "lime" if inlier else "red"
                ya, xa = pa[i]
                yb, xb = pb[j]
                ax_a.add_patch(Circle((xa, ya), 5, edgecolor=col,
                                          facecolor="none", lw=1.4))
                ax_b.add_patch(Circle((xb, yb), 5, edgecolor=col,
                                          facecolor="none", lw=1.4))
                ax_a.text(xa + 5, ya - 5, str(k), color=col, fontsize=7)
                ax_b.text(xb + 5, yb - 5, str(k), color=col, fontsize=7)
        a_idx, b_idx = self._last["a"], self._last["b"]
        n_in = int(inliers.sum()) if inliers is not None else 0
        ax_a.set_title(f"A = p{a_idx}   {pa.shape[0]} blobs", fontsize=10)
        ax_b.set_title(f"B = p{b_idx}   {pb.shape[0]} blobs   "
                        f"({n_in}/{pairs.shape[0]} inliers)",
                        fontsize=10)
        for a in (ax_a, ax_b):
            a.set_xticks([]); a.set_yticks([])
        # Reciprocal scale bars if calibrated.
        try:
            rp = float(self.app.recip_res.get()) if self.app else 0.0
        except Exception:
            rp = 0.0
        if rp > 0:
            from gui_app._calib_utils import (q_per_polar_bin,
                                                 get_raw_detector_size,
                                                 add_recip_scalebar)
            qpx = q_per_polar_bin(rp, get_raw_detector_size(self.sample))
            for a in (ax_a, ax_b):
                add_recip_scalebar(a, q_per_disp_px=qpx, length_q=0.2)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ----- save ---------------------------------------------------------
    def _save_snapshot(self):
        if self._last is None or self.outdir is None:
            messagebox.showinfo("Strain", "Run compute first."); return
        out_dir = os.path.join(self.outdir, "strain")
        os.makedirs(out_dir, exist_ok=True)
        a, b = self._last["a"], self._last["b"]
        tag = f"p{a}__p{b}"
        try:
            self._fig.savefig(
                os.path.join(out_dir, f"strain_{tag}.png"), dpi=140)
        except Exception as e:
            messagebox.showerror("Strain", f"PNG save failed:\n{e!r}")
            return
        try:
            blob = {
                "class_a": a, "class_b": b,
                "matched_pairs": int(self._last["pairs"].shape[0]),
                "inliers": int(self._last["inliers"].sum()),
                "residual_px": float(self._last["residual"]),
                "A": self._last["A"].tolist(),
                "t": self._last["t"].tolist(),
                "strain": {
                    k: v for k, v in self._last["strain"].items()
                    if isinstance(v, (int, float))
                },
                "principal_strains": list(map(
                    float, self._last["strain"]["principal_strains"])),
                "knobs": {
                    "max_d":          float(self._vars["max_d"].get()),
                    "ransac_thresh":  float(self._vars["ransac_thresh"].get()),
                    "ransac_iters":   int(self._vars["ransac_iters"].get()),
                    "use_ransac":     bool(self._vars["use_ransac"].get()),
                },
            }
            with open(os.path.join(out_dir,
                                       f"strain_{tag}.json"), "w") as fh:
                json.dump(blob, fh, indent=2)
            # Save the raw blob coords + matched-pair indices so the
            # caller can re-derive the strain externally / merge across
            # runs without re-running detection.
            np.savez(
                os.path.join(out_dir, f"strain_{tag}.npz"),
                coords_a=self._last["coords_a"],
                coords_b=self._last["coords_b"],
                pairs=self._last["pairs"],
                inliers=self._last["inliers"],
                A=self._last["A"],
                t=self._last["t"],
            )
        except Exception as e:
            messagebox.showerror("Strain", f"save failed:\n{e!r}")
            return
        self._status_lbl.configure(text=f"saved → {out_dir} (tag={tag})")
