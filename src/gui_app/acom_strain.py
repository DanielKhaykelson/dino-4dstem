"""acom_strain.py -- the ACOM window's "Strain" mode.

Per-pattern lattice strain against any reference in the Data-reference
list: a measured one (ROI / grain / class average / scan position ->
relative strain, that region = 0) or a CIF simulated at a zone axis
("+ CIF @ zone axis…" -> absolute strain, only as good as the 1/Å-per-px
calibration).  Maths in ``strain_core``.

Mixed into ACOMTabPanel next to DataRefMixin (shares its reference list,
tolerance, peak + beam-centre cache and in-plane rotation matching).
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

from data import SAMPLES
from gui_app.acom_dataref import _hint, _HINT_CLR


class StrainMixin:

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _st_build(self, parent):
        _hint(parent,
            "Lattice strain of every pattern relative to a reference from "
            "the 'Data reference' list: a measured region (relative "
            "strain: that region = 0) or a CIF at a zone axis (absolute "
            "strain: depends on the 1/Å-per-px calibration).")
        r = ctk.CTkFrame(parent, fg_color="transparent")
        r.pack(fill="x", padx=10, pady=(4, 2))
        ctk.CTkLabel(r, text="reference").pack(side="left")
        self._st_ref = ctk.StringVar(value="")
        self._st_ref_menu = ctk.CTkOptionMenu(
            r, variable=self._st_ref, values=["(none)"], width=170,
            command=lambda _v: self._st_show_lattice())
        self._st_ref_menu.pack(side="left", padx=6)
        ctk.CTkButton(r, text="↻", width=28,
                      command=self._st_refresh_refs).pack(side="left")
        ctk.CTkButton(parent, text="Show reference lattice", width=240,
                      command=self._st_show_lattice).pack(anchor="w",
                                                          padx=10, pady=2)
        g = ctk.CTkFrame(parent, fg_color="transparent")
        g.pack(fill="x", padx=10, pady=(4, 2))
        ctk.CTkLabel(g, text="min indexed spots").pack(side="left")
        self._st_min = ctk.StringVar(value="5")
        ctk.CTkEntry(g, textvariable=self._st_min, width=44).pack(
            side="left", padx=6)
        self._st_only_best = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(parent,
            text="only where this reference fits best",
            variable=self._st_only_best).pack(anchor="w", padx=10, pady=2)
        _hint(parent,
            "Spot tolerance, ring and score threshold are the "
            "Data-reference settings.  Spots are refined to sub-pixel "
            "positions and the fit includes an offset, so a slightly wrong "
            "beam centre does not show up as strain.")
        ctk.CTkButton(parent, text="Strain of current source ▶", width=240,
                      command=self._st_single).pack(anchor="w", padx=10,
                                                    pady=2)
        sr = ctk.CTkFrame(parent, fg_color="transparent")
        sr.pack(fill="x", padx=10, pady=(6, 0))
        ctk.CTkLabel(sr, text="stride:").pack(side="left")
        self._st_stride = ctk.IntVar(value=2)
        ctk.CTkEntry(sr, textvariable=self._st_stride, width=40).pack(
            side="left", padx=4)
        ctk.CTkButton(sr, text="Map strain ▶", width=150,
                      fg_color=("#2D7A2D", "#1F6A1F"),
                      command=self._st_run_full).pack(side="left",
                                                      padx=(8, 0))
        ctk.CTkButton(sr, text="■", width=30, fg_color=("#a33", "#722"),
                      hover_color=("#c33", "#822"),
                      command=self._stop_batch_run).pack(side="left",
                                                         padx=(4, 0))
        _hint(parent,
            "x = detector right, y = detector up; rotation ω > 0 = "
            "counter-clockwise.  These are real-space directions only if "
            "the scan is not rotated relative to the diffraction pattern.")
        self._st_status = ctk.CTkLabel(parent, text="",
            font=("Consolas", 9), text_color=_HINT_CLR, wraplength=290,
            justify="left", anchor="w")
        self._st_status.pack(anchor="w", padx=10, pady=(2, 6))

    def _st_set_status(self, msg):
        try:
            self._st_status.configure(text=msg)
        except Exception:
            pass

    def _st_refresh_refs(self):
        names = [r.name for r in getattr(self, "_dr_refs", [])]
        self._st_ref_menu.configure(values=names or ["(none)"])
        if self._st_ref.get() not in names:
            self._st_ref.set(names[0] if names else "(none)")

    def _st_target(self):
        self._st_refresh_refs()
        for k, r in enumerate(getattr(self, "_dr_refs", [])):
            if r.name == self._st_ref.get():
                return k, r
        messagebox.showinfo("Strain",
            "Add a reference on the 'Data reference' page first (a "
            "measured region, or '+ CIF @ zone axis…').")
        return None, None

    def _st_min_val(self):
        try:
            return max(int(self._st_min.get()), 3)
        except Exception:
            return 5

    # ------------------------------------------------------------------
    # reference lattice
    # ------------------------------------------------------------------
    def _st_ref_lattice(self, ref):
        """(B_ref (x, y-up columns), n_indexed) of a reference, from
        sub-pixel spot positions when its pattern is available."""
        from gui_app.strain_core import lattice_basis, refine_subpixel, \
            to_xy
        cache = ref.extra.get("_lattice")
        if cache is not None:
            return cache
        vecs = ref.vecs
        if ref.pattern is not None and not ref.extra.get("simulated"):
            cy, cx = ref.centre
            pk = np.column_stack([vecs[:, 0] + cy, vecs[:, 1] + cx,
                                  np.ones(len(vecs))])
            pk = refine_subpixel(ref.pattern, pk)
            vecs = pk[:, :2] - np.array([cy, cx])
        B, n = lattice_basis(to_xy(vecs), self._dr_tol_val())
        ref.extra["_lattice"] = (B, n)
        return B, n

    def _st_show_lattice(self):
        k, ref = self._st_target()
        if ref is None:
            return
        try:
            B, n = self._st_ref_lattice(ref)
        except Exception as e:
            messagebox.showerror("Strain", f"reference lattice: {e}")
            return
        self._ensure_quad_layout()
        ax = self._ax_pat
        ax.clear()
        if ref.pattern is not None:
            self._dr_imshow(ax, ref.pattern)
        cy, cx = ref.centre
        hk = np.array([(h, kk) for h in range(-8, 9) for kk in range(-8, 9)])
        pts = hk @ B.T
        lim = 1.05 * np.hypot(ref.vecs[:, 0], ref.vecs[:, 1]).max()
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= lim]
        ax.scatter(cx + pts[:, 0], cy - pts[:, 1], s=14, marker="+",
                   color="#ff5050", linewidths=1, label="lattice")
        ax.scatter(cx + ref.vecs[:, 1], cy + ref.vecs[:, 0], s=60,
                   facecolors="none", edgecolors="c", label="spots")
        for g, nm, c in ((B[:, 0], "g1", "#ffd000"),
                         (B[:, 1], "g2", "#40ff40")):
            ax.annotate("", xy=(cx + g[0], cy - g[1]), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color=c, lw=2))
            ax.text(cx + 1.15 * g[0], cy - 1.15 * g[1], nm, color=c,
                    fontsize=10, ha="center", va="center")
        ang = np.degrees(np.arccos(np.clip(
            B[:, 0] @ B[:, 1] / np.prod(np.linalg.norm(B, axis=0)), -1, 1)))
        ax.set_title(f"strain reference '{ref.name}': |g1| = "
                     f"{np.linalg.norm(B[:, 0]):.2f}, |g2| = "
                     f"{np.linalg.norm(B[:, 1]):.2f} px, {ang:.1f}°  "
                     f"({n} spots indexed)", fontsize=9)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        self._redraw_all()

    # ------------------------------------------------------------------
    # one pattern
    # ------------------------------------------------------------------
    def _st_fit_pattern(self, pat, peaks, k, ref, B_ref, centre=None):
        """Strain of one pattern vs reference k.  Returns (result dict
        or None, centre, refined (dy, dx) vectors, scores list)."""
        from gui_app.dataref_core import beam_centre, match_vectors, \
            spot_vectors
        from gui_app.strain_core import refine_subpixel, strain_one, to_xy
        tol = self._dr_tol_val()
        c = centre if centre is not None else beam_centre(
            pat, self._dr_guess(pat))
        lo = ref.extra.get("r_min", 0.0); hi = ref.extra.get("r_max", np.inf)
        v = spot_vectors(peaks, c, lo, hi)
        scores = []
        for rr in self._dr_refs:
            vv = spot_vectors(peaks, c, rr.extra.get("r_min", 0.0),
                              rr.extra.get("r_max", np.inf))
            scores.append(match_vectors(rr.vecs, vv, tol, rr.period_deg))
        m = scores[k]
        if len(v) < self._st_min_val() or not np.isfinite(m["theta_deg"]):
            return None, c, v, scores
        pk = np.column_stack([v[:, 0] + c[0], v[:, 1] + c[1],
                              np.ones(len(v))])
        vr = refine_subpixel(pat, pk)[:, :2] - np.asarray(c)
        res = strain_one(B_ref, to_xy(vr), m["theta_deg"], tol,
                         self._st_min_val())
        return res, c, vr, scores

    def _st_single(self):
        k, ref = self._st_target()
        if ref is None:
            return
        pat = getattr(self, "_test_pattern", None)
        pk = getattr(self, "_test_peaks", None)
        if pat is None or pk is None:
            messagebox.showinfo("Strain",
                "Load a source (step 1) and detect its peaks (step 2).")
            return
        try:
            B_ref, _ = self._st_ref_lattice(ref)
        except Exception as e:
            messagebox.showerror("Strain", f"reference lattice: {e}")
            return
        res, c, vr, _ = self._st_fit_pattern(pat, pk, k, ref, B_ref)
        self._ensure_quad_layout()
        self._st_draw_fit(self._ax_fit, pat, c, vr, res, ref,
                          getattr(self, "_test_origin", ""))
        self._redraw_all()
        if res is None or not np.isfinite(res["exx"]):
            self._st_set_status("no fit: too few spots indexed by the "
                                "reference lattice")
        else:
            self._st_set_status(
                f"εxx {100 * res['exx']:+.2f}%  εyy {100 * res['eyy']:+.2f}%"
                f"  εxy {100 * res['exy']:+.2f}%  ω {res['rot']:+.2f}°  "
                f"({res['n']} spots, rms {res['rms']:.2f} px)")

    def _st_draw_fit(self, ax, pat, c, vr, res, ref, title=""):
        ax.clear()
        self._dr_imshow(ax, pat)
        cy, cx = c
        ax.scatter(cx + vr[:, 1], cy + vr[:, 0], s=60, facecolors="none",
                   edgecolors="c", linewidths=1.0, label="spots (sub-px)")
        if res is not None and np.all(np.isfinite(res["B"])):
            B, t = res["B"], res["t"]
            hk = np.array([(h, k) for h in range(-8, 9)
                           for k in range(-8, 9)])
            pts = hk @ B.T + t
            lim = 1.05 * np.hypot(vr[:, 0], vr[:, 1]).max()
            pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= lim]
            ax.scatter(cx + pts[:, 0], cy - pts[:, 1], s=16, marker="+",
                       color="#ff5050", linewidths=1, label="fitted lattice")
            txt = (f"εxx {100 * res['exx']:+.2f}%  εyy "
                   f"{100 * res['eyy']:+.2f}%  εxy {100 * res['exy']:+.2f}%"
                   f"  ω {res['rot']:+.2f}°  ({res['n']} spots, rms "
                   f"{res['rms']:.2f} px)")
        else:
            txt = "no fit (too few spots indexed)"
        ax.plot(cx, cy, "+", color="w", ms=10)
        ax.set_title((title + "\n" if title else "")
                     + f"vs '{ref.name}': " + txt, fontsize=9)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])

    # ------------------------------------------------------------------
    # whole scan
    # ------------------------------------------------------------------
    def _st_run_full(self):
        k, ref = self._st_target()
        if ref is None:
            return
        sample = self._need_dataset()
        if sample is None:
            return
        if getattr(self, "_dr_busy", False):
            return
        try:
            B_ref, _ = self._st_ref_lattice(ref)
        except Exception as e:
            messagebox.showerror("Strain", f"reference lattice: {e}")
            return
        self._dr_busy = True
        self._stop_event.clear()
        threading.Thread(target=self._st_worker, daemon=True, kwargs=dict(
            k=k, ref=ref, B_ref=B_ref, sample=sample,
            stride=max(int(self._st_stride.get() or 1), 1),
            detect_kw=self._detect_kw_now(),
            only_best=bool(self._st_only_best.get()),
            thr=self._dr_thr_val())).start()

    def _st_worker(self, k, ref, B_ref, sample, stride, detect_kw,
                   only_best, thr):
        from gui_app.posthoc_panel import _open_lazy
        try:
            cfg = SAMPLES[sample]
            Ny, Nx = cfg["scan_shape"]
            cube = _open_lazy(cfg["path"], scan_shape=(Ny, Nx))
            t0 = time.time()
            peaks, cents = self._dr_detect_all(cube, stride, detect_kw)
            names = ("exx", "eyy", "exy", "rot", "rms")
            out = {n: np.full((Ny, Nx), np.nan, np.float32) for n in names}
            out["n"] = np.zeros((Ny, Nx), np.int16)
            out["state"] = np.zeros((Ny, Nx), np.int8)   # see legend
            pos = [(y, x) for y in range(0, Ny, stride)
                   for x in range(0, Nx, stride)]
            for i, (y, x) in enumerate(pos):
                if self._stop_event.is_set():
                    raise RuntimeError("stopped by user")
                n = y * Nx + x
                pk = peaks[n]
                if len(pk) < self._st_min_val():
                    out["state"][y, x] = 1                 # too few spots
                    continue
                pat = np.asarray(cube[y, x], np.float32)
                res, _c, _v, sc = self._st_fit_pattern(
                    pat, pk, k, ref, B_ref, centre=cents[n])
                s_all = [s["score"] for s in sc]
                if s_all[k] < thr or (only_best and
                                      int(np.argmax(s_all)) != k):
                    out["state"][y, x] = 2                 # other / none
                    continue
                if res is None or not np.isfinite(res["exx"]):
                    out["state"][y, x] = 3                 # no lattice fit
                    continue
                for nm in names:
                    out[nm][y, x] = res[nm]
                out["n"][y, x] = res["n"]
                out["state"][y, x] = 4                     # fitted
                if i % 100 == 0:
                    el = time.time() - t0
                    self._dr_post(self._st_set_status,
                        f"strain: {i}/{len(pos)}  ({el:.0f} s)")
            res = dict(out, stride=stride, scan_shape=(Ny, Nx), ref=ref,
                       k=k, cube=cube, peaks=peaks, centres=cents,
                       B_ref=B_ref, elapsed=time.time() - t0,
                       only_best=only_best, thr=thr)
            self._dr_post(self._st_show_full, res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = repr(e)
            self._dr_post(self._st_set_status, f"strain map failed: {msg}")
        finally:
            self._dr_busy = False

    def _st_show_full(self, res):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from gui_app.acom_tab import _fill_stride
        self._stfull = res
        Ny, Nx = res["scan_shape"]; st = res["stride"]
        maps = {n: np.array(res[n], dtype=float) for n in
                ("exx", "eyy", "exy", "rot", "rms", "n")}
        _fill_stride(maps, Ny, Nx, st)
        self._st_save_outputs(res)

        win = tk.Toplevel(self)
        win.title(f"ACOM - strain vs '{res['ref'].name}'  "
                  f"({self._active_sample()})")
        win.geometry("1250x880")
        bar = tk.Frame(win); bar.pack(side="top", fill="x")
        fig = Figure(figsize=(12.5, 8.5), dpi=95, facecolor="#f4f4f4",
                     layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        tk.Label(bar, fg="#555", text="left-click: pattern + fitted "
                 "lattice").pack(side="left", padx=10)
        # zoom to the fitted region (the rest is grey anyway) and pick
        # the grid from ITS shape
        fy, fx = np.nonzero(res["state"] == 4)
        if fy.size:
            y0, y1 = max(fy.min() - st, 0), min(fy.max() + st, Ny - 1)
            x0, x1 = max(fx.min() - st, 0), min(fx.max() + st, Nx - 1)
        else:
            y0, y1, x0, x1 = 0, Ny - 1, 0, Nx - 1
        wide = (x1 - x0) > 2.5 * (y1 - y0)
        axs = np.asarray(fig.subplots(6, 1) if wide
                         else fig.subplots(2, 3)).ravel()
        e_all = np.concatenate([maps[n][np.isfinite(maps[n])]
                                for n in ("exx", "eyy", "exy")]) * 100
        lim = float(np.percentile(np.abs(e_all), 98)) if e_all.size else 1
        lim = max(lim, 0.05)
        r_ok = maps["rot"][np.isfinite(maps["rot"])]
        rlim = max(float(np.percentile(np.abs(r_ok), 98)) if r_ok.size
                   else 1.0, 0.05)
        panels = (("exx", "εxx (%)", 100, "RdBu_r", -lim, lim),
                  ("eyy", "εyy (%)", 100, "RdBu_r", -lim, lim),
                  ("exy", "εxy (%)", 100, "RdBu_r", -lim, lim),
                  ("rot", "lattice rotation ω (°, CCW +)", 1, "PuOr_r",
                   -rlim, rlim),
                  ("n", "indexed spots", 1, "magma", None, None),
                  ("rms", "fit rms (px)", 1, "viridis", 0, None))
        for a, (key, lab, mul, cm, lo, hi) in zip(axs, panels):
            d = maps[key] * mul
            if key == "n":
                d = np.where(d > 0, d, np.nan)
            a.set_facecolor((0.82, 0.82, 0.82))
            im = a.imshow(d, cmap=cm, vmin=lo, vmax=hi,
                          interpolation="nearest")
            a.set_title(lab, fontsize=10)
            a.set_xlim(x0 - 0.5, x1 + 0.5); a.set_ylim(y1 + 0.5, y0 - 0.5)
            a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im, ax=a, fraction=0.035)
        n_fit = int((res["state"][::st, ::st] == 4).sum())
        n_all = int(res["state"][::st, ::st].size)
        mode = "absolute (CIF)" if res["ref"].extra.get("simulated") \
            else "relative to the reference region"
        fig.suptitle(f"strain vs '{res['ref'].name}' - {mode}; fitted "
                     f"{n_fit}/{n_all} positions (grey = not fitted), "
                     f"stride {st}, {res['elapsed']:.0f} s", fontsize=11)

        def _click(ev):
            if (ev.button != 1 or ev.inaxes not in axs or ev.xdata is None
                    or getattr(tb, "mode", "")):
                return
            x = int(np.clip(round(ev.xdata), 0, Nx - 1))
            y = int(np.clip(round(ev.ydata), 0, Ny - 1))
            self._st_inspect(y - y % st, x - x % st)
        cv.mpl_connect("button_press_event", _click)
        cv.draw_idle()
        self._st_set_status(
            f"fitted {n_fit}/{n_all} positions in {res['elapsed']:.0f} s "
            f"(colour limits ±{lim:.2f}% / ±{rlim:.2f}°)")

    def _st_save_outputs(self, res):
        try:
            from data import output_dir
            d = output_dir(self._active_sample(), "acom", "strain")
            np.savez_compressed(
                os.path.join(d, f"strain_{res['ref'].name}.npz"
                                .replace(" ", "_")),
                exx=res["exx"], eyy=res["eyy"], exy=res["exy"],
                rotation_deg=res["rot"], n_indexed=res["n"], rms=res["rms"],
                state=res["state"], B_ref=res["B_ref"],
                stride=res["stride"], reference=res["ref"].name,
                absolute=bool(res["ref"].extra.get("simulated")),
                axes="x = detector right, y = detector up; strain = "
                     "real-space (from the reciprocal lattice)")
        except Exception as e:
            print(f"[strain] save: {e!r}", flush=True)

    def _st_inspect(self, y, x):
        res = getattr(self, "_stfull", None)
        if res is None:
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        Ny, Nx = res["scan_shape"]
        n = y * Nx + x
        pat = np.asarray(res["cube"][y, x], np.float32)
        fit, c, vr, _ = self._st_fit_pattern(pat, res["peaks"][n], res["k"],
                                             res["ref"], res["B_ref"],
                                             centre=res["centres"][n])
        win = tk.Toplevel(self)
        win.title(f"strain at ({y}, {x})")
        win.geometry("720x760")
        bar = tk.Frame(win); bar.pack(side="bottom", fill="x")
        fig = Figure(figsize=(7, 7), dpi=95)
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        self._st_draw_fit(fig.add_subplot(111), pat, c, vr, fit,
                          res["ref"], f"scan ({y}, {x})")
        fig.tight_layout(); cv.draw_idle()
