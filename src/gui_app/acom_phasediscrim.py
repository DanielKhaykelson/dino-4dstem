"""acom_phasediscrim.py -- "Phase discrimination" on the ACOM window's
Data-reference page: tell similar phases apart by what DIFFERS between the
references (phase_discrim).  Fiber mode: one known viewing direction.

  Compare references  - how every pair differs: same main lattice ->
                        the cell positions where they differ (folded-cell
                        comparison, e.g. a half-order superstructure at
                        (0, 1/2)); different lattices -> decided by which
                        lattice indexes the spots.
  Map (the page's     - every pattern: best / second-best reference,
   Map button)
                        posterior, confidence (weakest pairwise win, in
                        log-likelihood units), uncertain / cannot-separate /
                        off-axis; click a pixel for the evidence.

Mixed into ACOMTabPanel with DataRefMixin (shares its references, ring,
tolerance, peak + beam-centre cache and STOP button).
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


class PhaseDiscrimMixin:

    def _pd_build(self, parent):
        hdr = ctk.CTkLabel(parent, text="Phase discrimination",
                           font=("Segoe UI", 12, "bold"), anchor="w")
        hdr.pack(anchor="w", padx=10, pady=(10, 0))
        ctk.CTkFrame(parent, height=2, fg_color=("#cccccc", "#444444")
                     ).pack(fill="x", padx=10, pady=(0, 4))
        _hint(parent,
            "For phases that share most reflections (a structure and its "
            "superstructure) or two similar phases.  The references are "
            "compared with each other to find WHERE they differ, and only "
            "that is measured in every pattern (intensity, no peak "
            "detection, positions from each pattern's own fitted lattice "
            "so strain does not leak in).  Fiber mode: all patterns at the "
            "same zone axis.  With data / ROI references you are "
            "responsible for the references being pure.")
        g = ctk.CTkFrame(parent, fg_color="transparent")
        g.pack(fill="x", padx=10, pady=(2, 2))
        ctk.CTkLabel(g, text="site radius px").pack(side="left")
        self._pd_w = ctk.StringVar(value="2")
        ctk.CTkEntry(g, textvariable=self._pd_w, width=40).pack(
            side="left", padx=(4, 10))
        ctk.CTkLabel(g, text="confident ≥").pack(side="left")
        self._pd_gate = ctk.StringVar(value="50")
        self._pd_conf = ctk.StringVar(value="3")
        ctk.CTkEntry(g, textvariable=self._pd_conf, width=40).pack(
            side="left", padx=4)
        g2 = ctk.CTkFrame(parent, fg_color="transparent")
        g2.pack(fill="x", padx=10, pady=(2, 2))
        ctk.CTkLabel(g2, text="lattice must index ≥").pack(side="left")
        ctk.CTkEntry(g2, textvariable=self._pd_gate, width=40).pack(
            side="left", padx=4)
        ctk.CTkLabel(g2, text="% (else off-axis)").pack(side="left")
        _hint(parent,
            "off-axis: no reference lattice indexes at least this share of "
            "a pattern's 20 brightest spots (above chance, relative to its "
            "own reference).  Lower it if real patterns come out off-axis.")
        _hint(parent,
            "confident: the best reference must beat every other by this "
            "log-likelihood ratio (3 ≈ 20:1 odds); below it the pixel is "
            "'uncertain'.")
        ctk.CTkButton(parent, text="Compare references", width=240,
                      command=self._pd_compare).pack(anchor="w", padx=10,
                                                     pady=(2, 2))
        _hint(parent, "'Map ▶' above uses this whenever there are two or "
                      "more references.")
        self._pd_status = ctk.CTkLabel(parent, text="",
            font=("Consolas", 9), text_color=_HINT_CLR, wraplength=290,
            justify="left", anchor="w")
        self._pd_status.pack(anchor="w", padx=10, pady=(2, 6))

    def _pd_set_status(self, msg):
        try:
            self._pd_status.configure(text=msg)
        except Exception:
            pass

    def _pd_conf_val(self):
        try:
            return float(self._pd_conf.get())
        except Exception:
            return 3.0

    # ------------------------------------------------------------------
    def _pd_model(self):
        """Calibrated phase_discrim.Model from the current references."""
        from gui_app import phase_discrim as pd
        from gui_app.acom_core import detect_peaks_2d
        refs = list(getattr(self, "_dr_refs", []))
        if len(refs) < 2:
            messagebox.showinfo("Phase discrimination",
                                "Add at least two references first.")
            return None
        try:
            w = max(float(self._pd_w.get()), 0.5)
        except Exception:
            w = 2.0
        model = pd.Model([pd.make_phase_ref(r, self._dr_tol_val())
                          for r in refs], tol=self._dr_tol_val(),
                         r_min=min(r.extra.get("r_min", 0.0) for r in refs),
                         r_max=max(r.extra.get("r_max", np.inf)
                                   for r in refs), w=w)
        try:
            model.min_index_frac = min(max(float(self._pd_gate.get()) / 100,
                                           0.0), 1.0)
        except Exception:
            pass
        kw = self._detect_kw_now()
        pd.calibrate(model, [(r.pattern, detect_peaks_2d(r.pattern, **kw),
                              r.centre) for r in refs])
        return model

    def _pd_compare(self):
        """Show how every pair of references differs."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from gui_app import phase_discrim as pd
        try:
            model = self._pd_model()
        except Exception as e:
            messagebox.showerror("Phase discrimination", repr(e))
            return
        if model is None:
            return
        pairs = list(model.pairs.items())
        win = tk.Toplevel(self)
        win.title("Phase discrimination - how the references differ")
        win.geometry("1100x%d" % (300 + 300 * len(pairs)))
        bar = tk.Frame(win); bar.pack(side="bottom", fill="x")
        fig = Figure(figsize=(11, 1 + 3 * len(pairs)), dpi=95,
                     layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        axs = np.atleast_2d(fig.subplots(len(pairs), 3))
        for row, ((a, b), ent) in zip(axs, pairs):
            na, nb = model.refs[a].name, model.refs[b].name
            for ax, k in zip(row[:2], (a, b)):
                ref = model.refs[k]
                if ref.pattern is not None:
                    self._dr_imshow(ax, ref.pattern)
                # the lattice found for this reference (g1 yellow, g2 green)
                cy, cx = ref.centre
                for g, col in ((ref.B[:, 0], "#ffd000"),
                               (ref.B[:, 1], "#40ff40")):
                    ax.annotate("", xy=(cx + g[0], cy - g[1]),
                                xytext=(cx, cy), arrowprops=dict(
                                    arrowstyle="->", color=col, lw=1.8))
                ax.set_title(f"{ref.name}  (lattice indexes "
                             f"{100 * ref.q_ref:.0f}% of its spots)",
                             fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            ax = row[2]
            if ent["mode"] == "shared":
                z = ent["z"]
                lab = [f"({f[0]:g},{f[1]:g})" for f in pd.OFFSETS]
                cols = ["#d62728" if d else "#999999" for d in ent["disc"]]
                ax.bar(range(len(z)), z, color=cols)
                ax.axhline(4.0, color="k", ls="--", lw=0.8)
                ax.set_xticks(range(len(z)))
                ax.set_xticklabels(lab, rotation=90, fontsize=7)
                ax.set_ylabel("|difference| (σ)")
                ax.set_title(f"{na} vs {nb}: same main lattice - cell "
                             f"positions that differ (red)", fontsize=9)
                # mark the discriminating sites on both references
                for axr, k in zip(row[:2], (a, b)):
                    ref = model.refs[k]
                    if ref.pattern is None:
                        continue
                    from gui_app.strain_core import lattice_basis  # noqa
                    fr = pd.fit_all(model, ref.pattern,
                                    self._pd_ref_peaks(ref), ref.centre,
                                    gate=False)[0]
                    c = ent["coarse"]
                    if fr[c] is None:
                        continue
                    B, t = fr[c][0], fr[c][1]
                    cy, cx = ref.centre
                    for f in pd.OFFSETS[ent["disc"]]:
                        xy, _ = pd.lattice_sites(B, t + B @ f, model.r_min,
                                                 model.r_max)
                        axr.scatter(cx + xy[:, 0], cy - xy[:, 1], s=40,
                                    facecolors="none", edgecolors="r",
                                    linewidths=0.8)
            else:
                ax.axis("off")
                ax.text(0.02, 0.6, f"{na} vs {nb}:\ndifferent lattices -\n"
                        f"decided by which one indexes\nthe detected spots",
                        fontsize=10, va="center")
        fig.suptitle("  ".join(model.notes), fontsize=9)
        cv.draw_idle()
        self._pd_set_status("\n".join(model.notes))

    def _pd_ref_peaks(self, ref):
        from gui_app.acom_core import detect_peaks_2d
        return detect_peaks_2d(ref.pattern, **self._detect_kw_now())

    # ------------------------------------------------------------------
    def _pd_run_full(self):
        sample = self._need_dataset()
        if sample is None or getattr(self, "_dr_busy", False):
            return
        try:
            model = self._pd_model()
        except Exception as e:
            messagebox.showerror("Phase discrimination", repr(e))
            return
        if model is None:
            return
        self._dr_busy = True
        self._stop_event.clear()
        threading.Thread(target=self._pd_worker, daemon=True, kwargs=dict(
            model=model, sample=sample,
            stride=max(int(self._dr_stride.get() or 1), 1),
            detect_kw=self._detect_kw_now())).start()

    def _pd_worker(self, model, sample, stride, detect_kw):
        from gui_app import phase_discrim as pd
        from gui_app.posthoc_panel import _open_lazy
        try:
            cfg = SAMPLES[sample]
            Ny, Nx = cfg["scan_shape"]
            cube = _open_lazy(cfg["path"], scan_shape=(Ny, Nx))
            t0 = time.time()
            peaks, cents = self._dr_detect_all(cube, stride, detect_kw)
            from gui_app.dataref_core import match_dataset
            samp = np.zeros((Ny, Nx), bool); samp[::stride, ::stride] = True
            drefs = list(self._dr_refs)
            dr_res = match_dataset(
                peaks, cents, drefs, (Ny, Nx), tol=model.tol,
                r_min=model.r_min, r_max=model.r_max, sampled=samp.ravel(),
                stop=self._stop_event.is_set)
            K = len(model.refs)
            best = np.full((Ny, Nx), -9, np.int16)      # -9 = not sampled
            second = np.full((Ny, Nx), -1, np.int16)
            margin = np.full((Ny, Nx), np.nan, np.float32)
            post = np.full((K, Ny, Nx), np.nan, np.float32)
            status = np.full((Ny, Nx), -1, np.int8)
            meter = pd.SiteMeter(model.w)
            pos = [(y, x) for y in range(0, Ny, stride)
                   for x in range(0, Nx, stride)]
            for i, (y, x) in enumerate(pos):
                if self._stop_event.is_set():
                    raise RuntimeError("stopped by user")
                n = y * Nx + x
                pat = np.asarray(cube[y, x], np.float32)
                r = pd.classify(model, pat, peaks[n], cents[n], meter)
                status[y, x] = r["status"]
                best[y, x] = r["best"]; second[y, x] = r["second"]
                margin[y, x] = r["margin"]
                post[:, y, x] = r["post"]
                if i % 100 == 0:
                    el = time.time() - t0
                    self._dr_post(self._pd_set_status,
                        f"phases: {i}/{len(pos)}  ({el:.0f} s)")
            res = dict(best=best, second=second, margin=margin, post=post,
                       dr=dr_res,
                       status=status, stride=stride, scan_shape=(Ny, Nx),
                       model=model, cube=cube, peaks=peaks, centres=cents,
                       elapsed=time.time() - t0)
            self._dr_post(self._pd_show_full, res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = repr(e)
            self._dr_post(self._pd_set_status, f"phase map failed: {msg}")
        finally:
            self._dr_busy = False

    def _pd_show_full(self, res):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        from gui_app.acom_tab import _fill_stride
        from gui_app import phase_discrim as pd
        self._pdfull = res
        model = res["model"]; K = len(model.refs)
        Ny, Nx = res["scan_shape"]; st = res["stride"]
        conf = self._pd_conf_val()
        cols = [r.colour for r in self._dr_refs[:K]]
        # category image: 0..K-1 phases, K uncertain, K+1 cannot separate,
        # K+2 off-axis, K+3 too few spots
        st_ = res["status"].astype(float)
        cat = np.where(st_ == pd.OK,
                       np.where(res["margin"] >= conf, res["best"], K),
                       np.where(st_ == pd.METRIC_ONLY, K + 1,
                                np.where(st_ == pd.OFF_AXIS, K + 2, K + 3))
                       ).astype(float)
        cat[st_ < 0] = np.nan
        sec = np.where(st_ == pd.OK, res["second"], np.nan).astype(float)
        pbest = np.where(st_ == pd.OK, np.nanmax(res["post"], 0), np.nan)
        mg = np.where(st_ == pd.OK, res["margin"], np.nan)
        # in-plane rotation and spot-match score of the assigned reference
        bi = np.clip(res["best"], 0, K - 1).astype(int)
        yy, xx = np.indices(bi.shape)
        th = np.where(st_ == pd.OK, res["dr"]["theta_k"][bi, yy, xx], np.nan)
        sc = np.where(st_ >= 0, res["dr"]["score_k"][bi, yy, xx], np.nan)
        maps = dict(cat=cat, sec=sec, pbest=pbest, mg=mg, th=th, sc=sc)
        _fill_stride(maps, Ny, Nx, st)
        self._pd_save(res, maps)

        win = tk.Toplevel(self)
        win.title(f"ACOM - phase discrimination  ({self._active_sample()})")
        win.geometry("1200x860")
        bar = tk.Frame(win); bar.pack(side="top", fill="x")
        fig = Figure(figsize=(12, 8), dpi=95, facecolor="#f4f4f4",
                     layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        tk.Label(bar, fg="#555", text="left-click: the evidence at that "
                 "pixel").pack(side="left", padx=10)
        wide = Nx > 1.6 * Ny
        axs = np.asarray(fig.subplots(6, 1) if wide
                         else fig.subplots(2, 3)).ravel()
        extra = [(0.75, 0.75, 0.75), (0.45, 0.45, 0.45), (0.0, 0.0, 0.0),
                 (1.0, 1.0, 1.0)]
        cm = ListedColormap(cols + extra)
        axs[0].imshow(maps["cat"], cmap=cm, vmin=-0.5, vmax=K + 3.5,
                      interpolation="nearest")
        axs[0].set_title(f"phase (confident: margin ≥ {conf:g})",
                         fontsize=10)
        names = [r.name for r in model.refs]
        axs[0].legend(handles=[Patch(color=c, label=n)
                               for c, n in zip(cols, names)]
                      + [Patch(color=extra[0], label="uncertain"),
                         Patch(color=extra[1], label="cannot separate"),
                         Patch(color=extra[2], label="off-axis"),
                         Patch(facecolor=extra[3], edgecolor="k",
                               label="too few spots")],
                      fontsize=7, loc="upper right", framealpha=0.8)
        im = axs[1].imshow(np.clip(maps["mg"], 0, 20), cmap="viridis",
                           vmin=0, vmax=20, interpolation="nearest")
        axs[1].set_title("confidence: weakest pairwise win of the best "
                         "(log-likelihood ratio; 3 ≈ 20:1)", fontsize=10)
        fig.colorbar(im, ax=axs[1], fraction=0.035)
        cm2 = ListedColormap(cols)
        axs[2].imshow(maps["sec"], cmap=cm2, vmin=-0.5, vmax=K - 0.5,
                      interpolation="nearest")
        axs[2].set_title("second-best reference", fontsize=10)
        im = axs[3].imshow(maps["pbest"], cmap="magma", vmin=1.0 / K,
                           vmax=1, interpolation="nearest")
        axs[3].set_title("posterior probability of the best", fontsize=10)
        fig.colorbar(im, ax=axs[3], fraction=0.035)
        P = float(max(r.period_deg for r in model.refs))
        im = axs[4].imshow(maps["th"], cmap="twilight", vmin=-P / 2,
                           vmax=P / 2, interpolation="nearest")
        axs[4].set_title("in-plane rotation vs its reference (°)",
                         fontsize=10)
        fig.colorbar(im, ax=axs[4], fraction=0.035)
        im = axs[5].imshow(maps["sc"], cmap="viridis", vmin=0, vmax=1,
                           interpolation="nearest")
        axs[5].set_title("spot-match score (F1)", fontsize=10)
        fig.colorbar(im, ax=axs[5], fraction=0.035)
        for a in axs:
            a.set_facecolor((0.9, 0.9, 0.9))
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{K} references, stride {st}  ({res['elapsed']:.0f} "
                     f"s)   " + "  ".join(model.notes), fontsize=9)

        def _click(ev):
            if (ev.button != 1 or ev.inaxes not in axs or ev.xdata is None
                    or getattr(tb, "mode", "")):
                return
            x = int(np.clip(round(ev.xdata), 0, Nx - 1))
            y = int(np.clip(round(ev.ydata), 0, Ny - 1))
            self._pd_inspect(y - y % st, x - x % st)
        cv.mpl_connect("button_press_event", _click)
        cv.draw_idle()
        s = res["status"][::st, ::st]
        ok = (s == pd.OK) & (res["margin"][::st, ::st] >= conf)
        self._pd_set_status(
            f"{int(ok.sum())}/{s.size} positions assigned confidently; "
            f"{int(((s == pd.OK) & ~ok).sum())} uncertain, "
            f"{int((s == pd.METRIC_ONLY).sum())} cannot separate, "
            f"{int((s == pd.OFF_AXIS).sum())} off-axis")

    def _pd_save(self, res, maps):
        try:
            from data import output_dir
            d = output_dir(self._active_sample(), "acom", "phases")
            np.savez_compressed(
                os.path.join(d, "phase_discrimination.npz"),
                best=res["best"], second=res["second"],
                margin=res["margin"], posterior=res["post"],
                status=res["status"], stride=res["stride"],
                names=np.array([r.name for r in res["model"].refs]),
                notes=np.array(res["model"].notes),
                status_codes="0 ok, 1 off-axis, 2 cannot separate, "
                             "3 too few spots, -1 not sampled")
        except Exception as e:
            print(f"[phases] save: {e!r}", flush=True)

    def _pd_inspect(self, y, x):
        """The pattern with the discriminating sites of every pair, and the
        pairwise log-likelihood ratios."""
        res = getattr(self, "_pdfull", None)
        if res is None:
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from gui_app import phase_discrim as pd
        model = res["model"]; Ny, Nx = res["scan_shape"]
        n = y * Nx + x
        pat = np.asarray(res["cube"][y, x], np.float32)
        c = res["centres"][n]
        r = pd.classify(model, pat, res["peaks"][n], c)
        win = tk.Toplevel(self)
        win.title(f"phase evidence at ({y}, {x})")
        win.geometry("760x820")
        bar = tk.Frame(win); bar.pack(side="bottom", fill="x")
        fig = Figure(figsize=(7.5, 8), dpi=95)
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        ax = fig.add_subplot(111)
        self._dr_imshow(ax, pat)
        cy, cx = c
        handles = []
        for (a, b), ent in model.pairs.items():
            if ent.get("mode") != "shared" or not ent["disc"].any():
                continue
            k = ent["coarse"]
            fk = (r.get("raw_fits") or r["fits"])[k]
            if fk is None:
                continue
            B, t = fk[0], fk[1]
            for f in pd.OFFSETS[ent["disc"]]:
                xy, _ = pd.lattice_sites(B, t + B @ f, model.r_min,
                                         model.r_max)
                h = ax.scatter(cx + xy[:, 0], cy - xy[:, 1], s=60,
                               facecolors="none", edgecolors="r",
                               linewidths=0.9,
                               label=f"sites where {model.refs[a].name} / "
                                     f"{model.refs[b].name} differ")
            handles.append(h)
        ax.plot(cx, cy, "+", color="w", ms=10)
        names = [rf.name for rf in model.refs]
        L = r["llr"]
        lines = [f"{names[i]} vs {names[j]}: {L[i, j]:+.1f}"
                 for i in range(len(names)) for j in range(i + 1, len(names))
                 if np.isfinite(L[i, j])]
        head = (f"best: {names[r['best']]}  (p {np.nanmax(r['post']):.2f}, "
                f"margin {r['margin']:.1f}); second: "
                f"{names[r['second']] if r['second'] >= 0 else '-'}"
                if r["status"] == pd.OK else pd.STATUS[r["status"]])
        if r["status"] == pd.OFF_AXIS and r.get("gate"):
            head += "  - " + "; ".join(
                f"{nm}: indexes {100 * g[0]:.0f}% of the brightest spots "
                f"(needs {100 * g[1]:.0f}%)" if g else f"{nm}: no fit"
                for nm, g in zip(names, r["gate"]))
        ax.set_title(f"scan ({y}, {x})   {head}\nlog-likelihood ratios: "
                     + ";  ".join(lines), fontsize=9)
        if handles:
            ax.legend(handles=handles[:3], loc="lower right", fontsize=7,
                      framealpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); cv.draw_idle()
