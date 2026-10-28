"""ACOM - automatic phase analysis from two initial references.

One button.  The user gives two data references (one with, one without
the superstructure satellites - they need not be pure); the code does the
rest:

  1. phase map with the initial references (phase_discrim);
  2. satellite strength S of every pattern (satellite sites over their own
     local background, relative to the main reflections) and the
     automatic weak / strong split;
  3. satellite-free domain: frames ranked by their neighbours' S, added
     from the weakest upward until the set's mean S is 3 sigma above zero
     (frames with satellites of their own are dropped);
  4. refined references: the satellite-free average replaces the
     satellite-poor reference, the average of the strong-satellite
     population replaces the satellite-rich one (each with its member
     positions, so the levels come from the patterns themselves);
  5. phase map with the refined references, domain averages, main-lattice
     vectors per phase and the profiles along the doubling direction.

One summary window (details on demand through its buttons); everything is
saved under <data>_DINO4DSTEM/acom/phases/ (auto_phases.npz / .png plus
the per-step files).
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import tkinter as tk
from tkinter import messagebox


class AutoPhasesMixin:

    # ------------------------------------------------------------ entry
    def _ap_run(self):
        sample = self._need_dataset()
        if sample is None or getattr(self, "_dr_busy", False):
            return
        refs = list(getattr(self, "_dr_refs", []))
        if len(refs) < 2:
            messagebox.showinfo("Auto phases",
                                "Add two references first: one with and "
                                "one without the extra reflections (they "
                                "need not be pure).")
            return
        try:
            model = self._pd_model()
        except Exception as e:
            messagebox.showerror("Auto phases", repr(e))
            return
        if model is None:
            return
        ctx = dict(sample=sample,
                   stride=max(int(self._dr_stride.get() or 1), 1),
                   detect_kw=self._detect_kw_now(), t0=time.time(),
                   initial=refs)
        self._dr_busy = True
        self._stop_event.clear()
        self._ap_ctx = ctx
        threading.Thread(target=self._ap_stage1, args=(model, ctx),
                         daemon=True).start()

    def _ap_status(self, msg):
        t = time.time() - self._ap_ctx["t0"]
        self._dr_post(self._pd_set_status, f"auto phases - {msg}  ({t:.0f} s)")

    def _ap_fail(self, e):
        import traceback
        traceback.print_exc()
        self._dr_post(self._pd_set_status, f"auto phases failed: {e!r}")
        self._dr_busy = False

    # ---------------------------------------------------- stage 1 (thread)
    def _ap_stage1(self, model, ctx):
        from gui_app import phase_discrim as pd
        try:
            self._ap_status("1/4 phase map with the initial references")
            res1 = self._pd_compute(model, ctx["sample"], ctx["stride"],
                                    ctx["detect_kw"])
            if pd._sat_pair(res1["model"])[0] is None:
                raise RuntimeError(
                    "the two references do not differ significantly at any "
                    "cell position (" + " ".join(res1["model"].notes)
                    + ") - pick references further apart (e.g. the "
                    "brightest and the darkest area of the satellite map)")
            self._ap_status("2/4 satellite strength")
            sm = self._pd_sat_compute(res1)
            self._ap_status("3/4 satellite-free domain")
            sf = self._sf_compute(res1)
            # the strong-satellite population for the refined rich reference
            Ny, Nx = res1["scan_shape"]; st = res1["stride"]
            ys, xs = sf["order"]
            strong = []
            if sm["tp"]["bimodal"]:
                lab = sm["lab"]
                strong = [(y, x) for y in range(0, Ny, st)
                          for x in range(0, Nx, st) if lab[y, x] == 1]
            if len(strong) < 20:
                n = max(20, int(0.2 * len(ys)))
                strong = list(zip(ys[-n:], xs[-n:]))
            H, W = np.asarray(res1["cube"][0, 0]).shape
            centre = getattr(self, "_dr_beam_guess", None)
            centre = centre if centre is not None else (H / 2, W / 2)
            self._ap_status(f"3/4 averaging the {len(strong)} strongest")
            avg_rich = self._dm_average(res1, strong, centre, cap=1500)
            mem_rich = np.array([y * Nx + x for (y, x) in strong])
            keep = sf["keep"]
            mem_poor = ys[keep] * Nx + xs[keep]
            if sf["avg_free"] is None or len(mem_poor) < 5:
                raise RuntimeError("no satellite-free frames found - the "
                                   "whole scan has satellites")
            self._dr_post(self._ap_stage2, res1, sm, sf,
                          sf["avg_free"], mem_poor, avg_rich, mem_rich)
        except Exception as e:
            self._ap_fail(e)

    # ------------------------------------------------ stage 2 (main thread)
    def _ap_stage2(self, res1, sm, sf, avg_poor, mem_poor, avg_rich,
                   mem_rich):
        """Refined references (tk variables are read here), then stage 3."""
        from gui_app.dataref_core import make_reference
        from gui_app.acom_core import detect_peaks_2d
        try:
            kw = self._detect_kw_now()
            rich, poor = sm["rich"], sm["poor"]
            old = self._ap_ctx["initial"]
            new = []
            for k, pat, mem, what in (
                    (poor, avg_poor, mem_poor,
                     f"auto: satellite-free {len(mem_poor)} frames"),
                    (rich, avg_rich, mem_rich,
                     f"auto: strong satellites {len(mem_rich)} frames")):
                o = old[k]
                pk = detect_peaks_2d(pat, **kw)
                lo, hi = self._dr_ring(pat)
                ref = make_reference(o.name, pat, pk, r_min=lo, r_max=hi,
                                     tol=self._dr_tol_val(),
                                     guess=self._dr_guess(pat), origin=what,
                                     colour=o.colour)
                if len(ref.vecs) < 3:
                    raise RuntimeError(f"refined '{o.name}': only "
                                       f"{len(ref.vecs)} spots")
                ref.extra["r_min"], ref.extra["r_max"] = lo, hi
                ref.extra["members"] = np.asarray(mem, np.int64)
                new.append((k, ref))
            new.sort(key=lambda t: t[0])
            self._dr_refs[:] = [r for _k, r in new]
            self._dr_refresh_list()
            model2 = self._pd_model()
            if model2 is None:
                raise RuntimeError("refined references give no model")
            self._ap_ctx["res1"] = res1
            threading.Thread(target=self._ap_stage3, args=(res1, model2),
                             daemon=True).start()
        except Exception as e:
            self._ap_fail(e)

    # ---------------------------------------------------- stage 3 (thread)
    def _ap_stage3(self, res1, model2):
        from gui_app.acom_domains import _cat_map, _radial, _line, _ordered
        from gui_app import phase_discrim as pd
        try:
            ctx = self._ap_ctx
            self._ap_status("4/4 phase map with the refined references")
            res2 = self._pd_compute(model2, ctx["sample"], ctx["stride"],
                                    ctx["detect_kw"])
            self._pd_sat_compute(res2)
            # domain averages + lattices of the two phases
            self._ap_status("4/4 domain averages")
            conf = self._pd_conf_val()
            # final phase: single = the satellite-free frames, double =
            # every other pattern with a measurement (it HAS satellites, to
            # whatever degree); the nearest-reference map stays in the
            # details
            sf = res1["satfree"]; sm = res1["satmaps"]
            cat = _cat_map(res2, conf)
            okm = res2["status"] == pd.OK
            cat = np.where(sf["mask"], sm["poor"],
                           np.where(okm, sm["rich"], cat)).astype(np.int16)
            st = res2["stride"]; Ny, Nx = res2["scan_shape"]
            H, W = np.asarray(res2["cube"][0, 0]).shape
            centre = getattr(self, "_dr_beam_guess", None)
            centre = centre if centre is not None else (H / 2, W / 2)
            rmax = int(min(H, W) / 2)
            sampled = np.zeros(cat.shape, bool); sampled[::st, ::st] = True
            dom = []
            for k, r in enumerate(model2.refs):
                pos = list(zip(*np.nonzero(sampled & (cat == k))))
                if not pos:
                    dom.append(dict(label=r.name, k=k, n=0, avg=None,
                                    lat=None, radial=None))
                    continue
                avg = self._dm_average(res2, pos, centre)
                dom.append(dict(label=r.name, k=k, n=len(pos), avg=avg,
                                lat=self._dm_lattice(res2, k, pos),
                                radial=_radial(avg, centre, rmax)))
            res2["domains"] = dict(out=dom, centre=centre, rmax=rmax,
                                   cat=cat, conf=conf)
            self._dr_post(self._ap_finish, res1, res2)
        except Exception as e:
            self._ap_fail(e)

    # ------------------------------------------------- finish (main thread)
    def _ap_finish(self, res1, res2):
        from gui_app.acom_domains import _line, _ordered
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        from gui_app.acom_tab import _fill_stride
        from gui_app import phase_discrim as pd
        self._dr_busy = False
        self._pdfull = res2
        model = res2["model"]; K = len(model.refs)
        names = [r.name for r in model.refs]
        Ny, Nx = res2["scan_shape"]; st = res2["stride"]
        sm = res2["satmaps"]; sf = res1["satfree"]
        dm = res2["domains"]; out = dm["out"]; centre = dm["centre"]
        rmax = dm["rmax"]
        maps, cols, conf = self._pd_maps(res2)
        catf = dm["cat"].astype(float)
        catf = np.where(catf >= 0, catf,
                        np.where(catf == -1, K, np.where(catf == -2, K + 2,
                                                         K + 3)))
        catf[res2["status"] < 0] = np.nan
        mf = dict(cat=catf); _fill_stride(mf, Ny, Nx, st)
        try:
            inv_a = float(self._inv_ang.get())
        except Exception:
            inv_a = np.nan
        od = None
        try:
            from data import output_dir
            od = output_dir(self._active_sample(), "acom", "phases")
            self._pd_save(res2, maps)
            self._dm_save(output_dir(self._active_sample(), "acom", "phases",
                                     "compare"), "domains", out, inv_a)
            lat = {o["label"]: (o["lat"]["B"] if o["lat"] is not None
                                else np.full((2, 2), np.nan)) for o in out}
            np.savez_compressed(
                os.path.join(od, "auto_phases.npz"),
                phase=mf["cat"], phase_nearest=maps["cat"],
                names=np.array(names),
                codes=f"0..{K - 1} phases, {K} uncertain, {K + 1} cannot "
                      f"separate, {K + 2} off-axis, {K + 3} too few spots; "
                      f"phase: '{sm['npo']}' = satellite-free frames, "
                      f"'{sm['nr']}' = frames with satellites; "
                      f"phase_nearest: nearest refined reference",
                margin=res2["margin"], S=sm["S"], S_se=sm["SE"],
                S_split=sm["tp"]["thr"], satellite_free=sf["mask"],
                n_satellite_free=sf["n_free"], n_initial_frames=sf["n_total"],
                g_vectors=np.array([lat[n] for n in names]),
                g_vector_note="per phase, 2x2 basis of the MAIN lattice in "
                              "detector px, columns = g1, g2 (x, y-up)",
                g1_px=np.array([o["lat"]["g1"].mean() if o["lat"] else
                                np.nan for o in out]),
                g2_px=np.array([o["lat"]["g2"].mean() if o["lat"] else
                                np.nan for o in out]),
                stride=st, notes=np.array(model.notes),
                initial_refs=np.array([r.name for r in
                                       self._ap_ctx["initial"]]))
        except Exception as e:
            print(f"[auto phases] save: {e!r}", flush=True)

        win = tk.Toplevel(self)
        win.title(f"ACOM - phases (automatic)  ({self._active_sample()})")
        win.geometry("1350x900")
        bar = tk.Frame(win); bar.pack(side="top", fill="x")
        fig = Figure(figsize=(13.5, 9), dpi=95, facecolor="#f4f4f4",
                     layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        tk.Label(bar, fg="#555", text="details:").pack(side="left",
                                                      padx=(10, 2))
        for txt, cmd in (("phase map", lambda: self._pd_show_full(res2)),
                         ("satellites", lambda: self._pd_satellites(res2)),
                         ("satellite-free", lambda: self._sf_show(res1, sf)),
                         ("domains / grains", lambda: self._dm_open(res2)),
                         ("initial map", lambda: self._pd_show_full(res1))):
            tk.Button(bar, text=txt, command=cmd).pack(side="left", padx=3)
        axs = fig.subplots(3, 3)
        extra = [(0.80, 0.80, 0.80), (0.55, 0.0, 0.0), (0.0, 0.0, 0.0),
                 (1.0, 1.0, 1.0)]
        # (0,0) final phase map
        ax = axs[0, 0]
        ax.imshow(mf["cat"], cmap=ListedColormap(cols + extra), vmin=-0.5,
                  vmax=K + 3.5, interpolation="nearest")
        ax.legend(handles=[Patch(color=c, label=n) for c, n in
                           zip(cols, names)]
                  + [Patch(color=extra[0], label="uncertain"),
                     Patch(color=extra[2], label="off-axis")],
                  fontsize=7, loc="upper right", framealpha=0.8)
        ax.set_title(f"phase: '{sm['npo']}' = satellite-free frames, "
                     f"'{sm['nr']}' = has satellites\n(nearest-reference "
                     f"map: 'phase map' button)", fontsize=9)
        # (0,1) confidence
        im = axs[0, 1].imshow(np.clip(maps["mg"], 0, 20), cmap="viridis",
                              vmin=0, vmax=20, interpolation="nearest")
        axs[0, 1].set_title("nearest refined reference: confidence "
                            "(log-likelihood ratio; 3 ≈ 20:1)", fontsize=9)
        fig.colorbar(im, ax=axs[0, 1], fraction=0.035)
        # (0,2) satellite strength
        Sfull = dict(S=sm["S"].copy()); _fill_stride(Sfull, Ny, Nx, st)
        fin = sm["S"][np.isfinite(sm["S"])]
        lo, hi = np.percentile(fin, [1, 99]) if fin.size else (0, 1)
        im = axs[0, 2].imshow(Sfull["S"], cmap="magma", vmin=min(lo, 0),
                              vmax=hi, interpolation="nearest")
        fig.colorbar(im, ax=axs[0, 2], fraction=0.035)
        axs[0, 2].set_title(f"satellite strength S (satellite / main "
                            f"reflection)\ncell position(s) {sm['pos']}",
                            fontsize=9)
        # (1,0) satellite-free domain
        mk = dict(mk=sf["mask"].astype(float)); _fill_stride(mk, Ny, Nx, st)
        ax = axs[1, 0]
        ax.imshow(Sfull["S"], cmap="magma", vmin=min(lo, 0), vmax=hi,
                  interpolation="nearest")
        ax.imshow(np.where(mk["mk"] == 1, 1.0, np.nan),
                  cmap=ListedColormap([(0.1, 0.6, 0.1)]), vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"satellite-free domain (green): {sf['n_free']} of "
                     f"{sf['n_total']} frames\n= refined '{sm['npo']}' "
                     f"reference", fontsize=9)
        # (1,1) running mean S / z
        ax = axs[1, 1]
        N, M, SEm, Z = sf["N"], sf["mean"], sf["se"], sf["Z"]
        ax.fill_between(N, M - SEm, M + SEm, color="#1f77b4", alpha=0.25,
                        lw=0)
        ax.plot(N, M, color="#1f77b4", lw=1.2, label="mean S of the set")
        ax.axhline(0, color="0.5", lw=0.6)
        if sf["n_star"] is not None:
            ax.axvline(sf["n_star"], color="g", lw=1.5,
                       label=f"stop N* = {sf['n_star']} (z ≥ 3)")
        ax.set_xlabel("N weakest frames (ranked by the neighbours' S)")
        ax.set_ylabel("mean S ± SE")
        ax.set_title("adding frames from the weakest satellites upward",
                     fontsize=9)
        ax.legend(fontsize=7)
        # (1,2) profile along the doubling direction, both domains
        ax = axs[1, 2]
        G = sf["G"]
        for o, c in zip(out, cols):
            if o["avg"] is None:
                continue
            s_, v_ = _line(o["avg"], centre, sf["g_sat"], sf["other"], rmax)
            ax.semilogy(s_, np.clip(v_, 1e-6, None) / max(v_[4:].max(), 1e-9),
                        color=c, label=f"{o['label']} ({o['n']})")
        for m_ in np.arange(0.5, rmax / G, 0.5):
            ax.axvline(m_ * G, color="0.85" if m_ % 1 else "0.7", lw=0.6,
                       zorder=0)
        ax.set_xlabel("|q| along the doubling direction (px)")
        ax.set_title("domain averages along g (all lattice rows); grey = ½g",
                     fontsize=9)
        ax.legend(fontsize=7)
        # (2,0) / (2,1) domain averages
        for ax, o in zip((axs[2, 0], axs[2, 1]), out):
            if o["avg"] is None:
                ax.axis("off"); continue
            self._dr_imshow(ax, o["avg"])
            if o["lat"] is not None:
                self._dm_arrows(ax, o["lat"]["B"], centre)
            ax.set_title(f"{o['label']}: mean of {o['n']} patterns",
                         fontsize=9)
        # (2,2) table
        ax = axs[2, 2]; ax.axis("off")
        txt = self._dm_table(out, inv_a)
        ax.text(0.0, 1.0, txt, family="monospace", fontsize=7, va="top",
                transform=ax.transAxes)
        ax.text(0.0, 0.42, "\n".join(_wrap(n, 60) for n in model.notes),
                fontsize=7, va="top", transform=ax.transAxes, color="#444")
        for a in (axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[2, 0],
                  axs[2, 1]):
            a.set_xticks([]); a.set_yticks([])
            a.set_facecolor((0.9, 0.9, 0.9))
        init = ", ".join(r.name for r in self._ap_ctx["initial"])
        fig.suptitle(
            f"automatic phases from the initial references ({init}): "
            f"satellite-free domain -> refined '{sm['npo']}', strong "
            f"satellites -> refined '{sm['nr']}', re-mapped.   "
            f"({time.time() - self._ap_ctx['t0']:.0f} s)", fontsize=9)
        if od:
            tk.Label(bar, fg="#555", text=f"saved -> {od}").pack(
                side="left", padx=10)
        cv.draw_idle()
        if od:
            try:
                fig.savefig(os.path.join(od, "auto_phases.png"), dpi=130,
                            facecolor="white")
            except Exception:
                pass
        s = res2["status"][::st, ::st]
        ok = (s == pd.OK) & (res2["margin"][::st, ::st] >= conf)
        self._pd_set_status(
            f"auto phases done: {int(ok.sum())}/{s.size} assigned "
            f"confidently; satellite-free {sf['n_free']} frames; the "
            f"reference list now holds the refined references.")


def _wrap(t, n):
    import textwrap
    return "\n".join(textwrap.wrap(t, n))
