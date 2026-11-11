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
  5. phase map with the refined references; every frame is then split on
     its OWN satellite signal: z = S / SE (S = satellite brightness relative
     to the main spots, SE = its noise in that frame), calibrated on the
     satellite-free frames (median / spread of their z, so "no satellites"
     is z = 0 +- 1 in this dataset).  single z < lower cut, double z >=
     upper cut, undecided between (or one fixed S cut, undecided within 2
     noise of it).  Bias checks: pooled z of every group, agreement with
     the nearest-reference map, one vs two populations of z;
  6. averages of the three groups, main-lattice vectors and the profiles
     along the doubling direction.

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


MIN_FREE = 30            # fewer satellite-free frames: keep the user's single


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
        sp = self._ap_split_settings()
        if sp is None:
            return
        ctx = dict(sample=sample,
                   stride=max(int(self._dr_stride.get() or 1), 1),
                   detect_kw=self._detect_kw_now(), t0=time.time(),
                   initial=refs, split=sp)
        self._dr_busy = True
        self._stop_event.clear()
        self._ap_ctx = ctx
        threading.Thread(target=self._ap_stage1, args=(model, ctx),
                         daemon=True).start()

    def _ap_split_settings(self):
        """The per-frame split settings from the panel (main thread) ->
        dict(mode 'z' | 'S', zs, zd, scut) or None (message shown)."""
        try:
            mode = "S" if "S" in self._ap_cutmode.get() else "z"
            zs = float(self._ap_zs.get()); zd = float(self._ap_zd.get())
            scut = float(self._ap_scut.get())
        except AttributeError:
            return dict(mode="z", zs=2.0, zd=3.0, scut=0.03)
        except ValueError:
            messagebox.showerror("Auto phases", "single z, double z and S "
                                 "cut must be numbers.")
            return None
        if not (0 <= zs <= zd and zd > 0):
            messagebox.showerror("Auto phases", "need 0 <= single z <= "
                                 "double z (e.g. 2 and 3).")
            return None
        return dict(mode=mode, zs=zs, zd=zd, scut=scut)

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
            sf = self._sf_compute(res1, ctx["split"]["zd"])
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
            few = len(mem_poor) < MIN_FREE
            self._ap_ctx["kept_initial_single"] = few
            for k, pat, mem, what in (
                    (poor, avg_poor, mem_poor,
                     f"auto: satellite-free {len(mem_poor)} frames"),
                    (rich, avg_rich, mem_rich,
                     f"auto: strong satellites {len(mem_rich)} frames")):
                o = old[k]
                if k == poor and few:
                    # too few satellite-free frames for a clean average:
                    # keep the user's reference
                    new.append((k, o))
                    continue
                pk = detect_peaks_2d(pat, **kw)
                lo, hi = self._dr_ring(pat)
                base = o.name[:-7] if o.name.endswith(" (auto)") else o.name
                ref = make_reference(base + " (auto)", pat, pk, r_min=lo,
                                     r_max=hi,
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
            # domain averages + lattices of the two phases
            self._ap_status("4/4 domain averages")
            conf = self._pd_conf_val()
            # final phase: every frame on its own satellite signal (z, or
            # one S cut), calibrated on the satellite-free frames; the
            # nearest-reference map stays in the details
            sf = res1["satfree"]; sm = res1["satmaps"]
            st = res2["stride"]; Ny, Nx = res2["scan_shape"]
            sp = ctx["split"]
            # the satellite signal is measured at the positions where the
            # user's INITIAL references differ (validated on them); the
            # refined pair is only checked against it
            okm = res2["status"] == pd.OK
            ok1 = res1["status"] == pd.OK
            S2 = np.where(ok1, res1["sat"][0], np.nan)
            SE2 = np.where(ok1, res1["sat"][1], np.nan)
            okm = okm | ok1
            spl = split_frames(S2, SE2, res1.get("sat_null"), **sp)
            near = _cat_map(res2, conf)
            cat = near.copy()
            lab = spl["lab"]
            cat[okm] = -1                       # undecided / no measurement
            cat[lab == 0] = sm["poor"]
            cat[lab == 2] = sm["rich"]
            cat = cat.astype(np.int16)
            spl["agree"] = agreement(lab, near, sm["poor"], sm["rich"])
            spl["warn"] = []
            e1, d1, r1, _p1 = pd._sat_pair(res1["model"])
            e2, d2, r2, _p2 = pd._sat_pair(model2)
            if e2 is None or r1 != r2 or not np.array_equal(d1, d2):
                spl["warn"].append(
                    "! the refined references differ at other cell "
                    "positions than the initial ones -> the nearest-"
                    "reference map / refined references are unreliable "
                    "(the split uses the initial positions)")
            if ctx.get("kept_initial_single"):
                spl["warn"].append(
                    f"! only {sf['n_free']} satellite-free frames (< "
                    f"{MIN_FREE}): the initial single reference was kept")
            res2["split"] = spl
            res2["sat_split"] = res1["sat"]
            H, W = np.asarray(res2["cube"][0, 0]).shape
            centre = getattr(self, "_dr_beam_guess", None)
            centre = centre if centre is not None else (H / 2, W / 2)
            rmax = int(min(H, W) / 2)
            sampled = np.zeros(cat.shape, bool); sampled[::st, ::st] = True
            dom = []
            groups = [(r.name, k, k) for k, r in enumerate(model2.refs)]
            groups.append(("undecided", -1, sm["rich"]))
            for name, k, klat in groups:
                sel = sampled & (cat == k)
                if k == -1:
                    sel &= lab == 1
                pos = list(zip(*np.nonzero(sel)))
                if not pos:
                    dom.append(dict(label=name, k=k, n=0, avg=None,
                                    lat=None, radial=None))
                    continue
                self._ap_status(f"4/4 averaging '{name}' ({len(pos)})")
                avg = self._dm_average(res2, pos, centre)
                dom.append(dict(label=name, k=k, n=len(pos), avg=avg,
                                lat=self._dm_lattice(res2, klat, pos),
                                radial=_radial(avg, centre, rmax)))
            res2["domains"] = dict(out=dom, centre=centre, rmax=rmax,
                                   cat=cat, conf=conf)
            # domains / statistics / masks use the automatic map; the
            # statistics window also shows the initial-reference map
            res2["auto_cat"] = cat
            res2["auto_res1"] = res1
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
        sm = res1["satmaps"]; sf = res1["satfree"]
        dm = res2["domains"]; out = dm["out"]; centre = dm["centre"]
        rmax = dm["rmax"]
        maps, cols, conf = self._pd_maps(res2)
        catf = dm["cat"].astype(float)
        catf = np.where(catf >= 0, catf,
                        np.where(catf == -1, K, np.where(catf == -2, K + 2,
                                                         K + 3)))
        catf[res2["status"] < 0] = np.nan
        mf = dict(cat=catf); _fill_stride(mf, Ny, Nx, st)
        spl = res2["split"]
        poor, rich = sm["poor"], sm["rich"]
        zf = dict(z=spl["z"].copy(), lab=spl["lab"].copy())
        _fill_stride(zf, Ny, Nx, st)
        Sfull = dict(S=sm["S"].copy()); _fill_stride(Sfull, Ny, Nx, st)
        # display order: single, undecided, double
        grp = [out[poor], out[K], out[rich]]
        gcol = [cols[poor], (0.55, 0.55, 0.55), cols[rich]]
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
                                     "compare"), "domains",
                          [o for o in grp if o["avg"] is not None], inv_a)
            lat = {o["label"]: (o["lat"]["B"] if o["lat"] is not None
                                else np.full((2, 2), np.nan)) for o in out}
            np.savez_compressed(
                os.path.join(od, "auto_phases.npz"),
                phase=mf["cat"], phase_nearest=maps["cat"],
                names=np.array(names),
                codes=f"0..{K - 1} phases, {K} undecided, {K + 1} cannot "
                      f"separate, {K + 2} off-axis, {K + 3} too few spots; "
                      f"phase: per-frame split on the satellite signal "
                      f"('{sm['npo']}' = single, '{sm['nr']}' = double); "
                      f"phase_nearest: nearest refined reference",
                split=zf["lab"], split_codes="0 single, 1 undecided, "
                                             "2 double, nan none",
                z=zf["z"], z_note="calibrated z = (S/SE - mu0) / s0, "
                                  "mu0 / s0 from the satellite-free frames",
                mu0=spl["mu0"], s0=spl["s0"], n_noise_ref=spl["n_null"],
                cut_mode=spl["mode"], z_single=spl["zs"],
                z_double=spl["zd"], S_cut=spl["scut"],
                pooled_z=np.array([spl["pooled"][g][1] for g in
                                   ("single", "undecided", "double")]),
                agreement=spl["agree"],
                agreement_note="rows single/undecided/double; columns "
                               "nearest map: single-ref, double-ref, "
                               "uncertain",
                margin=res2["margin"], S=sm["S"], S_se=sm["SE"],
                satellite_free=sf["mask"],
                n_satellite_free=sf["n_free"], n_initial_frames=sf["n_total"],
                g_vectors=np.array([lat[n] for n in names]),
                g_vector_note="per phase, 2x2 basis of the MAIN lattice in "
                              "detector px, columns = g1, g2 (x, y-up)",
                group_labels=np.array([o["label"] for o in grp]),
                g1_px=np.array([o["lat"]["g1"].mean() if o["lat"] else
                                np.nan for o in grp]),
                g2_px=np.array([o["lat"]["g2"].mean() if o["lat"] else
                                np.nan for o in grp]),
                stride=st, notes=np.array(model.notes),
                initial_refs=np.array([r.name for r in
                                       self._ap_ctx["initial"]]))
        except Exception as e:
            print(f"[auto phases] save: {e!r}", flush=True)

        win = tk.Toplevel(self)
        win.title(f"ACOM - phases (automatic, REFINED + re-mapped)  "
                  f"({self._active_sample()})")
        win.geometry("1500x930")
        bar = tk.Frame(win); bar.pack(side="top", fill="x")
        fig = Figure(figsize=(15, 9.3), dpi=95, facecolor="#f4f4f4",
                     layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        tk.Label(bar, fg="#555", text="details:").pack(side="left",
                                                      padx=(10, 2))
        for txt, cmd in (("phase map", lambda: self._pd_show_full(res2)),
                         ("satellites", lambda: self._pd_satellites(res1)),
                         ("satellite-free", lambda: self._sf_show(res1, sf)),
                         ("domains / grains", lambda: self._dm_open(res2)),
                         ("initial map", lambda: self._pd_show_full(res1))):
            tk.Button(bar, text=txt, command=cmd).pack(side="left", padx=3)
        axs = fig.subplots(3, 4)
        extra = [(0.55, 0.55, 0.55), (0.55, 0.0, 0.0), (0.0, 0.0, 0.0),
                 (1.0, 1.0, 1.0)]
        zmode = spl["mode"] != "S"
        zs, zd, scut = spl["zs"], spl["zd"], spl["scut"]
        cut_txt = (f"single z < {zs:g}, double z ≥ {zd:g}" if zmode else
                   f"S cut {scut:g} (undecided within 2 noise)")
        # (0,0) the class map
        ax = axs[0, 0]
        ax.imshow(mf["cat"], cmap=ListedColormap(cols + extra), vmin=-0.5,
                  vmax=K + 3.5, interpolation="nearest")
        ax.legend(handles=[Patch(color=cols[poor], label=f"single "
                                 f"({names[poor]})"),
                           Patch(color=extra[0], label="undecided"),
                           Patch(color=cols[rich], label=f"double "
                                 f"({names[rich]})"),
                           Patch(color=extra[2], label="off-axis")],
                  fontsize=7, loc="upper right", framealpha=0.8)
        lv = spl["lab"][np.isfinite(spl["lab"])]
        fr = [100 * np.mean(lv == g) if lv.size else 0 for g in (0, 1, 2)]
        ax.set_title(f"per-frame split\nsingle {fr[0]:.0f}% | undecided "
                     f"{fr[1]:.0f}% | double {fr[2]:.0f}%", fontsize=8.5)

        def _marks(cb, ax_, cuts):
            for v, c, t in cuts:
                cb.ax.axhline(v, color=c, lw=3)
        # (0,1) calibrated z map, cuts marked on the colour bar
        zfin = spl["z"][np.isfinite(spl["z"])]
        zlo, zhi = (np.percentile(zfin, [1, 99]) if zfin.size else (0, 1))
        zlo, zhi = min(zlo, -2.0), max(zhi, zd + 2.0)
        ax = axs[0, 1]
        im = ax.imshow(zf["z"], cmap="magma", vmin=zlo, vmax=zhi,
                       interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        if zmode:
            _marks(cb, ax, [(zs, "#00c8ff", f"single <"),
                            (zd, "#ff3030", f"double ≥")])
        ax.set_title("satellite signal z = S / noise (0 = none)\n"
                     + (f"bar: cyan = single < {zs:g}, red = double ≥ "
                        f"{zd:g}" if zmode else "is it there? (S: how much)"),
                     fontsize=8.5)
        # (0,2) S map (physical magnitude), S cut marked in fixed-S mode
        fin = sm["S"][np.isfinite(sm["S"])]
        lo, hi = np.percentile(fin, [1, 99]) if fin.size else (0, 1)
        ax = axs[0, 2]
        im = ax.imshow(Sfull["S"], cmap="magma", vmin=min(lo, 0), vmax=hi,
                       interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        if not zmode:
            _marks(cb, ax, [(scut, "#ff3030", "cut")])
        ax.set_title(f"S = satellite / main brightness {sm['pos']}\n"
                     + (f"bar: red = cut {scut:g}" if not zmode else
                        "how much satellite"), fontsize=8.5)
        # (0,3) histogram: two populations or one continuous?
        ax = axs[0, 3]
        two = spl["two_pop"]
        hv = spl["z"] if zmode else spl["lab"] * 0 + sm["S"]
        hv_ok = np.isfinite(hv) & np.isfinite(spl["lab"])
        if hv_ok.any():
            x_ = hv[hv_ok]
            lo_, hi_ = np.percentile(x_, [0.5, 99.5])
            bins = np.linspace(lo_, hi_, 60)
            ax.hist([x_[spl["lab"][hv_ok] == g] for g in (0, 1, 2)],
                    bins=bins, stacked=True, color=gcol,
                    label=["single", "undecided", "double"])
            for v, c in (((zs, "#00a0d0"), (zd, "#d02020")) if zmode
                         else ((scut, "#d02020"),)):
                ax.axvline(v, color=c, lw=1.5, ls="--")
            ax.legend(fontsize=7)
        ax.set_xlabel("z (calibrated)" if zmode else "S")
        ax.set_ylabel("frames")
        if two is not None and two.get("bimodal"):
            verdict = "TWO populations: a real two-phase split"
        elif two is not None:
            verdict = ("ONE continuous population: satellites vary "
                       "gradually,\nthe split is an arbitrary cut")
        else:
            verdict = "too few frames to tell one / two populations"
        ax.set_title(verdict, fontsize=9,
                     color="k" if (two or {}).get("bimodal") else "#b00")
        # (1,0) satellite-free domain (noise reference + single reference)
        mk = dict(mk=sf["mask"].astype(float)); _fill_stride(mk, Ny, Nx, st)
        ax = axs[1, 0]
        ax.imshow(Sfull["S"], cmap="magma", vmin=min(lo, 0), vmax=hi,
                  interpolation="nearest")
        ax.imshow(np.where(mk["mk"] == 1, 1.0, np.nan),
                  cmap=ListedColormap([(0.1, 0.6, 0.1)]), vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"satellite-free (green): {sf['n_free']} of "
                     f"{sf['n_total']}\n-> refined single reference",
                     fontsize=8.5)
        # (1,1) running mean S
        ax = axs[1, 1]
        N, M, SEm = sf["N"], sf["mean"], sf["se"]
        ax.fill_between(N, M - SEm, M + SEm, color="#1f77b4", alpha=0.25,
                        lw=0)
        ax.plot(N, M, color="#1f77b4", lw=1.2, label="mean S of the set")
        ax.axhline(0, color="0.5", lw=0.6)
        if sf["n_star"] is not None:
            ax.axvline(sf["n_star"], color="g", lw=1.5,
                       label=f"stop N* = {sf['n_star']} "
                             f"(z ≥ {sf.get('z_thr', 3):g})")
        ax.set_xlabel("N weakest frames (ranked by the neighbours' S)")
        ax.set_ylabel("mean S ± SE")
        ax.set_title("building the satellite-free reference",
                     fontsize=9)
        ax.legend(fontsize=7)
        # (1,2) profile along the doubling direction, three groups
        ax = axs[1, 2]
        G = sf["G"]
        for o, c in zip(grp, gcol):
            if o["avg"] is None:
                continue
            s_, v_ = _line(o["avg"], centre, sf["g_sat"], sf["other"], rmax)
            ax.semilogy(s_, np.clip(v_, 1e-6, None) / max(v_[4:].max(), 1e-9),
                        color=c, label=f"{o['label']} ({o['n']})")
        for m_ in np.arange(0.5, rmax / G, 0.5):
            ax.axvline(m_ * G, color="0.85" if m_ % 1 else "0.7", lw=0.6,
                       zorder=0)
        ax.set_xlabel("|q| along the doubling direction (px)")
        ax.set_title("group averages along g (all lattice rows); grey = ½g",
                     fontsize=9)
        ax.legend(fontsize=7)
        # (1,3) the checks
        ax = axs[1, 3]; ax.axis("off")
        ax.text(0.0, 1.0, self._ap_checks_text(spl, names[poor],
                                               names[rich]),
                family="monospace", fontsize=7, va="top",
                transform=ax.transAxes)
        # (2,0..2) group averages
        for ax, o in zip(axs[2, :3], grp):
            if o["avg"] is None:
                ax.text(0.5, 0.5, f"{o['label']}: no frames", ha="center",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            self._dr_imshow(ax, o["avg"])
            if o["lat"] is not None:
                self._dm_arrows(ax, o["lat"]["B"], centre)
            ax.set_title(f"{o['label']}: mean of {o['n']} patterns",
                         fontsize=9)
        # (2,3) lattice table + notes
        ax = axs[2, 3]; ax.axis("off")
        ax.text(0.0, 1.0, self._dm_table(grp, inv_a), family="monospace",
                fontsize=6.5, va="top", transform=ax.transAxes)
        ax.text(0.0, 0.5, "\n".join(_wrap(n, 55) for n in model.notes),
                fontsize=7, va="top", transform=ax.transAxes, color="#444")
        for a in (axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[2, 0],
                  axs[2, 1], axs[2, 2]):
            a.set_xticks([]); a.set_yticks([])
            a.set_facecolor((0.9, 0.9, 0.9))
        init = ", ".join(r.name for r in self._ap_ctx["initial"])
        fig.suptitle(
            f"automatic phases from ({init}): refined references, then every "
            f"frame split on its own satellite signal ({cut_txt}).   "
            f"({time.time() - self._ap_ctx['t0']:.0f} s)", fontsize=9)
        tk.Label(bar, fg="#555", text="click a map: its diffraction "
                 "pattern").pack(side="left", padx=10)
        self._pd_click_inspect(cv, [axs[0, 0], axs[0, 1], axs[0, 2],
                                    axs[1, 0]], res2, tb)
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
        self._pd_set_status(
            f"auto phases DONE (refined + re-mapped): single {fr[0]:.0f} %, "
            f"undecided {fr[1]:.0f} %, double {fr[2]:.0f} % ({cut_txt}); "
            + ("two populations." if (two or {}).get("bimodal") else
               "ONE continuous population - the split is a cut."))

    def _ap_checks_text(self, spl, n_single, n_double):
        """The bias checks as text."""
        L = ["BIAS CHECKS"] + [_wrap(w, 62) for w in spl.get("warn", [])]
        if spl["calibrated"]:
            L.append(f"noise reference: {spl['n_null']} empty cell-position "
                     f"values (all frames)")
            L.append(f"  their z: centre {spl['mu0']:+.2f}, spread "
                     f"{spl['s0']:.2f}  (theory 0, 1)")
            if spl["s0"] > 1.5 or spl["s0"] < 0.67:
                L.append(f"  ! the formula's noise is off by x{spl['s0']:.2f}"
                         f" - corrected")
            if abs(spl["mu0"]) > 1:
                L.append("  ! offset > 1: background bias at the satellite "
                         "sites - corrected")
        else:
            L.append(f"noise reference: only {spl['n_null']} frames -> "
                     f"theory (0, 1) used")
        L.append("")
        L.append("pooled z of every group (all its frames together):")
        warn = []
        for g in ("single", "undecided", "double"):
            n, zp = spl["pooled"][g]
            L.append(f"  {g:<10}{n:>6} frames   z {zp:+8.1f}" if n else
                     f"  {g:<10}{0:>6} frames")
        n, zp = spl["pooled"]["single"]
        if n and zp >= 3:
            warn.append(f"! 'single' frames TOGETHER show satellites "
                        f"(z {zp:.1f}):\n  weak satellites hide below the "
                        f"cut - continuous?")
        n, zp = spl["pooled"]["double"]
        if n and zp < 3:
            warn.append("! 'double' frames together are not clearly above 0")
        L += warn or ["  ok: single ~ 0, double clearly > 0"]
        L.append("")
        A = spl["agree"]
        L.append("split vs nearest-reference map:")
        L.append(f"  {'':<10}{'->single':>9}{'->double':>9}{'uncert.':>8}")
        for i, g in enumerate(("single", "undecided", "double")):
            L.append(f"  {g:<10}{A[i, 0]:>9}{A[i, 1]:>9}{A[i, 2]:>8}")
        tot = A[[0, 2]][:, :2].sum()
        if tot:
            agr = (A[0, 0] + A[2, 1]) / tot
            L.append(f"  agree on decided frames: {100 * agr:.0f} %")
        return "\n".join(L)


def split_frames(S, SE, znull, mode="z", zs=2.0, zd=3.0, scut=0.03):
    """Per-frame single / undecided / double from the frame's own satellite
    signal.

    S, SE: satellite strength and its noise per frame (nan = none);
    znull: z (value / SE) at the EMPTY cell positions of every frame
    (P, Ny, Nx) - what "no signal" looks like in this data, independent
    of which frames are single.  z = S / SE, calibrated so that empty
    positions are z = 0 +- 1: z_cal = (z - mu0) / s0, mu0 = median and
    s0 = robust spread (1.4826 MAD) of the empty-position z; the theory
    (0, 1) is used when fewer than 20 are available.
    mode 'z': single z < zs, double z >= zd, undecided between.
    mode 'S': one cut on S (minus the calibrated offset) for every frame;
    undecided within 2 calibrated noise of it.
    -> dict(lab (0 single, 1 undecided, 2 double, nan none), z, mu0, s0,
    n_null, calibrated, pooled {group: (n, pooled z)}, two_pop dict)."""
    S = np.asarray(S, float); SE = np.asarray(SE, float)
    ok = np.isfinite(S) & np.isfinite(SE) & (SE > 0)
    z = np.full(S.shape, np.nan)
    z[ok] = S[ok] / SE[ok]
    zn = (np.asarray(znull, float).ravel() if znull is not None
          else np.zeros(0))
    zn = zn[np.isfinite(zn)]
    if zn.size >= 20:
        mu0 = float(np.median(zn))
        s0 = float(1.4826 * np.median(np.abs(zn - mu0)))
        s0 = float(np.clip(s0, 0.3, 10.0))
        cal = True
    else:
        mu0, s0, cal = 0.0, 1.0, False
    zc = (z - mu0) / s0
    lab = np.full(S.shape, np.nan)
    if mode == "S":
        Sc = S - mu0 * SE                    # remove the offset (S units)
        nz = 2.0 * s0 * SE
        lab[ok] = np.where(np.abs(Sc[ok] - scut) < nz[ok], 1,
                           np.where(Sc[ok] < scut, 0, 2))
    else:
        lab[ok] = np.where(zc[ok] < zs, 0, np.where(zc[ok] >= zd, 2, 1))
    pooled = {}
    for g, nm in ((0, "single"), (1, "undecided"), (2, "double")):
        m = lab == g
        if m.any():
            num = np.sum(S[m] - mu0 * SE[m])
            den = s0 * np.sqrt(np.sum(SE[m] ** 2))
            pooled[nm] = (int(m.sum()), float(num / max(den, 1e-12)))
        else:
            pooled[nm] = (0, np.nan)
    two = None
    try:
        from gui_app import phase_discrim as pd
        bi, info = pd.bimodality(zc[ok] if mode != "S" else S[ok])
        two = dict(bimodal=bi, **info) if bi is not None else None
    except Exception:
        pass
    return dict(lab=lab, z=zc, z_raw=z, mu0=mu0, s0=s0, n_null=int(zn.size),
                calibrated=cal, pooled=pooled, two_pop=two, mode=mode,
                zs=zs, zd=zd, scut=scut)


def agreement(lab, near, poor, rich):
    """Rows: the split (single / undecided / double); columns: the
    nearest-refined-reference map (poor, rich, uncertain) -> 3x3 counts."""
    out = np.zeros((3, 3), int)
    for i in range(3):
        m = lab == i
        out[i, 0] = int(np.sum(m & (near == poor)))
        out[i, 1] = int(np.sum(m & (near == rich)))
        out[i, 2] = int(np.sum(m & (near == -1)))
    return out


def _wrap(t, n):
    import textwrap
    return "\n".join(textwrap.wrap(t, n))
