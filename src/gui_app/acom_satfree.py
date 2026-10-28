"""ACOM - satellite-free domain by a cumulative satellite test.

The phase map decides which reference a pattern is CLOSER to, so a pattern
with half-strength satellites lands on the satellite-free reference.  This
tool needs no reference: it ranks the scan positions from the weakest
satellites upward and adds them one by one to a "single" set; the set is
satellite-free as long as its mean satellite level (each frame's S = the
satellite sites over their own local background, relative to its main
reflections, exactly as in the satellite map) is not significantly above
zero.  The first N at which the mean reaches z >= 3 (and stays) is the
stop: those N frames are the satellite-free domain, everything beyond is
doubled to some degree.

Ranking uses the mean S of the 8 NEIGHBOURS (not the frame itself), so a
frame picked for a negative noise excursion cannot bias the running mean;
the running mean itself is taken over the frames' own S.

The window shows the running mean / z-score, the stop, the domain map, the
average pattern of the satellite-free domain vs the rest and their 1-D
profiles along the doubling direction (all lattice rows) with the
half-order positions marked, plus a 1-D half-order test on each (two
main-peak tails fitted between the peaks; advisory).  Outputs
<data>_DINO4DSTEM/acom/phases/satellite_free.npz / .png.  The
satellite-free average can be added as the 'single' reference.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import tkinter as tk
from tkinter import messagebox


class SatFreeMixin:

    def _sf_run(self, res, sat_win=None):
        if getattr(self, "_dr_busy", False):
            messagebox.showinfo("Satellite-free domain", "busy"); return
        self._dr_busy = True
        self._stop_event.clear()
        threading.Thread(target=self._sf_worker, args=(res,),
                         daemon=True).start()

    def _sf_compute(self, res):
        """The satellite-free domain -> out dict (no GUI; raises).  Cached
        in res["satfree"]."""
        if res.get("satfree") is not None:
            return res["satfree"]
        from gui_app import phase_discrim as pd
        from celldenoise_flatten import recentre
        t0 = time.time()
        model = res["model"]
        ent, d, rich, poor = pd._sat_pair(model)
        if ent is None:
            raise RuntimeError("no satellite positions in the model")
        c = ent["coarse"]
        Ny, Nx = res["scan_shape"]; st = res["stride"]
        cube = res["cube"]
        H, W = np.asarray(cube[0, 0]).shape
        centre = getattr(self, "_dr_beam_guess", None)
        centre = centre if centre is not None else (H / 2, W / 2)
        rmax = int(min(H, W) / 2)
        ok = ((res["status"] == pd.OK) & np.isfinite(res["sat"][0])
              & np.isfinite(res["sat"][1]))
        S = np.where(ok, res["sat"][0], np.nan)[::st, ::st]
        SE = np.where(ok, res["sat"][1], np.nan)[::st, ::st]
        # rank by the mean S of the 8 neighbours (own S only when
        # there is no neighbour)
        Rk = np.full_like(S, np.nan)
        for y in range(S.shape[0]):
            for x in range(S.shape[1]):
                if not np.isfinite(S[y, x]):
                    continue
                w = S[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].copy()
                w[min(y, 1), min(x, 1)] = np.nan
                Rk[y, x] = np.nanmean(w) if np.isfinite(w).any() \
                    else S[y, x]
        ys, xs = np.nonzero(np.isfinite(Rk))
        order = np.argsort(Rk[ys, xs], kind="stable")
        ys, xs = ys[order], xs[order]
        Sv, SEv = S[ys, xs], SE[ys, xs]
        ys, xs = ys * st, xs * st
        N = len(ys)
        if N < 20:
            raise RuntimeError("too few positions with a satellite "
                               "measurement")
        n = np.arange(1, N + 1)
        cum = np.cumsum(Sv)
        cse = np.sqrt(np.cumsum(SEv ** 2))
        Z = cum / np.maximum(cse, 1e-12)
        mean = cum / n
        Z[:9] = np.nan                       # fewer than 10 frames
        n_star = pd.satfree_stop(n, np.nan_to_num(Z, nan=-np.inf),
                                 run=max(3, N // 50))
        # the doubling direction (for the profiles): strongest
        # satellite position of the rich reference, median main lattice
        f = pd.OFFSETS[d][np.argmax(ent["mu"][rich][d])]
        lat = res["lat"][c][ys, xs].reshape(-1, 2, 2)
        lat = lat[np.all(np.isfinite(lat.reshape(len(lat), -1)), 1)]
        Bc = np.median(lat, 0)
        g_sat = 2.0 * (Bc @ f)
        G = float(np.linalg.norm(g_sat))
        u = g_sat / G
        cos = [abs(Bc[:, j] @ u) / max(np.linalg.norm(Bc[:, j]), 1e-9)
               for j in (0, 1)]
        other = Bc[:, int(np.argmin(cos))]
        hw = float(getattr(model, "w", 2.0))
        # the two averages (beam-recentred), capped at 1500 frames each
        n_free = n_star if n_star is not None else N
        # a frame ranked in by its neighbours (domain edge) but with
        # satellites of its own (S > 3 SE) is not satellite-free
        keep = np.ones(N, bool)
        keep[:n_free] = Sv[:n_free] <= 3 * SEv[:n_free]
        keep[n_free:] = False
        n_drop = int(n_free - keep.sum())
        mask = np.zeros((Ny, Nx), bool)
        mask[ys[keep], xs[keep]] = True
        rng = np.random.default_rng(0)

        def _avg(idx, what):
            if len(idx) == 0:
                return None
            if len(idx) > 1500:
                idx = np.sort(rng.choice(idx, 1500, replace=False))
            acc = np.zeros((H, W), np.float64)
            for j, i in enumerate(idx):
                if self._stop_event.is_set():
                    raise RuntimeError("stopped by user")
                acc += recentre(np.asarray(cube[ys[i], xs[i]],
                                           np.float32), centre)
                if j % 100 == 0:
                    self._dr_post(self._pd_set_status,
                                  f"satellite-free: averaging {what} "
                                  f"{j}/{len(idx)}  "
                                  f"({time.time() - t0:.0f} s)")
            return (acc / len(idx)).astype(np.float32)
        avg_free = _avg(np.flatnonzero(keep), "satellite-free")
        avg_rest = _avg(np.flatnonzero(~keep), "the rest")
        out = dict(N=n, Z=Z, mean=mean, se=cse / n, S_sorted=Sv,
                   n_star=n_star, n_total=N, mask=mask, keep=keep,
                   n_drop=n_drop, n_free=int(keep.sum()),
                   order=(ys, xs), avg_free=avg_free, avg_rest=avg_rest,
                   G=G, g_sat=g_sat, other=other, centre=centre,
                   rmax=rmax, hw=hw, f=f, elapsed=time.time() - t0)
        res["satfree"] = out
        return out

    def _sf_worker(self, res):
        try:
            out = self._sf_compute(res)
            self._dr_post(self._sf_show, res, out)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._dr_post(self._pd_set_status,
                          f"satellite-free domain failed: {e!r}")
        finally:
            self._dr_busy = False

    def _sf_show(self, res, out):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        from gui_app import phase_discrim as pd
        from gui_app.acom_domains import _line
        from gui_app.acom_tab import _fill_stride
        Ny, Nx = res["scan_shape"]; st = res["stride"]
        N, Z, M, SEm = out["N"], out["Z"], out["mean"], out["se"]
        n_star, n_tot = out["n_star"], out["n_total"]
        od = None
        try:
            from data import output_dir
            od = output_dir(self._active_sample(), "acom", "phases")
            np.savez_compressed(
                os.path.join(od, "satellite_free.npz"),
                N=N, z=Z, mean_S=out["mean"], se_S=out["se"],
                S_sorted=out["S_sorted"],
                n_star=-1 if n_star is None else n_star,
                n_total=n_tot, n_free=out["n_free"], n_dropped=out["n_drop"],
                mask=out["mask"], keep=out["keep"], avg_free=out["avg_free"],
                order_y=out["order"][0], order_x=out["order"][1],
                G=out["G"], g_sat=out["g_sat"], other=out["other"],
                centre=np.asarray(out["centre"], float), stride=st,
                note="mask = satellite-free frames (the n_star weakest, "
                     "ranked by the neighbours' S); mean_S / z = running "
                     "mean of S over the N weakest and its z-score")
        except Exception as e:
            od = None
            print(f"[phases] satellite-free save: {e!r}", flush=True)
        win = tk.Toplevel(self)
        win.title("ACOM - satellite-free domain (cumulative half-order test)")
        win.geometry("1250x820")
        bar = tk.Frame(win); bar.pack(side="top", fill="x")
        fig = Figure(figsize=(12.5, 8), dpi=95, facecolor="#f4f4f4",
                     layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        tk.Button(bar, text="Use as 'single' reference",
                  command=lambda: self._sf_add_reference(res, out)).pack(
                      side="right", padx=8)
        axs = fig.subplots(2, 3)
        # (0, 0): running mean of S
        ax = axs[0, 0]
        ax.fill_between(N, M - SEm, M + SEm, color="#1f77b4", alpha=0.25,
                        lw=0)
        ax.plot(N, M, color="#1f77b4", lw=1.2)
        ax.axhline(0, color="0.5", lw=0.6)
        if n_star is not None:
            ax.axvline(n_star, color="g", lw=1.5)
        ax.set_xlabel("N weakest frames in the set")
        ax.set_ylabel("mean satellite strength S (± SE)")
        ax.set_title("adding frames from the weakest satellites upward:\n"
                     "mean S of the set (satellite / main reflection)",
                     fontsize=9)
        # (0, 1): z
        ax = axs[0, 1]
        ax.plot(N, Z, color="#ff7f0e", lw=1.2)
        ax.axhline(3, color="r", ls="--", lw=1)
        ax.axhline(0, color="0.5", lw=0.6)
        if n_star is not None:
            ax.axvline(n_star, color="g", lw=1.5)
        zf = Z[np.isfinite(Z)]
        if zf.size:
            ax.set_ylim(min(-5, zf.min() - 1), min(max(10, zf.max() + 1),
                                                    60))
        ax.set_xlabel("N weakest frames in the set")
        ax.set_ylabel("z-score of the mean S")
        ax.set_title("is the mean S significantly > 0?  (red = 3σ, "
                     "green = stop N*)", fontsize=9)
        # (0, 2): map
        mk = out["mask"].astype(float)
        mk[~np.isfinite(np.where(res["status"] == pd.OK, res["sat"][0],
                                 np.nan))] = np.nan
        Sm = np.where(out["mask"], np.nan, res["sat"][0])
        maps = dict(mk=mk, Sm=Sm)
        _fill_stride(maps, Ny, Nx, st)
        ax = axs[0, 2]
        fin = res["sat"][0][np.isfinite(res["sat"][0])]
        hi = np.percentile(fin, 99) if fin.size else 1
        im = ax.imshow(maps["Sm"], cmap="magma", vmin=0, vmax=hi,
                       interpolation="nearest")
        ax.imshow(np.where(maps["mk"] == 1, 1.0, np.nan),
                  cmap=ListedColormap([(0.1, 0.6, 0.1)]), vmin=0, vmax=1,
                  interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.035)
        ax.legend(handles=[Patch(color=(0.1, 0.6, 0.1),
                                 label="satellite-free (single)")],
                  fontsize=7, loc="upper right", framealpha=0.8)
        nf, nd = out["n_free"], out["n_drop"]
        ax.set_title((f"satellite-free: {nf} of {n_tot} frames "
                      f"({100 * nf / n_tot:.0f} %)"
                      + (f"\n{nd} edge frames with own S > 3 SE dropped"
                         if nd else "")) if n_star is not None else
                     f"NO satellites appear even with all {n_tot} frames",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor((0.9,) * 3)
        # (1, 0) / (1, 1): the two averages
        for ax, img, lab in ((axs[1, 0], out["avg_free"],
                              f"satellite-free average ({nf})"),
                             (axs[1, 1], out["avg_rest"],
                              f"the rest ({n_tot - nf})")):
            if img is None:
                ax.axis("off"); continue
            self._dr_imshow(ax, img)
            ax.set_title(lab, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        # (1, 2): profiles along the doubling direction
        ax = axs[1, 2]
        G = out["G"]
        for img, col, lab in ((out["avg_free"], "#2ca02c", "satellite-free"),
                              (out["avg_rest"], "#d62728", "the rest")):
            if img is None:
                continue
            s_, v_ = _line(img, out["centre"], out["g_sat"], out["other"],
                           out["rmax"])
            t = pd.half_order_test(s_, v_, G, out["hw"])
            ax.semilogy(s_, np.clip(v_, 1e-6, None) / max(v_[4:].max(), 1e-9),
                        color=col, label=f"{lab}  (½g z {t['z']:+.1f})")
        for m_ in np.arange(0.5, out["rmax"] / G, 0.5):
            ax.axvline(m_ * G, color="0.85" if m_ % 1 else "0.7", lw=0.6,
                       zorder=0)
        ax.set_xlabel("|q| along the doubling direction (px)")
        ax.set_title("profile along g (all lattice rows); grey = ½g\n"
                     "(½g z = 1-D two-tail test, advisory)", fontsize=9)
        ax.legend(fontsize=7)
        fig.suptitle(
            (f"stop at N* = {n_star} (mean S ≥ 3σ above 0 and stays); "
             if n_star is not None else "no stop: ") +
            f"doubling along cell position ({out['f'][0]:g}, "
            f"{out['f'][1]:g}), G = {G:.1f} px, ranked by the neighbours' S   "
            f"({out['elapsed']:.0f} s)", fontsize=9)
        if od:
            tk.Label(bar, fg="#555", text=f"saved -> {od}").pack(
                side="left", padx=10)
        cv.draw_idle()
        if od:
            try:
                fig.savefig(os.path.join(od, "satellite_free.png"), dpi=130,
                            facecolor="white")
            except Exception:
                pass
        self._pd_set_status(
            f"satellite-free domain: {nf} of {n_tot} frames"
            + ("" if n_star is not None else " (no satellites at all)"))

    def _sf_add_reference(self, res, out):
        """The satellite-free average as a data reference (with its member
        positions, so the levels come from its own patterns)."""
        from tkinter import simpledialog
        from gui_app.acom_core import detect_peaks_2d
        keep = np.asarray(out["keep"], bool)
        n = int(keep.sum())
        name = simpledialog.askstring(
            "New reference", f"Add the satellite-free average ({n} frames) "
            f"as a reference.\n\nName:", initialvalue="single", parent=self)
        if name is None or not name.strip():
            return
        if any(r.name == name.strip() for r in self._dr_refs):
            messagebox.showerror("New reference",
                                 f"'{name.strip()}' already exists.")
            return
        Nx = res["scan_shape"][1]
        ys, xs = out["order"]
        avg = out["avg_free"]
        self._note_members(avg, ys[keep] * Nx + xs[keep])
        self._test_pattern = avg
        self._test_peaks = detect_peaks_2d(avg, **self._detect_kw_now())
        self._test_origin = f"satellite-free {n} frames"
        self._dr_name.set(name.strip())
        self._dr_add_reference()
        self._pd_set_status(f"added reference '{name.strip()}' "
                            f"({len(self._dr_refs)} now) - remove the old "
                            f"single, then Map ▶")
