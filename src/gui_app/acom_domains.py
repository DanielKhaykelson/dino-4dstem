"""acom_domains.py -- post-processing of a phase map: domain masks, and
phase A vs phase B compared as whole domains or grain by grain.

  Export masks   - one mask per phase (confident pixels), uncertain,
                   off-axis, and the grain labels (npz + png).
  Domains        - average pattern of every pixel of phase A and of B (each
                   pattern re-centred on its own beam first), their
                   difference; 1-D radial profiles and line profiles along
                   the two lattice directions (log y); per-pixel fitted
                   lattice |g1|, |g2|, cell area per phase (mean ± s.e.m.,
                   histograms) - sub-pixel, the sensitive test for a slightly
                   different spacing.
  Grains         - connected regions of each phase (>= N pixels): the same
                   numbers per grain.

Mixed into ACOMTabPanel; works on the result of the phase map (_pdfull).
"""
from __future__ import annotations

import os

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox


def _cat_map(res, conf):
    """Per scan pixel: phase index (confident), -1 uncertain, -2 off-axis,
    -3 other / not sampled."""
    from gui_app import phase_discrim as pd
    st = res["status"]
    cat = np.full(st.shape, -3, np.int16)
    ok = st == pd.OK
    cat[ok & (res["margin"] >= conf)] = res["best"][ok & (res["margin"] >= conf)]
    cat[ok & (res["margin"] < conf)] = -1
    cat[st == pd.OFF_AXIS] = -2
    return cat


def _fill(a, stride):
    """Strided result -> every pixel (copy each sample into its block)."""
    if stride <= 1:
        return a.copy()
    out = a.copy()
    Ny, Nx = a.shape
    for y in range(0, Ny, stride):
        for x in range(0, Nx, stride):
            out[y:y + stride, x:x + stride] = a[y, x]
    return out


def _radial(img, centre, rmax):
    H, W = img.shape
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - centre[0], xx - centre[1]).astype(int)
    m = r < rmax
    s = np.bincount(r[m], img[m], minlength=rmax)
    n = np.bincount(r[m], minlength=rmax)
    return s / np.maximum(n, 1)


def _line(img, centre, g, other, rmax, width=3):
    """Intensity along lattice direction g, averaged over EVERY lattice row
    parallel to it (rows through k * other, k = ..., -1, 0, 1, ...), 3 px
    wide, both directions folded.  A profile through the beam alone sees
    only the central row, where superlattice reflections are often
    systematically absent (glide / screw extinctions) although present on
    the other rows.  -> (position along g in px, mean intensity)."""
    from scipy.ndimage import map_coordinates
    g = np.asarray(g, float); other = np.asarray(other, float)
    u = g / np.linalg.norm(g)
    nrm = np.array([-u[1], u[0]])
    s = np.arange(0, rmax, 0.5)
    H, W = img.shape
    acc = np.zeros_like(s); cnt = np.zeros_like(s)
    offs = np.linspace(-(width - 1) / 2, (width - 1) / 2, width)
    kmax = int(rmax / max(abs(other @ nrm), 1e-6)) + 1
    for k in range(-kmax, kmax + 1):
        foot = k * other
        for sign in (1, -1):
            for o in offs:
                x = foot[0] + sign * s * u[0] + o * nrm[0]
                y = foot[1] + sign * s * u[1] + o * nrm[1]
                rows, cols = centre[0] - y, centre[1] + x
                ok = ((rows >= 0) & (rows < H - 1) & (cols >= 0)
                      & (cols < W - 1) & (np.hypot(x, y) < rmax))
                v = map_coordinates(img, [rows, cols], order=1,
                                    mode="nearest")
                acc += np.where(ok, v, 0); cnt += ok
    return s, acc / np.maximum(cnt, 1)


def _ordered(B):
    """Lattice basis with the shorter vector first (g1 = short, g2 = long,
    as in the tables)."""
    B = np.asarray(B, float)
    return B if np.linalg.norm(B[:, 0]) <= np.linalg.norm(B[:, 1])         else B[:, ::-1]


class DomainMixin:

    def _dm_arrows(self, ax, B, centre):
        """g1 (yellow, shorter) and g2 (green) from the beam."""
        B = _ordered(B)
        cy, cx = centre
        for j, (nm, col) in enumerate((("g1", "#ffd000"), ("g2", "#40ff40"))):
            g = B[:, j]
            ax.annotate("", xy=(cx + g[0], cy - g[1]), xytext=(cx, cy),
                        arrowprops=dict(arrowstyle="->", color=col, lw=2))
            ax.text(cx + 1.18 * g[0], cy - 1.18 * g[1], nm, color=col,
                    fontsize=11, weight="bold", ha="center", va="center")

    # ------------------------------------------------------------------
    def _dm_state(self, res):
        conf = self._pd_conf_val()
        cat = _cat_map(res, conf)
        names = [r.name for r in res["model"].refs]
        return cat, names

    def _dm_grains(self, cat, K, stride, min_px):
        """Connected (8-conn) regions of each phase on the sampled grid
        -> label map (sampled grid), list of (label, phase, n_samples)."""
        from scipy.ndimage import label
        sub = cat[::stride, ::stride]
        lab = np.zeros(sub.shape, np.int32)
        grains = []
        nxt = 1
        for k in range(K):
            l, n = label(sub == k, structure=np.ones((3, 3)))
            for g in range(1, n + 1):
                m = l == g
                if m.sum() >= min_px:
                    lab[m] = nxt
                    grains.append((nxt, k, int(m.sum())))
                    nxt += 1
        full = np.zeros(cat.shape, np.int32)
        full[::stride, ::stride] = lab
        return _fill(full, stride), lab, grains

    # ------------------------------------------------------------------
    def _dm_export(self, res):
        from data import output_dir
        cat, names = self._dm_state(res)
        st = res["stride"]
        catf = _fill(cat, st)
        try:
            mn = max(int(self._dm_min.get()), 1)
        except Exception:
            mn = 5
        glab, _sub, grains = self._dm_grains(cat, len(names), st, mn)
        d = output_dir(self._active_sample(), "acom", "phases", "masks")
        masks = {f"phase_{n}": catf == k for k, n in enumerate(names)}
        masks.update(uncertain=catf == -1, off_axis=catf == -2)
        np.savez_compressed(os.path.join(d, "phase_masks.npz"),
                            grain_labels=glab,
                            grain_phase=np.array([g[1] for g in grains]),
                            names=np.array(names), **masks)
        from matplotlib.figure import Figure
        for nm, m in masks.items():
            fig = Figure(figsize=(5, 5 * m.shape[0] / max(m.shape[1], 1) + .4))
            ax = fig.add_subplot(111)
            ax.imshow(m, cmap="gray", interpolation="nearest")
            ax.set_title(nm); ax.set_xticks([]); ax.set_yticks([])
            fig.savefig(os.path.join(d, f"mask_{nm}.png".replace(" ", "_")),
                        dpi=120, bbox_inches="tight")
        self._pd_set_status(f"masks ({', '.join(masks)}) + grain labels "
                            f"-> {d}")
        return d

    # ------------------------------------------------------------------
    def _dm_open(self, res):
        """Small control window: phases A / B, minimum grain size."""
        if getattr(self, "_dm_win", None) is not None:
            try:
                self._dm_win.destroy()
            except Exception:
                pass
        _cat, names = self._dm_state(res)
        w = tk.Toplevel(self)
        w.title("Compare domains")
        w.geometry("380x250")
        self._dm_win = w
        f = ctk.CTkFrame(w); f.pack(fill="both", expand=True, padx=6, pady=6)
        self._dm_a = ctk.StringVar(value=names[0])
        self._dm_b = ctk.StringVar(value=names[1] if len(names) > 1
                                   else names[0])
        for lab, var in (("phase A", self._dm_a), ("phase B", self._dm_b)):
            r = ctk.CTkFrame(f, fg_color="transparent"); r.pack(fill="x",
                                                               pady=2)
            ctk.CTkLabel(r, text=lab, width=70).pack(side="left")
            ctk.CTkOptionMenu(r, variable=var, values=names,
                              width=200).pack(side="left")
        r = ctk.CTkFrame(f, fg_color="transparent"); r.pack(fill="x", pady=2)
        ctk.CTkLabel(r, text="grains ≥", width=70).pack(side="left")
        self._dm_min = ctk.StringVar(value="5")
        ctk.CTkEntry(r, textvariable=self._dm_min, width=50).pack(side="left")
        ctk.CTkLabel(r, text="scan pixels (sampled)").pack(side="left",
                                                           padx=4)
        r = ctk.CTkFrame(f, fg_color="transparent"); r.pack(fill="x",
                                                           pady=(8, 2))
        ctk.CTkButton(r, text="Whole domains ▶", width=110,
                      command=lambda: self._dm_compare(res, "domains")
                      ).pack(side="left", padx=2)
        ctk.CTkButton(r, text="Grains ▶", width=80,
                      command=lambda: self._dm_compare(res, "grains")
                      ).pack(side="left", padx=2)
        ctk.CTkButton(r, text="Export masks", width=100,
                      command=lambda: self._dm_export(res)
                      ).pack(side="left", padx=2)
        ctk.CTkButton(f, text="Statistics (areas, grain sizes, interfaces) ▶",
                      command=lambda: self._dm_stats(res)
                      ).pack(fill="x", pady=(6, 0))

    # ------------------------------------------------------------------
    def _dm_average(self, res, positions, centre, cap=1024):
        """Mean pattern of the given (y, x) positions, each re-centred on
        its own beam (integer shift) so drift does not blur the spots."""
        from celldenoise_flatten import recentre
        cube = res["cube"]; Nx = res["scan_shape"][1]
        pos = list(positions)
        if len(pos) > cap:
            pos = [pos[i] for i in np.random.default_rng(0).choice(
                len(pos), cap, replace=False)]
        acc = None
        for (y, x) in pos:
            f = np.asarray(cube[y, x], np.float32)
            f = recentre(f, centre)
            acc = f.astype(np.float64) if acc is None else acc + f
        return (acc / max(len(pos), 1)).astype(np.float32)

    def _dm_lattice(self, res, k, positions):
        """|g1|, |g2| (px) and cell area of phase k's MAIN lattice at the
        given positions (per-pixel fits of the common / coarse lattice)."""
        lat = res.get("lat")
        if lat is None:
            return None
        model = res["model"]
        # main lattice: the coarse one of a shared pair containing k, else
        # k's own
        main = k
        for (a, b), ent in model.pairs.items():
            if k in (a, b) and ent.get("mode") == "shared":
                main = ent["coarse"]
        v = np.array([lat[main, y, x] for (y, x) in positions])
        v = v[np.all(np.isfinite(v), axis=1)]
        if not len(v):
            return None
        B = v.reshape(-1, 2, 2)
        g1 = np.hypot(B[:, 0, 0], B[:, 1, 0]); g2 = np.hypot(B[:, 0, 1],
                                                          B[:, 1, 1])
        # consistent order: shorter first
        a1, a2 = np.minimum(g1, g2), np.maximum(g1, g2)
        area = np.abs(B[:, 0, 0] * B[:, 1, 1] - B[:, 0, 1] * B[:, 1, 0])
        return dict(g1=a1, g2=a2, area=area, B=np.nanmedian(B, 0))

    def _dm_compare(self, res, mode):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from data import output_dir
        cat, names = self._dm_state(res)
        ka, kb = names.index(self._dm_a.get()), names.index(self._dm_b.get())
        st = res["stride"]; Ny, Nx = res["scan_shape"]
        centre = getattr(self, "_dr_beam_guess", None)
        H, W = np.asarray(res["cube"][0, 0]).shape
        centre = centre if centre is not None else (H / 2, W / 2)
        rmax = int(min(H, W) / 2)
        try:
            inv_a = float(self._inv_ang.get())
        except Exception:
            inv_a = np.nan
        try:
            mn = max(int(self._dm_min.get()), 1)
        except Exception:
            mn = 5
        sampled = np.zeros(cat.shape, bool); sampled[::st, ::st] = True
        groups = []                                  # (label, phase, positions)
        if mode == "domains":
            for k in (ka, kb):
                pos = list(zip(*np.nonzero(sampled & (cat == k))))
                groups.append((names[k], k, pos))
        else:
            glab, sub, grains = self._dm_grains(cat, len(names), st, mn)
            for (g, k, n) in grains:
                if k not in (ka, kb):
                    continue
                ys, xs = np.nonzero(sub == g)
                groups.append((f"{names[k]} #{g}", k, list(zip(ys * st,
                                                               xs * st))))
        groups = [g for g in groups if len(g[2])]
        if not groups:
            messagebox.showinfo("Domains", "No confident pixels for these "
                                "phases.")
            return
        self._pd_set_status(f"averaging {len(groups)} group(s)…")
        self.update_idletasks()
        out = []
        for lab, k, pos in groups:
            avg = self._dm_average(res, pos, centre)
            latt = self._dm_lattice(res, k, pos)
            out.append(dict(label=lab, k=k, n=len(pos), avg=avg, lat=latt,
                            radial=_radial(avg, centre, rmax)))
        d = output_dir(self._active_sample(), "acom", "phases", "compare")
        self._dm_save(d, mode, out, inv_a)

        win = tk.Toplevel(self)
        win.title(f"Domains - {names[ka]} vs {names[kb]} ({mode})")
        win.geometry("1300x900")
        bar = tk.Frame(win); bar.pack(side="top", fill="x")
        fig = Figure(figsize=(13, 9), dpi=95, layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        tk.Label(bar, fg="#555", text=f"saved -> {d}").pack(side="left",
                                                           padx=10)
        cols = {ka: "#1f77b4", kb: "#ff7f0e"}
        if mode == "domains":
            A = next(o for o in out if o["k"] == ka)
            Bd = next((o for o in out if o["k"] == kb), A)
            axs = fig.subplots(2, 3)
            for ax, o in zip(axs[0, :2], (A, Bd)):
                self._dr_imshow(ax, o["avg"])
                if o["lat"] is not None:
                    self._dm_arrows(ax, o["lat"]["B"], centre)
                ax.set_title(f"{o['label']}: mean of {o['n']} patterns",
                             fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
            na = A["avg"] / max(A["avg"].mean(), 1e-9)
            nb = Bd["avg"] / max(Bd["avg"].mean(), 1e-9)
            dif = na - nb
            lim = np.percentile(np.abs(dif), 99.5)
            axs[0, 2].imshow(dif, cmap="RdBu_r", vmin=-lim, vmax=lim)
            axs[0, 2].set_title(f"{A['label']} − {Bd['label']} (each / its "
                                f"mean)", fontsize=10)
            axs[0, 2].set_xticks([]); axs[0, 2].set_yticks([])
            ax = axs[1, 0]
            for o in (A, Bd):
                ax.semilogy(np.arange(len(o["radial"])),
                            np.clip(o["radial"], 1e-6, None) /
                            max(o["radial"][5:].max(), 1e-9),
                            color=cols[o["k"]], label=o["label"])
            ax.set_xlabel("r (px)"); ax.set_ylabel("radial mean (norm.)")
            ax.set_title("radial profile", fontsize=10); ax.legend(fontsize=8)
            ax = axs[1, 1]
            latA = A["lat"]
            if latA is not None:
                Bo = _ordered(latA["B"])
                for j, (gname, ls) in enumerate((("g1", "-"), ("g2", "--"))):
                    gv, ov = Bo[:, j], Bo[:, 1 - j]
                    for o in (A, Bd):
                        s, pr = _line(o["avg"], centre, gv, ov, rmax)
                        ax.semilogy(s, np.clip(pr, 1e-6, None)
                                    / max(pr[4:].max(), 1e-9), ls,
                                    color=cols[o["k"]],
                                    label=f"{o['label']} along {gname}")
                for j, gname in enumerate(("g1", "g2")):
                    g = np.linalg.norm(_ordered(latA["B"])[:, j])
                    for m_ in (0.5, 1, 1.5, 2, 2.5, 3):
                        ax.axvline(m_ * g, color="0.85", lw=0.6, zorder=0)
            ax.set_xlabel("|q| along the lattice direction (px)")
            ax.set_title("along g1 (solid) / g2 (dashed), averaged over all "
                         "lattice rows; grey = multiples of ½g", fontsize=9)
            ax.legend(fontsize=7)
            ax = axs[1, 2]
            txt = self._dm_table(out, inv_a)
            for o in (A, Bd):
                if o["lat"] is not None:
                    ax.hist(o["lat"]["g1"], bins=40, alpha=0.5,
                            color=cols[o["k"]], label=f"{o['label']} |g1|")
                    ax.hist(o["lat"]["g2"], bins=40, alpha=0.5,
                            color=cols[o["k"]], hatch="//",
                            label=f"{o['label']} |g2|")
            ax.set_xlabel("fitted |g| of the main lattice (px)")
            ax.set_title("per-pixel lattice", fontsize=10)
            ax.legend(fontsize=7)
            fig.suptitle(txt, fontsize=8, family="monospace", ha="left",
                         x=0.01)
        else:
            axs = fig.subplots(2, 2)
            glab, sub, grains = self._dm_grains(cat, len(names), st, mn)
            show = np.where(np.isin(glab, [int(o["label"].split("#")[1])
                                           for o in out]), glab, 0)
            from gui_app import display_prefs
            pal = display_prefs.class_palette(max(show.max(), 1) + 1)
            axs[0, 0].imshow(np.ma.masked_equal(show, 0), cmap=pal,
                             interpolation="nearest")
            for o in out:
                g = int(o["label"].split("#")[1])
                ys, xs = np.nonzero(glab == g)
                axs[0, 0].text(xs.mean(), ys.mean(), str(g), fontsize=7,
                               ha="center", va="center", color="k")
            axs[0, 0].set_title(f"grains (≥ {mn} sampled px)", fontsize=10)
            axs[0, 0].set_xticks([]); axs[0, 0].set_yticks([])
            ax = axs[0, 1]
            for o in out:
                if o["lat"] is None:
                    continue
                m1, m2 = o["lat"]["g1"].mean(), o["lat"]["g2"].mean()
                e1 = o["lat"]["g1"].std() / np.sqrt(len(o["lat"]["g1"]))
                e2 = o["lat"]["g2"].std() / np.sqrt(len(o["lat"]["g2"]))
                ax.errorbar(m1, m2, xerr=e1, yerr=e2, fmt="o",
                            color=cols[o["k"]], ms=4)
                ax.annotate(o["label"].split("#")[1], (m1, m2), fontsize=7)
            ax.set_xlabel("|g1| (px)"); ax.set_ylabel("|g2| (px)")
            ax.set_title("main lattice per grain (mean ± s.e.m.)",
                         fontsize=10)
            ax = axs[1, 0]
            for o in out:
                ax.semilogy(np.arange(len(o["radial"])),
                            np.clip(o["radial"], 1e-6, None) /
                            max(o["radial"][5:].max(), 1e-9),
                            color=cols[o["k"]], lw=0.8, alpha=0.7)
            ax.set_xlabel("r (px)"); ax.set_title(
                "radial profiles per grain (blue = A, orange = B)",
                fontsize=10)
            ax = axs[1, 1]; ax.axis("off")
            ax.text(0, 1, self._dm_table(out, inv_a), va="top",
                    family="monospace", fontsize=7)
        cv.draw_idle()
        self._pd_set_status(f"domain comparison saved -> {d}")

    def _dm_table(self, out, inv_a):
        rows = [f"{'group':<18}{'n':>6}{'|g1| px':>10}{'±':>7}{'|g2| px':>10}"
                f"{'±':>7}{'area px²':>10}" + ("   d1, d2 (Å)" if
                                                np.isfinite(inv_a) else "")]
        for o in out:
            L = o["lat"]
            if L is None:
                rows.append(f"{o['label']:<18}{o['n']:>6}   (no lattice fits)")
                continue
            n = len(L["g1"])
            s1, s2 = L["g1"].std() / np.sqrt(n), L["g2"].std() / np.sqrt(n)
            line = (f"{o['label'][:17]:<18}{o['n']:>6}{L['g1'].mean():>10.3f}"
                    f"{s1:>7.3f}{L['g2'].mean():>10.3f}{s2:>7.3f}"
                    f"{L['area'].mean():>10.1f}")
            if np.isfinite(inv_a) and inv_a > 0:
                line += (f"   {1 / (L['g1'].mean() * inv_a):.4f}, "
                         f"{1 / (L['g2'].mean() * inv_a):.4f}")
            rows.append(line)
        return "\n".join(rows)

    def _dm_save(self, d, mode, out, inv_a):
        import csv
        with open(os.path.join(d, f"{mode}_lattice.csv"), "w", newline="") \
                as fh:
            w = csv.writer(fh)
            w.writerow(["group", "n_patterns", "g1_px_mean", "g1_sem",
                        "g2_px_mean", "g2_sem", "cell_area_px2",
                        "inv_ang_per_px"])
            for o in out:
                L = o["lat"]
                if L is None:
                    w.writerow([o["label"], o["n"]]); continue
                n = len(L["g1"])
                w.writerow([o["label"], o["n"], L["g1"].mean(),
                            L["g1"].std() / np.sqrt(n), L["g2"].mean(),
                            L["g2"].std() / np.sqrt(n), L["area"].mean(),
                            inv_a])
        np.savez_compressed(
            os.path.join(d, f"{mode}_averages.npz"),
            labels=np.array([o["label"] for o in out]),
            averages=np.stack([o["avg"] for o in out]),
            radial=np.stack([o["radial"] for o in out]))


    # ------------------------------------------------------------------
    # statistics of the masks
    # ------------------------------------------------------------------
    def _dm_stats(self, res):
        """Per phase: total area, share of the mapped area, number of grains,
        grain size (mean +- std, median, range), interface length; per grain:
        area, equivalent diameter, elongation, orientation of its long axis,
        in-plane lattice rotation (mean, spread) and confidence."""
        from scipy.ndimage import label
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from data import output_dir
        import csv
        cat, names = self._dm_state(res)
        K = len(names); st = res["stride"]
        try:
            step = float(self.app.real_res.get())
        except Exception:
            step = 1.0
        unit = "nm" if step != 1.0 else "px"
        a_px = (st * step) ** 2                 # area of one sampled pixel
        try:
            mn = max(int(self._dm_min.get()), 1)
        except Exception:
            mn = 5
        sub = cat[::st, ::st]
        mapped = sub != -3
        theta = res["dr"]["theta_k"][:, ::st, ::st]
        marg = np.clip(res["margin"][::st, ::st], 0, 20)
        rows_p, rows_g = [], []
        for k, nm in enumerate(names):
            m = sub == k
            lab, n = label(m, structure=np.ones((3, 3)))
            areas = np.bincount(lab.ravel())[1:] * a_px
            big = areas >= mn * a_px
            ga = areas[big]
            eqd = 2 * np.sqrt(ga / np.pi)
            # boundary of this phase with each other phase (4-neighbours)
            inter = {}
            for j in range(K):
                if j == k:
                    continue
                c = (((sub[:, :-1] == k) & (sub[:, 1:] == j)).sum()
                     + ((sub[:, 1:] == k) & (sub[:, :-1] == j)).sum()
                     + ((sub[:-1, :] == k) & (sub[1:, :] == j)).sum()
                     + ((sub[1:, :] == k) & (sub[:-1, :] == j)).sum())
                inter[names[j]] = c * st * step
            rows_p.append(dict(
                phase=nm, area=m.sum() * a_px,
                share=m.sum() / max(mapped.sum(), 1),
                n_grains=int(big.sum()), n_small=int((~big).sum()),
                mean=ga.mean() if len(ga) else np.nan,
                std=ga.std() if len(ga) > 1 else np.nan,
                median=np.median(ga) if len(ga) else np.nan,
                min=ga.min() if len(ga) else np.nan,
                max=ga.max() if len(ga) else np.nan,
                eqd=eqd.mean() if len(eqd) else np.nan,
                eqd_std=eqd.std() if len(eqd) > 1 else np.nan,
                inter=inter))
            for g in range(1, n + 1):
                gm = lab == g
                ar = gm.sum() * a_px
                if ar < mn * a_px:
                    continue
                ys, xs = np.nonzero(gm)
                if len(ys) > 2:
                    cvm = np.cov(np.vstack([xs, ys]).astype(float))
                    ev, evec = np.linalg.eigh(cvm)
                    elong = float(np.sqrt(max(ev[1], 1e-9)
                                          / max(ev[0], 1e-9)))
                    ang = float(np.degrees(np.arctan2(-evec[1, 1],
                                                      evec[0, 1])) % 180)
                else:
                    elong, ang = 1.0, np.nan
                th = theta[k][gm]
                th = th[np.isfinite(th)]
                rows_g.append(dict(phase=nm, area=ar,
                                   eqd=2 * np.sqrt(ar / np.pi),
                                   elong=elong, axis_deg=ang,
                                   rot=th.mean() if len(th) else np.nan,
                                   rot_std=th.std() if len(th) > 1 else np.nan,
                                   conf=float(np.nanmean(marg[gm])),
                                   cy=float(ys.mean() * st),
                                   cx=float(xs.mean() * st)))
        n_unc = int((sub == -1).sum()); n_off = int((sub == -2).sum())
        d = output_dir(self._active_sample(), "acom", "phases", "stats")
        u2 = f"{unit}^2"
        with open(os.path.join(d, "phase_stats.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["phase", f"area_{u2}", "share_of_mapped", "n_grains",
                        "n_smaller_than_min", f"grain_area_mean_{u2}",
                        f"grain_area_std_{u2}", f"grain_area_median_{u2}",
                        f"grain_area_min_{u2}", f"grain_area_max_{u2}",
                        f"equiv_diameter_mean_{unit}",
                        f"equiv_diameter_std_{unit}"]
                       + [f"interface_with_{n}_{unit}" for n in names])
            for r in rows_p:
                w.writerow([r["phase"], r["area"], r["share"], r["n_grains"],
                            r["n_small"], r["mean"], r["std"], r["median"],
                            r["min"], r["max"], r["eqd"], r["eqd_std"]]
                           + [r["inter"].get(n, "") for n in names])
        with open(os.path.join(d, "grain_stats.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["phase", f"area_{u2}", f"equiv_diameter_{unit}",
                        "elongation", "long_axis_deg", "rotation_mean_deg",
                        "rotation_std_deg", "confidence_mean",
                        "centre_y_px", "centre_x_px"])
            for g in rows_g:
                w.writerow([g["phase"], g["area"], g["eqd"], g["elong"],
                            g["axis_deg"], g["rot"], g["rot_std"], g["conf"],
                            g["cy"], g["cx"]])

        lines = [f"pixel = {st} x {step:g} {unit} (stride x step);  grains: "
                 f">= {mn} sampled px, 8-connected;  uncertain "
                 f"{n_unc * a_px:.4g} {u2}, off-axis {n_off * a_px:.4g} {u2}",
                 "",
                 f"{'phase':<14}{'area':>11}{'share':>7}{'grains':>7}"
                 f"{'mean':>10}{'+- std':>10}{'median':>10}{'eq.diam':>9}"
                 f"{'+- std':>8}  ({u2}, {unit})"]
        for r in rows_p:
            lines.append(f"{r['phase'][:13]:<14}{r['area']:>11.4g}"
                         f"{100 * r['share']:>6.1f}%{r['n_grains']:>7d}"
                         f"{r['mean']:>10.4g}{r['std']:>10.4g}"
                         f"{r['median']:>10.4g}{r['eqd']:>9.4g}"
                         f"{r['eqd_std']:>8.3g}")
        lines.append("")
        for r in rows_p:
            for j, L in r["inter"].items():
                if names.index(j) > names.index(r["phase"]):
                    lines.append(f"interface {r['phase']} / {j}: "
                                 f"{L:.4g} {unit}")
        win = tk.Toplevel(self)
        win.title("Phase statistics")
        win.geometry("1100x760")
        bar = tk.Frame(win); bar.pack(side="top", fill="x")
        fig = Figure(figsize=(11, 7.5), dpi=95, layout="constrained")
        cvs = FigureCanvasTkAgg(fig, master=win)
        cvs.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cvs, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        tk.Label(bar, fg="#555", text=f"saved -> {d}").pack(side="left",
                                                           padx=10)
        gs = fig.add_gridspec(2, 2, height_ratios=[0.45, 1])
        axt = fig.add_subplot(gs[0, :]); axt.axis("off")
        axt.text(0, 1, "\n".join(lines), va="top", family="monospace",
                 fontsize=8.5)
        cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        ax = fig.add_subplot(gs[1, 0])
        for k, nm in enumerate(names):
            v = [g["eqd"] for g in rows_g if g["phase"] == nm]
            if v:
                ax.hist(v, bins=20, alpha=0.6, color=cols[k % 5], label=nm)
        ax.set_xlabel(f"grain equivalent diameter ({unit})")
        ax.set_ylabel("grains"); ax.legend(fontsize=8)
        ax.set_title("grain size distribution", fontsize=10)
        ax = fig.add_subplot(gs[1, 1])
        for k, nm in enumerate(names):
            gg = [g for g in rows_g if g["phase"] == nm]
            if gg:
                ax.scatter([g["eqd"] for g in gg], [g["elong"] for g in gg],
                           s=18, color=cols[k % 5], label=nm)
        ax.set_xlabel(f"equivalent diameter ({unit})")
        ax.set_ylabel("elongation (long / short axis)")
        ax.set_title("grain shape", fontsize=10); ax.legend(fontsize=8)
        cvs.draw_idle()
        self._pd_set_status(f"statistics -> {d}")
        return rows_p, rows_g
