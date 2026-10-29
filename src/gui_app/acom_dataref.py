"""acom_dataref.py -- the ACOM window's "Data reference" mode.

Instead of simulating spots from a CIF, a MEASURED pattern is the
reference: whatever is loaded as the step-1 source (scan position, grain,
ROI box or class average) with the spots step 2 detects on it.  Every
pattern of the scan is then compared with every reference
(``dataref_core``): best in-plane rotation, match score, and -- with
several references -- which one fits best (a phase / zone-axis map).

Mixed into ACOMTabPanel; uses its source (``_test_pattern`` /
``_test_peaks``), detection settings, axes and STOP button.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from data import SAMPLES

_HINT_CLR = ("#666", "#aaa")


def _hint(parent, text):
    ctk.CTkLabel(parent, text=text, font=("Segoe UI", 9),
                 text_color=_HINT_CLR, wraplength=280, justify="left",
                 anchor="w").pack(anchor="w", padx=10, pady=(0, 2))


def _palette(k):
    import matplotlib.pyplot as plt
    cols = plt.get_cmap("tab10").colors
    return tuple(cols[k % len(cols)])


class DataRefMixin:

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _dr_build(self, parent):
        self._dr_refs = []
        _hint(parent,
            "A MEASURED pattern is the reference instead of a CIF: its "
            "detected spots are the template.  Load a source in step 1 "
            "(scan position, grain, ROI box or class average), check its "
            "spots in step 2, then add it here.  Several references -> a "
            "phase / zone-axis map.")
        r = ctk.CTkFrame(parent, fg_color="transparent")
        r.pack(fill="x", padx=10, pady=(4, 2))
        self._dr_name = ctk.StringVar(value="ref 1")
        ctk.CTkEntry(r, textvariable=self._dr_name, width=96).pack(
            side="left")
        ctk.CTkButton(r, text="+ Add current source", width=170,
                      command=self._dr_add_reference).pack(side="left",
                                                           padx=(6, 0))
        ctk.CTkButton(parent, text="+ CIF @ zone axis…  (simulated)",
                      width=240, command=self._dr_add_cif_reference
                      ).pack(anchor="w", padx=10, pady=(2, 2))
        self._dr_list = tk.Listbox(parent, height=4, exportselection=False,
                                   font=("Consolas", 9),
                                   activestyle="none")
        self._dr_list.pack(fill="x", padx=10, pady=2)
        self._dr_list.bind("<<ListboxSelect>>",
                           lambda _e: self._dr_show_selected())
        r = ctk.CTkFrame(parent, fg_color="transparent")
        r.pack(fill="x", padx=10, pady=(0, 2))
        for text, cmd, w in (("− remove", self._dr_remove, 74),
                             ("save…", self._dr_save_refs, 60),
                             ("load…", self._dr_load_refs, 60)):
            ctk.CTkButton(r, text=text, width=w, height=26,
                          command=cmd).pack(side="left", padx=(0, 4))

        g = ctk.CTkFrame(parent, fg_color="transparent")
        g.pack(fill="x", padx=10, pady=(6, 2))
        self._dr_tol = ctk.StringVar(value="3")
        self._dr_rmin = ctk.StringVar(value="auto")
        self._dr_rmax = ctk.StringVar(value="auto")
        self._dr_thr = ctk.StringVar(value="0.3")
        for i, (lab, var, tip) in enumerate((
                ("tolerance px", self._dr_tol,
                 "Two spots match when closer than this (detector px)."),
                ("ring r px", None, ""),
                ("score ≥", self._dr_thr,
                 "Positions whose best score is below this are left "
                 "unassigned (grey) in the maps."))):
            ctk.CTkLabel(g, text=lab, anchor="w").grid(row=i, column=0,
                                                       sticky="w", pady=1)
            if var is not None:
                ctk.CTkEntry(g, textvariable=var, width=60).grid(
                    row=i, column=1, sticky="w", padx=4)
        rr = ctk.CTkFrame(g, fg_color="transparent")
        rr.grid(row=1, column=1, columnspan=2, sticky="w", padx=4)
        ctk.CTkEntry(rr, textvariable=self._dr_rmin, width=52).pack(
            side="left")
        ctk.CTkLabel(rr, text="–").pack(side="left", padx=2)
        ctk.CTkEntry(rr, textvariable=self._dr_rmax, width=52).pack(
            side="left")
        _hint(parent,
            "Only spots between r_min and r_max (px from each pattern's "
            "own beam) are compared.  auto: r_min just outside the direct "
            "beam, r_max = half the detector.")

        ctk.CTkButton(parent, text="Match current source ▶", width=240,
                      command=self._dr_match_single).pack(anchor="w",
                                                          padx=10, pady=2)
        sr = ctk.CTkFrame(parent, fg_color="transparent")
        sr.pack(fill="x", padx=10, pady=(6, 0))
        ctk.CTkLabel(sr, text="stride:").pack(side="left")
        self._dr_stride = ctk.IntVar(value=2)
        ctk.CTkEntry(sr, textvariable=self._dr_stride, width=40).pack(
            side="left", padx=4)
        ctk.CTkButton(sr, text="Map ▶", width=150,
                      fg_color=("#2D7A2D", "#1F6A1F"),
                      command=self._dr_run_full).pack(side="left",
                                                      padx=(8, 0))
        ctk.CTkButton(sr, text="■", width=30, fg_color=("#a33", "#722"),
                      hover_color=("#c33", "#822"),
                      command=self._stop_batch_run).pack(side="left",
                                                         padx=(4, 0))
        ctk.CTkButton(parent, text="Auto phases ▶  (refine + re-map)",
                      fg_color=("#7A2D7A", "#6A1F6A"),
                      hover_color=("#8A3D8A", "#7A2F7A"),
                      command=self._ap_run).pack(fill="x", padx=6,
                                                 pady=(6, 0))
        _hint(parent,
            "Auto phases: from two references (with / without the extra "
            "reflections, not necessarily pure) - maps, finds the "
            "satellite-free domain, refines both references from the data "
            "and maps again; one summary window, everything saved.")
        _hint(parent,
            "Peaks of every pattern are detected once (step-2 settings) "
            "and cached; re-mapping with other references only re-matches."
            "  ■ = stop.")
        self._dr_status = ctk.CTkLabel(parent, text="",
            font=("Consolas", 9), text_color=_HINT_CLR, wraplength=290,
            justify="left", anchor="w")
        self._dr_status.pack(anchor="w", padx=10, pady=(2, 6))

    def _dr_imshow(self, ax, pat):
        """Pattern with the ACOM view-bar stretch (log / power / linear
        + percentiles)."""
        img, lo, hi = self._stretch_img(pat)
        ax.imshow(img, cmap="gray", origin="upper", vmin=lo, vmax=hi)

    def _dr_set_status(self, msg):
        try:
            self._dr_status.configure(text=msg)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # references
    # ------------------------------------------------------------------
    def _dr_ring(self, pattern):
        """(r_min, r_max) px: typed values, or auto from the pattern."""
        H, W = np.asarray(pattern).shape

        def _v(s, auto):
            s = str(s).strip().lower()
            return float(auto) if s in ("", "auto") else float(s)
        auto_lo = 6.0
        try:
            from celldenoise_flatten import auto_params
            auto_lo = float(auto_params(pattern)["r_min"])
        except Exception:
            pass
        return _v(self._dr_rmin.get(), auto_lo), _v(self._dr_rmax.get(),
                                                    min(H, W) / 2.0)

    def _dr_tol_val(self):
        try:
            return max(float(self._dr_tol.get()), 0.5)
        except Exception:
            return 3.0

    def _dr_add_reference(self):
        from gui_app.dataref_core import make_reference
        pat = getattr(self, "_test_pattern", None)
        pk = getattr(self, "_test_peaks", None)
        if pat is None or pk is None:
            messagebox.showinfo("Data reference",
                "Load a source (step 1) and detect its peaks (step 2) "
                "first.")
            return
        name = self._dr_name.get().strip() or f"ref {len(self._dr_refs)+1}"
        if any(r.name == name for r in self._dr_refs):
            messagebox.showerror("Data reference",
                                 f"'{name}' already exists.")
            return
        lo, hi = self._dr_ring(pat)
        ref = make_reference(name, pat, pk, r_min=lo, r_max=hi,
                             tol=self._dr_tol_val(),
                             guess=self._dr_guess(pat),
                             origin=getattr(self, "_test_origin", ""),
                             colour=_palette(len(self._dr_refs)))
        if len(ref.vecs) < 3:
            messagebox.showerror("Data reference",
                f"only {len(ref.vecs)} spot(s) between r = {lo:.0f} and "
                f"{hi:.0f} px - adjust detection (step 2) or the ring.")
            return
        ref.extra["r_min"], ref.extra["r_max"] = lo, hi
        mem = getattr(self, "_members_of", {}).get(id(pat))
        if mem is not None and mem[0] is pat:
            ref.extra["members"] = mem[1]       # scan positions averaged
        self._dr_refs.append(ref)
        self._dr_name.set(f"ref {len(self._dr_refs) + 1}")
        self._dr_refresh_list()
        self._dr_list.selection_clear(0, "end")
        self._dr_list.selection_set(len(self._dr_refs) - 1)
        self._dr_show_selected()

    def _dr_add_cif_reference(self):
        """A reference SIMULATED from a built CIF phase at a zone axis
        [u v w] (kinematic spot positions, converted to detector px with
        the step-3 calibration).  Used like a measured reference; as a
        strain reference it gives ABSOLUTE strain."""
        from tkinter import simpledialog
        from gui_app._dialogs import ask_choice
        from gui_app.dataref_core import DataReference, self_symmetry
        crystals = getattr(self, "_phase_crystals", {}) or {}
        if not crystals:
            messagebox.showinfo("CIF reference",
                "Build a CIF first: switch the Reference to 'CIF', add it "
                "(+ CIF…) and press 'Build all'; check the 1/Å-per-px "
                "calibration there too.")
            return
        name = ask_choice(self, "CIF reference", "Phase:",
                          sorted(crystals))
        if not name:
            return
        s = simpledialog.askstring("CIF reference",
            f"Zone axis [u v w] of '{name}' (lattice indices):",
            initialvalue="0 0 1", parent=self)
        if not s:
            return
        try:
            uvw = np.array([float(t) for t in
                            s.replace(",", " ").replace("[", " ")
                            .replace("]", " ").split()], float)
            assert uvw.shape == (3,) and np.any(uvw)
        except Exception:
            messagebox.showerror("CIF reference",
                                 "Type three numbers, e.g. 0 0 1.")
            return
        inv_a = float(self._inv_ang.get())
        pat = getattr(self, "_test_pattern", None)
        H, W = (np.asarray(pat).shape if pat is not None else (256, 256))
        r_max_px = min(H, W) / 2.0
        try:
            pl = crystals[name].generate_diffraction_pattern(
                zone_axis_lattice=uvw, sigma_excitation_error=0.02,
                k_max=r_max_px * inv_a)
            qx = np.asarray(pl.data["qx"], float)
            qy = np.asarray(pl.data["qy"], float)
            I = np.asarray(pl.data["intensity"], float)
        except Exception as e:
            messagebox.showerror("CIF reference",
                                 f"simulation failed: {e!r}")
            return
        q = np.hypot(qx, qy)
        keep = q > 1e-6
        qx, qy, I = qx[keep], qy[keep], I[keep]
        keep = I >= 0.01 * I.max() if I.size else keep[:0]
        vecs = np.column_stack([qx[keep], qy[keep]]) / inv_a   # (dy, dx) px
        if len(vecs) < 3:
            messagebox.showerror("CIF reference",
                f"only {len(vecs)} reflection(s) at [{s}] inside the "
                f"detector - check the zone axis / calibration.")
            return
        # a picture of the simulated spots, for the reference views
        img = np.zeros((H, W), np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = H / 2.0, W / 2.0
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 8.0) * 2
        for (dy, dx), w in zip(vecs, I[keep] / I[keep].max()):
            img += np.sqrt(w) * np.exp(-((yy - cy - dy) ** 2
                                         + (xx - cx - dx) ** 2) / 4.0)
        lo = 0.5 * float(np.hypot(vecs[:, 0], vecs[:, 1]).min())
        if pat is not None:
            lo = min(lo, self._dr_ring(pat)[0])
        uvw_s = " ".join(f"{v:g}" for v in uvw)
        ref = DataReference(
            name=f"{name} [{uvw_s}]", vecs=vecs,
            period_deg=self_symmetry(vecs, self._dr_tol_val()),
            origin=f"CIF {name} @ [{uvw_s}], {inv_a:g} 1/Å/px",
            pattern=img, centre=(cy, cx),
            colour=_palette(len(self._dr_refs)),
            extra=dict(simulated=True, inv_ang=inv_a, r_min=lo,
                       r_max=r_max_px))
        if any(r.name == ref.name for r in self._dr_refs):
            messagebox.showerror("CIF reference",
                                 f"'{ref.name}' already exists.")
            return
        self._dr_refs.append(ref)
        self._dr_refresh_list()
        self._dr_list.selection_clear(0, "end")
        self._dr_list.selection_set(len(self._dr_refs) - 1)
        self._dr_show_selected()

    def _dr_guess(self, pattern):
        """Expected beam position: the step-1 source's centre if it is
        the beam, else the detector centre."""
        H, W = np.asarray(pattern).shape
        g = getattr(self, "_dr_beam_guess", None)
        return g if g is not None else (H / 2.0, W / 2.0)

    def _dr_refresh_list(self):
        self._dr_list.delete(0, "end")
        for k, r in enumerate(self._dr_refs):
            fold = f"{int(round(360 / r.period_deg))}-fold"
            self._dr_list.insert("end",
                f"■ {r.name}: {len(r.vecs)} spots, {fold}  [{r.origin}]")
            c = "#%02x%02x%02x" % tuple(int(255 * v) for v in r.colour[:3])
            self._dr_list.itemconfig(k, fg=c)

    def _dr_remove(self):
        sel = self._dr_list.curselection()
        if not sel:
            return
        del self._dr_refs[int(sel[0])]
        for k, r in enumerate(self._dr_refs):
            r.colour = _palette(k)
        self._dr_refresh_list()

    def _dr_show_selected(self):
        sel = self._dr_list.curselection()
        if not sel or getattr(self, "_ax_pat", None) is None:
            return
        ref = self._dr_refs[int(sel[0])]
        self._ensure_quad_layout()
        ax = self._ax_pat
        ax.clear()
        self._dr_imshow(ax, ref.pattern)
        cy, cx = ref.centre
        ax.scatter(cx + ref.vecs[:, 1], cy + ref.vecs[:, 0], s=70,
                   facecolors="none", edgecolors=[ref.colour],
                   linewidths=1.5)
        ax.plot(cx, cy, "+", color="w", ms=10)
        ax.set_title(f"reference '{ref.name}': {len(ref.vecs)} spots "
                     f"(r {ref.extra.get('r_min', 0):.0f}–"
                     f"{ref.extra.get('r_max', 0):.0f} px)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        self._redraw_all()

    def _dr_save_refs(self):
        if not self._dr_refs:
            return
        from data import output_dir
        d = output_dir(self._active_sample(), "acom", "dataref") \
            if self._active_sample() else os.getcwd()
        p = filedialog.asksaveasfilename(
            initialdir=d, defaultextension=".npz",
            initialfile="data_references.npz",
            filetypes=[("references", "*.npz")])
        if not p:
            return
        np.savez_compressed(p, refs=np.array([dict(
            name=r.name, vecs=r.vecs, period_deg=r.period_deg,
            origin=r.origin, pattern=r.pattern, centre=r.centre,
            extra=r.extra) for r in self._dr_refs], dtype=object))
        self._dr_set_status(f"saved {len(self._dr_refs)} reference(s) -> "
                            f"{p}")

    def _dr_load_refs(self):
        from gui_app.dataref_core import DataReference
        p = filedialog.askopenfilename(filetypes=[("references", "*.npz")])
        if not p:
            return
        try:
            items = np.load(p, allow_pickle=True)["refs"]
            for it in items:
                it = dict(it)
                if any(r.name == it["name"] for r in self._dr_refs):
                    continue
                self._dr_refs.append(DataReference(
                    name=it["name"], vecs=np.asarray(it["vecs"]),
                    period_deg=min(float(it["period_deg"]), 180.0),
                    origin=it.get("origin", ""), pattern=it.get("pattern"),
                    centre=tuple(it.get("centre", (0, 0))),
                    colour=_palette(len(self._dr_refs)),
                    extra=dict(it.get("extra", {}))))
        except Exception as e:
            messagebox.showerror("load references", repr(e))
            return
        self._dr_refresh_list()

    # ------------------------------------------------------------------
    # single pattern
    # ------------------------------------------------------------------
    def _dr_match_one(self, pattern, peaks):
        """Match one pattern against all references -> (centre, vecs,
        [result per ref])."""
        from gui_app.dataref_core import beam_centre, match_vectors, \
            spot_vectors
        c = beam_centre(pattern, self._dr_guess(pattern))
        out = []
        tol = self._dr_tol_val()
        for ref in self._dr_refs:
            lo = ref.extra.get("r_min", 0.0); hi = ref.extra.get(
                "r_max", np.inf)
            v = spot_vectors(peaks, c, lo, hi)
            out.append((v, match_vectors(ref.vecs, v, tol,
                                         ref.period_deg)))
        return c, out

    def _dr_match_single(self):
        if not self._dr_refs:
            messagebox.showinfo("Data reference",
                                "Add at least one reference first.")
            return
        pat = getattr(self, "_test_pattern", None)
        pk = getattr(self, "_test_peaks", None)
        if pat is None or pk is None:
            messagebox.showinfo("Data reference",
                "Load a source (step 1) and detect its peaks (step 2).")
            return
        c, res = self._dr_match_one(pat, pk)
        self._ensure_quad_layout()
        self._dr_draw_match(self._ax_fit, pat, c, res,
                            getattr(self, "_test_origin", ""))
        self._redraw_all()
        k = int(np.argmax([r["score"] for _, r in res]))
        self._dr_set_status(
            "  ".join(f"{ref.name}: {r['score']:.2f}"
                      for ref, (_, r) in zip(self._dr_refs, res))
            + f"   -> best '{self._dr_refs[k].name}'")

    def _dr_draw_match(self, ax, pat, c, res, title=""):
        """Pattern + its spots (cyan) + the best reference rotated onto
        it (reference colour; filled = matched)."""
        from gui_app.dataref_core import rotate_ccw
        ax.clear()
        self._dr_imshow(ax, pat)
        k = int(np.argmax([r["score"] for _, r in res]))
        v, r = res[k]
        ref = self._dr_refs[k]
        cy, cx = c
        ax.scatter(cx + v[:, 1], cy + v[:, 0], s=60, facecolors="none",
                   edgecolors="c", linewidths=1.0, label="detected")
        if np.isfinite(r["theta_deg"]):
            rv = rotate_ccw(ref.vecs, r["theta_deg"])
            hit = np.zeros(len(rv), bool); hit[r["ia"]] = True
            ax.scatter(cx + rv[~hit, 1], cy + rv[~hit, 0], s=26,
                       marker="s", facecolors="none",
                       edgecolors=[ref.colour], linewidths=1.2,
                       label=f"'{ref.name}' unmatched")
            ax.scatter(cx + rv[hit, 1], cy + rv[hit, 0], s=22, marker="s",
                       color=[ref.colour], label=f"'{ref.name}' matched")
        ax.plot(cx, cy, "+", color="w", ms=10)
        lines = [f"{rf.name}: {rr['score']:.2f}"
                 + (f" θ={rr['theta_deg']:.1f}°"
                    if np.isfinite(rr["theta_deg"]) else "")
                 for rf, (_, rr) in zip(self._dr_refs, res)]
        ax.set_title((title + "\n" if title else "")
                     + "   ".join(lines), fontsize=9)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])

    # ------------------------------------------------------------------
    # whole scan
    # ------------------------------------------------------------------
    def _dr_run_full(self):
        # two or more references: the phase discrimination map (it also
        # carries the match score and rotation)
        if len(getattr(self, "_dr_refs", [])) >= 2 and hasattr(
                self, "_pd_run_full"):
            return self._pd_run_full()
        if not self._dr_refs:
            messagebox.showinfo("Data reference",
                                "Add at least one reference first.")
            return
        sample = self._need_dataset()
        if sample is None:
            return
        if getattr(self, "_dr_busy", False):
            return
        self._dr_busy = True
        self._stop_event.clear()
        args = dict(stride=max(int(self._dr_stride.get() or 1), 1),
                    detect_kw=self._detect_kw_now(),
                    tol=self._dr_tol_val(), refs=list(self._dr_refs),
                    sample=sample)
        threading.Thread(target=self._dr_worker, kwargs=args,
                         daemon=True).start()

    def _dr_post(self, fn, *a):
        try:
            self.after(0, lambda: fn(*a))
        except Exception:
            pass

    def _dr_detect_all(self, cube, stride, detect_kw):
        """Peaks + own beam centre of every sampled pattern; cached next
        to the CIF-ACOM peak cache (same key), with the centres added."""
        from gui_app.acom_core import detect_peaks_2d
        from gui_app.dataref_core import beam_centre
        Ny, Nx, H, W = cube.shape
        cache = self._peak_cache_path(stride, detect_kw)
        peaks = cents = None
        if cache and os.path.exists(cache):
            try:
                d = np.load(cache, allow_pickle=True)
                peaks = [np.asarray(a, float) for a in d["peaks"]]
                if "centres" in d.files:
                    cents = [tuple(c) for c in d["centres"]]
                    self._dr_post(self._dr_set_status,
                                  "peaks + centres from cache")
            except Exception:
                peaks = cents = None
        # expected beam position = beam of the mean of ~200 patterns
        idx = np.linspace(0, Ny * Nx - 1, min(200, Ny * Nx)).astype(int)
        acc = np.zeros((H, W), np.float64)
        for i in idx:
            acc += np.asarray(cube[i // Nx, i % Nx], np.float64)
        guess = beam_centre(acc / len(idx))
        self._dr_beam_guess = guess
        need_pk, need_c = peaks is None, cents is None
        if need_pk:
            peaks = [np.zeros((0, 3))] * (Ny * Nx)
        if need_c:
            cents = [guess] * (Ny * Nx)
        if need_pk or need_c:
            t0 = time.time(); n_s = len(range(0, Ny, stride)) * len(
                range(0, Nx, stride)); done = 0
            for y in range(0, Ny, stride):
                if self._stop_event.is_set():
                    raise RuntimeError("stopped by user")
                for x in range(0, Nx, stride):
                    f = np.asarray(cube[y, x], np.float32)
                    if need_pk:
                        peaks[y * Nx + x] = detect_peaks_2d(f, **detect_kw)
                    if need_c:
                        cents[y * Nx + x] = beam_centre(f, guess)
                    done += 1
                el = time.time() - t0
                self._dr_post(self._dr_set_status,
                    f"{'detecting' if need_pk else 'beam centres'}: "
                    f"{done}/{n_s}  ({el:.0f} s, ~"
                    f"{el / max(done, 1) * (n_s - done):.0f} s left)")
            if cache:
                try:
                    os.makedirs(os.path.dirname(cache), exist_ok=True)
                    np.savez_compressed(cache,
                        peaks=np.array(peaks, dtype=object),
                        centres=np.array(cents, dtype=float))
                except Exception as e:
                    print(f"[dataref] cache save: {e!r}", flush=True)
        return peaks, cents

    def _dr_worker(self, stride, detect_kw, tol, refs, sample):
        from gui_app.dataref_core import match_dataset
        from gui_app.posthoc_panel import _open_lazy
        try:
            cfg = SAMPLES[sample]
            Ny, Nx = cfg["scan_shape"]
            cube = _open_lazy(cfg["path"], scan_shape=(Ny, Nx))
            t0 = time.time()
            peaks, cents = self._dr_detect_all(cube, stride, detect_kw)
            samp = np.zeros((Ny, Nx), bool)
            samp[::stride, ::stride] = True
            lo = min(r.extra.get("r_min", 0.0) for r in refs)
            hi = max(r.extra.get("r_max", np.inf) for r in refs)

            def _prog(n, N):
                self._dr_post(self._dr_set_status,
                              f"matching {n}/{N} ...")
            res = match_dataset(peaks, cents, refs, (Ny, Nx), tol=tol,
                                r_min=lo, r_max=hi, sampled=samp.ravel(),
                                progress=_prog,
                                stop=self._stop_event.is_set)
            res.update(stride=stride, scan_shape=(Ny, Nx), refs=refs,
                       cube=cube, peaks=peaks, centres=cents,
                       elapsed=time.time() - t0, tol=tol)
            self._dr_post(self._dr_show_full, res)
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = repr(e)
            self._dr_post(self._dr_set_status, f"map failed: {msg}")
        finally:
            self._dr_busy = False

    def _dr_thr_val(self):
        try:
            return float(self._dr_thr.get())
        except Exception:
            return 0.3

    def _dr_show_full(self, res):
        """Result window: reference map, score, in-plane rotation and
        spot count; left-click = inspect, right-drag = new reference."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        from gui_app.acom_tab import _fill_stride
        self._drfull = res
        Ny, Nx = res["scan_shape"]; st = res["stride"]
        refs = res["refs"]; K = len(refs)
        thr = self._dr_thr_val()
        maps = {k: np.array(res[k], dtype=float) for k in
                ("best", "score", "theta", "n_spots", "margin")}
        maps["theta_frac"] = maps["theta"] / np.asarray(res["periods"])[
            np.clip(res["best"], 0, K - 1)]
        _fill_stride(maps, Ny, Nx, st)
        ok = maps["score"] >= thr
        best = np.where(ok, maps["best"], -1)
        # two different rotations fit (almost) equally well: no angle
        amb = ok & (maps["margin"] < 0.15)
        self._dr_save_outputs(res, maps, thr)

        win = tk.Toplevel(self)
        win.title(f"ACOM - data references  ({self._active_sample()})")
        win.geometry("1200x860")
        bar = tk.Frame(win); bar.pack(side="top", fill="x")
        fig = Figure(figsize=(12, 8), dpi=95, facecolor="#f4f4f4",
                     layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        tk.Label(bar, fg="#555",
                 text="left-click: pattern + match    right-drag a box: "
                      "make it a new reference").pack(side="left", padx=10)
        wide = Nx > 1.6 * Ny
        axs = fig.subplots(4, 1) if wide else fig.subplots(2, 2)
        axs = np.asarray(axs).ravel()
        cols = [r.colour for r in refs]
        cm = ListedColormap([(0.82, 0.82, 0.82)] + cols)
        axs[0].imshow(best + 1, cmap=cm, vmin=-0.5, vmax=K + 0.5,
                      interpolation="nearest")
        axs[0].set_title(f"best reference (score ≥ {thr:g}; grey = none)",
                         fontsize=10)
        axs[0].legend(handles=[Patch(color=c, label=r.name)
                               for c, r in zip(cols, refs)]
                      + [Patch(color=(0.82, 0.82, 0.82), label="none")],
                      fontsize=7, loc="upper right", framealpha=0.7)
        im = axs[1].imshow(maps["score"], cmap="viridis", vmin=0, vmax=1,
                           interpolation="nearest")
        axs[1].set_title("match score (F1 of spot positions)", fontsize=10)
        fig.colorbar(im, ax=axs[1], fraction=0.035)
        same_p = len(set(res["periods"])) == 1
        if same_p:
            P = float(res["periods"][0])
            im = axs[2].imshow(np.where(ok & ~amb, maps["theta"], np.nan),
                               cmap="twilight", vmin=-P / 2, vmax=P / 2,
                               interpolation="nearest")
            lab = f"in-plane rotation vs reference (deg, ±{P / 2:g}°)"
        else:
            im = axs[2].imshow(np.where(ok & ~amb, maps["theta_frac"],
                                        np.nan),
                               cmap="twilight", vmin=-0.5, vmax=0.5,
                               interpolation="nearest")
            lab = "in-plane rotation (fraction of each reference's period)"
        axs[2].set_facecolor((0.82, 0.82, 0.82))
        axs[2].imshow(np.ma.masked_where(~amb, np.ones_like(maps["score"])),
                      cmap=ListedColormap(["black"]), vmin=0, vmax=1,
                      interpolation="nearest")
        axs[2].set_title(lab + "\nblack = ambiguous (another rotation fits "
                         "almost as well)", fontsize=10)
        fig.colorbar(im, ax=axs[2], fraction=0.035)
        im = axs[3].imshow(maps["n_spots"], cmap="magma",
                           interpolation="nearest")
        axs[3].set_title("detected spots per pattern (in the ring)",
                         fontsize=10)
        fig.colorbar(im, ax=axs[3], fraction=0.035)
        for a in axs:
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{K} data reference(s), stride {st}, tolerance "
                     f"{res['tol']:g} px  ({res['elapsed']:.0f} s)",
                     fontsize=11)

        from matplotlib.widgets import RectangleSelector

        def _click(ev):
            if (ev.button != 1 or ev.inaxes not in axs or ev.xdata is None
                    or getattr(tb, "mode", "")):
                return
            x = int(np.clip(round(ev.xdata), 0, Nx - 1))
            y = int(np.clip(round(ev.ydata), 0, Ny - 1))
            self._dr_inspect(y - y % st, x - x % st)

        def _box(e0, e1):
            y0, y1 = sorted((int(round(e0.ydata)), int(round(e1.ydata))))
            x0, x1 = sorted((int(round(e0.xdata)), int(round(e1.xdata))))
            self._dr_box_reference(max(y0, 0), min(y1, Ny - 1),
                                   max(x0, 0), min(x1, Nx - 1))
        cv.mpl_connect("button_press_event", _click)
        win._selectors = [RectangleSelector(a, _box, button=[3],
                                            useblit=False, interactive=False)
                          for a in axs]
        cv.draw_idle()
        n_ok = int(ok[::st, ::st].sum()); n_all = int(ok[::st, ::st].size)
        self._dr_set_status(
            f"mapped {n_all} positions in {res['elapsed']:.0f} s; "
            f"{n_ok} ({100 * n_ok / max(n_all, 1):.0f}%) match a reference "
            f"with score ≥ {thr:g}")

    def _dr_save_outputs(self, res, maps, thr):
        try:
            from data import output_dir
            d = output_dir(self._active_sample(), "acom", "dataref")
            np.savez_compressed(
                os.path.join(d, "dataref_maps.npz"),
                best=res["best"], score=res["score"], theta=res["theta"],
                n_match=res["n_match"], n_spots=res["n_spots"],
                margin=res["margin"],
                score_k=res["score_k"], theta_k=res["theta_k"],
                periods=res["periods"], names=np.array(res["names"]),
                stride=res["stride"], tol=res["tol"], score_threshold=thr)
            np.savez_compressed(os.path.join(d, "data_references.npz"),
                refs=np.array([dict(name=r.name, vecs=r.vecs,
                    period_deg=r.period_deg, origin=r.origin,
                    pattern=r.pattern, centre=r.centre, extra=r.extra)
                    for r in res["refs"]], dtype=object))
        except Exception as e:
            print(f"[dataref] save: {e!r}", flush=True)

    def _dr_inspect(self, y, x):
        res = getattr(self, "_drfull", None)
        if res is None:
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        Ny, Nx = res["scan_shape"]
        pat = np.asarray(res["cube"][y, x], np.float32)
        pk = res["peaks"][y * Nx + x]
        c, out = self._dr_match_one(pat, pk)
        win = tk.Toplevel(self)
        win.title(f"pattern ({y}, {x})")
        win.geometry("720x760")
        bar = tk.Frame(win); bar.pack(side="bottom", fill="x")
        fig = Figure(figsize=(7, 7), dpi=95)
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(cv, bar, pack_toolbar=False)
        tb.update(); tb.pack(side="left")
        self._dr_draw_match(fig.add_subplot(111), pat, c, out,
                            f"scan ({y}, {x})  {len(pk)} spots detected")

        def _use():
            self._test_pattern = pat
            self._test_peaks = pk
            self._test_origin = f"scan ({y}, {x})"
            self._dr_name.set(f"ref {len(self._dr_refs) + 1}")
            self._dr_add_reference()
        tk.Button(bar, text="Add as reference",
                  command=_use).pack(side="right", padx=8)
        fig.tight_layout(); cv.draw_idle()

    def _dr_box_reference(self, y0, y1, x0, x1, res=None):
        """Average of a scan box (all its patterns, beam-recentred) as a
        new reference, using the step-2 detection.  res: the map result
        the box was drawn on (data-reference, phase or satellite map)."""
        res = res if res is not None else getattr(self, "_drfull", None)
        if res is None:
            return
        n = (y1 - y0 + 1) * (x1 - x0 + 1)
        from tkinter import simpledialog
        name = simpledialog.askstring(
            "New reference",
            f"Average the {n} patterns in rows {y0}-{y1}, columns "
            f"{x0}-{x1} as a new reference.\n\nName:",
            initialvalue=f"ref {len(self._dr_refs) + 1}", parent=self)
        if name is None or not name.strip():
            return
        if any(r.name == name.strip() for r in self._dr_refs):
            messagebox.showerror("New reference",
                                 f"'{name.strip()}' already exists.")
            return
        from gui_app.acom_core import detect_peaks_2d
        from celldenoise_flatten import recentre
        cube = res["cube"]; Ny, Nx = res["scan_shape"]
        cents = res["centres"]
        guess = self._dr_beam_guess
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        pos = list(zip(ys.ravel(), xs.ravel()))
        if len(pos) > 1024:
            pos = [pos[i] for i in np.random.default_rng(0).choice(
                len(pos), 1024, replace=False)]
        acc = None
        for (y, x) in pos:
            f = np.asarray(cube[y, x], np.float32)
            # put each pattern's own beam on the common position so the
            # spots do not smear when the beam drifts across the box
            f = recentre(f, guess) if cents is not None else f
            acc = f.astype(np.float64) if acc is None else acc + f
        avg = (acc / len(pos)).astype(np.float32)
        self._note_members(avg, [y * Nx + x for (y, x) in pos])
        self._test_pattern = avg
        self._test_peaks = detect_peaks_2d(avg, **self._detect_kw_now())
        self._test_origin = f"box y{y0}-{y1} x{x0}-{x1}"
        self._dr_name.set(name.strip())
        self._dr_add_reference()
        try:
            self._draw_pattern_panel()
        except Exception:
            pass
