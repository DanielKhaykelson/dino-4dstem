"""interpret_panel.py — Analysis ▸ Interpretation sub-tab.

Runs the interpretability battery (port of tools/interpret_imc_*.py) on the
run + dataset currently linked in the Post-hoc tab, and shows the result
figures + an auto-written report.  Whatever can be computed from what is on
disk is computed; ACOM-dependent pieces (orientation / phase factors,
zone-axis cross-check) are enabled only when an ACOM full-dataset run exists,
otherwise the panel says so and asks for an ACOM run first.
"""
import os
import threading
import traceback

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data import SAMPLES
from gui_app import interpret_core as ic


# figures the battery can produce (label, relative path under run)
_FIG_SPECS = [
    ("Factor ranking (probe R² / η²)", "_interpretability/test1_factor_ranking.png"),
    ("Per-class signature heatmap",    "_interpretability/test5_class_signatures.png"),
    ("Ablation class maps",            "_interpretability/test2_4_ablation_maps.png"),
    ("Per-class mean patterns",        "_interpretability/class_mean_patterns.png"),
    ("Per-class radial profiles",      "_interpretability/class_radial_profiles.png"),
    ("Classical-baseline agreement",   "_interpretability/classical_baselines.png"),
    ("DINO vs classical maps",         "_interpretability/classical_vs_dino_maps.png"),
    ("Grad-CAM / IG (per prototype)",  "eval/paper_attribution/fig_paper_attribution.png"),
]


class InterpretPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self._busy = False
        self._factors = None
        self._build()
        self.after(200, self.refresh_from_posthoc)

    # ---- UI -------------------------------------------------------------
    def _build(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # left control column
        left = ctk.CTkScrollableFrame(self, width=320)
        left.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)

        ctk.CTkLabel(left, text="Interpretation",
                     font=ctk.CTkFont(size=15, weight="bold")
                     ).pack(anchor="w", pady=(2, 2))
        ctk.CTkLabel(left, justify="left", wraplength=290, text=(
            "What physical signal does the trained model cluster on? "
            "Runs probing, causal ablations, class averages, Grad-CAM/IG "
            "on the run linked in the Post-hoc tab.")
            ).pack(anchor="w", pady=(0, 6))

        ctk.CTkButton(left, text="↻ Refresh from Post-hoc",
                      command=self.refresh_from_posthoc).pack(fill="x", pady=2)
        self._link_lbl = ctk.CTkLabel(left, justify="left", wraplength=290,
                                      text="(no run linked)",
                                      font=ctk.CTkFont(size=11))
        self._link_lbl.pack(anchor="w", pady=(2, 2))
        self._acom_lbl = ctk.CTkLabel(left, justify="left", wraplength=290,
                                     text="", font=ctk.CTkFont(size=11))
        self._acom_lbl.pack(anchor="w", pady=(0, 6))

        ctk.CTkLabel(left, text="Analyses to run:",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        self._opt_probe = ctk.BooleanVar(value=True)
        self._opt_means = ctk.BooleanVar(value=True)
        self._opt_classical = ctk.BooleanVar(value=True)
        self._opt_abl = ctk.BooleanVar(value=True)
        self._opt_gcam = ctk.BooleanVar(value=False)
        for txt, var, hint in [
            ("Probing + signatures (fast)", self._opt_probe, ""),
            ("Class averages (1 cube pass)", self._opt_means, ""),
            ("Classical-baseline comparison", self._opt_classical, ""),
            ("Causal ablations (GPU, ~minutes)", self._opt_abl, ""),
            ("Grad-CAM / IG (GPU)", self._opt_gcam, ""),
        ]:
            ctk.CTkCheckBox(left, text=txt, variable=var).pack(anchor="w",
                                                               pady=1)
        ctk.CTkLabel(left, justify="left", wraplength=290,
                     font=ctk.CTkFont(size=10), text=(
                     "Orientation/phase factors + zone-axis cross-check are "
                     "added automatically when an ACOM run is found.")
                     ).pack(anchor="w", pady=(2, 6))

        self._run_btn = ctk.CTkButton(left, text="▶ Run interpretation",
                                      command=self._run)
        self._run_btn.pack(fill="x", pady=(4, 2))
        self._prog = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=11),
                                  wraplength=290, justify="left")
        self._prog.pack(anchor="w", pady=(2, 6))
        ctk.CTkButton(left, text="📂 Open output folder",
                      command=self._open_folder).pack(fill="x", pady=2)

        # right results area
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(right, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(top, text="Figure:").pack(side="left", padx=(4, 4))
        self._fig_menu = ctk.CTkOptionMenu(top, values=["(run first)"],
                                           command=self._show_figure, width=260)
        self._fig_menu.pack(side="left", padx=2)

        self._fig = Figure(figsize=(6.5, 5.2), facecolor="white")
        self._ax = self._fig.add_subplot(111)
        self._ax.axis("off")
        self._ax.text(0.5, 0.5, "Run interpretation to see results",
                      ha="center", va="center")
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew",
                                          padx=4, pady=4)

        self._report = tk.Text(right, height=9, wrap="word")
        self._report.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 4))
        self._report.insert("1.0", "(report appears here)")
        self._report.configure(state="disabled")

    # ---- linkage --------------------------------------------------------
    def _posthoc(self):
        return getattr(self.app, "posthoc", None)

    def refresh_from_posthoc(self):
        ph = self._posthoc()
        run = getattr(ph, "outdir", None) if ph else None
        sample = getattr(ph, "sample", None) if ph else None
        if not run:
            self._link_lbl.configure(text="(no run linked — load a run in the "
                                          "Post-hoc tab)")
            self._acom_lbl.configure(text="")
            return
        cached = "  [inference cached]" if getattr(ph, "_inf", None) \
            is not None else ""
        self._link_lbl.configure(
            text=f"run: …{os.sep}{os.path.basename(run)}\nsample = "
                 f"{sample}{cached}")
        acom = ic.find_acom_arrays(run)
        if acom is None:
            self._acom_lbl.configure(
                text="ACOM: not found — run ACOM (Diffraction ▸ ACOM, "
                     "full-dataset) to enable orientation/phase analysis.",
                text_color="#c08a2a")
        else:
            extra = []
            if "winning_rmat" in acom:
                extra.append("zone-axis")
            if "n_peaks" in acom:
                extra.append("n-peaks")
            ex = (" (+" + ", ".join(extra) + ")") if extra else ""
            self._acom_lbl.configure(
                text=f"ACOM: found{ex} in {os.path.basename(acom['src'])}/",
                text_color="#3a8a3a")

    # ---- run ------------------------------------------------------------
    def _set_prog(self, msg):
        self.after(0, lambda: self._prog.configure(text=msg))

    def _run(self):
        if self._busy:
            return
        ph = self._posthoc()
        if not ph or not getattr(ph, "outdir", None):
            messagebox.showinfo("no run", "Load a run + dataset in the "
                                "Post-hoc tab first."); return
        if not ph.sample or ph.sample not in SAMPLES:
            messagebox.showinfo("no dataset", "Link a dataset in the "
                                "Post-hoc tab first."); return
        # ensure inference (may be slow first time)
        self._set_prog("preparing inference …")
        if not ph._ensure_inference():
            self._set_prog(""); return
        self._busy = True
        self._run_btn.configure(state="disabled", text="running …")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            ph = self._posthoc()
            run, sample = ph.outdir, ph.sample
            cfg = SAMPLES[sample]
            polar = ph._read_polar_pipeline_cfg()
            ctx = ic.Ctx(run, sample, cfg["path"], cfg.get("vmax", 5.0),
                         cfg["scan_shape"], ph._inf["embeds"],
                         ph._inf["assigns"], polar)
            acom = ic.find_acom_arrays(run)

            def prog(d, t, s):
                self._set_prog(f"{s}: {d}/{t}")

            probe = abl = classical = None
            did_gcam = False

            want_classical = self._opt_classical.get()
            if (self._opt_probe.get() or self._opt_means.get()
                    or want_classical):
                self._set_prog("computing factors + class means (cube pass) …")
                self._factors = ic.compute_factors_and_means(
                    ctx, collect_classical=want_classical, progress=prog)
            if self._opt_probe.get():
                self._set_prog("probing (ridge R² / η² / MI) …")
                probe = ic.probe_and_signatures(ctx, self._factors, acom=acom)
            if self._opt_means.get():
                self._set_prog("rendering class-mean figures …")
                ic.figures_class_means(ctx, self._factors)
            if want_classical:
                self._set_prog("classical-baseline comparison …")
                classical = ic.classical_baselines(ctx, self._factors,
                                                   progress=prog)
            if self._opt_abl.get():
                self._set_prog("loading model for ablations …")
                model, dev = self._load_model(ph)
                abl = ic.run_ablations(ctx, model, dev, progress=prog)
            if self._opt_gcam.get():
                self._set_prog("Grad-CAM / IG (per prototype) …")
                try:
                    ic.run_gradcam(ctx)
                    did_gcam = True
                except Exception as e:
                    self._set_prog(f"Grad-CAM failed: {e}")

            rep = ic.write_report(ctx, probe=probe, ablations=abl,
                                  acom=acom, classical=classical,
                                  did_gradcam=did_gcam)
            self.after(0, lambda: self._finish(run, rep))
        except Exception as e:
            tb = traceback.format_exc()
            self.after(0, lambda: self._fail(e, tb))

    def _load_model(self, ph):
        import torch
        from dino_sr_contrastive_model import load_contrastive_checkpoint
        ckpt = ph._best_ckpt()
        if ckpt is None:
            raise RuntimeError("no checkpoint (best.pth) in the linked run")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _, _ = load_contrastive_checkpoint(ckpt, device=dev)
        model.eval()
        return model, dev

    def _finish(self, run, report_path):
        self._busy = False
        self._run_btn.configure(state="normal", text="▶ Run interpretation")
        self._set_prog("done.")
        # populate figure menu with whatever now exists
        avail = [(lbl, os.path.join(run, rel)) for lbl, rel in _FIG_SPECS
                 if os.path.exists(os.path.join(run, rel))]
        self._fig_paths = {lbl: p for lbl, p in avail}
        labels = list(self._fig_paths) or ["(no figures)"]
        self._fig_menu.configure(values=labels)
        self._fig_menu.set(labels[0])
        if avail:
            self._show_figure(labels[0])
        # load report
        try:
            txt = open(report_path, encoding="utf-8").read()
        except Exception:
            txt = "(report not found)"
        self._report.configure(state="normal")
        self._report.delete("1.0", "end")
        self._report.insert("1.0", txt)
        self._report.configure(state="disabled")

    def _fail(self, e, tb):
        self._busy = False
        self._run_btn.configure(state="normal", text="▶ Run interpretation")
        self._set_prog("failed.")
        print(tb)
        messagebox.showerror("interpretation failed", str(e))

    # ---- figure viewer --------------------------------------------------
    def _show_figure(self, label):
        path = getattr(self, "_fig_paths", {}).get(label)
        self._ax.clear(); self._ax.axis("off")
        if not path or not os.path.exists(path):
            self._ax.text(0.5, 0.5, "(no figure)", ha="center", va="center")
        else:
            try:
                self._ax.imshow(mpimg.imread(path))
            except Exception as e:
                self._ax.text(0.5, 0.5, f"load error:\n{e}",
                              ha="center", va="center")
        self._canvas.draw_idle()

    def _open_folder(self):
        ph = self._posthoc()
        run = getattr(ph, "outdir", None) if ph else None
        if not run:
            return
        out = os.path.join(run, "_interpretability")
        os.makedirs(out, exist_ok=True)
        try:
            os.startfile(out)            # Windows
        except Exception:
            messagebox.showinfo("output", out)
