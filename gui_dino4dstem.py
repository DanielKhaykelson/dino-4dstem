"""gui_dino4dstem.py -- main entry for the DINO-4DSTEM GUI.

Tabs:
    1. Pre-processing  : inspect cube + tune vmax / mask / crop interactively
    2. Training        : choose recipe + sub-tabs of knobs + live loss plot
    3. Eval            : live class map (during training) or load-ckpt mode

Run with:
    python gui_dino4dstem.py

Dependencies: customtkinter (auto-installed via the env), matplotlib,
numpy, pytorch, opencv-python.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import customtkinter as ctk


def _patch_ctk_resize_guard():
    """Global safety net for the customtkinter + matplotlib clash.

    When a matplotlib FigureCanvasTkAgg is embedded near CTk widgets,
    CTk's resize/scaling callbacks can fire `_update_dimensions_event`
    / `_draw` on a widget whose internal handle is the matplotlib
    canvas, which lacks `winfo_exists` → AttributeError that crashes
    the Tk callback.  We wrap the two offending CTk methods to swallow
    that specific AttributeError (harmless — it just means 'skip the
    redraw for this transient state').  Applied once at startup.
    """
    try:
        from customtkinter.windows.widgets.core_widget_classes \
            import ctk_base_class as _cbc
        Base = _cbc.CTkBaseClass
        for meth in ("_update_dimensions_event", "_draw"):
            orig = getattr(Base, meth, None)
            if orig is None or getattr(orig, "_resize_guarded", False):
                continue
            def _wrap(o):
                def inner(self, *a, **k):
                    try:
                        return o(self, *a, **k)
                    except AttributeError as e:
                        if "winfo_exists" in str(e):
                            return None
                        raise
                inner._resize_guarded = True
                return inner
            setattr(Base, meth, _wrap(orig))
    except Exception as _e:
        print(f"[ctk-guard] could not patch: {_e!r}", flush=True)


def _patch_ctk_entry_empty_guard():
    """Global fix for CTkEntry bound to IntVar/DoubleVar.

    When the user clears such an entry to retype, CTk's
    `_textvariable_callback` does `self._textvariable.get() == ""`,
    but IntVar/DoubleVar.get() raises TclError on an empty string
    instead of returning "".  Wrap the callback to swallow that
    TclError (the field is mid-edit; nothing to do).
    """
    try:
        import tkinter as _tk
        from customtkinter.windows.widgets import ctk_entry as _ce
        orig = _ce.CTkEntry._textvariable_callback
        if getattr(orig, "_empty_guarded", False):
            return
        def inner(self, *a, **k):
            try:
                return orig(self, *a, **k)
            except _tk.TclError:
                return None       # entry is empty mid-edit; ignore
        inner._empty_guarded = True
        _ce.CTkEntry._textvariable_callback = inner
    except Exception as _e:
        print(f"[ctk-guard] entry patch failed: {_e!r}", flush=True)


_patch_ctk_resize_guard()
_patch_ctk_entry_empty_guard()

from gui_app.pre_panel import PrePanel
from gui_app.train_panel import TrainPanel
from gui_app.eval_panel import EvalPanel
from gui_app.posthoc_panel import PostHocPanel
from gui_app.sam_panel import SAMPanel
from gui_app.blob_panel import BlobPanel
from gui_app.strain_panel import StrainPanel
from gui_app.fluct_panel import FluctPanel
from gui_app.symm_panel import SymmPanel
from gui_app.nmf_panel import NMFPanel
from gui_app.dino_pretrained_panel import DINOClusterPanel
from gui_app.transfer_panel import TransferPanel
from gui_app.acom_tab import ACOMTabPanel


class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.title("DINO-4DSTEM")
        self.geometry("1480x920")
        self.minsize(1240, 780)
        # Global session = single source of truth for (sample, run_dir,
        # inference).  All panels subscribe; no more per-panel
        # outdir/sample fan-out.
        from gui_app.session import Session
        self.session = Session()

        self._build_topbar()
        self._build_tabs()
        self._build_statusbar()

        # Intercept window-close so we can kill any orphan training
        # subprocesses cleanly. Without this, closing the GUI just
        # detaches the worker → it keeps eating GPU until it finishes
        # (or until the user finds it via Task Manager).
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----- shutdown / orphan-process protection ----------------------
    def _collect_running_jobs(self):
        """Return list of (label, job) tuples for every TrainingJob
        across the GUI that's still alive in a subprocess."""
        jobs = []
        for label, owner_attr, job_attr in (
            ("training",  "train",    "job"),
            ("fine-tune", "posthoc",  "_ft_job"),
            ("SAM",       "sam",      "job"),
            ("transfer",  "transfer", "job"),
        ):
            owner = getattr(self, owner_attr, None)
            if owner is None:
                continue
            j = getattr(owner, job_attr, None)
            try:
                if j is not None and j.is_running():
                    jobs.append((label, j))
            except Exception:
                # if is_running blows up, treat as not-running
                pass
        return jobs

    def _on_close(self):
        running = self._collect_running_jobs()
        if not running:
            self.destroy()
            return
        # Compose the confirmation message.
        from tkinter import messagebox
        msg_lines = [
            "There are running background subprocess(es):",
            "",
        ]
        for label, j in running:
            outdir = getattr(j, "outdir", "?")
            msg_lines.append(f"  • {label}  →  {outdir}")
        msg_lines += [
            "",
            "Closing the GUI will KILL them (any unsaved progress is lost; "
            "checkpoints already on disk are kept).",
            "",
            "Quit anyway?",
        ]
        ok = messagebox.askyesno("Confirm quit",
                                   "\n".join(msg_lines),
                                   default="no")
        if not ok:
            return
        # Tear down each subprocess. Stop is graceful (terminate then
        # SIGKILL after timeout); errors are swallowed so a single bad
        # job can't block window close.
        for label, j in running:
            try:
                j.stop()
            except Exception as e:
                print(f"[shutdown] failed to stop {label}: {e!r}",
                      flush=True)
        self.destroy()

    # ---- session badges --------------------------------------------
    def _on_session_change(self, sess):
        from gui_app._ui import COLOR
        ds = sess.sample or "(none)"
        try:
            self._badge_dataset.configure(
                text=f"● dataset:  {ds}")
        except Exception: pass
        run = "(none)"
        if sess.run_dir:
            run = sess.run_dir.replace("\\", "/").split("/")[-1]
        try:
            self._badge_run.configure(text=f"● run:  {run}")
        except Exception: pass
        try:
            if sess.has_inference():
                self._dot_inf.set("ok", "inference cached")
            elif sess.has_run():
                self._dot_inf.set("idle", "run loaded, no inference")
            else:
                self._dot_inf.set("idle", "no run loaded")
        except Exception: pass

    def _pick_dataset_badge(self):
        """Open a quick sample picker."""
        try:
            import tkinter as tk
            from data import SAMPLES
            top = tk.Toplevel(self); top.title("Pick a sample")
            top.geometry("420x460")
            lst = tk.Listbox(top, font=("Consolas", 10))
            lst.pack(fill="both", expand=True, padx=8, pady=8)
            keys = sorted(SAMPLES.keys())
            for k in keys: lst.insert("end", k)
            def _ok():
                sel = lst.curselection()
                if sel:
                    self.session.set(sample=keys[int(sel[0])])
                top.destroy()
            ctk.CTkButton(top, text="Use this sample",
                           command=_ok).pack(pady=6)
        except Exception as e:
            messagebox.showerror("Pick sample", repr(e))

    def _pick_run_badge(self):
        """Open a run-dir picker.  Routes through Session.load_run_dir
        which resolves the sample via SAMPLE_LOCK.json /
        _train_kwargs.json automatically."""
        from tkinter import filedialog, messagebox
        d = filedialog.askdirectory(initialdir="runs",
            title="Pick a run dir")
        if not d: return
        sample = self.session.load_run_dir(d)
        # if posthoc tab exists, also push the inference so it caches
        # for downstream tabs (ACOM, Crystallinity).
        ph = getattr(self, "posthoc", None)
        if ph is not None and sample is not None:
            try: ph.link_run(d, sample)
            except Exception: pass
        if sample is None:
            messagebox.showinfo("Run dir",
                "Run dir loaded but no sample could be auto-resolved "
                "from SAMPLE_LOCK.json / _train_kwargs.json.  Click "
                "the dataset badge to pick one manually.")

    def _open_run_inspector(self):
        """Generate runs/_gui/_inspect.html via inspect_runs.gather +
        HTML_TEMPLATE, then open it in the default browser.  Falls back
        to the full runs/ root if runs/_gui/ doesn't exist yet."""
        import os, webbrowser
        from tkinter import messagebox
        # Pick the root: prefer runs/_gui (GUI-launched runs only).
        cwd = os.getcwd()
        candidates = [
            os.path.join(cwd, "runs", "_gui"),
            os.path.join(cwd, "runs"),
        ]
        root = next((c for c in candidates if os.path.isdir(c)), None)
        if root is None:
            messagebox.showinfo("Inspect runs",
                "No runs/ directory found in the current working "
                "directory.  Run training first.")
            return
        out_path = os.path.join(root, "_inspect.html")
        try:
            from inspect_runs import gather, emit_html
            payload = gather(root)
            emit_html(payload, root, out_path)
        except Exception as e:
            messagebox.showerror("Inspect runs",
                f"could not build / write HTML:\n{e!r}"); return
        try:
            webbrowser.open("file:///" + out_path.replace("\\", "/"))
        except Exception as e:
            messagebox.showinfo("Inspect runs",
                f"wrote {out_path}\n(could not auto-open: {e!r})")
            return
        self._sb.configure(
            text=f"run inspector → {out_path}  "
                  f"({len(payload)} runs)")

    def _build_topbar(self):
        # ----- two-row toolbar -----
        bar = ctk.CTkFrame(self, height=42)
        bar.pack(side="top", fill="x", padx=4, pady=(4, 0))
        ctk.CTkLabel(bar, text="DINO-4DSTEM",
                      font=("Segoe UI", 16, "bold")).pack(side="left",
                                                            padx=12, pady=6)
        ctk.CTkButton(bar, text="Inspect runs  (HTML)",
                       width=170,
                       command=self._open_run_inspector
                       ).pack(side="left", padx=8, pady=4)
        # Session badges: dataset + run + inference status.  Click to
        # change.  Updated by any panel that drives a load.
        from gui_app._ui import session_badge, StatusDot
        self._badge_dataset = session_badge(bar,
            "● dataset:  (none)",
            command=self._pick_dataset_badge)
        self._badge_dataset.pack(side="left", padx=4, pady=4)
        self._badge_run = session_badge(bar,
            "● run:  (none)",
            command=self._pick_run_badge)
        self._badge_run.pack(side="left", padx=4, pady=4)
        self._dot_inf = StatusDot(bar, label="inference")
        self._dot_inf.set("idle", "no inference cached")
        self._dot_inf.pack(side="left", padx=6)
        self.session.subscribe(self._on_session_change)

        # Resolution fields (shared across all tabs).
        # Convention: both values are for the ORIGINAL un-resized data.
        #   real_res  : nm per scan-grid pixel  (the step size of your scan)
        #   recip_res : nm⁻¹ per RAW detector pixel  (your microscope value)
        # Every plot in the GUI applies the right resize/crop scaling
        # automatically — you only enter these two numbers.
        self.real_res = ctk.DoubleVar(value=1.0)
        self.recip_res = ctk.DoubleVar(value=0.0185)
        cal = ctk.CTkFrame(bar, fg_color="transparent")
        cal.pack(side="right", padx=12)
        ctk.CTkLabel(cal, text="real space:").grid(row=0, column=0, sticky="e")
        ctk.CTkEntry(cal, textvariable=self.real_res,
                       width=80).grid(row=0, column=1, padx=2)
        ctk.CTkLabel(cal, text="nm/px").grid(row=0, column=2, padx=(0, 4))
        from gui_app.tooltip import add_help_button
        add_help_button(cal,
            "Real-space calibration: nanometres per SCAN-grid pixel.\n\n"
            "This is the spacing between scan positions (the step size you "
            "set on the microscope when acquiring the 4D-STEM map). "
            "Used for the 100 nm scale bar on the class map / variance map "
            "and the y/x axis labels on real-space images.").grid(
                row=0, column=3, padx=(0, 12))
        ctk.CTkLabel(cal, text="reciprocal:").grid(row=0, column=4, sticky="e")
        ctk.CTkEntry(cal, textvariable=self.recip_res,
                       width=80).grid(row=0, column=5, padx=2)
        ctk.CTkLabel(cal, text="nm⁻¹/px").grid(row=0, column=6)
        add_help_button(cal,
            "Reciprocal calibration: nm⁻¹ per RAW detector pixel.\n\n"
            "Use the value reported by your microscope at the camera-"
            "length / acquisition you took the data with — i.e. the "
            "calibration of the ORIGINAL detector image, BEFORE any "
            "resize / crop the GUI applies internally.\n\n"
            "Every downstream plot (1D radial, GradCAM, class avg grid, "
            "Symm-map, scale bars on diffraction patterns) automatically "
            "applies the right resize/crop factor — so just enter your "
            "raw-detector value and you're done.\n\n"
            "Used for the 0.2 nm⁻¹ scale bars and the q-axis on 1D "
            "plots. The '1D advanced' popup overlays CIF reflections "
            "after converting their |g| from Å⁻¹ to nm⁻¹ (×10).").grid(
                row=0, column=7, padx=(2, 0))

    def _build_tabs(self):
        # Lazy-construct heavy tabs (SAM, Transfer) on first click rather
        # than at startup. Both hold matplotlib Figures + dense widget
        # trees that slow down the global Tk event loop and the
        # right-click popup path on the Post-hoc tab. Eager construction
        # of all panels was costing ~hundreds of ms per matplotlib
        # operation across the whole GUI.
        self._tabs = ctk.CTkTabview(self, anchor="nw",
                                       command=self._on_tab_change)
        self._tabs.pack(side="top", fill="both", expand=True,
                         padx=8, pady=(6, 4))
        # 5 high-level groups instead of 11 flat tabs.  Each group is
        # itself a CTkTabview holding the original panels.
        g_data    = self._tabs.add("Data")
        g_model   = self._tabs.add("Model")
        g_anal    = self._tabs.add("Analysis")
        g_clust   = self._tabs.add("Clustering")
        g_diff    = self._tabs.add("Diffraction")
        self._group_tabs = {}
        def _grp(parent, members):
            tv = ctk.CTkTabview(parent, anchor="nw",
                                  command=self._on_tab_change)
            tv.pack(fill="both", expand=True)
            return {name: tv.add(name) for name in members}, tv
        # Data: one tab "Pre-processing" with Synthetic nested inside.
        # Model: "Training" with Eval nested inside, plus Transfer.
        data_frames,  self._group_tabs["Data"]  = _grp(g_data,
            ["Pre-processing"])
        # Pre-processing sub-tabs: Load (the existing pre panel) +
        # Synthetic.
        pre_subtabs = ctk.CTkTabview(data_frames["Pre-processing"],
                                            anchor="nw",
                                            command=self._on_tab_change)
        pre_subtabs.pack(fill="both", expand=True)
        pre_load_tab  = pre_subtabs.add("Load + Pre-process")
        synth_tab     = pre_subtabs.add("Synthetic")
        self._group_tabs["Pre-processing"] = pre_subtabs
        model_frames, self._group_tabs["Model"] = _grp(g_model,
            ["Training", "Transfer"])
        # Training sub-tabs: Train + Eval (Eval moved from top-level).
        tr_subtabs = ctk.CTkTabview(model_frames["Training"],
                                          anchor="nw",
                                          command=self._on_tab_change)
        tr_subtabs.pack(fill="both", expand=True)
        train_tab = tr_subtabs.add("Training")
        eval_tab  = tr_subtabs.add("Eval")
        self._group_tabs["Training"] = tr_subtabs
        anal_frames,  self._group_tabs["Analysis"] = _grp(g_anal,
            ["Post-hoc"])
        clust_frames, self._group_tabs["Clustering"] = _grp(g_clust,
            ["NMF + cluster", "DINO + cluster", "SAM"])
        diff_frames,  self._group_tabs["Diffraction"] = _grp(g_diff,
            ["Blob", "ACOM"])
        pre_tab      = pre_load_tab
        transfer_tab = model_frames["Transfer"]
        posthoc_tab  = anal_frames["Post-hoc"]
        nmf_tab      = clust_frames["NMF + cluster"]
        dinoc_tab    = clust_frames["DINO + cluster"]
        sam_tab      = clust_frames["SAM"]
        blob_tab     = diff_frames["Blob"]
        acom2_tab    = diff_frames["ACOM"]

        # Eager — used immediately or cheap to build.
        self.pre = PrePanel(pre_tab,
                             on_state_change=self._on_pre_state)
        self.pre.pack(fill="both", expand=True)
        self.eval_panel = EvalPanel(eval_tab)
        self.eval_panel.pack(fill="both", expand=True)

        # Post-hoc tab is now a sub-tabview: "Analysis" (the original
        # PostHocPanel), "Fluct-map", "Symm-map". All three live under the
        # one top-level "Post-hoc" tab.
        self._posthoc_tabs = ctk.CTkTabview(posthoc_tab, anchor="nw",
                                              command=self._on_posthoc_subtab)
        self._posthoc_tabs.pack(fill="both", expand=True)
        ph_main_tab     = self._posthoc_tabs.add("Analysis")
        ph_fluct_tab    = self._posthoc_tabs.add("Fluct-map")
        ph_symm_tab     = self._posthoc_tabs.add("Symm-map")
        # 'Ordering' sub-tab removed — Crystallinity (under Blob) does
        # the same peak/background-ratio mapping with the r-window
        # the user actually wants.
        self.posthoc = PostHocPanel(ph_main_tab, app=self)
        self.posthoc.pack(fill="both", expand=True)
        self.fluct = None
        self.symm = None
        self.ordering = None    # kept for any back-compat callers
        self._posthoc_subtab_frames = {
            "Fluct-map": ph_fluct_tab,
            "Symm-map":  ph_symm_tab,
        }
        self._posthoc_subtab_built = set()

        # ACOM tab: step-by-step (source / detection / calibration /
        # single-pattern fit / batch) — pulls source data from the
        # posthoc tab.  `acom2` attribute name kept for back-compat
        # with posthoc → acom hand-off.
        self.acom2 = ACOMTabPanel(acom2_tab, app=self)
        self.acom2.pack(fill="both", expand=True)
        self.nmf = NMFPanel(nmf_tab, app=self)
        self.nmf.pack(fill="both", expand=True)
        self.dino_cluster = DINOClusterPanel(dinoc_tab, app=self)
        self.dino_cluster.pack(fill="both", expand=True)
        self.train = TrainPanel(train_tab,
                                 prepanel=self.pre,
                                 on_run_started=self._on_train_started,
                                 on_run_finished=self._on_train_finished)
        self.train.pack(fill="both", expand=True)

        # Lazy — built on first tab visit. Until then these are None.
        self.sam = None
        self.blob = None
        self.transfer = None
        self.synth = None
        self._lazy_tab_frames = {
            "SAM": sam_tab,
            "Blob": blob_tab,
            "Transfer": transfer_tab,
            "Synthetic": synth_tab,
        }
        self._lazy_tab_built = set()

    def _on_tab_change(self):
        # Walk down the group hierarchy until we hit a leaf.  Outer
        # group tabs (Data/Model/...) and intermediate tabs that are
        # themselves containers (Pre-processing, Training) never need
        # lazy build — only the leaves do.
        name = None
        try:
            cur = self._tabs.get()
            for _ in range(4):
                tv = self._group_tabs.get(cur)
                if tv is None:
                    name = cur; break
                cur = tv.get()
            else:
                name = cur
        except Exception:
            return
        if not name or name in self._lazy_tab_built:
            return
        if name == "SAM":
            self.sam = SAMPanel(self._lazy_tab_frames["SAM"], app=self)
            self.sam.pack(fill="both", expand=True)
            # Replay any pending auto-link from a training run that
            # finished before the user visited this tab.
            pl = getattr(self, "_pending_sam_link", None)
            if pl is not None:
                try: self.sam.link_run(*pl)
                except Exception: pass
                self._pending_sam_link = None
        elif name == "Blob":
            # Blob top-level tab → sub-tabview with Detect / Strain / ACOM.
            blob_tab = self._lazy_tab_frames["Blob"]
            self._blob_tabs = ctk.CTkTabview(blob_tab, anchor="nw",
                                                command=self._on_blob_subtab)
            self._blob_tabs.pack(fill="both", expand=True)
            blob_detect_tab = self._blob_tabs.add("Detect")
            blob_strain_tab = self._blob_tabs.add("Strain")
            blob_cryst_tab  = self._blob_tabs.add("Crystallinity")
            # Detect is the existing BlobPanel (eager — first visit).
            self.blob = BlobPanel(blob_detect_tab, app=self)
            self.blob.pack(fill="both", expand=True)
            self.strain = None
            self.crystallinity = None
            self._blob_subtab_frames = {
                "Strain":        blob_strain_tab,
                "Crystallinity": blob_cryst_tab,
            }
            self._blob_subtab_built = set()
            pl = getattr(self, "_pending_blob_link", None)
            if pl is not None:
                try: self.blob.link_run(*pl)
                except Exception: pass
                self._pending_blob_link = None
        elif name == "Transfer":
            self.transfer = TransferPanel(self._lazy_tab_frames["Transfer"],
                                            app=self)
            self.transfer.pack(fill="both", expand=True)
        elif name == "Synthetic":
            from gui_app.synth_panel import SynthPanel
            self.synth = SynthPanel(self._lazy_tab_frames["Synthetic"],
                                       app=self)
            self.synth.pack(fill="both", expand=True)
        else:
            return  # nothing lazy to build for this tab
        self._lazy_tab_built.add(name)

    def _on_blob_subtab(self):
        """Lazy-build Strain / ACOM sub-tabs inside Blob."""
        try:
            name = self._blob_tabs.get()
        except Exception:
            return
        if name in self._blob_subtab_built:
            return
        if name == "Strain":
            self.strain = StrainPanel(
                self._blob_subtab_frames["Strain"], app=self)
            self.strain.pack(fill="both", expand=True)
            try:
                if self.blob is not None and self.blob.outdir is not None:
                    self.strain.link_run(self.blob.outdir, self.blob.sample)
            except Exception:
                pass
        elif name == "Crystallinity":
            from gui_app.crystallinity_panel import CrystallinityPanel
            self.crystallinity = CrystallinityPanel(
                self._blob_subtab_frames["Crystallinity"], app=self)
            self.crystallinity.pack(fill="both", expand=True)
        else:
            return
        self._blob_subtab_built.add(name)

    def _on_posthoc_subtab(self):
        """Lazy-build Fluct-map / Symm-map sub-tabs inside Post-hoc."""
        try:
            name = self._posthoc_tabs.get()
        except Exception:
            return
        if name in self._posthoc_subtab_built:
            return
        if name == "Fluct-map":
            self.fluct = FluctPanel(
                self._posthoc_subtab_frames["Fluct-map"], app=self)
            self.fluct.pack(fill="both", expand=True)
            pl = getattr(self, "_pending_fluct_link", None)
            if pl is not None:
                try: self.fluct.link_run(*pl)
                except Exception: pass
                self._pending_fluct_link = None
        elif name == "Symm-map":
            self.symm = SymmPanel(
                self._posthoc_subtab_frames["Symm-map"], app=self)
            self.symm.pack(fill="both", expand=True)
            pl = getattr(self, "_pending_symm_link", None)
            if pl is not None:
                try: self.symm.link_run(*pl)
                except Exception: pass
                self._pending_symm_link = None
        else:
            return
        self._posthoc_subtab_built.add(name)

    def _build_statusbar(self):
        sb = ctk.CTkFrame(self, height=24)
        sb.pack(side="bottom", fill="x", padx=4, pady=(0, 4))
        self._sb = ctk.CTkLabel(sb, text="Ready.",
                                  font=("Segoe UI", 10), anchor="w")
        self._sb.pack(side="left", padx=10)
        ctk.CTkLabel(sb,
            text="© DINO-4DSTEM",
            font=("Segoe UI", 9), text_color=("#888", "#888")
            ).pack(side="right", padx=10)

    # ---- callbacks between tabs ----
    def _on_pre_state(self, kind, **kwargs):
        if kind == "pre_kwargs_changed":
            # The user moved a pre-processing knob (vmax / crop /
            # polar_mask_cols / COM).  Push it straight into the
            # Training tab so its Apply snapshot is never stale.
            tr = getattr(self, "train", None)
            if tr is not None:
                fn = getattr(tr, "_sync_from_prepanel", None)
                if callable(fn):
                    try: fn()
                    except Exception: pass
            return
        if kind == "pair_labels_changed":
            # Tell the Training tab to refresh its λ_pair info line
            # so the auto-default note shows up immediately.
            try:
                fn = getattr(self.train, "_refresh_pair_label_info", None)
                if callable(fn): fn()
            except Exception:
                pass
            # Also bump the visible status so the user knows it took.
            cnt = kwargs.get("label_count") or {}
            self._sb.configure(
                text=f"pair labels updated: {cnt.get('total', 0)} pairs "
                      f"({cnt.get('same', 0)} same / "
                      f"{cnt.get('diff', 0)} diff). "
                      f"Training will auto-apply λ_pair=0.1 unless you "
                      f"set it on the Training tab.")
            return
        if kind == "loaded":
            sk = kwargs.get("sample_key")
            path = kwargs.get("path", "")
            if sk:
                self._sb.configure(
                    text=f"loaded cube: {path}    (sample key: {sk})")
                # Tell the Training tab a new sample is available so it can
                # refresh its dropdown and auto-select it.
                try:
                    self.train.on_runtime_sample_added(sk)
                except AttributeError:
                    pass
                # Eval / Post-hoc / ACOM read SAMPLES on demand, so they
                # don't need an explicit notification — but if a panel
                # caches a dropdown, refresh it.
                for panel in (getattr(self, "eval_panel", None),
                                getattr(self, "posthoc", None),
                                getattr(self, "acom2", None)):
                    fn = getattr(panel, "on_runtime_sample_added", None)
                    if callable(fn):
                        try: fn(sk)
                        except Exception: pass
            else:
                self._sb.configure(text=f"loaded cube: {path}")

    def _on_train_started(self, outdir, sample, save_every):
        self.eval_panel.link_training_run(outdir, sample, save_every)
        self._sb.configure(
            text=f"training started → {outdir}  (eval tab is watching)")
        # auto-switch to Training tab so user can see loss live
        try:
            self._tabs.set("Training")
        except Exception:
            pass

    def _on_train_finished(self, outdir):
        self.eval_panel.on_training_finished(outdir)
        sample = self.train.var["sample"].get()
        # auto-link post-hoc to the just-finished run too
        try:
            self.posthoc.link_run(outdir, sample)
        except Exception:
            pass
        # auto-link SAM panel as well, but ONLY if the user has visited
        # that tab (i.e. it's been lazy-built).  If they haven't, it'll
        # link on first visit instead via _on_tab_change.
        try:
            if self.sam is not None:
                self.sam.link_run(outdir, sample)
            else:
                # Stash the pending link for when SAM gets built.
                self._pending_sam_link = (outdir, sample)
        except Exception:
            pass
        # Same dance for the Blob tab.
        try:
            if self.blob is not None:
                self.blob.link_run(outdir, sample)
            else:
                self._pending_blob_link = (outdir, sample)
        except Exception:
            pass
        # And Fluct-map.
        try:
            if self.fluct is not None:
                self.fluct.link_run(outdir, sample)
            else:
                self._pending_fluct_link = (outdir, sample)
        except Exception:
            pass
        # And Symm-map.
        try:
            if self.symm is not None:
                self.symm.link_run(outdir, sample)
            else:
                self._pending_symm_link = (outdir, sample)
        except Exception:
            pass
        # And the NMF + clustering tab.
        try:
            if getattr(self, "nmf", None) is not None:
                self.nmf.link_run(outdir, sample)
        except Exception:
            pass
        # And the DINO + clustering tab.
        try:
            if getattr(self, "dino_cluster", None) is not None:
                self.dino_cluster.link_run(outdir, sample)
        except Exception:
            pass
        self._sb.configure(text=f"training done.  outputs in {outdir}")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
