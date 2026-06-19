"""eval_panel.py -- Tab 3 (Evaluation).

Two modes:
  * LIVE: while training is running, watch <outdir>/ckpt_ep<N>.pth and
    re-render class map + per-class averages each time a new
    checkpoint appears.
  * LOAD: pick an existing run directory and render off best.pth (or
    the latest ckpt).

Inference uses contrastive_eval.infer_scan with dense_remap=True so
the displayed class IDs are 0..K_active-1 sorted by count.
"""
from __future__ import annotations
import os, sys, json, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap, BoundaryNorm

from data import SAMPLES, LoadPRZ, loaded_sample_keys

# Dataset dropdowns list only data LOADED BY PATH this session (no built-in
# 46-name catalogue).  This hint shows when nothing is loaded yet.
_NO_DATA_HINT = "(load a cube…)"
from gui_app.runner import list_ckpts


def _find_sample_lock_eval(start_dir, max_walk=5):
    """Walk up from `start_dir` looking for a SAMPLE_LOCK.json (dropped
    by tools/sweep_m_K.py at the sample-level dir)."""
    cur = os.path.abspath(start_dir)
    for _ in range(int(max_walk)):
        p = os.path.join(cur, "SAMPLE_LOCK.json")
        if os.path.exists(p):
            return p
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _adaptive_cmap(K_act):
    if K_act <= 10:
        base = list(plt_get_tab10()[:K_act])
    elif K_act <= 20:
        base = list(plt_get_tab20()[:K_act])
    else:
        import matplotlib.pyplot as plt
        base = [plt.get_cmap("turbo")(i / max(K_act - 1, 1))
                for i in range(K_act)]
    return ListedColormap(base, name=f"K{K_act}")


def plt_get_tab10():
    import matplotlib.pyplot as plt
    return list(plt.get_cmap("tab10").colors)


def plt_get_tab20():
    import matplotlib.pyplot as plt
    return list(plt.get_cmap("tab20").colors)


class EvalPanel(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master)
        # state
        self.outdir: "str | None" = None
        self.sample: "str | None" = None
        self.save_every: int = 10
        self._last_ckpt_epoch: int = -1
        self._poll_after = None
        self._busy = False
        self._build()

    def _build(self):
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)

        self._mode_var = ctk.StringVar(value="LIVE")
        ctk.CTkLabel(top, text="Mode:",
                      font=("Segoe UI", 11, "bold")).pack(side="left",
                                                            padx=(6, 4))
        ctk.CTkSegmentedButton(top, values=["LIVE", "LOAD"],
                                 variable=self._mode_var,
                                 command=lambda _v: self._sync_mode()
                                 ).pack(side="left")

        # Live-mode info row (read-only)
        self._live_info = ctk.CTkLabel(top, text="(no training run linked)",
                                         font=("Consolas", 10), anchor="w",
                                         justify="left")
        self._live_info.pack(side="left", padx=8)

        # Load-mode controls (hidden by default).  Two logical halves:
        #   (1) MODEL  : the run dir whose checkpoint we load.
        #   (2) DATASET: ANY cube to run that model on — a built-in key,
        #                a constituent cube of a multi-run, or an
        #                arbitrary cube file via "Browse cube…".
        self._load_box = ctk.CTkFrame(top, fg_color="transparent")
        ctk.CTkButton(self._load_box, text="Load run dir… (model)",
                       width=150,
                       command=self._load_dir_dialog).pack(side="left", padx=4)
        self._load_path_var = ctk.StringVar()
        ctk.CTkEntry(self._load_box, textvariable=self._load_path_var,
                       width=300).pack(side="left", padx=2)
        ctk.CTkLabel(self._load_box, text="dataset:").pack(side="left",
                                                           padx=(10, 2))
        self._sample_var = ctk.StringVar(value="Na007b")
        self._sample_menu = ctk.CTkOptionMenu(
            self._load_box, variable=self._sample_var,
            values=(loaded_sample_keys() or [_NO_DATA_HINT]), width=200)
        self._sample_menu.pack(side="left", padx=2)
        ctk.CTkButton(self._load_box, text="Browse cube…", width=110,
                       command=self._browse_dataset).pack(side="left", padx=2)
        ctk.CTkButton(self._load_box, text="Render", width=90,
                       command=self._render_from_load
                       ).pack(side="left", padx=4)

        # status line
        self._status = ctk.CTkLabel(self, text="",
                                      font=("Consolas", 10), anchor="w",
                                      justify="left")
        self._status.pack(side="top", fill="x", padx=8, pady=(0, 2))

        # figure
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)
        self._fig = Figure(figsize=(12, 7), dpi=95, facecolor="#f4f4f4")
        # 1 row class map (wide), then class averages grid below
        self._fig.subplots_adjust(left=0.04, right=0.98, top=0.94,
                                    bottom=0.04, hspace=0.35)
        self._canvas = FigureCanvasTkAgg(self._fig, master=body)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self._sync_mode()

        # When the user navigates back to this tab, make sure the live
        # watch is still running (re-arm if it stopped) so it stays
        # "always connected to the current kmap".
        # NOTE: self._canvas is the matplotlib canvas (shadows CTkFrame's
        # internal _canvas), so self.bind() is broken here — bind the
        # underlying tk widget instead.
        try:
            self._canvas.get_tk_widget().bind(
                "<Map>", lambda _e: self._ensure_polling(), add="+")
        except Exception:
            pass

    def _ensure_polling(self):
        """Re-arm the LIVE poll loop if a run is linked but the
        self.after chain isn't currently scheduled."""
        try:
            if (self._mode_var.get() == "LIVE" and self.outdir is not None
                    and self._poll_after is None):
                self._start_polling()
        except Exception:
            pass

    def _sync_mode(self):
        if self._mode_var.get() == "LIVE":
            self._load_box.pack_forget()
            self._live_info.pack(side="left", padx=8)
        else:
            self._live_info.pack_forget()
            self._load_box.pack(side="left", padx=8)

    # ---- API for main app ----
    def link_training_run(self, outdir, sample, save_every):
        """Called when training starts. Switch to LIVE and start polling."""
        self.outdir = outdir
        self.sample = sample
        self.save_every = max(5, int(save_every))
        self._last_ckpt_epoch = -1
        self._mode_var.set("LIVE")
        self._sync_mode()
        self._live_info.configure(
            text=f"watching {outdir}\nsample={sample}  save_every={self.save_every}  "
                  f"(class map updates per checkpoint)")
        self._status.configure(text="awaiting first checkpoint...")
        self._start_polling()

    def on_training_finished(self, outdir):
        # one final refresh once best.pth lands
        self._status.configure(text="training done — rendering best.pth …")
        # try best.pth one more time
        self._refresh_from_latest_ckpt(prefer_best=True)

    def _start_polling(self):
        if self._poll_after is not None:
            try: self.after_cancel(self._poll_after)
            except Exception: pass
        self._poll_after = self.after(2000, self._poll)

    def _poll(self):
        # Self-healing watch loop: ANY transient error (a half-written
        # mid-training checkpoint, an infer hiccup) must NOT kill the
        # poll chain — otherwise the tab "dies" and only a full reload
        # revives it.  Always reschedule in finally.
        try:
            if self.outdir is not None and os.path.isdir(self.outdir):
                cks = list_ckpts(self.outdir)
                if cks:
                    ep, p = cks[-1]
                    if ep > self._last_ckpt_epoch and not self._busy:
                        self._last_ckpt_epoch = ep
                        self._render_from_ckpt(p, label=f"ep{ep}")
        except Exception as e:
            try:
                print(f"[eval] poll error (continuing): {e!r}", flush=True)
            except Exception:
                pass
        finally:
            try:
                self._poll_after = self.after(2000, self._poll)
            except Exception:
                self._poll_after = None

    # ---- LOAD mode ----
    def _load_dir_dialog(self):
        d = filedialog.askdirectory(initialdir="runs",
            title="Pick a run dir (contains best.pth + ckpt_ep*.pth)")
        if not d:
            return
        self._load_path_var.set(d)
        rs = os.path.join(d, "run_summary.json")
        sample_inferred = None
        if os.path.exists(rs):
            try:
                js = json.load(open(rs))
                sample_inferred = (js.get("sample")
                                    or js.get("cfg", {}).get("sample"))
            except Exception:
                pass
        # Sweep runs drop SAMPLE_LOCK.json at <sweep_root>/<sample>/
        # with the exact cube_path + vmax + crop + polar mask the sweep
        # used.  Use it when present so the GUI matches the sweep's
        # pre-processing (and so non-built-in keys like "IMC_SI5" auto-
        # register).
        if sample_inferred:
            lock_path = _find_sample_lock_eval(d)
            if lock_path:
                try:
                    spec = json.load(open(lock_path, encoding="utf-8"))
                    cube_p = spec.get("cube_path")
                    if cube_p and os.path.exists(cube_p):
                        from data import register_runtime_sample
                        vmax = float(spec.get("vmax", 2.0))
                        pmc = int(spec.get("polar_mask_cols", 0))
                        derived_cmr = pmc // 2
                        register_runtime_sample(
                            cube_p, vmax=vmax,
                            center_mask_radius=derived_cmr,
                            key=sample_inferred,
                        )
                        try:
                            self._sample_menu.configure(
                                values=(loaded_sample_keys() or [_NO_DATA_HINT]))
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[eval] SAMPLE_LOCK auto-register "
                           f"failed: {e!r}", flush=True)
        # Auto-register runtime sample (loaded__*) from
        # _train_kwargs.json so the user doesn't have to remember
        # to pick the matching sample on Tab 1 first.
        if sample_inferred and sample_inferred not in SAMPLES:
            tk_path = os.path.join(d, "_train_kwargs.json")
            if os.path.exists(tk_path):
                try:
                    kw = json.load(open(tk_path, encoding="utf-8"))
                    cfg = kw.get("_sample_config")
                    from data import register_runtime_sample
                    if cfg and cfg.get("is_multi") and cfg.get("paths"):
                        # MULTI run: register each constituent cube as its
                        # own dataset so eval can show each one.  The
                        # combined key has no single 2D scan grid.
                        reg = []
                        for pth in cfg["paths"]:
                            try:
                                reg.append(register_runtime_sample(
                                    pth, vmax=float(cfg.get("vmax", 2.0)),
                                    center_mask_radius=int(
                                        cfg.get("center_mask_radius", 15))))
                            except Exception as e:
                                print(f"[eval] multi cube register "
                                       f"failed for {pth}: {e!r}", flush=True)
                        self._sync_sample_choices(
                            select=(reg[0] if reg else None))
                        if reg:
                            self._status.configure(
                                text=f"multi-run: {len(reg)} cubes — pick "
                                      f"each in the 'dataset' dropdown, "
                                      f"then Render")
                    elif cfg and cfg.get("path"):
                        scan_shape = cfg.get("scan_shape")
                        register_runtime_sample(
                            cfg["path"],
                            scan_shape=(tuple(scan_shape)
                                          if scan_shape else None),
                            vmax=float(cfg.get("vmax", 2.0)),
                            center_mask_radius=int(
                                cfg.get("center_mask_radius", 15)),
                            key=sample_inferred,
                        )
                        self._sync_sample_choices()
                except Exception as e:
                    print(f"[eval] auto-register from "
                           f"_train_kwargs failed: {e!r}", flush=True)
        # If the inferred sample IS a multi entry already present, expand
        # its constituent cubes into the dataset dropdown too.
        if (sample_inferred and sample_inferred in SAMPLES
                and SAMPLES[sample_inferred].get("is_multi")):
            from data import register_runtime_sample
            reg = []
            for pth in SAMPLES[sample_inferred].get("paths", []):
                try:
                    reg.append(register_runtime_sample(
                        pth,
                        vmax=float(SAMPLES[sample_inferred].get("vmax", 2.0)),
                        center_mask_radius=int(
                            SAMPLES[sample_inferred].get(
                                "center_mask_radius", 15))))
                except Exception:
                    pass
            self._sync_sample_choices(select=(reg[0] if reg else None))
            if reg:
                self._status.configure(
                    text=f"multi-run: {len(reg)} cubes — pick each in the "
                          f"'dataset' dropdown, then Render")
        elif sample_inferred and sample_inferred in SAMPLES:
            self._sync_sample_choices(select=sample_inferred)

    def _sync_sample_choices(self, select: "str | None" = None):
        """Refresh the dataset dropdown's options from SAMPLES, optionally
        selecting `select`."""
        try:
            keys = loaded_sample_keys()
            # Keep a run-resolved built-in sample selectable even though
            # it isn't a path-loaded dataset.
            if select and select in SAMPLES and select not in keys:
                keys = keys + [select]
            self._sample_menu.configure(values=(keys or [_NO_DATA_HINT]))
            if select and select in SAMPLES:
                self._sample_var.set(select)
        except Exception:
            pass

    def _browse_dataset(self):
        """Pick ANY cube file to run the currently-loaded model on
        (decouples the model run-dir from the dataset)."""
        p = filedialog.askopenfilename(
            title="Pick a dataset cube to run the model on",
            filetypes=[("Cube files", "*.prz *.npz *.npy *.h5 *.hdf5"),
                        ("All files", "*.*")])
        if not p:
            return
        scan_override = None
        if p.lower().endswith((".h5", ".hdf5")):
            # 3D Eiger/Dectris masters may not carry the scan grid.
            try:
                import h5py
                from data import (_h5_find_data_path, _h5_infer_scan_shape)
                with h5py.File(p, "r") as fh:
                    dpath, ndim = _h5_find_data_path(fh)
                    s = tuple(fh[dpath].shape)
                    if ndim == 3:
                        scan_override = _h5_infer_scan_shape(fh, s[0])
                        if scan_override is None:
                            from gui_app._dialogs import ask_scan_shape
                            scan_override = ask_scan_shape(
                                self, s[0], s[1], s[2])
                            if scan_override is None:
                                return
            except Exception as e:
                messagebox.showerror("dataset",
                    f"could not read h5 shape:\n{e}"); return
        try:
            from data import register_runtime_sample
            key = register_runtime_sample(
                p, scan_shape=(tuple(scan_override) if scan_override
                                else None))
        except Exception as e:
            messagebox.showerror("dataset",
                f"could not register cube:\n{e}"); return
        self._sync_sample_choices(select=key)
        self._status.configure(
            text=f"dataset → {key}   (now click Render to run the "
                  f"loaded model on it)")

    def _render_from_load(self):
        d = self._load_path_var.get().strip()
        if not d or not os.path.isdir(d):
            messagebox.showerror("Error", "Pick a run dir first.")
            return
        self.outdir = d
        self.sample = self._sample_var.get()
        self._refresh_from_latest_ckpt(prefer_best=True)

    def _refresh_from_latest_ckpt(self, prefer_best=False):
        if self.outdir is None:
            return
        ckpt = None; label = ""
        if prefer_best:
            cand = os.path.join(self.outdir, "best.pth")
            if os.path.exists(cand):
                ckpt = cand; label = "best"
        if ckpt is None:
            cks = list_ckpts(self.outdir)
            if cks:
                ep, p = cks[-1]; ckpt = p; label = f"ep{ep}"
        if ckpt is None:
            self._status.configure(
                text="no checkpoint found in this run dir.")
            return
        self._render_from_ckpt(ckpt, label=label)

    # ---- core inference + render (background thread) ----
    def _render_from_ckpt(self, ckpt_path, label=""):
        if self._busy:
            return
        self._busy = True
        self._status.configure(text=f"rendering {label} from {os.path.basename(ckpt_path)} …")
        # spin a worker thread so the GUI stays responsive
        threading.Thread(target=self._render_worker,
                          args=(ckpt_path, label), daemon=True).start()

    def _render_worker(self, ckpt_path, label):
        try:
            assigns, soft, K_act, K_orig_ids, scan_shape = self._infer(ckpt_path)
            avgs = self._compute_class_averages(assigns, soft, K_act)
            self.after(0, lambda: self._render_figure(assigns, K_act, scan_shape,
                                              avgs, label, ckpt_path))
        except Exception as e:
            self.after(0, lambda: self._render_failed(e))
        finally:
            self._busy = False

    def _infer(self, ckpt_path):
        """Run infer_scan on the linked sample. Returns dense-remapped
        assigns, soft_probs, K_active, K_original_ids, scan_shape."""
        import torch
        from dino_sr_contrastive_model import load_contrastive_checkpoint
        from contrastive_eval import infer_scan
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _, _ = load_contrastive_checkpoint(ckpt_path, device=device)
        model.eval()
        cfg = SAMPLES[self.sample]
        ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
        # use training-time mask values from run_summary.json if present
        mask_r, mask_cols, ccrop = 15, 45, 140; com = True
        rs = os.path.join(self.outdir, "run_summary.json")
        if os.path.exists(rs):
            try:
                js = json.load(open(rs))
                c = js.get("cfg", {})
                mask_r = int(c.get("center_mask_radius", mask_r))
                mask_cols = int(c.get("polar_mask_cols", mask_cols))
                ccrop = int(c.get("center_crop_size", ccrop))
                com = bool(c.get("com_centering", com))
            except Exception:
                pass
        inf = infer_scan(model, ds, device, dense_remap=True,
                          polar_size=192, polar_mask_cols=mask_cols,
                          center_crop_size=ccrop,
                          com_centering=com, center_mask_radius=mask_r,
                          eval_temp=0.06, batch_size=128)
        K_act = int(inf["soft_probs"].shape[1])
        return (inf["assigns"], inf["soft_probs"], K_act,
                inf.get("K_original_ids", []), cfg["scan_shape"])

    def _compute_class_averages(self, assigns, soft_probs, K_act, top_n=200):
        """Confidence-weighted class average per class (raw Cartesian, no
        mask). Returns list of (K_act,) numpy arrays."""
        import torch
        import torch.nn.functional as F
        from torchvision.transforms import v2 as T
        from torchvision.transforms import InterpolationMode
        cfg = SAMPLES[self.sample]
        ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
        H = 192
        cart_pre = T.Compose([
            T.CenterCrop(140),
            T.Resize(H, interpolation=InterpolationMode.BILINEAR, antialias=True),
        ])
        avgs = []
        for c in range(K_act):
            idx = np.where(assigns == c)[0]
            if idx.size == 0:
                avgs.append(np.zeros((H, H), dtype=np.float32)); continue
            sc = soft_probs[idx, c]
            top = idx[np.argsort(-sc)[:min(top_n, len(idx))]]
            patterns = np.stack([ds.get_raw(int(i)) for i in top], 0).astype(np.float32)
            w = sc[np.argsort(-sc)[:len(top)]].astype(np.float32)
            wavg = (patterns * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
            wavg_norm = np.clip(wavg / max(float(cfg["vmax"]), 1e-6), 0.0, 1.0)
            x = torch.from_numpy(wavg_norm).unsqueeze(0).unsqueeze(0).float()
            x = F.interpolate(x, size=(H, H), mode="bilinear",
                                align_corners=False)
            x_cart = cart_pre(x)[0, 0].cpu().numpy()
            avgs.append(x_cart)
        return avgs

    def _render_figure(self, assigns, K_act, scan_shape, avgs, label, ckpt_path):
        Ny, Nx = scan_shape
        cm_array = assigns.reshape(Ny, Nx)
        self._fig.clear()
        # layout: class map on top (3 rows), class averages grid below
        cols_grid = min(K_act, 8)
        rows_grid = (K_act + cols_grid - 1) // cols_grid
        gs = self._fig.add_gridspec(2, cols_grid,
                                      height_ratios=[1.4, 1.0 * rows_grid],
                                      hspace=0.30, wspace=0.10,
                                      left=0.04, right=0.98, top=0.94,
                                      bottom=0.04)
        ax_cm = self._fig.add_subplot(gs[0, :])
        cmap = _adaptive_cmap(K_act)
        norm = BoundaryNorm(np.arange(K_act + 1) - 0.5, K_act)
        im = ax_cm.imshow(cm_array, cmap=cmap, norm=norm,
                            aspect="equal", interpolation="nearest")
        counts = np.bincount(assigns, minlength=K_act).tolist()
        ax_cm.set_title(
            f"{self.sample} — class map ({label}, K_active = {K_act})  "
            f"counts = {counts}", fontsize=10)
        ax_cm.set_xticks([]); ax_cm.set_yticks([])
        cbar = self._fig.colorbar(im, ax=ax_cm, fraction=0.025, pad=0.01,
                                    ticks=list(range(K_act)))
        cbar.set_label("class id", fontsize=9)

        # class averages
        gs2 = self._fig.add_gridspec(rows_grid, cols_grid,
                                       hspace=0.30, wspace=0.08,
                                       left=0.04, right=0.98,
                                       top=0.49, bottom=0.04)
        for c in range(K_act):
            r = c // cols_grid; cc = c % cols_grid
            ax = self._fig.add_subplot(gs2[r, cc])
            avg = avgs[c]
            # display: percentile clip + log1p (no mask)
            ref = avg.flatten()
            if ref.size and ref.max() > 0:
                lo, hi = np.percentile(ref, 2), np.percentile(ref, 99.5)
                disp = np.log1p(np.clip(avg, lo, hi) - lo)
            else:
                disp = avg
            ax.imshow(disp, cmap="inferno", aspect="equal",
                       interpolation="nearest")
            ax.set_title(f"p{c}  N={counts[c]}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        self._canvas.draw_idle()
        self._status.configure(
            text=f"rendered {label}  ({os.path.basename(ckpt_path)})  "
                  f"K_active={K_act}, counts={counts}")
        # Auto-save a copy of the class map + averages next to the data.
        try:
            import re, assistant_io
            safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(label)).strip("_") \
                if label else "ckpt"
            assistant_io.gui_autosave(
                self, "classmap", self._fig,
                name=f"classmap_{safe}",
                summary=f"{self.sample} — class map ({label}), "
                        f"K_active={K_act}, counts={counts}, "
                        f"ckpt={os.path.basename(ckpt_path)}")
        except Exception:
            pass

    def _render_failed(self, e):
        self._status.configure(text=f"render failed: {e!r}")
