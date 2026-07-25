"""pre_panel.py -- Tab 1 (Pre-processing).

Lets the user browse a .prz cube and interactively choose vmax,
center_mask_radius, and center_crop_size by seeing the resulting
input the model would actually train on.

Display:
    Left  : raw frame (Cartesian, vmax-normalised)
    Right : after CenterCrop + (optional COM) + Resize, with a circular
            overlay marking the beam mask used by the model
"""
from __future__ import annotations
import os, sys, math, time, threading, atexit, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def simpledialog_int(parent, title, prompt, *, default=None,
                       min_=None, max_=None):
    """Small int-input prompt. Wraps tk.simpledialog.askinteger so the
    file using this can stay self-contained."""
    from tkinter import simpledialog as _sd
    kw = {}
    if default is not None: kw["initialvalue"] = default
    if min_ is not None: kw["minvalue"] = min_
    if max_ is not None: kw["maxvalue"] = max_
    return _sd.askinteger(title, prompt, parent=parent, **kw)


def _register_atexit_cleanup(path, cleanup_dir=None):
    """Schedule deletion of a temp NBED file (and optionally its dir)
    when the process exits.  Used when the user picks 'No' in the save
    modal — the centred cube is still usable for this session, but the
    temp file shouldn't linger after the GUI closes."""
    def _cleanup(p=path, d=cleanup_dir):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
        if d:
            try: os.rmdir(d)
            except Exception: pass
    atexit.register(_cleanup)

import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle

from data import rescale_like_vmax, register_runtime_sample
from gui_app.tooltip import add_help_button
from gui_app.nbed_center import (
    find_bf_center, recenter_to, nbed_center_cube)


def _open_lazy(path, scan_shape=None,
                  apply_dectris_corrections: bool = False):
    """Thin shim — delegates to data.open_lazy_cube so all panels share
    a single h5/.prz/.npy loader (and a single allow_pickle=True
    convention)."""
    from data import open_lazy_cube
    return open_lazy_cube(path, scan_shape=scan_shape,
                              apply_dectris_corrections=apply_dectris_corrections)


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    H, W = img.shape
    y0 = max(0, (H - size) // 2); x0 = max(0, (W - size) // 2)
    return img[y0:y0 + size, x0:x0 + size]


def _com_shift(img: np.ndarray, search_radius: int) -> np.ndarray:
    """Mirrors CenterOnCOM: shift so the COM within search_radius is at
    center."""
    H, W = img.shape
    cy, cx = H / 2.0, W / 2.0
    yy, xx = np.ogrid[:H, :W]
    bm = ((yy - cy) ** 2 + (xx - cx) ** 2) <= search_radius ** 2
    msk = img * bm
    s = msk.sum()
    if s <= 0:
        return img
    com_y = (yy * msk).sum() / s
    com_x = (xx * msk).sum() / s
    sy = int(round(cy - com_y)); sx = int(round(cx - com_x))
    out = np.roll(np.roll(img, sy, axis=0), sx, axis=1)
    return out


class PrePanel(ctk.CTkFrame):

    def __init__(self, master, on_state_change=None):
        super().__init__(master)
        self.on_state_change = on_state_change or (lambda *_: None)

        # state
        self.cube = None        # mmap (Ny, Nx, H, W)
        self.path = ""
        self.sample_key: "str | None" = None  # key into data.SAMPLES once loaded
        # NBED-centering state
        self._nbed_busy = False
        self._blur_busy = False
        self._bin_busy = False
        self._ellip_busy = False
        self._nbed_overlay_center: "tuple[float, float] | None" = None
        # Last-fit ellipse for the raw-pane overlay.  None if no fit
        # has been run yet for the current frame.  (cx, cy, axis_minor,
        # axis_major, theta_deg) in RAW-frame pixel coords.
        self._ellip_overlay: "tuple[float, float, float, float, float] | None" = None
        # Defaults will be auto-tuned to the loaded cube's frame size on
        # first load; provide initial scalars here.
        self._nbed_search_radius = ctk.IntVar(value=40)
        self._nbed_disk_radius   = ctk.IntVar(value=20)
        self._nbed_threshold     = ctk.DoubleVar(value=0.50)

        # Display defaults
        self.vmax = ctk.DoubleVar(value=2.0)
        # center_mask_radius (cart-frame disk mask) has been
        # DEPRECATED in the GUI — `polar_mask_cols` covers the same
        # role (mask the low-radius region before the model).  The
        # field is kept as an IntVar fixed at 0 so back-end calls
        # that still take this kwarg keep working without further
        # plumbing changes.  Old run_summary.json files with a
        # non-zero radius continue to be read correctly by the eval
        # and post-hoc paths.
        self.center_mask_radius = ctk.IntVar(value=0)
        self.center_crop_size = ctk.IntVar(value=140)
        # The single, canonical low-radius mask control.  Number of
        # leftmost (low-r) columns of the polar image zeroed BEFORE
        # the model sees them.  Pushed through get_pre_kwargs into
        # Training tab defaults.
        self.polar_mask_cols = ctk.IntVar(value=45)
        self.com = ctk.BooleanVar(value=True)
        self.log_stretch = ctk.BooleanVar(value=False)   # display only
        self.use_blur    = ctk.BooleanVar(value=False)   # display only
        self.blur_sigma  = ctk.DoubleVar(value=1.0)      # px in display frame
        # Ellipticity correction state.  ``ellip_mode`` selects the
        # auto-fit strategy: "BF disk edge" works for any 4D-STEM
        # dataset (the bright-field disk has an elliptical boundary
        # under lens/detector distortion); "Amorphous ring" uses a
        # weighted-moment fit on an annulus around the ring peak;
        # "Manual" disables auto-fit and just uses the slider values.
        self.ellip_mode  = ctk.StringVar(value="BF disk edge")
        self.ellip_ab    = ctk.DoubleVar(value=1.000)    # major/minor ratio
        self.ellip_theta = ctk.DoubleVar(value=0.0)      # deg
        self.idx = ctk.IntVar(value=0)
        self.cmap = ctk.StringVar(value="inferno")

        self._build()

    # ------------------------------------------------------------------
    # Effective beam-mask radius (cart-frame, 192-px units).
    #
    # The standalone slider is gone (polar_mask_cols is the canonical
    # control), but the TRAINING pipeline still uses center_mask_radius
    # for two things:
    #   (a) CenterMask before PolarTransform — REDUNDANT with
    #       polar_mask_cols, so technically fine at 0
    #   (b) COM-centering search radius = com_search_radius_factor *
    #       center_mask_radius — REQUIRES a non-zero value or COM
    #       silently no-ops, leaving patterns mis-aligned and the
    #       polar profiles inconsistent across the scan (=> the model
    #       underclusters because every input looks like noise).
    #
    # So we ALWAYS derive a sensible mask radius from polar_mask_cols
    # using the PolarTransform's radius-to-column relation (cart_r ≈
    # polar_col * 96 / 192 = polar_col / 2).  This keeps the COM step
    # functional and matches the OLD default exactly when
    # polar_mask_cols=45 (→ cart_r=22, close to the legacy 15).
    # ------------------------------------------------------------------
    def _effective_center_mask_radius(self) -> int:
        try:
            pmc = int(self.polar_mask_cols.get())
        except Exception:
            pmc = 0
        return max(1, pmc // 2)

    # state pulled by other tabs
    def get_pre_kwargs(self) -> dict:
        return dict(
            center_mask_radius=self._effective_center_mask_radius(),
            center_crop_size=int(self.center_crop_size.get()),
            polar_mask_cols=int(self.polar_mask_cols.get()),
            com_centering=bool(self.com.get()),
            vmax=float(self.vmax.get()),
        )

    def _load_params_to_model(self):
        """Push current crop / beam-mask / polar-mask / COM into the
        Training tab (its _sync_from_prepanel pulls from get_pre_kwargs)."""
        try:
            tp = getattr(self.winfo_toplevel(), "train", None)
            if tp is None or not hasattr(tp, "_sync_from_prepanel"):
                self._load_params_status.configure(
                    text="(Training tab not available)"); return
            tp._sync_from_prepanel()
            # Read back what the Training tab ACTUALLY holds now, so the
            # transfer is verifiable (and the derived beam mask is explained).
            def _gv(k):
                try:
                    return tp.var[k].get()
                except Exception:
                    return "?"
            self._load_params_status.configure(
                text=f"Training now uses  crop={_gv('center_crop_size')}, "
                     f"beam-mask r={_gv('center_mask_radius')} "
                     f"(= polar_mask//2), polar_mask={_gv('polar_mask_cols')}, "
                     f"COM={_gv('com_centering')}.  (vmax is display-only — "
                     f"training normalizes internally.)")
        except Exception as e:
            self._load_params_status.configure(text=f"failed: {e}")

    def get_loaded_path(self) -> str:
        return self.path

    def get_sample_key(self) -> "str | None":
        """Return the key under which this loaded cube is registered in
        data.SAMPLES (created by register_runtime_sample on load)."""
        return self.sample_key

    # ------------------------------------------------------------------
    def _build(self):
        # Top toolbar
        tb = ctk.CTkFrame(self)
        tb.pack(side="top", fill="x", padx=6, pady=(6, 4))
        ctk.CTkLabel(tb, text="Cube:").pack(side="left", padx=(6, 4))
        self._path_var = ctk.StringVar()
        ctk.CTkEntry(tb, textvariable=self._path_var, width=520).pack(
            side="left", padx=4)
        ctk.CTkButton(tb, text="Browse", width=90,
                       command=self._browse).pack(side="left", padx=2)
        ctk.CTkButton(tb, text="Load", width=70,
                       command=self._load).pack(side="left", padx=2)
        ctk.CTkButton(tb, text="Browse multi…", width=130,
                       command=self._browse_multi).pack(side="left",
                                                            padx=2)

        # main split: figure (left) + controls (right)
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        # figure — two panels:
        #   LEFT  (_ax_proc)  : cart view of the pre-processed input,
        #                        i.e. the 192x192 frame after CenterCrop
        #                        and optional COM-centering.  This is
        #                        what was on the right before.
        #   RIGHT (_ax_polar) : polar transform of the same frame with
        #                        polar_mask_cols applied — the actual
        #                        tensor the encoder receives.
        # The old raw-frame view has moved into the NBED preview popup
        # (it was only really used for clicking through scan positions
        # and the NBED-marker overlay, both still available below).
        fig_holder = ctk.CTkFrame(body)
        fig_holder.pack(side="left", fill="both", expand=True)
        self._fig = Figure(figsize=(10, 5), dpi=95, facecolor="#f4f4f4")
        self._ax_proc = self._fig.add_subplot(121)
        self._ax_polar = self._fig.add_subplot(122)
        # Legacy alias — code paths that touch _ax_raw will still find
        # the leftmost axis (now the cart-processed one).
        self._ax_raw = self._ax_proc
        for ax in (self._ax_proc, self._ax_polar):
            ax.set_xticks([]); ax.set_yticks([])
        self._fig.tight_layout()
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_holder)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._info = ctk.CTkLabel(fig_holder, text="(no cube loaded)",
                                    font=("Consolas", 10), anchor="w",
                                    justify="left")
        self._info.pack(fill="x", padx=4, pady=(2, 4))

        # control panel — scrollable so the whole NBED section (and
        # any future additions below it) stay reachable on small
        # windows / shorter screens.
        ctrl = ctk.CTkScrollableFrame(body, width=340)
        ctrl.pack(side="left", fill="y", padx=(8, 0))
        ctk.CTkLabel(ctrl, text="Pattern index",
                      font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=8, pady=(8, 0))
        self._idx_lbl = ctk.CTkLabel(ctrl, text="—")
        self._idx_lbl.pack(anchor="w", padx=8)
        ctk.CTkLabel(ctrl,
            text="(click slider or image, then ←/→ = ±1, "
                  "PgUp/PgDn = ±10)",
            font=("Segoe UI", 8), text_color=("#777", "#999")
            ).pack(anchor="w", padx=8)
        self._idx_slider = ctk.CTkSlider(ctrl, from_=0, to=1,
                                           command=self._on_idx)
        self._idx_slider.pack(fill="x", padx=8, pady=(0, 8))
        self._wire_idx_keys()

        self._row_slider(ctrl, "vmax", self.vmax, 0.1, 250.0, 0.1,
                          "Per-pattern intensity normalisation cap. "
                          "Pixel values above this clip to 1.0 in the "
                          "rescaled image. Lower vmax saturates more of "
                          "the central beam halo.\n\n"
                          "The slider covers 0.1-250. For raw-count data "
                          "(e.g. un-normalised detector counts) you can "
                          "TYPE a larger value in the box, up to 10000 - "
                          "the slider then just rests at its right end.",
                          entry_hi=1e4)
        self._row_slider(ctrl, "center crop (px, in 192 frame)",
                          self.center_crop_size, 32, 192, 1,
                          "Cartesian center-crop applied AFTER the cube "
                          "is resized to 192×192 (matches the training "
                          "pipeline order: LoadPRZ resize → CenterCrop "
                          "→ COM → Resize → Polar). 140 = central 73% "
                          "of the 192-frame; cropped result is then "
                          "blown back up to 192 for the polar transform.")
        # The OLD `beam-mask radius` slider has been removed: it
        # represented the same physical thing as polar_mask_cols (mask
        # the low-radius / direct-beam region) and confused users.
        # polar_mask_cols is the canonical control; the equivalent
        # beam-mask radius in 192-frame cart pixels is just
        # polar_mask_cols / 2.
        self._row_slider(ctrl, "polar_mask_cols (low-r mask in polar)",
                          self.polar_mask_cols, 0, 96, 1,
                          "Number of leftmost columns of the polar "
                          "image (= low-radius region) zeroed before "
                          "the model sees them.  This is the SINGLE "
                          "low-r mask control — it replaces the old "
                          "'beam-mask radius' slider, which did the "
                          "same thing in cart space.  In the right-"
                          "pane polar preview the masked columns "
                          "appear black.  Equivalent cart-frame beam "
                          "radius ≈ polar_mask_cols / 2 (so 45 here "
                          "= ~22 px disk in 192-frame).")

        # COM centering
        com_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        com_row.pack(fill="x", padx=8, pady=4)
        ctk.CTkCheckBox(com_row, text="COM-center the direct beam",
                         variable=self.com,
                         command=self._refresh).pack(side="left")
        h = add_help_button(com_row,
            "Before resizing, find the center of mass of intensity "
            "within 2 x beam-mask radius and shift so the direct beam "
            "is centered. Removes residual probe drift.")
        h.pack(side="left", padx=(6, 0))

        # Log-stretch toggle for the display panes only — does NOT
        # change what the model sees during training.
        log_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        log_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkCheckBox(log_row, text="log-stretch",
                          variable=self.log_stretch,
                          command=self._propagate_filters
                          ).pack(side="left")
        h_log = add_help_button(log_row,
            "Log stretch: applies log1p(I × 50) / log1p(50) AFTER vmax "
            "normalisation, then forwards the result to every "
            "downstream tab (Training, NMF, Post-hoc, DINO+cluster, …) "
            "via the LoadPRZ pipeline.  Useful for compressing dynamic "
            "range so weak rings / Bragg peaks contribute more to the "
            "model's loss.  Stored on the active sample's SAMPLES entry "
            "— toggle live.")
        h_log.pack(side="left", padx=(6, 0))

        # Gaussian blur — display-only checkbox + permanent-apply button.
        blur_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        blur_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkCheckBox(blur_row, text="Gaussian blur",
                          variable=self.use_blur,
                          command=self._propagate_filters
                          ).pack(side="left")
        ctk.CTkLabel(blur_row, text="σ (px):").pack(side="left",
                                                        padx=(8, 2))
        e_blur = ctk.CTkEntry(blur_row, textvariable=self.blur_sigma,
                                 width=60)
        e_blur.pack(side="left", padx=2)
        e_blur.bind("<Return>", lambda _e: self._propagate_filters())
        h_blur = add_help_button(blur_row,
            "Gaussian blur σ (in 192-px model-input pixels) applied "
            "in LoadPRZ after vmax normalisation.  Toggle live — "
            "every downstream tab that re-opens this sample sees "
            "the blurred frames.  Does NOT modify the cube file on "
            "disk; if you want a permanent baked-in copy (faster "
            "downstream reads), click 'Apply blur to dataset…' below.")
        h_blur.pack(side="left", padx=(6, 0))
        blur_apply_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        blur_apply_row.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkButton(blur_apply_row,
            text="Apply blur to dataset…  (write new .cube.npy)",
            width=320,
            command=self._blur_apply_all).pack(side="left")
        self._blur_status = ctk.CTkLabel(ctrl, text="",
            font=("Consolas", 9), justify="left",
            text_color=("#444", "#aaa"))
        self._blur_status.pack(anchor="w", padx=10, pady=(0, 4))

        # Push current crop / beam-mask / polar-mask / COM into the Training tab.
        load_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        load_row.pack(fill="x", padx=8, pady=(2, 4))
        ctk.CTkButton(load_row,
            text="Load parameters to model",
            width=320,
            command=self._load_params_to_model).pack(side="left")
        self._load_params_status = ctk.CTkLabel(ctrl, text="",
            font=("Consolas", 9), text_color=("#444", "#aaa"))
        self._load_params_status.pack(anchor="w", padx=10, pady=(0, 4))

        # ---- Binning section -----------------------------------------
        # Down-sample the dataset two ways:
        #   real-space : average every n×n block of scan positions
        #                (fewer, higher-SNR diffraction patterns).
        #   q-space    : average every n×n block of detector pixels
        #                (coarser reciprocal sampling per pattern).
        # Writes a brand-new .cube.npy and switches to it — the ORIGINAL
        # cube file is never modified.
        bin_box = ctk.CTkFrame(ctrl, border_width=1)
        bin_box.pack(fill="x", padx=6, pady=(8, 4))
        bt = ctk.CTkFrame(bin_box, fg_color="transparent")
        bt.pack(fill="x", padx=4, pady=(4, 0))
        ctk.CTkLabel(bt, text="Binning (down-sample → new dataset)",
                      font=("Segoe UI", 11, "bold")).pack(side="left")
        add_help_button(bt,
            "Real-space binning: every n×n block of scan positions is "
            "averaged into one diffraction pattern (scan grid shrinks "
            "by n, detector unchanged).\n\n"
            "Q-space binning: every n×n block of detector pixels is "
            "averaged within each pattern (detector shrinks by n, scan "
            "grid unchanged).\n\n"
            "A NEW .cube.npy is written and the GUI switches to it; the "
            "original cube file is untouched.").pack(side="left",
                                                     padx=(6, 0))
        bin_row = ctk.CTkFrame(bin_box, fg_color="transparent")
        bin_row.pack(fill="x", padx=8, pady=(4, 2))
        self._bin_mode = ctk.StringVar(value="real")
        ctk.CTkRadioButton(bin_row, text="real-space (n×n scan positions)",
                            variable=self._bin_mode, value="real"
                            ).pack(side="left", padx=(2, 10))
        ctk.CTkRadioButton(bin_row, text="q-space (n×n detector px)",
                            variable=self._bin_mode, value="q"
                            ).pack(side="left", padx=2)
        bin_row2 = ctk.CTkFrame(bin_box, fg_color="transparent")
        bin_row2.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(bin_row2, text="bin factor n:").pack(side="left",
                                                          padx=(2, 2))
        self._bin_factor = ctk.StringVar(value="2")
        ctk.CTkEntry(bin_row2, textvariable=self._bin_factor,
                     width=50).pack(side="left", padx=2)
        ctk.CTkButton(bin_row2,
            text="Apply binning →  (write new .cube.npy)",
            width=300, command=self._bin_apply_all).pack(side="left",
                                                         padx=8)
        self._bin_status = ctk.CTkLabel(bin_box, text="",
            font=("Consolas", 9), justify="left",
            text_color=("#444", "#aaa"))
        self._bin_status.pack(anchor="w", padx=10, pady=(0, 4))

        # Pair-labels section (semi-supervised, pre-train labels).
        # Lets the user click through random pairs and answer
        # "same / different / skip" before training starts. The
        # collected labels feed `pair_assignment_loss` if the Training
        # tab's λ_pair knob is non-zero.
        pair_box = ctk.CTkFrame(ctrl, border_width=1)
        pair_box.pack(fill="x", padx=6, pady=(8, 4))
        ph = ctk.CTkFrame(pair_box, fg_color="transparent")
        ph.pack(fill="x", padx=4, pady=(4, 0))
        ctk.CTkLabel(ph, text="Pair labelling (pre-train)",
                      font=("Segoe UI", 11, "bold")).pack(side="left")
        h = add_help_button(ph,
            "Optional: before training, label N random pattern pairs as "
            "same/different physical phase. Labels are saved to "
            "<basename>.pair_labels.json next to the cube and used by the "
            "training loop as an extra loss term (pair_assignment_loss).")
        h.pack(side="left", padx=(6, 0))
        pair_btns = ctk.CTkFrame(pair_box, fg_color="transparent")
        pair_btns.pack(fill="x", padx=4, pady=(4, 2))
        ctk.CTkButton(pair_btns, text="Label pairs…",
                       width=120,
                       command=self._open_pair_labeler
                       ).pack(side="left", padx=2)
        ctk.CTkButton(pair_btns, text="Review labels…",
                       width=120,
                       command=self._open_pair_reviewer
                       ).pack(side="left", padx=2)
        ctk.CTkButton(pair_btns, text="Import labels…",
                       width=120,
                       command=self._import_pair_labels
                       ).pack(side="left", padx=2)
        self._pair_label_status = ctk.CTkLabel(pair_box, text="",
            font=("Consolas", 9), wraplength=290, justify="left",
            text_color=("#444", "#aaa"))
        self._pair_label_status.pack(fill="x", padx=8, pady=(0, 6))

        # NBED-centering section (BF-disk threshold-centroid) — runs as
        # a one-shot pre-processing step on the FULL cube, not as a
        # per-batch transform.
        nbed_box = ctk.CTkFrame(ctrl, border_width=1)
        nbed_box.pack(fill="x", padx=6, pady=(8, 4))
        title_row = ctk.CTkFrame(nbed_box, fg_color="transparent")
        title_row.pack(fill="x", padx=4, pady=(4, 0))
        ctk.CTkLabel(title_row, text="NBED centering",
                      font=("Segoe UI", 11, "bold")).pack(side="left")
        h = add_help_button(title_row,
            "Robust BF-disk centring for low-mag / NBED data, where "
            "COM fails due to saturated central beam, beam stops or "
            "diffuse haloes. Runs once over the whole cube; the "
            "result becomes the active dataset for this session and "
            "can optionally be saved as <basename>_nbed.cube.npy.")
        h.pack(side="left", padx=(6, 0))

        self._row_slider(nbed_box, "search radius (raw px)",
                          self._nbed_search_radius, 4, 256, 1,
                          "Coarse-search disk for the brightest pixel "
                          "near the centre. Should be larger than your "
                          "expected misalignment but smaller than the "
                          "distance to the first Bragg ring.")
        self._row_slider(nbed_box, "BF disk radius (raw px)",
                          self._nbed_disk_radius, 2, 128, 1,
                          "Refinement disk: after the bright spot is "
                          "found, the threshold-centroid is taken over "
                          "pixels within this radius. Set to roughly "
                          "the BF-disk radius in raw-frame pixels.")
        self._row_slider(nbed_box, "threshold fraction",
                          self._nbed_threshold, 0.10, 0.90, 0.01,
                          "Pixels above (threshold × local max) "
                          "contribute to the centroid. 0.5 is the "
                          "FWHM. Lower = include the disk skirt; "
                          "higher = use only the saturated core.")

        nbed_btns = ctk.CTkFrame(nbed_box, fg_color="transparent")
        nbed_btns.pack(fill="x", padx=4, pady=(2, 6))
        ctk.CTkButton(nbed_btns, text="Preview on this frame",
                       width=140, command=self._nbed_preview_one
                       ).pack(side="left", padx=2)
        ctk.CTkButton(nbed_btns, text="Center entire dataset",
                       width=160,
                       fg_color=("#2D7A2D", "#1F7A1F"),
                       command=self._nbed_center_all
                       ).pack(side="left", padx=2)
        self._nbed_status = ctk.CTkLabel(nbed_box, text="",
            font=("Consolas", 9), wraplength=290, justify="left",
            text_color=("#444", "#aaa"))
        self._nbed_status.pack(fill="x", padx=8, pady=(0, 6))

        # ---- Ellipticity correction ------------------------------
        # Auto-fit a 5-tuple (cx, cy, a, b, theta) on either the BF
        # disk edge, an amorphous ring, or by direct user input.
        # Apply writes a new <base>_ellipFix.cube.npy via cv2.warpAffine
        # and re-registers the sample.
        ellip_box = ctk.CTkFrame(ctrl, border_width=1)
        ellip_box.pack(fill="x", padx=6, pady=(8, 4))
        et = ctk.CTkFrame(ellip_box, fg_color="transparent")
        et.pack(fill="x", padx=4, pady=(4, 0))
        ctk.CTkLabel(et, text="Ellipticity correction",
                       font=("Segoe UI", 11, "bold")).pack(side="left")
        add_help_button(et,
            "Corrects elliptical distortion of diffraction features "
            "(BF disk / amorphous ring / Bragg lattice) — usually "
            "caused by lens astigmatism or detector geometry.\n\n"
            "Modes:\n"
            "  • BF disk edge — fits an ellipse to the Sobel-edge "
            "pixels of the bright-field disk.  Works for any 4D-STEM "
            "dataset (every cube has a BF disk).  Default.\n"
            "  • Amorphous ring — weighted-moment fit on an annulus "
            "around the ring peak.  Use when you have a clean "
            "amorphous halo.\n"
            "  • Manual — type a/b and θ directly; auto-fit disabled."
            ).pack(side="left", padx=4)

        em_row = ctk.CTkFrame(ellip_box, fg_color="transparent")
        em_row.pack(fill="x", padx=4, pady=(2, 2))
        ctk.CTkLabel(em_row, text="Fit on:").pack(side="left")
        ctk.CTkOptionMenu(em_row, variable=self.ellip_mode,
                            values=["BF disk edge",
                                       "Amorphous ring",
                                       "Manual"],
                            width=140,
                            command=lambda _v: self._refresh()
                            ).pack(side="left", padx=4)
        ctk.CTkButton(em_row, text="Auto-fit", width=90,
                        command=self._ellip_autofit
                        ).pack(side="left", padx=4)
        # a/b and theta sliders (live preview).
        self._row_slider(ellip_box, "a/b ratio",
                          self.ellip_ab, 0.85, 1.15, 0.001,
                          "Major-axis / minor-axis ratio of the "
                          "elliptical feature.  1.0 = circle (no "
                          "correction).  Live preview on the "
                          "processed pane.")
        self._row_slider(ellip_box, "θ (deg)",
                          self.ellip_theta, -90.0, 90.0, 0.5,
                          "Angle of the major axis in the detector "
                          "frame (0° = +x axis, +ve = CCW).")
        e_btns = ctk.CTkFrame(ellip_box, fg_color="transparent")
        e_btns.pack(fill="x", padx=4, pady=(2, 6))
        ctk.CTkButton(e_btns, text="Apply to dataset",
                        width=180,
                        fg_color=("#2D7A2D", "#1F7A1F"),
                        command=self._ellip_apply_all
                        ).pack(side="left", padx=2)
        ctk.CTkButton(e_btns, text="Reset (1.0, 0°)",
                        width=120,
                        command=self._ellip_reset
                        ).pack(side="left", padx=2)
        self._ellip_status = ctk.CTkLabel(ellip_box, text="",
            font=("Consolas", 9), wraplength=290, justify="left",
            text_color=("#444", "#aaa"))
        self._ellip_status.pack(fill="x", padx=8, pady=(0, 6))

        # cmap
        cmap_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        cmap_row.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(cmap_row, text="cmap").pack(side="left")
        ctk.CTkOptionMenu(cmap_row, variable=self.cmap, values=[
            "inferno", "viridis", "magma", "gray", "plasma", "cividis"],
            command=lambda _v: self._refresh()).pack(side="left", padx=4)

        # nav buttons
        nav = ctk.CTkFrame(ctrl, fg_color="transparent")
        nav.pack(fill="x", padx=8, pady=(8, 4))
        for label, dx in (("<<", -10), ("<", -1), (">", +1), (">>", +10)):
            ctk.CTkButton(nav, text=label, width=44,
                           command=lambda d=dx: self._step(d)).pack(
                side="left", padx=1)
        ctk.CTkButton(nav, text="Random", width=70,
                       command=self._random).pack(side="left", padx=4)

        ctk.CTkLabel(ctrl,
            text="\nThese values are passed through to the\n"
                  "training tab as defaults if you start a run.",
            font=("Segoe UI", 9), justify="left").pack(
                anchor="w", padx=8, pady=(8, 4))

    def _row_slider(self, parent, label, var, lo, hi, step, help_text,
                      entry_hi=None):
        """Slider with a two-way-bound editable text entry on the right.

        `hi` is the SLIDER's maximum.  `entry_hi` (default = hi) is the
        largest value the user may TYPE into the box -- set it higher when
        occasional extreme values are useful but would ruin the slider's
        resolution (e.g. vmax: slider to 250, typed up to 1e4).  A typed
        value above `hi` parks the slider handle at its right end while the
        real value is kept."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        head = ctk.CTkFrame(row, fg_color="transparent")
        head.pack(fill="x")
        ctk.CTkLabel(head, text=label).pack(side="left")
        h = add_help_button(head, help_text); h.pack(side="left", padx=(4, 0))

        is_int = isinstance(var, ctk.IntVar)
        ehi = float(hi if entry_hi is None else entry_hi)

        def _fmt(v):
            return str(int(v)) if is_int else f"{float(v):.2f}"

        entry_var = ctk.StringVar(value=_fmt(var.get()))
        entry = ctk.CTkEntry(head, textvariable=entry_var, width=72,
                              justify="right", font=("Consolas", 10))
        entry.pack(side="right", padx=(4, 0))

        s = ctk.CTkSlider(row, from_=lo, to=hi)
        s.set(var.get())
        s.pack(fill="x", pady=(2, 4))

        def _from_slider(v):
            v = float(v)
            if is_int:
                v = int(round(v))
            var.set(v)
            entry_var.set(_fmt(v))
            self._refresh()

        def _from_entry(_=None):
            try:
                v = float(entry_var.get())
            except ValueError:
                entry_var.set(_fmt(var.get()))
                return
            # Typed values may exceed the slider's range (up to entry_hi).
            v = max(float(lo), min(ehi, v))
            if is_int:
                v = int(round(v))
            var.set(v)
            s.set(v)          # parks at the slider's max if v > hi
            entry_var.set(_fmt(v))
            self._refresh()

        def _from_key(n_steps):
            """Nudge the value by n_steps × step, clamped to [lo, entry_hi]."""
            v = float(var.get()) + float(n_steps) * float(step)
            v = max(float(lo), min(ehi, v))
            if is_int:
                v = int(round(v))
            var.set(v)
            s.set(v)
            entry_var.set(_fmt(v))
            self._refresh()

        s.configure(command=_from_slider)
        entry.bind("<Return>", _from_entry)
        entry.bind("<FocusOut>", _from_entry)
        # Keyboard nav: click this slider, then ←/→ nudge by one step,
        # PgUp/PgDn by ten.  Focus is per-widget so only the slider
        # you clicked moves — not the pattern-index slider.
        self._bind_slider_arrows(s, _from_key)

    def _bind_slider_arrows(self, slider, on_nudge):
        """Wire arrow / PageUp / PageDown keys to a CTkSlider so it is
        keyboard-navigable once clicked.

        `on_nudge(n)` is called with n ∈ {±1, ±10} — the caller does
        the actual value change + clamp + redraw.  Each slider's
        internal canvas takes its OWN keyboard focus, so clicking a
        given slider routes the arrows to THAT slider only.
        """
        canvas = None
        for attr in ("_canvas", "canvas"):
            canvas = getattr(slider, attr, None)
            if canvas is not None:
                break
        if canvas is None:
            return

        def _mk(n):
            def _h(_e=None):
                on_nudge(n)
                return "break"   # suppress CTkSlider's own key handling
            return _h

        try:
            canvas.configure(takefocus=True)
        except Exception:
            pass
        for seq, n in (("<Left>", -1), ("<Down>", -1),
                        ("<Right>", +1), ("<Up>", +1),
                        ("<Prior>", -10), ("<Next>", +10)):
            try:
                canvas.bind(seq, _mk(n), add="+")
            except Exception:
                pass
        try:
            canvas.bind("<Button-1>",
                          lambda _e: canvas.focus_set(), add="+")
        except Exception:
            pass

    def _browse(self):
        p = filedialog.askopenfilename(
            filetypes=[("Cube files",
                          "*.prz *.npz *.npy *.h5 *.hdf5 *.dm4 *.dm3 *.raw"),
                        ("PRZ/NPZ", "*.prz *.npz"),
                        ("Numpy", "*.npy"),
                        ("HDF5", "*.h5 *.hdf5"),
                        ("Gatan DM", "*.dm4 *.dm3"),
                        ("EMPAD raw", "*.raw"),
                        ("All", "*.*")])
        if p:
            self._path_var.set(p); self._load()

    def _browse_multi(self):
        """Pick multiple cubes from any folders, register them as one
        multi-sample entry (Training routes through LoadPRZMulti).
        Uses an incremental add-one-at-a-time dialog so cubes living
        in different sub-folders work cleanly."""
        from gui_app._dialogs import add_cubes_dialog
        paths = add_cubes_dialog(self)
        if not paths:
            return
        try:
            from data import register_runtime_multi_sample
            try:
                vmax = float(self.vmax.get())
            except Exception:
                vmax = 2.0
            key = register_runtime_multi_sample(
                list(paths), vmax=vmax,
                center_mask_radius=self._effective_center_mask_radius())
        except Exception as e:
            messagebox.showerror("Load multi",
                f"failed:\n{e}"); return
        # Preview the FIRST cube.  NOTE: _load() re-registers cube-0 as a
        # SINGLE-cube sample and makes THAT the active sample (firing
        # on_state_change with the single key) — which is exactly why
        # "multi" training silently used only one cube.  After the
        # preview we restore the MULTI key as the active sample so
        # Training defaults to ALL cubes (LoadPRZMulti).
        self._path_var.set(paths[0])
        self._load()
        self.sample_key = key
        try:
            from data import SAMPLES as _S
            self.on_state_change("loaded", path=paths[0], sample_key=key,
                                  scan_shape=_S[key].get("scan_shape"))
        except Exception:
            pass
        messagebox.showinfo("Load multi",
            f"Registered {len(paths)} cubes as «{key}» and made it the "
            f"ACTIVE sample.\n\n"
            f"Training (DINO) will use ALL {len(paths)} cubes via "
            f"LoadPRZMulti — the Training tab is already set to «{key}».\n\n"
            f"(This preview pane shows only the first cube.)")

    def _load(self):
        p = self._path_var.get().strip()
        if not p or not os.path.exists(p):
            messagebox.showerror("Error", f"Not found: {p}")
            return
        # Gatan .dm4/.dm3: confirm we detected a 4D-STEM cube (and surface any
        # sanity warnings) before committing — these files can also hold EELS
        # spectrum images, image stacks, etc.
        self._dm4_calib = None
        if p.lower().endswith((".dm4", ".dm3")):
            try:
                from data import dm4_probe, dm4_calibration
                info = dm4_probe(p)
            except Exception as e:
                messagebox.showerror(
                    "DM file",
                    f"Could not read {os.path.basename(p)} as a 4D-STEM cube:\n"
                    f"{e}")
                return
            sh = info["shape4d"]
            msg = (f"Detected a 4D-STEM cube in {os.path.basename(p)}:\n\n"
                   f"   Ny × Nx × H × W = {sh[0]} × {sh[1]} × {sh[2]} × {sh[3]}\n"
                   f"   dtype = {info['dtype']}  (raw {info['raw_ndim']}D)\n")
            # Best-effort calibration straight from the dm metadata.
            try:
                cal = dm4_calibration(p)
            except Exception:
                cal = None
            if cal and (cal.get("real_nm_per_px") or cal.get("recip_nm_per_px")):
                self._dm4_calib = cal
                msg += f"\nCalibration (from dm metadata):\n   {cal['note']}\n"
                msg += "   → will be applied to the real/reciprocal fields.\n"
            elif cal and cal.get("note"):
                msg += f"\nCalibration: {cal['note']}\n"
            if info["warnings"]:
                msg += "\n⚠ " + "\n⚠ ".join(info["warnings"]) + "\n"
            msg += "\nLoad this as a 4D-STEM cube?"
            if not messagebox.askyesno("Confirm DM load", msg):
                return
        # EMPAD .raw: confirm the detected scan shape (the 2 metadata rows
        # per frame are cropped automatically -> 128x128 detector).
        if p.lower().endswith(".raw"):
            try:
                from data import empad_probe
                info = empad_probe(p)
            except Exception as e:
                messagebox.showerror(
                    "EMPAD raw",
                    f"Could not read {os.path.basename(p)} as an EMPAD "
                    f".raw:\n{e}\n\nExpected an EMPAD scan_x{{Nx}}_y{{Ny}}.raw "
                    f"with a sibling acquisition_*.xml.")
                return
            sh = info["shape4d"]
            msg = (f"Detected an EMPAD cube in {os.path.basename(p)}:\n\n"
                   f"   Ny × Nx × H × W = {sh[0]} × {sh[1]} × {sh[2]} × {sh[3]}\n"
                   f"   dtype = float32\n"
                   f"   {'(cropped 2 metadata rows/frame)' if info['has_metadata_rows'] else '(no metadata rows)'}\n")
            if info["warnings"]:
                msg += "\n⚠ " + "\n⚠ ".join(info["warnings"]) + "\n"
            msg += "\nLoad this as a 4D-STEM cube?"
            if not messagebox.askyesno("Confirm EMPAD load", msg):
                return
        # For 3D HDF5 masters (Eiger / Dectris), peek the dataset shape
        # and pop a scan-shape dialog. For 4D HDF5 / .prz / .npy, no
        # prompt is needed.
        scan_shape_override = None
        if p.lower().endswith((".h5", ".hdf5")):
            try:
                import h5py
                from data import (_h5_find_data_path,
                                      _h5_infer_scan_shape,
                                      _h5_dectris_external_data)
                with h5py.File(p, "r") as fh:
                    try:
                        dpath, ndim = _h5_find_data_path(fh)
                        s = tuple(fh[dpath].shape)
                    except ValueError:
                        # Dectris master case: stitch external links.
                        pairs = _h5_dectris_external_data(fh, p)
                        if not pairs:
                            raise
                        total = 0; H_ = W_ = None
                        for fp, dp in pairs:
                            with h5py.File(fp, "r") as gf:
                                sh = tuple(gf[dp].shape)
                                if len(sh) != 3: continue
                                total += sh[0]
                                H_, W_ = sh[1], sh[2]
                        s = (total, H_, W_); ndim = 3
                    if ndim == 3:
                        scan_shape_override = _h5_infer_scan_shape(
                            fh, s[0])
            except Exception as e:
                messagebox.showerror("Error",
                    f"HDF5 peek failed:\n{e}"); return
            # Only prompt if the master doesn't carry the scan grid.
            if ndim == 3 and scan_shape_override is None:
                from gui_app._dialogs import ask_scan_shape
                N, H, W = s
                scan_shape_override = ask_scan_shape(self, N, H, W)
                if scan_shape_override is None:
                    return
        # --- Peek shape/dtype cheaply (no frame load), estimate memory,
        #     and prompt to real-space bin before loading. ---
        shape4d = None; peek_dtype = None; lazy = True
        if p.lower().endswith((".h5", ".hdf5")):
            try:
                import h5py
                from data import (_h5_find_data_path,
                                      _h5_dectris_external_data)
                with h5py.File(p, "r") as fh:
                    try:
                        dpath, _nd = _h5_find_data_path(fh)
                        ds0 = fh[dpath]; peek_dtype = ds0.dtype
                        if ds0.ndim == 4:
                            shape4d = tuple(int(x) for x in ds0.shape)
                        else:
                            oy, ox = scan_shape_override
                            shape4d = (oy, ox, int(ds0.shape[1]),
                                       int(ds0.shape[2]))
                    except ValueError:
                        pairs = _h5_dectris_external_data(fh, p)
                        with h5py.File(pairs[0][0], "r") as gf:
                            d0 = gf[pairs[0][1]]; peek_dtype = d0.dtype
                            H_, W_ = int(d0.shape[1]), int(d0.shape[2])
                        oy, ox = scan_shape_override
                        shape4d = (oy, ox, H_, W_)
            except Exception as e:
                messagebox.showerror("Error", f"HDF5 peek failed:\n{e}")
                return
        else:
            try:
                from data import peek_cube_info
                shape4d, peek_dtype, lazy = peek_cube_info(p)
            except Exception as e:
                messagebox.showerror("Error",
                    f"Could not read file header:\n{e}")
                return
        Ny0, Nx0, H0, W0 = (int(x) for x in shape4d)
        try:
            itemsize = int(np.dtype(peek_dtype).itemsize)
        except Exception:
            itemsize = 2
        total_bytes = Ny0 * Nx0 * H0 * W0 * itemsize
        try:
            import psutil
            avail_bytes = int(psutil.virtual_memory().available)
        except Exception:
            avail_bytes = total_bytes * 100      # psutil missing → don't block
        pref = getattr(self, "_session_load_pref", None)
        if pref is not None:
            decision = dict(pref)
        else:
            from gui_app._dialogs import ask_load_options
            decision = ask_load_options(
                self, shape=(Ny0, Nx0, H0, W0), dtype=peek_dtype,
                total_bytes=total_bytes, avail_bytes=avail_bytes, lazy=lazy)
        if decision.get("action") == "cancel":
            return
        if decision.get("remember"):
            self._session_load_pref = {
                "action": decision["action"],
                "n": int(decision.get("n", 1)), "remember": True}
        load_path = p
        used_scan_override = scan_shape_override
        if (decision.get("action") == "bin"
                and int(decision.get("n", 1)) > 1):
            n = int(decision["n"])
            base, _ext = os.path.splitext(p)
            out = base + f"_bin{n}.cube.npy"
            if not os.path.exists(out):
                try:
                    src = _open_lazy(p, scan_shape=scan_shape_override)
                except Exception as e:
                    messagebox.showerror("Error", f"Load failed:\n{e}")
                    return
                if not self._run_binning(src, n, out):
                    return
            load_path = out
            used_scan_override = None             # binned .cube.npy is 4D
        try:
            self.cube = _open_lazy(load_path, scan_shape=used_scan_override)
        except Exception as e:
            messagebox.showerror("Error", f"Load failed:\n{e}")
            return
        p = load_path
        # Diagnostic: peek the central frame so the user knows what
        # vmax order-of-magnitude to set (Eiger raw counts can range
        # from ~10 to ~10^4 depending on detector / dose).
        try:
            Ny_, Nx_ = self.cube.shape[:2]
            mid = self.cube[Ny_ // 2, Nx_ // 2]
            arr = np.asarray(mid, dtype=np.float32)
            print(f"[load] {os.path.basename(p)}: frame stats "
                   f"min={float(arr.min()):.2f}  "
                   f"max={float(arr.max()):.2f}  "
                   f"mean={float(arr.mean()):.2f}  "
                   f"p99={float(np.percentile(arr, 99)):.2f}  "
                   f"  → suggest vmax ≈ p99",
                   flush=True)
        except Exception:
            pass
        Ny, Nx, H, W = self.cube.shape
        self.path = p
        self.idx.set(Ny * Nx // 2)
        self._idx_slider.configure(from_=0, to=Ny * Nx - 1)
        self._idx_slider.set(self.idx.get())
        # Auto-tune NBED defaults to this cube's frame size if the user
        # hasn't moved the sliders away from the previous defaults yet.
        suggested_search = max(8, min(H, W) // 4)
        suggested_disk   = max(4, min(H, W) // 12)
        try:
            self._nbed_search_radius.set(int(suggested_search))
            self._nbed_disk_radius.set(int(suggested_disk))
        except Exception:
            pass
        self._nbed_overlay_center = None  # reset any stale overlay
        # Register this cube as a runtime sample so the Training / Eval /
        # Post-hoc / ACOM tabs can address it by key.
        try:
            self.sample_key = register_runtime_sample(
                p, scan_shape=(Ny, Nx),
                vmax=float(self.vmax.get()),
                center_mask_radius=self._effective_center_mask_radius(),
                blur_sigma=(float(self.blur_sigma.get())
                             if bool(self.use_blur.get()) else 0.0),
                log_stretch=bool(self.log_stretch.get()))
        except Exception as e:
            messagebox.showwarning(
                "Runtime-sample register",
                f"Could not register loaded cube as a sample:\n{e}\n\n"
                "Training will only see built-in samples.")
            self.sample_key = None
        self._refresh()
        # Update the pair-labels counter so the user can see whether
        # they already have labels for this cube before training.
        try:
            self._refresh_pair_label_count()
        except Exception:
            pass
        self.on_state_change("loaded", path=p,
                              sample_key=self.sample_key,
                              scan_shape=(Ny, Nx),
                              frame_shape=(H, W),
                              calib=getattr(self, "_dm4_calib", None))

    def _run_binning(self, src, n, out):
        """Stream a real-space n×n bin of `src` → `out` (.cube.npy) with a
        modal progress bar.  Returns True on success."""
        import tkinter as tk
        import data
        win = tk.Toplevel(self)
        win.title("Binning…")
        win.geometry("440x140")
        try:
            win.transient(self.winfo_toplevel())
        except Exception:
            pass
        win.grab_set()
        ctk.CTkLabel(win, text=f"Real-space binning n={n} — writing a "
                     f"smaller cube to disk.\nThis runs once; please wait.",
                     justify="left").pack(padx=12, pady=(12, 6))
        bar = ctk.CTkProgressBar(win, width=380); bar.pack(padx=12, pady=4)
        bar.set(0)
        pct = ctk.CTkLabel(win, text="0%", font=("Consolas", 10))
        pct.pack()
        err = {"e": None}
        def cb(done, total, stage):
            try:
                bar.set(done / max(total, 1))
                pct.configure(text=f"{100 * done // max(total, 1)}%  "
                                   f"({done}/{total})")
                win.update()
            except Exception:
                pass
        try:
            data.bin_realspace_to_npy(src, n, out, progress=cb)
        except Exception as e:
            err["e"] = e
        try:
            win.grab_release(); win.destroy()
        except Exception:
            pass
        if err["e"] is not None:
            messagebox.showerror("Binning failed", repr(err["e"]))
            try:
                os.remove(out)
            except Exception:
                pass
            return False
        return True

    def _on_idx(self, v):
        self.idx.set(int(round(float(v))))
        self._refresh()

    def _step(self, d):
        if self.cube is None: return
        self.idx.set(max(0, min(self.cube.shape[0] * self.cube.shape[1] - 1,
                                  self.idx.get() + d)))
        self._idx_slider.set(self.idx.get())
        self._refresh()

    def _wire_idx_keys(self):
        """Keyboard frame-stepping.  Click the index slider OR the
        figure, then:  ←/↓ = −1 frame,  →/↑ = +1,  PgUp/PgDn = ±10,
        Home/End = first/last.  Bindings are attached to both the
        slider's internal canvas and the matplotlib widget, and a
        click on either gives it keyboard focus."""
        def _stepper(d):
            def _h(_e=None):
                self._step(d)
                return "break"   # stop the widget's own arrow handling
            return _h

        def _jump(to_end):
            def _h(_e=None):
                if self.cube is None:
                    return "break"
                n = self.cube.shape[0] * self.cube.shape[1]
                self.idx.set(n - 1 if to_end else 0)
                self._idx_slider.set(self.idx.get())
                self._refresh()
                return "break"
            return _h

        targets = []
        # CTkSlider routes key/focus through an internal tk Canvas.
        for attr in ("_canvas", "canvas"):
            w = getattr(self._idx_slider, attr, None)
            if w is not None:
                targets.append(w)
                break
        # The matplotlib figure widget — arrowing while looking at the
        # diffraction image is the most natural interaction.
        try:
            targets.append(self._canvas.get_tk_widget())
        except Exception:
            pass

        for w in targets:
            try:
                w.configure(takefocus=True)
            except Exception:
                pass
            for seq, handler in (
                ("<Left>",  _stepper(-1)), ("<Down>",  _stepper(-1)),
                ("<Right>", _stepper(+1)), ("<Up>",    _stepper(+1)),
                ("<Prior>", _stepper(-10)),   # PageUp
                ("<Next>",  _stepper(+10)),   # PageDown
                ("<Home>",  _jump(False)),
                ("<End>",   _jump(True)),
            ):
                try:
                    w.bind(seq, handler, add="+")
                except Exception:
                    pass
            # Click → take keyboard focus so the arrows go here.
            try:
                w.bind("<Button-1>",
                       lambda _e, ww=w: ww.focus_set(), add="+")
            except Exception:
                pass

    def _random(self):
        if self.cube is None: return
        import random
        n = self.cube.shape[0] * self.cube.shape[1]
        self.idx.set(random.randrange(n))
        self._idx_slider.set(self.idx.get())
        self._refresh()

    # ------------------------------------------------------------------
    # NBED centring
    # ------------------------------------------------------------------
    def _nbed_preview_one(self):
        """Run BF-disk centring on the currently-displayed pattern only,
        store the centre, and re-render with a marker overlay so the
        user can check whether the parameters are good before running
        the full cube."""
        if self.cube is None:
            messagebox.showinfo("cube", "Load a cube first."); return
        try:
            Ny, Nx, H, W = self.cube.shape
            idx = int(self.idx.get())
            y, x = divmod(idx, Nx)
            raw = np.asarray(self.cube[y, x]).astype(np.float32)
            cy, cx = find_bf_center(
                raw,
                search_radius=int(self._nbed_search_radius.get()),
                disk_radius=int(self._nbed_disk_radius.get()),
                threshold_frac=float(self._nbed_threshold.get()))
        except Exception as e:
            messagebox.showerror("NBED preview failed", repr(e))
            return
        # store overlay for _refresh to draw
        self._nbed_overlay_center = (float(cy), float(cx))
        cy0 = (H - 1) / 2.0; cx0 = (W - 1) / 2.0
        dy = cy - cy0; dx = cx - cx0
        self._nbed_status.configure(
            text=f"BF centre @ idx={idx}:  (cy={cy:.1f}, cx={cx:.1f})\n"
                  f"offset from geom centre:  Δy={dy:+.1f}px, "
                  f"Δx={dx:+.1f}px  → {np.hypot(dy, dx):.2f}px shift")
        self._refresh()

    # ==================================================================
    # Ellipticity correction
    # ==================================================================
    def _ellip_warp_matrix(self, H: int, W: int):
        """Build the 2×3 affine matrix that maps elliptical features
        of ratio ``a/b`` at angle ``theta`` to circular ones, centred
        on the image centre (H/2, W/2).  Returns None if no
        correction is needed (ratio ≈ 1.0).
        """
        ab = float(self.ellip_ab.get())
        if abs(ab - 1.0) < 1e-4:
            return None
        th = float(np.deg2rad(float(self.ellip_theta.get())))
        c, s = float(np.cos(th)), float(np.sin(th))
        # cv2.warpAffine convention: dst[p] = src[M·p].  To turn a
        # source ellipse of semi-axes (a, b) with a/b = ab into a dst
        # circle of radius b, M must SCALE dst coords along the major
        # axis by `ab` (so each dst pixel at radius r along major
        # samples the src at radius ab·r — i.e. further from centre,
        # which is where the elongated source feature actually lies).
        # The previous code used 1/ab and produced the inverse warp
        # (made the feature MORE elliptical instead of less).
        scale_x, scale_y = ab, 1.0
        # Rotation matrix R(theta) and its transpose R(-theta).
        # The warp is R · S · R^T applied to the image centred at (cx, cy).
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        S = np.diag([scale_x, scale_y])
        A = R @ S @ R.T
        cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
        # cv2.warpAffine maps dst[y,x] = src[A · ([x,y] - centre) + centre]
        # So the affine matrix M is [A | (centre - A·centre)].
        t = np.array([cx, cy]) - A @ np.array([cx, cy])
        M = np.zeros((2, 3), dtype=np.float64)
        M[:2, :2] = A
        M[:, 2] = t
        return M

    def _ellip_warp_image(self, img: np.ndarray) -> np.ndarray:
        """Apply the current ellipticity correction to a 2D image.
        No-op when ratio is 1."""
        M = self._ellip_warp_matrix(*img.shape)
        if M is None:
            return img
        import cv2
        return cv2.warpAffine(img.astype(np.float32),
                                  M.astype(np.float32),
                                  (img.shape[1], img.shape[0]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0.0)

    def _ellip_autofit(self):
        """Run the per-mode auto-fit on the current frame (or sum-
        PACBED for robustness)."""
        if self.cube is None:
            messagebox.showinfo("ellipticity",
                "Load a cube first."); return
        mode = self.ellip_mode.get()
        if mode == "Manual":
            self._ellip_status.configure(
                text="manual mode: type a/b and θ above, then Apply.")
            return
        try:
            # Build a high-SNR target image — sum-PACBED across a
            # small random subset is robust to one bad pattern.
            ref = self._ellip_make_reference_image()
            if mode.startswith("BF"):
                fit = self._ellip_fit_bf_disk_edge(ref)
            else:
                fit = self._ellip_fit_amorphous_ring(ref)
        except Exception as e:
            messagebox.showerror("ellipticity fit failed", repr(e))
            self._ellip_status.configure(
                text=f"fit failed: {e}")
            self._ellip_overlay = None
            return
        if fit is None:
            self._ellip_status.configure(
                text="auto-fit returned no result — try the other mode "
                      "or set manually.")
            return
        (cy, cx, axis_minor, axis_major, theta_deg) = fit
        # cv2.fitEllipse returns angle in [0, 180); fold to [-90, 90]
        # so our θ slider stays well-behaved.
        if theta_deg > 90.0:
            theta_deg -= 180.0
        ab = float(axis_major) / max(float(axis_minor), 1e-6)
        # Sanity clamp — wildly-off fits land outside the slider range.
        if ab > 1.15 or ab < 0.85:
            self._ellip_status.configure(
                text=f"fit gave a/b={ab:.3f}, θ={theta_deg:.2f}° — "
                      f"outside ±15% range; clamped.  Try a different "
                      f"mode if this looks wrong.")
            ab = max(0.85, min(1.15, ab))
        self.ellip_ab.set(round(ab, 3))
        self.ellip_theta.set(round(theta_deg, 2))
        self._ellip_overlay = (float(cx), float(cy),
                                  float(axis_minor),
                                  float(axis_major),
                                  float(theta_deg))
        self._ellip_status.configure(
            text=f"fit @ {mode}:  a/b = {ab:.4f}   "
                  f"θ = {theta_deg:+.2f}°   "
                  f"centre = ({cx:.1f}, {cy:.1f}) raw px")
        self._refresh()

    def _ellip_make_reference_image(self, n_samples: int = 32):
        """Sum-PACBED-ish reference: mean of `n_samples` random
        patterns across the scan, used for the auto-fit.  Single
        patterns are noisy and the BF disk edge wobbles; averaging
        a few patterns gives a robust edge."""
        Ny, Nx, H, W = self.cube.shape
        N = Ny * Nx
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=min(int(n_samples), N), replace=False)
        acc = np.zeros((H, W), dtype=np.float64)
        for i in idx:
            y, x = divmod(int(i), Nx)
            acc += np.asarray(self.cube[y, x], dtype=np.float64)
        return (acc / float(len(idx))).astype(np.float32)

    def _ellip_fit_bf_disk_edge(self, img: np.ndarray):
        """Sobel edges → cv2.fitEllipse on the bright edge pixels.
        Returns (cy, cx, axis_minor, axis_major, theta_deg) or None.
        """
        import cv2
        try:
            from scipy.ndimage import sobel as _sobel
        except Exception:
            _sobel = None
        if _sobel is not None:
            gx = _sobel(img, axis=1).astype(np.float32)
            gy = _sobel(img, axis=0).astype(np.float32)
        else:
            gx = cv2.Sobel(img.astype(np.float32),
                              cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img.astype(np.float32),
                              cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        # Adaptive threshold: top-decile of edge magnitudes.  This is
        # the BF disk boundary for a typical 4D-STEM pattern.
        thr = np.percentile(mag, 90.0)
        ys, xs = np.where(mag > thr)
        if len(xs) < 32:
            return None
        pts = np.column_stack([xs.astype(np.float32),
                                  ys.astype(np.float32)])
        (cx_cy), sizes, angle_deg = cv2.fitEllipse(pts)
        cx, cy = float(cx_cy[0]), float(cx_cy[1])
        # cv2.fitEllipse returns sizes as (size_along_angle,
        # size_perpendicular_to_angle) — NEITHER is guaranteed to be
        # the major axis.  Sort explicitly, and recover the major-
        # axis angle: if the FIRST size is the larger one the angle
        # is already for the major axis; otherwise add 90°.
        s_along, s_perp = float(sizes[0]), float(sizes[1])
        if s_along >= s_perp:
            ax_major, ax_minor = s_along, s_perp
            theta_major = float(angle_deg)
        else:
            ax_major, ax_minor = s_perp, s_along
            theta_major = float(angle_deg) + 90.0
        # cv2 returns full-axis lengths (diameter) — divide by 2 for
        # consistency with semi-axis convention.
        return (cy, cx,
                ax_minor / 2.0,
                ax_major / 2.0,
                theta_major)

    def _ellip_fit_amorphous_ring(self, img: np.ndarray,
                                       r_inner_frac: float = 0.25,
                                       r_outer_frac: float = 0.85):
        """Weighted-moment ellipse fit on an annulus.  Locates the
        radial-profile peak inside [r_inner_frac, r_outer_frac] of
        the image half-size, builds an annulus around it, then
        computes the 2nd moments of intensity inside the annulus
        and diagonalises the covariance matrix.
        """
        H, W = img.shape
        cy0, cx0 = (H - 1) * 0.5, (W - 1) * 0.5
        # Full (H, W) grids so the boolean mask below indexes
        # cleanly (np.ogrid gives (H,1) + (1,W) which can't be
        # mask-indexed with an (H,W) bool).
        yy, xx = np.mgrid[:H, :W].astype(np.float64)
        dy = yy - cy0; dx = xx - cx0
        r = np.sqrt(dx * dx + dy * dy)
        r_max = min(H, W) * 0.5
        # Coarse 1D radial profile → find ring peak.
        r_bins = np.linspace(r_inner_frac * r_max,
                                r_outer_frac * r_max, 80)
        prof = np.zeros(len(r_bins) - 1, dtype=np.float64)
        for i in range(len(r_bins) - 1):
            m = (r >= r_bins[i]) & (r < r_bins[i + 1])
            if m.any():
                prof[i] = float(img[m].mean())
        if prof.max() <= 0:
            return None
        peak_i = int(np.argmax(prof))
        r_peak = 0.5 * (r_bins[peak_i] + r_bins[peak_i + 1])
        ring_half_width = max(3.0, (r_bins[1] - r_bins[0]) * 2)
        mask = (r > r_peak - ring_half_width) & \
                (r < r_peak + ring_half_width)
        if mask.sum() < 64:
            return None
        w = img[mask].astype(np.float64).clip(min=0.0)
        wsum = float(w.sum())
        if wsum <= 0:
            return None
        rx = (dx[mask].astype(np.float64) - 0).ravel()
        ry = (dy[mask].astype(np.float64) - 0).ravel()
        # 2nd-moment covariance of the ring.
        Cxx = float((w * rx * rx).sum() / wsum)
        Cyy = float((w * ry * ry).sum() / wsum)
        Cxy = float((w * rx * ry).sum() / wsum)
        C = np.array([[Cxx, Cxy], [Cxy, Cyy]])
        eigvals, eigvecs = np.linalg.eigh(C)
        # eigvals sorted ascending; semi-axes ∝ sqrt(eigvals).
        axis_minor = float(np.sqrt(max(eigvals[0], 1e-12)))
        axis_major = float(np.sqrt(max(eigvals[1], 1e-12)))
        v_major = eigvecs[:, 1]   # column for the larger eigenvalue
        theta_deg = float(np.degrees(np.arctan2(v_major[1],
                                                       v_major[0])))
        return (cy0, cx0, axis_minor, axis_major, theta_deg)

    def _ellip_reset(self):
        self.ellip_ab.set(1.0)
        self.ellip_theta.set(0.0)
        self._ellip_overlay = None
        self._ellip_status.configure(text="reset to identity.")
        self._refresh()

    def _ellip_apply_all(self):
        """Bake the current ellipticity correction into a new
        <base>_ellipFix.cube.npy and swap to it.  Mirrors the NBED
        workflow (worker thread, save/keep modal, runtime
        re-registration).
        """
        if self.cube is None:
            messagebox.showinfo("cube", "Load a cube first."); return
        if self._ellip_busy:
            messagebox.showinfo("ellipticity",
                "Ellipticity correction is already running."); return
        if self._nbed_busy or self._blur_busy:
            messagebox.showinfo("ellipticity",
                "Another bake (NBED / blur) is in progress."); return
        try:
            ab = float(self.ellip_ab.get())
            th = float(self.ellip_theta.get())
        except Exception:
            messagebox.showerror("ellipticity",
                "Bad a/b or θ."); return
        if abs(ab - 1.0) < 1e-4:
            messagebox.showinfo("ellipticity",
                "a/b = 1.0 (identity); nothing to apply.  Auto-fit "
                "or move the sliders first."); return
        Ny, Nx, H, W = self.cube.shape
        if not messagebox.askyesno("Apply ellipticity correction",
                f"Re-warp every diffraction pattern in this "
                f"{Ny}×{Nx} scan with a/b = {ab:.4f}, θ = {th:+.2f}°?"
                f"\n\nWrites a new .cube.npy alongside the original. "
                f"You'll be asked whether to save permanently after."):
            return
        self._ellip_busy = True
        self._ellip_status.configure(text="ellipFix: starting…")
        threading.Thread(target=self._ellip_worker,
                            args=(ab, th), daemon=True).start()

    def _ellip_worker(self, ab: float, th_deg: float):
        try:
            import cv2
            base = os.path.splitext(self.path)[0]
            base = base[:-5] if base.endswith(".cube") else base
            tag = f"_ellip_ab{ab:.4f}_th{th_deg:+.2f}".replace("+", "p"
                ).replace("-", "m").replace(".", "p")
            new_basename = os.path.basename(base) + tag
            tmp_dir = tempfile.mkdtemp(prefix="dinosr_ellip_")
            tmp_path = os.path.join(tmp_dir,
                                       new_basename + ".cube.npy")
            Ny, Nx, H, W = self.cube.shape
            # Always float32 — the affine warp (INTER_LINEAR) produces
            # fractional values that an integer source dtype (e.g. uint16
            # dm4) would truncate, zeroing out low-count data.
            out = np.lib.format.open_memmap(
                tmp_path, mode="w+",
                dtype=np.float32,
                shape=(Ny, Nx, H, W))
            # Pre-compute the affine matrix once (constant for whole cube).
            M = self._ellip_warp_matrix(H, W)
            if M is None:
                raise RuntimeError("identity warp; nothing to do")
            M32 = M.astype(np.float32)
            t0 = time.time()
            for y in range(Ny):
                row = np.asarray(self.cube[y]).astype(np.float32)
                for x in range(Nx):
                    out[y, x] = cv2.warpAffine(
                        row[x], M32, (W, H),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0)
                if (y + 1) % max(1, Ny // 40) == 0:
                    dt = time.time() - t0
                    eta = dt * (Ny - y - 1) / max(y + 1, 1)
                    self.after(0, lambda y=y, eta=eta:
                        self._ellip_status.configure(
                            text=f"ellipFix: {y+1}/{Ny} rows "
                                  f"({100*(y+1)/Ny:.0f}%)  "
                                  f"eta {eta:.0f}s"))
            out.flush(); del out
            self.after(0, lambda: self._ellip_finish(
                tmp_path, ab, th_deg, base, new_basename))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "ellipticity failed", err))
            self.after(0, lambda: self._ellip_status.configure(
                text=f"failed: {err}"))
        finally:
            self._ellip_busy = False

    def _ellip_finish(self, tmp_path: str, ab: float, th_deg: float,
                         original_base: str, new_basename: str):
        permanent_path = original_base + os.path.basename(
            new_basename.replace(os.path.basename(original_base), "")
            ) + ".cube.npy"
        # Cleaner permanent path
        permanent_path = os.path.join(
            os.path.dirname(original_base),
            new_basename + ".cube.npy")
        save = messagebox.askyesno(
            "Save ellipticity-corrected cube?",
            f"Ellipticity correction applied "
            f"(a/b={ab:.4f}, θ={th_deg:+.2f}°).\n\n"
            f"Save permanently to:\n  {permanent_path}\n\n"
            f"  Yes → keep file (re-loadable in future sessions).\n"
            f"  No  → keep this run only; temp file deleted on exit.")
        if save:
            try:
                shutil.move(tmp_path, permanent_path)
                final_path = permanent_path
                try: os.rmdir(os.path.dirname(tmp_path))
                except Exception: pass
            except Exception as e:
                messagebox.showerror("save failed",
                    f"Could not move temp file:\n{e}\n"
                    f"Cube usable from temp: {tmp_path}")
                final_path = tmp_path
                _register_atexit_cleanup(tmp_path)
        else:
            final_path = tmp_path
            _register_atexit_cleanup(
                tmp_path,
                cleanup_dir=os.path.dirname(tmp_path))
        # Swap the active cube to the corrected one.
        try:
            self.cube = np.load(final_path, mmap_mode="r",
                                  allow_pickle=True)
            Ny, Nx, _H, _W = self.cube.shape
            self.path = final_path
            self._path_var.set(final_path)
            self.sample_key = register_runtime_sample(
                final_path, scan_shape=(Ny, Nx),
                vmax=float(self.vmax.get()),
                center_mask_radius=self._effective_center_mask_radius())
        except Exception as e:
            messagebox.showerror("reload failed",
                f"Corrected cube is at:\n{final_path}\n\nbut reload "
                f"failed: {e}"); return
        # Reset the live correction (it's baked into the new cube now).
        self.ellip_ab.set(1.0)
        self.ellip_theta.set(0.0)
        self._ellip_overlay = None
        self._refresh()
        self._ellip_status.configure(
            text=f"ellipFix done.  active sample: {self.sample_key}\n"
                  f"sliders reset to (1, 0°) — correction is baked "
                  f"into the new cube.")
        try:
            self.on_state_change("loaded", path=final_path,
                                    sample_key=self.sample_key,
                                    scan_shape=(Ny, Nx),
                                    frame_shape=self.cube.shape[2:])
        except Exception:
            pass

    def _propagate_filters_quiet(self):
        """Push blur / log / ellipticity into the SAMPLES entry WITHOUT
        triggering a redraw.  Safe to call from inside ``_refresh``.
        """
        if self.sample_key is None:
            return
        try:
            from data import SAMPLES
            cfg = SAMPLES.get(self.sample_key)
            if cfg is None:
                return
            sig = (float(self.blur_sigma.get())
                    if bool(self.use_blur.get()) else 0.0)
            cfg["blur_sigma"] = float(sig)
            cfg["log_stretch"] = bool(self.log_stretch.get())
            cfg["ellipticity_ab"] = float(self.ellip_ab.get())
            cfg["ellipticity_theta_deg"] = float(self.ellip_theta.get())
        except Exception as e:
            print(f"[pre] propagate filters: {e!r}", flush=True)

    def _propagate_filters(self):
        """Push the current blur / log-stretch / ellipticity settings
        into the registered SAMPLES entry, so every downstream tab
        that opens a LoadPRZ on this cube sees the SAME image the
        preview shows.  Then redraw."""
        self._propagate_filters_quiet()
        self._refresh()

    # =================================================================
    # Permanent Gaussian blur: write a new <base>_blurσ.cube.npy and
    # register as a runtime sample. Same pattern as NBED — heavy I/O
    # in a worker thread, main-thread modal afterwards.
    # =================================================================
    def _blur_apply_all(self):
        if self.cube is None:
            messagebox.showinfo("cube", "Load a cube first."); return
        if self._blur_busy:
            messagebox.showinfo("blur",
                "Blur is already in progress."); return
        if self._nbed_busy:
            messagebox.showinfo("blur",
                "NBED is in progress — wait for it to finish first.")
            return
        try:
            sig = float(self.blur_sigma.get())
        except Exception:
            messagebox.showerror("blur",
                "Bad σ value."); return
        if sig <= 0:
            messagebox.showinfo("blur",
                "σ must be > 0 to apply."); return
        Ny, Nx, H, W = self.cube.shape
        if not messagebox.askyesno(
                "Apply Gaussian blur to entire cube",
                f"Apply σ = {sig:g} Gaussian blur to every diffraction "
                f"pattern in this {Ny}×{Nx} scan (H×W={H}×{W})?\n\n"
                f"Writes a new .cube.npy alongside the original.  "
                f"You'll be asked whether to save permanently after.\n\n"
                f"This is independent of the per-batch training "
                f"augmentation (`aug_blur` in the Training tab)."):
            return
        self._blur_busy = True
        self._blur_status.configure(text="blur: starting…")
        threading.Thread(target=self._blur_worker,
                            args=(sig,), daemon=True).start()

    def _blur_worker(self, sig: float):
        try:
            base = os.path.splitext(self.path)[0]
            base = base[:-5] if base.endswith(".cube") else base
            new_basename = os.path.basename(base) + f"_blur{sig:g}"
            tmp_dir = tempfile.mkdtemp(prefix="dinosr_blur_")
            tmp_path = os.path.join(tmp_dir,
                                       new_basename + ".cube.npy")
            Ny, Nx, H, W = self.cube.shape
            from scipy.ndimage import gaussian_filter
            # Blur produces fractional values, so the output MUST be
            # float32.  Inheriting an integer source dtype (e.g. uint16
            # dm4 cubes) would truncate every smoothed pixel to an int
            # and zero out low-count data (an ~84% intensity loss).
            out = np.lib.format.open_memmap(
                tmp_path, mode="w+",
                dtype=np.float32,
                shape=(Ny, Nx, H, W))
            t0 = time.time()
            for y in range(Ny):
                # Index per-frame with the universal 2-tuple cube[y, x]
                # so this works for BOTH 4D .npy cubes and 3D-backed h5
                # wrappers (the latter reject a single-int row index).
                for x in range(Nx):
                    frame = np.asarray(self.cube[y, x], dtype=np.float32)
                    out[y, x] = gaussian_filter(frame, sigma=sig)
                if (y + 1) % max(1, Ny // 40) == 0:
                    dt = time.time() - t0
                    eta = dt * (Ny - y - 1) / max(y + 1, 1)
                    self.after(0, lambda y=y, dt=dt, eta=eta:
                        self._blur_status.configure(
                            text=f"blur σ={sig:g}: "
                                  f"{y+1}/{Ny} rows  "
                                  f"({100*(y+1)/Ny:.0f}%)  "
                                  f"eta {eta:.0f}s"))
            out.flush(); del out
            self.after(0, lambda: self._blur_finish(
                tmp_path, sig, base, new_basename))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "blur failed", err))
            self.after(0, lambda: self._blur_status.configure(
                text=f"blur failed: {err}"))
        finally:
            self._blur_busy = False

    def _blur_finish(self, tmp_path: str, sig: float,
                       original_base: str, new_basename: str):
        permanent_path = original_base + f"_blur{sig:g}.cube.npy"
        save = messagebox.askyesno(
            "Save blurred cube?",
            f"Blur σ={sig:g} applied to all patterns.\n\n"
            f"Save permanently to:\n  {permanent_path}\n\n"
            f"  Yes → keep file (re-loadable in future sessions).\n"
            f"  No  → keep this run only; temp file deleted on exit.")
        if save:
            try:
                shutil.move(tmp_path, permanent_path)
                final_path = permanent_path
                try: os.rmdir(os.path.dirname(tmp_path))
                except Exception: pass
            except Exception as e:
                messagebox.showerror("save failed",
                    f"Could not move temp file:\n{e}\n"
                    f"Cube usable from: {tmp_path}")
                final_path = tmp_path
                _register_atexit_cleanup(tmp_path)
        else:
            final_path = tmp_path
            _register_atexit_cleanup(
                tmp_path,
                cleanup_dir=os.path.dirname(tmp_path))
        # Swap the active cube to the blurred one and re-register.
        try:
            self.cube = np.load(final_path, mmap_mode="r",
                                  allow_pickle=True)
            Ny, Nx, _H, _W = self.cube.shape
            self.path = final_path
            self._path_var.set(final_path)
            self.sample_key = register_runtime_sample(
                final_path, scan_shape=(Ny, Nx),
                vmax=float(self.vmax.get()),
                center_mask_radius=self._effective_center_mask_radius())
        except Exception as e:
            messagebox.showerror("reload failed",
                f"Blurred cube is at:\n{final_path}\n\nbut reload "
                f"failed: {e}"); return
        self._refresh()
        self._blur_status.configure(
            text=f"blur σ={sig:g} done.  active sample: {self.sample_key}")
        # Status bar so other tabs see the new sample.
        try:
            if self.on_state_change is not None:
                self.on_state_change(
                    "loaded", sample_key=self.sample_key,
                    path=final_path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Binning (real-space n×n scan positions, or q-space n×n detector px)
    # Writes a NEW cube and switches to it; original file untouched.
    # ------------------------------------------------------------------
    def _bin_apply_all(self):
        if self.cube is None:
            messagebox.showinfo("cube", "Load a cube first."); return
        if self._bin_busy:
            messagebox.showinfo("binning",
                "Binning already in progress."); return
        if self._blur_busy or self._nbed_busy or self._ellip_busy:
            messagebox.showinfo("binning",
                "Another full-cube op is in progress — wait first.")
            return
        try:
            n = int(self._bin_factor.get())
        except Exception:
            messagebox.showerror("binning",
                "Bin factor must be an integer."); return
        if n < 2:
            messagebox.showinfo("binning",
                "Bin factor must be ≥ 2."); return
        mode = self._bin_mode.get()
        Ny, Nx, H, W = self.cube.shape
        if mode == "real":
            nY, nX = Ny // n, Nx // n
            if nY < 1 or nX < 1:
                messagebox.showerror("binning",
                    f"n={n} too large for {Ny}×{Nx} scan."); return
            msg = (f"Real-space bin n={n}: average every {n}×{n} block "
                    f"of scan positions.\n\nScan grid {Ny}×{Nx} → "
                    f"{nY}×{nX}  (detector {H}×{W} unchanged).")
            if (Ny % n) or (Nx % n):
                msg += (f"\n\nRemainder dropped: {Ny%n} row(s), "
                         f"{Nx%n} col(s).")
        else:
            hH, wW = H // n, W // n
            if hH < 1 or wW < 1:
                messagebox.showerror("binning",
                    f"n={n} too large for {H}×{W} detector."); return
            msg = (f"Q-space bin n={n}: average every {n}×{n} block of "
                    f"detector pixels per pattern.\n\nDetector {H}×{W} → "
                    f"{hH}×{wW}  (scan grid {Ny}×{Nx} unchanged).")
            if (H % n) or (W % n):
                msg += (f"\n\nRemainder px dropped: {H%n} row(s), "
                         f"{W%n} col(s).")
        msg += ("\n\nWrites a NEW .cube.npy and switches to it.  The "
                 "ORIGINAL cube file is NOT modified.")
        if not messagebox.askyesno("Apply binning to entire cube", msg):
            return
        self._bin_busy = True
        self._bin_status.configure(text="binning: starting…")
        threading.Thread(target=self._bin_worker,
                          args=(mode, n), daemon=True).start()

    def _bin_worker(self, mode: str, n: int):
        try:
            base = os.path.splitext(self.path)[0]
            base = base[:-5] if base.endswith(".cube") else base
            tag = f"_binr{n}" if mode == "real" else f"_binq{n}"
            new_basename = os.path.basename(base) + tag
            tmp_dir = tempfile.mkdtemp(prefix="dinosr_bin_")
            tmp_path = os.path.join(tmp_dir, new_basename + ".cube.npy")
            Ny, Nx, H, W = self.cube.shape
            # Always float32 — averaging produces fractional counts that
            # an integer source dtype (e.g. uint16) would truncate.
            dtype = np.float32
            t0 = time.time()
            # NOTE: index with the universal 2-tuple cube[y, x] so this
            # works for BOTH 4D .npy cubes and 3D-backed h5 wrappers
            # (the latter only accept a 2-tuple index).
            if mode == "real":
                nY, nX = Ny // n, Nx // n
                inv = 1.0 / float(n * n)
                out = np.lib.format.open_memmap(
                    tmp_path, mode="w+", dtype=dtype,
                    shape=(nY, nX, H, W))
                # HDF5-backed cubes expose read_block(): ONE contiguous
                # read per scan row instead of one per frame.  Frame-by-
                # frame reads make a chunked/compressed master
                # re-decompress the same chunks over and over (this is what
                # made a 2x2 bin take ~20 min).  Memmap cubes are already
                # fast frame-by-frame, so they keep the simple path.
                _reader = getattr(self.cube, "read_block", None)
                _cols = max(1, min(nX, (256 * 1024 ** 2)
                                      // max(n * n * H * W * 4, 1)))
                for Y in range(nY):
                    if _reader is not None:
                        for X0 in range(0, nX, _cols):
                            X1 = min(nX, X0 + _cols)
                            blk = np.asarray(
                                _reader(Y * n, n, X0 * n, (X1 - X0) * n),
                                dtype=np.float32)
                            out[Y, X0:X1] = blk.reshape(
                                n, X1 - X0, n, H, W).mean(axis=(0, 2)
                                                            ).astype(dtype,
                                                                     copy=False)
                    else:
                        for X in range(nX):
                            acc = np.zeros((H, W), dtype=np.float32)
                            for dy in range(n):
                                for dx in range(n):
                                    acc += np.asarray(
                                        self.cube[Y*n + dy, X*n + dx],
                                        dtype=np.float32)
                            out[Y, X] = (acc * inv).astype(dtype, copy=False)
                    if (Y + 1) % max(1, nY // 40) == 0:
                        dt = time.time() - t0
                        eta = dt * (nY - Y - 1) / max(Y + 1, 1)
                        self.after(0, lambda Y=Y, eta=eta, nY=nY:
                            self._bin_status.configure(
                                text=f"real-space bin n={n}: "
                                      f"{Y+1}/{nY} rows "
                                      f"({100*(Y+1)/nY:.0f}%)  "
                                      f"eta {eta:.0f}s"))
            else:
                hH, wW = H // n, W // n
                out = np.lib.format.open_memmap(
                    tmp_path, mode="w+", dtype=dtype,
                    shape=(Ny, Nx, hH, wW))
                for y in range(Ny):
                    for x in range(Nx):
                        f = np.asarray(self.cube[y, x], dtype=np.float32)
                        f = (f[:hH*n, :wW*n]
                                 .reshape(hH, n, wW, n).mean(axis=(1, 3)))
                        out[y, x] = f.astype(dtype, copy=False)
                    if (y + 1) % max(1, Ny // 40) == 0:
                        dt = time.time() - t0
                        eta = dt * (Ny - y - 1) / max(y + 1, 1)
                        self.after(0, lambda y=y, eta=eta, Ny=Ny:
                            self._bin_status.configure(
                                text=f"q-space bin n={n}: "
                                      f"{y+1}/{Ny} rows "
                                      f"({100*(y+1)/Ny:.0f}%)  "
                                      f"eta {eta:.0f}s"))
            out.flush(); del out
            self.after(0, lambda: self._bin_finish(
                tmp_path, mode, n, base, new_basename))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "binning failed", err))
            self.after(0, lambda: self._bin_status.configure(
                text=f"binning failed: {err}"))
        finally:
            self._bin_busy = False

    def _bin_finish(self, tmp_path: str, mode: str, n: int,
                     original_base: str, new_basename: str):
        tag = f"_binr{n}" if mode == "real" else f"_binq{n}"
        permanent_path = original_base + tag + ".cube.npy"
        kind = "real-space" if mode == "real" else "q-space"
        save = messagebox.askyesno(
            "Save binned cube?",
            f"Binning ({kind} n={n}) applied.\n\n"
            f"Save permanently to:\n  {permanent_path}\n\n"
            f"  Yes → keep file (re-loadable in future sessions).\n"
            f"  No  → keep this run only; temp file deleted on exit.")
        if save:
            try:
                shutil.move(tmp_path, permanent_path)
                final_path = permanent_path
                try: os.rmdir(os.path.dirname(tmp_path))
                except Exception: pass
            except Exception as e:
                messagebox.showerror("save failed",
                    f"Could not move temp file:\n{e}\n"
                    f"Cube usable from: {tmp_path}")
                final_path = tmp_path
                _register_atexit_cleanup(tmp_path)
        else:
            final_path = tmp_path
            _register_atexit_cleanup(
                tmp_path, cleanup_dir=os.path.dirname(tmp_path))
        # Swap the active cube to the binned one and re-register so every
        # downstream tab (training / eval / post-hoc / ACOM) runs on it.
        try:
            self.cube = np.load(final_path, mmap_mode="r",
                                  allow_pickle=True)
            Ny, Nx, _H, _W = self.cube.shape
            self.path = final_path
            self._path_var.set(final_path)
            self.sample_key = register_runtime_sample(
                final_path, scan_shape=(Ny, Nx),
                vmax=float(self.vmax.get()),
                center_mask_radius=self._effective_center_mask_radius())
        except Exception as e:
            messagebox.showerror("reload failed",
                f"Binned cube is at:\n{final_path}\n\nbut reload "
                f"failed: {e}"); return
        self._refresh()
        self._bin_status.configure(
            text=f"binning {kind} n={n} done.  "
                  f"active sample: {self.sample_key}")
        try:
            if self.on_state_change is not None:
                self.on_state_change(
                    "loaded", sample_key=self.sample_key,
                    path=final_path)
        except Exception:
            pass

    def _nbed_center_all(self):
        if self.cube is None:
            messagebox.showinfo("cube", "Load a cube first."); return
        if self._nbed_busy:
            messagebox.showinfo("nbed",
                "NBED centering is already in progress."); return
        if not messagebox.askyesno(
                "NBED-center entire cube",
                f"Re-centre every diffraction pattern in this "
                f"{self.cube.shape[0]}×{self.cube.shape[1]} scan using "
                f"the BF-disk threshold-centroid?\n\n"
                f"Search radius:    {self._nbed_search_radius.get()} px\n"
                f"BF disk radius:   {self._nbed_disk_radius.get()} px\n"
                f"Threshold:        {self._nbed_threshold.get():.2f}\n\n"
                f"You'll be asked whether to save the result "
                f"permanently after it's done."):
            return
        self._nbed_busy = True
        self._nbed_status.configure(text="NBED centering: starting…")
        threading.Thread(target=self._nbed_worker, daemon=True).start()

    def _nbed_worker(self):
        try:
            sr = int(self._nbed_search_radius.get())
            dr = int(self._nbed_disk_radius.get())
            tf = float(self._nbed_threshold.get())
            base = os.path.splitext(self.path)[0]
            base = base[:-5] if base.endswith(".cube") else base
            base_name = os.path.basename(base) + "_nbed"
            # Always write to a temp location first; on "yes save" we'll
            # move it next to the original.
            tmp_dir = tempfile.mkdtemp(prefix="dinosr_nbed_")
            tmp_path = os.path.join(tmp_dir, base_name + ".cube.npy")

            Ny = self.cube.shape[0]
            t0 = time.time()

            def _progress(done, total):
                # called from worker thread; marshal via after()
                dt = time.time() - t0
                eta = dt * (total - done) / max(done, 1)
                self.after(0, lambda d=done, t=total, e=eta:
                    self._nbed_status.configure(
                        text=f"NBED centering: {d}/{t} rows  "
                              f"({100*d/t:.0f}%)   eta {e:.0f}s"))

            centers = nbed_center_cube(
                self.cube, tmp_path,
                search_radius=sr, disk_radius=dr,
                threshold_frac=tf, progress_cb=_progress)

            # Hand back to the main thread for the modal + state swap
            self.after(0, lambda: self._nbed_finish(
                tmp_path, centers, base, base_name))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "NBED centering failed", err))
            self.after(0, lambda: self._nbed_status.configure(
                text="NBED centering failed."))
        finally:
            self._nbed_busy = False

    def _nbed_finish(self, tmp_path, centers, original_base, new_basename):
        """Modal: save permanently or keep only for this session.
        Either way the centred cube becomes the active dataset and a
        runtime sample is registered."""
        try:
            self._nbed_finish_impl(tmp_path, centers, original_base,
                                     new_basename)
        except Exception as e:
            import traceback as _tb
            tb = _tb.format_exc()
            messagebox.showerror("NBED finish failed",
                f"Centred file is at:\n{tmp_path}\n\n"
                f"…but post-processing failed:\n{e}\n\n{tb}")
            self._nbed_status.configure(
                text=f"NBED finish failed; cube still at {tmp_path}")

    def _nbed_finish_impl(self, tmp_path, centers, original_base,
                            new_basename):
        # centers is (Ny, Nx, 2). H, W come from the ORIGINAL cube
        # before we swap it out (they're identical to the new cube's
        # since we only roll patterns, never crop).
        Ny, Nx = centers.shape[:2]
        H, W = self.cube.shape[2:]
        cy0 = (H - 1) / 2.0; cx0 = (W - 1) / 2.0
        dy = centers[..., 0] - cy0
        dx = centers[..., 1] - cx0
        shift_mag = np.hypot(dy, dx)

        msg_stats = (
            f"NBED centring complete. {Ny*Nx} patterns recentered.\n"
            f"per-pattern shift  median = {np.median(shift_mag):.2f} px,  "
            f"95% = {np.percentile(shift_mag, 95):.2f} px,  "
            f"max = {shift_mag.max():.2f} px.\n\n")

        permanent_path = original_base + "_nbed.cube.npy"
        save = messagebox.askyesno(
            "Save NBED-centered cube?",
            msg_stats +
            f"Save permanently to:\n  {permanent_path}\n\n"
            f"  Yes → keep file (re-loadable in future sessions).\n"
            f"  No  → keep this run only; temp file is deleted on exit.")

        if save:
            try:
                # Move temp file next to the original
                shutil.move(tmp_path, permanent_path)
                final_path = permanent_path
                # tmp_dir is now empty; remove it
                try: os.rmdir(os.path.dirname(tmp_path))
                except Exception: pass
            except Exception as e:
                messagebox.showerror("save failed",
                    f"Could not move temp file:\n{e}\n"
                    f"Cube is still usable from temp path: {tmp_path}")
                final_path = tmp_path
                _register_atexit_cleanup(tmp_path)
        else:
            final_path = tmp_path
            _register_atexit_cleanup(tmp_path,
                                       cleanup_dir=os.path.dirname(tmp_path))

        # Auto-port pair-labels sidecar from the source cube to the
        # NBED-centred cube. NBED only rolls within each diffraction
        # pattern; the FLAT scan indices stay valid against the new
        # cube (same Ny, Nx). Without this auto-port, labels collected
        # before NBED look like they "vanished" because the active
        # sample's path changed and the auto-bump in train_panel can't
        # find the sidecar.
        try:
            from gui_app.pair_labels import (
                label_path_for_cube, load_pair_labels, save_pair_labels,
                label_count)
            src_label_path = label_path_for_cube(self.path)
            dst_label_path = label_path_for_cube(final_path)
            if (os.path.abspath(src_label_path)
                    != os.path.abspath(dst_label_path)
                    and os.path.exists(src_label_path)):
                src_lbl = load_pair_labels(self.path)
                cnt = label_count(src_lbl)
                if cnt.get("total", 0) > 0:
                    # If the destination already has labels, MERGE; else
                    # just write a fresh copy.
                    if os.path.exists(dst_label_path):
                        dst_lbl = load_pair_labels(final_path)
                        existing = {(int(p["a"]), int(p["b"]))
                                     for p in dst_lbl.get("pairs", [])}
                        added = 0
                        for p in src_lbl.get("pairs", []):
                            ab = (int(p["a"]), int(p["b"]))
                            ba = (int(p["b"]), int(p["a"]))
                            if ab in existing or ba in existing: continue
                            dst_lbl.setdefault("pairs", []).append(p)
                            added += 1
                        save_pair_labels(final_path, dst_lbl)
                        msg_port = (f"merged {added} new pair labels "
                                    f"into the NBED cube's sidecar "
                                    f"({len(dst_lbl.get('pairs', []))} "
                                    f"total).")
                    else:
                        # Fresh write — copy the source object as-is.
                        save_pair_labels(final_path, src_lbl)
                        msg_port = (f"copied {cnt['total']} pair labels "
                                    f"from the source cube to the NBED "
                                    f"sidecar.")
                    print(f"[nbed] {msg_port}", flush=True)
                    self._nbed_status.configure(
                        text=(self._nbed_status.cget("text") +
                              "\n" + msg_port))
        except Exception as _e:
            # Non-fatal: training will simply not see the labels.
            print(f"[nbed] could not auto-port pair labels: {_e!r}",
                  flush=True)

        # Replace the active cube with the centered one (memmap'd).
        try:
            new_cube = np.load(final_path, mmap_mode="r",
                                  allow_pickle=True)
        except Exception as e:
            messagebox.showerror("reload failed",
                f"Centred file written but could not reopen:\n{e}")
            return
        self.cube = new_cube
        self.path = final_path
        self._path_var.set(final_path)
        self._nbed_overlay_center = None

        # Register the new cube as a runtime sample so Training, Eval,
        # Post-hoc, ACOM all see it.
        try:
            new_key = register_runtime_sample(
                final_path, scan_shape=(Ny, Nx),
                vmax=float(self.vmax.get()),
                center_mask_radius=self._effective_center_mask_radius())
            self.sample_key = new_key
            self.on_state_change("loaded", path=final_path,
                                  sample_key=new_key,
                                  scan_shape=(Ny, Nx),
                                  frame_shape=(H, W))
        except Exception as e:
            messagebox.showwarning("register sample",
                f"NBED cube is loaded but could not register as a "
                f"runtime sample:\n{e}")

        kind = "permanently"  if save else "for this session only"
        self._nbed_status.configure(
            text=f"active dataset is now NBED-centered ({kind}).\n"
                  f"path: {final_path}\n"
                  f"shift: median={np.median(shift_mag):.2f}px, "
                  f"max={shift_mag.max():.2f}px")
        self._refresh()

    # ------------------------------------------------------------------
    # Pair labelling (pre-train)
    # ------------------------------------------------------------------
    def _refresh_pair_label_count(self):
        """Read the pair-labels sidecar for the loaded cube and update
        the small status label next to the 'Label pairs…' button."""
        if not getattr(self, "_pair_label_status", None):
            return
        if not self.path:
            self._pair_label_status.configure(text="(load a cube first)")
            return
        try:
            from gui_app.pair_labels import (
                load_pair_labels, label_count, label_path_for_cube)
            d = load_pair_labels(self.path)
            c = label_count(d)
            sp = os.path.basename(label_path_for_cube(self.path))
            if c["total"] == 0:
                self._pair_label_status.configure(
                    text=f"no labels yet ({sp})")
            else:
                self._pair_label_status.configure(
                    text=f"{c['total']} pairs  "
                          f"({c['same']} same / {c['diff']} diff)\n"
                          f"-> {sp}")
        except Exception as e:
            self._pair_label_status.configure(
                text=f"(label-count error: {e})")

    def _import_pair_labels(self):
        """Import labels from another source into this cube's sidecar.

        Supported formats (auto-detected from extension):
          .json  → another pair_labels.json (our schema)
          .csv   → 'a, b, y' or 'rx_a, ry_a, rx_b, ry_b, y'
          .npy / .png / .json → per-pixel class-map; pairs are
                                derived by sampling.
        """
        if self.cube is None or not self.path:
            messagebox.showinfo("cube", "Load a cube first."); return
        src = filedialog.askopenfilename(
            title="Import pair labels",
            filetypes=[
                ("All supported", "*.json;*.csv;*.npy;*.png;*.tif;*.tiff;*.bmp"),
                ("pair_labels.json or class-map JSON", "*.json"),
                ("CSV (a,b,y or rx_a,ry_a,rx_b,ry_b,y)", "*.csv"),
                ("Class-map .npy",  "*.npy"),
                ("Class-map image", "*.png;*.tif;*.tiff;*.bmp"),
                ("All files", "*.*"),
            ])
        if not src: return
        ext = os.path.splitext(src)[1].lower()
        Ny, Nx, _, _ = self.cube.shape
        scan_shape = (int(Ny), int(Nx))

        try:
            from gui_app.pair_labels import (
                import_from_json, import_from_csv,
                import_from_classmap)
        except Exception as e:
            messagebox.showerror("import failed", repr(e)); return

        # Heuristic: a .json file might be either pair_labels.json
        # (has a top-level "pairs" key) or a class-map dict. Peek at
        # the contents to decide.
        kind = None
        if ext == ".json":
            try:
                import json as _json
                with open(src, "r", encoding="utf-8") as f:
                    obj = _json.load(f)
                if isinstance(obj, dict) and "pairs" in obj:
                    kind = "json_pairs"
                else:
                    kind = "json_classmap"
            except Exception:
                kind = "json_pairs"   # let the importer fail loudly
        elif ext == ".csv":
            kind = "csv"
        elif ext in (".npy", ".png", ".tif", ".tiff", ".bmp",
                       ".jpg", ".jpeg"):
            kind = "classmap"
        else:
            messagebox.showerror("import",
                f"Unsupported extension: {ext}"); return

        # If class-map, ask for the pair budget.
        max_pairs = 500; same_frac = 0.5; ignore_value = -1
        if kind in ("classmap", "json_classmap"):
            ans = simpledialog_int(self,
                "Class-map import",
                "How many pairs to derive?  (50/50 same/diff balance)",
                default=500, min_=10, max_=10000)
            if ans is None: return
            max_pairs = int(ans)

        # Run the import (fast; in-thread is fine).
        try:
            if kind == "json_pairs":
                stats = import_from_json(self.path, src,
                                            scan_shape=scan_shape)
                src_kind = "pair_labels JSON"
            elif kind == "csv":
                stats = import_from_csv(self.path, src,
                                           scan_shape=scan_shape)
                src_kind = "CSV"
            else:  # classmap from npy/png/json
                stats = import_from_classmap(
                    self.path, src,
                    scan_shape=scan_shape,
                    max_pairs=max_pairs,
                    same_frac=same_frac,
                    ignore_value=ignore_value)
                src_kind = ("class-map JSON" if kind == "json_classmap"
                            else f"class-map {ext}")
        except Exception as e:
            messagebox.showerror("import failed",
                f"{src}\n\n{e}")
            return

        # Confirmation modal
        msg = (
            f"Imported from {src_kind}:\n  {src}\n\n"
            f"  added         : {stats.get('added', 0)}\n"
            f"  skipped (dup) : {stats.get('skipped_dup', 0)}\n"
            f"  skipped (oor) : {stats.get('skipped_oor', 0)}\n"
            f"  skipped (bad) : {stats.get('skipped_bad_y', 0)}\n")
        if "n_same_added" in stats:
            msg += (f"  same / diff   : "
                    f"{stats['n_same_added']} / {stats['n_diff_added']}\n")
        msg += (f"\nTotal labels on disk: {stats.get('n_after', 0)}")
        messagebox.showinfo("Import complete", msg)

        # Refresh count + notify Training tab
        self._refresh_pair_label_count()
        try:
            from gui_app.pair_labels import load_pair_labels, label_count
            self.on_state_change(
                "pair_labels_changed",
                sample_key=self.sample_key,
                label_count=label_count(load_pair_labels(self.path)))
        except Exception:
            pass

    def _spawn_pair_labeler(self, mode):
        """Launch the labelling Toplevel in `mode` ∈ {'random','review'}."""
        if self.cube is None or not self.path:
            messagebox.showinfo("cube", "Load a cube first."); return
        try:
            from gui_app.pair_labeler import PairLabelerWindow
        except Exception as e:
            messagebox.showerror("pair-labeller import failed", repr(e))
            return
        # Review mode requires existing labels.
        if mode == "review":
            try:
                from gui_app.pair_labels import (
                    load_pair_labels, label_count)
                cnt = label_count(load_pair_labels(self.path))
            except Exception:
                cnt = {"total": 0}
            if cnt.get("total", 0) == 0:
                messagebox.showinfo("no labels yet",
                    "There are no labels for this cube to review. "
                    "Click 'Label pairs…' first.")
                return
        sample_key = self.sample_key or os.path.basename(self.path)

        def _after_close(counts):
            try: self._refresh_pair_label_count()
            except Exception: pass
            try:
                self.on_state_change(
                    "pair_labels_changed",
                    sample_key=self.sample_key,
                    label_count=counts)
            except Exception:
                pass

        try:
            win = PairLabelerWindow(
                self,
                cube_path=self.path,
                sample_key=sample_key,
                cube=self.cube,
                vmax=float(self.vmax.get()),
                cmap=self.cmap.get(),
                mode=mode,
                on_close=_after_close,
            )
            win.transient(self)
            win.focus_set()
        except Exception as e:
            messagebox.showerror(
                f"pair-labeller failed ({mode})", repr(e))

    def _open_pair_labeler(self):
        self._spawn_pair_labeler("random")

    def _open_pair_reviewer(self):
        self._spawn_pair_labeler("review")

    def _refresh(self):
        if self.cube is None:
            return
        # Keep the runtime SAMPLES entry in sync with the user's current
        # vmax choice on this tab.  ``center_mask_radius`` is no longer
        # set from PrePanel — the equivalent role is now played by
        # ``polar_mask_cols`` and we leave SAMPLES at whatever value
        # was set when the cube was registered.
        if self.sample_key is not None:
            try:
                from data import SAMPLES as _S
                if self.sample_key in _S:
                    _S[self.sample_key]["vmax"] = float(self.vmax.get())
            except Exception:
                pass
        Ny, Nx, H, W = self.cube.shape
        idx = int(self.idx.get())
        y, x = divmod(idx, Nx)
        raw = np.asarray(self.cube[y, x]).astype(np.float32)
        vmax = float(self.vmax.get())
        norm = rescale_like_vmax(raw, vmax=vmax)
        # Live ellipticity correction (a/b ≠ 1).  Applied BEFORE
        # the cart pipeline (resize → crop → COM → polar) so every
        # downstream step sees the corrected geometry.  Identity
        # warp short-circuits inside ``_ellip_warp_image``.
        norm = self._ellip_warp_image(norm)

        # ---- processed pane: mirror the TRAINING pipeline exactly ----
        # LoadPRZ.__getitem__ does cv2.resize(raw → 192) FIRST, then the
        # transform pipeline applies CenterCrop(center_crop_size) to the
        # 192-frame, then COM, then Resize back to 192, then Polar.
        # Earlier versions of this preview cropped the raw frame *before*
        # the resize, which made `center_crop_size=140` mean different
        # spatial extents in the preview vs in training (140/raw vs
        # 140/192). Fixed.
        import cv2
        resized192 = cv2.resize(norm, (192, 192),
                                  interpolation=cv2.INTER_LINEAR)
        crop_size = int(self.center_crop_size.get())
        crop_size = max(8, min(crop_size, 192))
        cropped = _center_crop(resized192, crop_size)
        if self.com.get():
            # COM search radius is 2 × effective beam-mask radius.
            # Derive that radius from polar_mask_cols (≈ pmc/2) since
            # the standalone center_mask_radius slider is gone.
            search_r = 2 * self._effective_center_mask_radius()
            cropped = _com_shift(cropped, search_radius=search_r)
        proc = cv2.resize(cropped, (192, 192),
                           interpolation=cv2.INTER_LINEAR)

        # ---- draw ----
        # Display-only filters (blur + log stretch). Order: blur first
        # (denoise / inspect), then log (compress dynamic range).
        use_blur = bool(self.use_blur.get())
        use_log = bool(self.log_stretch.get())
        try:
            sig = float(self.blur_sigma.get())
        except Exception:
            sig = 0.0
        if use_blur and sig > 0:
            from scipy.ndimage import gaussian_filter
            proc_disp = gaussian_filter(proc.astype(np.float32),
                                            sigma=sig)
        else:
            proc_disp = proc
        if use_log:
            proc_disp = np.log1p(np.clip(proc_disp, 0.0, None) * 50.0)
        tags = []
        if use_blur and sig > 0:
            tags.append(f"blur σ={sig:g}")
        if use_log:
            tags.append("log1p×50")
        log_tag = ("  [" + ", ".join(tags) + "]") if tags else ""
        com_str = "COM on" if self.com.get() else "COM off"

        # ---- LEFT panel: cart-processed (192×192, what the model
        # sees in CART space — after CenterCrop + optional COM-shift +
        # Resize) ----
        self._ax_proc.clear()
        self._ax_proc.set_xticks([]); self._ax_proc.set_yticks([])
        self._ax_proc.imshow(proc_disp, cmap=self.cmap.get(),
                                aspect="equal",
                                interpolation="nearest")
        # NBED overlay (was on the dropped raw pane).  After CenterCrop
        # +Resize the raw center maps to (96, 96) in 192-frame; the
        # BF-disk centre maps proportionally.  We draw it here so the
        # user can still tune NBED parameters visually.
        if self._nbed_overlay_center is not None:
            cy_r, cx_r = self._nbed_overlay_center
            # raw-frame size from cube
            H_r, W_r = self.cube.shape[2:]
            # raw → 192 cart mapping: raw_px * 192 / size
            cy_192 = (cy_r / max(H_r - 1, 1)) * 191.0
            cx_192 = (cx_r / max(W_r - 1, 1)) * 191.0
            # CenterCrop(crop_size) then resize back to 192:
            # cart_px = 96 + (192_px - 96) * 192 / crop_size
            scale = 192.0 / max(crop_size, 1)
            cy_disp = 96.0 + (cy_192 - 96.0) * scale
            cx_disp = 96.0 + (cx_192 - 96.0) * scale
            self._ax_proc.plot([cx_disp], [cy_disp], marker="+",
                                  color="lime", markersize=12,
                                  markeredgewidth=1.8,
                                  label="BF centre")
            self._ax_proc.plot([96], [96], marker="x", color="cyan",
                                  markersize=8, markeredgewidth=1.2,
                                  label="geom centre")
        # Ellipticity overlay — only when the slider is still at
        # identity (a/b == 1).  Once the user moves the slider the
        # corrected image speaks for itself (the elliptical feature
        # is now circular); showing the elliptical outline on top
        # would look wrong.  Drawn on the LEFT pane in 192-frame
        # coords scaled from raw px.
        try:
            ab_cur = float(self.ellip_ab.get())
        except Exception:
            ab_cur = 1.0
        if (self._ellip_overlay is not None
                and abs(ab_cur - 1.0) < 1e-4):
            from matplotlib.patches import Ellipse as _Ellipse
            cx_r, cy_r, axis_minor_r, axis_major_r, theta_deg = \
                self._ellip_overlay
            H_r, W_r = self.cube.shape[2:]
            # raw → cart 192 frame.  Then CenterCrop+Resize again maps
            # (192, 192) box → cart_size = 192 / crop_size · 192.
            # For an overlay drawn directly on the 192-cart pane,
            # scale raw-px by (192 / H_raw) and apply the same crop
            # rescaling as the NBED overlay above.
            s_192 = 192.0 / max(H_r - 1, 1)
            scale = 192.0 / max(crop_size, 1)
            cx_disp = 96.0 + (cx_r * s_192 - 96.0) * scale
            cy_disp = 96.0 + (cy_r * s_192 - 96.0) * scale
            w_disp = 2.0 * axis_minor_r * s_192 * scale
            h_disp = 2.0 * axis_major_r * s_192 * scale
            # matplotlib's Ellipse(width=, height=) treats width as
            # x-extent and height as y-extent; the angle rotates the
            # x-axis CCW.  We stored the major-axis angle directly
            # (already normalised in _ellip_fit_*), so map width →
            # 2·major and height → 2·minor and pass the angle as-is.
            self._ax_proc.add_patch(_Ellipse(
                (cx_disp, cy_disp), h_disp, w_disp,
                angle=theta_deg,
                fill=False, edgecolor=(0.20, 1.0, 0.80, 0.95),
                linestyle="--", lw=1.4,
                label="fitted ellipse"))

        # Cart-frame visualisation of what `polar_mask_cols` blocks.
        # The PolarTransform samples cart pixels with normalised
        # radius r ∈ [0, 1] (r=1 = inscribed disk of radius 96 px),
        # so polar column k corresponds to cart radius ≈ k * 96 / 192
        # = k / 2 pixels.  We draw a translucent disk + dashed
        # outline at that radius so the user can see exactly which
        # part of the cart image is being blanked before the
        # polar transform.
        pmc = max(0, int(self.polar_mask_cols.get()))
        if pmc > 0:
            cart_mask_r = pmc * 96.0 / 192.0   # = pmc/2
            self._ax_proc.add_patch(Circle(
                (96, 96), cart_mask_r,
                fill=True, facecolor=(1.0, 0.2, 0.2, 0.30),
                edgecolor=(1.0, 0.4, 0.4, 0.95),
                linestyle="--", lw=1.2,
                label=f"polar_mask_cols={pmc}"))
        self._ax_proc.set_title(
            f"What the model sees — cart 192×192  "
            f"(crop={crop_size}, {com_str}, mask≈{pmc/2:g}px)"
            f"{log_tag}", fontsize=10)

        # ---- RIGHT panel: polar (192×192, model encoder input) ----
        # Run the actual PolarTransform + PolarMaskLeft on proc_disp
        # so the user sees exactly the tensor going into the encoder.
        polar_disp = None
        try:
            import torch
            from dino_sr_contrastive_model import (
                PolarTransform, PolarMaskLeft)
            _pt = PolarTransform(output_size=192)
            _pml = PolarMaskLeft(
                k_cols=max(0, int(self.polar_mask_cols.get())))
            _t = torch.from_numpy(proc_disp.astype(np.float32)
                                       ).unsqueeze(0).unsqueeze(0)
            _t = _pml(_pt(_t))
            polar_disp = _t.squeeze().detach().cpu().numpy()
        except Exception as _e:
            print(f"[pre] polar preview failed: {_e!r}", flush=True)

        self._ax_polar.clear()
        self._ax_polar.set_xticks([]); self._ax_polar.set_yticks([])
        if polar_disp is not None:
            self._ax_polar.imshow(polar_disp, cmap=self.cmap.get(),
                                      aspect="auto",
                                      interpolation="nearest")
            self._ax_polar.set_xlabel("r (low → high)", fontsize=9)
            self._ax_polar.set_ylabel("θ (0 → 2π)", fontsize=9)
            self._ax_polar.set_title(
                f"Polar (encoder input) — 192×192, "
                f"polar_mask_cols={int(self.polar_mask_cols.get())}",
                fontsize=10)
        else:
            self._ax_polar.text(0.5, 0.5,
                "(polar preview unavailable)",
                ha="center", va="center",
                transform=self._ax_polar.transAxes,
                fontsize=10, color="#888")

        self._fig.tight_layout()
        self._canvas.draw_idle()

        # info line
        sat = float((raw >= vmax).mean()) * 100
        self._info.configure(
            text=f"  idx={idx}  (y={y}, x={x})    "
                  f"raw min/max/mean = {raw.min():.3g} / {raw.max():.3g} "
                  f"/ {raw.mean():.3g}    "
                  f"saturated@vmax = {sat:.1f}% pixels")
        self._idx_lbl.configure(text=f"idx = {idx}    (y={y}, x={x})")

        # Live-propagate the pre-processing knobs (vmax, crop,
        # polar_mask_cols, COM) to the Training tab so its Apply uses
        # whatever the user just set here.  Without this, the Training
        # tab keeps a STALE copy synced only at cube-load time and the
        # user's edits silently don't reach the run.
        try:
            self.on_state_change("pre_kwargs_changed")
            # Also push blur / log / ellipticity into SAMPLES so the
            # next LoadPRZ open on this cube picks them up.  (Cheap;
            # ~one dict write.)
            try:
                self._propagate_filters_quiet()
            except Exception:
                pass
        except Exception:
            pass
