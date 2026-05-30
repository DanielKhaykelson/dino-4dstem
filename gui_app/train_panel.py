"""train_panel.py -- Tab 2 (Training).

A single 'Use paper recipe' toggle gives the locked DINO + cluster1d
recipe with K=6 used in the manuscript. When unchecked, four
sub-tabs (Core / Schedule / Augmentation / Polar) expose every knob.

Live x=epoch, y=loss plot reads <outdir>/training_log.csv every second.
The training runs in a daemon thread (gui_app.runner.TrainingJob) so
the GUI stays responsive.
"""
from __future__ import annotations
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import customtkinter as ctk
from tkinter import messagebox

import matplotlib
matplotlib.use("TkAgg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data import SAMPLES
from gui_app.tooltip import add_help_button
from gui_app.runner import TrainingJob, make_outdir, read_training_log


# ---------------------------------------------------------------------------
# variant -> active loss-knob set
# ---------------------------------------------------------------------------
VARIANTS = {
    "1D":               {"cluster1d": True, "supcon": False, "centroid": False,
                          "weight":   True,  "spatial": False},
    # All-physics recipe: DINO + γ-weight + 1D intra/inter + spatial-gated.
    # Picking this variant auto-bumps lam_spatial to 0.5 (custom mode).
    "1D + spatial":     {"cluster1d": True, "supcon": False, "centroid": False,
                          "weight":   True,  "spatial": True},
    # Sinkhorn-Knopp teacher target (SwAV / DINOv2-SK style) + 1D radial.
    # Same loss profile as "1D" but DINO centering is replaced by SK
    # equipartition. Set λ_1D = 0 from this variant for vanilla SK.
    "SK + 1D":          {"cluster1d": True, "supcon": False, "centroid": False,
                          "weight":   True,  "spatial": False,
                          "target_mode": "sinkhorn"},
    "Contrastive1D":    {"cluster1d": False, "supcon": True, "centroid": False,
                          "weight":   False, "spatial": False},
    "Vanilla":          {"cluster1d": False, "supcon": False, "centroid": False,
                          "weight":   False, "spatial": False},
    "Vanilla+centroid": {"cluster1d": False, "supcon": False, "centroid": True,
                          "weight":   False, "spatial": False},
    "Vanilla+weight":   {"cluster1d": False, "supcon": False, "centroid": False,
                          "weight":   True,  "spatial": False},
    "Vanilla+spatial":  {"cluster1d": False, "supcon": False, "centroid": False,
                          "weight":   False, "spatial": True},
}

# Sensible non-zero defaults for each loss term, applied when the user
# picks a variant that turns the term on (and the term's current λ is
# still 0). Lets variant selection actually train the implied loss
# without forcing the user to also type a value.
VARIANT_LAMBDA_DEFAULTS = {
    "cluster1d_lambda_intra": 0.1,
    "cluster1d_lambda_inter": 0.1,
    "supcon_lambda":          0.1,
    "centroid_lambda":        0.1,
    "lam_spatial":            0.5,
    "conf_weight_gamma":      0.5,
}
# Maps the variant flag name to the lambda(s) it controls.
VARIANT_FLAG_TO_LAMBDAS = {
    "cluster1d": ("cluster1d_lambda_intra", "cluster1d_lambda_inter"),
    "supcon":    ("supcon_lambda",),
    "centroid":  ("centroid_lambda",),
    "spatial":   ("lam_spatial",),
    "weight":    ("conf_weight_gamma",),
}

# Locked paper recipe (= "1D" variant with K=6 v=2).
PAPER_DEFAULTS = dict(
    variant="1D",
    K=6, epochs=30, batch_size=128,
    lr=3e-4, weight_decay=1e-6,
    T0=0.04, Tfin=0.07,
    warmup_frac=0.2,                        # teacher temp warmup
    center_momentum=0.9,                    # DINO center EMA
    EMA0=0.990, EMAfin=0.999,               # teacher param EMA schedule
    n_layers=1,                              # ResNet18 stages kept (1..4)
    cluster1d_lambda=0.1, cluster1d_margin=0.4,
    # Per-term cluster1d weights.  Paper used the unified
    # `cluster1d_lambda` for both intra and inter (the values below
    # are the defaults that match that).  Setting `intra=0.1,
    # inter=0.0` is the "auto-K via 1D" recipe — keeps physics-
    # baked centroid concentration via intra, removes the
    # anti-shrinkage centroid push-apart via inter.
    cluster1d_lambda_intra=0.1, cluster1d_lambda_inter=0.1,
    cluster1d_warmup_frac=0.0, cluster1d_ramp_frac=0.0,
    conf_weight_gamma=0.5,
    supcon_lambda=0.0, supcon_temperature=0.3,
    centroid_lambda=0.0, centroid_margin=0.3,
    lam_spatial=0.0,
    # Gated 4-neighbor spatial Potts thresholds.  -2 sentinel means
    # "auto-load from per-sample gate_thresholds.json (same calibration
    # as radial-gated SupCon)".  User can override with explicit values.
    spatial_tau_pos=-2.0, spatial_tau_neg=-2.0,
    polar_size=192, polar_mask_cols=45,
    center_crop_size=140, center_mask_radius=15,
    com_centering=True, com_search_radius_factor=2.0,
    theta_shift_student=192, theta_shift_teacher=16,
    aug_hflip=False, aug_vflip=False, aug_colorjitter=False,
    # Gaussian blur was used in the paper (its tag is NOT in the
    # paper's `aug_disable` list).  Default ON to match.
    aug_blur=True,
    # Per-augmentation params (defaults match the previous hardcoded
    # values in dino_sr_contrastive_model.get_contrastive_transforms).
    cj_brightness=0.2, cj_contrast=0.2,
    blur_kernel_max=5, blur_sigma_max=0.3,
    save_every=10, seed=42,
    # Pair-supervised assignment loss (defaults to off for paper recipe)
    lambda_pair=0.0,
    pair_entropy_reg=0.0,
    pair_per_batch=32,
)


def _seg(parent, title):
    """Small section header used inside the sub-tabs."""
    f = ctk.CTkFrame(parent, fg_color="transparent")
    f.pack(fill="x", padx=2, pady=(8, 0))
    ctk.CTkLabel(f, text=title, font=("Segoe UI", 11, "bold")).pack(
        anchor="w")
    sep = ctk.CTkFrame(parent, height=2, fg_color=("#cccccc", "#444444"))
    sep.pack(fill="x", padx=2, pady=(0, 6))
    return parent


def _entry_row(parent, label, var, help_text="", width=110):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=4, pady=2)
    ctk.CTkLabel(row, text=label, width=220, anchor="w"
                  ).pack(side="left")
    e = ctk.CTkEntry(row, textvariable=var, width=width)
    e.pack(side="left")
    if help_text:
        h = add_help_button(row, help_text)
        h.pack(side="left", padx=(8, 0))
    return e


def _check_row(parent, label, var, help_text=""):
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=4, pady=2)
    ctk.CTkCheckBox(row, text=label, variable=var).pack(side="left")
    if help_text:
        h = add_help_button(row, help_text)
        h.pack(side="left", padx=(8, 0))


class TrainPanel(ctk.CTkFrame):

    def __init__(self, master, on_run_started=None, on_run_finished=None,
                 prepanel=None):
        super().__init__(master)
        self.on_run_started = on_run_started or (lambda *_, **__: None)
        self.on_run_finished = on_run_finished or (lambda *_, **__: None)
        self.prepanel = prepanel              # for "use loaded path" option

        self.use_defaults = ctk.BooleanVar(value=True)
        # All editable variables
        self.var = self._build_vars()

        # Job state
        self.job: "TrainingJob | None" = None
        self._poll_after = None
        self._sample_keys = sorted(SAMPLES.keys())
        # Apply / snapshot state. The user clicks "Apply changes" to
        # commit current entry values into a snapshot dict.  Train
        # builds its kwargs from the snapshot (not the live entries).
        # An entry typed after Apply doesn't affect anything until
        # the user clicks Apply again.  If Train is clicked without
        # ever pressing Apply, we auto-snapshot the current entries
        # on the fly so the workflow still works without ceremony.
        self._applied_snapshot: "dict | None" = None
        self._applied_at: "str | None" = None

        self._build()
        self._sync_defaults_lock()

    # ---- runtime-sample notifications ----
    def on_runtime_sample_added(self, key: str):
        """Called by the main app when a new sample has been registered
        in data.SAMPLES (e.g. from the Pre-processing tab loading an
        arbitrary .prz). Refresh the dropdown, auto-select the new key,
        and seed vmax / beam-mask / center-crop from the Pre-processing
        tab so the user doesn't have to re-tune them here."""
        try:
            self._sample_keys = sorted(SAMPLES.keys())
            if hasattr(self, "_sample_menu"):
                self._sample_menu.configure(values=self._sample_keys)
            if key in self._sample_keys:
                self.var["sample"].set(key)
            # Pull what the user tuned on Tab 1 into the recipe knobs.
            if self.prepanel is not None:
                pk = self.prepanel.get_pre_kwargs()
                if "center_mask_radius" in pk:
                    self.var["center_mask_radius"].set(int(pk["center_mask_radius"]))
                if "center_crop_size" in pk:
                    self.var["center_crop_size"].set(int(pk["center_crop_size"]))
                if "polar_mask_cols" in pk:
                    self.var["polar_mask_cols"].set(int(pk["polar_mask_cols"]))
                if "com_centering" in pk:
                    self.var["com_centering"].set(bool(pk["com_centering"]))
            if hasattr(self, "_update_defaults_summary"):
                self._update_defaults_summary()
            if hasattr(self, "_refresh_pair_label_info"):
                self._refresh_pair_label_info()
        except Exception:
            # never crash the load path because of a UI sync hiccup
            pass

    # ---- variable factory ----
    def _build_vars(self):
        d = PAPER_DEFAULTS
        return dict(
            variant=ctk.StringVar(value=d["variant"]),
            sample=ctk.StringVar(value="Na007b"),
            run_name=ctk.StringVar(value=""),  # blank = timestamp
            K=ctk.IntVar(value=d["K"]),
            epochs=ctk.IntVar(value=d["epochs"]),
            batch_size=ctk.IntVar(value=d["batch_size"]),
            lr=ctk.DoubleVar(value=d["lr"]),
            weight_decay=ctk.DoubleVar(value=d["weight_decay"]),
            T0=ctk.DoubleVar(value=d["T0"]),
            Tfin=ctk.DoubleVar(value=d["Tfin"]),
            warmup_frac=ctk.DoubleVar(value=d["warmup_frac"]),
            center_momentum=ctk.DoubleVar(value=d["center_momentum"]),
            EMA0=ctk.DoubleVar(value=d["EMA0"]),
            EMAfin=ctk.DoubleVar(value=d["EMAfin"]),
            n_layers=ctk.IntVar(value=d["n_layers"]),
            cluster1d_lambda=ctk.DoubleVar(value=d["cluster1d_lambda"]),
            cluster1d_lambda_intra=ctk.DoubleVar(
                value=d["cluster1d_lambda_intra"]),
            cluster1d_lambda_inter=ctk.DoubleVar(
                value=d["cluster1d_lambda_inter"]),
            cluster1d_margin=ctk.DoubleVar(value=d["cluster1d_margin"]),
            cluster1d_warmup_frac=ctk.DoubleVar(value=d["cluster1d_warmup_frac"]),
            cluster1d_ramp_frac=ctk.DoubleVar(value=d["cluster1d_ramp_frac"]),
            conf_weight_gamma=ctk.DoubleVar(value=d["conf_weight_gamma"]),
            supcon_lambda=ctk.DoubleVar(value=d["supcon_lambda"]),
            supcon_temperature=ctk.DoubleVar(value=d["supcon_temperature"]),
            centroid_lambda=ctk.DoubleVar(value=d["centroid_lambda"]),
            centroid_margin=ctk.DoubleVar(value=d["centroid_margin"]),
            lam_spatial=ctk.DoubleVar(value=d["lam_spatial"]),
            spatial_tau_pos=ctk.DoubleVar(value=d["spatial_tau_pos"]),
            spatial_tau_neg=ctk.DoubleVar(value=d["spatial_tau_neg"]),
            polar_size=ctk.IntVar(value=d["polar_size"]),
            polar_mask_cols=ctk.IntVar(value=d["polar_mask_cols"]),
            center_crop_size=ctk.IntVar(value=d["center_crop_size"]),
            center_mask_radius=ctk.IntVar(value=d["center_mask_radius"]),
            com_centering=ctk.BooleanVar(value=d["com_centering"]),
            com_search_radius_factor=ctk.DoubleVar(value=d["com_search_radius_factor"]),
            theta_shift_student=ctk.IntVar(value=d["theta_shift_student"]),
            theta_shift_teacher=ctk.IntVar(value=d["theta_shift_teacher"]),
            aug_hflip=ctk.BooleanVar(value=d["aug_hflip"]),
            aug_vflip=ctk.BooleanVar(value=d["aug_vflip"]),
            aug_colorjitter=ctk.BooleanVar(value=d["aug_colorjitter"]),
            aug_blur=ctk.BooleanVar(value=d["aug_blur"]),
            cj_brightness=ctk.DoubleVar(value=d["cj_brightness"]),
            cj_contrast=ctk.DoubleVar(value=d["cj_contrast"]),
            blur_kernel_max=ctk.IntVar(value=d["blur_kernel_max"]),
            blur_sigma_max=ctk.DoubleVar(value=d["blur_sigma_max"]),
            save_every=ctk.IntVar(value=d["save_every"]),
            seed=ctk.IntVar(value=d["seed"]),
            # Pair-supervised assignment loss
            lambda_pair=ctk.DoubleVar(value=d["lambda_pair"]),
            pair_entropy_reg=ctk.DoubleVar(value=d["pair_entropy_reg"]),
            pair_per_batch=ctk.IntVar(value=d["pair_per_batch"]),
        )

    # ---- UI ----
    def _build(self):
        # Top: sample + recipe toggle + run name
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)

        ctk.CTkLabel(top, text="Sample:").pack(side="left", padx=(6, 4))
        self._sample_menu = ctk.CTkOptionMenu(top, variable=self.var["sample"],
                                                values=self._sample_keys, width=240,
                                                command=lambda _v: (
                                                    self._refresh_pair_label_info()
                                                    if hasattr(self, "_refresh_pair_label_info")
                                                    else None))
        self._sample_menu.pack(side="left")

        ctk.CTkLabel(top, text="  Run name (optional):").pack(side="left",
                                                                padx=(12, 4))
        ctk.CTkEntry(top, textvariable=self.var["run_name"], width=150
                      ).pack(side="left")
        h = add_help_button(top,
            "Folder name under runs/_gui/. Leave blank to use a "
            "timestamp.")
        h.pack(side="left", padx=(4, 12))

        ctk.CTkSwitch(top, text="Use paper recipe (locked)",
                       variable=self.use_defaults,
                       command=self._sync_defaults_lock,
                       font=("Segoe UI", 11, "bold")).pack(side="left",
                                                            padx=12)

        # Body: when defaults ON show summary; OFF show sub-tabs
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        # 'use defaults' summary panel
        self._defaults_box = ctk.CTkFrame(body)
        ctk.CTkLabel(self._defaults_box,
            text="Paper recipe (locked).  Click the switch above to edit.",
            font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10,
                                                  pady=(8, 0))
        self._defaults_summary = ctk.CTkLabel(self._defaults_box,
            text="", justify="left", font=("Consolas", 10))
        self._defaults_summary.pack(anchor="w", padx=12, pady=8)
        self._update_defaults_summary()

        # custom-mode sub-tabs
        self._tabs = ctk.CTkTabview(body, anchor="nw")
        self._build_core_tab(self._tabs.add("Core"))
        self._build_schedule_tab(self._tabs.add("Schedule"))
        self._build_aug_tab(self._tabs.add("Augmentation"))
        self._build_polar_tab(self._tabs.add("Polar pipeline"))

        # Bottom: status + run + plot
        bottom = ctk.CTkFrame(self)
        bottom.pack(side="bottom", fill="x", padx=6, pady=(0, 6))

        run_row = ctk.CTkFrame(bottom, fg_color="transparent")
        run_row.pack(side="top", fill="x", pady=4)
        self._apply_btn = ctk.CTkButton(run_row, text="Apply changes",
                                          width=130, height=32,
                                          font=("Segoe UI", 11, "bold"),
                                          fg_color=("#5070A0", "#3a5a8a"),
                                          command=self._on_apply)
        self._apply_btn.pack(side="left", padx=4)
        self._train_btn = ctk.CTkButton(run_row, text="Train",
                                          width=120, height=32,
                                          font=("Segoe UI", 12, "bold"),
                                          fg_color=("#2D7A2D", "#1F7A1F"),
                                          command=self._on_train)
        self._train_btn.pack(side="left", padx=4)
        self._stop_btn = ctk.CTkButton(run_row, text="Stop",
                                         width=90, height=32,
                                         font=("Segoe UI", 12, "bold"),
                                         fg_color=("#A02020", "#7A1010"),
                                         hover_color=("#C04040", "#A03030"),
                                         state="disabled",
                                         command=self._on_stop)
        self._stop_btn.pack(side="left", padx=4)
        self._status_lbl = ctk.CTkLabel(run_row, text="Idle.",
                                          font=("Consolas", 10))
        self._status_lbl.pack(side="left", padx=8)
        # Second row: applied-snapshot status
        snap_row = ctk.CTkFrame(bottom, fg_color="transparent")
        snap_row.pack(side="top", fill="x", padx=4, pady=(0, 4))
        self._apply_status = ctk.CTkLabel(snap_row,
            text="(no params applied yet — Train will auto-apply current "
                  "entries)",
            font=("Consolas", 9),
            text_color=("#666", "#888"))
        self._apply_status.pack(side="left", padx=4)

        # plot
        plot_holder = ctk.CTkFrame(bottom)
        plot_holder.pack(side="top", fill="both", expand=True)
        self._fig = Figure(figsize=(9, 3.0), dpi=95, facecolor="#f4f4f4")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel("epoch"); self._ax.set_ylabel("loss")
        self._fig.tight_layout()
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_holder)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def _update_defaults_summary(self):
        d = PAPER_DEFAULTS
        s = (
            f"  variant       :  DINO + cluster1d  (1D)\n"
            f"  K (prototypes):  {d['K']}\n"
            f"  epochs        :  {d['epochs']}\n"
            f"  loss recipe   :  L = L_DINO + λ_intra·L_1d_intra + λ_inter·L_1d_inter\n"
            f"                  λ_intra = {d['cluster1d_lambda_intra']}, "
            f"λ_inter = {d['cluster1d_lambda_inter']}, "
            f"margin = {d['cluster1d_margin']}, γ = {d['conf_weight_gamma']}\n"
            f"  teacher temp  :  T0 = {d['T0']} → Tfin = {d['Tfin']}, "
            f"warmup_frac = {d['warmup_frac']}\n"
            f"  optimizer     :  AdamW(lr={d['lr']}, wd={d['weight_decay']})\n"
            f"  polar         :  {d['polar_size']}², mask_cols = "
            f"{d['polar_mask_cols']}, center_crop = {d['center_crop_size']}\n"
            f"  beam mask     :  r = {d['center_mask_radius']} px,  "
            f"COM-center: {d['com_centering']}\n"
            f"  augmentation  :  θ-roll  student/teacher = "
            f"{d['theta_shift_student']}/{d['theta_shift_teacher']}, "
            f"hflip/vflip/colorjitter = OFF\n"
            f"  determinism   :  seed = {d['seed']}, cudnn deterministic, "
            f"Tensor.max(dim=2) global pool\n"
            f"  ckpt cadence  :  every {d['save_every']} epochs"
        )
        self._defaults_summary.configure(text=s)

    def _sync_defaults_lock(self):
        if self.use_defaults.get():
            self._tabs.pack_forget()
            self._defaults_box.pack(fill="both", expand=True, padx=4, pady=4)
        else:
            self._defaults_box.pack_forget()
            self._tabs.pack(fill="both", expand=True, padx=4, pady=4)

    # --- sub-tab builders ---
    def _build_core_tab(self, tab):
        # Wrap in a scrollable frame so the sub-tab scrolls when its
        # content overflows the window height (the Train button at
        # the bottom of the parent stays reachable regardless).
        sf = ctk.CTkScrollableFrame(tab); sf.pack(fill="both", expand=True)
        tab = sf
        _seg(tab, "Model variant")
        var_row = ctk.CTkFrame(tab, fg_color="transparent")
        var_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(var_row, text="variant", width=220, anchor="w"
                      ).pack(side="left")
        m = ctk.CTkOptionMenu(var_row, variable=self.var["variant"],
                                values=list(VARIANTS.keys()), width=160,
                                command=lambda _v: self._sync_variant_visibility())
        m.pack(side="left")
        h = add_help_button(var_row,
            "1D = DINO + cluster1d (paper recipe). "
            "Contrastive1D = DINO + radial-gated SupCon. "
            "Vanilla = pure DINO. "
            "+centroid: embedding-centroid loss. "
            "+weight: γ-confidence weighting on DINO. "
            "+spatial: 4-neighbor Potts regularizer.")
        h.pack(side="left", padx=(8, 0))

        # Two-column layout: Optimizer & training on the left, DINO
        # hyperparameters on the right.  Loss knobs continue full-width
        # below.
        two_col = ctk.CTkFrame(tab, fg_color="transparent")
        two_col.pack(fill="x", padx=2, pady=2)
        left  = ctk.CTkFrame(two_col, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = ctk.CTkFrame(two_col, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        _seg(left, "Optimizer & training")
        _entry_row(left, "K (prototypes)", self.var["K"],
                    "Number of prototype clusters. K=6 was used in the paper.")
        _entry_row(left, "epochs", self.var["epochs"],
                    "Training epochs. Paper uses 30.")
        _entry_row(left, "batch_size", self.var["batch_size"],
                    "Per-step batch. Lower if GPU OOM.")
        _entry_row(left, "lr", self.var["lr"],
                    "AdamW learning rate. Paper: 3e-4.")
        _entry_row(left, "weight_decay", self.var["weight_decay"],
                    "AdamW weight decay. Paper: 1e-6.")
        _entry_row(left, "save_every (epochs)", self.var["save_every"],
                    "Checkpoint cadence. The Eval tab refreshes its "
                    "live class map on every checkpoint. Min 5.")
        _entry_row(left, "seed", self.var["seed"],
                    "Random seed. Paper: 42.")

        _seg(right, "DINO hyperparameters")
        _entry_row(right, "T0 (start)", self.var["T0"],
            "Initial teacher softmax temperature. Paper: 0.04. "
            "Lower = sharper teacher targets early.")
        _entry_row(right, "Tfin (end)", self.var["Tfin"],
            "Final teacher temperature after warmup. Paper: 0.07.")
        _entry_row(right, "warmup_frac", self.var["warmup_frac"],
            "Fraction of total epochs used for the teacher-temp "
            "warmup ramp from T0 to Tfin. Paper: 0.2.")
        _entry_row(right, "center_momentum", self.var["center_momentum"],
            "EMA momentum on the DINO 'center' (anti-collapse). "
            "Paper: 0.9. Higher (e.g. 0.99) makes centering more "
            "sluggish — weaker prototype rescue, but slower to "
            "respond to drift. The user has chosen NOT to tune "
            "this for the paper recipe.")
        _entry_row(right, "EMA0 (teacher start)", self.var["EMA0"],
            "Initial EMA momentum for teacher params. Paper: 0.990. "
            "Higher = teacher tracks student more slowly.")
        _entry_row(right, "EMAfin (teacher end)", self.var["EMAfin"],
            "Final EMA momentum for teacher params. Paper: 0.999.")
        _entry_row(right, "n_layers (ResNet18 depth)", self.var["n_layers"],
            "How many ResNet18 stages to keep in the backbone.\n"
            "  1 = layer1 only  (default; paper recipe)\n"
            "  2 = layer1+layer2\n"
            "  3 = layer1+layer2+layer3\n"
            "  4 = full ResNet18  (NOT recommended — overfits + much "
            "slower; the paper uses 1).")

        _seg(tab, "Loss-specific knobs (active for the chosen variant)")
        self._loss_rows = {}
        self._loss_rows["cluster1d_lambda_intra"] = self._loss_entry_row(
            tab, "cluster1d λ intra", "cluster1d_lambda_intra",
            "Weight of cluster1d INTRA term: pulls each pattern's "
            "1D radial profile toward its assigned cluster's "
            "centroid. Physics-baked. Paper used 0.1.")
        self._loss_rows["cluster1d_lambda_inter"] = self._loss_entry_row(
            tab, "cluster1d λ inter", "cluster1d_lambda_inter",
            "Weight of cluster1d INTER term: pushes pairs of "
            "cluster centroids APART (margin-based). Set to 0 to "
            "let weak prototypes drift toward heavy ones and die "
            "naturally — `hard_K_active` will fall to the data-"
            "driven natural K. Paper used 0.1 (matched intra).")
        self._loss_rows["cluster1d_margin"] = self._loss_entry_row(tab,
            "cluster1d margin", "cluster1d_margin",
            "Hinge margin for inter-cluster centroid repulsion. "
            "Only relevant if cluster1d λ inter > 0.")
        self._loss_rows["conf_weight_gamma"] = self._loss_entry_row(tab,
            "γ (conf weight)", "conf_weight_gamma",
            "γ-confidence weighting on per-sample DINO loss. "
            "γ=0 disables. Helps layered samples.")
        self._loss_rows["supcon_lambda"] = self._loss_entry_row(tab,
            "supcon λ", "supcon_lambda",
            "Weight of radial-gated SupCon loss (Contrastive1D variant).")
        self._loss_rows["supcon_temperature"] = self._loss_entry_row(tab,
            "supcon τ", "supcon_temperature",
            "InfoNCE temperature for SupCon.")
        self._loss_rows["centroid_lambda"] = self._loss_entry_row(tab,
            "centroid λ", "centroid_lambda",
            "Weight of embedding-centroid loss (+centroid variant).")
        self._loss_rows["lam_spatial"] = self._loss_entry_row(tab,
            "spatial λ", "lam_spatial",
            "Weight of GATED 4-neighbor Potts loss (+spatial variant). "
            "When >0, a tile of ~batch_size patterns sweeps the scan "
            "row-major (one tile per epoch) and contributes a 4-neighbor "
            "loss GATED by 1D-radial agreement: pull together when "
            "neighbors share a 1D profile, push apart when they don't. "
            "Fixes 'spatial-close, physics-different' merging (e.g. "
            "vacuum vs lacy carbon). Requires precomputed radials.")
        self._loss_rows["spatial_tau_pos"] = self._loss_entry_row(tab,
            "  ↳ spatial τ_pos", "spatial_tau_pos",
            "Pull threshold: 4-neighbor pairs with 1D cosine ABOVE "
            "this get pull pressure (assignments forced together). "
            "Sentinel -2 = auto-load from per-sample "
            "gate_thresholds.json (same calibration as supcon). "
            "Set explicit value to override (e.g. 0.5).")
        self._loss_rows["spatial_tau_neg"] = self._loss_entry_row(tab,
            "  ↳ spatial τ_neg", "spatial_tau_neg",
            "Push threshold: 4-neighbor pairs with 1D cosine BELOW "
            "this get push pressure (assignments forced apart). "
            "Sentinel -2 = auto-load from gate_thresholds.json. "
            "Pairs in [τ_neg, τ_pos] dead band contribute nothing.")

        _seg(tab, "Pair-supervised loss (semi-supervised, optional)")
        self._loss_rows["lambda_pair"] = self._loss_entry_row(tab,
            "pair λ", "lambda_pair",
            "Weight of pair_assignment_loss. 0 = off (default). "
            "Set 0.05–0.2 if you have pre-train labels and want them "
            "to nudge the unsupervised solution. Labels are loaded from "
            "<basename>.pair_labels.json next to the cube.")
        self._loss_rows["pair_entropy_reg"] = self._loss_entry_row(tab,
            "pair entropy reg", "pair_entropy_reg",
            "Optional non-negative entropy bonus on the labelled "
            "pair members. Counteracts the pair loss's pull toward "
            "one-hot when labels are sparse. Try 0.01 if pairs collapse.")
        self._loss_rows["pair_per_batch"] = self._loss_entry_row(tab,
            "pair per batch", "pair_per_batch",
            "How many random labelled pairs to forward per batch. "
            "Each adds 2 forward passes. 32 is a good default.")
        # Apply-default button + info line.  Apply explicitly sets
        # λ_pair = 0.1 if labels exist; nothing happens silently.
        pair_apply_row = ctk.CTkFrame(tab, fg_color="transparent")
        pair_apply_row.pack(fill="x", padx=4, pady=(2, 0))
        ctk.CTkButton(pair_apply_row,
            text="Apply auto-default (λ_pair = 0.1)",
            width=240,
            command=self._apply_pair_default
            ).pack(side="left", padx=4)
        self._pair_apply_status = ctk.CTkLabel(pair_apply_row, text="",
            font=("Consolas", 9),
            text_color=("#444", "#aaa"))
        self._pair_apply_status.pack(side="left", padx=8)
        self._pair_label_info = ctk.CTkLabel(tab,
            text="(no sample selected)",
            font=("Consolas", 9), justify="left",
            text_color=("#444", "#aaa"))
        self._pair_label_info.pack(anchor="w", padx=8, pady=(2, 6))

        self._sync_variant_visibility()
        self._refresh_pair_label_info()

    def _apply_pair_default(self):
        """Explicit "apply auto-default" — sets λ_pair to 0.1 if there
        are labels for the active sample. Without this button, the
        user's λ_pair value is never auto-changed (so an explicit 0
        sticks across sample changes / re-loads)."""
        try:
            sample = self.var["sample"].get()
            cfg = SAMPLES.get(sample, {})
            path = cfg.get("path") or (cfg.get("paths") or [None])[0]
            if not path:
                self._pair_apply_status.configure(
                    text="(no path)"); return
            from gui_app.pair_labels import (
                label_path_for_cube, load_pair_labels, label_count)
            sp = label_path_for_cube(path)
            if not os.path.exists(sp):
                self._pair_apply_status.configure(
                    text="(no labels file)")
                return
            c = label_count(load_pair_labels(path))
            if c["total"] == 0:
                self._pair_apply_status.configure(
                    text="(0 pairs in file)")
                return
            self.var["lambda_pair"].set(0.1)
            self._pair_apply_status.configure(
                text=f"applied: λ_pair = 0.1  ({c['total']} pairs)")
            self._refresh_pair_label_info()
        except Exception as e:
            self._pair_apply_status.configure(text=f"error: {e}")

    def _refresh_pair_label_info(self):
        """Update the small pair-label-count line under the λ_pair knob.
        Called on sample-change, runtime-sample-add, and once at build."""
        if not getattr(self, "_pair_label_info", None):
            return
        try:
            sample = self.var["sample"].get()
            cfg = SAMPLES.get(sample)
            if not cfg:
                self._pair_label_info.configure(
                    text="(unknown sample)")
                return
            path = cfg.get("path") or (cfg.get("paths") or [None])[0]
            if not path:
                self._pair_label_info.configure(
                    text="(sample has no path)")
                return
            from gui_app.pair_labels import (
                label_path_for_cube, load_pair_labels, label_count)
            sp = label_path_for_cube(path)
            if not os.path.exists(sp):
                self._pair_label_info.configure(
                    text=f"no pair-labels sidecar at {os.path.basename(sp)}\n"
                          f"(go to Pre-processing → 'Label pairs…' to "
                          f"create one)")
                return
            d = load_pair_labels(path)
            c = label_count(d)
            cur = float(self.var.get("lambda_pair").get()
                          if hasattr(self.var.get("lambda_pair"), "get")
                          else 0.0)
            # NO silent auto-set anymore. The label info line just
            # describes the current state; if labels exist and
            # λ_pair = 0, the user can press the "Apply auto-default"
            # button to set λ_pair = 0.1 explicitly.
            if c["total"] > 0:
                if cur > 0:
                    note = f"  →  λ_pair = {cur:.3g}  (will be applied)"
                else:
                    note = ("  →  λ_pair = 0  (labels NOT used; "
                            "click 'Apply auto-default' or set "
                            "λ_pair > 0 manually)")
            else:
                note = ""
            self._pair_label_info.configure(
                text=f"pair-labels: {c['total']} total  "
                      f"({c['same']} same / {c['diff']} diff)  "
                      f"@ {os.path.basename(sp)}{note}")
        except Exception as e:
            self._pair_label_info.configure(
                text=f"(label-info error: {e})")

    def _loss_entry_row(self, tab, label, key, help_text):
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(row, text=label, width=220, anchor="w"
                      ).pack(side="left")
        e = ctk.CTkEntry(row, textvariable=self.var[key], width=110)
        e.pack(side="left")
        h = add_help_button(row, help_text); h.pack(side="left", padx=(8, 0))
        return row

    def _sync_variant_visibility(self):
        v = VARIANTS[self.var["variant"].get()]
        # When the user picks a variant that turns a loss term ON,
        # auto-bump the term's λ from 0 to a sensible default so the
        # variant actually trains the implied loss. Existing non-zero
        # values are preserved (so user overrides survive variant
        # flips). This is the "preset behaviour" the user expects when
        # picking from the dropdown.
        for flag, lam_keys in VARIANT_FLAG_TO_LAMBDAS.items():
            if not v.get(flag, False):
                continue
            for lam_key in lam_keys:
                if lam_key not in self.var:
                    continue
                try:
                    cur = float(self.var[lam_key].get())
                except Exception:
                    continue
                if cur == 0.0 and lam_key in VARIANT_LAMBDA_DEFAULTS:
                    self.var[lam_key].set(VARIANT_LAMBDA_DEFAULTS[lam_key])
        # cluster1d rows visible if cluster1d active OR γ used (1D uses both)
        for key, active in [
            ("cluster1d_lambda_intra", v["cluster1d"]),
            ("cluster1d_lambda_inter", v["cluster1d"]),
            ("cluster1d_margin", v["cluster1d"]),
            ("supcon_lambda", v["supcon"]),
            ("supcon_temperature", v["supcon"]),
            ("centroid_lambda", v["centroid"]),
            ("conf_weight_gamma", v["weight"] or v["cluster1d"]),
            ("lam_spatial", v["spatial"]),
            ("spatial_tau_pos", v["spatial"]),
            ("spatial_tau_neg", v["spatial"]),
        ]:
            if key in self._loss_rows:
                self._loss_rows[key].pack_forget()
                if active:
                    self._loss_rows[key].pack(fill="x", padx=4, pady=2)

    def _build_schedule_tab(self, tab):
        sf = ctk.CTkScrollableFrame(tab); sf.pack(fill="both", expand=True)
        tab = sf
        # Teacher temperature schedule (T0/Tfin/warmup_frac) and other
        # DINO hyperparams now live in the Core sub-tab next to the
        # optimizer params.  This Schedule sub-tab is only for the
        # cluster1d λ ramp.
        _seg(tab, "Cluster1d λ schedule (only used if variant=1D)")
        _entry_row(tab, "warmup_frac", self.var["cluster1d_warmup_frac"],
                    "Fraction of epochs at λ=0 before ramping up.")
        _entry_row(tab, "ramp_frac", self.var["cluster1d_ramp_frac"],
                    "Fraction of epochs spent linearly ramping λ to its "
                    "target value.")

    def _build_aug_tab(self, tab):
        sf = ctk.CTkScrollableFrame(tab); sf.pack(fill="both", expand=True)
        tab = sf
        _seg(tab, "θ-roll (rotation invariance)")
        _entry_row(tab, "student shift range", self.var["theta_shift_student"],
                    "Maximum θ-roll for the student view. 192 = full 360°.")
        _entry_row(tab, "teacher shift range", self.var["theta_shift_teacher"],
                    "Maximum θ-roll for the teacher view. Paper: 16.")
        _seg(tab, "Other augmentations")
        _check_row(tab, "Horizontal flip", self.var["aug_hflip"],
                    "Random hflip. Disabled in the paper recipe.")
        _check_row(tab, "Vertical flip", self.var["aug_vflip"],
                    "Random vflip. Disabled in the paper recipe.")
        _check_row(tab, "Color jitter", self.var["aug_colorjitter"],
                    "Brightness/contrast jitter. Disabled in the paper "
                    "recipe (unphysical for diffraction).")
        # Color-jitter params (used only when checkbox is on)
        _entry_row(tab, "  ↳ brightness", self.var["cj_brightness"],
            "Strength of brightness perturbation (uniform in "
            "[1-x, 1+x]). 0 = no perturbation.", width=70)
        _entry_row(tab, "  ↳ contrast", self.var["cj_contrast"],
            "Strength of contrast perturbation (same uniform "
            "[1-x, 1+x]). 0 = no perturbation.", width=70)

        _check_row(tab, "Gaussian blur", self.var["aug_blur"],
                    "Random Gaussian blur. ENABLED in the paper recipe "
                    "(noise robustness). Apply a Gaussian kernel of "
                    "random odd size in [3, max_kernel] and σ uniform "
                    "in [0.1, max_sigma].")
        _entry_row(tab, "  ↳ max kernel (odd)", self.var["blur_kernel_max"],
            "Upper bound on kernel size. Must be odd; 5 = paper "
            "default. Larger = more aggressive smoothing.",
            width=70)
        _entry_row(tab, "  ↳ max σ", self.var["blur_sigma_max"],
            "Upper bound on Gaussian σ. 0.3 = paper default. "
            "Larger = more aggressive smoothing.", width=70)

    def _build_polar_tab(self, tab):
        sf = ctk.CTkScrollableFrame(tab); sf.pack(fill="both", expand=True)
        tab = sf
        _seg(tab, "Polar transform")
        _entry_row(tab, "polar_size", self.var["polar_size"],
                    "Output side length of the polar tensor. Paper: 192.")
        _entry_row(tab, "polar_mask_cols", self.var["polar_mask_cols"],
                    "Number of leftmost columns of the polar tensor "
                    "(small radii, central beam) to zero out. Paper: 45. "
                    "Independent of PrePanel's beam-mask radius (which "
                    "is Cartesian); polar_mask_cols zeros the inner "
                    "POLAR columns.")
        _seg(tab, "Cartesian crop & beam mask")
        _entry_row(tab, "center_crop_size", self.var["center_crop_size"],
                    "Cartesian crop size before resizing to polar_size. "
                    "Paper: 140.")
        _entry_row(tab, "center_mask_radius", self.var["center_mask_radius"],
                    "Radius of the central beam mask in 192-resized space. "
                    "Paper: 15.  Pulled from PrePanel ONCE at sample-load; "
                    "if you change PrePanel's beam-mask AFTER load, click "
                    "the refresh button below to re-sync.")
        _check_row(tab, "COM-center the direct beam",
                    self.var["com_centering"],
                    "Shift each pattern so the direct beam lies at "
                    "(96, 96) before the polar transform.")
        _entry_row(tab, "COM search radius factor",
                    self.var["com_search_radius_factor"],
                    "Multiplier on center_mask_radius for the COM "
                    "search region. Paper: 2.0.")
        # Manual sync button — pulls current PrePanel state into the
        # Training vars (center_crop_size, center_mask_radius,
        # com_centering). Useful after the user adjusts PrePanel
        # mid-workflow.
        sync_row = ctk.CTkFrame(tab, fg_color="transparent")
        sync_row.pack(fill="x", padx=4, pady=(8, 4))
        ctk.CTkButton(sync_row,
            text="↻ Refresh crop/mask/COM from PrePanel",
            width=320,
            command=self._sync_from_prepanel
            ).pack(side="left", padx=4)
        self._prepanel_sync_status = ctk.CTkLabel(sync_row, text="",
            font=("Consolas", 9), text_color=("#444", "#aaa"))
        self._prepanel_sync_status.pack(side="left", padx=8)

    def _sync_from_prepanel(self):
        """Pull the live PrePanel values for crop / beam-mask / COM
        into this tab's vars. Used when the user has changed those
        sliders AFTER first loading the sample (the sample-load
        callback already does this once, automatically)."""
        if self.prepanel is None:
            try: self._prepanel_sync_status.configure(
                text="(no PrePanel reference)")
            except Exception: pass
            return
        try:
            pk = self.prepanel.get_pre_kwargs()
            n_changed = 0
            for k_pre, k_train in (
                ("center_mask_radius", "center_mask_radius"),
                ("center_crop_size",   "center_crop_size"),
                ("polar_mask_cols",    "polar_mask_cols"),
                ("com_centering",      "com_centering"),
            ):
                if k_pre in pk:
                    cur = self.var[k_train].get()
                    new = (int(pk[k_pre]) if isinstance(cur, int)
                            else (bool(pk[k_pre])
                                   if isinstance(cur, bool)
                                   else float(pk[k_pre])))
                    if cur != new:
                        self.var[k_train].set(new); n_changed += 1
            if n_changed:
                self._prepanel_sync_status.configure(
                    text=f"synced {n_changed} value(s) from PrePanel")
            else:
                self._prepanel_sync_status.configure(
                    text="already in sync with PrePanel")
        except Exception as e:
            self._prepanel_sync_status.configure(
                text=f"sync error: {e}")

    # ---- Train action ----
    # ---- master Apply button ----
    @staticmethod
    def _vals_equal(a, b):
        try:
            if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                return abs(float(a) - float(b)) < 1e-9
        except Exception:
            pass
        return a == b

    def _capture_snapshot(self) -> dict:
        """Read every var's current value into a plain dict. Used by
        the Apply button and as the auto-apply fallback when Train
        is clicked without a prior Apply."""
        snap = {}
        for k, v in self.var.items():
            try:
                snap[k] = v.get()
            except Exception:
                snap[k] = None
        return snap

    def _on_apply(self):
        """Capture current entry values into `_applied_snapshot`. Train
        builds its run config from the snapshot, NOT the live entries
        — so subsequent edits don't affect already-running or
        about-to-launch runs unless you Apply again."""
        snap = self._capture_snapshot()
        self._applied_snapshot = snap
        from datetime import datetime as _dt
        stamp = _dt.now().strftime("%H:%M:%S")
        self._applied_at = stamp
        # Diff vs PAPER_DEFAULTS for visible reassurance
        diffs = []
        for k, val in snap.items():
            if k in PAPER_DEFAULTS:
                if not self._vals_equal(val, PAPER_DEFAULTS[k]):
                    diffs.append(f"{k}={val}")
        if not diffs:
            self._apply_status.configure(
                text=f"✓ applied @ {stamp}  —  no changes from paper "
                      f"defaults  (Train will use the paper recipe values)",
                text_color=("#2D7A2D", "#7AC07A"))
        else:
            n = len(diffs)
            preview = "; ".join(diffs[:4])
            more = f"  +{n-4} more" if n > 4 else ""
            self._apply_status.configure(
                text=f"✓ applied @ {stamp}  —  {n} change(s):  "
                      f"{preview}{more}",
                text_color=("#2D7A2D", "#7AC07A"))

    def _gather_kwargs(self) -> dict:
        d = self.var
        if self.use_defaults.get():
            P = PAPER_DEFAULTS
            chosen = dict(P)
        else:
            # Custom mode: ALWAYS re-capture the live entries on Train,
            # so edits made after the last explicit Apply still take
            # effect for the new run.  (The previous behaviour of
            # locking-in the prior snapshot caused: edit → Train →
            # nothing changed because a stale snapshot was reused.)
            #
            # The snapshot still drives the diff display + the "RUNNING"
            # status line — we just refresh it here so Train can never
            # use stale values.  Mid-run edits are still safe because
            # the kwargs dict is serialised to `_train_kwargs.json` at
            # spawn time and the subprocess reads from that file.
            self._on_apply()
            chosen = dict(self._applied_snapshot)
        # determine sample
        sample = d["sample"].get()
        # decide aug list
        aug_disable = []
        for tag, key in (("hflip", "aug_hflip"),
                          ("vflip", "aug_vflip"),
                          ("colorjitter", "aug_colorjitter"),
                          ("blur", "aug_blur")):
            if not bool(chosen.get(key, False)):
                aug_disable.append(tag)
        # radials path (only if variant uses cluster1d or supcon)
        cfg = SAMPLES[sample]
        path = cfg.get("path") or (cfg.get("paths") or [None])[0]
        rad_path = None; th_path = None
        v = VARIANTS[chosen["variant"]]
        # Spatial loss is gated by 1D radials too, so set the radials
        # path whenever ANY radial-using term is enabled.
        if (v["cluster1d"] or v["supcon"] or v["spatial"]) and path:
            base = path[:-4] if path.endswith(".prz") else path
            rad_path = base + ".radial.npy"
            th_path = base + ".gate_thresholds.json"
        # pair-labels sidecar — attach the file path if labels exist
        # so the loss CAN load them, but let the user's λ_pair value
        # be authoritative.  The visible auto-set (in
        # `_refresh_pair_label_info`) already pre-fills 0.1 when
        # labels exist + entry is 0.  If the user types 0 after that
        # to disable, the explicit 0 here is honoured (no silent
        # bump).
        pair_path = None
        lambda_pair_used = float(chosen.get("lambda_pair", 0.0))
        if path and lambda_pair_used > 0.0:
            try:
                from gui_app.pair_labels import (
                    label_path_for_cube, load_pair_labels, label_count)
                cand = label_path_for_cube(path)
                if os.path.exists(cand):
                    cnt = label_count(load_pair_labels(path))
                    if cnt["total"] > 0:
                        pair_path = cand
                        print(
                            f"[train] pair loss enabled: "
                            f"{cnt['total']} labels @ {cand}  →  "
                            f"λ_pair={lambda_pair_used}", flush=True)
            except Exception:
                pass
        # construct full kwargs for run_config
        warmup_epochs = int(round((2.0 / 3.0) * chosen["epochs"]))
        ramp_epochs = int(round((1.0 / 3.0) * chosen["epochs"]))
        return dict(
            sample=sample,
            kwargs=dict(
                epochs=int(chosen["epochs"]),
                seed=int(chosen["seed"]),
                batch_size=int(chosen["batch_size"]),
                lr=float(chosen["lr"]),
                weight_decay=float(chosen["weight_decay"]),
                num_prototypes=int(chosen["K"]),
                t0=float(chosen["T0"]), tfin=float(chosen["Tfin"]),
                center_momentum=float(chosen.get("center_momentum", 0.9)),
                EMA0=float(chosen.get("EMA0", 0.990)),
                EMAfin=float(chosen.get("EMAfin", 0.999)),
                warmup_frac=float(chosen.get("warmup_frac", 0.2)),
                warmup_epochs=warmup_epochs, ramp_epochs=ramp_epochs,
                entropy_gate=False,
                projection_dim=128, projection_hidden=256,
                theta_shift_range=None,
                theta_shift_range_student=int(chosen["theta_shift_student"]),
                theta_shift_range_teacher=int(chosen["theta_shift_teacher"]),
                center_mask_radius=int(chosen["center_mask_radius"]),
                center_crop_size=int(chosen["center_crop_size"]),
                vmax=None,
                polar_size=int(chosen["polar_size"]),
                polar_mask_cols=int(chosen["polar_mask_cols"]),
                pipeline="polar",
                centroid_lambda=float(chosen["centroid_lambda"]) if v["centroid"] else 0.0,
                centroid_margin=float(chosen["centroid_margin"]),
                conf_weight_gamma=float(chosen["conf_weight_gamma"]) if (v["weight"] or v["cluster1d"]) else 0.0,
                entropy_gate_override=None,
                lam_spatial=float(chosen["lam_spatial"]) if v["spatial"] else 0.0,
                spatial_tau_pos=float(chosen.get("spatial_tau_pos", -2.0)),
                spatial_tau_neg=float(chosen.get("spatial_tau_neg", -2.0)),
                architecture="resnet",
                n_layers=int(chosen.get("n_layers", 1)),
                w_ent=0.0,
                com_centering=bool(chosen["com_centering"]),
                com_search_radius_factor=float(chosen["com_search_radius_factor"]),
                aug_disable=aug_disable,
                cj_brightness=float(chosen.get("cj_brightness", 0.2)),
                cj_contrast=float(chosen.get("cj_contrast", 0.2)),
                blur_kernel_max=int(chosen.get("blur_kernel_max", 5)),
                blur_sigma_max=float(chosen.get("blur_sigma_max", 0.3)),
                supcon_radials_path=rad_path,
                supcon_thresholds_path=th_path,
                supcon_lambda=float(chosen["supcon_lambda"]) if v["supcon"] else 0.0,
                supcon_temperature=float(chosen["supcon_temperature"]),
                # Teacher-target mode (default "dino"). Variants like
                # "SK + 1D" set this to "sinkhorn" to swap centering for
                # Sinkhorn-Knopp equipartition; the rest of the loss
                # profile is whatever the variant flags say.
                target_mode=str(v.get("target_mode", "dino")),
                sinkhorn_eps=float(chosen.get("sinkhorn_eps", 0.05)),
                sinkhorn_iters=int(chosen.get("sinkhorn_iters", 3)),
                contrastive_lambda_override=0.0,
                proto_repel_lambda=0.0, proto_repel_threshold=0.5,
                cluster1d_lambda=float(chosen["cluster1d_lambda"]) if v["cluster1d"] else 0.0,
                cluster1d_lambda_intra=(float(chosen["cluster1d_lambda_intra"])
                                          if v["cluster1d"] else 0.0),
                cluster1d_lambda_inter=(float(chosen["cluster1d_lambda_inter"])
                                          if v["cluster1d"] else 0.0),
                cluster1d_margin=float(chosen["cluster1d_margin"]),
                cluster1d_min_cluster_mass=1.0,
                cluster1d_warmup_frac=float(chosen["cluster1d_warmup_frac"]),
                cluster1d_ramp_frac=float(chosen["cluster1d_ramp_frac"]),
                pair_labels_path=pair_path,
                lambda_pair=float(lambda_pair_used),
                pair_entropy_reg=float(chosen.get("pair_entropy_reg", 0.0)),
                pair_per_batch=int(chosen.get("pair_per_batch", 32)),
                save_every=max(5, int(chosen["save_every"])),
            ),
            run_name=d["run_name"].get().strip(),
        )

    def _on_train(self):
        if self.job is not None and self.job.is_running():
            messagebox.showinfo("Already running",
                "A training job is in progress. Wait for it to finish.")
            return
        try:
            cfg = self._gather_kwargs()
        except Exception as e:
            messagebox.showerror("Bad parameters", str(e)); return
        sample = cfg["sample"]
        if sample not in SAMPLES:
            messagebox.showerror("Sample", f"unknown sample key: {sample}")
            return
        # output dir.  If the user supplied a run_name and a folder
        # with that name already exists (e.g. they stopped a previous
        # run and changed parameters), suffix `_2`, `_3`, … so the new
        # run gets a clean folder — otherwise best.pth, training_log,
        # and _train_kwargs.json from the previous attempt get
        # silently overwritten / mixed with new params.  Auto-stamped
        # runs (run_name == "") are already unique by timestamp.
        run_name = cfg["run_name"]
        if run_name:
            base = os.path.join("runs", "_gui", run_name)
            candidate = base
            suffix = 1
            while os.path.exists(candidate):
                # If the existing folder is genuinely empty (just a
                # placeholder), reuse it.  Otherwise, walk to _2/_3/…
                try:
                    contents = os.listdir(candidate)
                except OSError:
                    contents = ["?"]
                if not contents:
                    break
                suffix += 1
                candidate = f"{base}_{suffix}"
            outdir = candidate
            os.makedirs(outdir, exist_ok=True)
            if outdir != base:
                messagebox.showinfo(
                    "Run name already exists",
                    f"'{base}' already contains a previous run, so "
                    f"this run will be written to:\n\n  {outdir}\n\n"
                    f"(Delete the old folder if you want to reuse "
                    f"the original name.)")
        else:
            outdir = make_outdir()
        # check radials if needed
        kw = cfg["kwargs"]
        if (kw["cluster1d_lambda"] > 0
                or kw.get("cluster1d_lambda_intra", 0) > 0
                or kw.get("cluster1d_lambda_inter", 0) > 0
                or kw["supcon_lambda"] > 0
                or kw.get("lam_spatial", 0) > 0):
            rp = kw["supcon_radials_path"]
            if rp is None or not os.path.exists(rp):
                ok = messagebox.askyesno("Missing radials",
                    f"This variant needs precomputed 1D radials but "
                    f"{rp} does not exist. Compute them now? "
                    f"(may take 1-2 min)")
                if not ok:
                    return
                # compute radials inline (CPU/GPU)
                try:
                    from compute_radial_profile import (
                        compute_radial as compute_radial_for_sample,
                        calibrate_thresholds)
                    import numpy as np, json as jsonlib
                    rad = compute_radial_for_sample(sample)
                    np.save(rp, rad)
                    th = calibrate_thresholds(rad, n_pairs=50_000,
                                                frac_pos=0.15, frac_neg=0.50)
                    th["sample"] = sample
                    with open(kw["supcon_thresholds_path"], "w") as fjs:
                        jsonlib.dump(th, fjs, indent=2)
                except Exception as e:
                    messagebox.showerror("Radial compute failed", str(e))
                    return
        # spawn job
        self.job = TrainingJob(sample=sample, outdir=outdir, kwargs=kw)
        self.job.start()
        self._train_btn.configure(state="disabled")
        self._apply_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._sample_menu.configure(state="disabled")
        self._status_lbl.configure(
            text=f"Training… outdir = {outdir}")
        # Make it explicit: the spec is FROZEN in the subprocess.
        # Edits to entries below won't reach this run.  The message
        # confirms that the user's Apply was captured AND tells them
        # how to start a *new* run with different params.
        running_stamp = self._applied_at or "(auto-applied just now)"
        self._apply_status.configure(
            text=f"⏵ RUNNING — using snapshot from {running_stamp} "
                  f"(your applied params).  Live edits below only "
                  f"affect the NEXT run.",
            text_color=("#A04030", "#E07060"))
        self.on_run_started(outdir=outdir, sample=sample,
                              save_every=kw["save_every"])
        self._poll()

    def _on_stop(self):
        if self.job is None or not self.job.is_running():
            return
        from tkinter import messagebox
        if not messagebox.askyesno("Stop training",
                "Stop the running training? Progress so far will be "
                "saved in the run dir but training will end immediately."):
            return
        self._status_lbl.configure(text="stopping training …")
        self._stop_btn.configure(state="disabled")
        self.job.stop()
        # _poll() will pick up the new status next tick

    def _poll(self):
        if self.job is None:
            return
        # update plot
        rows = read_training_log(self.job.csv_path)
        self.job.note_rows(len(rows))   # anchors training-timer on first row
        self._draw_plot(rows)
        if self.job.is_running():
            n_done = len(rows)
            total = self.job.kwargs.get("epochs", 30)
            if n_done == 0:
                # Still in startup (Python imports, dataset load, model
                # build). Show a separate 'starting up' counter so the user
                # knows the run hasn't stalled.
                self._status_lbl.configure(
                    text=f"Starting up…  {self.job.elapsed():5.0f} s    "
                          f"(loading dataset, building model)    "
                          f"outdir = {self.job.outdir}")
            else:
                t_train = self.job.elapsed_training()
                t_per   = self.job.time_per_epoch(n_done)
                if n_done >= 2 and t_per > 0:
                    pe_str  = f"{t_per:5.1f} s/ep"
                    eta_s   = max(0.0, (total - n_done) * t_per)
                    eta_str = (f"{eta_s/60:4.1f} min" if eta_s >= 60
                               else f"{eta_s:4.0f} s")
                else:
                    pe_str  = "  — s/ep"
                    eta_str = "  —"
                startup = self.job.startup_seconds()
                self._status_lbl.configure(
                    text=(f"Training…  epoch {n_done}/{total}    "
                          f"train-time {t_train:5.0f} s    "
                          f"{pe_str}    eta {eta_str}    "
                          f"(startup {startup:.0f} s)    "
                          f"outdir = {self.job.outdir}"))
            self._poll_after = self.after(1000, self._poll)
        else:
            self._train_btn.configure(state="normal")
            self._apply_btn.configure(state="normal")
            self._stop_btn.configure(state="disabled")
            self._sample_menu.configure(state="normal")
            # Restore the apply-status line to its post-run state.
            stamp = self._applied_at or "(unapplied)"
            self._apply_status.configure(
                text=f"run finished.  last applied @ {stamp}.  edit + "
                      f"Apply + Train for a new run.",
                text_color=("#666", "#888"))
            n_done = len(rows)
            t_train = self.job.elapsed_training()
            t_per   = self.job.time_per_epoch(n_done)
            tail = ""
            if n_done >= 1 and t_train > 0:
                tail = (f"   (training {t_train:.0f} s, "
                        f"~{t_per:.1f} s/ep over {n_done} ep, "
                        f"startup {self.job.startup_seconds():.0f} s)")
            self._status_lbl.configure(
                text=(f"{self.job.status().upper()} after "
                      f"{self.job.elapsed():.0f} s.{tail}    "
                      f"outdir = {self.job.outdir}"))
            if self.job.status() == "failed":
                messagebox.showerror("Training failed",
                    (self.job.error() or "")[:1000])
            self.on_run_finished(outdir=self.job.outdir)

    def _draw_plot(self, rows):
        self._ax.clear()
        if not rows:
            self._ax.set_title("(waiting for first epoch …)")
        else:
            ep = [int(r["epoch"]) for r in rows]
            # (csv_col, label, color, lw, gating_lambda_col)
            # If `gating_lambda_col` is given and is 0 in every row,
            # the line is hidden — the term isn't contributing to
            # the total loss, so plotting its raw value would be
            # misleading.
            cols = [
                ("avg_loss",                "total",      "black", 1.5, None),
                ("avg_loss_dino",           "L_DINO",     "C0",    1.0, None),
                ("avg_loss_supcon",         "L_supcon",   "C2",    1.0, None),
                ("avg_loss_cluster1d_intra","L_1d intra", "C3",    1.0,
                    "lambda_cluster1d_intra_eff"),
                ("avg_loss_cluster1d_inter","L_1d inter", "C4",    1.0,
                    "lambda_cluster1d_inter_eff"),
                ("avg_loss_centroid_intra", "L_cen intra","C5",    1.0, None),
                ("avg_loss_centroid_inter", "L_cen inter","C6",    1.0, None),
                ("avg_loss_spatial",        "L_spatial",  "C7",    1.5,
                    "lambda_spatial_eff"),
                ("avg_loss_spatial_pull",   "L_sp pull",  "C9",    1.0,
                    "lambda_spatial_eff"),
                ("avg_loss_spatial_push",   "L_sp push",  "C1",    1.0,
                    "lambda_spatial_eff"),
                ("avg_loss_pair",           "L_pair",     "C8",    1.0,
                    "lambda_pair_eff"),
            ]
            def _all_zero(col):
                if col is None or col not in rows[0]: return False
                try:
                    return all(float(r[col]) == 0 for r in rows)
                except Exception:
                    return False
            for key, label, color, lw, gate in cols:
                if key not in rows[0]: continue
                if _all_zero(gate):
                    # this term is OFF for every epoch — don't show
                    continue
                try:
                    y = [float(r[key]) for r in rows]
                except Exception:
                    continue
                if any(yy != 0 for yy in y):
                    self._ax.plot(ep, y, label=label, color=color, lw=lw)
            self._ax.legend(fontsize=8, ncol=4, loc="upper right")
        self._ax.set_xlabel("epoch")
        self._ax.set_ylabel("loss")
        self._ax.grid(alpha=0.3)
        self._fig.tight_layout()
        self._canvas.draw_idle()
