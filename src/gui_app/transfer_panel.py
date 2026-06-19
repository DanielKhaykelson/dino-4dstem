"""transfer_panel.py -- Transfer learning tab.

Workflow:
    1. User picks a folder; we recursively walk it for .prz/.npz files
       >= 100 MB (heuristic for "is a datacube").  Smaller .prz are
       metadata sidecars and skipped.
    2. User checks which cubes to include (max 20).
    3. User picks an init checkpoint:
         - dropdown of recent runs (auto-populated from runs/_gui/),
           OR
         - manual browse to any .pth file.
    4. User adjusts a small fine-tune-friendly recipe (epochs, lr,
       freeze-encoder, etc.).
    5. Click "Run sequential transfer ▶".  Spawns _transfer_worker.py
       which loops:
         for each cube:
             register runtime sample
             run_config(...)  # subprocess-internal; uses prev cube's
                              # best.pth as init for the next iteration
                              # (chain mode), or the original ckpt
                              # for every cube (independent mode).
"""
from __future__ import annotations
import os, sys, json, time, subprocess

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

# Re-use the existing TrainPanel knob defaults so Transfer mirrors the
# Training tab's recipe surface but with fine-tune-friendly defaults.
from gui_app.train_panel import PAPER_DEFAULTS, VARIANTS


WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_transfer_worker.py")

MIN_CUBE_BYTES = 100 * 1024 * 1024     # 100 MB heuristic
MAX_CUBES = 20

TRANSFER_DEFAULTS = dict(
    PAPER_DEFAULTS,
    epochs=10,
    lr=1e-4,
    cluster1d_lambda_inter=0.0,        # don't fight ckpt during adapt
    warmup_frac=0.0,                    # no temp ramp on a short run
)


# ---------------------------------------------------------------------------
# Sequential job wrapper (mirrors TrainingJob interface for the close-handler)
# ---------------------------------------------------------------------------

class TransferJob:
    def __init__(self, outdir: str, kwargs: dict):
        self.outdir = outdir
        self.kwargs = kwargs
        self._proc = None
        self._t_start = 0.0
        self._t_end = 0.0
        self._stopped = False

    def start(self):
        if self.is_running(): return
        os.makedirs(self.outdir, exist_ok=True)
        spec = dict(self.kwargs); spec["outdir"] = self.outdir
        spec_path = os.path.join(self.outdir, "_transfer_kwargs.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2, default=str)
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self._t_start = time.perf_counter()
        self._proc = subprocess.Popen(
            [sys.executable, "-u", WORKER, spec_path],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            stdout=open(os.path.join(self.outdir, "_stdout.log"), "w",
                          encoding="utf-8"),
            stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                            if os.name == "nt" else 0),
        )

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, kill_after_s: float = 4.0):
        if not self.is_running(): return
        self._stopped = True
        try:
            self._proc.terminate()
            try: self._proc.wait(timeout=kill_after_s)
            except subprocess.TimeoutExpired:
                self._proc.kill(); self._proc.wait(timeout=2)
        except Exception: pass
        self._t_end = time.perf_counter()

    def progress(self) -> str:
        p = os.path.join(self.outdir, "_progress.txt")
        if os.path.exists(p):
            try: return open(p).read().strip()
            except Exception: return ""
        return ""


# ---------------------------------------------------------------------------
# Folder discovery
# ---------------------------------------------------------------------------

def discover_cubes(folder: str, min_bytes: int = MIN_CUBE_BYTES) -> tuple:
    """Recursively walk `folder` for .prz/.npz files >= min_bytes.

    Returns (cubes, n_skipped) where cubes is a list of dicts:
        {path, name, size_mb}
    """
    cubes = []
    n_skipped = 0
    for root, _, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith((".prz", ".npz", ".npy")):
                continue
            full = os.path.join(root, fn)
            try:
                sz = os.path.getsize(full)
            except OSError:
                continue
            if sz < min_bytes:
                n_skipped += 1
                continue
            cubes.append({
                "path": full,
                "name": os.path.relpath(full, folder),
                "size_mb": sz / (1024 * 1024),
            })
    cubes.sort(key=lambda c: c["name"])
    return cubes, n_skipped


def list_recent_runs(runs_root: str = "runs/_gui",
                       limit: int = 30) -> list:
    """List recent run dirs that contain a best.pth or latest.pth.
    Returns list of (display_label, ckpt_path) tuples, newest first."""
    out = []
    if not os.path.isdir(runs_root):
        return out
    for entry in sorted(os.listdir(runs_root), reverse=True):
        d = os.path.join(runs_root, entry)
        if not os.path.isdir(d):
            continue
        for cand in ("best.pth", "latest.pth"):
            p = os.path.join(d, cand)
            if os.path.exists(p):
                out.append((f"{entry}  ({cand})", p))
                break
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# TransferPanel
# ---------------------------------------------------------------------------

class TransferPanel(ctk.CTkFrame):

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.job = None
        self._poll_after = None
        self._cubes = []                   # list of dicts from discover_cubes
        self._cube_check_vars = {}         # path -> BooleanVar
        self._var = self._build_vars()
        self._build()

    # ----- vars (subset of TrainPanel's) ----------------------------
    def _build_vars(self):
        d = TRANSFER_DEFAULTS
        return dict(
            folder=ctk.StringVar(value=""),
            init_ckpt=ctk.StringVar(value=""),
            chain_ckpts=ctk.BooleanVar(value=True),
            run_name=ctk.StringVar(value=""),
            epochs=ctk.IntVar(value=int(d["epochs"])),
            batch_size=ctk.IntVar(value=int(d["batch_size"])),
            lr=ctk.DoubleVar(value=float(d["lr"])),
            K=ctk.IntVar(value=int(d["K"])),
            variant=ctk.StringVar(value=d["variant"]),
            cluster1d_lambda_intra=ctk.DoubleVar(
                value=float(d["cluster1d_lambda_intra"])),
            cluster1d_lambda_inter=ctk.DoubleVar(
                value=float(d["cluster1d_lambda_inter"])),
            freeze_encoder=ctk.BooleanVar(value=True),
            seed=ctk.IntVar(value=int(d["seed"])),
        )

    # ----- UI -------------------------------------------------------
    def _build(self):
        # Top: folder picker + run name + ckpt picker
        top = ctk.CTkFrame(self)
        top.pack(side="top", fill="x", padx=6, pady=6)
        ctk.CTkLabel(top, text="Cubes folder:").pack(side="left", padx=(8, 4))
        ctk.CTkEntry(top, textvariable=self._var["folder"], width=460
                       ).pack(side="left", padx=2)
        ctk.CTkButton(top, text="Browse…", width=80,
                       command=self._pick_folder).pack(side="left", padx=2)
        ctk.CTkButton(top, text="Refresh", width=80,
                       command=self._rescan).pack(side="left", padx=2)
        ctk.CTkLabel(top, text="    Run name:").pack(side="left", padx=(16, 2))
        ctk.CTkEntry(top, textvariable=self._var["run_name"], width=160
                       ).pack(side="left", padx=2)

        # Body: left = cube list, right = config
        body = ctk.CTkFrame(self)
        body.pack(side="top", fill="both", expand=True, padx=6, pady=4)

        left = ctk.CTkFrame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))
        ctk.CTkLabel(left,
            text="Discovered cubes  (≥100 MB; tick up to 20)",
            font=("Segoe UI", 11, "bold")
            ).pack(anchor="w", padx=8, pady=(6, 2))
        self._cube_list = ctk.CTkScrollableFrame(left)
        self._cube_list.pack(fill="both", expand=True, padx=4, pady=4)
        self._scan_status = ctk.CTkLabel(left, text="(no folder picked)",
            font=("Consolas", 9), text_color=("#444", "#aaa"))
        self._scan_status.pack(anchor="w", padx=8, pady=(0, 4))

        right = ctk.CTkScrollableFrame(body, width=420)
        right.pack(side="left", fill="y")

        self._section(right, "Init checkpoint")
        ck_row = ctk.CTkFrame(right, fg_color="transparent")
        ck_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(ck_row, text="ckpt:", width=60, anchor="w"
                       ).pack(side="left")
        ctk.CTkEntry(ck_row, textvariable=self._var["init_ckpt"], width=240
                       ).pack(side="left", padx=2)
        ctk.CTkButton(ck_row, text="Browse…", width=70,
                        command=self._pick_ckpt).pack(side="left", padx=2)
        recent_row = ctk.CTkFrame(right, fg_color="transparent")
        recent_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(recent_row, text="recent:", width=60, anchor="w"
                       ).pack(side="left")
        recent = list_recent_runs()
        recent_labels = ["(none)"] + [r[0] for r in recent]
        self._recent_map = {r[0]: r[1] for r in recent}
        self._recent_var = ctk.StringVar(value="(none)")
        ctk.CTkOptionMenu(recent_row, variable=self._recent_var,
            values=recent_labels, width=300,
            command=self._on_recent_pick).pack(side="left", padx=2)
        ctk.CTkCheckBox(right, text="Chain ckpts (each cube initialises "
                                       "from the previous cube's best.pth)",
                          variable=self._var["chain_ckpts"]
                          ).pack(anchor="w", padx=8, pady=4)

        self._section(right, "Recipe (per-cube)")
        self._entry(right, "epochs", "epochs",
                       "Per-cube epochs.  10 is a sensible adapt default.")
        self._entry(right, "batch_size", "batch_size", "")
        self._entry(right, "lr", "lr",
                       "Lower (1e-4) for adapt; higher (3e-4) for full pre-train.")
        self._entry(right, "K (prototypes)", "K", "")
        var_row = ctk.CTkFrame(right, fg_color="transparent")
        var_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(var_row, text="variant", width=120, anchor="w"
                       ).pack(side="left")
        ctk.CTkOptionMenu(var_row, variable=self._var["variant"],
                            values=list(VARIANTS.keys()), width=160
                            ).pack(side="left", padx=2)
        self._entry(right, "cluster1d λ intra", "cluster1d_lambda_intra", "")
        self._entry(right, "cluster1d λ inter", "cluster1d_lambda_inter",
                       "Set 0 to avoid fighting the ckpt while adapting.")
        ctk.CTkCheckBox(right, text="freeze encoder (head-only adapt)",
                          variable=self._var["freeze_encoder"]
                          ).pack(anchor="w", padx=8, pady=4)
        self._entry(right, "seed", "seed", "")

        # Buttons
        btn_row = ctk.CTkFrame(self)
        btn_row.pack(side="bottom", fill="x", padx=6, pady=(0, 6))
        self._run_btn = ctk.CTkButton(btn_row,
            text="Run sequential transfer  ▶",
            fg_color=("#2D7A2D", "#1F7A1F"),
            font=("Segoe UI", 12, "bold"), width=240, height=34,
            command=self._on_run)
        self._run_btn.pack(side="left", padx=8, pady=4)
        self._stop_btn = ctk.CTkButton(btn_row, text="Stop ■", width=110,
            height=34, fg_color=("#A04030", "#882F22"),
            command=self._on_stop, state="disabled")
        self._stop_btn.pack(side="left", padx=4)
        self._status = ctk.CTkLabel(btn_row, text="(no transfer running)",
            font=("Consolas", 10), anchor="w")
        self._status.pack(side="left", padx=10)

    # ----- helpers --------------------------------------------------
    def _section(self, parent, title):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=2, pady=(8, 0))
        ctk.CTkLabel(f, text=title, font=("Segoe UI", 11, "bold")
                       ).pack(anchor="w")
        sep = ctk.CTkFrame(parent, height=2,
                              fg_color=("#cccccc", "#444444"))
        sep.pack(fill="x", padx=2, pady=(0, 4))

    def _entry(self, parent, label, key, _help):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(row, text=label, width=180, anchor="w").pack(side="left")
        ctk.CTkEntry(row, textvariable=self._var[key], width=110
                       ).pack(side="left")

    # ----- folder + cube discovery ----------------------------------
    def _pick_folder(self):
        p = filedialog.askdirectory(title="Pick a folder of .prz cubes")
        if p:
            self._var["folder"].set(p)
            self._rescan()

    def _rescan(self):
        folder = self._var["folder"].get().strip()
        if not folder or not os.path.isdir(folder):
            self._scan_status.configure(
                text=f"folder not found: {folder}")
            return
        self._scan_status.configure(text=f"scanning {folder} …")
        self.update_idletasks()
        cubes, skipped = discover_cubes(folder)
        self._cubes = cubes
        # Rebuild the checkbox list
        for w in self._cube_list.winfo_children():
            w.destroy()
        self._cube_check_vars = {}
        for cube in cubes:
            v = ctk.BooleanVar(value=False)
            self._cube_check_vars[cube["path"]] = v
            txt = (f"{cube['name']}    "
                   f"({cube['size_mb']:.0f} MB)")
            ctk.CTkCheckBox(self._cube_list, text=txt, variable=v,
                              command=self._enforce_max).pack(
                anchor="w", padx=4, pady=1)
        self._scan_status.configure(
            text=f"{len(cubes)} datacubes ≥100 MB    "
                  f"(skipped {skipped} smaller .prz/.npz)")

    def _enforce_max(self):
        n_on = sum(1 for v in self._cube_check_vars.values() if v.get())
        if n_on > MAX_CUBES:
            messagebox.showinfo("Transfer",
                f"Max {MAX_CUBES} cubes per transfer run "
                f"(memory + sequential-time budget).  "
                f"Untick some.")

    def _pick_ckpt(self):
        p = filedialog.askopenfilename(
            title="Pick a .pth checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth"),
                       ("All files", "*.*")])
        if p:
            self._var["init_ckpt"].set(p)

    def _on_recent_pick(self, label):
        if label in self._recent_map:
            self._var["init_ckpt"].set(self._recent_map[label])

    # ----- run / stop -----------------------------------------------
    def _checked_cubes(self) -> list:
        return [c for c in self._cubes
                if self._cube_check_vars.get(c["path"], ctk.BooleanVar()).get()]

    def _gather_shared_kwargs(self) -> dict:
        v = self._var
        # Mirror the relevant TrainPanel knobs.  Transfer doesn't expose
        # the full sub-tab tree to keep the UI tight; missing knobs use
        # the paper defaults via TRANSFER_DEFAULTS.
        kw = dict(TRANSFER_DEFAULTS)
        kw.update(dict(
            epochs=int(v["epochs"].get()),
            batch_size=int(v["batch_size"].get()),
            lr=float(v["lr"].get()),
            num_prototypes=int(v["K"].get()),
            seed=int(v["seed"].get()),
        ))
        # Translate variant -> active loss flags (mirrors TrainPanel
        # _gather_kwargs logic, simplified).
        var_dict = VARIANTS.get(v["variant"].get(), {})
        kw["cluster1d_lambda"] = (float(v["cluster1d_lambda_intra"].get())
                                    if var_dict.get("cluster1d") else 0.0)
        kw["cluster1d_lambda_intra"] = (float(v["cluster1d_lambda_intra"].get())
                                          if var_dict.get("cluster1d") else 0.0)
        kw["cluster1d_lambda_inter"] = (float(v["cluster1d_lambda_inter"].get())
                                          if var_dict.get("cluster1d") else 0.0)
        kw["freeze_encoder"] = bool(v["freeze_encoder"].get())
        # Strip GUI-only keys that run_config doesn't accept.
        for drop in ("variant", "cj_brightness", "cj_contrast",
                       "blur_kernel_max", "blur_sigma_max",
                       "polar_mask_cols", "polar_size",
                       "center_crop_size", "center_mask_radius",
                       "com_centering", "com_search_radius_factor",
                       "theta_shift_student", "theta_shift_teacher",
                       "aug_hflip", "aug_vflip", "aug_colorjitter",
                       "aug_blur", "T0", "Tfin", "warmup_frac",
                       "lambda_pair", "pair_entropy_reg", "pair_per_batch",
                       "spatial_tau_pos", "spatial_tau_neg",
                       "lam_spatial", "supcon_lambda", "supcon_temperature",
                       "centroid_lambda", "centroid_margin",
                       "conf_weight_gamma", "K"):
            kw.pop(drop, None)
        return kw

    def _on_run(self):
        if self.job is not None and self.job.is_running():
            messagebox.showinfo("Transfer",
                "A transfer run is already in progress."); return
        cubes = self._checked_cubes()
        if not cubes:
            messagebox.showinfo("Transfer",
                "No cubes selected."); return
        if len(cubes) > MAX_CUBES:
            messagebox.showinfo("Transfer",
                f"Too many cubes ({len(cubes)} > {MAX_CUBES})."); return
        init_ckpt = self._var["init_ckpt"].get().strip() or None
        if init_ckpt and not os.path.exists(init_ckpt):
            messagebox.showerror("Transfer",
                f"init checkpoint not found:\n{init_ckpt}"); return
        # Output dir
        run_name = self._var["run_name"].get().strip() or \
                   time.strftime("transfer_%Y%m%d_%H%M%S")
        outdir = os.path.join("runs", "_gui", run_name)
        # Build worker spec
        spec = dict(
            outdir=outdir,
            cubes=[{"path": c["path"], "name": c["name"]} for c in cubes],
            init_ckpt=init_ckpt,
            chain_ckpts=bool(self._var["chain_ckpts"].get()),
            shared_kwargs=self._gather_shared_kwargs(),
        )
        self.job = TransferJob(outdir, spec)
        self.job.start()
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._status.configure(
            text=f"transfer started  ({len(cubes)} cubes)  →  {outdir}")
        self._poll()

    def _on_stop(self):
        if self.job is None or not self.job.is_running():
            return
        if not messagebox.askyesno("Stop transfer",
            "Stop the running transfer?  Progress for completed cubes "
            "is kept on disk; the in-flight cube is killed."):
            return
        self.job.stop()
        self._status.configure(text="stopping…")

    def _poll(self):
        if self.job is None: return
        if self.job.is_running():
            prog = self.job.progress()
            elapsed = time.perf_counter() - self.job._t_start
            self._status.configure(
                text=f"transfer running…  {prog}    elapsed {elapsed:.0f}s")
            self._poll_after = self.after(2000, self._poll)
        else:
            self._run_btn.configure(state="normal")
            self._stop_btn.configure(state="disabled")
            self._status.configure(
                text=f"transfer finished.  outputs in {self.job.outdir}")
