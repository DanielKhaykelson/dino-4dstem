"""synth_panel.py — Synthetic-data tab.

Builds 4D-STEM phantoms from a list of crystal structures (either CIF
files or ASE builders), simulates with abTEM (PRISM by default), and
writes a cube + three ground-truth class maps next to it:

    <out>.cube.npy            (Ny, Nx, H, W)  — drop-in for LoadPRZ
    <out>.classmap_A.npy      phase only
    <out>.classmap_B.npy      phase + zone-axis
    <out>.classmap_C.npy      phase + full orientation
    <out>.sim_meta.json       all abTEM/grain params + RNG seed

The same tab houses the validation block: once the user has run
inference on this cube with some trained model, the Score button reads
back the chosen classmap, runs Hungarian alignment vs the model's
hard assignments, and prints a metric table inline.

Phase 1 scope: CIF | ASE builder rows, block layout, PRISM, single
zone-axis per structure.  Voronoi / continuous tilt / multislice are
added in later phases.
"""
from __future__ import annotations
import os, sys, json, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data import register_runtime_sample
from gui_app.tooltip import add_help_button


# =========================================================================
# 0. Built-in materials + crystallinity presets
# =========================================================================
# Common crystals buildable with ASE (no CIF file needed).  Each entry is
# an ``ase_builder`` dict {fn, args, kwargs} passed to the same builder
# dispatch used for user ASE rows.  TMDs (mx2) need a `vacuum` so the cell
# has a c-axis; cubic metals/semiconductors need cubic=True for an
# orthogonal conventional cell (abTEM refuses non-orthogonal cells).
PREDEFINED_CRYSTALS: "dict[str, dict]" = {
    # TMDs: mx2 builds one layer; vacuum=1.5 sets the c-axis so tiling in
    # Z stacks layers at the realistic 2H van-der-Waals spacing (~6.15 Å)
    # → a proper few-layer 3D film (Tile Z = number of layers), NOT an
    # isolated monolayer.  (True 2H AB stacking needs a bulk CIF.)
    "WS2 (2H layered)":  {"fn": "mx2", "args": [],
        "kwargs": {"formula": "WS2", "kind": "2H", "a": 3.153,
                    "thickness": 3.14, "vacuum": 1.5}},
    "MoS2 (2H layered)": {"fn": "mx2", "args": [],
        "kwargs": {"formula": "MoS2", "kind": "2H", "a": 3.16,
                    "thickness": 3.17, "vacuum": 1.54}},
    "Si (diamond)":   {"fn": "bulk", "args": ["Si"],
        "kwargs": {"crystalstructure": "diamond", "a": 5.43, "cubic": True}},
    "Cu (fcc)":       {"fn": "bulk", "args": ["Cu"],
        "kwargs": {"crystalstructure": "fcc", "a": 3.61, "cubic": True}},
    "Au (fcc)":       {"fn": "bulk", "args": ["Au"],
        "kwargs": {"crystalstructure": "fcc", "a": 4.08, "cubic": True}},
    "Al (fcc)":       {"fn": "bulk", "args": ["Al"],
        "kwargs": {"crystalstructure": "fcc", "a": 4.05, "cubic": True}},
    "Fe (bcc)":       {"fn": "bulk", "args": ["Fe"],
        "kwargs": {"crystalstructure": "bcc", "a": 2.87, "cubic": True}},
    "NaCl (rocksalt)": {"fn": "bulk", "args": ["NaCl"],
        "kwargs": {"crystalstructure": "rocksalt", "a": 5.64, "cubic": True}},
}

# Crystallinity levels → (static/thermal displacement sigma in Å,
# n frozen-phonon configs to average).  Averaging over configs turns
# the sharp Bragg speckle into smooth diffuse scattering; larger sigma
# damps the Bragg peaks (Debye-Waller) and raises the diffuse background.
#   crystalline : sharp spots (no disorder averaging)
#   partial     : weakened spots + mild diffuse (thermal-diffuse look)
#   amorphous   : spots washed out, diffuse rings dominate
CRYSTALLINITY_LEVELS: "dict[str, tuple[float, int]]" = {
    "crystalline": (0.0,  1),
    "partial":     (0.12, 4),
    "amorphous":   (0.60, 8),
}
CRYSTALLINITY_ORDER = ("crystalline", "partial", "amorphous")


# =========================================================================
# 1. Structure description (one per grain "phase")
# =========================================================================
class _Structure:
    """One row in the structure list.

    Construction sources:
      ``cif_path``  — load from .cif
      ``ase_builder`` — dict, e.g. {"fn":"bulk","args":["Si"],"kwargs":{...}}

    Per-structure fields:
      ``zone_axes``        — list of (h,k,l).  A grain picks one at
                              random.
      ``tile_xyz``         — (nx, ny, nz) supercell repetitions
                              AFTER zone-axis orientation.  nz controls
                              physical thickness in unit cells along
                              the beam direction.
      ``thickness_crop_A`` — optional post-tile z crop in Å.  0 →
                              no crop (use full nz·c_unit thickness).
      ``tilt_range_deg``   — small off-zone-axis tilt range, KEEP ≤ 5°.
      ``in_plane_range_deg`` — (lo, hi) in-plane rotation range.
    """
    __slots__ = ("kind", "cif_path", "ase_builder", "label",
                  "zone_axes", "tile_xyz", "thickness_crop_A",
                  "tilt_mrad_range", "in_plane_range_deg",
                  "crystallinity",
                  # legacy slot kept so unpickled / old structures
                  # don't AttributeError; never read.
                  "tilt_range_deg")

    def __init__(self, kind, cif_path=None, ase_builder=None,
                  label="", zone_axes=((0, 0, 1),),
                  tile_xyz=(2, 2, 4),
                  thickness_crop_A=0.0,
                  tilt_mrad_range=0.0,
                  in_plane_range_deg=(0.0, 360.0),
                  crystallinity="crystalline",
                  tilt_range_deg=0.0):  # legacy kwarg — ignored
        self.kind = kind
        # Crystallinity domain level (crystalline | partial | amorphous).
        cl = str(crystallinity or "crystalline").lower()
        self.crystallinity = cl if cl in CRYSTALLINITY_LEVELS else "crystalline"
        self.cif_path = cif_path
        self.ase_builder = ase_builder
        self.label = label or "?"
        # Normalise zone_axes → list[tuple[int,int,int]]
        if isinstance(zone_axes, (tuple, list)) and len(zone_axes) > 0 \
                and isinstance(zone_axes[0], (int, float)):
            zone_axes = [tuple(int(x) for x in zone_axes)]
        else:
            zone_axes = [tuple(int(x) for x in za) for za in zone_axes]
        self.zone_axes = zone_axes or [(0, 0, 1)]
        if isinstance(tile_xyz, int):
            tile_xyz = (tile_xyz, tile_xyz, tile_xyz)
        self.tile_xyz = tuple(max(1, int(x)) for x in tile_xyz)
        self.thickness_crop_A = float(thickness_crop_A or 0.0)
        # Off-axis tilt — implemented as BEAM TILT (passed to abTEM's
        # Probe/SMatrix tilt kwarg, in mrad).  No cell rotation, no
        # orthogonalize call, no OOM.  Each grain draws a tilt
        # magnitude in [0, tilt_mrad_range] and a random azimuth.
        # Cap at ~50 mrad for physical accuracy (small-angle small-
        # tilt approximation).
        self.tilt_mrad_range = float(tilt_mrad_range or 0.0)
        self.tilt_range_deg = 0.0  # legacy field; not used
        self.in_plane_range_deg = (float(in_plane_range_deg[0]),
                                       float(in_plane_range_deg[1]))

    def describe(self) -> str:
        if self.kind == "cif":
            src = f"CIF: {os.path.basename(self.cif_path or '?')}"
        else:
            b = self.ase_builder or {}
            args = ",".join(map(str, b.get("args", [])))
            src = f"ASE: {b.get('fn', '?')}({args})"
        zas = ",".join("[" + ",".join(str(x) for x in za) + "]"
                          for za in self.zone_axes)
        nx, ny, nz = self.tile_xyz
        thk = (f"trim_to={self.thickness_crop_A:g}Å"
                 if self.thickness_crop_A > 0
                 else f"thickness = {nz}·c_unit (full)")
        tilt = (f" beam_tilt≤{self.tilt_mrad_range:g}mrad"
                  if self.tilt_mrad_range > 0 else "")
        cry = f"  [{self.crystallinity}]"
        return f"{src}  ZA={zas}  tile=({nx},{ny},{nz})  {thk}{tilt}{cry}"


class _SimAborted(Exception):
    """Raised inside the simulation worker when the user hits Stop.
    Distinct exception type so the worker can catch it specifically
    and report a clean "aborted" status (vs a real crash)."""


# =========================================================================
# 2. Lightweight ASE / abTEM gating — only imported when the user clicks
#    "Simulate".  Lets the GUI start without these heavy deps installed.
# =========================================================================
def _check_deps():
    """Return list of (pkg_name, error_repr) for any dep that can't
    import.  Empty list = all OK.  We deliberately catch *all*
    exceptions because heavy deps like abtem can fail at module-load
    time with non-ImportError errors (cupy GPU detection, dask init,
    parametrization file lookups, ...) and the user needs to see the
    underlying message rather than a generic 'not installed' lie.
    """
    failures: list[tuple[str, str]] = []
    for pkg in ("ase", "abtem"):
        try:
            mod = __import__(pkg)
            # Reach into the file path so a stale shim re-exporting the
            # name without the real module surfaces here too.
            _ = getattr(mod, "__file__", None)
        except BaseException as e:   # noqa: BLE001 — yes, we want it ALL
            failures.append((pkg, f"{type(e).__name__}: {e!r}"))
    return failures


def _env_diag() -> str:
    """One-line summary of the active interpreter — used to flag env
    mismatches when the user 'has' abtem in a different conda env than
    the one the GUI is running under."""
    import sys, os
    return (f"python = {sys.executable}\n"
            f"prefix = {sys.prefix}\n"
            f"conda env = {os.environ.get('CONDA_DEFAULT_ENV') or '?'}")


# =========================================================================
# 3. The panel
# =========================================================================
class SynthPanel(ctk.CTkFrame):
    """Build → simulate → validate, all in one tab."""

    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app

        # State
        self._structures: list[_Structure] = []
        self._sim_busy = False
        # Cooperative abort flag — checked between grains in the
        # simulation loop.  We can't kill abTEM mid-multislice safely
        # (would leak GPU memory and corrupt FFT plans), but per-grain
        # granularity is fine: each grain bounded to seconds–minutes.
        self._sim_stop_requested = False
        self._last_cube_path: "str | None" = None
        self._last_classmaps: "dict | None" = None  # {"A":path, "B":path, "C":path}
        self._last_meta: "dict | None" = None

        # Scan / abTEM params (defaults match Na007b 58 mm CL)
        self.scan_ny = ctk.IntVar(value=32)
        self.scan_nx = ctk.IntVar(value=32)
        self.scan_step_A = ctk.DoubleVar(value=2.0)   # Å between probe positions
        self.beam_kV = ctk.DoubleVar(value=200.0)
        self.conv_mrad = ctk.DoubleVar(value=1.0)     # convergence semi-angle
        self.det_size = ctk.IntVar(value=256)         # detector pixel grid
        self.det_max_mrad = ctk.DoubleVar(value=80.0)
        self.dose_e_A2 = ctk.DoubleVar(value=1e3)     # 0 → no scaling/noise
        # Noise mode: "shot" = Poisson-sample at the dose (realistic shot
        # noise, SNR ~ sqrt(counts)); "exact" = deterministic pattern
        # scaled to the same electron count but with NO Poisson sampling.
        self.noise_mode = ctk.StringVar(value="shot")
        self.engine = ctk.StringVar(value="finite")   # finite | multislice | PRISM
        self.layout = ctk.StringVar(value="block")    # block | voronoi
        self.n_voronoi_seeds = ctk.IntVar(value=4)    # Phase 2: Voronoi grains (low default → safer)
        # Potential gpts — was 512 (fine, but each SMatrix ≈ 100s of MB).
        # 256 cuts memory ~4× with negligible visual difference for
        # validation cubes.  Exposed in the UI for paper-grade runs.
        self.pot_gpts = ctk.IntVar(value=512)   # 512 suits the finite engine
        # Vacuum padding around the rotated atom cluster (Å).  Used by
        # the NaPHI-style ``_build_atoms_vacuum_box``: any tilt/ZA is
        # handled by rotating the atoms inside an orthogonal box that
        # ASE rebuilds around the rotated cluster + ``vacuum_pad_A``
        # padding on each side.  No orthogonalize_cell call needed.
        # 10–20 Å is plenty for most cases.
        self.vacuum_pad_A = ctk.DoubleVar(value=15.0)
        self.rng_seed = ctk.IntVar(value=0)
        self.out_basename = ctk.StringVar(value="phantom")

        # Validation state
        self._val_classmap_choice = ctk.StringVar(
            value="main — phase + orientation + crystallinity")
        self._val_status = None

        self._build()
        # Start with Voronoi-seeds entry greyed out (default layout=block).
        try: self._refresh_layout_visibility()
        except Exception: pass

    # ===================================================================
    # UI
    # ===================================================================
    def _build(self):
        # Two sub-tabs: build the phantom (Simulate) vs score a model
        # against it (Validate).  Validate gets its own full-height tab
        # so the confusion-matrix figure isn't squeezed.
        tabs = ctk.CTkTabview(self)
        tabs.pack(fill="both", expand=True, padx=6, pady=6)
        sim_tab = tabs.add("Simulate")
        val_tab = tabs.add("Validate")

        # --- top: structure list -------------------------------------
        top = ctk.CTkFrame(sim_tab)
        top.pack(side="top", fill="x", pady=(0, 6))
        head = ctk.CTkFrame(top, fg_color="transparent")
        head.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(head, text="Structures",
                       font=("Segoe UI", 12, "bold")).pack(side="left")
        add_help_button(head,
            "Each structure → one phase in the phantom.  Add as many "
            "as you like; the layout dropdown decides how grains are "
            "tiled.  Either pick a .cif or build via ASE "
            "(bulk, surface, mx2, etc.) — both produce the same "
            "downstream atomic model.").pack(side="left", padx=4)
        btn_row = ctk.CTkFrame(top, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=2)
        ctk.CTkButton(btn_row, text="+ from CIF…", width=130,
                       fg_color=("#2D7A2D", "#1F7A1F"),
                       command=self._add_from_cif).pack(side="left",
                                                            padx=2)
        ctk.CTkButton(btn_row, text="+ built-in crystal…", width=155,
                       fg_color=("#2D7A2D", "#1F7A1F"),
                       command=self._add_from_builtin).pack(side="left",
                                                               padx=2)
        ctk.CTkButton(btn_row, text="+ from ASE builder…", width=160,
                       fg_color=("#2D7A2D", "#1F7A1F"),
                       command=self._add_from_ase).pack(side="left",
                                                            padx=2)
        ctk.CTkButton(btn_row, text="Edit selected…", width=120,
                       command=self._edit_selected).pack(side="left",
                                                             padx=2)
        ctk.CTkButton(btn_row, text="Remove selected", width=140,
                       command=self._remove_selected).pack(side="left",
                                                                padx=2)
        ctk.CTkButton(btn_row, text="🔬 View 3D", width=95,
                       command=self._view_structure).pack(side="left",
                                                              padx=2)
        ctk.CTkButton(btn_row, text="⚡ WS₂ example", width=140,
                       fg_color=("#3A5380", "#2A3F63"),
                       command=self._load_ws2_example).pack(side="right",
                                                               padx=2)

        list_holder = ctk.CTkFrame(top)
        list_holder.pack(fill="x", padx=8, pady=(2, 6))
        self._listbox = tk.Listbox(list_holder, height=6,
                                       font=("Consolas", 10),
                                       activestyle="dotbox")
        self._listbox.pack(side="left", fill="x", expand=True)
        sb = tk.Scrollbar(list_holder, command=self._listbox.yview)
        sb.pack(side="right", fill="y")
        self._listbox.config(yscrollcommand=sb.set)

        # --- middle: scan / abTEM params ----------------------------
        mid = ctk.CTkFrame(sim_tab)
        mid.pack(side="top", fill="x", pady=(0, 6))
        ctk.CTkLabel(mid, text="Scan & acquisition",
                       font=("Segoe UI", 12, "bold")).pack(
            anchor="w", padx=8, pady=(6, 2))

        grid = ctk.CTkFrame(mid, fg_color="transparent")
        grid.pack(fill="x", padx=8, pady=(0, 6))
        self._row_pair(grid, 0, "Scan Ny", self.scan_ny,
                         "Scan Nx", self.scan_nx)
        self._row_pair(grid, 1, "Step (Å)", self.scan_step_A,
                         "RNG seed", self.rng_seed)
        self._row_pair(grid, 2, "Beam (kV)", self.beam_kV,
                         "Conv α (mrad)", self.conv_mrad)
        self._row_pair(grid, 3, "Detector px", self.det_size,
                         "Det α_max (mrad)", self.det_max_mrad)
        self._row_pair(grid, 4, "Dose (e/Å²)", self.dose_e_A2,
                         "Out basename", self.out_basename)

        noise_row = ctk.CTkFrame(mid, fg_color="transparent")
        noise_row.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(noise_row, text="Electrons:").pack(side="left")
        ctk.CTkSegmentedButton(
            noise_row, values=["shot", "exact"],
            variable=self.noise_mode, width=180).pack(side="left", padx=6)
        add_help_button(noise_row,
            "How the dose is applied to each diffraction pattern.  Each "
            "pattern is scaled to  Dose × step²  electrons (electrons per "
            "probe position), then:\n\n"
            "  shot  — Poisson-SAMPLE at that count → realistic shot "
            "noise.  Higher dose ⇒ cleaner (SNR ~ √counts).  Low dose or "
            "weak Bragg peaks ⇒ visibly grainy.\n\n"
            "  exact — use the EXACT expected counts with NO Poisson "
            "sampling → a clean, noise-free pattern at the same "
            "brightness.  Use this when you want the ideal pattern "
            "regardless of dose.\n\n"
            "Set Dose = 0 to skip scaling/noise entirely (raw normalized "
            "intensity).").pack(side="left", padx=4)

        sel_row = ctk.CTkFrame(mid, fg_color="transparent")
        sel_row.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(sel_row, text="Engine:").pack(side="left")
        ctk.CTkOptionMenu(sel_row, variable=self.engine,
                            values=["PRISM", "multislice", "finite"],
                            width=130).pack(side="left", padx=4)
        add_help_button(sel_row,
            "Simulation engine:\n\n"
            "  multislice — scanned probe per pixel, projection='infinite' "
            "(fast, but finite-crystal features smear into streaks/'diffuse' "
            "between spots).\n\n"
            "  finite — projection='finite' + ONE high-quality diffraction "
            "pattern per grain, broadcast to that grain's pixels.  Removes "
            "the inter-spot streaks (clean spots), and is often faster since "
            "it's one sim per grain, not one per pixel.  Use gpts≥512 for "
            "sharpest spots.\n\n"
            "  PRISM — scattering-matrix scan (experimental here).").pack(
            side="left", padx=4)
        ctk.CTkLabel(sel_row, text="Layout:").pack(side="left",
                                                        padx=(12, 4))
        ctk.CTkOptionMenu(sel_row, variable=self.layout,
                            values=["block", "voronoi"],
                            width=120,
                            command=lambda _v: self._refresh_layout_visibility()
                            ).pack(side="left", padx=4)
        ctk.CTkLabel(sel_row, text="Voronoi seeds:").pack(side="left",
                                                                padx=(12, 4))
        self._vor_seeds_entry = ctk.CTkEntry(sel_row,
            textvariable=self.n_voronoi_seeds, width=60)
        self._vor_seeds_entry.pack(side="left", padx=2)
        ctk.CTkLabel(sel_row, text="potential gpts:").pack(side="left",
                                                                padx=(12, 4))
        ctk.CTkEntry(sel_row, textvariable=self.pot_gpts,
                       width=60).pack(side="left", padx=2)
        ctk.CTkLabel(sel_row, text="vacuum pad (Å):").pack(side="left",
                                                                padx=(12, 4))
        ctk.CTkEntry(sel_row, textvariable=self.vacuum_pad_A,
                       width=60).pack(side="left", padx=2)
        add_help_button(sel_row,
            "PRISM: caches the scattering matrix and samples per probe "
            "position — 10–100× faster than full multislice for 4D-STEM "
            "scans, <5% intensity error for thin samples.  Use multislice "
            "for paper-figure validation only.\n\n"
            "Layout 'block': N phases → N tiles in a grid (e.g. 2×2 "
            "for 4 phases).  Layout 'voronoi': N_voronoi_seeds random "
            "grain seeds; each grain gets a random (phase, zone-axis, "
            "in-plane rot) assignment.  More realistic, harder to "
            "score."
            ).pack(side="left", padx=(12, 0))

        # --- Simulate row -------------------------------------------
        sim_row = ctk.CTkFrame(sim_tab)
        sim_row.pack(side="top", fill="x", pady=(0, 6))
        ctk.CTkButton(sim_row, text="Simulate ▶", width=130,
                       height=34,
                       fg_color=("#2D7A2D", "#1F7A1F"),
                       command=self._on_simulate).pack(side="left",
                                                           padx=8, pady=6)
        self._stop_btn = ctk.CTkButton(sim_row, text="Stop ■",
                       width=90, height=34,
                       fg_color=("#8b2020", "#a83232"),
                       hover_color=("#a73030", "#c84545"),
                       state="disabled",
                       command=self._on_stop)
        self._stop_btn.pack(side="left", padx=4, pady=6)
        ctk.CTkButton(sim_row, text="Load existing phantom…",
                        width=180, height=34,
                        command=self._on_load_existing
                        ).pack(side="left", padx=4, pady=6)
        ctk.CTkButton(sim_row, text="Multislice comparison ▶",
                        width=200, height=34,
                        command=self._on_multislice_compare
                        ).pack(side="left", padx=4, pady=6)
        self._sim_status = ctk.CTkLabel(sim_row, text="(no sim yet)",
            font=("Consolas", 10),
            text_color=("#444", "#aaa"), wraplength=600,
            justify="left")
        self._sim_status.pack(side="left", fill="x", expand=True,
                                  padx=8)

        # --- Validate tab: score a trained model against the phantom --
        val = ctk.CTkFrame(val_tab, border_width=1)
        val.pack(side="top", fill="both", expand=True, pady=(0, 4))
        vh = ctk.CTkFrame(val, fg_color="transparent")
        vh.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(vh, text="Validate (score against trained model)",
                       font=("Segoe UI", 12, "bold")).pack(side="left")
        add_help_button(vh,
            "After a sim run is loaded AND a trained model has been "
            "evaluated on it (via Post-hoc → Run Inference, or Eval "
            "tab), pick a labeling convention and hit Score.\n\n"
            "  A — phase only: same CIF → one class regardless of "
            "orientation. Tests rotation invariance.\n"
            "  B — phase + zone axis: distinguishes [001] vs [111] of "
            "the same crystal as different classes. Probably the most "
            "physically honest.\n"
            "  C — phase + full orientation: every distinct (CIF, "
            "rotation) is its own class. Hardest to recover.").pack(
            side="left", padx=4)

        vrow = ctk.CTkFrame(val, fg_color="transparent")
        vrow.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(vrow, text="Labeling:").pack(side="left")
        ctk.CTkOptionMenu(vrow,
            variable=self._val_classmap_choice,
            values=["main — phase + orientation + crystallinity",
                       "A — phase only",
                       "B — phase + orientation",
                       "crys — phase + crystallinity",
                       "C — full incl in-plane (control)"],
            width=360).pack(side="left", padx=4)
        ctk.CTkButton(vrow, text="Score ▶", width=120,
                        fg_color=("#2D7A2D", "#1F7A1F"),
                        command=self._on_validate).pack(side="left",
                                                             padx=8)
        ctk.CTkButton(vrow, text="Show preview",
                        width=130,
                        command=self._show_preview_btn
                        ).pack(side="left", padx=4)
        ctk.CTkButton(vrow, text="Open last sim folder",
                        width=180,
                        command=self._open_last_outdir
                        ).pack(side="left", padx=4)

        # Metric panel: text + confusion-matrix figure side by side.
        body2 = ctk.CTkFrame(val)
        body2.pack(fill="both", expand=True, padx=8, pady=(4, 6))
        self._metrics_text = tk.Text(body2, height=12, width=44,
                                          font=("Consolas", 10),
                                          wrap="word")
        self._metrics_text.pack(side="left", fill="y", padx=(0, 4))
        self._metrics_text.insert("end",
            "Metrics will appear here after Score.")
        self._metrics_text.config(state="disabled")

        fig_holder = ctk.CTkFrame(body2)
        fig_holder.pack(side="left", fill="both", expand=True)
        self._fig = Figure(figsize=(5, 4), dpi=95, facecolor="#f4f4f4")
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xticks([]); self._ax.set_yticks([])
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_holder)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    # ----- small helpers -------------------------------------------
    def _row_pair(self, parent, row, lab1, var1, lab2, var2):
        ctk.CTkLabel(parent, text=lab1).grid(row=row, column=0,
                                                  sticky="e",
                                                  padx=(4, 4), pady=2)
        ctk.CTkEntry(parent, textvariable=var1, width=110).grid(
            row=row, column=1, sticky="w", padx=2, pady=2)
        ctk.CTkLabel(parent, text=lab2).grid(row=row, column=2,
                                                  sticky="e",
                                                  padx=(16, 4), pady=2)
        ctk.CTkEntry(parent, textvariable=var2, width=110).grid(
            row=row, column=3, sticky="w", padx=2, pady=2)

    def _refresh_list(self):
        self._listbox.delete(0, tk.END)
        for i, s in enumerate(self._structures):
            self._listbox.insert(tk.END,
                f"  [{i}] {s.label!r:20s}  {s.describe()}")

    # ===================================================================
    # Structure-list actions
    # ===================================================================
    def _add_from_cif(self):
        p = filedialog.askopenfilename(
            title="Pick CIF",
            filetypes=[("CIF", "*.cif"), ("All", "*.*")])
        if not p: return
        label = os.path.splitext(os.path.basename(p))[0]
        s = _Structure(kind="cif", cif_path=p, label=label)
        s = self._edit_dialog(s, allow_kind_change=False)
        if s is not None:
            self._structures.append(s); self._refresh_list()

    def _add_from_ase(self):
        # Sensible default: bulk silicon along [001].  IMPORTANT:
        # cubic=True forces the CONVENTIONAL cubic cell — without it
        # `bulk(...)` returns the primitive (rhombohedral) cell, which
        # is non-orthogonal and abTEM will refuse to run on it.  Same
        # gotcha for any cubic structure built via ASE.
        s = _Structure(kind="ase",
                          ase_builder={"fn": "bulk",
                                          "args": ["Si"],
                                          "kwargs": {"crystalstructure":
                                                       "diamond",
                                                       "a": 5.43,
                                                       "cubic": True}},
                          label="Si")
        s = self._edit_dialog(s, allow_kind_change=False)
        if s is not None:
            self._structures.append(s); self._refresh_list()

    def _add_from_builtin(self):
        """Pick one of the built-in ASE crystals (no CIF needed)."""
        dlg = ctk.CTkToplevel(self)
        dlg.title("Built-in crystal")
        dlg.geometry("360x150")
        try: dlg.transient(self.winfo_toplevel())
        except Exception: pass
        dlg.grab_set()
        ctk.CTkLabel(dlg, text="Choose a built-in crystal:").pack(
            anchor="w", padx=10, pady=(12, 4))
        names = list(PREDEFINED_CRYSTALS.keys())
        choice = ctk.StringVar(value=names[0])
        ctk.CTkOptionMenu(dlg, variable=choice, values=names,
                            width=320).pack(padx=10, pady=2)
        picked = {"name": None}
        def _ok():
            picked["name"] = choice.get(); dlg.destroy()
        bb = ctk.CTkFrame(dlg, fg_color="transparent")
        bb.pack(side="bottom", fill="x", padx=10, pady=10)
        ctk.CTkButton(bb, text="OK", width=80,
                        fg_color=("#2D7A2D", "#1F7A1F"),
                        command=_ok).pack(side="right", padx=2)
        ctk.CTkButton(bb, text="Cancel", width=80,
                        command=dlg.destroy).pack(side="right", padx=2)
        self.wait_window(dlg)
        name = picked["name"]
        if not name:
            return
        import copy
        s = _Structure(kind="ase",
                          ase_builder=copy.deepcopy(PREDEFINED_CRYSTALS[name]),
                          label=name.split(" ")[0])
        s = self._edit_dialog(s, allow_kind_change=False)
        if s is not None:
            self._structures.append(s); self._refresh_list()

    def _build_base_atoms(self, s: "_Structure"):
        """Build the raw crystal (before ZA orientation / vacuum box) for
        the given structure row — shared by the 3D viewer."""
        if s.kind == "cif":
            from ase.io import read as ase_read
            if not s.cif_path or not os.path.exists(s.cif_path):
                raise RuntimeError(f"CIF not found: {s.cif_path!r}")
            return ase_read(s.cif_path)
        from ase.build import bulk, surface, mx2, fcc100, fcc111  # noqa
        b = s.ase_builder or {}
        fn = {"bulk": bulk, "surface": surface, "mx2": mx2,
              "fcc100": fcc100, "fcc111": fcc111}.get(b.get("fn", "bulk"))
        if fn is None:
            raise RuntimeError(f"unknown ASE builder: {b.get('fn')!r}")
        return fn(*b.get("args", []), **b.get("kwargs", {}))

    def _view_structure(self):
        """Open the selected structure in ASE's interactive 3D viewer
        (a separate window with live mouse rotation).  Shows the real
        crystal repeated by Tile X,Y,Z so you can see the 3D packing /
        thickness — press a number key (1/2/3) in the viewer to look
        down x/y/z."""
        sel = list(self._listbox.curselection())
        if not sel:
            messagebox.showinfo("View 3D",
                "Select a structure row first."); return
        s = self._structures[int(sel[0])]
        try:
            atoms = self._build_base_atoms(s)
            nx, ny, nz = (max(1, min(int(t), 6)) for t in s.tile_xyz)
            atoms = atoms * (nx, ny, nz)
        except Exception as e:
            messagebox.showerror("View 3D",
                f"Could not build structure:\n{e}"); return
        import tempfile, subprocess, sys
        from ase.io import write
        try:
            d = tempfile.mkdtemp(prefix="dinosr_view_")
            p = os.path.join(d, f"{s.label or 'structure'}.xyz")
            write(p, atoms)
            # Launch ASE's GUI in a SEPARATE process → live rotation
            # without clashing with this app's Tk mainloop.
            subprocess.Popen([sys.executable, "-m", "ase", "gui", p])
        except Exception as e:
            messagebox.showerror("View 3D",
                f"Could not launch the ASE 3D viewer:\n{e!r}\n\n"
                f"(Structure has {len(atoms)} atoms, formula "
                f"{atoms.get_chemical_formula()}.)")

    def _load_ws2_example(self):
        """One-click demo: WS2 domains that vary by 3D orientation AND
        crystallinity, with in-plane rotation as a within-class nuisance.
        Doubles as the runnable example and the paper validation phantom."""
        if self._structures and not messagebox.askyesno(
                "WS₂ example",
                "Replace the current structure list with the WS₂ example "
                "(3 domains: 3 distinct 3-D orientations)?"):
            return
        import copy
        ws2 = PREDEFINED_CRYSTALS["WS2 (2H layered)"]
        # 3 domains: same material, differing by zone axis.  In-plane spun
        # freely (0–360°) → must collapse to a single class per orientation.
        specs = [
            ("WS2 [001] cryst",  (0, 0, 1), "crystalline"),
            ("WS2 [110] cryst",  (1, 1, 0), "crystalline"),
            ("WS2 [100] cryst",  (1, 0, 0), "crystalline"),
        ]
        # tile_z=6 → a ~6-layer WS2 film.  BIG in-plane tiling (14×14) so
        # the crystal patch nearly fills the box: a small patch in a large
        # vacuum-padded box gives each Bragg spot a wide shape-function
        # (sinc) → smeared spots with side-lobe "added spots".  A large
        # patch → sharp, clean Bragg spots.  (Multislice cost is set by
        # gpts, not atom count, so bigger patches are ~free.)
        # View any row with "🔬 View 3D" to see the stacking.
        self._structures = [
            _Structure(kind="ase", ase_builder=copy.deepcopy(ws2),
                        label=lab, zone_axes=[za], tile_xyz=(14, 14, 6),
                        in_plane_range_deg=(0.0, 360.0),
                        crystallinity=cry)
            for (lab, za, cry) in specs
        ]
        # Larger + more DIVERSE defaults.  The dataset's real diversity =
        # number of grains (each grain = one crystal at one in-plane
        # rotation, so all its pixels are ~identical).  So we use MANY
        # Voronoi seeds (64) — this is cheap because the expensive
        # multislice is cached per (structure, ZA, crystallinity) and the
        # per-grain in-plane rotation is just a 2-D image rotation.  A big
        # 96×96 scan then gives plenty of patterns for self-supervised
        # training.  det_max=40 mrad makes the WS2 diffraction fill the
        # frame (first reflection ~9 mrad → ~29 px, disks ~5 px) instead
        # of being crammed into the central ~14 px at 80 mrad.
        self.scan_ny.set(96); self.scan_nx.set(96)
        self.scan_step_A.set(3.0)
        # conv 1.5 mrad → disks a few px wide (NOT sub-pixel) so in-plane
        # rotation is smooth/alias-free; det_max 28 mrad → the WS2
        # reflections (out to ~27 mrad) fill the frame instead of sitting
        # in the central ~half with a blank outer ring.
        self.beam_kV.set(200.0); self.conv_mrad.set(1.5)
        self.det_size.set(256); self.det_max_mrad.set(28.0)
        self.dose_e_A2.set(1e4)
        self.engine.set("finite")          # clean spots, no inter-spot streaks
        self.layout.set("voronoi"); self.n_voronoi_seeds.set(64)
        self.pot_gpts.set(512); self.rng_seed.set(0)
        # Small vacuum pad → the big crystal patch fills more of the box →
        # sharper Bragg spots (fewer finite-size side-lobe "added spots").
        self.vacuum_pad_A.set(6.0)
        self.out_basename.set("WS2_example")
        try: self._refresh_layout_visibility()
        except Exception: pass
        self._refresh_list()
        messagebox.showinfo(
            "WS₂ example",
            "Loaded 3 WS₂ domains (3 distinct 3-D orientations), "
            "96×96 scan, Voronoi with 64 grains, det_max=28 mrad "
            "(diffraction fills the frame), conv=1.5 mrad (disks a few px, "
            "so in-plane rotation stays alias-free), FINITE engine "
            "(gpts=512, clean spots) @200 kV.\n\n"
            "The 64 grains give the model ~64 distinct in-plane rotations "
            "to learn from — the pixel count matters less than the grain "
            "count.\n\n"
            "The finite engine runs one high-quality pattern per grain "
            "(~64 sims), so it stays in the minutes range at gpts=512.")

    def _edit_selected(self):
        sel = list(self._listbox.curselection())
        if not sel:
            messagebox.showinfo("edit", "Select a row first."); return
        i = int(sel[0])
        new = self._edit_dialog(self._structures[i])
        if new is not None:
            self._structures[i] = new; self._refresh_list()

    def _remove_selected(self):
        sel = list(self._listbox.curselection())
        for i in reversed(sel):
            if 0 <= i < len(self._structures):
                del self._structures[i]
        self._refresh_list()

    def _edit_dialog(self, s: _Structure,
                       *, allow_kind_change: bool = True
                       ) -> "_Structure | None":
        """Editor: label, zone-axis list, thickness range, tile, tilt
        range, in-plane rotation range.  For CIFs the path is read-
        only; for ASE the builder fn/args/kwargs are text fields."""
        dlg = ctk.CTkToplevel(self)
        dlg.title("Structure")
        dlg.geometry("620x560")
        try: dlg.transient(self.winfo_toplevel())
        except Exception: pass
        dlg.grab_set()

        lab = ctk.StringVar(value=s.label)
        za_str = ctk.StringVar(value="; ".join(
            "(" + ",".join(str(int(x)) for x in za) + ")"
            for za in s.zone_axes))
        tile_str = ctk.StringVar(value=",".join(map(str, s.tile_xyz)))
        crop_str = ctk.StringVar(value=f"{s.thickness_crop_A:g}")
        tilt_mrad = ctk.DoubleVar(value=float(s.tilt_mrad_range))
        ip0, ip1 = s.in_plane_range_deg
        ip_str = ctk.StringVar(value=f"{ip0:g},{ip1:g}")
        cryst_var = ctk.StringVar(value=s.crystallinity)

        # CIF path or ASE builder block (mutually exclusive).
        cif_var = ctk.StringVar(value=s.cif_path or "")
        ase_fn = ctk.StringVar(value=(s.ase_builder or {}).get(
            "fn", "bulk"))
        ase_args = ctk.StringVar(value=",".join(
            map(str, (s.ase_builder or {}).get("args", []))))
        ase_kwargs = ctk.StringVar(value=json.dumps(
            (s.ase_builder or {}).get("kwargs", {})))

        ctk.CTkLabel(dlg, text="Label").pack(anchor="w", padx=8,
                                                    pady=(6, 0))
        ctk.CTkEntry(dlg, textvariable=lab, width=200
                       ).pack(anchor="w", padx=8)
        if s.kind == "cif":
            ctk.CTkLabel(dlg, text="CIF path").pack(anchor="w",
                                                        padx=8, pady=(6, 0))
            ctk.CTkEntry(dlg, textvariable=cif_var, width=480
                           ).pack(anchor="w", padx=8)
        else:
            box = ctk.CTkFrame(dlg, border_width=1)
            box.pack(fill="x", padx=8, pady=(8, 0))
            ctk.CTkLabel(box, text="ASE builder").pack(anchor="w",
                                                            padx=4)
            r = ctk.CTkFrame(box, fg_color="transparent")
            r.pack(fill="x", padx=4, pady=2)
            ctk.CTkLabel(r, text="fn (ase.build.…)").pack(side="left")
            ctk.CTkEntry(r, textvariable=ase_fn, width=120
                           ).pack(side="left", padx=4)
            ctk.CTkLabel(box, text="args (comma-separated, e.g. Si)"
                           ).pack(anchor="w", padx=4)
            ctk.CTkEntry(box, textvariable=ase_args, width=420
                           ).pack(anchor="w", padx=4, pady=2)
            ctk.CTkLabel(box,
                           text="kwargs (JSON, e.g. {\"a\":5.43,"
                                  "\"crystalstructure\":\"diamond\","
                                  "\"cubic\":true})"
                           ).pack(anchor="w", padx=4)
            ctk.CTkEntry(box, textvariable=ase_kwargs, width=420
                           ).pack(anchor="w", padx=4, pady=2)
            ctk.CTkLabel(box,
                text=("⚠ IMPORTANT: for cubic/diamond/zincblende/rocksalt "
                       "structures pass cubic:true so bulk() returns the "
                       "CONVENTIONAL (orthogonal) cell.  Without it, "
                       "bulk() returns the primitive rhombohedral cell — "
                       "non-orthogonal — and abTEM will refuse to "
                       "simulate it."),
                justify="left", wraplength=500,
                font=("Segoe UI", 9), text_color=("#a50", "#fb0")
                ).pack(anchor="w", padx=4, pady=(2, 4))

        ctk.CTkLabel(dlg,
            text=("Zone axes — semicolon-separated, each (h,k,l).  "
                   "A grain picks one at random.  Examples:\n"
                   "   (0,0,1)              one ZA only (Phase-1 style)\n"
                   "   (0,0,1); (1,1,0); (1,1,1)   three discrete ZAs"),
            justify="left", wraplength=580, font=("Segoe UI", 9)
            ).pack(anchor="w", padx=8, pady=(8, 0))
        ctk.CTkEntry(dlg, textvariable=za_str, width=560
                       ).pack(anchor="w", padx=8, pady=2)

        r = ctk.CTkFrame(dlg, fg_color="transparent")
        r.pack(fill="x", padx=8, pady=(8, 0))
        ctk.CTkLabel(r, text="Tile X,Y,Z (unit cells)").pack(side="left")
        ctk.CTkEntry(r, textvariable=tile_str, width=110
                       ).pack(side="left", padx=4)
        add_help_button(r,
            "Supercell repetitions along the in-plane (X, Y) and "
            "beam (Z) directions, AFTER zone-axis orientation.  Z = "
            "thickness in unit cells.  For Si cubic (5.43 Å) along "
            "[001], (2,2,15) gives ~80 Å foil.  X and Y are tiled to "
            "make the in-plane cell large enough that auto-square-"
            "tiling doesn't have to do much extra work.").pack(
            side="left", padx=4)
        ctk.CTkLabel(r,
            text="Trim foil thickness to (Å, 0 = no trim)").pack(
            side="left", padx=(12, 4))
        ctk.CTkEntry(r, textvariable=crop_str, width=80
                       ).pack(side="left", padx=2)
        add_help_button(r,
            "OPTIONAL post-tile trim of the foil along the beam (Z).\n\n"
            "The Z tile (third number in 'Tile X,Y,Z') sets the "
            "structure thickness in WHOLE unit cells:\n"
            "    thickness_Å  =  Z_tile  ×  c_unit\n"
            "e.g. for cubic Si (c=5.43 Å), Z_tile=15 → ~81 Å foil.\n\n"
            "If you need a NON-integer-cell thickness (say exactly "
            "60 Å), set this field to 60.  The atoms outside that "
            "centred slab are simply dropped before the multislice "
            "integration.\n\n"
            "Leave 0 to use the full Z_tile × c_unit thickness — "
            "which is what you usually want.").pack(side="left", padx=4)

        r2 = ctk.CTkFrame(dlg, fg_color="transparent")
        r2.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(r2, text="Beam tilt range (mrad)").pack(
            side="left")
        ctk.CTkEntry(r2, textvariable=tilt_mrad, width=80
                       ).pack(side="left", padx=4)
        add_help_button(r2,
            "Off-axis beam tilt in mrad.  Implemented as BEAM TILT "
            "(abTEM's Probe(tilt=…) kwarg), not sample rotation — "
            "the atomic cell stays orthogonal, no orthogonalize_cell "
            "call, no RAM blow-up.\n\n"
            "Each grain draws a tilt magnitude in [0, this value] "
            "and a random azimuth.  0 mrad = strictly on-zone.\n\n"
            "Recommended range:\n"
            "   0–10 mrad   ≈ small mistilt, very accurate\n"
            "   10–50 mrad  ≈ moderate off-axis, still OK\n"
            "   >50 mrad    ≈ small-angle approximation breaks\n"
            "Reference: 1 mrad ≈ 0.057°, so 50 mrad ≈ 2.9°.\n\n"
            "For LARGE off-axis variation, add another zone axis to "
            "the list above instead of using big tilts.").pack(
            side="left", padx=4)
        ctk.CTkLabel(r2, text="In-plane (lo,hi deg)").pack(side="left",
                                                                  padx=(12, 4))
        ctk.CTkEntry(r2, textvariable=ip_str, width=110
                       ).pack(side="left", padx=2)

        # ---- Crystallinity (click to set) --------------------------------
        r3 = ctk.CTkFrame(dlg, fg_color="transparent")
        r3.pack(fill="x", padx=8, pady=(8, 0))
        ctk.CTkLabel(r3, text="Crystallinity").pack(side="left")
        ctk.CTkSegmentedButton(
            r3, values=list(CRYSTALLINITY_ORDER),
            variable=cryst_var, width=280).pack(side="left", padx=6)
        add_help_button(r3,
            "Crystallinity of this domain — a separate GROUND-TRUTH class "
            "axis (so crystalline vs amorphous grains of the same material "
            "and orientation are different classes the model should "
            "separate).\n\n"
            "  crystalline — sharp Bragg spots (no disorder).\n"
            "  partial     — weakened spots + mild diffuse (thermal-"
            "diffuse look), via frozen-phonon averaging (σ≈0.12 Å).\n"
            "  amorphous   — spots washed out, diffuse rings dominate "
            "(σ≈0.6 Å, 8 configs).\n\n"
            "Physically this is static/thermal displacement disorder "
            "averaged over frozen-phonon configurations (Debye-Waller "
            "damping of the peaks + rising diffuse background).  "
            "Amorphous domains cost more (more configs → more "
            "multislice passes).").pack(side="left", padx=4)

        result = {"out": None}
        def _ok():
            try:
                # Parse zone axes list
                raw = za_str.get().replace(" ", "")
                if not raw:
                    raise ValueError("at least one zone axis required")
                items = [x for x in raw.split(";") if x]
                zas = []
                for item in items:
                    item = item.strip("()")
                    parts = [p for p in item.split(",") if p != ""]
                    if len(parts) != 3:
                        raise ValueError(
                            f"ZA must be (h,k,l): bad item {item!r}")
                    zas.append(tuple(int(p) for p in parts))
                if not zas:
                    raise ValueError("no zone axes parsed")
                tile = tuple(int(x) for x in
                              tile_str.get().replace(" ", "").split(",")
                              if x != "")
                if len(tile) != 3:
                    raise ValueError("Tile must be 3 ints: nx,ny,nz")
                try:
                    crop = float(crop_str.get())
                except Exception:
                    crop = 0.0
                ip_parts = [p.strip() for p in ip_str.get().split(",")
                             if p.strip()]
                if len(ip_parts) != 2:
                    raise ValueError("in-plane: 'lo,hi'")
                ip_lo, ip_hi = float(ip_parts[0]), float(ip_parts[1])
                new = _Structure(
                    kind=s.kind,
                    cif_path=(cif_var.get() or None
                                if s.kind == "cif" else None),
                    ase_builder=(None if s.kind == "cif"
                                  else {"fn": ase_fn.get(),
                                          "args": [a.strip() for a in
                                                    ase_args.get().split(",")
                                                    if a.strip()],
                                          "kwargs": json.loads(
                                              ase_kwargs.get() or "{}")}),
                    label=lab.get() or "?",
                    zone_axes=zas,
                    tile_xyz=tile,
                    thickness_crop_A=crop,
                    tilt_mrad_range=float(tilt_mrad.get()),
                    in_plane_range_deg=(ip_lo, ip_hi),
                    crystallinity=cryst_var.get())
                result["out"] = new
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("Structure", str(e), parent=dlg)
        def _cancel(): dlg.destroy()

        bb = ctk.CTkFrame(dlg, fg_color="transparent")
        bb.pack(side="bottom", fill="x", padx=8, pady=8)
        ctk.CTkButton(bb, text="OK", width=80,
                        fg_color=("#2D7A2D", "#1F7A1F"),
                        command=_ok).pack(side="right", padx=2)
        ctk.CTkButton(bb, text="Cancel", width=80,
                        command=_cancel).pack(side="right", padx=2)
        self.wait_window(dlg)
        return result["out"]

    # ===================================================================
    # Simulation
    # ===================================================================
    def _on_stop(self):
        """Cooperative stop: sets a flag the worker thread polls
        between grains.  Mid-multislice work can't be killed safely
        (would corrupt abTEM's CUDA/FFT state), so the cancel takes
        effect at the next grain boundary — usually within seconds."""
        if not self._sim_busy:
            return
        self._sim_stop_requested = True
        try:
            self._stop_btn.configure(state="disabled",
                                          text="Stopping…")
        except Exception:
            pass
        self._sim_status.configure(
            text=(self._sim_status.cget("text")
                  + "\n[stop requested — finishing current grain "
                     "then aborting]"))

    def _check_stop(self):
        """Raise _SimAborted if the user hit Stop.  Called at safe
        points in the worker loop."""
        if self._sim_stop_requested:
            raise _SimAborted("simulation stopped by user")

    def _set_running_ui(self, running: bool):
        """Toggle Stop/Simulate enable state."""
        try:
            if running:
                self._stop_btn.configure(state="normal", text="Stop ■")
            else:
                self._stop_btn.configure(state="disabled",
                                              text="Stop ■")
        except Exception:
            pass

    def _on_simulate(self):
        if self._sim_busy:
            messagebox.showinfo("Simulate",
                "A simulation is already running."); return
        if len(self._structures) < 1:
            messagebox.showinfo("Simulate",
                "Add at least one structure first."); return
        failures = _check_deps()
        if failures:
            names = ", ".join(n for n, _ in failures)
            details = "\n".join(f"  {n}:  {e}" for n, e in failures)
            messagebox.showerror("Dependencies",
                f"The following package(s) could not be imported:  "
                f"{names}\n\n"
                f"Underlying errors:\n{details}\n\n"
                f"Active Python:\n{_env_diag()}\n\n"
                f"If you're SURE the package is installed, the GUI is "
                f"likely running in a different env than the one you "
                f"installed it into.  Activate that env first, or run "
                f"   {os.path.basename(__import__('sys').executable)} "
                f"-m pip install <pkg>\n"
                f"to install into the env the GUI is using.")
            # Also print to stdout so the terminal log captures it.
            print(f"[synth] dep check failed:\n{_env_diag()}\n"
                   f"{details}", flush=True)
            return
        # Output dir = <runs root>/synth/<basename>_<timestamp>/, where the
        # runs root is the SAME one the rest of the app writes to (runs/_gui,
        # etc.) — resolved relative to the launch/working directory via
        # os.getcwd(), exactly like runner.py's os.path.join("runs","_gui").
        # NOT relative to the src/ package dir, so synth lands next to the
        # GUI's runs on any machine without hard-coding a path.
        base = self.out_basename.get().strip() or "phantom"
        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(os.getcwd(), "runs", "synth",
                                f"{base}_{ts}")
        os.makedirs(outdir, exist_ok=True)
        self._sim_busy = True
        self._sim_stop_requested = False
        self._set_running_ui(True)
        self._sim_status.configure(
            text=f"simulating → {outdir} …")
        threading.Thread(target=self._sim_worker,
                            args=(outdir,), daemon=True).start()

    def _sim_worker(self, outdir: str):
        try:
            cube, classmaps, meta = self._run_simulation(outdir)
            cube_path = os.path.join(outdir, "phantom.cube.npy")
            np.save(cube_path, cube.astype(np.float32))
            cmap_paths = {}
            for k, m in classmaps.items():
                p = os.path.join(outdir, f"phantom.classmap_{k}.npy")
                np.save(p, m.astype(np.int32))
                cmap_paths[k] = p
            meta_path = os.path.join(outdir, "phantom.sim_meta.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2,
                            default=lambda o: repr(o))
            # Register as a SAMPLES entry so Pre-processing / Training
            # pick it up.
            try:
                key = register_runtime_sample(
                    cube_path,
                    scan_shape=tuple(cube.shape[:2]),
                    vmax=float(meta.get("recommended_vmax", 1.0)),
                    center_mask_radius=0)
            except Exception:
                key = None

            self._last_cube_path = cube_path
            self._last_classmaps = cmap_paths
            self._last_meta = meta

            def _done():
                # Auto-flag the just-simulated phantom in the Dataset
                # (Pre-processing) tab: pre-fill its path so it's one
                # click from Load — no need to browse for it.
                flagged = False
                try:
                    pre = getattr(self.app, "pre", None)
                    if pre is not None and hasattr(pre, "_path_var"):
                        pre._path_var.set(cube_path)
                        flagged = True
                except Exception:
                    pass
                self._sim_status.configure(
                    text=(f"sim done.  cube: {cube_path}\n"
                          f"classmaps written; registered as "
                          f"SAMPLES['{key}'].\n"
                          + ("→ path pre-filled in the Dataset tab — "
                             "go there and click Load."
                             if flagged else
                             "Switch to Pre-processing or Training "
                             "to use it.")))
                # Auto-render the phantom preview (class-map +
                # per-phase structure views) in the metric pane and
                # save as phantom_preview.png next to the cube.
                try:
                    self._render_phantom_preview(
                        cube_path=cube_path,
                        classmap_B_path=cmap_paths.get("B"),
                        meta=meta,
                        save_path=os.path.join(
                            os.path.dirname(cube_path),
                            "phantom_preview.png"))
                except Exception as e:
                    print(f"[synth] preview render failed: {e!r}",
                          flush=True)
            self.after(0, _done)
        except _SimAborted:
            print("[synth] sim aborted by user (partial cube NOT "
                  "written).", flush=True)
            self.after(0, lambda: self._sim_status.configure(
                text="aborted by user.  Partial cube was NOT saved."))
        except Exception as e:
            err = repr(e)
            self.after(0, lambda: messagebox.showerror(
                "Simulate failed", err))
            self.after(0, lambda: self._sim_status.configure(
                text=f"failed: {err}"))
        finally:
            self._sim_busy = False
            self._sim_stop_requested = False
            self.after(0, lambda: self._set_running_ui(False))

    # ===================================================================
    # Simulation engine (Phase 1/2/3 unified):
    #   1) Build a grain_id map (block | voronoi)
    #   2) Per grain → draw (phase_id, ZA_index, in-plane rot,
    #                          thickness, tilt off-axis)
    #   3) Build / cache per-(phase, ZA, thickness, tilt) atomic model
    #      and PRISM S-matrix (or multislice Potential)
    #   4) Scan probe-by-probe with dispatch by grain assignment
    #   5) Write cube + classmap_A/B/C + sim_meta
    # ===================================================================
    def _run_simulation(self, outdir: str,
                          *, engine_override: "str | None" = None,
                          reuse_grain_assignments: "dict | None" = None,
                          out_basename: "str | None" = None):
        from ase.io import read as ase_read
        from ase import Atoms
        import abtem

        # Resolve abTEM classes/functions ONCE per sim — the user's
        # abTEM version may stash them under different sub-modules
        # than the top-level we expected.
        PotentialCls   = _resolve_abtem("Potential")
        ProbeCls       = _resolve_abtem("Probe")
        SMatrixCls     = _resolve_abtem("SMatrix")
        PixelDetCls    = _resolve_abtem("PixelatedDetector")
        try:
            scan_fn    = _resolve_abtem("scan")
        except RuntimeError:
            scan_fn    = None  # multislice path may use SMatrix.scan
        print(f"[synth] abtem version = "
              f"{getattr(abtem, '__version__', '?')}\n"
              f"        Potential -> {PotentialCls.__module__}."
              f"{PotentialCls.__name__}\n"
              f"        SMatrix   -> {SMatrixCls.__module__}."
              f"{SMatrixCls.__name__}\n"
              f"        Probe     -> {ProbeCls.__module__}."
              f"{ProbeCls.__name__}",
              flush=True)

        rng = np.random.default_rng(int(self.rng_seed.get()))
        Ny, Nx = int(self.scan_ny.get()), int(self.scan_nx.get())
        step = float(self.scan_step_A.get())
        engine = engine_override or self.engine.get()
        kV = float(self.beam_kV.get())
        sa = float(self.conv_mrad.get())
        det_size = int(self.det_size.get())
        det_max = float(self.det_max_mrad.get())
        dose = float(self.dose_e_A2.get())

        # ---- 1) grain_id map ----------------------------------------
        if reuse_grain_assignments is not None:
            grain_map = reuse_grain_assignments["grain_map"]
            grain_props = reuse_grain_assignments["grain_props"]
        else:
            layout = self.layout.get()
            if layout == "voronoi":
                n_seeds = max(2, int(self.n_voronoi_seeds.get()))
                grain_map = _make_voronoi_grain_map(Ny, Nx, n_seeds, rng)
            else:
                grain_map = _make_block_grain_map(Ny, Nx,
                                                       len(self._structures))
            grain_ids = np.unique(grain_map)
            # ---- 2) per-grain draws ------------------------------------
            grain_props: dict = {}
            for gid in grain_ids:
                # Phase choice — uniform over structures (Voronoi);
                # for block layout the grain_map already encodes phase_id
                # so we lock phase to that id.
                if self.layout.get() == "block":
                    phase_id = int(gid) % len(self._structures)
                else:
                    phase_id = int(rng.integers(0, len(self._structures)))
                s = self._structures[phase_id]
                za_idx = int(rng.integers(0, len(s.zone_axes)))
                za = tuple(s.zone_axes[za_idx])
                ip_lo, ip_hi = s.in_plane_range_deg
                in_plane_deg = float(rng.uniform(ip_lo, ip_hi))
                # Beam tilt — uniform-random magnitude in [0, range]
                # and random azimuth.  Converted to (tx, ty) mrad
                # below and passed to abTEM's Probe(tilt=...) /
                # SMatrix(tilt=...).  No cell rotation needed —
                # cell stays exactly as built.
                if s.tilt_mrad_range > 0:
                    tilt_mag_mrad = float(rng.uniform(
                        0.0, float(s.tilt_mrad_range)))
                    tilt_az_deg = float(rng.uniform(0.0, 360.0))
                else:
                    tilt_mag_mrad = 0.0
                    tilt_az_deg = 0.0
                # Quantise to 1 mrad bins so the S-matrix cache hits.
                tilt_mag_q = round(tilt_mag_mrad)
                tilt_az_q = round(tilt_az_deg / 5.0) * 5.0  # 5° bins
                grain_props[int(gid)] = {
                    "phase_id":        phase_id,
                    "za":              za,
                    "za_idx":          za_idx,
                    "tile_xyz":        list(s.tile_xyz),
                    "thickness_crop_A": float(s.thickness_crop_A),
                    "tilt_mag_mrad":   tilt_mag_q,
                    "tilt_az_deg":     tilt_az_q,
                    "in_plane_deg":    in_plane_deg,
                    "crystallinity":   s.crystallinity,
                }

        # ---- Define cache cap up front so the pre-flight warnings
        #      that reference it (combo count, RAM estimate) can use
        #      it without an UnboundLocalError.  The actual sm_cache
        #      OrderedDict is constructed just before the scan loop.
        MAX_CACHE = 1

        # ---- Diagnostic dump: list every distinct (phase, ZA,
        #      thickness, tilt) combo the sim is about to run, so the
        #      user can confirm visually that ZA / tilt / thickness
        #      variation actually fired.  If every grain prints the
        #      same combo, the user's structure list / layout choice
        #      didn't exercise the variability they set up.
        from collections import Counter
        combo_counter: Counter = Counter()
        for gid, gp in grain_props.items():
            combo_counter[(gp["phase_id"], gp["za"],
                              tuple(gp.get("tile_xyz", (1, 1, 1))),
                              gp.get("thickness_crop_A", 0.0),
                              gp.get("tilt_mag_mrad", 0.0),
                              gp["in_plane_deg"] // 30 * 30)] += 1
        print(f"\n[synth] grain summary — {len(grain_props)} grains, "
              f"{len(combo_counter)} distinct (phase, ZA, tile_xyz, "
              f"z-crop, beam_tilt_mrad, rot-bin) combos:", flush=True)
        for combo, n in combo_counter.most_common():
            ph, za, tile, crop, tlt_mrad, ipb = combo
            thk = (f"crop={crop:g}Å" if crop > 0
                     else f"tile_z={tile[2]}")
            print(f"   ×{n:>3d}  phase={ph} ZA={za} tile={tile} "
                  f"{thk} beam_tilt={tlt_mrad:g}mrad "
                  f"rot~{ipb:g}°", flush=True)
        print(f"[synth] if you expected more variation: switch layout "
              f"to 'voronoi' with more seeds (currently "
              f"{self.layout.get()}, "
              f"n_seeds={int(self.n_voronoi_seeds.get())})\n",
              flush=True)

        # ---- Pre-flight sanity: beam tilt > 50 mrad pushes the
        # small-angle approximation past its validity range.
        big_tilts = [(gid, gp.get("tilt_mag_mrad", 0.0))
                       for gid, gp in grain_props.items()
                       if gp.get("tilt_mag_mrad", 0.0) > 50.0]
        if big_tilts:
            print(f"[synth] WARNING: {len(big_tilts)} grain(s) have "
                   f"beam tilt > 50 mrad (max = "
                   f"{max(t for _, t in big_tilts):.1f} mrad ≈ "
                   f"{max(t for _, t in big_tilts)*0.0573:.2f}°).  "
                   f"Diffraction patterns will be heavily shifted "
                   f"and may exit the detector window.  Lower "
                   f"tilt_mrad_range or use another zone axis.",
                   flush=True)
        # ---- Pre-flight sanity: too many distinct combos = OOM risk.
        if len(combo_counter) > MAX_CACHE * 4:
            print(f"[synth] WARNING: {len(combo_counter)} distinct "
                   f"(phase,ZA,thickness,tilt,rot) combos — cache "
                   f"will thrash (size {MAX_CACHE}).  Build cost per "
                   f"grain ≈ one full multislice; sim will be slow.",
                   flush=True)
        # ---- Peak-RAM estimate (very rough): one SMatrix lives at a
        #      time given MAX_CACHE=1.  Size scales with gpts²·n_slices
        #      ·n_plane_waves.  For Si@200kV, 256 gpts, ~40 slices,
        #      ~20 plane-waves → ~80 MB float32 per SMatrix.  Cube
        #      itself is Ny·Nx·det²·4 bytes.
        gpts = int(self.pot_gpts.get())
        est_sm = gpts * gpts * 40 * 20 * 4 / 1e9  # GB, rough
        est_cube = Ny * Nx * det_size * det_size * 4 / 1e9  # GB
        print(f"[synth] peak-RAM estimate:  SMatrix ≈ {est_sm:.2f} GB "
               f"+  cube ≈ {est_cube:.2f} GB "
               f"+ Python/abtem overhead.\n"
               f"        If your RAM is < 8 GB, drop scan size, "
               f"detector px, or potential gpts before hitting "
               f"Simulate.", flush=True)

        # ---- 3) cache per-(phase, ZA, thickness, tilt) S-matrix.
        #          STRICT MEMORY CAP.  Each SMatrix can be hundreds of
        #          MB; with multi-ZA + tilts + thicknesses the
        #          combinatorial space inflates to GBs and starves
        #          the OS into swap.  Keep only the most-recent
        #          MAX_CACHE entries; evict the oldest as needed.
        # ------------------------------------------------------------
        from collections import OrderedDict
        # Cache size = MAX_CACHE (defined up top so pre-flight
        # warnings can reference it).  Default 1 = drop the old
        # SMatrix before building the next.  Cost: every distinct
        # combo rebuilds = slower.  Benefit: peak RAM = ONE potential
        # + ONE SMatrix, bounded regardless of how many grains / ZAs
        # / tilts you set up.  Raise MAX_CACHE manually if you have
        # lots of free RAM.
        sm_cache: "OrderedDict[tuple, tuple]" = OrderedDict()

        def _cache_get(key):
            v = sm_cache.get(key)
            if v is not None:
                sm_cache.move_to_end(key)
            return v

        def _cache_put(key, value):
            sm_cache[key] = value
            sm_cache.move_to_end(key)
            while len(sm_cache) > MAX_CACHE:
                evicted_key, _ = sm_cache.popitem(last=False)
                print(f"[synth] evicting cache entry {evicted_key} "
                      f"(MAX_CACHE={MAX_CACHE})", flush=True)
            # Trigger GC after eviction so SMatrix arrays are
            # actually returned to the OS, not just dropped from a
            # Python dict.
            import gc; gc.collect()

        def _get_engine_obj(phase_id, za, tile_xyz, thickness_crop_A,
                              tilt_mag_mrad, tilt_az_deg,
                              crystallinity="crystalline"):
            key = (phase_id, za, tuple(tile_xyz),
                   round(thickness_crop_A, 3),
                   round(tilt_mag_mrad, 1), round(tilt_az_deg, 1),
                   crystallinity, engine)
            hit = _cache_get(key)
            if hit is not None:
                return hit
            s = self._structures[phase_id]
            # ---- Base atoms in the ORIGINAL crystal frame ----------
            if s.kind == "cif":
                if not s.cif_path or not os.path.exists(s.cif_path):
                    raise RuntimeError(
                        f"CIF not found: {s.cif_path!r}")
                atoms_base = ase_read(s.cif_path)
            else:
                from ase.build import bulk, surface, mx2, fcc100, fcc111  # noqa
                fn = (s.ase_builder or {}).get("fn", "bulk")
                args = (s.ase_builder or {}).get("args", [])
                kwargs = (s.ase_builder or {}).get("kwargs", {})
                builder_fn = {
                    "bulk": bulk, "surface": surface, "mx2": mx2,
                    "fcc100": fcc100, "fcc111": fcc111,
                }.get(fn)
                if builder_fn is None:
                    raise RuntimeError(f"unknown ASE builder: {fn!r}")
                atoms_base = builder_fn(*args, **kwargs)
            # ---- NaPHI-style: tile → wipe cell → rotate to ZA →
            #      additional rotation for off-axis tilt → center with
            #      vacuum padding so the bounding box is orthogonal.
            #      No orthogonalize_cell, no surface(), no auto-tile.
            try:
                atoms = _build_atoms_vacuum_box(
                    atoms_base,
                    zone_axis=za,
                    tile_xyz=tile_xyz,
                    tilt_mag_mrad=float(tilt_mag_mrad),
                    tilt_az_deg=float(tilt_az_deg),
                    vacuum_pad_A=float(self.vacuum_pad_A.get()),
                    thickness_crop_A=float(thickness_crop_A))
            except Exception as e:
                raise RuntimeError(
                    f"atom build failed for {s.label!r}, ZA={za}, "
                    f"tilt={tilt_mag_mrad} mrad: {e!r}") from e
            # ---- Crystallinity: wrap atoms in FrozenPhonons so the
            #      multislice averages over displacement configs → smooth
            #      diffuse scattering + Debye-Waller damping of the Bragg
            #      peaks.  sigma=0 → sharp crystalline (plain atoms).
            sigma_A, n_cfg = CRYSTALLINITY_LEVELS.get(
                str(crystallinity), (0.0, 1))
            if sigma_A > 0 and n_cfg > 1:
                from abtem import FrozenPhonons
                # Seed per-combo so the disorder is reproducible but
                # differs between grains/levels.
                fp_seed = (abs(hash((phase_id, za, crystallinity)))
                             % (2 ** 31))
                pot_src = FrozenPhonons(atoms, num_configs=int(n_cfg),
                                          sigmas=float(sigma_A),
                                          seed=int(fp_seed))
            else:
                pot_src = atoms
            print(f"[synth] feeding Potential (phase={phase_id}, "
                  f"za={za}, tilt={tilt_mag_mrad} mrad, "
                  f"crystallinity={crystallinity} σ={sigma_A}Å×{n_cfg}):\n"
                  f"{_cell_diag(atoms)}", flush=True)
            # NaPHI pattern: abTEM auto-periodic in xy; z_periodic is
            # True by default in 1.0.0beta33.  Vacuum padding ensures
            # the periodic wrap-around lands in zero-potential space,
            # so this works for non-periodic clusters as well as the
            # original periodic crystal supercells.
            # The 'finite' engine uses the accurate finite-projection
            # potential with fine slices — this is what removes the
            # inter-spot streak/'diffuse' artifacts of the fast infinite
            # projection.  The scanned engines keep 'infinite' for speed.
            _proj = "finite" if engine == "finite" else "infinite"
            _slth = 1.0 if engine == "finite" else 2.0
            pot = PotentialCls(pot_src, gpts=int(self.pot_gpts.get()),
                                  slice_thickness=_slth,
                                  parametrization="kirkland",
                                  projection=_proj)
            if engine == "finite":
                _cache_put(key, ("finite", pot))
            elif engine == "PRISM":
                # abTEM 1.0.0beta33 PRISM workflow:
                #   sm = SMatrix(energy, expansion_cutoff, interp)
                #   sm.match_grid(potential)
                #   sma = sm.multislice(potential)   # → SMatrixArray
                #   sma.collapse(positions=[...])    # batched probes
                #
                # `expansion_cutoff` (mrad) bounds the angular range
                # the scattering matrix represents.  It MUST cover
                # everything the detector will measure — otherwise the
                # SMatrix is band-limited below the detector cutoff and
                # high-q Bragg peaks vanish.  Use the user's detector
                # max_angle as the cutoff, not the probe convergence.
                exp_cutoff = max(float(det_max), 4.0 * float(sa))
                # Tilt is already baked into the atoms (NaPHI pattern);
                # the beam is along +z.  No `tilt=` kwarg needed.
                sm = SMatrixCls(energy=kV * 1e3,
                                   expansion_cutoff=exp_cutoff,
                                   interpolation=1)
                sm.match_grid(pot)
                sma = sm.multislice(pot)
                _cache_put(key, ("prism", sma))
            else:
                _cache_put(key, ("ms", pot))
            return sm_cache[key]

        probe = ProbeCls(energy=kV * 1e3,
                            semiangle_cutoff=sa,
                            sampling=0.1)

        # ---- 4) scan, grouped BY GRAIN so each grain runs one
        #          batched multislice / SMatrix.collapse call. The
        #          abTEM 1.0.0betaXX API needs either a GridScan
        #          (rectangular only) or a position list passed to
        #          probe.multislice / SMatrixArray.collapse — so we
        #          gather positions per grain then dispatch once.
        # ------------------------------------------------------------
        try:
            import cv2 as _cv2
        except Exception:
            _cv2 = None
        cube = np.zeros((Ny, Nx, det_size, det_size), dtype=np.float32)
        cm_A = np.zeros((Ny, Nx), dtype=np.int32)   # phase
        cm_B = np.zeros((Ny, Nx), dtype=np.int32)   # phase + orientation
        cm_crys = np.zeros((Ny, Nx), dtype=np.int32)  # phase + crystallinity
        cm_main = np.zeros((Ny, Nx), dtype=np.int32)  # phase+orient+cryst (in-plane inv.)
        cm_C = np.zeros((Ny, Nx), dtype=np.int32)   # full incl in-plane (control)
        b_keys: dict[tuple, int] = {}
        d_keys: dict[tuple, int] = {}
        m_keys: dict[tuple, int] = {}
        c_keys: dict[tuple, int] = {}

        # Bucket pixels by grain id.
        from collections import defaultdict
        grain_pixels: "dict[int, list[tuple[int, int]]]" = \
            defaultdict(list)
        for yy in range(Ny):
            for xx in range(Nx):
                grain_pixels[int(grain_map[yy, xx])].append((yy, xx))

        n_pix = Ny * Nx
        done = 0
        t0 = time.time()
        for gid, pixels in grain_pixels.items():
            # Cooperative cancel point — between grains is the only
            # safe place; mid-multislice / mid-collapse can't be
            # interrupted without leaking abTEM internal state.
            self._check_stop()
            gp = grain_props[gid]
            phase_id = gp["phase_id"]
            kind, obj = _get_engine_obj(
                phase_id, gp["za"],
                gp.get("tile_xyz", (1, 1, 1)),
                gp.get("thickness_crop_A", 0.0),
                gp.get("tilt_mag_mrad", 0.0),
                gp.get("tilt_az_deg", 0.0),
                gp.get("crystallinity", "crystalline"))
            # Tilt is in the atoms now (NaPHI pattern); the probe is
            # along +z for every grain.  Match_grid is required so the
            # probe's FFT grid matches whatever vacuum-padded cell the
            # potential ended up with — that varies grain-by-grain.
            if kind in ("ms", "finite"):
                try:
                    probe.match_grid(obj)
                except Exception:
                    pass
            positions = [(xx * step, yy * step) for (yy, xx) in pixels]
            det = PixelDetCls(max_angle=det_max, resample=False)
            try:
                if kind == "prism":
                    # SMatrix.build() → SMatrixArray; .collapse() takes
                    # arbitrary probe positions.  Cached `obj` is
                    # already the SMatrixArray.
                    waves = obj.collapse(positions=positions)
                elif kind == "finite":
                    # ONE high-quality pattern for the whole grain: a
                    # single probe at the (square) box centre.  It is
                    # broadcast to every pixel of the grain below — each
                    # pixel then gets independent shot noise (and the
                    # per-grain in-plane rotation) downstream.
                    try:
                        ext = obj.extent
                        cen = (float(ext[0]) / 2.0, float(ext[1]) / 2.0)
                    except Exception:
                        cen = positions[len(positions) // 2]
                    waves = probe.multislice(positions=[cen],
                                                 potential=obj, pbar=False)
                else:
                    waves = probe.multislice(positions=positions,
                                                 potential=obj,
                                                 pbar=False)
            except Exception as e:
                raise RuntimeError(
                    f"abTEM scan failed for grain {gid} "
                    f"(phase={phase_id} za={gp['za']} "
                    f"tile_xyz={gp.get('tile_xyz')} "
                    f"crop_A={gp.get('thickness_crop_A')} "
                    f"engine={kind}):  {e!r}") from e
            arr = np.asarray(det.detect(waves))
            # arr shape: (N_positions, h, w) for batched, or (h, w)
            # for single position — collapse to a 3-D array.
            if arr.ndim == 2:
                arr = arr[None]
            # 'finite' produced ONE pattern — replicate it to every pixel
            # of the grain so the downstream per-pixel noise/rotation loop
            # works unchanged.
            if kind == "finite" and arr.shape[0] == 1 and len(pixels) > 1:
                arr = np.repeat(arr, len(pixels), axis=0)
            # Resample each pattern to the user's detector grid.
            # IMPORTANT: pad to square FIRST so the angular geometry
            # of kx vs ky is preserved.  abTEM emits patterns whose
            # pixel aspect matches the in-plane cell aspect (after
            # max-angle clip), which can be wildly non-square.  Resize
            # then becomes an isotropic downscale, not an anisotropic
            # stretch.
            h, w = arr.shape[-2:]
            if h != w:
                side = max(h, w)
                pad_h_top = (side - h) // 2
                pad_h_bot = side - h - pad_h_top
                pad_w_l   = (side - w) // 2
                pad_w_r   = side - w - pad_w_l
                arr = np.pad(arr,
                                ((0, 0),
                                 (pad_h_top, pad_h_bot),
                                 (pad_w_l, pad_w_r)),
                                mode="constant", constant_values=0.0)
                h = w = side
            if (h, w) != (det_size, det_size):
                if _cv2 is not None:
                    arr_r = np.empty((arr.shape[0], det_size, det_size),
                                          dtype=np.float32)
                    for i in range(arr.shape[0]):
                        arr_r[i] = _cv2.resize(
                            arr[i].astype(np.float32),
                            (det_size, det_size),
                            interpolation=_cv2.INTER_AREA)
                    arr = arr_r
                else:
                    from scipy.ndimage import zoom as _zoom
                    fy = det_size / h; fx = det_size / w
                    arr = np.stack([_zoom(a.astype(np.float32),
                                              (fy, fx), order=1)
                                       for a in arr], axis=0)
            # ---- Dose scaling + (optional) shot noise ------------------
            # Both "shot" and "exact" scale each pattern to the SAME
            # electron count — total_e = dose × step² electrons per probe
            # position — so the only difference between the two modes is
            # whether we Poisson-SAMPLE.  Per-pattern normalisation makes
            # the electron count well-defined regardless of how abTEM
            # normalises the raw detected intensity, so higher dose is
            # correctly correlated with a cleaner pattern in "shot" mode.
            if dose > 0:
                total_e = float(dose) * (float(step) ** 2)
                flat_sum = arr.reshape(arr.shape[0], -1).sum(axis=1)
                scale = np.where(flat_sum > 0,
                                    total_e / np.maximum(flat_sum, 1e-30),
                                    0.0).astype(np.float32)
                expected = (arr * scale[:, None, None]).astype(np.float32)
                if self.noise_mode.get() == "exact":
                    arr = np.clip(expected, 0, None)
                else:  # "shot" — Poisson sample at that electron count
                    arr = rng.poisson(np.clip(expected, 0, None)
                                          ).astype(np.float32)
            ipd = gp["in_plane_deg"]
            for i, (yy, xx) in enumerate(pixels):
                pat = arr[i]
                if abs(ipd) > 1e-3:
                    from scipy.ndimage import rotate as _rotate
                    # order=3 (cubic) NOT order=1: the Bragg spots are
                    # near sub-pixel sharp, and bilinear rotation aliases
                    # them differently at each angle (~28% round-trip
                    # error) — which breaks the in-plane-rotation
                    # invariance the model relies on and makes it split
                    # one orientation into several classes.  Cubic cuts
                    # that to ~4%.  Clip the small spline undershoot back
                    # to >=0 (diffraction intensity can't be negative).
                    pat = _rotate(pat, ipd, reshape=False,
                                     order=3, mode="constant",
                                     cval=0.0)
                    np.clip(pat, 0.0, None, out=pat)
                cube[yy, xx] = pat.astype(np.float32)
                cry = gp.get("crystallinity", "crystalline")
                cm_A[yy, xx] = phase_id
                bk = (phase_id, gp["za"])
                if bk not in b_keys: b_keys[bk] = len(b_keys)
                cm_B[yy, xx] = b_keys[bk]
                # phase + crystallinity
                dk = (phase_id, cry)
                if dk not in d_keys: d_keys[dk] = len(d_keys)
                cm_crys[yy, xx] = d_keys[dk]
                # PRIMARY: phase + orientation + crystallinity, in-plane
                # rotation deliberately EXCLUDED (nuisance → one class).
                mk = (phase_id, gp["za"], cry)
                if mk not in m_keys: m_keys[mk] = len(m_keys)
                cm_main[yy, xx] = m_keys[mk]
                # CONTROL: everything incl in-plane rotation bin.
                rot_bin = int(ipd // 30) % 12
                ck = (phase_id, gp["za"], cry, rot_bin)
                if ck not in c_keys: c_keys[ck] = len(c_keys)
                cm_C[yy, xx] = c_keys[ck]
            done += len(pixels)
            dt = time.time() - t0
            eta = dt * (n_pix - done) / max(done, 1)
            msg = (f"sim {done}/{n_pix} "
                    f"({100*done/n_pix:.0f}%)  eta {eta:.0f}s  "
                    f"|cache|={len(sm_cache)}  "
                    f"grain {gid}: {len(pixels)} px in one call")
            self.after(0, lambda m=msg:
                self._sim_status.configure(text=m))
            print(f"[synth] {msg}", flush=True)

        classmaps = {"A": cm_A, "B": cm_B, "crys": cm_crys,
                     "main": cm_main, "C": cm_C}
        meta = {
            "engine": engine,
            "beam_kV": kV,
            "conv_mrad": sa,
            "det_size": det_size,
            "det_max_mrad": det_max,
            "dose_e_A2": dose,
            "noise_mode": self.noise_mode.get(),
            "scan_shape": [Ny, Nx],
            "scan_step_A": step,
            "layout": self.layout.get(),
            "n_voronoi_seeds": (int(self.n_voronoi_seeds.get())
                                  if self.layout.get() == "voronoi"
                                  else None),
            "rng_seed": int(self.rng_seed.get()),
            "structures": [
                {"label": s.label, "kind": s.kind,
                 "cif_path": s.cif_path,
                 "ase_builder": s.ase_builder,
                 "zone_axes": [list(za) for za in s.zone_axes],
                 "tile_xyz": list(s.tile_xyz),
                 "thickness_crop_A": s.thickness_crop_A,
                 "tilt_mrad_range": s.tilt_mrad_range,
                 "in_plane_range_deg": list(s.in_plane_range_deg),
                 "crystallinity": s.crystallinity}
                for s in self._structures
            ],
            "grain_props": {str(k): v for k, v in grain_props.items()},
            "n_classes_A": int(cm_A.max()) + 1,
            "n_classes_B": int(cm_B.max()) + 1,
            "n_classes_crys": int(cm_crys.max()) + 1,
            "n_classes_main": int(cm_main.max()) + 1,
            "n_classes_C": int(cm_C.max()) + 1,
            "n_unique_smatrix_entries": len(sm_cache),
            "recommended_vmax": float(np.percentile(cube, 99.5)),
        }
        # Stash grain_map so callers (e.g. multislice comparison) can
        # reuse the *exact same* grain layout.  Written to disk too,
        # for offline re-validation.
        np.save(os.path.join(outdir, "phantom.grain_map.npy"),
                  grain_map.astype(np.int32))
        return cube, classmaps, meta

    # ===================================================================
    # Validation: Hungarian alignment + metrics
    # ===================================================================
    def _on_validate(self):
        if self._last_cube_path is None or self._last_classmaps is None:
            messagebox.showinfo("Validate",
                "Run a simulation first (or load a phantom — coming "
                "in P2)."); return
        # Find a trained model run that has inferenced this cube. The
        # cleanest signal is `runs/.../eval/inference.npz` containing
        # an `assigns` array of length Ny*Nx that matches our shape.
        choice_key = self._val_classmap_choice.get().split(" ")[0]
        gt = np.load(self._last_classmaps[choice_key])
        # Ask the user to point at the inference.npz to score against.
        infz = filedialog.askopenfilename(
            title="Pick inference.npz for the trained model",
            initialdir=os.path.dirname(self._last_cube_path),
            filetypes=[("inference.npz", "inference.npz"),
                          ("Numpy NPZ", "*.npz"),
                          ("All", "*.*")])
        if not infz: return
        try:
            data = np.load(infz, allow_pickle=True)
            assigns = np.asarray(data["assigns"])
        except Exception as e:
            messagebox.showerror("Validate",
                f"Failed to load inference.npz:\n{e}"); return
        Ny, Nx = gt.shape
        if assigns.size != Ny * Nx:
            messagebox.showerror("Validate",
                f"Shape mismatch:  GT {gt.shape}  vs assigns "
                f"size={assigns.size}"); return
        pred = assigns.reshape(Ny, Nx)
        self._score_and_render(gt, pred, choice_key, infz=infz)

    def _score_and_render(self, gt: np.ndarray, pred: np.ndarray,
                            choice_key: str, infz: "str | None" = None):
        from scipy.optimize import linear_sum_assignment
        K_true = int(gt.max()) + 1
        K_pred = int(pred.max()) + 1
        K = max(K_true, K_pred)
        # Confusion matrix (rows=GT, cols=pred)
        C = np.zeros((K, K), dtype=np.int64)
        for g, p in zip(gt.ravel(), pred.ravel()):
            C[int(g), int(p)] += 1
        # Hungarian on -C to maximise diagonal
        row_ind, col_ind = linear_sum_assignment(-C)
        perm = np.zeros(K, dtype=int)
        for r, c in zip(row_ind, col_ind):
            perm[c] = r
        aligned = perm[pred]
        acc = float((aligned == gt).mean())

        # Per-class P/R/F1 after alignment
        per = []
        for k in range(K_true):
            tp = int(((aligned == k) & (gt == k)).sum())
            fp = int(((aligned == k) & (gt != k)).sum())
            fn = int(((aligned != k) & (gt == k)).sum())
            P = tp / max(tp + fp, 1)
            R = tp / max(tp + fn, 1)
            F = 2 * P * R / max(P + R, 1e-9)
            per.append((k, tp + fn, P, R, F))

        # ARI / NMI (permutation-invariant)
        try:
            from sklearn.metrics import (
                adjusted_rand_score as _ari,
                normalized_mutual_info_score as _nmi)
            ari = float(_ari(gt.ravel(), pred.ravel()))
            nmi = float(_nmi(gt.ravel(), pred.ravel()))
        except Exception:
            ari = nmi = float("nan")

        # K_eff from the soft probs if available (cheap fall-back: from
        # hard-assignment frequencies)
        counts = np.bincount(pred.ravel(),
                                minlength=K_pred).astype(np.float64)
        p_bar = counts / max(counts.sum(), 1)
        pb = np.clip(p_bar, 1e-12, 1.0)
        k_eff = float(np.exp(-(pb * np.log(pb)).sum()))

        # Column order that puts each Hungarian-matched pred cluster on
        # the diagonal (so a good result shows a bright diagonal).
        C2 = C[:K_true, :K_pred].copy()
        col_order, used = [], set()
        for r in range(K_true):
            for c in [c for c in range(K_pred) if perm[c] == r]:
                if c not in used:
                    col_order.append(c); used.add(c)
        for c in range(K_pred):
            if c not in used: col_order.append(c); used.add(c)
        C2 = C2[:, col_order]

        # ---- human-readable report ----
        label_full = self._val_classmap_choice.get()
        lines = ["=== Synthetic-phantom validation report ===", ""]
        lines.append(f"phantom : {self._last_cube_path}")
        if infz:
            lines.append(f"model   : {infz}")
        lines.append(f"labeling: {label_full}")
        lines.append("")
        lines.append(f"K_true = {K_true}    K_pred = {K_pred}    "
                     f"K_eff = {k_eff:.2f}")
        lines.append(f"Pixel accuracy (Hungarian-aligned) = {acc*100:.2f}%")
        lines.append(f"ARI = {ari:.4f}    NMI = {nmi:.4f}")
        lines.append("")
        lines.append("Per-class (after alignment):")
        lines.append(f"  {'gt':>3} {'N':>7} {'P':>6} {'R':>6} {'F1':>6}")
        for k, n, P, R, F in per:
            lines.append(f"  {k:>3} {n:>7} {P:>6.3f} {R:>6.3f} {F:>6.3f}")
        report_txt = "\n".join(lines)

        # ---- inline text (quick glance) ----
        self._metrics_text.config(state="normal")
        self._metrics_text.delete("1.0", "end")
        self._metrics_text.insert("end", report_txt + "\n")
        self._metrics_text.config(state="disabled")

        # ---- structured + text report files next to the phantom ----
        saved_to = None
        try:
            folder = os.path.dirname(self._last_cube_path)
            # tag by the model run so scoring several models doesn't
            # overwrite (run dir = parent of .../eval/inference.npz).
            tag = "model"
            if infz:
                parts = os.path.normpath(infz).split(os.sep)
                tag = (parts[parts.index("eval") - 1]
                       if "eval" in parts and parts.index("eval") > 0
                       else os.path.basename(os.path.dirname(infz)) or "model")
            base = os.path.join(folder, f"validation_{tag}_{choice_key}")
            report = {
                "phantom_cube": self._last_cube_path, "inference": infz,
                "labeling": choice_key, "labeling_full": label_full,
                "K_true": K_true, "K_pred": K_pred, "K_eff": k_eff,
                "pixel_accuracy": acc, "ARI": ari, "NMI": nmi,
                "per_class": [{"gt": k, "N": n, "precision": P,
                                 "recall": R, "f1": F}
                                for k, n, P, R, F in per],
                "confusion_matrix_gt_x_pred": C[:K_true, :K_pred].tolist(),
            }
            with open(base + ".txt", "w", encoding="utf-8") as f:
                f.write(report_txt + "\n")
            with open(base + ".json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            saved_to = base + ".png"
        except Exception as e:
            print(f"[synth] report save failed: {e!r}", flush=True)
            base = None

        # ---- a proper, readable figure: confusion matrix + metrics ----
        from matplotlib.figure import Figure as _Fig
        fig = _Fig(figsize=(12.5, max(5.0, 0.45 * K_true + 4.0)),
                     dpi=120, facecolor="white")
        axm = fig.add_subplot(1, 2, 1)
        im = axm.imshow(C2, cmap="viridis", aspect="auto")
        axm.set_xlabel(f"predicted cluster (Hungarian-permuted), "
                        f"K_pred={K_pred}")
        axm.set_ylabel(f"ground truth ({choice_key}), K_true={K_true}")
        axm.set_title("Confusion matrix (bright diagonal = good)",
                        fontsize=11)
        fig.colorbar(im, ax=axm, fraction=0.046, pad=0.03)
        # annotate counts when the matrix is small enough to read
        if K_true <= 12 and K_pred <= 12:
            thr = C2.max() * 0.5 if C2.max() > 0 else 1
            for r in range(C2.shape[0]):
                for c in range(C2.shape[1]):
                    v = int(C2[r, c])
                    if v:
                        axm.text(c, r, str(v), ha="center", va="center",
                                   fontsize=8,
                                   color="black" if C2[r, c] > thr else "white")
        axt = fig.add_subplot(1, 2, 2); axt.axis("off")
        axt.text(0.0, 1.0, report_txt, family="monospace", fontsize=11,
                   va="top", ha="left", transform=axt.transAxes)
        fig.suptitle(f"Validation — acc={acc*100:.1f}%  ARI={ari:.3f}  "
                      f"NMI={nmi:.3f}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if base is not None:
            try:
                fig.savefig(saved_to, dpi=120, facecolor="white",
                              bbox_inches="tight")
            except Exception as e:
                print(f"[synth] report png failed: {e!r}", flush=True)
                saved_to = None

        # ---- compact inline confusion matrix + open the big window ----
        try:
            self._ax.clear()
            self._ax.imshow(C2, cmap="viridis", aspect="auto")
            self._ax.set_title(f"acc={acc*100:.1f}%  ARI={ari:.3f}",
                                 fontsize=9)
            self._ax.set_xticks([]); self._ax.set_yticks([])
            self._fig.tight_layout(); self._canvas.draw_idle()
        except Exception:
            pass
        if base is not None:
            # Note where the report was saved (inline, no extra dialog).
            self._metrics_text.config(state="normal")
            self._metrics_text.insert(
                "end", f"\nsaved report → {os.path.basename(base)}"
                       f".png / .txt / .json\n")
            self._metrics_text.config(state="disabled")
        if saved_to and os.path.exists(saved_to):
            self._open_image_window(
                saved_to, title=f"Validation — {choice_key}")

    # ===================================================================
    # Phase 2: layout-visibility, load existing
    # ===================================================================
    def _refresh_layout_visibility(self):
        """Grey out the Voronoi-seeds entry when layout != voronoi."""
        try:
            state = ("normal" if self.layout.get() == "voronoi"
                       else "disabled")
            self._vor_seeds_entry.configure(state=state)
        except Exception:
            pass

    def _on_load_existing(self):
        """Pick a previously-simulated phantom folder, rehydrate
        ``_last_*`` state, so the Validate block can score against
        a new inference.npz without re-simulating."""
        d = filedialog.askdirectory(
            title="Pick a previous sim folder (contains "
                  "phantom.cube.npy + phantom.classmap_*.npy)",
            initialdir=os.path.join(os.getcwd(), "runs", "synth"))
        if not d: return
        cube_path = os.path.join(d, "phantom.cube.npy")
        if not os.path.exists(cube_path):
            messagebox.showerror("Load",
                f"No phantom.cube.npy in:\n{d}"); return
        cmaps = {}
        for k in ("A", "B", "crys", "main", "C"):
            p = os.path.join(d, f"phantom.classmap_{k}.npy")
            if os.path.exists(p):
                cmaps[k] = p
        if not cmaps:
            messagebox.showerror("Load",
                f"No phantom.classmap_*.npy in:\n{d}"); return
        meta_path = os.path.join(d, "phantom.sim_meta.json")
        meta = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f: meta = json.load(f)
            except Exception:
                meta = None
        self._last_cube_path = cube_path
        self._last_classmaps = cmaps
        self._last_meta = meta
        # Re-register so other tabs see it.
        try:
            cube = np.load(cube_path, mmap_mode="r")
            register_runtime_sample(
                cube_path,
                scan_shape=tuple(cube.shape[:2]),
                vmax=float((meta or {}).get("recommended_vmax", 1.0)),
                center_mask_radius=0)
        except Exception:
            pass
        self._sim_status.configure(
            text=f"loaded existing phantom: {cube_path}\n"
                  f"classmaps: {sorted(cmaps.keys())}")
        # Auto-render the preview on load too.
        try:
            self._render_phantom_preview(
                cube_path=cube_path,
                classmap_B_path=cmaps.get("B"),
                meta=meta or {},
                save_path=None)
        except Exception as e:
            print(f"[synth] preview render on load failed: {e!r}",
                  flush=True)

    # ===================================================================
    # Phase 3: multislice comparison
    # ===================================================================
    def _on_multislice_compare(self):
        """Re-run the LAST simulation with engine='multislice' on the
        EXACT same grain layout + per-grain assignments, save the
        second cube next to the first, and render per-pixel pattern
        correlation distribution.  Slow — intended as a paper-figure
        sanity check on 1–2 phantoms, not a routine validation step.
        """
        if self._sim_busy:
            messagebox.showinfo("Multislice", "Sim already running.")
            return
        if not self._last_cube_path:
            messagebox.showinfo("Multislice",
                "Need a PRISM sim first."); return
        if (self._last_meta or {}).get("engine") == "multislice":
            messagebox.showinfo("Multislice",
                "Last sim was ALREADY multislice — nothing to "
                "compare against."); return
        grain_path = os.path.join(os.path.dirname(self._last_cube_path),
                                     "phantom.grain_map.npy")
        if not os.path.exists(grain_path):
            messagebox.showerror("Multislice",
                "Missing phantom.grain_map.npy — can't replay "
                "the same grain layout."); return
        if not self._last_meta or "grain_props" not in self._last_meta:
            messagebox.showerror("Multislice",
                "Missing grain_props in sim_meta.json — can't "
                "replay assignments."); return
        # Confirm — multislice is slow.
        if not messagebox.askyesno("Multislice",
            "Multislice on the same grain layout — typically "
            "10–100× slower than PRISM.  Proceed?"):
            return
        # Prepare reuse dict
        grain_map = np.load(grain_path)
        # JSON stores grain_props with string keys.
        grain_props_raw = self._last_meta["grain_props"]
        grain_props = {}
        for k, v in grain_props_raw.items():
            v = dict(v)
            v["za"] = tuple(v["za"])
            grain_props[int(k)] = v
        reuse = {"grain_map": grain_map, "grain_props": grain_props}
        # Output dir → sibling of original
        parent = os.path.dirname(self._last_cube_path)
        outdir = parent + "_ms"
        os.makedirs(outdir, exist_ok=True)
        self._sim_busy = True
        self._sim_stop_requested = False
        self._set_running_ui(True)
        self._sim_status.configure(
            text=f"multislice sim → {outdir} …")

        def _worker():
            try:
                # Force engine = multislice but keep all other params
                cube_ms, classmaps_ms, meta_ms = self._run_simulation(
                    outdir, engine_override="multislice",
                    reuse_grain_assignments=reuse,
                    out_basename="phantom_ms")
                cube_ms_path = os.path.join(outdir,
                                              "phantom_ms.cube.npy")
                np.save(cube_ms_path, cube_ms.astype(np.float32))
                for k, m in classmaps_ms.items():
                    np.save(os.path.join(outdir,
                                             f"phantom_ms.classmap_{k}.npy"),
                              m.astype(np.int32))
                with open(os.path.join(outdir,
                                          "phantom_ms.sim_meta.json"),
                            "w") as f:
                    json.dump(meta_ms, f, indent=2,
                                default=lambda o: repr(o))
                # ---- per-pixel pattern correlation vs original ----
                cube_pr = np.load(self._last_cube_path, mmap_mode="r")
                Ny, Nx = cube_pr.shape[:2]
                corr = np.zeros((Ny, Nx), dtype=np.float32)
                for yy in range(Ny):
                    for xx in range(Nx):
                        a = cube_pr[yy, xx].ravel().astype(np.float32)
                        b = cube_ms[yy, xx].ravel().astype(np.float32)
                        a = a - a.mean(); b = b - b.mean()
                        na = float(np.linalg.norm(a))
                        nb = float(np.linalg.norm(b))
                        corr[yy, xx] = (
                            float(np.dot(a, b) / (na * nb))
                            if na * nb > 1e-9 else 0.0)
                np.save(os.path.join(outdir,
                                         "prism_vs_multislice.corr.npy"),
                          corr)
                med = float(np.median(corr))
                p05 = float(np.percentile(corr, 5))
                msg = (f"multislice done.\n"
                       f"  PRISM-vs-MS pattern correlation:\n"
                       f"    median={med:.4f}   p5={p05:.4f}\n"
                       f"  saved: {cube_ms_path}\n"
                       f"  corr map: prism_vs_multislice.corr.npy")
                self.after(0, lambda: self._sim_status.configure(
                    text=msg))
                # Render histogram of correlations in the metric pane.
                def _draw():
                    self._ax.clear()
                    self._ax.hist(corr.ravel(), bins=40,
                                     color="#2D7A2D", alpha=0.85)
                    self._ax.set_xlabel("PRISM-vs-MS pattern correlation")
                    self._ax.set_ylabel("# pixels")
                    self._ax.set_title(
                        f"PRISM vs multislice  (median={med:.3f}, "
                        f"p5={p05:.3f}, N={corr.size})", fontsize=10)
                    self._fig.tight_layout()
                    self._canvas.draw_idle()
                self.after(0, _draw)
            except _SimAborted:
                print("[synth] multislice comparison aborted by user.",
                      flush=True)
                self.after(0, lambda: self._sim_status.configure(
                    text="multislice comparison aborted by user."))
            except Exception as e:
                err = repr(e)
                self.after(0, lambda: messagebox.showerror(
                    "Multislice", err))
                self.after(0, lambda: self._sim_status.configure(
                    text=f"multislice failed: {err}"))
            finally:
                self._sim_busy = False
                self._sim_stop_requested = False
                self.after(0, lambda: self._set_running_ui(False))
        threading.Thread(target=_worker, daemon=True).start()

    # ===================================================================
    # Phantom preview — class-map (B) + per-phase structure XY/XZ/YZ
    # views.  Class-map B = "phase + zone axis", which is the labeling
    # convention where each viewing direction is its OWN class but
    # in-plane rotations of the same (phase, ZA) collapse together.
    # ===================================================================
    def _show_preview_btn(self):
        if not self._last_cube_path or not self._last_classmaps:
            messagebox.showinfo("Preview",
                "No sim loaded — Simulate or Load existing first.")
            return
        png = os.path.join(os.path.dirname(self._last_cube_path),
                             "phantom_preview.png")
        try:
            # (Re)render to a full-size PNG, then open it BIG in its own
            # scrollable window — the inline canvas is height-starved.
            self._render_phantom_preview(
                cube_path=self._last_cube_path,
                classmap_B_path=self._last_classmaps.get("B"),
                meta=self._last_meta or {},
                save_path=png)
        except Exception as e:
            messagebox.showerror("Preview", repr(e)); return
        self._open_image_window(png, title="Phantom preview")

    def _open_image_window(self, png_path, title="Image"):
        """Show a PNG at full size in a resizable window with horizontal
        and vertical scrollbars + mouse-wheel scrolling."""
        if not (png_path and os.path.exists(png_path)):
            messagebox.showinfo("Preview", "No preview image found.")
            return
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("1150x720")
        try:
            win.lift(); win.focus_force()
        except Exception:
            pass
        cv = tk.Canvas(win, background="white", highlightthickness=0)
        hbar = tk.Scrollbar(win, orient="horizontal", command=cv.xview)
        vbar = tk.Scrollbar(win, orient="vertical", command=cv.yview)
        cv.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        hbar.pack(side="bottom", fill="x")
        cv.pack(side="left", fill="both", expand=True)
        try:
            img = tk.PhotoImage(file=png_path)
        except Exception as e:
            win.destroy()
            messagebox.showerror("Preview",
                f"Could not load image:\n{e!r}"); return
        # Keep a reference on the window so it isn't garbage-collected
        # (which would blank the canvas).
        win._img_ref = img
        cv.create_image(0, 0, anchor="nw", image=img)
        cv.configure(scrollregion=(0, 0, img.width(), img.height()))
        def _wheel(e):
            cv.yview_scroll(-1 if e.delta > 0 else 1, "units")
        def _wheel_h(e):
            cv.xview_scroll(-1 if e.delta > 0 else 1, "units")
        cv.bind("<MouseWheel>", _wheel)
        cv.bind("<Shift-MouseWheel>", _wheel_h)

    def _render_phantom_preview(self, cube_path, classmap_B_path,
                                  meta, save_path=None):
        """Clean phantom quick-look: the ground-truth class map plus one
        class-average diffraction pattern per class (up to 8).  Written to
        phantom_preview.png (properly sized) and shown compactly on the
        GUI validation canvas.  Prefers the primary 'main' classmap.
        """
        import matplotlib.pyplot as _plt
        from matplotlib.colors import ListedColormap, BoundaryNorm

        # Prefer the primary in-plane-invariant map ('main'); fall back
        # to B.  Both live next to the cube.
        folder = os.path.dirname(cube_path)
        cm_path = None
        for cand in (os.path.join(folder, "phantom.classmap_main.npy"),
                       classmap_B_path,
                       os.path.join(folder, "phantom.classmap_B.npy")):
            if cand and os.path.exists(cand):
                cm_path = cand; break
        if cm_path is None:
            raise RuntimeError("no phantom.classmap_*.npy for preview")
        which = "main" if cm_path.endswith("_main.npy") else "B"
        cm = np.load(cm_path)
        Kc = int(cm.max()) + 1
        cube = np.load(cube_path, mmap_mode="r")
        Ny, Nx, Hh, Ww = cube.shape

        # Class-average diffraction for up to 8 largest classes.
        ids, counts = np.unique(cm, return_counts=True)
        cnt = dict(zip(ids.tolist(), counts.tolist()))
        show_ids = [int(k) for k in ids[np.argsort(counts)[::-1]][:8]]
        vmax = float((meta or {}).get("recommended_vmax") or 0.0)
        avgs = []
        for k in show_ids:
            ys, xs = np.where(cm == k)
            sel = (np.linspace(0, len(ys) - 1, 64).astype(int)
                     if len(ys) > 64 else np.arange(len(ys)))
            acc = np.zeros((Hh, Ww), np.float64)
            for i in sel:
                acc += np.asarray(cube[ys[i], xs[i]], dtype=np.float64)
            avgs.append((k, cnt.get(k, len(ys)), acc / max(len(sel), 1)))

        # ---- Build a properly-sized standalone figure for the PNG ----
        from matplotlib.figure import Figure as _Figure
        n_show = max(1, len(avgs))
        ncols = min(4, n_show)
        nrows_p = int(np.ceil(n_show / ncols))
        base = _plt.get_cmap("tab20" if Kc <= 20 else "viridis", max(Kc, 1))
        cmap = ListedColormap(base(np.arange(max(Kc, 1))))
        norm = BoundaryNorm(np.arange(Kc + 1) - 0.5, Kc)

        def _draw(fig, with_patterns):
            fig.clear()
            if with_patterns:
                gs = fig.add_gridspec(max(nrows_p, 1), ncols + 2,
                                        wspace=0.3, hspace=0.35)
                axm = fig.add_subplot(gs[:, :2])
            else:
                axm = fig.add_subplot(111)
            im = axm.imshow(cm, cmap=cmap, norm=norm,
                              interpolation="nearest", aspect="equal")
            axm.set_title(f"ground-truth class map ({which}) — K={Kc}",
                            fontsize=11)
            axm.set_xticks([]); axm.set_yticks([])
            fig.colorbar(im, ax=axm, fraction=0.045, pad=0.02,
                           ticks=np.arange(Kc))
            if with_patterns:
                for idx, (k, n, av) in enumerate(avgs):
                    ax = fig.add_subplot(gs[idx // ncols, 2 + idx % ncols])
                    # Per-pattern 99.5th-percentile stretch: the central
                    # (unscattered) beam is orders brighter than the Bragg
                    # disks, so scaling by av.max() would crush everything
                    # to black.  The percentile saturates the beam and
                    # reveals the disks / diffuse rings.
                    ref = (float(np.percentile(av, 99.5))
                             or float(av.max()) or 1.0)
                    disp = np.log1p(np.clip(av / ref, 0, None) * 50.0)
                    ax.imshow(disp, cmap="inferno", interpolation="nearest")
                    ax.set_title(f"class {k}  ({n}px)", fontsize=9)
                    ax.set_xticks([]); ax.set_yticks([])
                fig.suptitle(
                    f"Phantom preview — {Ny}×{Nx} scan, {Kc} classes, "
                    f"class-average diffraction", fontsize=12)

        if save_path:
            try:
                png = _Figure(figsize=(4 + 3 * ncols, max(4.0, 3.2 * nrows_p)),
                                dpi=130, facecolor="white")
                _draw(png, with_patterns=True)
                png.savefig(save_path, dpi=130, facecolor="white",
                              bbox_inches="tight")
                print(f"[synth] preview PNG → {save_path}", flush=True)
            except Exception as e:
                print(f"[synth] preview save failed: {e!r}", flush=True)

        # Compact class map on the GUI validation canvas.
        try:
            _draw(self._fig, with_patterns=False)
            self._fig.tight_layout()
            self._canvas.draw_idle()
        except Exception:
            pass

    def _open_last_outdir(self):
        if not self._last_cube_path:
            messagebox.showinfo("folder",
                "No sim run yet."); return
        d = os.path.dirname(self._last_cube_path)
        try:
            if sys.platform == "win32":
                os.startfile(d)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            messagebox.showerror("folder", repr(e))


# =========================================================================
# 4. Atomic-model helpers
# =========================================================================
def _is_cell_orthogonal(cell, tol: float = 1e-6) -> bool:
    """Strict orthogonality check — abTEM's internal check uses
    near-zero tolerance, so any residual rotation in off-diagonals
    (e.g. 1e-4 from a manual rotation matrix) will still trigger its
    "not orthogonal" exception even though it looks "fine" by eye.
    """
    import numpy as _np
    cell = _np.asarray(cell, dtype=float)
    off_diag = (abs(cell[0, 1]) + abs(cell[0, 2]) +
                 abs(cell[1, 0]) + abs(cell[1, 2]) +
                 abs(cell[2, 0]) + abs(cell[2, 1]))
    if off_diag >= tol:
        return False
    # Also verify positive diagonal — abTEM doesn't like flipped axes.
    if (cell[0, 0] <= 0 or cell[1, 1] <= 0 or cell[2, 2] <= 0):
        return False
    return True


def _orthogonalize(atoms, max_repetitions: int = 5):
    """Wrap abTEM's orthogonalize_cell, tolerating API moves across
    versions.  Returns atoms with a strictly orthogonal cell, or
    raises if no orthogonal supercell exists within max_repetitions.

    Reports the MOST INFORMATIVE error — i.e. an error from an actual
    call to the function takes precedence over a trivial "function
    not at this import path" miss.  The previous version reported
    the import-miss when the real failure was the algorithmic one,
    which made debugging hard.
    """
    import_errors: list[tuple[str, Exception]] = []
    call_error: "Exception | None" = None
    for path in ("abtem.structures", "abtem.potentials",
                  "abtem.atoms", "abtem.dft",
                  "abtem.core.atoms", "abtem"):
        try:
            mod = __import__(path, fromlist=["orthogonalize_cell"])
        except Exception as e:
            import_errors.append((path, e))
            continue
        fn = getattr(mod, "orthogonalize_cell", None)
        if fn is None:
            import_errors.append(
                (path, AttributeError("not present")))
            continue
        try:
            return fn(atoms, max_repetitions=max_repetitions)
        except Exception as e:
            # Real algorithmic failure — preserve it; do NOT continue
            # to less-likely paths just to mask this with a trivial
            # AttributeError.
            call_error = e
            break
    if call_error is not None:
        raise call_error
    miss_summary = ", ".join(f"{p}({type(e).__name__})"
                                for p, e in import_errors)
    raise RuntimeError(
        f"orthogonalize_cell not found in any known abtem submodule: "
        f"{miss_summary}")


def _resolve_abtem(name: str):
    """Find `abtem.<name>` (or its equivalent under one of the
    version-dependent sub-modules) as a real class/function — NOT a
    submodule.  Different abTEM releases re-shuffle where Potential /
    SMatrix / Probe / PixelatedDetector / scan live; this helper
    papers over that.  Raises if no callable can be located.
    """
    import importlib, inspect
    candidates = [
        "abtem",
        "abtem.potentials",          "abtem.potentials.potential",
        "abtem.atomic_models",
        "abtem.waves",               "abtem.waves.waves",
        "abtem.scan",                "abtem.scans",
        "abtem.detectors",
        "abtem.prism",               "abtem.prism.s_matrix",
        "abtem.core.potentials",     "abtem.core.waves",
    ]
    last_err: "Exception | None" = None
    for path in candidates:
        try:
            mod = importlib.import_module(path)
            obj = getattr(mod, name, None)
            if obj is None:
                continue
            # Reject submodules — they're callable-looking but raise
            # the 'module is not callable' error at use.
            if inspect.ismodule(obj):
                continue
            if inspect.isclass(obj) or inspect.isfunction(obj) \
                    or callable(obj):
                return obj
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"abtem.{name} not found as a class/function in any of: "
        f"{', '.join(candidates)}.  Last import error: {last_err!r}.\n"
        f"abtem version layout has shifted; tell the maintainer "
        f"which abtem version you have (python -m pip show abtem).")


def _cell_diag(atoms, tag: str = "") -> str:
    """Compact one-line cell description for stdout logging."""
    import numpy as _np
    cell = _np.asarray(atoms.get_cell(), dtype=float)
    diag = (f"  {tag}: cell=\n  {cell[0]}\n  {cell[1]}\n  {cell[2]}\n"
            f"  off-diag sum = {abs(cell[0,1])+abs(cell[0,2])+abs(cell[1,0])+abs(cell[1,2])+abs(cell[2,0])+abs(cell[2,1]):.6g}\n"
            f"  n_atoms = {len(atoms)}")
    return diag


def _orient_to_zone_axis(atoms, zone_axis):
    """Return an atoms object with an ORTHOGONAL cell whose c-axis
    points along the requested zone axis (i.e. the (hkl) plane is
    perpendicular to the beam ‖ +z).

    abTEM hard-requires orthorhombic cells (off-diagonal cell terms
    must be zero), so we MUST end with one no matter the input.

    Strategy — try in order, return the first that yields an
    orthogonal cell:
      1. ``ase.build.surface(atoms, idx, layers, vacuum=0)`` and
         keep if orthogonal.
      2. Same, then ``orthogonalize_cell(..., max_repetitions=5)``
         on the slab.
      3. Manual rotation: rotate cell so [hkl] ‖ z, then keep if
         orthogonal.
      4. Same, then ``orthogonalize_cell(..., max_repetitions=10)``.
      5. ``orthogonalize_cell`` directly on the original atoms
         (works for trivial ZAs along a cell axis, e.g. cubic [001]
         on a `cubic=True` cell).

    All steps catch their own exceptions; the final ``raise`` reports
    the last error so the caller knows whether it's a `surface()`
    failure, an orthogonalize_cell-not-found failure, or a "no
    orthogonal supercell within max_repetitions" failure.
    """
    import numpy as _np
    from ase.build import surface
    try:
        idx = tuple(int(x) for x in zone_axis)
    except Exception:
        raise RuntimeError(f"bad zone axis: {zone_axis!r}")

    last_err: "Exception | None" = None

    # ---- Path 1+2: surface(), with orthogonalize fallback ----------
    try:
        slab = surface(atoms, indices=idx, layers=8,
                          vacuum=0.0, periodic=True)
        if _is_cell_orthogonal(slab.get_cell()):
            return slab
        try:
            ortho = _orthogonalize(slab, max_repetitions=5)
            if _is_cell_orthogonal(ortho.get_cell()):
                return ortho
        except Exception as e:
            last_err = e
    except Exception as e:
        last_err = e

    # ---- Path 3+4: manual rotation, with orthogonalize fallback ----
    try:
        h, k, l = (float(x) for x in idx)
        g = _np.array([h, k, l], dtype=float)
        if _np.linalg.norm(g) >= 1e-9:
            g = g / _np.linalg.norm(g)
            z = _np.array([0.0, 0.0, 1.0])
            v = _np.cross(g, z)
            s_ = _np.linalg.norm(v)
            c_ = float(_np.dot(g, z))
            if s_ < 1e-9:
                R = (_np.eye(3) if c_ > 0
                       else _np.diag([1.0, -1.0, -1.0]))
            else:
                vx = _np.array([[0, -v[2], v[1]],
                                  [v[2], 0, -v[0]],
                                  [-v[1], v[0], 0]])
                R = _np.eye(3) + vx + vx @ vx * ((1 - c_) / (s_ ** 2))
            rotated = atoms.copy()
            rotated.set_cell(rotated.get_cell() @ R.T, scale_atoms=True)
            if _is_cell_orthogonal(rotated.get_cell()):
                return rotated
            try:
                ortho = _orthogonalize(rotated, max_repetitions=10)
                if _is_cell_orthogonal(ortho.get_cell()):
                    return ortho
            except Exception as e:
                last_err = e
    except Exception as e:
        last_err = e

    # ---- Path 5: orthogonalize the ORIGINAL atoms directly ----------
    # Catches the common "cubic=True + ZA=(0,0,1)" case where the cell
    # is already orthogonal and no rotation is needed.
    try:
        ortho = _orthogonalize(atoms, max_repetitions=10)
        if _is_cell_orthogonal(ortho.get_cell()):
            return ortho
    except Exception as e:
        last_err = e

    raise RuntimeError(
        f"could not produce an orthogonal cell for zone axis {idx}. "
        f"Last error: {last_err!r}.\n\n"
        f"Most common cause: an ASE builder was called without "
        f"`cubic=True` (or `orthorhombic=True`), so it returned the "
        f"PRIMITIVE cell, which is non-orthogonal for cubic / "
        f"hexagonal / etc. crystals.  Add `cubic=True` (or "
        f"`orthorhombic=True`) to the builder kwargs in the structure "
        f"editor.\n\nFor CIFs: try a lower-index zone axis, or pre-"
        f"orient in VESTA/Avogadro before loading."
    )


def _apply_off_axis_tilt(atoms, tilt_deg: float, axis_deg: float):
    """Phase 3: tilt the oriented crystal by `tilt_deg` around an
    in-plane axis at azimuth `axis_deg`.  Rotates atoms relative to
    the lab frame so the beam (along +z) now hits the crystal off
    its nominal zone axis.

    Cell becomes non-orthogonal after the rotation → re-orthogonalize
    via abTEM's ``orthogonalize_cell`` (max_repetitions=5).  If the
    requested tilt has no rational orthogonal supercell within that
    bound the call raises; user should reduce tilt_range_deg or pre-
    tile their structure.
    """
    import numpy as _np
    if abs(tilt_deg) < 1e-6:
        return atoms
    th = _np.deg2rad(tilt_deg)
    ax = _np.deg2rad(axis_deg)
    u = _np.array([_np.cos(ax), _np.sin(ax), 0.0])
    c, s = _np.cos(th), _np.sin(th)
    ux, uy, uz = u
    R = _np.array([
        [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)],
    ])
    out = atoms.copy()
    out.set_cell(out.get_cell() @ R.T, scale_atoms=True)
    if _is_cell_orthogonal(out.get_cell()):
        return out
    return _orthogonalize(out, max_repetitions=5)


def _make_block_grain_map(Ny: int, Nx: int, n_phases: int) -> np.ndarray:
    """Block layout: divide the (Ny, Nx) scan into N tiles in an
    approx-square grid; each tile is one grain whose id == phase id.
    """
    if n_phases <= 1:
        return np.zeros((Ny, Nx), dtype=np.int32)
    nrows = int(round(n_phases ** 0.5))
    ncols = int(np.ceil(n_phases / max(1, nrows)))
    block_ny = Ny // max(1, nrows)
    block_nx = Nx // max(1, ncols)
    out = np.zeros((Ny, Nx), dtype=np.int32)
    for yy in range(Ny):
        for xx in range(Nx):
            bi = min(yy // max(1, block_ny), nrows - 1)
            bj = min(xx // max(1, block_nx), ncols - 1)
            out[yy, xx] = min(bi * ncols + bj, n_phases - 1)
    return out


def _make_voronoi_grain_map(Ny: int, Nx: int, n_seeds: int,
                                rng) -> np.ndarray:
    """Voronoi layout: scatter `n_seeds` uniformly in the scan rect
    and assign each pixel to the nearest seed (Euclidean).  Returns a
    (Ny, Nx) int32 grain_id map with values in [0, n_seeds)."""
    from scipy.spatial import cKDTree
    seeds = rng.uniform(low=[0.0, 0.0], high=[Nx, Ny],
                           size=(int(n_seeds), 2))
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    _, idx = cKDTree(seeds).query(pts, k=1)
    return idx.reshape(Ny, Nx).astype(np.int32)


def _build_atoms_vacuum_box(atoms_base, *, zone_axis, tile_xyz,
                                tilt_mag_mrad=0.0, tilt_az_deg=0.0,
                                vacuum_pad_A=20.0,
                                thickness_crop_A=0.0):
    """NaPHI-style structure builder.

    1.  ``atoms = atoms_base * tile_xyz``  (in the ORIGINAL crystal frame).
    2.  ``atoms.cell = [0,0,0]``           (discard the cell — atoms only).
    3.  Rotate atoms so the requested zone axis ``[hkl]`` (in the original
        frame) points along +z (beam direction).
    4.  Apply the off-axis beam-tilt as an ADDITIONAL atomic rotation:
        ``tilt_mag_mrad`` around an in-plane axis at azimuth
        ``tilt_az_deg``.
    5.  ``atoms.center(vacuum=vacuum_pad_A)`` — ASE rebuilds an orthogonal
        axis-aligned bounding box around the rotated atoms and pads
        vacuum on all sides.  The result is always orthogonal regardless
        of the rotation, so abTEM's ``Potential`` is happy and we
        NEVER call ``orthogonalize_cell``.
    6.  (Optional) Z-crop to a specific thickness in Å.

    Returns the prepared Atoms.  Pass to ``abtem.Potential(..., periodic
    =True)``.
    """
    import numpy as _np
    atoms = atoms_base.copy()
    nx, ny, nz = (max(int(x), 1) for x in tile_xyz)
    atoms = atoms * (nx, ny, nz)
    cell0 = _np.asarray(atoms.get_cell(), dtype=float).copy()
    # ---- Step 1: zone-axis alignment ---------------------------------
    h, k, l = (float(x) for x in zone_axis)
    # Real-space crystallographic direction [hkl] in cartesian.
    target = h * cell0[0] + k * cell0[1] + l * cell0[2]
    nrm = float(_np.linalg.norm(target))
    if nrm > 1e-9:
        target = target / nrm
        z_hat = _np.array([0.0, 0.0, 1.0])
        cosang = float(_np.dot(target, z_hat))
        if cosang < -1.0 + 1e-9:
            # 180° flip
            atoms.cell = [0, 0, 0]
            atoms.rotate(v=[1.0, 0.0, 0.0], a=180.0, center="COM")
        elif cosang < 1.0 - 1e-9:
            axis = _np.cross(target, z_hat)
            axis = axis / (_np.linalg.norm(axis) + 1e-12)
            angle = float(_np.degrees(_np.arccos(cosang)))
            atoms.cell = [0, 0, 0]
            atoms.rotate(v=axis.tolist(), a=angle, center="COM")
        else:
            atoms.cell = [0, 0, 0]
    else:
        atoms.cell = [0, 0, 0]

    # ---- Step 2: off-axis beam tilt as atomic rotation ---------------
    if tilt_mag_mrad and float(tilt_mag_mrad) > 0:
        tilt_deg = float(tilt_mag_mrad) / 1000.0 * (180.0 / _np.pi)
        az = float(_np.radians(tilt_az_deg))
        axis = [float(_np.cos(az)), float(_np.sin(az)), 0.0]
        atoms.rotate(v=axis, a=tilt_deg, center="COM")

    # ---- Step 3: rebuild orthogonal bounding box with vacuum ---------
    atoms.center(vacuum=float(vacuum_pad_A))

    # ---- Step 4: optional Z crop after orientation -------------------
    if thickness_crop_A and float(thickness_crop_A) > 0:
        pos = atoms.get_positions()
        z = pos[:, 2]
        z0 = 0.5 * (float(z.min()) + float(z.max()))
        keep = ((z >= z0 - float(thickness_crop_A) / 2) &
                 (z <= z0 + float(thickness_crop_A) / 2))
        atoms = atoms[keep]
        if len(atoms) > 0:
            atoms.center(vacuum=float(vacuum_pad_A))

    # ---- Step 5: SQUARE the in-plane box -----------------------------
    # A non-square (Lx != Ly) cell makes abTEM sample reciprocal space
    # anisotropically with square gpts → ELLIPTICAL diffraction rings,
    # and it breaks in-plane-rotation invariance (the ellipse axes stay
    # fixed while the crystal rotates).  Pad the shorter in-plane axis
    # with vacuum so Lx == Ly → isotropic sampling → circular rings.
    cell = _np.array(atoms.get_cell(), dtype=float)
    Lx, Ly = float(cell[0, 0]), float(cell[1, 1])
    if Lx > 0 and Ly > 0 and abs(Lx - Ly) > 1e-6:
        L = max(Lx, Ly)
        pos = atoms.get_positions()
        pos[:, 0] += (L - Lx) / 2.0
        pos[:, 1] += (L - Ly) / 2.0
        atoms.set_positions(pos)
        cell[0, 0] = L
        cell[1, 1] = L
        atoms.set_cell(cell)
    return atoms


def _autotile_inplane_to_square(atoms, max_tile: int = 8,
                                    aspect_tol: float = 1.15):
    """Tile the slab in the in-plane (a, b) directions so the two
    extents are within `aspect_tol` of each other.  Pure supercell
    operation — no physics change, just sampling fix.

    Without this, `surface()` on hexagonal / off-axis cubic CIFs
    often returns a 1:√3 or 1:3 cell, which makes abTEM's reciprocal
    sampling anisotropic and the displayed diffraction look like a
    vertical/horizontal lens (the kx vs ky angular range you reach
    at max_angle=80 mrad is wildly different on the two axes).

    Returns a new Atoms with tile_xy chosen to minimise aspect ratio
    while keeping each tile factor ≤ max_tile.  Falls back to the
    original atoms if the cell is already near-square.
    """
    import numpy as _np
    cell = _np.asarray(atoms.get_cell(), dtype=float)
    a = float(_np.linalg.norm(cell[0]))
    b = float(_np.linalg.norm(cell[1]))
    if a <= 0 or b <= 0:
        return atoms
    cur_ratio = max(a, b) / min(a, b)
    if cur_ratio < aspect_tol:
        return atoms
    # Try every (nx, ny) ≤ max_tile and pick the one that minimises
    # max(nx*a, ny*b) / min(nx*a, ny*b).
    best = (1, 1, cur_ratio)
    for nx in range(1, max_tile + 1):
        for ny in range(1, max_tile + 1):
            ax = nx * a; by = ny * b
            r = max(ax, by) / min(ax, by)
            if r < best[2]:
                best = (nx, ny, r)
    nx, ny, r = best
    if (nx, ny) == (1, 1):
        return atoms
    print(f"[synth] auto-tile in-plane: original a={a:.2f}Å "
          f"b={b:.2f}Å (ratio={cur_ratio:.2f}) → "
          f"tile ({nx},{ny},1) → "
          f"new a={nx*a:.2f}Å b={ny*b:.2f}Å (ratio={r:.2f})",
          flush=True)
    return atoms * (nx, ny, 1)


def _crop_thickness(atoms, thickness_A: float):
    """Trim atoms whose z is outside [0, thickness_A] (after centring
    around the cell's z midplane).  Adjusts cell c-vector and forces
    the cell to remain orthogonal — abTEM requires an orthorhombic
    cell for the multislice/PRISM integration."""
    import numpy as _np
    if thickness_A <= 0:
        return atoms
    atoms = atoms.copy()
    pos = atoms.get_positions()
    if pos.shape[0] == 0:
        return atoms
    z_min, z_max = pos[:, 2].min(), pos[:, 2].max()
    z0 = 0.5 * (z_min + z_max)
    keep = (pos[:, 2] >= z0 - thickness_A / 2) & (
            pos[:, 2] <= z0 + thickness_A / 2)
    atoms = atoms[keep]
    pos = atoms.get_positions()
    if pos.shape[0] > 0:
        pos[:, 2] -= pos[:, 2].min()
    atoms.set_positions(pos)
    cell = _np.asarray(atoms.get_cell(), dtype=float).copy()
    # Force strict orthogonality (zero out off-diagonals) — Path 1 of
    # _orient_to_zone_axis already gives us orthogonal, this is a
    # paranoid safety net for the manual-rotation fallback path.
    a = float(cell[0, 0]) or float(_np.linalg.norm(cell[0]))
    b = float(cell[1, 1]) or float(_np.linalg.norm(cell[1]))
    atoms.set_cell(_np.diag([a, b, float(thickness_A)]))
    return atoms
