"""pair_labeler.py -- modal dialog for labelling pattern pairs.

Two flavours, both share the same UI; only how pairs are *drawn*
differs:
  * mode='random'        — uniform random pairs from the cube.
                            Used PRE-training (no model required).
  * mode='active'        — pairs surfaced from the trained model's
                            inference (Phase B). The constructor
                            accepts a `pair_proposer` callable that
                            returns (a, b, source) tuples.

Layout:
+--------------------------------------------------+
| Pair labelling — <sample>                  [X]   |
|                                                  |
| Are these patterns the SAME phase or DIFFERENT?  |
|                                                  |
| +---------------+      +---------------+         |
| |               |      |               |         |
| |   pattern A   |      |   pattern B   |         |
| | idx 1234      |      | idx 5678      |         |
| | (y=12, x=34)  |      | (y=56, x=78)  |         |
| +---------------+      +---------------+         |
|                                                  |
| [SAME (s)] [DIFFERENT (d)] [skip (k)] [next (n)] |
|                                                  |
| Labelled this session: 12 same · 8 different     |
| Total on disk:         47 pairs                  |
|                                                  |
|                              [save & close]      |
+--------------------------------------------------+

Saves to the cube's sidecar JSON via `pair_labels.save_pair_labels`
on close (and on every label commit, so accidental window-close
doesn't lose work).

Keyboard shortcuts: s=same, d=diff, k=skip, n=next-without-saving,
Escape = save and close.
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import customtkinter as ctk
import tkinter as tk

import matplotlib
matplotlib.use("TkAgg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui_app.pair_labels import (
    load_pair_labels, save_pair_labels, append_pair, label_count,
    sample_random_pair, already_labelled,
    pop_last_pair, set_pair_label_at, delete_pair_at,
    LABEL_SAME, LABEL_DIFF,
)


class PairLabelerWindow(ctk.CTkToplevel):
    """Modal-ish (non-blocking) Toplevel for labelling pattern pairs.

    Parameters
    ----------
    master : Tk widget
    cube_path : str
        Path to the cube file; used to derive the sidecar path.
    sample_key : str
        Display name (e.g. 'Na007b' or 'loaded__IMC_…').
    cube : numpy array, shape (Ny, Nx, H, W)
        Mmap-friendly handle. Used to fetch raw patterns lazily.
    vmax : float
        For normalised display.
    cmap : str
    mode : 'random' | 'active' | 'review'
        - 'random'  : draw uniform random pairs from the cube.
        - 'active'  : use `pair_proposer` to surface candidate pairs.
        - 'review'  : iterate through pairs ALREADY in the sidecar so
                      the user can flip / delete them. Buttons swap to
                      'change to SAME / change to DIFFERENT / delete /
                      keep & next / back / save & close'.
    pair_proposer : callable() -> (idx_a, idx_b, source) | None
        Required if mode='active'. Returning None ends the queue.
    on_close : callable(label_count_dict) | None
        Called once when the window closes, with the latest count
        on disk. Useful for the parent panel to refresh its display.
    """

    def __init__(self, master, *, cube_path, sample_key, cube, vmax,
                  cmap="inferno", mode="random",
                  pair_proposer=None, on_close=None):
        super().__init__(master)
        self.title(f"Pair labelling — {sample_key}  ({mode})")
        self.geometry("980x720")

        self._cube_path = cube_path
        self._sample_key = sample_key
        self._cube = cube
        Ny, Nx, H, W = cube.shape
        self._scan_shape = (Ny, Nx)
        self._N_total = Ny * Nx
        self._vmax = float(vmax)
        self._cmap = cmap
        self._mode = mode
        self._pair_proposer = pair_proposer
        self._on_close = on_close

        # Load existing labels (if any)
        self._labels = load_pair_labels(cube_path)
        self._labels.setdefault("scan_shape", list(self._scan_shape))

        # Session counters (this window only)
        self._session_same = 0
        self._session_diff = 0
        self._session_skip = 0
        self._session_changed = 0   # review mode: flips
        self._session_deleted = 0   # review mode: deletes

        # Random generator for pair sampling
        self._rng = np.random.default_rng()

        # Current proposed pair
        self._cur_a = None
        self._cur_b = None
        self._cur_source = mode  # default for random; active overrides

        # Review-mode cursor (only used when mode == 'review')
        self._review_cursor = 0

        self._build()
        self._next_pair()

        # Keyboard shortcuts
        self.bind("<Key-s>", lambda e: self._commit(LABEL_SAME))
        self.bind("<Key-d>", lambda e: self._commit(LABEL_DIFF))
        self.bind("<Key-k>", lambda e: self._skip())
        self.bind("<Key-n>", lambda e: self._next_pair())
        self.bind("<Key-u>", lambda e: self._undo_last())  # all modes
        # Review-only:
        self.bind("<Key-x>",       lambda e: self._delete_current())
        self.bind("<Key-Left>",    lambda e: self._review_back())
        self.bind("<Key-Right>",   lambda e: self._review_keep_and_next())
        self.bind("<Escape>", lambda e: self._save_and_close())
        self.protocol("WM_DELETE_WINDOW", self._save_and_close)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build(self):
        # Header
        hdr = ctk.CTkFrame(self)
        hdr.pack(side="top", fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(hdr,
            text="Are these patterns the SAME phase or DIFFERENT?",
            font=("Segoe UI", 13, "bold")
            ).pack(side="left", padx=8, pady=4)

        # Two-pane figure
        fig_holder = ctk.CTkFrame(self)
        fig_holder.pack(side="top", fill="both", expand=True,
                          padx=8, pady=4)
        self._fig = Figure(figsize=(8.5, 4.5), dpi=100,
                              facecolor="#f4f4f4")
        self._ax_a = self._fig.add_subplot(121)
        self._ax_b = self._fig.add_subplot(122)
        for ax in (self._ax_a, self._ax_b):
            ax.set_xticks([]); ax.set_yticks([])
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_holder)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # Banner above the buttons (mostly for review-mode current label).
        self._banner = ctk.CTkLabel(self, text="", font=("Segoe UI", 11),
                                       justify="left")
        self._banner.pack(side="top", fill="x", padx=12, pady=(0, 4))

        # Buttons row — mode-dependent.
        btns = ctk.CTkFrame(self, fg_color="transparent")
        btns.pack(side="top", fill="x", padx=12, pady=(2, 8))
        if self._mode == "review":
            ctk.CTkButton(btns, text="Change to SAME  (s)",
                width=160, height=36,
                fg_color=("#2D7A2D", "#1F7A1F"),
                font=("Segoe UI", 12, "bold"),
                command=lambda: self._commit(LABEL_SAME)
                ).pack(side="left", padx=4)
            ctk.CTkButton(btns, text="Change to DIFFERENT  (d)",
                width=190, height=36,
                fg_color=("#A04030", "#7A1010"),
                font=("Segoe UI", 12, "bold"),
                command=lambda: self._commit(LABEL_DIFF)
                ).pack(side="left", padx=4)
            ctk.CTkButton(btns, text="Delete  (x)",
                width=100, height=36,
                fg_color=("#666", "#444"),
                command=self._delete_current
                ).pack(side="left", padx=4)
            ctk.CTkButton(btns, text="◀ back  (←)", width=100, height=36,
                command=self._review_back).pack(side="left", padx=4)
            ctk.CTkButton(btns, text="keep & next  (→)", width=140, height=36,
                command=self._review_keep_and_next
                ).pack(side="left", padx=4)
        else:
            ctk.CTkButton(btns, text="SAME  (s)", width=140, height=36,
                fg_color=("#2D7A2D", "#1F7A1F"),
                font=("Segoe UI", 12, "bold"),
                command=lambda: self._commit(LABEL_SAME)
                ).pack(side="left", padx=4)
            ctk.CTkButton(btns, text="DIFFERENT  (d)", width=160, height=36,
                fg_color=("#A04030", "#7A1010"),
                font=("Segoe UI", 12, "bold"),
                command=lambda: self._commit(LABEL_DIFF)
                ).pack(side="left", padx=4)
            ctk.CTkButton(btns, text="skip  (k)", width=100, height=36,
                command=self._skip).pack(side="left", padx=4)
            ctk.CTkButton(btns, text="next pair  (n)", width=120, height=36,
                command=self._next_pair).pack(side="left", padx=4)
            ctk.CTkButton(btns, text="↶ undo last  (u)",
                width=140, height=36,
                fg_color=("#888", "#555"),
                command=self._undo_last
                ).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="save & close  (esc)", width=160, height=36,
                       command=self._save_and_close
                       ).pack(side="right", padx=4)

        # Counters
        self._stats = ctk.CTkLabel(self, text="", font=("Consolas", 10),
                                      justify="left")
        self._stats.pack(side="top", fill="x", padx=12, pady=(0, 8))

        self._update_stats()

    def _update_stats(self):
        on_disk = label_count(self._labels)
        bys = on_disk.get("by_source", {})
        bys_str = ", ".join(f"{k}={v}" for k, v in sorted(bys.items())) \
                    or "—"
        if self._mode == "review":
            session_line = (
                f"this session :  {self._session_changed} re-labelled   "
                f"{self._session_deleted} deleted")
        else:
            session_line = (
                f"this session :  {self._session_same} same   "
                f"{self._session_diff} different   "
                f"{self._session_skip} skipped")
        self._stats.configure(text=(
            f"{session_line}\n"
            f"on disk total:  {on_disk['total']} pairs   "
            f"({on_disk['same']} same / {on_disk['diff']} diff)   "
            f"by source: [{bys_str}]"
        ))

    # ------------------------------------------------------------------
    # Pair sampling + display
    # ------------------------------------------------------------------
    def _next_pair(self):
        if self._mode == "review":
            self._review_show()
            return
        # Random / active modes: get a fresh proposal.
        max_tries = 50
        for _ in range(max_tries):
            if self._mode == "random":
                a, b = sample_random_pair(self._N_total, self._rng)
                src = "random"
            else:
                if self._pair_proposer is None:
                    self._set_pair_unavailable(
                        "active mode requires a pair_proposer")
                    return
                prop = self._pair_proposer()
                if prop is None:
                    self._set_pair_unavailable(
                        "no more candidate pairs from the proposer.")
                    return
                a, b, src = prop
            if not already_labelled(self._labels, a, b):
                self._cur_a = int(a); self._cur_b = int(b)
                self._cur_source = str(src)
                self._banner.configure(text="")
                self._draw_pair()
                return
        # All tries exhausted
        self._set_pair_unavailable(
            "couldn't find an unlabelled pair after 50 tries.")

    # ---- review mode ----
    def _review_show(self):
        """Show the labelled pair at self._review_cursor and update
        the banner with its current label."""
        pairs = self._labels.get("pairs", [])
        if not pairs:
            self._set_pair_unavailable(
                "no labelled pairs to review yet.")
            self._banner.configure(text="")
            return
        if self._review_cursor < 0:
            self._review_cursor = 0
        if self._review_cursor >= len(pairs):
            self._set_pair_unavailable(
                f"reviewed all {len(pairs)} pairs. "
                f"close to commit, or click ◀ back to go again.")
            self._banner.configure(
                text=f"end of list  ({len(pairs)} pairs)",
                text_color=("#444", "#aaa"))
            return
        p = pairs[self._review_cursor]
        self._cur_a = int(p["a"]); self._cur_b = int(p["b"])
        self._cur_source = str(p.get("source", "?"))
        cur_y = int(p.get("y", 0))
        cur_t = str(p.get("t", ""))
        # Banner:  "Reviewing 5 / 47   currently SAME (random, ts)"
        y_str = "SAME" if cur_y == LABEL_SAME else (
            "DIFFERENT" if cur_y == LABEL_DIFF else f"y={cur_y}")
        y_color = (("#2D7A2D", "#7AC07A")
                    if cur_y == LABEL_SAME else
                    ("#A04030", "#E07060"))
        self._banner.configure(
            text=(f"Reviewing  {self._review_cursor + 1} / "
                  f"{len(pairs)}     currently labelled "
                  f"{y_str}     (source: {self._cur_source}, {cur_t})"),
            text_color=y_color)
        self._draw_pair()

    def _review_back(self):
        if self._mode != "review":
            return
        self._review_cursor = max(0, self._review_cursor - 1)
        self._review_show()

    def _review_keep_and_next(self):
        if self._mode != "review":
            return
        self._review_cursor += 1
        self._review_show()

    def _delete_current(self):
        if self._mode != "review":
            return
        if self._cur_a is None:
            return
        deleted = delete_pair_at(self._labels, self._review_cursor)
        if deleted is not None:
            self._session_deleted += 1
            try: save_pair_labels(self._cube_path, self._labels)
            except Exception: pass
            self._update_stats()
            # cursor stays on the same index → now points at the next
            # pair (the list shifted down). If we were at end, clamp.
            self._review_show()

    def _draw_pair(self):
        Ny, Nx = self._scan_shape
        ay, ax_ = divmod(self._cur_a, Nx)
        by, bx_ = divmod(self._cur_b, Nx)
        try:
            pat_a = np.asarray(self._cube[ay, ax_]).astype(np.float32)
            pat_b = np.asarray(self._cube[by, bx_]).astype(np.float32)
        except Exception as e:
            self._set_pair_unavailable(f"failed to read pattern: {e}")
            return
        # log-stretched normalised display
        def _disp(p):
            n = np.clip(p / max(self._vmax, 1e-6), 0.0, 1.0)
            return np.log1p(n * 50.0)
        self._ax_a.clear(); self._ax_b.clear()
        self._ax_a.imshow(_disp(pat_a), cmap=self._cmap,
                            aspect="equal", interpolation="nearest")
        self._ax_b.imshow(_disp(pat_b), cmap=self._cmap,
                            aspect="equal", interpolation="nearest")
        self._ax_a.set_xticks([]); self._ax_a.set_yticks([])
        self._ax_b.set_xticks([]); self._ax_b.set_yticks([])
        self._ax_a.set_title(
            f"A — idx {self._cur_a}  (y={ay}, x={ax_})", fontsize=10)
        self._ax_b.set_title(
            f"B — idx {self._cur_b}  (y={by}, x={bx_})\n"
            f"source: {self._cur_source}", fontsize=10)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _set_pair_unavailable(self, msg):
        self._cur_a = None; self._cur_b = None
        self._ax_a.clear(); self._ax_b.clear()
        for ax in (self._ax_a, self._ax_b):
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                     transform=ax.transAxes, color="#888",
                     wrap=True, fontsize=10)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Commit / skip / save
    # ------------------------------------------------------------------
    def _commit(self, y):
        if self._cur_a is None or self._cur_b is None:
            return
        if self._mode == "review":
            # Re-label the existing pair at the current cursor.
            pairs = self._labels.get("pairs", [])
            if not (0 <= self._review_cursor < len(pairs)):
                return
            old_y = int(pairs[self._review_cursor].get("y", 0))
            if int(y) == old_y:
                # No change → just advance.
                self._review_cursor += 1
                self._review_show()
                return
            ok = set_pair_label_at(self._labels,
                                       self._review_cursor, int(y))
            if ok:
                self._session_changed += 1
            try: save_pair_labels(self._cube_path, self._labels)
            except Exception: pass
            self._update_stats()
            self._review_cursor += 1
            self._review_show()
            return
        # random / active modes: append.
        append_pair(self._labels, self._cur_a, self._cur_b, int(y),
                     source=self._cur_source)
        if y == LABEL_SAME: self._session_same += 1
        else:               self._session_diff += 1
        try:
            save_pair_labels(self._cube_path, self._labels)
        except Exception:
            pass
        self._update_stats()
        self._next_pair()

    def _undo_last(self):
        """Pop the most recently appended pair and re-show it as the
        current proposal so the user can re-label it."""
        if self._mode == "review":
            # In review mode 'undo' doesn't make sense — we already
            # have explicit Back/Delete/Change buttons.
            self._banner.configure(
                text="(undo isn't used in review mode — "
                      "use ◀ back / Delete / Change to SAME|DIFFERENT)",
                text_color=("#a00", "#f99"))
            return
        popped = pop_last_pair(self._labels)
        if popped is None:
            self._banner.configure(
                text="(nothing to undo — no labels yet)",
                text_color=("#a00", "#f99"))
            return
        # Decrement the right session counter
        old_y = int(popped.get("y", 0))
        if old_y == LABEL_SAME:
            self._session_same = max(0, self._session_same - 1)
        elif old_y == LABEL_DIFF:
            self._session_diff = max(0, self._session_diff - 1)
        try:
            save_pair_labels(self._cube_path, self._labels)
        except Exception:
            pass
        self._update_stats()
        # Re-display the popped pair so the user can re-label it.
        self._cur_a = int(popped["a"]); self._cur_b = int(popped["b"])
        self._cur_source = str(popped.get("source", self._mode))
        old_str = ("SAME" if old_y == LABEL_SAME else
                    "DIFFERENT" if old_y == LABEL_DIFF else f"y={old_y}")
        self._banner.configure(
            text=f"undid last label (was {old_str}). "
                  f"re-label or skip.",
            text_color=("#444", "#aaa"))
        self._draw_pair()

    def _skip(self):
        if self._cur_a is None or self._cur_b is None:
            return
        self._session_skip += 1
        self._update_stats()
        self._next_pair()

    def _save_and_close(self):
        try:
            save_pair_labels(self._cube_path, self._labels)
        except Exception as e:
            try:
                from tkinter import messagebox
                messagebox.showerror("save failed",
                    f"Could not write labels:\n{e}")
            except Exception:
                pass
        finally:
            try:
                if callable(self._on_close):
                    self._on_close(label_count(self._labels))
            except Exception:
                pass
            try: self.destroy()
            except Exception: pass
