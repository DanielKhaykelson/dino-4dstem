"""session.py -- single global session state for the GUI.

All panels that need (sample, run_dir, inference) read from one
`Session` instance held on `app.session`.  Changes broadcast to
subscribers — no more push/pull between panels.

Replaces the per-panel `outdir / sample / _inf` proliferation +
the ad-hoc `posthoc.refresh_from_acom2()` style fan-out.

Usage in a panel:
    app.session.subscribe(self._on_session_change)
    s = app.session
    if s.sample and s.inference is not None:
        ...

To change session state from anywhere:
    app.session.set(sample="IMC_SI5", run_dir="...")
"""
from __future__ import annotations
import os
from typing import Callable


class Session:
    def __init__(self):
        self.sample: "str | None" = None
        self.run_dir: "str | None" = None
        self.inference: "dict | None" = None
        self._subs: list[Callable] = []

    # --- pub/sub ---
    def subscribe(self, fn: Callable):
        if fn not in self._subs:
            self._subs.append(fn)

    def unsubscribe(self, fn: Callable):
        try: self._subs.remove(fn)
        except ValueError: pass

    def _emit(self):
        for fn in list(self._subs):
            try: fn(self)
            except Exception as e:
                print(f"[session] subscriber {fn} raised: {e!r}",
                      flush=True)

    # --- setters ---
    def set(self, *, sample=None, run_dir=None, inference=...,
              emit: bool = True):
        changed = False
        if sample is not None and sample != self.sample:
            self.sample = sample; changed = True
        if run_dir is not None and run_dir != self.run_dir:
            self.run_dir = run_dir; changed = True
            # auto-reset inference on new run unless explicitly set
            if inference is ...:
                self.inference = None
        if inference is not ...:
            self.inference = inference; changed = True
        if changed and emit:
            self._emit()

    # --- handy queries ---
    def has_dataset(self) -> bool:
        return bool(self.sample)

    def has_run(self) -> bool:
        return bool(self.run_dir and os.path.isdir(self.run_dir))

    def has_inference(self) -> bool:
        return self.inference is not None

    # ------------------------------------------------------------------
    # High-level loader: resolves sample + cube path from any of the
    # config dumps the project produces.  Replaces the per-panel
    # ad-hoc loaders.  Returns the resolved sample key (or None).
    # ------------------------------------------------------------------
    def load_run_dir(self, run_dir: str) -> "str | None":
        import json
        from data import SAMPLES, register_runtime_sample
        if not os.path.isdir(run_dir):
            print(f"[session] not a dir: {run_dir}", flush=True)
            return None
        sample_inferred = None
        # 1) run_summary.json (sweep + GUI runs)
        rs = os.path.join(run_dir, "run_summary.json")
        if os.path.exists(rs):
            try:
                js = json.load(open(rs, encoding="utf-8"))
                sample_inferred = (js.get("sample")
                                    or js.get("cfg", {}).get("sample")
                                    or js.get("cfg", {}).get("target_sample"))
            except Exception: pass
        # 2) walk up for SAMPLE_LOCK.json (sweep convention)
        lock = self._find_lock(run_dir)
        if sample_inferred and lock:
            try:
                spec = json.load(open(lock, encoding="utf-8"))
                cube_p = spec.get("cube_path")
                if cube_p and os.path.exists(cube_p):
                    vmax = float(spec.get("vmax", 2.0))
                    pmc = int(spec.get("polar_mask_cols", 0))
                    register_runtime_sample(
                        cube_p, vmax=vmax,
                        center_mask_radius=pmc // 2,
                        key=sample_inferred)
            except Exception as e:
                print(f"[session] SAMPLE_LOCK failed: {e!r}",
                      flush=True)
        # 3) _train_kwargs.json (GUI-side fallback)
        if sample_inferred and sample_inferred not in SAMPLES:
            tk = os.path.join(run_dir, "_train_kwargs.json")
            if os.path.exists(tk):
                try:
                    kw = json.load(open(tk, encoding="utf-8"))
                    cfg = kw.get("_sample_config") or {}
                    if cfg.get("path"):
                        ss = cfg.get("scan_shape")
                        register_runtime_sample(
                            cfg["path"],
                            scan_shape=(tuple(ss) if ss else None),
                            vmax=float(cfg.get("vmax", 2.0)),
                            center_mask_radius=int(
                                cfg.get("center_mask_radius", 15)),
                            key=sample_inferred)
                except Exception: pass
        # Commit to session.
        self.set(run_dir=run_dir,
                  sample=(sample_inferred
                            if sample_inferred in SAMPLES else None))
        if sample_inferred and sample_inferred not in SAMPLES:
            print(f"[session] '{sample_inferred}' could not be "
                    f"registered — no SAMPLE_LOCK / _train_kwargs / "
                    f"static SAMPLES entry resolved a cube path.",
                    flush=True)
        return self.sample

    @staticmethod
    def _find_lock(start_dir, max_walk=5):
        cur = os.path.abspath(start_dir)
        for _ in range(int(max_walk)):
            p = os.path.join(cur, "SAMPLE_LOCK.json")
            if os.path.exists(p): return p
            parent = os.path.dirname(cur)
            if parent == cur: break
            cur = parent
        return None

    def __repr__(self):
        return (f"Session(sample={self.sample!r}, "
                f"run={'...' + (self.run_dir or '')[-30:]}, "
                f"inf={'✓' if self.has_inference() else '✗'})")
