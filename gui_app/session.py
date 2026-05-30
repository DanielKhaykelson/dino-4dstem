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

    def __repr__(self):
        return (f"Session(sample={self.sample!r}, "
                f"run={'...' + (self.run_dir or '')[-30:]}, "
                f"inf={'✓' if self.has_inference() else '✗'})")
