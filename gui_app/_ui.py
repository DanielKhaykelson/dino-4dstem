"""_ui.py -- shared UI primitives for consistent look + feel.

  - status_dot()         tiny ⚪/⏳/✅/❌ widget
  - session_badge()      button-style badge with colored dot
  - palette              single source of color truth
  - btn()                consistent button factory (4 sizes only)

All panels import these and stop reinventing them.
"""
from __future__ import annotations
import customtkinter as ctk


COLOR = {
    "neutral":   ("#7f7f7f", "#999"),
    "primary":   ("#4D6FB0", "#3A5380"),    # main interactive
    "run":       ("#2D7A2D", "#1F7A1F"),    # start a long job
    "stop":      ("#a33",    "#722"),       # halt / destructive
    "multi":     ("#A23BB0", "#7A2680"),    # multi-phase actions
    "warn":      ("#c97c20", "#a35e0a"),
    "ok":        "#2D7A2D",
    "err":       "#c0392b",
    "pending":   "#888",
}

DOT = {"idle": "○", "busy": "◔", "ok": "✓", "err": "✗"}

SIZES = {"icon": 32, "small": 90, "med": 160, "full": 240}


def btn(parent, text, command=None, *, size="med", kind="neutral",
         **kw):
    """Factory for consistent CTkButtons.

    kind ∈ {neutral, primary, run, stop, multi, warn}
    size ∈ {icon, small, med, full}
    """
    width = kw.pop("width", SIZES.get(size, 160))
    fg = COLOR.get(kind, COLOR["neutral"])
    return ctk.CTkButton(parent, text=text, command=command,
                            width=width, fg_color=fg, **kw)


class StatusDot(ctk.CTkLabel):
    """Small ⚪/⏳/✅/❌ dot with hover text.

    Usage:
        d = StatusDot(parent, "dataset")
        d.set("ok",   "IMC_SI5 ready")
        d.set("err",  "no sample loaded")
        d.set("busy", "computing dp_max…")
        d.set("idle", "not loaded")
    """
    def __init__(self, parent, label: str = ""):
        self._label = label
        self._state = "idle"
        self._tooltip = ""
        super().__init__(parent, text=self._render(),
                          width=22, anchor="w",
                          font=("Segoe UI Symbol", 14))
        self.bind("<Enter>", self._show_tip)
        self.bind("<Leave>", self._hide_tip)

    def _render(self):
        glyph = DOT.get(self._state, "○")
        return f"{glyph} {self._label}".strip()

    def _color(self):
        return {"ok":   COLOR["ok"],
                  "err":  COLOR["err"],
                  "busy": COLOR["warn"][0],
                  "idle": COLOR["pending"]}.get(self._state,
                                                  COLOR["pending"])

    def set(self, state: str, tooltip: str = ""):
        self._state = state
        self._tooltip = tooltip
        try:
            self.configure(text=self._render(),
                              text_color=self._color())
        except Exception:
            pass

    def _show_tip(self, _e=None):
        if not self._tooltip: return
        try:
            import tkinter as tk
            self._tip = tk.Toplevel(self)
            self._tip.wm_overrideredirect(True)
            x = self.winfo_rootx() + 22
            y = self.winfo_rooty() + 4
            self._tip.geometry(f"+{x}+{y}")
            tk.Label(self._tip, text=self._tooltip,
                      bg="#222", fg="white", font=("Segoe UI", 9),
                      padx=6, pady=3).pack()
        except Exception:
            pass

    def _hide_tip(self, _e=None):
        try: self._tip.destroy()
        except Exception: pass


def session_badge(parent, label: str, command=None):
    """Pill-shaped clickable badge for the topbar showing session info.

    Returns the CTkButton.  Update its text via `.configure(text=...)`.
    """
    return ctk.CTkButton(parent, text=label, command=command,
                            width=260, height=28,
                            corner_radius=14,
                            fg_color=("#dde6f4", "#2c3a4f"),
                            text_color=("#1a3a66", "#cfe0fa"),
                            hover_color=("#cad6e7", "#3a4d68"),
                            font=("Segoe UI", 10))
