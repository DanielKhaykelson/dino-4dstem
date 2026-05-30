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
        self._caption = label
        self._state = "idle"
        self._tooltip = ""
        super().__init__(parent, text=self._render(),
                          width=22, anchor="w",
                          font=("Segoe UI Symbol", 14))
        self.bind("<Enter>", self._show_tip)
        self.bind("<Leave>", self._hide_tip)

    def _render(self):
        glyph = DOT.get(self._state, "○")
        return f"{glyph} {self._caption}".strip()

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


def attach_hover_q(canvas, ax, *, center, q_per_disp_px,
                       units="nm⁻¹"):
    """Wire a follow-the-cursor q-readout to a diffraction-image axes.

    canvas         : FigureCanvasTkAgg the axes belongs to
    ax             : matplotlib Axes hosting the imshow
    center         : (cy, cx) image-center in *display* pixels of `ax`
    q_per_disp_px  : reciprocal-space units per display pixel of `ax`.
                       This MUST be computed for the actual view
                       (raw-detector / cart-crop+resize / polar bin /
                       …) — use gui_app._calib_utils helpers.
    units          : just the label on the readout ("nm⁻¹", "1/Å").

    Returns the mpl cid so the caller can disconnect.
    """
    import math
    state = {"annot": None}
    cy, cx = float(center[0]), float(center[1])
    def _hover(event):
        if event.inaxes is not ax or event.xdata is None:
            if state["annot"] is not None:
                state["annot"].set_visible(False)
                canvas.draw_idle()
            return
        dx = event.xdata - cx; dy = event.ydata - cy
        r_disp = math.sqrt(dx * dx + dy * dy)
        q = r_disp * q_per_disp_px
        d_real = (1.0 / q) if q > 1e-12 else float("inf")
        # d-spacing in nm when q is in nm⁻¹, in Å when q is in 1/Å
        d_unit = "nm" if "nm" in units else "Å"
        txt = (f"r = {r_disp:.1f} px\n"
                 f"q = {q:.4f} {units}\n"
                 f"d = {d_real:.3f} {d_unit}")
        if state["annot"] is None:
            state["annot"] = ax.annotate(
                txt, xy=(event.xdata, event.ydata),
                xytext=(14, 14), textcoords="offset points",
                fontsize=8, color="#111",
                bbox=dict(boxstyle="round,pad=0.3",
                            fc="#fff7c2", ec="#666", lw=1.0,
                            alpha=0.92),
                annotation_clip=False)
            state["annot"].set_zorder(20)
        else:
            state["annot"].xy = (event.xdata, event.ydata)
            state["annot"].set_text(txt)
            state["annot"].set_visible(True)
        canvas.draw_idle()
    return canvas.mpl_connect("motion_notify_event", _hover)


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
