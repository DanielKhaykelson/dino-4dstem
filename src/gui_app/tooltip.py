"""tooltip.py -- lightweight hover tooltip for any tkinter widget. Used
to add a small "?" icon next to a parameter that explains what it does.
"""
from __future__ import annotations
import tkinter as tk


class ToolTip:
    """Floating tooltip on hover. Pure tkinter (no extra deps)."""

    def __init__(self, widget: tk.Widget, text: str,
                 delay_ms: int = 350, wraplength: int = 320):
        self.widget = widget
        self.text = text
        self.delay = delay_ms
        self.wraplength = wraplength
        self._after_id: "str | None" = None
        self._tip: "tk.Toplevel | None" = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event=None):
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, _event=None):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None

    def _show(self):
        if self._tip is not None:
            return
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass
        lbl = tk.Label(tw, text=self.text, justify="left",
                        background="#FFFFE0", relief="solid", borderwidth=1,
                        wraplength=self.wraplength,
                        font=("Segoe UI", 9))
        lbl.pack(ipadx=6, ipady=4)
        # Measure the tooltip and clamp/flip so it never goes off-screen.
        tw.update_idletasks()
        tip_w = tw.winfo_reqwidth()
        tip_h = tw.winfo_reqheight()
        screen_w = tw.winfo_screenwidth()
        screen_h = tw.winfo_screenheight()
        wx = self.widget.winfo_rootx()
        wy = self.widget.winfo_rooty()
        wh = self.widget.winfo_height()
        ww = self.widget.winfo_width()
        margin = 8
        # Default: below + right of widget.
        x = wx + 18
        y = wy + wh + 4
        # If overflowing right, anchor to the right edge of widget instead.
        if x + tip_w + margin > screen_w:
            x = max(margin, wx + ww - tip_w)
        # If overflowing left, push to margin.
        if x < margin:
            x = margin
        # If overflowing bottom, flip above the widget.
        if y + tip_h + margin > screen_h:
            y = max(margin, wy - tip_h - 4)
        tw.wm_geometry(f"+{x}+{y}")
        self._tip = tw


def add_help_button(parent, help_text: str, **kwargs):
    """Place a small clickable '?' label that opens a tooltip on hover.

    Returns the label widget so the caller can grid/pack it.
    """
    import customtkinter as ctk
    btn = ctk.CTkLabel(parent, text="?", width=18, height=18,
                        fg_color=("#dddddd", "#444444"),
                        text_color=("#333333", "#dddddd"),
                        corner_radius=9,
                        font=ctk.CTkFont(size=11, weight="bold"))
    ToolTip(btn, help_text, **kwargs)
    return btn
