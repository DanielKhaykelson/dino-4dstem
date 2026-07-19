"""assistant_gui.py -- standalone Assistant window (no full GUI, no console).

Opens just the chat assistant in its own window, driving the panel-free
(headless) toolset: load by path, NMF, infer, class average, interpretation,
ACOM, score, plus advice/troubleshoot/assess and inline figures. Reuses
gui_app.chat_panel.ChatPanel, so it inherits the hardware-aware model pick
(14b when there's room, else 7b/3b), the Gemini option, markup cleanup, and
loop guards.

Launched windowless via `pythonw src/assistant_gui.py` (see launch_assistant.bat).
"""
from __future__ import annotations
import os
import sys
import traceback

# When run as `python src/assistant_gui.py`, src/ is already sys.path[0].
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _fatal(msg: str):
    """Show a fatal error in a dialog + log, since there's no console."""
    try:
        with open(os.path.join(os.path.expanduser("~"),
                               "dino4dstem_assistant_error.log"),
                  "w", encoding="utf-8") as f:
            f.write(msg)
    except Exception:
        pass
    try:
        import tkinter as tk
        from tkinter import messagebox
        r = tk.Tk(); r.withdraw()
        messagebox.showerror("DINO-4DSTEM Assistant — failed to start", msg)
        r.destroy()
    except Exception:
        pass


def main():
    try:
        import customtkinter as ctk
        from assistant_headless import _HeadlessApp, build_headless_registry
        from gui_app.chat_panel import ChatPanel
    except Exception:
        _fatal("Could not import the assistant:\n\n" + traceback.format_exc())
        return

    try:
        ctk.set_appearance_mode("system")
        root = ctk.CTk()
        root.title("DINO-4DSTEM Assistant")
        root.geometry("920x700")
        try:
            from gui_dino4dstem import set_app_icon
            set_app_icon(root)
        except Exception:
            pass

        app = _HeadlessApp()                      # session holder, no panels
        panel = ChatPanel(root, app=app,
                          registry=build_headless_registry())
        panel.pack(fill="both", expand=True)

        # Optional: a cube path passed on the command line -> load on start.
        if len(sys.argv) > 1 and sys.argv[1].strip():
            panel.after(600, lambda: panel._send_prompt(
                f"load {sys.argv[1].strip()}"))

        root.mainloop()
    except Exception:
        _fatal("The assistant crashed:\n\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
