"""chat_panel.py -- in-GUI natural-language assistant for DINO-4DSTEM.

A microscopist can drive the whole pipeline in plain English:
    "load IMC SI4, set vmax 5, train, show the class map, run the
     interpretation battery, compare to NMF and ACOM, what do the
     classes mean, what parameters should I use for this new material?"

Architecture (built incrementally — see SPEC PART F):
    ChatPanel (this file)  : transcript + entry + send + backend/model
                             dropdown + cancel + status line.  Holds a
                             reference to `app` so tools can drive the
                             GUI and call the programmatic APIs.
    [Step 2] LLMBackend    : pluggable chat backend (Ollama default,
                             optional cloud).  Streaming + tool calls.
    [Step 3] tool registry : ~16 tools wrapping data.py / contrastive_eval
                             / interpret_core / panels, each gated by a
                             confirm dialog.

This file is STEP 1: the UI skeleton only.  `_send` currently echoes a
placeholder so the layout is testable; the agent loop drops in next.

Tk is single-threaded: the agent loop will run in a background
threading.Thread and marshal every widget update back via
`self.after(0, ...)`.  Nothing here spawns processes, so no
multiprocessing guard is needed in this module.
"""
from __future__ import annotations
import queue
import threading
import customtkinter as ctk

from gui_app._ui import btn, COLOR, StatusDot
from gui_app.chat_backends import OllamaBackend, BackendError
from gui_app import chat_tools

# Safety cap on tool-call rounds per user turn, so a confused model
# can't loop forever calling tools.
MAX_TOOL_ROUNDS = 8


# Backend / model / device choices surfaced in the dropdowns.  The
# actual backends are wired in Step 2; these are the labels the user
# picks from.
BACKENDS = ["Ollama (local, free)", "Cloud (bring your own key)"]
OLLAMA_MODELS = ["qwen2.5:7b", "qwen2.5:3b", "qwen2.5:14b"]
DEVICES = ["GPU", "CPU"]

# Default per the confirmed scope: 7b on GPU, with an automatic 3b
# fallback (handled in the backend in Step 2) if the 7b won't load for
# lack of resources.
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_DEVICE = "GPU"

_LATEX_SYMBOLS = {
    r"\approx": "≈", r"\times": "×", r"\cdot": "·", r"\leq": "≤", r"\le": "≤",
    r"\geq": "≥", r"\ge": "≥", r"\neq": "≠", r"\pm": "±", r"\rightarrow": "→",
    r"\to": "→", r"\Rightarrow": "⇒", r"\alpha": "α", r"\beta": "β",
    r"\gamma": "γ", r"\theta": "θ", r"\sigma": "σ", r"\mu": "μ", r"\lambda": "λ",
    r"\eta": "η", r"\Delta": "Δ", r"\sum": "Σ", r"\sqrt": "√", r"\infty": "∞",
    r"\partial": "∂", r"\nabla": "∇", r"\approxeq": "≈",
}


def _clean_markup(s: str) -> str:
    """Render LLM LaTeX/markdown to readable plain text for the Tk transcript
    (which doesn't render either).

    IMPORTANT: LaTeX symbol conversion happens ONLY inside math delimiters
    (\\(..\\), \\[..\\], $..$), so Windows paths like D:\\tools\\out — which
    contain backslash sequences like \\to — are never corrupted.
    """
    if not s:
        return s
    import re

    def _conv(m):
        inner = m.group(1)
        for k, v in _LATEX_SYMBOLS.items():
            inner = inner.replace(k, v)
        inner = re.sub(
            r"\\(?:text|mathrm|mathbf|mathit|operatorname|boldsymbol)\{([^{}]*)\}",
            r"\1", inner)
        inner = re.sub(r"[_^]\{([^{}]*)\}", r"\1", inner)
        inner = re.sub(r"\\([A-Za-z]+)", r"\1", inner)  # safe: math span only
        return inner.strip()

    s = re.sub(r"\\\[(.+?)\\\]", _conv, s, flags=re.S)
    s = re.sub(r"\\\((.+?)\\\)", _conv, s, flags=re.S)
    s = re.sub(r"\$\$(.+?)\$\$", _conv, s, flags=re.S)
    s = re.sub(r"\$([^$\n]+?)\$", _conv, s, flags=re.S)
    # drop any leftover (unmatched) math delimiters — safe, not in paths
    for d in (r"\(", r"\)", r"\[", r"\]"):
        s = s.replace(d, "")
    # markdown: **bold**, `code`, ### headers -> plain (none occur in paths)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)
    return s


class ChatPanel(ctk.CTkFrame):
    """Natural-language assistant tab.

    Built as ``ChatPanel(parent_frame, app=self)`` like every other
    panel, so tools can reach shared state via ``self.app.session`` and
    call sibling panels.
    """

    def __init__(self, parent, app=None):
        super().__init__(parent)
        self.app = app

        # Conversation state (the message list handed to the backend).
        # Kept here so the worker thread can read/append it.  Holds the
        # full role/content/tool_calls history minus the system prompt
        # (which is rebuilt fresh each turn from live app state).
        self.messages: list[dict] = []
        self._busy = False          # an agent turn is in flight
        self._cancel = False        # cancel flag the loop polls
        self._assistant_open = False  # streaming an assistant bubble?
        self._assistant_buf = ""      # raw streamed text (cleaned on close)
        self._assistant_start = None  # transcript index where the bubble began

        # Tool registry: {name: ToolSpec}.  Built once; reads live state
        # through `app` at call time.
        self.registry = chat_tools.build_registry(app)
        # Tools the user has chosen to allow for the rest of the session
        # (confirm dialog "allow for session" checkbox).  Default: empty
        # -> every tool is gated, per the confirmed policy.
        self._auto_allow: set[str] = set()
        self._confirm_win = None
        # When True, tools run without a per-call confirm dialog (the user
        # asked "just do everything, don't ask me to approve").  Read by
        # the worker thread, so keep it a plain bool (not a Tk var).
        self._autorun = False

        # Context object handed to every tool fn (worker thread).  Widget
        # access funnels through call_ui / post / status so tools never
        # touch Tk directly.
        # Keep PhotoImage refs alive (Tk drops un-referenced images).
        self._img_refs: list = []
        self._tool_ctx = chat_tools.ToolContext(
            app=app,
            call_ui=self._call_ui,
            post=self._post,
            status=lambda t: self._after(self.set_status, t),
            cancel=lambda: self._cancel,
            post_image=lambda p, cap=None: self._after(self._post_image, p, cap))

        # Cross-thread UI marshaling.  Tkinter is NOT thread-safe and
        # even `widget.after(...)` raises if called off the Tk thread, so
        # the worker pushes (fn, args) onto this queue and a periodic
        # poller (started on the Tk thread) drains it.  See _after /
        # _poll_ui_queue.
        self._ui_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._greet()
        self.after(100, self._poll_ui_queue)
        # Kick off a lightweight "loading" readiness check on open.
        self.after(250, lambda: threading.Thread(
            target=self._startup_probe, daemon=True).start())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)   # transcript expands
        self.grid_columnconfigure(0, weight=1)

        self._build_topbar()
        self._build_transcript()
        self._build_statusline()
        self._build_inputbar()
        self._build_suggestions()

    def _build_topbar(self):
        bar = ctk.CTkFrame(self)
        bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))

        ctk.CTkLabel(bar, text="Assistant",
                     font=("Segoe UI", 14, "bold")).pack(
                         side="left", padx=(8, 14), pady=6)

        ctk.CTkLabel(bar, text="backend:").pack(side="left", padx=(4, 2))
        self.var_backend = ctk.StringVar(value=BACKENDS[0])
        self.dd_backend = ctk.CTkOptionMenu(
            bar, values=BACKENDS, variable=self.var_backend,
            width=190, command=self._on_backend_change)
        self.dd_backend.pack(side="left", padx=(0, 12))

        ctk.CTkLabel(bar, text="model:").pack(side="left", padx=(4, 2))
        self.var_model = ctk.StringVar(value=DEFAULT_MODEL)
        self.dd_model = ctk.CTkOptionMenu(
            bar, values=OLLAMA_MODELS, variable=self.var_model,
            width=130)
        self.dd_model.pack(side="left", padx=(0, 12))

        ctk.CTkLabel(bar, text="device:").pack(side="left", padx=(4, 2))
        self.var_device = ctk.StringVar(value=DEFAULT_DEVICE)
        self.dd_device = ctk.CTkOptionMenu(
            bar, values=DEVICES, variable=self.var_device,
            width=80, command=self._on_device_change)
        self.dd_device.pack(side="left", padx=(0, 12))

        # Load data by PATH (browse).  Loading is always path-based — no
        # named-sample list.
        self.btn_load = btn(bar, "📂 Load data…",
                            command=self._on_browse_load,
                            size="med", kind="primary")
        self.btn_load.pack(side="left", padx=(0, 12))

        # Auto-run: skip the per-call confirm dialog.
        self.var_autorun = ctk.BooleanVar(value=False)
        self.chk_autorun = ctk.CTkCheckBox(
            bar, text="Auto-run (no confirm)", variable=self.var_autorun,
            command=self._on_autorun_toggle, onvalue=True, offvalue=False)
        self.chk_autorun.pack(side="left", padx=(0, 12))

        # Connection / readiness indicator (lights up in Step 2 once we
        # can actually ping Ollama).
        self.dot_conn = StatusDot(bar, label="backend")
        self.dot_conn.set("idle", "not connected yet (Step 2)")
        self.dot_conn.pack(side="left", padx=(4, 0))

        # Feedback on the last answer → learned KB (👎 asks what was wrong).
        ctk.CTkButton(bar, text="👎", width=34,
                      command=self._on_feedback_down).pack(side="right", padx=2)
        ctk.CTkButton(bar, text="👍", width=34,
                      command=self._on_feedback_up).pack(side="right", padx=(8, 2))

    def _build_transcript(self):
        # CTkTextbox wraps a tkinter Text widget — gives us easy
        # append-only token streaming plus per-role color tags.  We keep
        # it disabled (read-only) except while programmatically writing.
        self.transcript = ctk.CTkTextbox(
            self, wrap="word", font=("Segoe UI", 12),
            activate_scrollbars=True)
        self.transcript.grid(row=1, column=0, sticky="nsew",
                             padx=6, pady=2)
        self.transcript.configure(state="disabled")

        # Color tags for the three roles + tool traces.  CTkTextbox
        # exposes the underlying Text as ._textbox.
        tb = self.transcript._textbox
        tb.tag_config("user", foreground="#4D6FB0",
                      font=("Segoe UI", 12, "bold"))
        tb.tag_config("assistant", foreground=("#1f1f1f"))
        tb.tag_config("assistant_name", foreground="#2D7A2D",
                      font=("Segoe UI", 12, "bold"))
        tb.tag_config("tool", foreground="#A23BB0",
                      font=("Consolas", 10))
        tb.tag_config("system", foreground="#888",
                      font=("Segoe UI", 11, "italic"))

    def _build_statusline(self):
        line = ctk.CTkFrame(self, fg_color="transparent")
        line.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 2))
        self.status = ctk.CTkLabel(
            line, text="", anchor="w", text_color=COLOR["warn"][0],
            font=("Segoe UI", 10))
        self.status.pack(side="left")
        # Persistent contention warning (confirmed scope item).
        self.warn = ctk.CTkLabel(
            line,
            text=("⚠ chatting on GPU while training can OOM/crash — "
                  "switch device to CPU (or pick qwen2.5:3b) if a run "
                  "is active"),
            anchor="e", text_color=COLOR["warn"][0],
            font=("Segoe UI", 9))
        self.warn.pack(side="right")

    def _build_inputbar(self):
        bar = ctk.CTkFrame(self)
        bar.grid(row=3, column=0, sticky="ew", padx=6, pady=(2, 6))
        bar.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(
            bar, placeholder_text=(
                "Ask in plain English — e.g. \"load IMC SI4, set vmax "
                "5, show the class map\""))
        self.entry.grid(row=0, column=0, sticky="ew", padx=(6, 6),
                        pady=6)
        self.entry.bind("<Return>", lambda _e: self._on_send())

        self.btn_send = btn(bar, "Send", command=self._on_send,
                            size="small", kind="primary")
        self.btn_send.grid(row=0, column=1, padx=(0, 4), pady=6)

        self.btn_cancel = btn(bar, "■ Stop", command=self._on_cancel,
                              size="small", kind="stop")
        self.btn_cancel.grid(row=0, column=2, padx=(0, 6), pady=6)
        self.btn_cancel.configure(state="disabled")

    def _build_suggestions(self):
        """A row of one-click example prompts (helps discovery)."""
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 6))
        for label in ("What can you do?", "What should I do next?",
                      "Assess this run", "Fix overclustering"):
            ctk.CTkButton(
                row, text=label, height=24,
                fg_color=("#E5E5E5", "#3A3A3A"),
                text_color=("#222", "#DDD"), hover_color=("#D5D5D5", "#4A4A4A"),
                command=lambda t=label: self._send_prompt(t)
            ).pack(side="left", padx=3)

    def _send_prompt(self, text):
        if self._busy:
            return
        self.entry.delete(0, "end")
        self.entry.insert(0, text)
        self._on_send()

    # ---- feedback (thumbs) → learned KB ----
    def _on_feedback_up(self):
        self.set_status("👍 thanks — glad that helped.")

    def _on_feedback_down(self):
        from tkinter import simpledialog
        last = next((m.get("content", "") for m in reversed(self.messages)
                     if m.get("role") == "assistant"), "")
        what = simpledialog.askstring(
            "Correct me",
            "What was wrong, or what should I do instead?", parent=self)
        if not what:
            return
        try:
            import gui_app.chat_kb as kb
            note = "User correction: " + what
            if last:
                note += f"  (re: {last[:160]})"
            kb.add_note("correction", note)
            self.add_message(
                "system", "Thanks — saved that correction; I'll apply it.")
        except Exception as e:
            self.add_message("system", f"(couldn't save feedback: {e})")

    # ------------------------------------------------------------------
    # Transcript helpers (always called on the Tk thread; the worker
    # marshals through self.after).
    # ------------------------------------------------------------------
    def _write(self, text, tag=None):
        self.transcript.configure(state="normal")
        if tag:
            self.transcript._textbox.insert("end", text, tag)
        else:
            self.transcript._textbox.insert("end", text)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def add_message(self, role, text):
        """Append a complete message bubble for role ∈ {user, assistant,
        system, tool}."""
        if role == "user":
            self._write("\nYou\n", "user")
            self._write(text + "\n", "user")
        elif role == "assistant":
            self._write("\nAssistant\n", "assistant_name")
            self._write(_clean_markup(text) + "\n", "assistant")
        elif role == "tool":
            self._write("  • " + text + "\n", "tool")
        else:
            self._write("\n" + text + "\n", "system")

    def set_status(self, text):
        """One-line status under the transcript (e.g. 'calling tool: …').
        Safe to call from the worker via self.after."""
        try:
            self.status.configure(text=text)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Greeting / current-state summary
    # ------------------------------------------------------------------
    def _greet(self):
        s = getattr(self.app, "session", None)
        sample = getattr(s, "sample", None) or "(none)"
        run = getattr(s, "run_dir", None) or "(none)"
        self.add_message(
            "system",
            "👋 Hello — I'm your DINO-4DSTEM assistant.\n"
            "I can load data, tune pre-processing, train, run NMF/ACOM, "
            "interpret the classes, and even show you where to click.\n"
            f"Loaded data: {sample}   |   current run: {run}\n"
            "Load a cube with the 📂 Load data button (or tell me a file "
            "path), then just ask in plain English.\n"
            "New here? Ask \"what can you do?\" (or click a suggestion below).")

    # ------------------------------------------------------------------
    # Startup readiness probe (the "loading" state on open).  Lightweight:
    # checks whether the local model is reachable WITHOUT triggering any
    # download — that happens on the first message.
    # ------------------------------------------------------------------
    def _startup_probe(self):
        from gui_app.chat_backends import OllamaBackend
        self._after(self.dot_conn.set, "busy", "checking local model…")
        self._after(self.set_status, "starting up…")
        try:
            up = OllamaBackend.server_up()
        except Exception:
            up = False
        if up:
            try:
                models = OllamaBackend.list_models()
            except Exception:
                models = []
            if any(str(m).startswith("qwen2.5") for m in models):
                self._after(self.dot_conn.set, "ok", "local model ready")
                self._after(self.add_message, "system",
                            "🟢 Local model ready — ask me anything.")
            else:
                self._after(self.dot_conn.set, "idle", "no model pulled")
                self._after(self.add_message, "system",
                            "Ollama is running but no model is downloaded "
                            "yet — I'll fetch one on your first message.")
        elif OllamaBackend.find_binary():
            self._after(self.dot_conn.set, "idle", "Ollama not running")
            self._after(self.add_message, "system",
                        "Ollama is installed — I'll start it automatically "
                        "on your first message.")
        else:
            self._after(self.dot_conn.set, "idle", "Ollama not installed")
            self._after(self.add_message, "system",
                        "Ollama isn't installed yet — send a message and "
                        "I'll download + set it up for you.")
        # Hardware-aware model pick (14b if there's room, else downgrade + note).
        try:
            model, note = self._auto_pick_model()
            if model and note:
                self._after(self.var_model.set, model)
                if up:
                    try:
                        have = any(str(m).startswith(model.split(":")[0]) and
                                   model in str(m) for m in models)
                    except Exception:
                        have = False
                    if not have:
                        sz = {"qwen2.5:14b": "~9 GB", "qwen2.5:7b": "~4.7 GB",
                              "qwen2.5:3b": "~2 GB"}.get(model, "")
                        note += f" Will download {sz} on first message."
                self._after(self.add_message, "system", note)
        except Exception:
            pass
        self._after(self.set_status, "")

    def _auto_pick_model(self):
        """Return (model, note): measure free VRAM (GPU) or RAM (CPU) and pick
        the largest qwen2.5 that should fit. Conservative — downgrades when
        unsure rather than risking an OOM."""
        import shutil, subprocess
        try:
            device = self.var_device.get()
        except Exception:
            device = "GPU"
        free_gb, where = None, ""
        if device == "GPU" and shutil.which("nvidia-smi"):
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=6)
                free_gb = float(r.stdout.strip().splitlines()[0]) / 1024.0
                where = "GPU VRAM"
            except Exception:
                free_gb = None
        if free_gb is None:
            try:
                import psutil
                free_gb = psutil.virtual_memory().available / 1e9
                where = "system RAM"
            except Exception:
                return DEFAULT_MODEL, ""
        if where == "GPU VRAM":
            model = ("qwen2.5:14b" if free_gb >= 11 else
                     "qwen2.5:7b" if free_gb >= 6 else "qwen2.5:3b")
        else:  # CPU: 14b is painfully slow — only with lots of RAM
            model = ("qwen2.5:14b" if free_gb >= 24 else
                     "qwen2.5:7b" if free_gb >= 10 else "qwen2.5:3b")
        note = f"Auto-selected {model} ({free_gb:.0f} GB free {where})."
        if model != "qwen2.5:14b":
            note += " (Not enough free memory for 14b — using a smaller model.)"
        elif where == "system RAM":
            note += " Note: 14b on CPU is slow."
        return model, note

    # ------------------------------------------------------------------
    # Event handlers (skeleton behavior)
    # ------------------------------------------------------------------
    def _on_backend_change(self, choice):
        if choice.startswith("Cloud"):
            self.dd_model.configure(values=["claude / openai (Settings)"])
            self.var_model.set("claude / openai (Settings)")
            self.set_status("Cloud backend selected — key entry comes "
                            "in a later step.")
        else:
            self.dd_model.configure(values=OLLAMA_MODELS)
            self.var_model.set(DEFAULT_MODEL)
            self.set_status("")

    def _on_autorun_toggle(self):
        # Runs on the Tk thread; mirror into the plain bool the worker reads.
        self._autorun = bool(self.var_autorun.get())
        self.add_message("system",
            "Auto-run ON — tools will run without asking. (Tip: turn off "
            "for risky actions like train.)" if self._autorun
            else "Auto-run OFF — I'll confirm each tool.")

    def _on_browse_load(self):
        """Browse to a 4D-STEM cube and load it as the active dataset.
        Runs on the Tk thread (button callback)."""
        import os
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Choose a 4D-STEM cube",
            filetypes=[("4D-STEM cubes", "*.npy *.npz *.prz *.h5 *.hdf5"),
                       ("All files", "*.*")])
        if not path:
            return
        self.add_message("system", f"loading {path} …")
        try:
            self.app.pre._path_var.set(path)
            self.app.pre._load()
            key = self.app.pre.get_sample_key()
            self.add_message("system",
                f"loaded {os.path.basename(path)} — it's now the active "
                f"dataset.")
        except Exception as e:
            self.add_message("system", f"load failed: {e!r}")

    def _on_device_change(self, choice):
        if choice == "GPU":
            self.warn.configure(
                text=("⚠ chatting on GPU while training can OOM/crash — "
                      "switch to CPU (or qwen2.5:3b) if a run is active"))
        else:
            self.warn.configure(
                text="CPU mode — slower replies, no VRAM contention")

    def _on_send(self):
        text = self.entry.get().strip()
        if not text or self._busy:
            return
        self.entry.delete(0, "end")
        self.add_message("user", text)
        self.messages.append({"role": "user", "content": text})
        self._start_turn()

    def _on_cancel(self):
        if not self._busy:
            return
        self._cancel = True
        self.set_status("cancelling…")

    # ------------------------------------------------------------------
    # Agent loop (runs in a worker thread; all widget writes marshalled
    # back to Tk via self.after).
    # ------------------------------------------------------------------
    def _start_turn(self):
        self._busy = True
        self._cancel = False
        self._set_busy_ui(True)
        # Read Tk variables HERE (Tk thread) — they must not be touched
        # from the worker thread.  Pass the captured config in.
        cfg = {"backend": self.var_backend.get(),
               "model": self.var_model.get(),
               "device": self.var_device.get()}
        t = threading.Thread(target=self._run_turn, args=(cfg,),
                             daemon=True)
        t.start()

    def _make_backend(self, cfg):
        if cfg["backend"].startswith("Cloud"):
            return None   # cloud wired in a later step
        return OllamaBackend(model=cfg["model"], device=cfg["device"])

    def _system_prompt(self) -> str:
        s = getattr(self.app, "session", None)
        sample = getattr(s, "sample", None) or "(none)"
        run = getattr(s, "run_dir", None) or "(none)"
        has_inf = bool(getattr(s, "has_inference", lambda: False)())
        base = (
            "You are the in-app assistant for DINO-4DSTEM: it trains a "
            "self-supervised DINO classifier on 4D-STEM electron "
            "diffraction data and analyses the class maps.\n\n"
            "HONESTY (highest priority): NEVER fabricate file paths, parameter "
            "values, metrics, or results. State a number only if you read it "
            "from a tool (get_state / assess_run / score_run / answer_from_docs) "
            "in this conversation. If you are not sure, SAY SO plainly and offer "
            "to check — 'I'm not certain, let me look' — rather than guessing. A "
            "hedged 'I don't know' is always better than a confident wrong "
            "answer. You may give expert judgement, but label it as judgement, "
            "not fact.\n\n"
            "HOW TO ACT:\n"
            "- Drive the app ONLY by calling the provided functions/tools. "
            "NEVER write a tool call as JSON or a code block in your reply, "
            "and never paste example JSON.\n"
            "- Do NOT ask the user for permission or whether to 'proceed' — "
            "the app pops its own confirmation. Just call the tool.\n"
            "- Call ONE tool, wait for its result, then continue. Don't "
            "narrate a multi-step plan or repeat yourself.\n"
            "- Once a tool's result lets you answer, REPLY IN PLAIN TEXT "
            "immediately and STOP — never call the same tool twice, and don't "
            "keep calling tools after you already have what you need.\n"
            "- If the user asks what you can do / for an overview / says they're "
            "new, call help ONCE and then stop (its output is shown to them "
            "directly — don't restate or summarize it).\n"
            "- Loading is BY FILE PATH ONLY. There are NO named samples/sample "
            "configurations — NEVER ask the user for a sample name. If nothing "
            "is loaded, tell them to click 📂 Load data or give a file path.\n"
            "- NEVER put JSON, ```code```, or pseudo-calls like train(sample=…) "
            "in your reply. Speak in plain English; to act, use the real tool "
            "call mechanism.\n"
            "- When the user asks an OPEN-ENDED question ('what can I do?', "
            "'any other ways?', 'show me something'), briefly LIST the "
            "relevant options instead of pushing a single one.\n"
            "- NEVER repeat the same question/proposal twice. If a tool "
            "call fails, FIX the arguments and try once more, then explain "
            "— do not loop. Never invent argument values (e.g. a fake "
            "sample name or run dir); use the real loaded data / runs.\n"
            "- Use ONLY parameters that exist in a tool's schema; never "
            "invent options (there is no 'log_stretch').\n"
            "- Data is loaded BY FILE PATH (load_data(path=…)); there are no "
            "named samples. All data tools act on the loaded dataset.\n"
            "- NEVER call load_data with a placeholder/example path (e.g. "
            "'/path/to/…'). If the user wants to load but gave no real path, "
            "ASK for the file path, or call show_me_how(target='load data') "
            "to point them to the dataset badge.\n"
            "- Don't GUESS what a control or term means. If you're not "
            "certain, say so (and offer to highlight it) rather than giving a "
            "confident answer that may be wrong.\n"
            "- LEARNING: when the user corrects you, states a fact, or gives "
            "a preference, call remember(text=…, kind=fact|correction|"
            "preference) to persist it (so you don't repeat the mistake). "
            "When they ask you to learn about their setup or 'interview me', "
            "ask a few focused questions ONE AT A TIME (data type & file, "
            "camera length, materials, default vmax/recipe, their goal) and "
            "remember each answer. You can also use answer_from_docs to look "
            "up facts in their paper/notes instead of guessing.\n"
            "- TEACHING / pointing in the GUI: when the user wants to FIND, "
            "LOCATE, or learn to do something THEMSELVES in the interface — "
            "ANY phrasing like 'where is X', 'how do I X', 'which button…', "
            "'I can't find…', 'point me to…', 'show me…' — call "
            "show_me_how(target='X') where X is the control/setting in plain "
            "words (e.g. 'vmax', 'run NMF', 'train', 'class map'). Do this "
            "EVERY time, even if repeated, and NEVER reply with written "
            "directions instead. (If instead they want YOU to perform the "
            "action, call the action tool, e.g. run_nmf / train.)\n"
            "- ADVISING / 'what should I do next', 'which method or "
            "algorithm', 'what parameters', 'how do I proceed': you ARE a "
            "4D-STEM + ML + clustering expert. Call suggest_next_step (reads "
            "the current state) and/or recommend_params (tailored by sample "
            "type: layered vs non-layered), and consult answer_from_docs "
            "about METHOD_GUIDE, then give a SHORT concrete recommendation "
            "grounded in those tools — not generic ML advice. After finishing "
            "an action, briefly offer the logical next step. Prefer the "
            "project's validated recipes.\n"
            "- FIXING A BAD RESULT: when the user says the result is wrong — "
            "'overclustered', 'collapsed', 'one class took everything', "
            "'salt-and-pepper', 'tracks thickness', 'unstable' — call "
            "troubleshoot(symptom=their words) and relay the concrete data + "
            "model fixes; offer to apply one (e.g. lower K and retrain, or "
            "merge/re-cluster post-hoc).\n"
            "- JUDGING A RUN: for 'is my model good / assess this run', call "
            "assess_run (it reads the real cached inference and reports what it "
            "cannot determine — do not invent a verdict). For 'what does this "
            "parameter do / what should it be', call explain_parameter.\n"
            "- CAPABILITIES: if the user asks what you can do / for help / how to "
            "start, call help.\n"
            "- SHOWING IMAGES: to display a saved figure (class map, pattern, "
            "NMF/interpretation output) inline, call show_figure(path) with a "
            "REAL path (find one via list_runs / get_state / the analysis output "
            "folder). Never invent a path.\n"
            "- COMPOUND requests ('load X then train then score'): carry out the "
            "steps in order, calling each tool in turn across rounds — don't stop "
            "after the first step.\n\n"
            "DOMAIN TIPS (4D-STEM — STARTING points; verify on the data then "
            "refine):\n"
            "- vmax: contrast only; ~2 to view (show halo+rings without "
            "saturating), ~5 for training.\n"
            "- center_crop_size: detector field of view; keep the rings "
            "INSIDE and cut high-q noise (~120–150, e.g. 140) — NOT 256.\n"
            "- polar_mask_cols: masks the low-radius beam; ~30–45 (not "
            "single digits).\n"
            "- center_mask_radius: ~15–20 px.\n"
            "- K (num_prototypes): leave ~60 (auto-prunes); epochs ~30–50.\n"
            "- 'bin factor n' = n×n binning (n×n scan positions in "
            "real-space, or n×n detector pixels in q-space, per the radio) "
            "— it is NOT pairwise/2-position averaging.\n"
            "- CLASS MAPS have TWO routes: (a) run_nmf → a CLASSICAL class "
            "map shown ON THE NMF TAB immediately, no training needed; "
            "(b) train a DINO model, then infer + show_class_map. If the "
            "user wants a class map and has no trained run, OFFER BOTH and "
            "note NMF is the quick no-training option. NMF results appear on "
            "the NMF tab — do NOT call show_class_map/infer for NMF (those "
            "are only for trained DINO runs).\n"
            "- To let the user eyeball data: show_pattern(index=…). To try "
            "settings: set_preproc(...) (it redraws the preview live).\n"
            "- NMF has TWO tools: run_nmf (FIT the decomposition + cluster "
            "— slow) and recluster_nmf (re-cluster the EXISTING NMF: change "
            "K or the clustering methods, NO re-fit — fast). If an NMF was "
            "ALREADY run this session and the user only wants to change K or "
            "the clustering method(s), use recluster_nmf, NOT run_nmf. If "
            "it's ambiguous whether they want a fresh fit or just a "
            "re-cluster, ASK. To enable multiple clustering methods pass "
            "methods=['kmeans','aglo','hdbscan','fcm'] (or 'all') — those "
            "actually tick the GUI boxes; never invent method names (only "
            "kmeans/aglo/hdbscan/fcm exist). Don't auto-pick sizes: for the "
            "first run, ask whether to set n_components/n_clusters or choose "
            "automatically (auto_components=knee, auto_clusters=silhouette).\n\n"
            f"CURRENT STATE: loaded_data={sample}, run={run}, "
            f"inference_cached={has_inf}.")
        # In-context learning: inject what the user has taught us.
        try:
            import gui_app.chat_kb as _kb
            learned = _kb.render_for_prompt()
        except Exception:
            learned = ""
        if learned:
            base += ("\n\nLEARNED FROM THIS USER (always honour these; they "
                     "override defaults and correct your knowledge):\n"
                     + learned)
        return base

    def _run_turn(self, cfg):
        backend = self._make_backend(cfg)
        if backend is None:
            self._post("system", "Cloud backend isn't wired up yet — "
                       "switch the backend dropdown to 'Ollama'.")
            self._finish_turn()
            return
        # Auto-provision: start the server if installed, download+launch
        # the installer if not, and pull the model if missing.  Progress
        # shows in the status line; Cancel aborts a long download.
        self._after(self.dot_conn.set, "busy", "preparing Ollama…")
        ok, msg = backend.provision(
            on_progress=lambda t: self._after(self.set_status, t),
            cancel=lambda: self._cancel)
        self._after(self.set_status, "")
        if not ok:
            self._post("system", msg)
            self._after(self.dot_conn.set, "err", "Ollama not ready")
            self._finish_turn()
            return
        if msg:                      # non-fatal note (e.g. fell back to 3b)
            self._post("system", msg)
        self._after(self.dot_conn.set, "ok",
                    f"connected · {backend.active_model}")

        convo = [{"role": "system", "content": self._system_prompt()}]
        convo += self.messages
        schemas = chat_tools.tool_schemas(self.registry)

        try:
            import json as _json
            executed = {}          # sig -> output (dedup within this turn)
            last_out = None        # last useful tool output (answer fallback)
            repeats = 0
            produced_text = False
            self._turn_displayed = False
            for _round in range(MAX_TOOL_ROUNDS):
                if self._cancel:
                    self._post("system", "(cancelled)")
                    break
                result = backend.chat(
                    convo, tools=schemas,
                    on_token=self._stream_token,
                    cancel=lambda: self._cancel)
                self._end_assistant()       # close any streamed bubble
                text = result.get("text", "")
                calls = result.get("tool_calls") or []

                # Fallback: the model may have emitted a tool call as JSON
                # text instead of a real function call — recover it.
                from_text = False
                if not calls and text:
                    tcalls = chat_tools.parse_text_tool_calls(
                        text, self.registry)
                    # Only auto-run a tool parsed from text when the message is
                    # basically JUST that call — never execute example/pseudo
                    # snippets embedded in an explanation.
                    if tcalls and len(text.strip()) < 240:
                        calls = tcalls
                        from_text = True

                # Record the assistant message (with tool_calls) so the
                # model sees its own call when we feed results back.
                amsg = {"role": "assistant", "content": text}
                if calls:
                    amsg["tool_calls"] = [
                        {"function": {"name": c["name"],
                                      "arguments": c["arguments"]}}
                        for c in calls]
                convo.append(amsg)

                if not calls:
                    if text.strip():
                        self.messages.append({"role": "assistant",
                                              "content": text})
                        produced_text = True
                    break

                # Execute each tool call, append its result, loop again.
                for c in calls:
                    if self._cancel:
                        break
                    try:
                        sig = c["name"] + "|" + _json.dumps(
                            c.get("arguments") or {}, sort_keys=True,
                            default=str)
                    except Exception:
                        sig = c["name"]
                    if sig in executed:
                        # Same tool+args already ran this turn — don't repeat;
                        # push the model to give its final answer.
                        repeats += 1
                        self._after(self.add_message, "tool",
                                    f"↳ {c['name']} (repeat — skipped)")
                        convo.append({"role": "tool", "name": c["name"],
                            "content": ("You ALREADY called this with the same "
                            "arguments; its result is above. Do NOT call any "
                            "tool again — write your final answer to the user "
                            "now in plain text.")})
                        continue
                    out = self._execute_tool(c, force_confirm=from_text)
                    executed[sig] = out
                    last_out = out
                    convo.append({"role": "tool",
                                  "name": c["name"],
                                  "content": out})
                if repeats >= 2:
                    break

            # Turn ended without a plain-text answer: surface the last tool
            # output as the answer rather than a bare "(stopped)" note.
            if not produced_text and not self._cancel and not self._turn_displayed:
                if last_out and not last_out.startswith("(Already shown"):
                    self.messages.append({"role": "assistant",
                                          "content": last_out})
                    self._post("assistant", last_out)
                else:
                    self._post("system",
                               f"(stopped after {MAX_TOOL_ROUNDS} tool rounds "
                               "without a final answer)")
        except BackendError as e:
            self._post("system", f"Backend error: {e}")
        except Exception as e:
            self._post("system", f"Unexpected error: {e!r}")
        finally:
            self._finish_turn()

    def _execute_tool(self, call, force_confirm=False) -> str:
        """Run a single tool call in the worker thread; return a short
        text result for the model.  Gated tools pop a confirm dialog
        (blocking this worker thread until the user decides).

        force_confirm: always confirm (used for calls parsed out of the
        model's text, which are heuristic), even if auto-run is on."""
        name = call.get("name", "")
        args = call.get("arguments") or {}
        spec = self.registry.get(name)
        self._after(self.add_message, "tool",
                    f"{name}({self._fmt_args(args)})")
        if spec is None:
            return f"ERROR: unknown tool '{name}'."

        # Human-in-the-loop: confirm before running, unless the user has
        # turned on auto-run or allowed this tool for the session.  Calls
        # parsed from text always confirm (force_confirm).
        gated = spec.confirm and not self._autorun and name not in self._auto_allow
        if gated or force_confirm:
            try:
                summ = spec.summary(args) if spec.summary else name
            except Exception:
                summ = name
            approved, remember = self._confirm_tool(name, summ, args)
            if not approved:
                self._after(self.set_status, "")
                self._after(self.add_message, "tool",
                            f"↳ {name} declined by user")
                return (f"User DECLINED to run '{name}'. Do not retry it; "
                        f"ask what they would prefer instead.")
            if remember:
                self._auto_allow.add(name)

        self._after(self.set_status, f"running tool: {name}…")
        try:
            out = spec.fn(self._tool_ctx, args)
        except Exception as e:
            out = f"ERROR running {name}: {e!r}"
        self._after(self.set_status, "")
        out = str(out)
        # Informational tools: show their curated text DIRECTLY to the user so
        # a weak model can't mangle/ignore it; tell the model not to repeat it.
        if getattr(spec, "display", False) and not out.startswith("ERROR"):
            self._after(self.add_message, "assistant", out)
            self._turn_displayed = True
            return ("(Already shown to the user verbatim — do NOT repeat it. "
                    "Add at most one short follow-up sentence, or just stop.)")
        return out

    # ------------------------------------------------------------------
    # Blocking run-on-Tk-thread helper (used by tools via ToolContext).
    # ------------------------------------------------------------------
    def _call_ui(self, fn, *a, timeout=60.0):
        """Schedule fn(*a) on the Tk thread and BLOCK the calling (worker)
        thread until it returns.  Re-raises any exception fn raised."""
        ev = threading.Event()
        box = {}
        def run():
            try:
                box["r"] = fn(*a)
            except Exception as e:
                box["e"] = e
            finally:
                ev.set()
        self._ui_queue.put((run, ()))
        if not ev.wait(timeout):
            raise TimeoutError(f"UI call timed out after {timeout}s")
        if "e" in box:
            raise box["e"]
        return box.get("r")

    # ------------------------------------------------------------------
    # Confirm dialog (human-in-the-loop).  Built on the Tk thread; the
    # worker thread blocks on an Event until the user decides.
    # ------------------------------------------------------------------
    def _confirm_tool(self, name, summary, args):
        """Returns (approved: bool, remember: bool).  Blocks the worker."""
        ev = threading.Event()
        res = {"approved": False, "remember": False}
        self._ui_queue.put(
            (self._build_confirm_dialog, (name, summary, args, res, ev)))
        # Wait, but bail out if the whole turn is cancelled.
        while not ev.wait(0.1):
            if self._cancel:
                self._after(self._close_confirm_dialog)
                return False, False
        return res["approved"], res["remember"]

    def _build_confirm_dialog(self, name, summary, args, res, ev):
        # Runs on the Tk thread.
        try:
            win = ctk.CTkToplevel(self)
            self._confirm_win = win
            win.title("Confirm action")
            win.geometry("440x300")
            win.transient(self.winfo_toplevel())
            try:
                win.grab_set()
            except Exception:
                pass

            ctk.CTkLabel(win, text="The assistant wants to run a tool:",
                         font=("Segoe UI", 12, "bold")).pack(
                             anchor="w", padx=14, pady=(14, 4))
            ctk.CTkLabel(win, text=f"  {name}", text_color="#A23BB0",
                         font=("Consolas", 12, "bold")).pack(
                             anchor="w", padx=14)
            ctk.CTkLabel(win, text=summary, wraplength=400,
                         justify="left").pack(anchor="w", padx=14, pady=(4, 2))
            argtxt = self._fmt_args(args) or "(no arguments)"
            ctk.CTkLabel(win, text=f"args: {argtxt}", wraplength=400,
                         justify="left", text_color=("#555", "#aaa"),
                         font=("Consolas", 10)).pack(
                             anchor="w", padx=14, pady=(0, 8))

            remember_var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(win, text=f"Allow '{name}' for the rest of "
                            "this session", variable=remember_var).pack(
                                anchor="w", padx=14, pady=(2, 10))

            def finish(approved):
                res["approved"] = approved
                res["remember"] = bool(remember_var.get())
                try:
                    win.grab_release()
                except Exception:
                    pass
                try:
                    win.destroy()
                except Exception:
                    pass
                self._confirm_win = None
                ev.set()

            row = ctk.CTkFrame(win, fg_color="transparent")
            row.pack(side="bottom", fill="x", padx=14, pady=12)
            btn(row, "Skip", command=lambda: finish(False),
                size="small", kind="stop").pack(side="right", padx=(6, 0))
            btn(row, "Confirm", command=lambda: finish(True),
                size="small", kind="run").pack(side="right")
            win.protocol("WM_DELETE_WINDOW", lambda: finish(False))
            win.lift()
            win.focus_force()
        except Exception as e:
            print(f"[chat] confirm dialog error: {e!r}", flush=True)
            res["approved"] = False
            ev.set()

    def _close_confirm_dialog(self):
        win = self._confirm_win
        if win is not None:
            try:
                win.grab_release()
            except Exception:
                pass
            try:
                win.destroy()
            except Exception:
                pass
            self._confirm_win = None

    @staticmethod
    def _fmt_args(args) -> str:
        if not args:
            return ""
        return ", ".join(f"{k}={v!r}" for k, v in args.items())

    # ------------------------------------------------------------------
    # Tk-thread marshaling helpers
    # ------------------------------------------------------------------
    def _after(self, fn, *a):
        """Schedule fn(*a) to run on the Tk thread.  Safe to call from
        the worker thread (enqueue only — no Tk access here)."""
        self._ui_queue.put((fn, a))

    def _poll_ui_queue(self):
        """Runs on the Tk thread: drain queued UI callbacks, then
        reschedule.  Started once from __init__."""
        try:
            while True:
                fn, a = self._ui_queue.get_nowait()
                try:
                    fn(*a)
                except Exception as e:
                    print(f"[chat] UI callback error: {e!r}", flush=True)
        except queue.Empty:
            pass
        try:
            self.after(60, self._poll_ui_queue)
        except Exception:
            pass

    def _post(self, role, text):
        self._after(self.add_message, role, text)

    def _post_image(self, path, caption=None):
        """Embed an image inline in the transcript (Tk thread). Degrades to
        a text line if Pillow is missing or the file can't be read."""
        try:
            from PIL import Image, ImageTk
        except Exception:
            self.add_message("system", f"(install Pillow to show images) {path}")
            return
        try:
            img = Image.open(path)
            maxw = 420
            if img.width > maxw:
                img = img.resize((maxw, int(img.height * maxw / img.width)))
            photo = ImageTk.PhotoImage(img)
            self._img_refs.append(photo)
            tb = self.transcript._textbox
            self.transcript.configure(state="normal")
            if caption:
                tb.insert("end", f"\n{caption}\n", "system")
            else:
                tb.insert("end", "\n")
            tb.image_create("end", image=photo)
            tb.insert("end", "\n")
            self.transcript.configure(state="disabled")
            self.transcript.see("end")
        except Exception as e:
            self.add_message("system", f"(couldn't show image {path}: {e})")

    def _stream_token(self, tok):
        """Backend calls this from the worker thread per token."""
        self._after(self._append_token, tok)

    def _append_token(self, tok):
        # Runs on the Tk thread.  Opens the assistant bubble lazily so we
        # don't print an empty header when a turn is pure tool-calling.
        tb = self.transcript._textbox
        if not self._assistant_open:
            self._write("\nAssistant\n", "assistant_name")
            self._assistant_open = True
            self._assistant_buf = ""
            self._assistant_start = tb.index("end-1c")
        self._assistant_buf += tok
        self._write(tok, "assistant")

    def _end_assistant(self):
        self._after(self._close_assistant)

    def _close_assistant(self):
        if self._assistant_open:
            # Re-render the streamed text cleaned (LaTeX/markdown -> plain),
            # since we couldn't clean it token-by-token while streaming.
            buf = getattr(self, "_assistant_buf", "")
            start = getattr(self, "_assistant_start", None)
            cleaned = _clean_markup(buf)
            if buf and start and cleaned != buf:
                try:
                    tb = self.transcript._textbox
                    self.transcript.configure(state="normal")
                    tb.delete(start, "end-1c")
                    tb.insert(start, cleaned, "assistant")
                    self.transcript.configure(state="disabled")
                except Exception:
                    pass
            self._write("\n", "assistant")
            self._assistant_open = False
            self._assistant_buf = ""
            self._assistant_start = None

    # ------------------------------------------------------------------
    # Busy / cancel UI state
    # ------------------------------------------------------------------
    def _set_busy_ui(self, busy):
        try:
            self.btn_send.configure(state="disabled" if busy else "normal")
            self.btn_cancel.configure(state="normal" if busy else "disabled")
        except Exception:
            pass

    def _finish_turn(self):
        def done():
            self._busy = False
            self._cancel = False
            self._set_busy_ui(False)
            self.set_status("")
        self._after(done)
