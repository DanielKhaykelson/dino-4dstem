"""chat_backends.py -- pluggable LLM backends for the Assistant.

A backend turns a message list (+ optional tool schemas) into either
streamed assistant text or a set of tool calls.  The Assistant's agent
loop (chat_panel.py) is backend-agnostic: it only speaks the
``LLMBackend.chat`` contract below.

Backends:
  - OllamaBackend  (DEFAULT, local, free, no account)  -- this file.
        Talks to a local Ollama server over its HTTP /api/chat endpoint
        via `requests` (no extra python dependency).  Supports
        OpenAI-style function/tool calling and NDJSON streaming.
        Includes a 7b -> 3b automatic fallback when the larger model
        won't load for lack of resources, and CPU/GPU control.
  - CloudBackend   (OPTIONAL, bring-your-own-key)  -- added in a later
        step.

Threading note: ``chat`` is always called from the Assistant's worker
thread, never the Tk thread.  The ``on_token`` callback it invokes is
responsible for marshaling back to Tk (chat_panel does this).
"""
from __future__ import annotations
import json
import os
from typing import Callable, Optional

import requests


DEFAULT_HOST = "http://localhost:11434"


class BackendError(Exception):
    """Raised for any backend-level failure (server down, model missing,
    HTTP error, model-reported error)."""


# ----------------------------------------------------------------------
# Base contract
# ----------------------------------------------------------------------
class LLMBackend:
    def ensure_ready(self) -> "tuple[bool, str]":
        """Return (ok, message).  When ok is False, `message` is a
        human-friendly hint (e.g. how to install / pull a model) that
        the Assistant prints to the transcript instead of chatting."""
        return True, ""

    def chat(self, messages, tools=None, on_token: Optional[Callable] = None,
             cancel: Optional[Callable] = None) -> dict:
        """One assistant turn.

        messages : list of {role, content, [tool_calls], [name]} dicts.
        tools    : list of OpenAI-style tool schemas, or None.
        on_token : called with each streamed text token (worker thread).
        cancel   : zero-arg predicate; if it returns True, stop early.

        Returns {"text": str, "tool_calls": [ {id, name, arguments(dict)} ]}.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------
# Ollama (local, default)
# ----------------------------------------------------------------------
class OllamaBackend(LLMBackend):
    _LADDER = ["qwen2.5:14b", "qwen2.5:7b", "qwen2.5:3b"]

    def __init__(self, model: str = "qwen2.5:7b",
                 host: str = DEFAULT_HOST,
                 device: str = "GPU",
                 fallback_model: str = None,
                 temperature: float = 0.2):
        self.model = model
        self.host = host.rstrip("/")
        self.device = device                 # "GPU" | "CPU"
        if fallback_model is None:           # next-smaller model in the ladder
            try:
                i = self._LADDER.index(model)
                fallback_model = (self._LADDER[i + 1]
                                  if i + 1 < len(self._LADDER)
                                  else "qwen2.5:3b")
            except ValueError:
                fallback_model = "qwen2.5:3b"
        self.fallback_model = fallback_model
        self.temperature = float(temperature)
        self.active_model = model            # may switch to fallback
        self._fell_back = False

    # --- discovery -------------------------------------------------
    @staticmethod
    def server_up(host: str = DEFAULT_HOST, timeout: float = 2.0) -> bool:
        try:
            r = requests.get(host.rstrip("/") + "/api/tags",
                             timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def list_models(host: str = DEFAULT_HOST, timeout: float = 3.0):
        try:
            r = requests.get(host.rstrip("/") + "/api/tags",
                             timeout=timeout)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            return []

    @staticmethod
    def _model_present(models, name) -> bool:
        # Ollama tags carry a :tag suffix (e.g. "qwen2.5:7b"); also accept
        # an implicit ":latest".
        if name in models:
            return True
        base = name.split(":")[0]
        return any(m == name or m.split(":")[0] == base and
                   (name.endswith(":latest") or ":" not in name)
                   for m in models)

    def ensure_ready(self) -> "tuple[bool, str]":
        if not self.server_up(self.host):
            return False, (
                "Ollama doesn't appear to be running.\n"
                "  1. Install it from https://ollama.com/download\n"
                "  2. Then pull a model:  ollama pull qwen2.5:7b\n"
                "     (or the lighter  ollama pull qwen2.5:3b)\n"
                "Ollama runs a local server on http://localhost:11434 and "
                "auto-unloads the model when idle.")
        models = self.list_models(self.host)
        if self._model_present(models, self.model):
            self.active_model = self.model
            return True, ""
        # Requested model missing -- can we fall back?
        if self._model_present(models, self.fallback_model):
            self.active_model = self.fallback_model
            self._fell_back = True
            return True, (f"'{self.model}' isn't pulled — using "
                          f"'{self.fallback_model}' instead. To get the "
                          f"bigger model:  ollama pull {self.model}")
        have = ", ".join(models) if models else "(none)"
        return False, (
            f"Model '{self.model}' isn't pulled. Run:\n"
            f"  ollama pull {self.model}\n"
            f"Installed models: {have}")

    # --- auto-provisioning (start / install / pull) ----------------
    INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"

    @staticmethod
    def find_binary():
        """Locate the ollama executable (PATH or default install dirs)."""
        import shutil
        p = shutil.which("ollama")
        if p:
            return p
        for c in (os.path.expandvars(
                      r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
                  os.path.expandvars(r"%ProgramFiles%\Ollama\ollama.exe")):
            if os.path.exists(c):
                return c
        return None

    def is_installed(self) -> bool:
        return self.find_binary() is not None

    def start_server(self, timeout: float = 25.0) -> bool:
        """Start `ollama serve` in the background and wait until the HTTP
        endpoint answers.  Returns True if the server came up."""
        import subprocess
        import time as _t
        if self.server_up(self.host, timeout=1.5):
            return True                       # already running
        binp = self.find_binary()
        if not binp:
            return False
        try:
            subprocess.Popen(
                [binp, "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            return False
        t0 = _t.time()
        while _t.time() - t0 < timeout:
            if self.server_up(self.host, timeout=1.5):
                return True
            _t.sleep(0.7)
        return self.server_up(self.host)

    def pull_model(self, model, on_progress=None, cancel=None):
        """Pull a model via the server's /api/pull (streams progress).
        Returns (ok, message)."""
        try:
            with requests.post(self.host + "/api/pull",
                               json={"model": model, "stream": True},
                               stream=True, timeout=(10, None)) as r:
                if r.status_code != 200:
                    return False, f"pull failed: {self._http_err(r)}"
                for line in r.iter_lines(decode_unicode=True):
                    if cancel and cancel():
                        return False, "model download cancelled."
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if o.get("error"):
                        return False, f"pull error: {o['error']}"
                    if on_progress:
                        status = o.get("status", "")
                        tot, comp = o.get("total"), o.get("completed")
                        if tot and comp:
                            on_progress(f"downloading {model}: {status} "
                                        f"{comp * 100 // tot}%")
                        elif status:
                            on_progress(f"downloading {model}: {status}")
            return True, f"downloaded {model}"
        except requests.exceptions.RequestException as e:
            return False, f"pull failed: {e}"

    def ensure_model(self, pull=True, on_progress=None, cancel=None):
        """Make sure a usable model is present; pull it if allowed.
        Returns (ok, message)."""
        models = self.list_models(self.host)
        if self._model_present(models, self.model):
            self.active_model = self.model
            return True, ""
        if self._model_present(models, self.fallback_model):
            self.active_model = self.fallback_model
            self._fell_back = True
            return True, (f"using {self.fallback_model} (pull "
                          f"{self.model} for the bigger model)")
        if not pull:
            return False, f"model {self.model} isn't pulled."
        ok, msg = self.pull_model(self.model, on_progress=on_progress,
                                  cancel=cancel)
        if ok:
            self.active_model = self.model
            return True, msg
        if cancel and cancel():
            return False, msg
        # Big model failed — try the lighter fallback.
        if on_progress:
            on_progress(f"falling back to {self.fallback_model}…")
        ok2, msg2 = self.pull_model(self.fallback_model,
                                    on_progress=on_progress, cancel=cancel)
        if ok2:
            self.active_model = self.fallback_model
            self._fell_back = True
            return True, msg2
        return False, msg

    def install(self, on_progress=None):
        """Download the official Ollama installer and run it SILENTLY
        (Ollama installs per-user, so no admin/UAC).  Falls back to the
        interactive installer if silent mode doesn't complete.  Returns
        (ok, message)."""
        import tempfile
        import subprocess
        import time as _t
        try:
            dest = os.path.join(tempfile.gettempdir(), "OllamaSetup.exe")
            with requests.get(self.INSTALLER_URL, stream=True,
                              timeout=(10, None)) as r:
                if r.status_code != 200:
                    return False, (f"couldn't download the installer "
                                   f"(HTTP {r.status_code}). Get it from "
                                   f"https://ollama.com/download")
                total = int(r.headers.get("Content-Length", 0))
                got = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk); got += len(chunk)
                        if on_progress and total:
                            on_progress(f"downloading Ollama installer: "
                                        f"{got * 100 // total}%")
            if on_progress:
                on_progress("installing Ollama silently… (about a minute)")
            # Inno Setup silent flags; per-user install → no UAC prompt.
            flags = ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]
            try:
                subprocess.run([dest] + flags,
                               creationflags=getattr(subprocess,
                                                     "CREATE_NO_WINDOW", 0),
                               timeout=900)
            except Exception as e:
                self._launch_installer(dest)
                return False, (f"automatic install couldn't run ({e}); I "
                               f"opened the installer — finish it, then send "
                               f"your message again.")
            # The installer may take a moment to drop the binary on disk.
            for _ in range(40):
                if self.find_binary():
                    break
                _t.sleep(0.5)
            if not self.find_binary():
                self._launch_installer(dest)
                return False, ("the silent install didn't finish, so I opened "
                               "the installer — complete it, then send your "
                               "message again.")
            return True, "Ollama installed automatically."
        except Exception as e:
            return False, (f"installer download/install failed: {e}. You can "
                           f"install manually from https://ollama.com/download")

    @staticmethod
    def _launch_installer(path):
        import subprocess
        try:
            os.startfile(path)            # Windows
        except Exception:
            try:
                subprocess.Popen([path])
            except Exception:
                pass

    def provision(self, on_progress=None, cancel=None):
        """One call that makes Ollama ready: install it if missing, start
        the server, then ensure a model is pulled — fully automatic.
        Returns (ready, message)."""
        if not self.server_up(self.host):
            if not self.is_installed():
                if on_progress:
                    on_progress("Ollama not found — downloading + installing…")
                ok, msg = self.install(on_progress=on_progress)
                if not ok:
                    return False, msg          # fell back to interactive
                if on_progress:
                    on_progress(msg)
            if on_progress:
                on_progress("starting Ollama…")
            if not self.start_server():
                return False, ("installed Ollama but the server wouldn't "
                               "start — open the Ollama app once, then retry.")
        return self.ensure_model(pull=True, on_progress=on_progress,
                                 cancel=cancel)

    # --- one turn --------------------------------------------------
    def chat(self, messages, tools=None, on_token=None, cancel=None) -> dict:
        try:
            return self._chat_once(self.active_model, messages, tools,
                                   on_token, cancel)
        except BackendError as e:
            # Resource / load failure on the big model -> drop to 3b once.
            if (not self._fell_back
                    and self.active_model != self.fallback_model
                    and self._looks_like_load_failure(str(e))):
                self._fell_back = True
                self.active_model = self.fallback_model
                if on_token:
                    on_token(f"\n[resource limit — retrying on "
                             f"{self.fallback_model}]\n")
                return self._chat_once(self.active_model, messages, tools,
                                       on_token, cancel)
            raise

    @staticmethod
    def _looks_like_load_failure(msg: str) -> bool:
        m = msg.lower()
        return any(s in m for s in (
            "memory", "out of memory", "oom", "unable to load",
            "failed to load", "not enough", "cuda", "vram"))

    def _options(self) -> dict:
        opts = {"temperature": self.temperature}
        if self.device == "CPU":
            opts["num_gpu"] = 0          # 0 layers offloaded -> pure CPU
        return opts

    def _chat_once(self, model, messages, tools, on_token, cancel) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": self._options(),
        }
        if tools:
            payload["tools"] = tools
        url = self.host + "/api/chat"
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        try:
            with requests.post(url, json=payload, stream=True,
                               timeout=(5, 600)) as r:
                if r.status_code != 200:
                    raise BackendError(self._http_err(r))
                for line in r.iter_lines(decode_unicode=True):
                    if cancel and cancel():
                        break
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("error"):
                        raise BackendError(str(obj["error"]))
                    msg = (obj.get("message") or {}) if isinstance(obj, dict) \
                        else {}
                    tok = msg.get("content")
                    if tok:
                        text_parts.append(tok)
                        if on_token:
                            on_token(tok)
                    for tc in msg.get("tool_calls") or []:
                        norm = self._norm_tool_call(tc, len(tool_calls))
                        if norm:
                            tool_calls.append(norm)
                    if obj.get("done"):
                        break
        except requests.exceptions.RequestException as e:
            raise BackendError(f"HTTP error talking to Ollama: {e}")
        return {"text": "".join(text_parts), "tool_calls": tool_calls}

    @staticmethod
    def _http_err(r) -> str:
        try:
            j = r.json()
            return j.get("error") or f"HTTP {r.status_code}"
        except Exception:
            return f"HTTP {r.status_code}: {r.text[:200]}"

    @staticmethod
    def _norm_tool_call(tc, idx) -> "dict | None":
        """Normalize an Ollama tool_call to {id, name, arguments(dict)}."""
        fn = tc.get("function") if isinstance(tc, dict) else None
        if not fn:
            return None
        name = fn.get("name")
        if not name:
            return None
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {}
        return {"id": tc.get("id") or f"call_{idx}",
                "name": name, "arguments": args}
