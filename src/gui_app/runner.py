"""runner.py -- background training as a SUBPROCESS so the GUI can kill
it cleanly via terminate(). The training itself writes:

  <outdir>/training_log.csv     (one row per epoch)
  <outdir>/ckpt_ep<N>.pth       (every save_every epochs)
  <outdir>/best.pth             (when training is done)
  <outdir>/run_summary.json
  <outdir>/_done.flag           (worker writes this on clean exit)
  <outdir>/_error.txt           (worker writes this on exception)

The GUI polls these files for live updates.
"""
from __future__ import annotations
import os, sys, time, json, csv, subprocess
from datetime import datetime


def make_outdir(root: str = None) -> str:
    if root is None:
        root = os.path.join("runs", "_gui")
    os.makedirs(root, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, stamp)
    os.makedirs(out, exist_ok=True)
    return out


def read_training_log(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path) as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def list_ckpts(outdir: str) -> list:
    out = []
    if not os.path.isdir(outdir):
        return out
    for fn in os.listdir(outdir):
        if fn.startswith("ckpt_ep") and fn.endswith(".pth"):
            try:
                ep = int(fn[len("ckpt_ep"):-len(".pth")])
                out.append((ep, os.path.join(outdir, fn)))
            except ValueError:
                pass
    out.sort()
    return out


WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_train_worker.py")


class TrainingJob:
    """Wraps a background training run as a subprocess. Status is
    file-system-derived so the GUI thread can poll without locks."""

    def __init__(self, sample: str, outdir: str, kwargs: dict):
        self.sample = sample
        self.outdir = outdir
        self.kwargs = kwargs           # the run_config kwargs (excluding
                                        # sample/outdir/device, which the
                                        # worker fills in itself)
        self._proc: "subprocess.Popen | None" = None
        self._t_start: float = 0.0      # subprocess spawn (includes Python
                                         # startup + dataset load + model
                                         # build — i.e. the time BEFORE
                                         # epoch 1 starts)
        self._t_first_row: float = 0.0  # wall-clock when the first CSV row
                                         # appears (≈ end of epoch 1).  All
                                         # user-visible "elapsed" / "per-
                                         # epoch" stats are anchored here so
                                         # startup overhead is excluded.
        self._t_end: float = 0.0
        self._stopped: bool = False     # user clicked stop
        self._last_n_rows: int = 0

    @property
    def csv_path(self):
        return os.path.join(self.outdir, "training_log.csv")

    @property
    def done_flag(self):
        return os.path.join(self.outdir, "_done.flag")

    @property
    def error_flag(self):
        return os.path.join(self.outdir, "_error.txt")

    def start(self):
        if self.is_running():
            return
        # write kwargs json
        os.makedirs(self.outdir, exist_ok=True)
        spec = dict(self.kwargs)
        spec["sample"] = self.sample
        spec["outdir"] = self.outdir
        # Pass the resolved sample-config dict so the worker subprocess
        # can register runtime-only samples (loaded ad-hoc on the Pre-
        # processing tab) into its own data.SAMPLES before run_contrastive
        # tries to look up cfg = EVAL_SAMPLES[sample].
        try:
            from data import SAMPLES
            cfg = SAMPLES.get(self.sample)
            if cfg is not None:
                # only keep JSON-friendly fields (drop anything pickled)
                safe = {k: v for k, v in cfg.items()
                        if isinstance(v, (str, int, float, bool, list,
                                            tuple, dict, type(None)))}
                spec["_sample_config"] = safe
        except Exception:
            pass
        # any non-JSON-serialisable items must be cast first; we already
        # pass plain ints/floats/bools/strings/lists from the GUI
        spec_path = os.path.join(self.outdir, "_train_kwargs.json")
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2, default=str)
        # spawn
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self._t_start = time.perf_counter()
        # use the SAME interpreter as the GUI
        self._proc = subprocess.Popen(
            [sys.executable, "-u", WORKER, spec_path],
            # Run in the GUI's working dir so the relative outdir / kwargs
            # paths resolve identically. (Do NOT derive this from __file__:
            # after the move into src/, dirname(dirname(__file__)) is src/,
            # not the repo root, which broke training.)
            cwd=os.getcwd(),
            env=env,
            stdout=open(os.path.join(self.outdir, "_stdout.log"), "w",
                          encoding="utf-8"),
            stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                            if os.name == "nt" else 0),
        )

    def stop(self, kill_after_s: float = 4.0):
        """Ask the worker to stop. terminate() first; SIGKILL after
        timeout if it ignores it."""
        if not self.is_running():
            return
        self._stopped = True
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=kill_after_s)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
        except Exception:
            pass
        self._t_end = time.perf_counter()

    def status(self) -> str:
        """idle | running | done | failed | stopped"""
        if self._proc is None:
            return "idle"
        rc = self._proc.poll()
        if rc is None:
            return "running"
        # process has exited
        if not self._t_end:
            self._t_end = time.perf_counter()
        if self._stopped:
            return "stopped"
        if os.path.exists(self.done_flag):
            return "done"
        if os.path.exists(self.error_flag) or rc != 0:
            return "failed"
        return "done"

    def error(self) -> "str | None":
        if os.path.exists(self.error_flag):
            try:
                return open(self.error_flag).read()
            except Exception:
                return None
        return None

    def elapsed(self) -> float:
        """Wall time since subprocess spawn (i.e. INCLUDES startup
        overhead). Useful for the post-mortem 'TOTAL after Xs' line."""
        if not self._t_start:
            return 0.0
        end = self._t_end if self._t_end else time.perf_counter()
        return end - self._t_start

    # ---- training-only timing (excludes Python/data startup) ----
    def note_rows(self, n_rows: int) -> None:
        """Tell the job how many rows are currently in training_log.csv.
        The first time this exceeds zero we anchor the 'training really
        started' timestamp."""
        if n_rows >= 1 and self._t_first_row == 0.0:
            self._t_first_row = time.perf_counter()
        self._last_n_rows = int(n_rows)

    def startup_seconds(self) -> float:
        """Time spent before the first epoch logged. Returns 0.0 until
        epoch 1 finishes (and the runner has called note_rows)."""
        if self._t_first_row == 0.0 or not self._t_start:
            return 0.0
        return self._t_first_row - self._t_start

    def elapsed_training(self) -> float:
        """Time since the first epoch row appeared in the CSV. 0.0 while
        we're still in startup."""
        if self._t_first_row == 0.0:
            return 0.0
        end = self._t_end if self._t_end else time.perf_counter()
        return max(0.0, end - self._t_first_row)

    def time_per_epoch(self, n_rows: int = None) -> float:
        """Average wall time per epoch since the first row landed.

        Computed as elapsed_training / (n_rows - 1), since the very
        first row arrives at t_first_row itself (so by definition
        elapsed_training is 0 right after epoch 1 finishes).  Returns
        0.0 until at least two epochs are done."""
        if n_rows is None:
            n_rows = self._last_n_rows
        if n_rows < 2 or self._t_first_row == 0.0:
            return 0.0
        return self.elapsed_training() / float(n_rows - 1)

    def is_running(self) -> bool:
        return self.status() == "running"
