"""chat_tools.py -- tool registry for the Assistant agent loop.

Each tool is a small, typed function the LLM can call to drive the
pipeline or read state.  A tool entry (ToolSpec) has:

    name        : str
    description : str                  (shown to the model)
    parameters  : JSON-schema dict     (OpenAI/Ollama function-call shape)
    fn          : callable(ctx, args) -> str
    confirm     : bool                 (gate behind a confirm dialog)
    summary     : callable(args) -> str   (one-line "about to do X")

`fn` runs in the Assistant's WORKER THREAD.  It must NOT touch Tk
widgets directly; instead it uses the ToolContext:

    ctx.app                  the main App (shared state, panels)
    ctx.call_ui(fn,*a)       run fn on the Tk thread, block, return result
    ctx.post(role, text)     fire-and-forget transcript message
    ctx.status(text)         set the one-line status

DATA HYGIENE: tool fns return compact text (shapes, scalars, paths) —
never raw arrays.

Implemented so far (Step 3a):
    list_samples, get_state, list_runs, open_tab, load_sample,
    set_preproc, show_class_map
Compute tools (infer, train, class_average, run_interpretation,
run_acom, run_nmf, score_run) land in Step 3b; recommend_params in
Step 4; answer_from_docs in Step 5.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Callable


# ----------------------------------------------------------------------
# Tool context + spec
# ----------------------------------------------------------------------
class ToolContext:
    """Handed to every tool fn.  Bound to a ChatPanel by the panel."""
    def __init__(self, app, call_ui, post, status, cancel=None):
        self.app = app
        self.call_ui = call_ui      # (fn, *a, timeout=?) -> result (blocks)
        self.post = post            # (role, text) -> None
        self.status = status        # (text) -> None
        self.cancel = cancel or (lambda: False)   # () -> bool (Stop pressed)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable                 # (ctx, args_dict) -> str
    confirm: bool = True
    summary: Callable = None     # (args_dict) -> str


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _resolve_sample(name_or_path: str):
    """Map a user string to a SAMPLES key (and path), or a filesystem
    path.  Returns (key|None, path|None, note)."""
    from data import SAMPLES
    s = (name_or_path or "").strip()
    if not s:
        return None, None, "empty name"
    # 1) exact key
    if s in SAMPLES:
        return s, SAMPLES[s].get("path"), ""
    # 2) normalized fuzzy match (spaces/dashes -> underscore, casefold)
    def norm(x):
        return x.replace(" ", "_").replace("-", "_").casefold()
    ns = norm(s)
    exact = [k for k in SAMPLES if norm(k) == ns]
    if len(exact) == 1:
        return exact[0], SAMPLES[exact[0]].get("path"), ""
    contains = [k for k in SAMPLES if ns in norm(k) or norm(k) in ns]
    if len(contains) == 1:
        return contains[0], SAMPLES[contains[0]].get("path"), \
            f"matched '{contains[0]}'"
    if len(contains) > 1:
        return None, None, ("ambiguous — matches: "
                            + ", ".join(sorted(contains)[:10]))
    # 3) a filesystem path
    if os.path.exists(s):
        return None, s, "path"
    return None, None, f"no sample or file matches '{s}'"


def _find_run_dirs(root="runs", cap=60):
    """Walk `root` for directories that look like training runs (contain
    best.pth / latest.pth / run_summary.json).  Returns list of dicts."""
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        fset = set(filenames)
        is_run = bool(fset & {"best.pth", "latest.pth", "run_summary.json"}
                      or any(f.startswith("ckpt_ep") for f in filenames))
        if not is_run:
            continue
        info = {"dir": dirpath.replace("\\", "/"),
                "has_best": "best.pth" in fset,
                "has_inference": os.path.exists(
                    os.path.join(dirpath, "eval", "inference.npz"))}
        rs = os.path.join(dirpath, "run_summary.json")
        if os.path.exists(rs):
            try:
                import json
                js = json.load(open(rs, encoding="utf-8"))
                info["sample"] = (js.get("sample")
                                  or js.get("cfg", {}).get("sample"))
            except Exception:
                pass
        out.append(info)
        # don't descend into a run dir's own subfolders (eval/, acom/…)
        dirnames[:] = [d for d in dirnames
                       if d not in ("eval", "acom", "_interpretability",
                                    "paper_attribution", "_imc_report")]
        if len(out) >= cap:
            break
    return out


# Friendly tab name -> navigation thunk(app).
def _nav(app, *path_subtab):
    """path_subtab is a sequence of (tabview_attr_or_key, name) steps."""
    app._tabs.set(path_subtab[0])
    cur = path_subtab[0]
    for key, name in path_subtab[1:]:
        tv = app._group_tabs.get(key) if key in app._group_tabs else None
        if tv is not None:
            tv.set(name)


TAB_ROUTES = {
    "pre-processing": lambda app: (app._tabs.set("Data"),
        app._group_tabs["Data"].set("Pre-processing"),
        app._group_tabs["Pre-processing"].set("Load + Pre-process")),
    "training": lambda app: (app._tabs.set("Model"),
        app._group_tabs["Model"].set("Training"),
        app._group_tabs["Training"].set("Training")),
    "eval": lambda app: (app._tabs.set("Model"),
        app._group_tabs["Model"].set("Training"),
        app._group_tabs["Training"].set("Eval")),
    "post-hoc": lambda app: (app._tabs.set("Analysis"),
        app._group_tabs["Analysis"].set("Post-hoc"),
        app._posthoc_tabs.set("Analysis")),
    "interpretation": lambda app: (app._tabs.set("Analysis"),
        app._group_tabs["Analysis"].set("Post-hoc"),
        app._posthoc_tabs.set("Interpretation"),
        app._on_posthoc_subtab()),
    "nmf": lambda app: (app._tabs.set("Clustering"),
        app._group_tabs["Clustering"].set("NMF + cluster")),
    "dino-cluster": lambda app: (app._tabs.set("Clustering"),
        app._group_tabs["Clustering"].set("DINO + cluster")),
    "sam": lambda app: (app._tabs.set("Clustering"),
        app._group_tabs["Clustering"].set("SAM"), app._on_tab_change()),
    "blob": lambda app: (app._tabs.set("Diffraction"),
        app._group_tabs["Diffraction"].set("Blob"), app._on_tab_change()),
    "acom": lambda app: (app._tabs.set("Diffraction"),
        app._group_tabs["Diffraction"].set("ACOM")),
    "transfer": lambda app: (app._tabs.set("Model"),
        app._group_tabs["Model"].set("Transfer"), app._on_tab_change()),
}


# ----------------------------------------------------------------------
# Tool implementations
# ----------------------------------------------------------------------
def _list_samples(ctx, args) -> str:
    from data import SAMPLES
    if not SAMPLES:
        return "No samples are configured."
    rows = []
    for k in sorted(SAMPLES):
        cfg = SAMPLES[k] or {}
        base = str(cfg.get("path", "")).replace("\\", "/").split("/")[-1]
        rows.append(f"  - {k}: scan={cfg.get('scan_shape')}, "
                    f"vmax={cfg.get('vmax')}, file={base}")
    cur = getattr(getattr(ctx.app, "session", None), "sample", None)
    head = f"{len(SAMPLES)} samples configured"
    if cur:
        head += f" (current: {cur})"
    return head + ":\n" + "\n".join(rows)


def _get_state(ctx, args) -> str:
    s = getattr(ctx.app, "session", None)
    sample = getattr(s, "sample", None) or "(none)"
    run = getattr(s, "run_dir", None) or "(none)"
    has_inf = bool(getattr(s, "has_inference", lambda: False)())
    # pre-processing kwargs live in Tk vars -> read on the Tk thread.
    pre = {}
    try:
        pre = ctx.call_ui(ctx.app.pre.get_pre_kwargs)
    except Exception:
        pre = {}
    pre_str = ", ".join(f"{k}={v}" for k, v in pre.items()) or "(n/a)"
    return (f"sample={sample}\nrun_dir={run}\ninference_cached={has_inf}\n"
            f"preproc: {pre_str}")


def _list_runs(ctx, args) -> str:
    runs = _find_run_dirs()
    if not runs:
        return "No training runs found under runs/."
    lines = []
    for r in runs:
        tags = []
        if r.get("sample"):
            tags.append(f"sample={r['sample']}")
        tags.append("best" if r["has_best"] else "no-best")
        if r["has_inference"]:
            tags.append("inference✓")
        lines.append(f"  - {r['dir']}  [{', '.join(tags)}]")
    return f"{len(runs)} run dir(s):\n" + "\n".join(lines)


_TAB_ALIASES = {
    "pre": "pre-processing", "preprocess": "pre-processing",
    "preprocessing": "pre-processing", "data": "pre-processing",
    "post hoc": "post-hoc", "posthoc": "post-hoc", "analysis": "post-hoc",
    "interpret": "interpretation", "dino": "dino-cluster",
    "dino cluster": "dino-cluster", "nmf cluster": "nmf",
    "nmf + cluster": "nmf", "model": "training", "train": "training",
    "class map": "eval", "evaluation": "eval", "diffraction": "acom",
}


def _open_tab(ctx, args) -> str:
    # Accept several arg names the model might use (tab/name/target/…).
    raw = (args.get("name") or args.get("tab") or args.get("tabname")
           or args.get("target") or "")
    key = str(raw).strip().lower().replace("_", " ").replace("+", " ")
    key = " ".join(key.split())
    route = TAB_ROUTES.get(key) or TAB_ROUTES.get(key.replace(" ", "-"))
    if route is None and key in _TAB_ALIASES:
        key = _TAB_ALIASES[key]; route = TAB_ROUTES.get(key)
    if route is None:                       # fuzzy contains-match
        for k in TAB_ROUTES:
            if k in key or key in k:
                key, route = k, TAB_ROUTES[k]; break
    if route is None:
        return ("unknown tab '" + str(raw) + "'. options: "
                + ", ".join(sorted(TAB_ROUTES)))
    ctx.call_ui(route, ctx.app)
    return f"switched to the '{key}' tab."


def _load_data(ctx, args) -> str:
    """Load a 4D-STEM cube BY PATH only.  Drives the Pre-processing
    panel's tested loader (which registers the cube under the hood and
    makes it the active dataset via the session)."""
    path = args.get("path") or args.get("name_or_path")
    if not path:
        return "ERROR: provide a file path (path='D:/.../cube.npy')."
    path = str(path)
    if not os.path.exists(path):
        return f"ERROR: no file at {path}"
    def do_load():
        ctx.app.pre._path_var.set(path)
        ctx.app.pre._load()
        return ctx.app.pre.get_sample_key()
    try:
        ctx.call_ui(do_load, timeout=120.0)
    except Exception as e:
        return f"ERROR loading cube: {e!r}"
    overrides = {k: args[k] for k in
                 ("vmax", "center_crop", "center_crop_size",
                  "center_mask_radius", "polar_mask_cols")
                 if k in args and args[k] is not None}
    applied = ("\n" + _set_preproc(ctx, overrides)) if overrides else ""
    sess = getattr(ctx.app, "session", None)
    return (f"Loaded {os.path.basename(path)} — it's now the active "
            f"dataset." + applied)


def _active_sample(ctx, args):
    """Resolve which loaded dataset a tool should act on: an explicit
    sample handle if given, else the currently-loaded data.  Returns
    (sample_or_None, error_msg_or_None)."""
    from data import SAMPLES
    s = getattr(ctx.app, "session", None)
    sample = args.get("sample") or getattr(s, "sample", None)
    if not sample:
        return None, ("ERROR: no data is loaded. Use the 'Load data' "
                      "button or load_data(path=…) first.")
    if sample not in SAMPLES:
        return None, (f"ERROR: '{sample}' isn't a loaded dataset. Load a "
                      f"cube by path first.")
    return sample, None


def _set_preproc(ctx, args) -> str:
    def apply():
        pre = ctx.app.pre
        changed = []
        if args.get("vmax") is not None:
            pre.vmax.set(float(args["vmax"])); changed.append("vmax")
        cc = args.get("center_crop_size", args.get("center_crop"))
        if cc is not None:
            pre.center_crop_size.set(int(cc)); changed.append("center_crop_size")
        if args.get("polar_mask_cols") is not None:
            pre.polar_mask_cols.set(int(args["polar_mask_cols"]))
            changed.append("polar_mask_cols")
        if args.get("center_mask_radius") is not None:
            try:
                pre.center_mask_radius.set(int(args["center_mask_radius"]))
                changed.append("center_mask_radius")
            except Exception:
                pass
        if args.get("com_centering") is not None:
            pre.com.set(bool(args["com_centering"]))
            changed.append("com_centering")
        # Push the change to the Training tab (keeps its snapshot fresh).
        try:
            pre.on_state_change("pre_kwargs_changed")
        except Exception:
            pass
        # Redraw the Pre-processing preview so the change is VISIBLE.
        try:
            pre._refresh()
        except Exception:
            pass
        return pre.get_pre_kwargs(), changed
    try:
        kw, changed = ctx.call_ui(apply)
    except Exception as e:
        return f"ERROR setting preproc: {e!r}"
    if not changed:
        return "no preproc fields changed (nothing recognized)."
    return ("set " + ", ".join(changed) + ". preproc now: "
            + ", ".join(f"{k}={v}" for k, v in kw.items()))


def _show_class_map(ctx, args) -> str:
    run_dir = args.get("run_dir")
    s = getattr(ctx.app, "session", None)
    if not run_dir:
        run_dir = getattr(s, "run_dir", None)
    if not run_dir or not os.path.isdir(run_dir):
        return ("ERROR: no valid run_dir (pass run_dir=… or load a run "
                "first).")
    sample = args.get("sample") or getattr(s, "sample", None)

    def drive():
        ep = ctx.app.eval_panel
        ep._mode_var.set("LOAD")
        ep._load_path_var.set(run_dir)
        if sample:
            try: ep._sample_var.set(sample)
            except Exception: pass
        TAB_ROUTES["eval"](ctx.app)
        ep._render_from_load()    # spawns its own worker thread
    try:
        ctx.call_ui(drive)
    except Exception as e:
        return f"ERROR: {e!r}"
    return (f"rendering class map for {run_dir}"
            + (f" (sample {sample})" if sample else "")
            + " on the Eval tab — it will appear shortly.")


def _show_pattern(ctx, args) -> str:
    """Display a single diffraction pattern (by scan index) in the
    Pre-processing preview, applying current vmax/crop."""
    idx = args.get("index", args.get("idx"))
    def drive():
        pre = ctx.app.pre
        if getattr(pre, "cube", None) is None:
            return "ERROR: no data loaded — load a cube first."
        Ny, Nx = pre.cube.shape[0], pre.cube.shape[1]
        N = Ny * Nx
        i = 0 if idx is None else max(0, min(int(idx), N - 1))
        pre.idx.set(i)
        pre._refresh()
        TAB_ROUTES["pre-processing"](ctx.app)
        return f"showing pattern {i} of {N} on the Pre-processing tab."
    try:
        return ctx.call_ui(drive)
    except Exception as e:
        return f"ERROR: {e!r}"


# ----------------------------------------------------------------------
# Compute helpers (run inline in the worker thread; no Tk access)
# ----------------------------------------------------------------------
def _pick_ckpt(run_dir):
    from gui_app.runner import list_ckpts
    best = os.path.join(run_dir, "best.pth")
    if os.path.exists(best):
        return best, "best"
    cks = list_ckpts(run_dir)
    if cks:
        ep, p = cks[-1]
        return p, f"ep{ep}"
    return None, None


def _polar_cfg_from_run(run_dir):
    """Recover the training-time polar/mask config so inference matches."""
    mask_r, mask_cols, ccrop, com = 15, 45, 140, True
    rs = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(rs):
        try:
            import json
            c = json.load(open(rs, encoding="utf-8")).get("cfg", {})
            mask_r = int(c.get("center_mask_radius", mask_r))
            mask_cols = int(c.get("polar_mask_cols", mask_cols))
            ccrop = int(c.get("center_crop_size", ccrop))
            com = bool(c.get("com_centering", com))
        except Exception:
            pass
    return mask_r, mask_cols, ccrop, com


def _sample_of_run(run_dir, fallback=None):
    rs = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(rs):
        try:
            import json
            js = json.load(open(rs, encoding="utf-8"))
            return (js.get("sample") or js.get("cfg", {}).get("sample")
                    or fallback)
        except Exception:
            pass
    return fallback


def _run_inference(run_dir, sample, device=None, save=True):
    """Load best/latest ckpt + dataset, run infer_scan, optionally write
    eval/inference.npz.  Mirrors EvalPanel._infer.  Returns
    (inf_dict, scan_shape, ckpt_label)."""
    import numpy as np
    import torch
    from data import LoadPRZ, SAMPLES
    from dino_sr_contrastive_model import load_contrastive_checkpoint
    from contrastive_eval import infer_scan
    ckpt, label = _pick_ckpt(run_dir)
    if ckpt is None:
        raise RuntimeError(f"no checkpoint found in {run_dir}")
    if sample not in SAMPLES:
        raise RuntimeError(f"sample '{sample}' is not in data.SAMPLES")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "cpu"))
    out = load_contrastive_checkpoint(ckpt, device=dev)
    model = out[0] if isinstance(out, tuple) else out
    model.eval()
    cfg = SAMPLES[sample]
    ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    mr, mc, cc, com = _polar_cfg_from_run(run_dir)
    inf = infer_scan(model, ds, dev, dense_remap=True, polar_size=192,
                     polar_mask_cols=mc, center_crop_size=cc,
                     com_centering=com, center_mask_radius=mr,
                     eval_temp=0.06, batch_size=128)
    if save:
        eval_dir = os.path.join(run_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        kw = dict(soft_probs=inf["soft_probs"], assigns=inf["assigns"],
                  embeds=inf["embeds"])
        if inf.get("teacher_probs") is not None:
            kw["teacher_probs"] = inf["teacher_probs"]
        kw["K_original_ids"] = np.asarray(inf.get("K_original_ids", []),
                                          dtype=np.int64)
        kw["K_original"] = np.int64(inf.get("K_original", 0))
        np.savez_compressed(os.path.join(eval_dir, "inference.npz"), **kw)
    return inf, cfg["scan_shape"], label


def _load_or_infer(run_dir, sample):
    """Return (assigns, soft_probs) — from cached inference.npz if present,
    else by running inference."""
    import numpy as np
    npz = os.path.join(run_dir, "eval", "inference.npz")
    if os.path.exists(npz):
        try:
            d = np.load(npz, allow_pickle=True)
            return d["assigns"], d["soft_probs"], "cached"
        except Exception:
            pass
    inf, _, label = _run_inference(run_dir, sample, save=True)
    return inf["assigns"], inf["soft_probs"], label


# ----------------------------------------------------------------------
# Compute tool implementations (Step 3b-1)
# ----------------------------------------------------------------------
def _infer(ctx, args) -> str:
    s = getattr(ctx.app, "session", None)
    run_dir = args.get("run_dir") or getattr(s, "run_dir", None)
    if not run_dir or not os.path.isdir(run_dir):
        return "ERROR: need a valid run_dir (pass run_dir=… or load a run)."
    sample = (args.get("sample") or getattr(s, "sample", None)
              or _sample_of_run(run_dir))
    if not sample:
        return ("ERROR: could not determine the sample; pass sample=… "
                "(must be a configured dataset).")
    ctx.status(f"running inference on {sample}…")
    try:
        inf, scan_shape, label = _run_inference(run_dir, sample)
    except Exception as e:
        return f"ERROR during inference: {e!r}"
    import numpy as np
    assigns = inf["assigns"]
    K = int(inf["soft_probs"].shape[1])
    counts = np.bincount(assigns, minlength=K)
    conf = float(inf["soft_probs"].max(1).mean())
    # Update session on the Tk thread (emit touches badge widgets).
    def upd():
        ctx.app.session.set(run_dir=run_dir, sample=sample,
                            inference=dict(soft_probs=inf["soft_probs"],
                                           assigns=assigns,
                                           embeds=inf["embeds"]))
    try:
        ctx.call_ui(upd)
    except Exception:
        pass
    order = np.argsort(-counts)
    sizes = ", ".join(f"c{int(i)}:{int(counts[i])}" for i in order)
    return (f"Inference done ({label} ckpt) on {sample}: N={assigns.size} "
            f"positions, K_active={K}, mean confidence={conf:.3f}, "
            f"scan_shape={tuple(scan_shape)}. Class sizes: {sizes}. "
            f"Wrote eval/inference.npz and cached it in the session.")


def _class_average(ctx, args) -> str:
    s = getattr(ctx.app, "session", None)
    run_dir = args.get("run_dir") or getattr(s, "run_dir", None)
    if not run_dir or not os.path.isdir(run_dir):
        return "ERROR: need a valid run_dir."
    cls = args.get("class_id")
    if cls is None:
        return "ERROR: provide class_id (integer)."
    cls = int(cls)
    sample = (args.get("sample") or getattr(s, "sample", None)
              or _sample_of_run(run_dir))
    if not sample:
        return "ERROR: could not determine the sample; pass sample=…"
    ctx.status(f"computing class average for class {cls}…")
    try:
        import numpy as np
        from data import LoadPRZ, SAMPLES
        assigns, soft, src = _load_or_infer(run_dir, sample)
        K = int(soft.shape[1])
        if cls < 0 or cls >= K:
            return f"ERROR: class_id {cls} out of range (0..{K-1})."
        idx = np.where(assigns == cls)[0]
        if idx.size == 0:
            return f"class {cls} has no members."
        cfg = SAMPLES[sample]
        ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
        sc = soft[idx, cls]
        top = idx[np.argsort(-sc)[:min(200, idx.size)]]
        w = soft[top, cls].astype(np.float32)
        pats = np.stack([ds.get_raw(int(i)) for i in top], 0).astype(np.float32)
        wavg = (pats * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
        eval_dir = os.path.join(run_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        out_png = os.path.join(eval_dir, f"class_avg_{cls}.png")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        disp = np.clip(wavg / max(float(cfg["vmax"]), 1e-6), 0, 1)
        plt.imsave(out_png, disp, cmap="inferno")
    except Exception as e:
        return f"ERROR computing class average: {e!r}"
    return (f"Class {cls}: {idx.size} members (averaged top "
            f"{min(200, idx.size)} by confidence; assigns from {src}). "
            f"Saved confidence-weighted mean diffraction pattern to "
            f"{out_png.replace(chr(92), '/')}.")


def _train(ctx, args) -> str:
    key, err = _active_sample(ctx, args)
    if err:
        return err
    overrides = args.get("cfg_overrides") or {}
    if not isinstance(overrides, dict):
        overrides = {}

    def start():
        tp = ctx.app.train
        if getattr(tp, "job", None) is not None and tp.job.is_running():
            return {"err": "a training job is already running; wait for it "
                    "to finish or stop it on the Training tab."}
        tp.var["sample"].set(key)
        applied, skipped = [], []
        for k, v in overrides.items():
            if k in tp.var:
                try:
                    tp.var[k].set(v); applied.append(k)
                except Exception:
                    skipped.append(k)
            else:
                skipped.append(k)
        tp._on_train()   # may pop confirm/messagebox dialogs (radials etc.)
        job = getattr(tp, "job", None)
        return {"outdir": getattr(job, "outdir", None),
                "status": job.status() if job else "unknown",
                "applied": applied, "skipped": skipped}
    try:
        r = ctx.call_ui(start, timeout=300.0)
    except Exception as e:
        return f"ERROR starting training: {e!r}"
    if r.get("err"):
        return "ERROR: " + r["err"]
    msg = (f"Training started for {key} → {r.get('outdir')} "
           f"(status: {r.get('status')}).")
    if r.get("applied"):
        msg += " Applied overrides: " + ", ".join(r["applied"]) + "."
    if r.get("skipped"):
        msg += " Ignored (unknown/invalid): " + ", ".join(r["skipped"]) + "."
    msg += (" The Training tab now shows the live loss; this runs in a "
            "separate process. Use score_run or infer when it finishes.")
    return msg


def _score_run(ctx, args) -> str:
    s = getattr(ctx.app, "session", None)
    sample = args.get("sample") or getattr(s, "sample", None)
    if not sample:
        return "ERROR: provide sample."
    label = args.get("label")
    run_dir = args.get("run_dir") or getattr(s, "run_dir", None)
    ckpt = None
    if run_dir:
        cand = os.path.join(run_dir, "best.pth")
        if os.path.exists(cand):
            ckpt = cand
    ctx.status(f"scoring {sample}…")
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        from scorecard import score_run as _sr
        res = _sr(sample=sample, label=label, device=dev,
                  save_outputs=True, ckpt_path=ckpt)
    except Exception as e:
        return (f"ERROR scoring: {e!r}. Note: score_run only supports "
                f"samples present in the scorecard's approved config "
                f"(the paper datasets), and needs a label or an "
                f"approved_label for that sample.")
    crit = getattr(res, "critical_failures", None)
    return (f"Scorecard for {sample}/{getattr(res, 'label', '?')}: "
            f"verdict={res.verdict}, overall={res.overall:.3f}, "
            f"weakest={res.weakest_component}"
            + (f", {len(crit)} critical failure(s)" if crit else ""))


def _load_or_infer_full(run_dir, sample):
    """Like _load_or_infer but also returns embeds (for interpretation)."""
    import numpy as np
    npz = os.path.join(run_dir, "eval", "inference.npz")
    if os.path.exists(npz):
        try:
            d = np.load(npz, allow_pickle=True)
            return d["assigns"], d["soft_probs"], d["embeds"], "cached"
        except Exception:
            pass
    inf, _, label = _run_inference(run_dir, sample, save=True)
    return inf["assigns"], inf["soft_probs"], inf["embeds"], label


# ----------------------------------------------------------------------
# Compute tool implementations (Step 3b-2)
# ----------------------------------------------------------------------
def _run_interpretation(ctx, args) -> str:
    s = getattr(ctx.app, "session", None)
    run_dir = args.get("run_dir") or getattr(s, "run_dir", None)
    if not run_dir or not os.path.isdir(run_dir):
        return "ERROR: need a valid run_dir."
    sample = (args.get("sample") or getattr(s, "sample", None)
              or _sample_of_run(run_dir))
    if not sample:
        return "ERROR: could not determine the sample; pass sample=…"
    try:
        from data import SAMPLES
        import gui_app.interpret_core as ic
        if sample not in SAMPLES:
            return f"ERROR: '{sample}' is not a configured dataset."
        ctx.status("interpretation: loading assignments…")
        assigns, soft, embeds, src = _load_or_infer_full(run_dir, sample)
        cfg = SAMPLES[sample]
        polar = _polar_cfg_from_run(run_dir)   # (mask_r, mask_cols, ccrop, com)
        cx = ic.Ctx(run_dir, sample, cfg["path"], cfg.get("vmax", 5.0),
                    cfg["scan_shape"], embeds, assigns, polar)
        acom = ic.find_acom_arrays(run_dir)
        ctx.status("interpretation: factors + class means (cube pass)…")
        factors = ic.compute_factors_and_means(cx, collect_classical=True)
        ctx.status("interpretation: probing (R²/η²/MI)…")
        probe = ic.probe_and_signatures(cx, factors, acom=acom)
        ctx.status("interpretation: classical baselines…")
        classical = ic.classical_baselines(cx, factors)
        rep = ic.write_report(cx, probe=probe, classical=classical, acom=acom)
    except Exception as e:
        return f"ERROR during interpretation: {e!r}"
    # Summarize.
    lines = [f"Interpretation done for {sample} (run {os.path.basename(run_dir)}, "
             f"K={cx.K}, assigns from {src})."]
    try:
        rows = sorted(probe.get("rows", []),
                      key=lambda r: (r.get("probe_R2") or 0), reverse=True)
        if rows:
            top = rows[0]
            lines.append(f"Embedding most encodes '{top['factor']}' "
                         f"(probe R²={top.get('probe_R2'):.2f}, "
                         f"η²={top.get('eta2'):.2f}).")
    except Exception:
        pass
    try:
        b = classical.get("best"); bari = classical.get("best_ARI")
        bami = classical.get("methods", {}).get(b, {}).get("AMI")
        verdict = ("distinctive (not reproduced by a classical pipeline)"
                   if (bari is not None and bari < 0.8)
                   else "reproducible by a classical pipeline")
        lines.append(f"Best classical baseline: '{b}' ARI={bari}"
                     + (f", AMI={bami}" if bami is not None else "")
                     + f" → the DINO partition is {verdict}.")
    except Exception:
        pass
    if acom is None:
        lines.append("(ACOM not available for this run — orientation "
                     "cross-checks skipped; run run_acom first to include "
                     "them.)")
    lines.append(f"Full report: {rep.replace(chr(92), '/')}")
    return " ".join(lines)


def _nmf_best_k(X, X_aug, krange=(2, 11), sweep_rows=4000,
                cancel=None, on_progress=None):
    """Pick n_components by best silhouette score.  For speed the SWEEP
    fits on a random subset of positions (the final fit on all data is
    done by the caller).  Polls `cancel` between Ks.

    Returns (best_k|None, scores_dict, cancelled_bool)."""
    import numpy as np
    from sklearn.metrics import silhouette_score
    from gui_app.nmf_panel import fit_nmf
    n = X.shape[0]
    if n > sweep_rows:
        rng = np.random.default_rng(0)
        sub = rng.choice(n, sweep_rows, replace=False)
        Xs, Xa = X[sub], None      # drop augmentation in the sweep (speed)
    else:
        Xs, Xa = X, X_aug
    best_k, best_s, scores = None, -1.0, {}
    for k in range(krange[0], krange[1]):
        if cancel and cancel():
            return None, scores, True
        if on_progress:
            on_progress(f"NMF: testing K={k} (silhouette)…")
        try:
            W, _, _ = fit_nmf(Xs, Xa, k, max_iter=120)
        except Exception:
            scores[k] = float("nan"); continue
        lab = W.argmax(1)
        if len(set(lab.tolist())) < 2:
            scores[k] = float("nan"); continue
        try:
            sc = float(silhouette_score(W, lab,
                       sample_size=int(min(2000, len(lab))), random_state=0))
        except Exception:
            sc = float("nan")
        scores[k] = sc
        if sc == sc and sc > best_s:
            best_k, best_s = k, sc
    return best_k, scores, False


# Map free-text clustering-method names to the NMF panel's checkbox vars.
_NMF_METHOD_MAP = {
    "kmeans": "use_kmeans", "k-means": "use_kmeans", "km": "use_kmeans",
    "aglo": "use_aglo", "agglomerative": "use_aglo", "hierarchical": "use_aglo",
    "ward": "use_aglo", "agglo": "use_aglo",
    "hdbscan": "use_hdbscan", "hdb": "use_hdbscan", "density": "use_hdbscan",
    "fcm": "use_fcm", "fuzzy": "use_fcm", "cmeans": "use_fcm",
    "fuzzy c-means": "use_fcm", "fuzzy-c-means": "use_fcm",
}
_NMF_METHOD_KEYS = ("use_kmeans", "use_aglo", "use_hdbscan", "use_fcm")


def _apply_nmf_methods(vars_dict, methods):
    """Set the panel's clustering-method checkboxes from a free-text list
    (or 'all').  Defaults to K-means.  Returns the enabled method keys."""
    keys = set()
    if methods:
        if isinstance(methods, str):
            methods = [methods]
        for m in methods:
            mn = str(m).strip().lower()
            if mn in ("all", "everything", "all methods", "every"):
                keys.update(_NMF_METHOD_KEYS); continue
            k = _NMF_METHOD_MAP.get(mn)
            if k:
                keys.add(k)
    if not keys:
        keys = {"use_kmeans"}
    for kk in _NMF_METHOD_KEYS:
        try:
            vars_dict[kk].set(kk in keys)
        except Exception:
            pass
    return [kk for kk in _NMF_METHOD_KEYS if kk in keys]


# Map a friendly input choice to one of the NMF panel's named variants.
_NMF_VARIANT_BY_INPUT = {
    "polar": "Polar + log  (Kimoto et al. 2025)",
    "polar_theta": "Polar + θ-shift  (Krajnak & Etheridge 2020)",
    "cart": "Cartesian flat  (Spurgeon et al. 2020)",
    "radial": "1D radial  (baseline)",
}


def _run_nmf(ctx, args) -> str:
    """Drive the NMF panel so its NATIVE progress + figure show and the
    user watches the run happen on the NMF tab.  Components (n_comp) and
    clusters (K) are separate; auto-selection is OPT-IN."""
    import time
    sample, err = _active_sample(ctx, args)
    if err:
        return err
    nmf = getattr(ctx.app, "nmf", None)
    if nmf is None:
        return "ERROR: the NMF panel isn't available."
    if getattr(nmf, "_compute_running", False):
        return "NMF is already running on the NMF tab — wait or press Stop."

    inp = str(args.get("input", "polar"))
    theta = bool(args.get("theta_shift", False))
    variant = args.get("variant")
    if not variant:
        key = "polar_theta" if (inp == "polar" and theta) else inp
        variant = _NMF_VARIANT_BY_INPUT.get(key,
                                            _NMF_VARIANT_BY_INPUT["polar"])
    auto_comp = bool(args.get("auto_components", False))
    auto_clu = bool(args.get("auto_clusters", args.get("auto_k", False)))
    n_comp = int(args.get("n_components") or args.get("K") or 6)
    n_clu = int(args.get("n_clusters") or n_comp)

    def setup():
        from gui_app.nmf_panel import NMF_VARIANTS
        v = nmf._vars
        v["sample"].set(sample)
        nmf._on_sample_change()              # sets self.sample + scan_shape
        if variant in NMF_VARIANTS:
            v["variant"].set(variant)
        v["auto_n"].set(auto_comp)
        if not auto_comp:
            v["n_comp"].set(n_comp)
        v["auto_K"].set(auto_clu)
        if not auto_clu:
            v["K"].set(n_clu)
        _apply_nmf_methods(v, args.get("methods"))
        TAB_ROUTES["nmf"](ctx.app)           # show the NMF tab
        nmf._kickoff_run()                   # start the panel's own worker
        return getattr(nmf, "sample", None)
    try:
        ok = ctx.call_ui(setup, timeout=30)
    except Exception as e:
        return f"ERROR starting NMF: {e!r}"
    if not ok:
        return "ERROR: could not set the dataset on the NMF panel."

    # The panel runs its own worker + shows native progress/figure.  Wait
    # for it, mirroring progress to the chat; Stop aborts.
    last = None
    while getattr(nmf, "_compute_running", False):
        if ctx.cancel():
            try:
                ctx.call_ui(nmf._on_stop_clicked)
            except Exception:
                pass
            return "NMF stop requested (the panel finishes its current step)."
        prog = getattr(nmf, "_compute_progress", "")
        if prog and prog != last:
            ctx.status(f"NMF: {prog}")
            last = prog
        time.sleep(0.4)

    def readback():
        return (int(nmf._vars["n_comp"].get()), int(nmf._vars["K"].get()),
                getattr(nmf, "_compute_progress", ""))
    try:
        nc, kk, prog = ctx.call_ui(readback)
    except Exception:
        nc, kk, prog = n_comp, n_clu, ""
    if prog and ("fail" in prog.lower() or "error" in prog.lower()):
        return f"NMF finished with an error: {prog}"
    return (f"NMF done on {sample} (variant '{variant}'): {nc} components, "
            f"{kk} clusters. The decomposition + class map are shown on "
            f"the NMF tab.")


def _run_recluster(ctx, args) -> str:
    """Re-cluster the EXISTING NMF (change K / methods) without re-fitting
    the decomposition — drives the panel's 'Cluster' button."""
    import time
    nmf = getattr(ctx.app, "nmf", None)
    if nmf is None:
        return "ERROR: the NMF panel isn't available."
    if getattr(nmf, "_last", None) is None or nmf._last.get("W") is None:
        return ("There's no NMF to re-cluster yet — run NMF first "
                "(run_nmf), then re-clustering can change K/methods fast.")
    if getattr(nmf, "_compute_running", False):
        return "NMF is busy right now — wait or press Stop."
    n_clusters = args.get("n_clusters") or args.get("K")
    auto_clusters = bool(args.get("auto_clusters", False))
    methods = args.get("methods")

    def setup():
        v = nmf._vars
        v["auto_K"].set(auto_clusters)
        if not auto_clusters and n_clusters:
            v["K"].set(int(n_clusters))
        _apply_nmf_methods(v, methods)
        TAB_ROUTES["nmf"](ctx.app)
        nmf._kickoff_recluster()
    try:
        ctx.call_ui(setup, timeout=30)
    except Exception as e:
        return f"ERROR starting re-cluster: {e!r}"

    last = None
    while getattr(nmf, "_compute_running", False):
        if ctx.cancel():
            try:
                ctx.call_ui(nmf._on_stop_clicked)
            except Exception:
                pass
            return "re-cluster stop requested."
        prog = getattr(nmf, "_compute_progress", "")
        if prog and prog != last:
            ctx.status(f"NMF: {prog}"); last = prog
        time.sleep(0.3)

    def readback():
        d = getattr(nmf, "_last", None) or {}
        return int(nmf._vars["K"].get()), list(d.get("labels", {}).keys())
    try:
        K, ms = ctx.call_ui(readback)
    except Exception:
        K, ms = (n_clusters or 0), []
    return (f"Re-clustered the existing NMF (no re-fit): K={K}, methods="
            f"{', '.join(ms) or '(none)'}. Updated on the NMF tab.")


def _run_acom(ctx, args) -> str:
    cif = args.get("cif")
    if not cif or not os.path.exists(str(cif)):
        return ("ERROR: run_acom needs a valid CIF file path "
                "(cif='C:/path/to/structure.cif').")
    s = getattr(ctx.app, "session", None)
    sample, err = _active_sample(ctx, args)
    if err:
        return err
    run_dir = args.get("run_dir") or getattr(s, "run_dir", None)
    k_max = float(args.get("k_max", 0.35))
    inv_a = float(args.get("inv_ang_per_pixel", 0.00185))
    stride = int(args.get("subsample_stride", 4))
    try:
        import numpy as np
        from data import SAMPLES
        from gui_app.acom_core import (load_crystal, prepare_crystal,
                                       acom_full_dataset, zone_axis_from_matrix)
        from gui_app.posthoc_panel import _open_lazy
        cfg = SAMPLES[sample]
        ctx.status("ACOM: loading crystal + building orientation plan…")
        cr = load_crystal(str(cif))
        prepare_crystal(cr, k_max=k_max)
        cube = _open_lazy(cfg["path"], scan_shape=cfg["scan_shape"])
        ctx.status(f"ACOM: matching full dataset (stride={stride})…")
        omap, bv, scan_shape = acom_full_dataset(
            cr, cube, inv_ang_per_pixel=inv_a, subsample_stride=stride)
        Ny, Nx = scan_shape
        cv = np.asarray(getattr(omap, "corr", None))
        win_corr = ((cv[..., 0] if cv.ndim == 3 else cv).astype(np.float32)
                    if cv is not None and cv.size
                    else np.full((Ny, Nx), np.nan, np.float32))
        mv = np.asarray(getattr(omap, "matrix", None))
        win_rmat = np.full((Ny, Nx, 3, 3), np.nan, np.float32)
        if mv is not None and mv.size:
            win_rmat = (mv[..., 0, :, :] if mv.ndim == 5 else mv).astype(np.float32)
        finite = win_corr[np.isfinite(win_corr)]
        thr = float(np.nanmedian(finite)) if finite.size else 0.0
        matched = np.isfinite(win_corr) & (win_corr > max(thr, 1e-6))
        phase_id = np.where(matched, 0, -1).astype(np.int32)
        # Save where interpretation's find_acom_arrays will discover it.
        base = (os.path.join(run_dir, "acom", "maps") if run_dir
                else os.path.join("runs", "_gui", "_chat_acom", sample))
        os.makedirs(base, exist_ok=True)
        np.save(os.path.join(base, "mpfull_phase_id.npy"), phase_id)
        np.save(os.path.join(base, "mpfull_winning_corr.npy"), win_corr)
        np.save(os.path.join(base, "mpfull_winning_rmat.npy"), win_rmat)
    except Exception as e:
        return f"ERROR during ACOM: {e!r}"
    pct = 100.0 * matched.sum() / max(matched.size, 1)
    mean_corr = float(np.nanmean(win_corr[matched])) if matched.any() else 0.0
    msg = (f"ACOM done on {sample} (stride={stride}, k_max={k_max}, "
           f"inv_Å/px={inv_a}): {matched.sum()}/{matched.size} positions "
           f"indexed ({pct:.0f}%), mean correlation={mean_corr:.3f}. "
           f"Saved orientation arrays to {base.replace(chr(92), '/')}.")
    return msg


# ----------------------------------------------------------------------
# Teacher mode: LIVE-highlight the real control anywhere in the GUI by
# finding it from a free-text target (no screenshots, no fixed registry).
# ----------------------------------------------------------------------
# Keyword -> which tab the control lives on.  First matching group wins;
# checked in order, longest/most-specific first.
_TAB_KEYWORDS = [
    (("nmf", "non-negative", "component", "n_comp", "decomp"), "nmf"),
    (("interpret", "probing", "baseline", "ablation", "grad-cam",
      "gradcam", "what do the classes", "class mean", "signature"),
     "interpretation"),
    (("acom", "orientation", "crystal", "cif", "bragg", "zone axis",
      "phase", "peaks", "index pattern"), "acom"),
    (("train", "epoch", "recipe", "prototype", "augment", "apply changes",
      "learning rate"), "training"),
    (("eval", "class map", "render", "inference", "checkpoint"), "eval"),
    (("umap", "virtual bf", "virtual haadf", "fine-tune", "finetune",
      "centroid", "radial", "distribution", "occupancy", "overlay",
      "post-hoc", "posthoc"), "post-hoc"),
    (("vmax", "crop", "mask", "beam", "com", "blur", "binning", "index",
      "pattern", "ellip", "load", "browse", "preprocess", "pre-process",
      "contrast", "slider"), "pre-processing"),
]

# tab key -> the panel object whose subtree we search.
_TAB_PANEL = {
    "pre-processing": lambda app: getattr(app, "pre", None),
    "training": lambda app: getattr(app, "train", None),
    "eval": lambda app: getattr(app, "eval_panel", None),
    "nmf": lambda app: getattr(app, "nmf", None),
    "interpretation": lambda app: getattr(app, "interpret", None),
    "post-hoc": lambda app: getattr(app, "posthoc", None),
    "acom": lambda app: getattr(app, "acom2", None),
}

_INTERACTIVE = ("CTkButton", "CTkOptionMenu", "CTkCheckBox", "CTkSwitch",
                "CTkSlider", "CTkEntry", "CTkSegmentedButton")


def _norm(s):
    s = str(s).lower()
    for ch in ("…", "→", "▶", "/", "-", "_", "(", ")", ",", "."):
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _resolve_tab(query):
    qn = _norm(query)
    for kws, tab in _TAB_KEYWORDS:
        for kw in kws:
            if kw in qn:
                return tab
    return None


def _is_descendant(w, ancestor):
    if ancestor is None:
        return False
    cur = w
    for _ in range(60):
        if cur is None:
            return False
        if cur is ancestor:
            return True
        cur = getattr(cur, "master", None)
    return False


def _best_match(root, query, prefer=None):
    """Find the widget under `root` whose visible text best matches the
    free-text query.  Widgets inside `prefer` (the keyword-routed panel)
    get a boost.  Long help-text labels are ignored (control names are
    short).  Returns (widget, its_text, score)."""
    qn = _norm(query)
    qtok = set(qn.split())
    best = (0, None, None)
    def walk(w):
        nonlocal best
        for c in w.winfo_children():
            try:
                t = c.cget("text")
            except Exception:
                t = None
            tn = _norm(t) if t else ""
            # control labels are short; skip long help/description text.
            if tn and len(tn) <= 80:
                ttok = set(tn.split())
                sc = 0
                if qn == tn:
                    sc = 100
                elif tn in qtok:               # widget text IS one query word
                    sc = 70                      # e.g. 'Stop' for 'stop …'
                elif qtok and qtok <= ttok:    # ALL query words appear in text
                    sc = 60                      # (word-level, no substrings)
                else:
                    ov = len(qtok & ttok)
                    if ov:
                        sc = 18 * ov
                if sc:
                    if type(c).__name__ in _INTERACTIVE:
                        sc += 6
                    if prefer is not None and _is_descendant(c, prefer):
                        sc += 30                 # favour the routed tab
                    if sc > best[0]:
                        best = (sc, c, t)
            walk(c)
    walk(root)
    return best[1], best[2], best[0]


def _build_all_tabs(app):
    """Build every lazy tab/sub-tab once so ALL controls are searchable.
    Lazy panel constructors are cheap (no model loading)."""
    if getattr(app, "_chat_tabs_built", False):
        return
    # Remember where the user was so we can put them back (building lazy
    # tabs requires navigating to them).
    orig = None
    try:
        orig = app._tabs.get()
    except Exception:
        pass
    # Top-level + interpretation (these route through the app's lazy builders).
    for k in ("interpretation", "sam", "blob", "transfer", "dino-cluster",
              "acom", "nmf", "eval", "training", "pre-processing"):
        r = TAB_ROUTES.get(k)
        if r:
            try:
                r(app)
            except Exception:
                pass
    # Blob sub-tabs (Strain / Crystallinity) — built on sub-tab visit.
    try:
        bt = getattr(app, "_blob_tabs", None)
        if bt is not None:
            for nm in ("Strain", "Crystallinity", "Detect"):
                try:
                    bt.set(nm); app._on_blob_subtab()
                except Exception:
                    pass
    except Exception:
        pass
    # Synthetic sub-tab under Data ▸ Pre-processing.
    try:
        app._tabs.set("Data")
        app._group_tabs["Data"].set("Pre-processing")
        app._group_tabs["Pre-processing"].set("Synthetic")
        app._on_tab_change()
    except Exception:
        pass
    # Put the user back on the tab they were viewing.
    if orig:
        try:
            app._tabs.set(orig)
        except Exception:
            pass
    try:
        app.update_idletasks()
    except Exception:
        pass
    app._chat_tabs_built = True


def _reveal_widget(app, w):
    """Make `w` visible by walking its ancestry and switching EVERY
    enclosing CTkTabview to the tab that contains it — handles arbitrary
    tab/sub-tab nesting with no hardcoded routes."""
    import customtkinter as ctk
    chain = set()
    cur = w
    for _ in range(80):
        if cur is None:
            break
        chain.add(id(cur))
        cur = getattr(cur, "master", None)
    tvs = []
    cur = w
    for _ in range(80):
        if cur is None:
            break
        if isinstance(cur, ctk.CTkTabview):
            tvs.append(cur)
        cur = getattr(cur, "master", None)

    def _apply():
        for tv in reversed(tvs):              # outermost first
            td = getattr(tv, "_tab_dict", None) or {}
            for name, frame in td.items():
                if id(frame) in chain:
                    try:
                        tv.set(name)
                    except Exception:
                        pass
                    break
        # CTkTabview.set() doesn't fire the tabview 'command' the app uses
        # to lazy-build content → a just-selected tab can be an empty gray
        # frame.  Fire the app's build handlers (what a real click does).
        for h in ("_on_tab_change", "_on_blob_subtab", "_on_posthoc_subtab"):
            fn = getattr(app, h, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        try:
            app.update_idletasks()
        except Exception:
            pass
        _redraw_owner_canvas(app, w)

    _apply()
    # CTkTabview.set() also schedules `after(100, grid_forget_all_tabs)`
    # excluding the tab selected AT THAT MOMENT.  When several tabs were
    # switched in quick succession (e.g. during lazy-build), those stale
    # deferred callbacks fire ~100ms later and hide the tab we just
    # revealed.  Re-apply the target path after they've fired.
    try:
        app.after(260, _apply)
    except Exception:
        pass


_PANEL_ATTRS = ("pre", "train", "eval_panel", "nmf", "posthoc", "acom2",
                "sam", "blob", "strain", "crystallinity", "transfer",
                "synth", "interpret", "dino_cluster")


def _redraw_owner_canvas(app, w):
    """Redraw matplotlib canvases on the panel that owns widget `w` (or
    all panels if the owner can't be identified)."""
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    except Exception:
        return
    objs = {}
    for a in _PANEL_ATTRS:
        p = getattr(app, a, None)
        if p is not None:
            objs[id(p)] = p
    owner, cur = None, w
    for _ in range(60):
        if cur is None:
            break
        if id(cur) in objs:
            owner = objs[id(cur)]
            break
        cur = getattr(cur, "master", None)
    for p in ([owner] if owner else list(objs.values())):
        try:
            for v in list(vars(p).values()):
                if isinstance(v, FigureCanvasTkAgg):
                    try:
                        v.draw()
                    except Exception:
                        pass
        except Exception:
            pass


def _blink_widget(w, n=8):
    """Pulse a CTk widget's red border to draw the eye to it (live, in the
    real GUI — no screenshots)."""
    try:
        orig_bc = w.cget("border_color")
    except Exception:
        orig_bc = None
    try:
        orig_bw = w.cget("border_width")
    except Exception:
        orig_bw = 0
    st = {"i": 0}
    def tick():
        if not w.winfo_exists():
            return
        on = (st["i"] % 2 == 0)
        try:
            if on:
                w.configure(border_color="#ff2d2d", border_width=4)
            else:
                w.configure(border_color=orig_bc if orig_bc is not None
                            else "#ff2d2d", border_width=orig_bw or 0)
        except Exception:
            pass
        st["i"] += 1
        if st["i"] <= n:
            w.after(280, tick)
        else:
            try:
                w.configure(border_color=orig_bc if orig_bc is not None
                            else "#1f6aa5",
                            border_width=orig_bw if orig_bw is not None else 0)
            except Exception:
                pass
    tick()


def _floating_tip(app, w, caption, ms=8000):
    """Borderless 'Click here' tip placed next to a widget using Tk's own
    geometry (robust across DPI/monitors — no pixel grabbing)."""
    import tkinter as tk
    # Remove any previous tip.
    old = getattr(app, "_howto_tip", None)
    try:
        if old is not None and old.winfo_exists():
            old.destroy()
    except Exception:
        pass
    tip = tk.Toplevel(app)
    tip.overrideredirect(True)
    try:
        tip.attributes("-topmost", True)
    except Exception:
        pass
    lbl = tk.Label(tip, text="👉  Click here:\n" + caption,
                   bg="#1b1b1b", fg="#ffe070",
                   font=("Segoe UI", 11, "bold"), justify="left",
                   wraplength=300, padx=12, pady=9, bd=2, relief="solid")
    lbl.pack()
    tip.update_idletasks()
    x = w.winfo_rootx() + w.winfo_width() + 12
    y = w.winfo_rooty()
    tw = tip.winfo_reqwidth()
    sw = app.winfo_screenwidth()
    if x + tw > sw:                         # overflow → place to the left
        x = max(0, w.winfo_rootx() - tw - 12)
    tip.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    app._howto_tip = tip
    tip.after(ms, lambda: (tip.winfo_exists() and tip.destroy()))


def _widget_tab(app, w):
    """Walk a widget's ancestry to find which known panel/tab contains it."""
    panel_to_tab = {}
    for k in _TAB_PANEL:
        p = _TAB_PANEL[k](app)
        if p is not None:
            panel_to_tab[id(p)] = k
    cur = w
    for _ in range(40):
        if cur is None:
            break
        if id(cur) in panel_to_tab:
            return panel_to_tab[id(cur)]
        cur = getattr(cur, "master", None)
    return None


def _curated(app, query):
    """Hand-picked targets that the generic text match gets wrong or that
    should point at a specific control.  Returns (widget, caption) or
    (None, None).  Checked BEFORE the fuzzy matcher."""
    qn = _norm(query)
    # LOAD DATA -> the topbar dataset badge (opens a file browser).  The
    # bare 'Load' button only loads a path already typed in, so it's the
    # wrong thing to point at for "how do I load data".
    load_phrases = ("load data", "load cube", "load a cube", "load the data",
                    "open data", "open cube", "import data", "load my data",
                    "load dataset", "open a cube")
    if qn in ("load", "open", "load data", "open data") \
            or any(p in qn for p in load_phrases):
        w = getattr(app, "_badge_dataset", None)
        if w is not None and w.winfo_exists():
            return w, ("Click the dataset badge in the top bar to browse for "
                       "and load a 4D-STEM cube (.prz / _nbed.cube.npy / "
                       ".npz). You can also use Browse → Load on the "
                       "Pre-processing tab.")
    return None, None


def _do_highlight(app, w, caption):
    _reveal_widget(app, w)
    try:
        w.focus_set()
    except Exception:
        pass
    _blink_widget(w)
    _floating_tip(app, w, caption)


def _highlight_target(app, query):
    """Find the control matching `query` and live-highlight it (blink +
    tip).  Does NOT open tabs unnecessarily: curated answers and controls
    on already-built tabs are handled with no tab-building; lazy tabs are
    only built as a last resort, after which the user's tab is restored.
    Returns (ok, caption|error)."""
    orig_tab = None
    try:
        orig_tab = app._tabs.get()
    except Exception:
        pass

    # 1) curated overrides (e.g. load → topbar dataset badge) — no tabs.
    cw, ccap = _curated(app, query)
    if cw is not None and cw.winfo_exists():
        _do_highlight(app, cw, ccap)
        return True, ccap

    tab = _resolve_tab(query)

    def _prefer():
        return _TAB_PANEL.get(tab, lambda a: None)(app) if tab else None

    # 2) search ALREADY-BUILT widgets first (covers the common, eager
    #    panels — no tab building, no flicker).
    w, txt, sc = _best_match(app, query, prefer=_prefer())

    # 3) if there's no match OR only a weak one, build the lazy tabs once
    #    and retry, then keep whichever match is stronger.  (A weak early
    #    hit must not block a strong control on a lazy tab.)
    if w is None or sc < 55:
        _build_all_tabs(app)        # restores the active tab itself
        w2, txt2, sc2 = _best_match(app, query, prefer=_prefer())
        if w2 is not None and sc2 > sc:
            w, txt, sc = w2, txt2, sc2

    if w is None or sc < 18 or not w.winfo_exists():
        if orig_tab:                # leave the user where they were
            try:
                app._tabs.set(orig_tab)
            except Exception:
                pass
        return False, ("I couldn't find a matching control for "
                       f"'{query}'. Try naming the button or setting (e.g. "
                       "'vmax', 'Run', 'Train', 'class map', 'detect peaks').")
    _do_highlight(app, w, txt)
    return True, txt


def _show_me_how(ctx, args) -> str:
    target = str(args.get("target") or args.get("action") or "").strip()
    if not target:
        return ("Tell me which control or step to point you to (e.g. "
                "'vmax', 'run NMF', 'train', 'class map').")
    try:
        ok, info = ctx.call_ui(_highlight_target, ctx.app, target, timeout=30)
    except Exception as e:
        return f"ERROR creating the guide: {e!r}"
    if not ok:
        return info
    return (f"Done — I switched to the right tab and highlighted '{info}' in "
            f"the GUI (it's blinking red with a '👉 Click here' tip next to "
            f"it).")


# ----------------------------------------------------------------------
# Learned-knowledge tools (in-context learning)
# ----------------------------------------------------------------------
def _remember(ctx, args) -> str:
    import gui_app.chat_kb as kb
    text = args.get("text") or args.get("note") or args.get("fact")
    kind = str(args.get("kind") or "fact").lower()
    if not text:
        return "ERROR: provide the text to remember."
    ok = kb.add_note(kind, str(text))
    return (f"Got it — I'll remember [{kind}]: {text}" if ok
            else "Couldn't save that note.")


def _list_knowledge(ctx, args) -> str:
    import gui_app.chat_kb as kb
    notes = kb.list_notes()
    if not notes:
        return ("I haven't learned anything from you yet. Tell me facts or "
                "corrections, or ask me to learn about your setup.")
    return ("Here's what I've learned from you:\n"
            + "\n".join(f"  - [{n.get('kind', 'fact')}] {n.get('text', '')}"
                        for n in notes[-50:]))


def _forget(ctx, args) -> str:
    import gui_app.chat_kb as kb
    sub = str(args.get("about") or args.get("text") or "")
    n = kb.forget(sub)
    return (f"Removed {n} note(s) mentioning '{sub}'." if n
            else f"No saved notes mention '{sub}'.")


def _answer_from_docs(ctx, args) -> str:
    query = args.get("query") or args.get("question") or args.get("q")
    if not query:
        return "ERROR: provide a query/question."
    ctx.status("searching the project docs…")
    try:
        import gui_app.chat_rag as rag
        hits = rag.search(str(query), k=4,
                          on_progress=ctx.status, cancel=ctx.cancel)
    except Exception as e:
        return f"ERROR searching docs: {e!r}"
    if not hits:
        return ("No relevant text found in the project documents for that "
                "question.")
    parts = ["Relevant excerpts from the project documents — base your "
             "answer on these and cite the file name(s):"]
    for score, src, text in hits:
        name = src.replace("\\", "/")
        parts.append(f"\n--- {name} ---\n{text.strip()[:700]}")
    return "\n".join(parts)


def _suggest_next_step(ctx, args) -> str:
    s = getattr(ctx.app, "session", None)
    sample = getattr(s, "sample", None)
    run = getattr(s, "run_dir", None)
    has_inf = bool(getattr(s, "has_inference", lambda: False)())
    runs = _find_run_dirs()
    scored = bool(run and (
        os.path.exists(os.path.join(run, "eval", "scorecard.json")) or
        os.path.exists(os.path.join(run, "scorecard.json"))))
    out = ["Suggested next step:"]
    if not sample:
        out.append("1) Load a 4D-STEM cube — Pre-processing tab → Browse → Load "
                   "(.prz / .npz / .h5 master / .npy).")
        out.append("   Then set beam mask + crop + COM while watching the DP-max.")
    elif not run and not runs:
        out.append(f"1) Data '{sample}' is loaded. Tune mask/crop/COM, click "
                   "'Load parameters to model', then Train (Training tab).")
        out.append("   For a quick label-free map without training, run NMF + K-means.")
    elif run and not has_inf:
        out.append(f"1) Run '{os.path.basename(run)}' exists. Eval it (infer) to get "
                   "the class map + class-average patterns.")
    elif has_inf and not scored:
        out.append("1) Inference is cached → Score the run, then open Interpretation "
                   "to see WHY the classes split.")
        out.append("2) Compare with NMF (and ACOM if the sample is crystalline) to "
                   "validate the map.")
    else:
        out.append("1) You have a scored run → interpret it (Grad-CAM / ablations / "
                   "radial), compare with NMF/ACOM, and refine K or preprocessing if "
                   "the map is fragmented or collapsed.")
    out.append("(Full decision guide + validity checks: ask answer_from_docs about "
               "METHOD_GUIDE.)")
    return "\n".join(out)


def _recommend_params(ctx, args) -> str:
    stype = str(args.get("sample_type", "auto")).lower().replace("-", "_")
    goal = str(args.get("goal", "")).strip()
    common = ("Validated base recipe: polar pipeline + theta-roll aug, "
              "center_momentum=0.97, EMA 0.99->0.999, ~50 epochs, COM-centering ON, "
              "beam mask sized to the central disk, crop to the informative FOV. "
              "Set these on the loader then click 'Load parameters to model'.")
    layered = ("LAYERED / zone-axis sample (e.g. EuInAs): orientation dominates. "
               "Enable the confidence/weight loss (these show avg_conf >~0.85 by "
               "epoch ~5). K small (~6) for a phase/zone map. Cross-check with ACOM "
               "zone axis (judge by AMI vs ACOM).")
    nonlayered = ("NON-LAYERED / crystallinity sample (e.g. IMC, NaPHI): the class "
                  "parameter is crystallinity + azimuthal spottiness (2-D Bragg "
                  "excess), NOT orientation. Plain polar DINO, no weight loss needed "
                  "(avg_conf stays low, ~0.3). K~6 focused or large (~60) then merge. "
                  "Compare to NMF + classical descriptors and SAM masks "
                  "(IoU / Dice / count r).")
    out = [common, ""]
    if stype.startswith("layer"):
        out.append(layered)
    elif "non" in stype or stype in ("imc", "naphi"):
        out.append(nonlayered)
    else:
        out += ["Choose by sample type:", "- " + layered, "- " + nonlayered,
                "Decide: known crystal structure + orientation matters → treat as "
                "layered/crystalline (ACOM-comparable); contrast is "
                "crystallinity/texture → non-layered."]
    if goal:
        out.append(f"\nGoal: {goal}. Quick no-train map → NMF+K-means; "
                   "orientation/strain → ACOM (needs a CIF).")
    return "\n".join(out)


def _troubleshoot(ctx, args) -> str:
    sym = str(args.get("symptom", "")).lower()
    def has(*ks): return any(k in sym for k in ks)
    blocks = []
    if has("overclust", "over-clust", "over clust", "too many", "split",
           "indistinct", "fragmented class", "redundant class"):
        blocks.append(("OVER-CLUSTERING (one real class split into several / "
            "class-averages look alike)", [
            "DATA: strengthen invariances so nuisance variation stops spawning "
            "classes — keep COM-centering ON, size the beam mask to cover the "
            "central disk, use the polar pipeline + theta-roll aug (so rotated "
            "grains don't split), add mild Gaussian blur to suppress shot-noise "
            "splits; check crop/vmax aren't clipping real signal.",
            "MODEL: lower K (primary fix); train longer; raise center_momentum "
            "(~0.97) + teacher EMA for steadier prototypes; add a consolidation "
            "loss (centroid_lambda or cluster1d) to pull same-class together; "
            "lower conf_weight_gamma if it over-sharpens.",
            "QUICK (no retrain): merge classes in the Post-hoc panel, or "
            "re-cluster with a smaller K (NMF 'Cluster' button; or agglomerative "
            "then cut the dendrogram)."]))
    if has("collapse", "underclust", "under-clust", "too few", "one class",
           "single class", "everything one", "merged everything"):
        blocks.append(("COLLAPSE / UNDER-CLUSTERING (one class swallows most "
            "pixels)", [
            "DATA: reduce over-aggressive augmentation; make sure the beam mask "
            "isn't deleting the discriminative signal.",
            "MODEL: raise K; enable the confidence/weight loss (esp. layered / "
            "zone-axis samples, avg_conf >~0.85 at epoch ~5); train longer; "
            "lower center_momentum slightly if it over-stabilizes."]))
    if has("salt", "pepper", "incoherent", "speckle", "noisy map", "random",
           "scattered pixel"):
        blocks.append(("SALT-AND-PEPPER / spatially incoherent map", [
            "DATA: stronger preprocessing (mask / crop / COM), mild blur.",
            "MODEL: add the spatial-smoothness loss (lam_spatial); train longer; "
            "consider lowering K."]))
    if has("thickness", "beam", "brightness", "intensity", "central"):
        blocks.append(("MAP TRACKS THICKNESS / BEAM INTENSITY only", [
            "DATA: enlarge the beam mask, enable COM-centering, use log-stretch so "
            "scattered intensity isn't dwarfed by the (000) beam.",
            "MODEL: ensure the polar pipeline + masking are actually active."]))
    if has("unstable", "different each", "changes each", "not reproduc", "seed"):
        blocks.append(("UNSTABLE (map changes run-to-run)", [
            "MODEL: fix the seed; train longer; raise EMA/center_momentum; add a "
            "consolidation loss. Large run-to-run change usually means the classes "
            "aren't well separated — also revisit preprocessing/K."]))
    if not blocks:
        blocks.append(("Describe the symptom (overclustered, collapsed, "
            "salt-and-pepper, tracks-thickness, unstable)", [
            "Run score_run + open Interpretation to diagnose; see METHOD_GUIDE "
            "validity section. Then adjust preprocessing (mask/crop/COM/blur) "
            "and/or K and retrain, or merge / re-cluster post-hoc."]))
    out = []
    for title, fixes in blocks:
        out.append(title + ":")
        out += ["  - " + f for f in fixes]
    out.append("After any change: retrain (or re-cluster for NMF), then score_run "
               "and re-check spatial coherence + class-average distinctness.")
    return "\n".join(out)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
def build_registry(app) -> "dict[str, ToolSpec]":
    specs = [
        ToolSpec("get_state",
            "Report the current session state: loaded sample, current run "
            "directory, whether inference is cached, and the active "
            "pre-processing settings (vmax, crop, masks).",
            {"type": "object", "properties": {}, "required": []},
            _get_state, confirm=True,
            summary=lambda a: "read current state"),

        ToolSpec("list_runs",
            "List training runs found under runs/, with their sample, "
            "whether a best checkpoint exists, and whether inference is "
            "cached.",
            {"type": "object", "properties": {}, "required": []},
            _list_runs, confirm=True,
            summary=lambda a: "list training runs"),

        ToolSpec("open_tab",
            "Switch the GUI to a named tab so the user can see a panel. "
            "Valid names: pre-processing, training, eval, post-hoc, "
            "interpretation, nmf, dino-cluster, sam, blob, acom, transfer.",
            {"type": "object",
             "properties": {"name": {"type": "string",
                "description": "tab name (see list)"}},
             "required": ["name"]},
            _open_tab, confirm=True,
            summary=lambda a: f"open the '{a.get('name','?')}' tab"),

        ToolSpec("load_data",
            "Load a 4D-STEM cube by FILE PATH (e.g. a *_nbed.cube.npy / "
            ".prz / .npz). This becomes the active dataset for all "
            "downstream steps. Optionally override vmax / center_crop_size "
            "/ center_mask_radius / polar_mask_cols.",
            {"type": "object",
             "properties": {
                "path": {"type": "string",
                    "description": "absolute path to the 4D-STEM cube file"},
                "vmax": {"type": "number"},
                "center_crop_size": {"type": "integer"},
                "center_mask_radius": {"type": "integer"},
                "polar_mask_cols": {"type": "integer"}},
             "required": ["path"]},
            _load_data, confirm=True,
            summary=lambda a: f"load data from "
                f"{os.path.basename(str(a.get('path','?')))}"),

        ToolSpec("set_preproc",
            "Change pre-processing knobs on the loaded dataset: vmax "
            "(display/contrast), center_crop_size (field of view), "
            "polar_mask_cols (low-radius beam mask), center_mask_radius, "
            "com_centering (bool). Only pass the fields you want to change.",
            {"type": "object",
             "properties": {
                "vmax": {"type": "number"},
                "center_crop_size": {"type": "integer"},
                "polar_mask_cols": {"type": "integer"},
                "center_mask_radius": {"type": "integer"},
                "com_centering": {"type": "boolean"}},
             "required": []},
            _set_preproc, confirm=True,
            summary=lambda a: "change pre-processing: "
                + ", ".join(f"{k}={v}" for k, v in a.items())),

        ToolSpec("show_class_map",
            "Render the class map (cluster assignment per scan position) "
            "for a run directory on the Eval tab. If run_dir is omitted, "
            "uses the current run. Optionally pass sample.",
            {"type": "object",
             "properties": {
                "run_dir": {"type": "string"}},
             "required": []},
            _show_class_map, confirm=True,
            summary=lambda a: "render the class map"
                + (f" for {a['run_dir']}" if a.get("run_dir") else "")),

        ToolSpec("remember",
            "Persist something the user taught you so you DON'T repeat "
            "mistakes and honour their preferences in future turns. Call "
            "this whenever the user corrects you, states a domain fact, or "
            "gives a preference. kind ∈ {fact, correction, preference, "
            "example}.",
            {"type": "object",
             "properties": {
                "text": {"type": "string",
                    "description": "the fact/correction/preference to store"},
                "kind": {"type": "string",
                    "enum": ["fact", "correction", "preference", "example"]}},
             "required": ["text"]},
            _remember, confirm=False,
            summary=lambda a: "remember a note"),

        ToolSpec("answer_from_docs",
            "Look up an answer in the project's OWN documents (manuscript, "
            "Methods, per-material notes, reports) and return the most "
            "relevant excerpts. Use this for factual/domain questions "
            "(e.g. what a term means, methods, results) so you answer from "
            "the user's writing instead of guessing. Cite the file name(s).",
            {"type": "object",
             "properties": {
                "query": {"type": "string",
                    "description": "the question to look up in the docs"}},
             "required": ["query"]},
            _answer_from_docs, confirm=False,
            summary=lambda a: f"search docs for "
                f"'{str(a.get('query', ''))[:40]}'"),

        ToolSpec("list_knowledge",
            "Show everything the user has taught you so far (the learned "
            "knowledge store).",
            {"type": "object", "properties": {}, "required": []},
            _list_knowledge, confirm=False,
            summary=lambda a: "list learned knowledge"),

        ToolSpec("forget",
            "Remove learned notes that mention a given phrase (use when the "
            "user says to forget or fix something previously taught).",
            {"type": "object",
             "properties": {
                "about": {"type": "string",
                    "description": "phrase contained in the note(s) to remove"}},
             "required": ["about"]},
            _forget, confirm=False,
            summary=lambda a: f"forget notes about '{a.get('about','?')}'"),

        ToolSpec("show_me_how",
            "TEACH MODE: point the user to WHERE a control is in the GUI by "
            "switching to its tab and live-highlighting the real button or "
            "field (it blinks red with a '👉 Click here' tip). Works for any "
            "labelled control in the app. Call this whenever the user wants "
            "to FIND, LOCATE, or learn to do something THEMSELVES in the "
            "interface — e.g. 'where do I change vmax?', 'how do I train?', "
            "'which button runs NMF?', 'I can't find the crop setting'. "
            "(If instead they want YOU to do it, use the action tool.) "
            "'target' is a free-text name of the control/action.",
            {"type": "object",
             "properties": {
                "target": {"type": "string",
                    "description": "the control/setting/action to highlight "
                    "in plain words (e.g. 'vmax', 'run NMF', 'train', 'load "
                    "data', 'class map', 'COM center', 'interpretation')"}},
             "required": ["target"]},
            _show_me_how, confirm=True,
            summary=lambda a: "highlight '"
                + str(a.get('target', a.get('action', '?'))) + "' in the GUI"),

        ToolSpec("show_pattern",
            "Display one raw diffraction pattern from the loaded cube (by "
            "scan-position index) in the Pre-processing preview, using the "
            "current vmax/crop. Use this to let the user eyeball the data.",
            {"type": "object",
             "properties": {
                "index": {"type": "integer",
                    "description": "scan-position index (0-based)"}},
             "required": []},
            _show_pattern, confirm=True,
            summary=lambda a: f"show diffraction pattern "
                f"#{a.get('index', 0)}"),

        ToolSpec("infer",
            "Run model inference on a trained run: load its best/latest "
            "checkpoint, classify every scan position, write "
            "eval/inference.npz, and cache the result. Reports the number "
            "of active classes, class sizes, and mean confidence. Runs on "
            "GPU and takes up to a couple of minutes.",
            {"type": "object",
             "properties": {
                "run_dir": {"type": "string"}},
             "required": []},
            _infer, confirm=True,
            summary=lambda a: "run inference"
                + (f" on {a['run_dir']}" if a.get("run_dir") else "")),

        ToolSpec("class_average",
            "Compute the confidence-weighted average diffraction pattern "
            "for one class of a run and save it as a PNG. Useful to see "
            "what a class 'looks like'. Uses cached inference if available, "
            "else runs inference first.",
            {"type": "object",
             "properties": {
                "class_id": {"type": "integer",
                    "description": "class index (0-based)"},
                "run_dir": {"type": "string"}},
             "required": ["class_id"]},
            _class_average, confirm=True,
            summary=lambda a: f"average diffraction of class "
                f"{a.get('class_id','?')}"),

        ToolSpec("train",
            "Start a new DINO training run for a configured sample, reusing "
            "the Training tab's machinery (runs as a separate GPU "
            "subprocess; returns immediately with the output dir). Optional "
            "cfg_overrides is a dict of training params to override (e.g. "
            "{'K':6,'epochs':30,'lr':0.0003}); unknown keys are ignored. "
            "This is expensive and long-running.",
            {"type": "object",
             "properties": {
                "sample": {"type": "string",
                    "description": "configured dataset key"},
                "cfg_overrides": {"type": "object",
                    "description": "optional training-param overrides"}},
             "required": ["sample"]},
            _train, confirm=True,
            summary=lambda a: f"START TRAINING on {a.get('sample','?')}"
                + (f" with overrides {a.get('cfg_overrides')}"
                   if a.get("cfg_overrides") else "")),

        ToolSpec("score_run",
            "Run the validity scorecard for a sample (verdict "
            "APPROVE/RETUNE/FAIL + overall score + weakest component). Only "
            "works for samples in the scorecard's approved config (the "
            "paper datasets); pass run_dir to score that run's best.pth.",
            {"type": "object",
             "properties": {
                "sample": {"type": "string"},
                "label": {"type": "string"},
                "run_dir": {"type": "string"}},
             "required": ["sample"]},
            _score_run, confirm=True,
            summary=lambda a: f"score run for {a.get('sample','?')}"),

        ToolSpec("run_interpretation",
            "Run the 'what do the classes mean' battery for a run: a "
            "full-cube factor pass (scattered intensity, crystallinity, "
            "spottiness), embedding probing (R²/η²/MI), and classical "
            "baselines (whether a classical pipeline reproduces the DINO "
            "partition). Writes a report and figures, returns a summary. "
            "Several minutes; blocks the chat while it runs.",
            {"type": "object",
             "properties": {
                "run_dir": {"type": "string"}},
             "required": []},
            _run_interpretation, confirm=True,
            summary=lambda a: "run the interpretation battery"
                + (f" on {a['run_dir']}" if a.get("run_dir") else "")),

        ToolSpec("run_nmf",
            "Run NMF on the loaded data ON THE NMF TAB (the panel shows its "
            "own progress bar + figure). Components and clusters are "
            "separate: n_components = NMF rank (default 6), n_clusters = how "
            "many clusters to group the loadings into (default = "
            "n_components). Auto-selection is OPT-IN: set auto_components "
            "(knee) or auto_clusters (silhouette) only if the user wants "
            "automatic. input ∈ {polar, cart, radial}.",
            {"type": "object",
             "properties": {
                "n_components": {"type": "integer",
                    "description": "NMF rank / number of components (default 6)"},
                "n_clusters": {"type": "integer",
                    "description": "clusters to group loadings into "
                    "(default = n_components)"},
                "auto_components": {"type": "boolean",
                    "description": "auto-pick n_components (knee); opt-in"},
                "auto_clusters": {"type": "boolean",
                    "description": "auto-pick clusters by silhouette; opt-in"},
                "methods": {"type": "array", "items": {"type": "string"},
                    "description": "clustering method(s) to enable: any of "
                    "kmeans, aglo, hdbscan, fcm — or 'all'. Default kmeans."},
                "input": {"type": "string",
                    "enum": ["polar", "cart", "radial"]},
                "theta_shift": {"type": "boolean"}},
             "required": []},
            _run_nmf, confirm=True,
            summary=lambda a: "RUN NMF (fit + cluster) on the NMF tab ("
                + ("auto components" if a.get("auto_components")
                   else f"{a.get('n_components', 6)} components")
                + (", auto clusters" if a.get("auto_clusters")
                   else (f", {a['n_clusters']} clusters"
                         if a.get("n_clusters") else "")) + ")"),

        ToolSpec("recluster_nmf",
            "Re-cluster the EXISTING NMF result WITHOUT re-fitting the "
            "decomposition (fast) — use this when an NMF has already been "
            "run and the user only wants to change the number of clusters K "
            "or the clustering method(s). Drives the panel's 'Cluster' "
            "button. methods: any of kmeans/aglo/hdbscan/fcm or 'all'.",
            {"type": "object",
             "properties": {
                "n_clusters": {"type": "integer",
                    "description": "number of clusters K"},
                "auto_clusters": {"type": "boolean",
                    "description": "pick K by silhouette instead"},
                "methods": {"type": "array", "items": {"type": "string"},
                    "description": "clustering method(s): kmeans/aglo/"
                    "hdbscan/fcm or 'all'"}},
             "required": []},
            _run_recluster, confirm=True,
            summary=lambda a: "re-cluster the existing NMF ("
                + (f"K={a['n_clusters']}" if a.get("n_clusters")
                   else "auto K" if a.get("auto_clusters") else "current K")
                + (f", methods={a['methods']}" if a.get("methods") else "")
                + ")"),

        ToolSpec("run_acom",
            "Run classical ACOM (orientation/phase mapping via py4DSTEM) "
            "over the full dataset for a sample, given a crystal structure "
            "CIF file. Saves orientation arrays (so interpretation can use "
            "them) and reports the indexed fraction and mean correlation. "
            "Heavy; use subsample_stride to speed up.",
            {"type": "object",
             "properties": {
                "cif": {"type": "string",
                    "description": "path to a .cif crystal structure file"},
                "run_dir": {"type": "string"},
                "k_max": {"type": "number",
                    "description": "max scattering vector (Å⁻¹), default 0.35"},
                "inv_ang_per_pixel": {"type": "number",
                    "description": "reciprocal calibration; default 0.00185 — "
                    "OVERRIDE for non-115mm camera lengths"},
                "subsample_stride": {"type": "integer",
                    "description": "scan-grid downsample factor (default 4)"}},
             "required": ["cif"]},
            _run_acom, confirm=True,
            summary=lambda a: f"run full-dataset ACOM on "
                f"{a.get('sample','current sample')} with CIF "
                f"{os.path.basename(str(a.get('cif','?')))}"),

        ToolSpec("suggest_next_step",
            "Recommend the next action in the GUI workflow based on the current "
            "state (data loaded? run trained? inference cached? scored?). Use when "
            "the user asks 'what should I do', 'what next', or how to proceed.",
            {"type": "object", "properties": {}, "required": []},
            _suggest_next_step, confirm=False,
            summary=lambda a: "suggest the next step"),

        ToolSpec("recommend_params",
            "Recommend which method to use plus preprocessing/training parameters, "
            "tailored to the sample type. Use when the user asks which "
            "algorithm/method or what parameters they need. "
            "sample_type: 'layered' | 'non_layered' | 'auto'.",
            {"type": "object",
             "properties": {
                "sample_type": {"type": "string",
                    "description": "layered | non_layered | auto"},
                "goal": {"type": "string",
                    "description": "what the user wants to achieve (optional)"}},
             "required": []},
            _recommend_params, confirm=False,
            summary=lambda a: "recommend method/params"),

        ToolSpec("troubleshoot",
            "Diagnose a problem the user reports about their result and give "
            "concrete data + model fixes. Use when they say the model/map is "
            "wrong — e.g. over-clustered, collapsed/under-clustered, "
            "salt-and-pepper, tracks thickness only, or unstable. Pass their "
            "words as 'symptom'.",
            {"type": "object",
             "properties": {
                "symptom": {"type": "string",
                    "description": "the problem in the user's words "
                    "(e.g. 'overclustered', 'one class took everything')"}},
             "required": ["symptom"]},
            _troubleshoot, confirm=False,
            summary=lambda a: "troubleshoot the result"),
    ]
    return {s.name: s for s in specs}


def parse_text_tool_calls(text, registry) -> list:
    """Fallback: small local models sometimes EMIT a tool call as JSON in
    their message text (```json {"name":...,"arguments":{...}} ```) instead
    of using the function-calling interface.  Extract any such calls whose
    name matches a real tool, so the loop can still execute them (always
    behind a confirm dialog, since this is heuristic).

    Returns a list of {id, name, arguments} dicts (possibly empty).
    """
    import json as _json
    if not text:
        return []
    # Extract every top-level balanced {...} / [...] region (handles
    # nested arguments and ```json fences alike).
    candidates = []
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start:i + 1])
                    start = None
    out = []
    for c in candidates:
        try:
            obj = _json.loads(c)
        except Exception:
            continue
        items = obj if isinstance(obj, list) else [obj]
        for it in items:
            if not isinstance(it, dict):
                continue
            name = it.get("name") or it.get("tool") or it.get("function")
            if isinstance(name, dict):           # OpenAI-ish {function:{name}}
                args = name.get("arguments")
                name = name.get("name")
            else:
                args = (it.get("arguments") or it.get("parameters")
                        or it.get("args") or {})
            if name in registry:
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except Exception:
                        args = {}
                out.append({"id": f"text_{len(out)}", "name": name,
                            "arguments": args if isinstance(args, dict) else {}})
    return out


def tool_schemas(registry) -> list:
    return [{
        "type": "function",
        "function": {"name": s.name,
                     "description": s.description,
                     "parameters": s.parameters},
    } for s in registry.values()]
