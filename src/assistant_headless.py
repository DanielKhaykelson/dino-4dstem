"""assistant_headless.py -- run the DINO-4DSTEM assistant WITHOUT the GUI.

Same plain-English assistant + local Ollama model, but no Tk window: it
computes analyses directly from the core code and SAVES every figure to
disk, next to the data, in sequential per-analysis folders
(<data>_assistant/<analysis>_NNN/).

Usage:
    python assistant_headless.py "D:/path/to/cube.prz"
        -> interactive terminal chat (type plain English; 'exit' to quit)
    python assistant_headless.py "D:/path/to/cube.prz" --task "run nmf with 6 components, then interpret"
        -> run that task once, save figures, exit
    optional:  --model qwen2.5:14b
"""
from __future__ import annotations
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import numpy as np

from assistant_io import new_output_dir, save_fig, save_array_png, write_summary
from gui_app.chat_backends import OllamaBackend, BackendError
from gui_app import chat_tools as ct
from gui_app.chat_tools import ToolSpec, ToolContext

MAX_TOOL_ROUNDS = 8


# ----------------------------------------------------------------------
# Minimal session/app + tool context (no Tk)
# ----------------------------------------------------------------------
class _Session:
    def __init__(self):
        self.sample = None
        self.run_dir = None
        self.inference = None

    def has_inference(self):
        return self.inference is not None

    def set(self, *, sample=None, run_dir=None, inference=Ellipsis):
        if sample is not None:
            self.sample = sample
        if run_dir is not None:
            self.run_dir = run_dir
        if inference is not Ellipsis:
            self.inference = inference


class _HeadlessApp:
    def __init__(self):
        self.session = _Session()


def _data_path_for(app):
    from data import SAMPLES
    s = app.session.sample
    if s and s in SAMPLES:
        return SAMPLES[s].get("path")
    return None


# ----------------------------------------------------------------------
# Headless analysis tools (compute + save figures)
# ----------------------------------------------------------------------
def _h_load_data(ctx, args):
    from data import register_runtime_sample
    path = args.get("path") or args.get("name_or_path")
    if not path or not os.path.exists(str(path)):
        return f"ERROR: no file at {path!r}"
    kw = {}
    if args.get("vmax") is not None:
        kw["vmax"] = float(args["vmax"])
    if args.get("scan_shape"):
        try:
            kw["scan_shape"] = tuple(int(x) for x in args["scan_shape"])
        except Exception:
            pass
    key = register_runtime_sample(str(path), **kw)
    ctx.app.session.set(sample=key)
    from data import SAMPLES
    cfg = SAMPLES[key]
    return (f"Loaded {os.path.basename(str(path))} as '{key}': "
            f"scan={cfg.get('scan_shape')}, vmax={cfg.get('vmax')}.")


def _component_grid(H, comp_shape):
    """matplotlib Figure: NMF components reshaped to comp_shape."""
    import matplotlib.pyplot as plt
    n = H.shape[0]
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        comp = H[i].reshape(comp_shape)
        if comp.shape[0] == 1:                      # 1-D radial
            ax.plot(comp.ravel())
        else:
            ax.imshow(comp, cmap="inferno", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"component {i}", fontsize=9)
    fig.suptitle("NMF components")
    fig.tight_layout()
    return fig


def _h_run_nmf(ctx, args):
    from data import SAMPLES
    from gui_app.nmf_panel import (build_nmf_input, fit_nmf, cluster_W,
                                   auto_n_comp, auto_K)
    s = ctx.app.session.sample
    if not s or s not in SAMPLES:
        return "ERROR: no data loaded — use load_data first."
    cfg = SAMPLES[s]
    inp = str(args.get("input", "polar"))
    variant_cfg = {"input": inp, "log": bool(args.get("log", True)),
                   "theta_shift": bool(args.get("theta_shift",
                                                inp == "polar")),
                   "sparse": False, "variant": "headless"}
    ctx.status("NMF: building input (cube pass)…")
    X, X_aug, comp_shape, info = build_nmf_input(s, variant_cfg)
    # components
    if args.get("auto_components"):
        ctx.status("NMF: auto n_components (knee)…")
        n_comp, _ = auto_n_comp(X, X_aug)
    else:
        n_comp = int(args.get("n_components") or 6)
    ctx.status(f"NMF: fitting {n_comp} components…")
    W, H, err = fit_nmf(X, X_aug, n_comp, max_iter=300)
    # clusters
    if args.get("auto_clusters"):
        ctx.status("NMF: auto K (silhouette)…")
        K, _sil = auto_K(W)
    else:
        K = int(args.get("n_clusters") or n_comp)
    methods = args.get("methods") or ["kmeans"]
    if isinstance(methods, str):
        methods = [methods]
    name_map = {"kmeans": ("K-means", dict(method="K-means", k=K)),
                "aglo": ("Aglo", dict(method="Aglo", k=K)),
                "hdbscan": ("HDBSCAN", dict(method="HDBSCAN")),
                "fcm": ("FCM", dict(method="FCM", k=K))}
    if any(str(m).lower() in ("all", "every") for m in methods):
        methods = list(name_map)
    labels = {}
    for m in methods:
        key = str(m).lower().replace("k-means", "kmeans")
        if key not in name_map:
            continue
        disp, kw = name_map[key]
        ctx.status(f"NMF: clustering {disp}…")
        labels[disp] = cluster_W(W, **kw)
    # save
    out = new_output_dir(cfg["path"], "nmf")
    Ny, Nx = cfg["scan_shape"]
    cmap = "tab20" if K > 10 else "tab10"
    for disp, lab in labels.items():
        if np.asarray(lab).size == Ny * Nx:
            save_array_png(np.asarray(lab).reshape(Ny, Nx), out,
                           f"classmap_{disp}", cmap=cmap)
    try:
        save_fig(_component_grid(H, comp_shape), out, "nmf_components")
    except Exception:
        pass
    np.savez_compressed(os.path.join(out, "nmf_result.npz"),
                        W=W, H=H, **{f"labels_{d}": np.asarray(l)
                                     for d, l in labels.items()})
    write_summary(out, (f"NMF on {s}\ninput={inp} log={variant_cfg['log']} "
                        f"theta_shift={variant_cfg['theta_shift']}\n"
                        f"n_components={n_comp}\nK={K}\n"
                        f"methods={list(labels)}\n"
                        f"reconstruction_err={err:.2f}\n"
                        f"positions={X.shape[0]} features={X.shape[1]}\n"))
    return (f"NMF done on {s}: {n_comp} components, {K} clusters, methods="
            f"{list(labels)}. Figures saved to {out.replace(chr(92), '/')}.")


def _h_infer(ctx, args):
    from data import SAMPLES
    s = ctx.app.session
    run_dir = args.get("run_dir") or s.run_dir
    if not run_dir or not os.path.isdir(run_dir):
        return "ERROR: need a valid run_dir (a trained run)."
    sample = (args.get("sample") or s.sample
              or ct._sample_of_run(run_dir))
    if not sample or sample not in SAMPLES:
        return "ERROR: could not determine a configured sample."
    ctx.status(f"inference on {sample}…")
    inf, scan_shape, label = ct._run_inference(run_dir, sample)
    s.set(run_dir=run_dir, sample=sample,
          inference=dict(soft_probs=inf["soft_probs"],
                         assigns=inf["assigns"], embeds=inf["embeds"]))
    out = new_output_dir(SAMPLES[sample]["path"], "classmap")
    K = int(inf["soft_probs"].shape[1])
    Ny, Nx = scan_shape
    save_array_png(inf["assigns"].reshape(Ny, Nx), out, "class_map",
                   cmap="tab20" if K > 10 else "tab10")
    write_summary(out, f"Inference on {sample} (run {os.path.basename(run_dir)})"
                       f"\nK_active={K}\nN={inf['assigns'].size}\n")
    return (f"Inference done: K_active={K}, class map saved to "
            f"{out.replace(chr(92), '/')}.")


def _h_class_average(ctx, args):
    from data import SAMPLES, LoadPRZ
    s = ctx.app.session
    run_dir = args.get("run_dir") or s.run_dir
    cls = args.get("class_id")
    if cls is None:
        return "ERROR: provide class_id."
    if not run_dir:
        return "ERROR: need run_dir."
    sample = args.get("sample") or s.sample or ct._sample_of_run(run_dir)
    assigns, soft, src = ct._load_or_infer(run_dir, sample)
    cls = int(cls); K = int(soft.shape[1])
    if cls < 0 or cls >= K:
        return f"ERROR: class_id {cls} out of range 0..{K-1}."
    idx = np.where(assigns == cls)[0]
    if idx.size == 0:
        return f"class {cls} is empty."
    cfg = SAMPLES[sample]
    ds = LoadPRZ(cfg["path"], resize=192, vmax=cfg["vmax"])
    top = idx[np.argsort(-soft[idx, cls])[:min(200, idx.size)]]
    w = soft[top, cls].astype(np.float32)
    pats = np.stack([ds.get_raw(int(i)) for i in top], 0).astype(np.float32)
    avg = (pats * w[:, None, None]).sum(0) / (w.sum() + 1e-12)
    out = new_output_dir(cfg["path"], "class_average")
    save_array_png(avg, out, f"class_avg_{cls}", cmap="inferno",
                   vmax=float(cfg["vmax"]))
    return (f"Class {cls}: {idx.size} members; average saved to "
            f"{out.replace(chr(92), '/')}.")


def _h_run_interpretation(ctx, args):
    from data import SAMPLES
    import gui_app.interpret_core as ic
    import shutil
    s = ctx.app.session
    run_dir = args.get("run_dir") or s.run_dir
    if not run_dir or not os.path.isdir(run_dir):
        return "ERROR: need a valid run_dir."
    sample = args.get("sample") or s.sample or ct._sample_of_run(run_dir)
    if not sample or sample not in SAMPLES:
        return "ERROR: could not determine a configured sample."
    cfg = SAMPLES[sample]
    assigns, soft, embeds, _src = ct._load_or_infer_full(run_dir, sample)
    polar = ct._polar_cfg_from_run(run_dir)
    cx = ic.Ctx(run_dir, sample, cfg["path"], cfg.get("vmax", 5.0),
                cfg["scan_shape"], embeds, assigns, polar)
    acom = ic.find_acom_arrays(run_dir)
    ctx.status("interpretation: factors + class means…")
    factors = ic.compute_factors_and_means(cx, collect_classical=True)
    ctx.status("interpretation: probing…")
    probe = ic.probe_and_signatures(cx, factors, acom=acom)
    ctx.status("interpretation: classical baselines…")
    classical = ic.classical_baselines(cx, factors)
    rep = ic.write_report(cx, probe=probe, classical=classical, acom=acom)
    # copy the figures interpret_core wrote into a fresh assistant folder
    out = new_output_dir(cfg["path"], "interpretation")
    interp_dir = os.path.join(run_dir, "_interpretability")
    copied = 0
    if os.path.isdir(interp_dir):
        for f in os.listdir(interp_dir):
            if f.lower().endswith((".png", ".md")):
                try:
                    shutil.copy2(os.path.join(interp_dir, f),
                                 os.path.join(out, f)); copied += 1
                except Exception:
                    pass
    best = classical.get("best"); bari = classical.get("best_ARI")
    return (f"Interpretation done for {sample}: best classical baseline "
            f"'{best}' ARI={bari}. {copied} figure(s) + report saved to "
            f"{out.replace(chr(92), '/')}.")


def _h_run_acom(ctx, args):
    from data import SAMPLES
    from gui_app.acom_core import (load_crystal, prepare_crystal,
                                   acom_full_dataset)
    from gui_app.posthoc_panel import _open_lazy
    cif = args.get("cif")
    if not cif or not os.path.exists(str(cif)):
        return "ERROR: run_acom needs a valid cif path."
    sample = args.get("sample") or ctx.app.session.sample
    if not sample or sample not in SAMPLES:
        return "ERROR: no configured sample loaded."
    cfg = SAMPLES[sample]
    k_max = float(args.get("k_max", 0.35))
    inv_a = float(args.get("inv_ang_per_pixel", 0.00185))
    stride = int(args.get("subsample_stride", 4))
    ctx.status("ACOM: building plan…")
    cr = load_crystal(str(cif)); prepare_crystal(cr, k_max=k_max)
    cube = _open_lazy(cfg["path"], scan_shape=cfg["scan_shape"])
    ctx.status(f"ACOM: matching (stride={stride})…")
    omap, bv, scan_shape = acom_full_dataset(
        cr, cube, inv_ang_per_pixel=inv_a, subsample_stride=stride)
    Ny, Nx = scan_shape
    cv = np.asarray(getattr(omap, "corr", None))
    win_corr = ((cv[..., 0] if cv.ndim == 3 else cv).astype(np.float32)
                if cv is not None and cv.size else
                np.zeros((Ny, Nx), np.float32))
    out = new_output_dir(cfg["path"], "acom")
    np.save(os.path.join(out, "winning_corr.npy"), win_corr)
    save_array_png(np.nan_to_num(win_corr), out, "acom_correlation",
                   cmap="viridis")
    matched = np.isfinite(win_corr) & (win_corr > np.nanmedian(
        win_corr[np.isfinite(win_corr)] if np.isfinite(win_corr).any()
        else [0]))
    pct = 100.0 * matched.sum() / max(matched.size, 1)
    write_summary(out, f"ACOM on {sample}\ncif={cif}\nstride={stride} "
                       f"k_max={k_max} inv_ang/px={inv_a}\nindexed={pct:.0f}%\n")
    return (f"ACOM done on {sample}: {pct:.0f}% indexed; maps saved to "
            f"{out.replace(chr(92), '/')}.")


def _h_score_run(ctx, args):
    return ct._score_run(ctx, args)


def _h_get_state(ctx, args):
    s = ctx.app.session
    from data import SAMPLES
    cfg = SAMPLES.get(s.sample, {}) if s.sample else {}
    return (f"sample={s.sample}\nrun_dir={s.run_dir}\n"
            f"inference_cached={s.has_inference()}\n"
            f"data_path={cfg.get('path')}\nscan_shape={cfg.get('scan_shape')}")


def _h_answer_from_docs(ctx, args):
    return ct._answer_from_docs(ctx, args)


def build_headless_registry():
    P = ct.build_registry(None)              # reuse JSON schemas
    H = {
        "load_data": _h_load_data,
        "run_nmf": _h_run_nmf,
        "infer": _h_infer,
        "class_average": _h_class_average,
        "run_interpretation": _h_run_interpretation,
        "run_acom": _h_run_acom,
        "score_run": _h_score_run,
        "get_state": _h_get_state,
        "answer_from_docs": _h_answer_from_docs,
        "list_runs": lambda ctx, a: ct._list_runs(ctx, a),
        "recommend_params": lambda ctx, a: ct._recommend_params(ctx, a),
        "suggest_next_step": lambda ctx, a: ct._suggest_next_step(ctx, a),
    }
    reg = {}
    for name, fn in H.items():
        if name in P:
            sp = P[name]
            reg[name] = ToolSpec(sp.name, sp.description, sp.parameters,
                                 fn, confirm=False, summary=sp.summary)
        else:
            reg[name] = ToolSpec(name, name, {"type": "object",
                                 "properties": {}, "required": []}, fn,
                                 confirm=False)
    return reg


# ----------------------------------------------------------------------
# Agent loop + CLI
# ----------------------------------------------------------------------
_SYS = (
    "You are the headless DINO-4DSTEM analysis assistant (no GUI). You run "
    "analyses by calling tools; each saves its figures to disk next to the "
    "data. There are no named samples — data is loaded by file path. Be "
    "concise. Tools: load_data, run_nmf, infer, class_average, "
    "run_interpretation, run_acom, score_run, get_state, list_runs, "
    "answer_from_docs, recommend_params, suggest_next_step. For method/"
    "parameter advice call recommend_params; you are a 4D-STEM + ML expert. "
    "For NMF, multiple methods: methods=['kmeans','aglo',"
    "'hdbscan','fcm'] or 'all'. NMF is model-free; infer/interpretation/"
    "acom need a trained run_dir.")


def run_turn(backend, registry, ctx, history, user_text):
    history.append({"role": "user", "content": user_text})
    convo = [{"role": "system", "content": _SYS}] + history
    schemas = ct.tool_schemas(registry)
    for _ in range(MAX_TOOL_ROUNDS):
        if ctx.cancel():
            break
        result = backend.chat(convo, tools=schemas,
                              on_token=lambda t: print(t, end="", flush=True),
                              cancel=ctx.cancel)
        print()
        text = result.get("text", "")
        calls = result.get("tool_calls") or []
        if not calls and text:
            calls = ct.parse_text_tool_calls(text, registry)
        amsg = {"role": "assistant", "content": text}
        if calls:
            amsg["tool_calls"] = [{"function": {"name": c["name"],
                                   "arguments": c["arguments"]}} for c in calls]
        convo.append(amsg)
        if not calls:
            if text.strip():
                history.append({"role": "assistant", "content": text})
            return
        for c in calls:
            name = c["name"]; a = c.get("arguments") or {}
            print(f"  • {name}({a})", flush=True)
            spec = registry.get(name)
            if spec is None:
                out = f"ERROR: unknown tool '{name}'."
            else:
                try:
                    out = str(spec.fn(ctx, a))
                except Exception as e:
                    out = f"ERROR running {name}: {e!r}"
            print(f"    ↳ {out}", flush=True)
            convo.append({"role": "tool", "name": name, "content": out})


def main():
    ap = argparse.ArgumentParser(description="Headless DINO-4DSTEM assistant")
    ap.add_argument("data", nargs="?", help="cube file to load on start")
    ap.add_argument("--task", help="run one task and exit")
    ap.add_argument("--model", default="qwen2.5:7b")
    args = ap.parse_args()

    app = _HeadlessApp()
    cancel = [False]
    ctx = ToolContext(app=app,
                      call_ui=lambda fn, *a, **k: fn(*a),
                      post=lambda role, t: print(t, flush=True),
                      status=lambda t: print("   …", t, flush=True),
                      cancel=lambda: cancel[0])

    backend = OllamaBackend(model=args.model)
    print("Preparing local model…", flush=True)
    ok, msg = backend.provision(on_progress=lambda t: print("   " + t,
                                                            flush=True))
    if not ok:
        print(msg); return
    print(f"Model ready: {backend.active_model}\n", flush=True)

    history = []
    if args.data:
        print(_h_load_data(ctx, {"path": args.data}), flush=True)

    if args.task:
        run_turn(backend, build_headless_registry(), ctx, history, args.task)
        return

    print("Headless assistant ready. Type a request ('exit' to quit).\n")
    registry = build_headless_registry()
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break
        try:
            run_turn(backend, registry, ctx, history, line)
        except Exception as e:
            print(f"[error] {e!r}", flush=True)


if __name__ == "__main__":
    main()
