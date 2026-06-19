"""render_existing_classmaps.py -- retroactive class-map renderer.

Walks any sweep root (or a single sample subdir) produced by
``sweep_m_K.py`` and generates, for every completed run:

  <run_outdir>/class_map.png        per-run image (same format as the
                                       live sweep writes now)

It also writes per-sample browse pages:

  <root>/<sample>/run_gallery.html  grid of all class maps with
                                       (m, K, seed, K_eff) labels

Usage
-----
    python tools/render_existing_classmaps.py --root <sweep_root>
    python tools/render_existing_classmaps.py --root <sweep_root> --force
        (re-render even if class_map.png already exists)

The script registers each sample's cube into ``SAMPLES`` first so the
scan shape is known.  Runs that lack ``eval/inference.npz`` are
skipped with a printed message (incomplete training).
"""
from __future__ import annotations
import argparse, html, os, sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path: sys.path.insert(0, REPO)
if HERE not in sys.path: sys.path.insert(0, HERE)

# Reuse the live-sweep helpers so the rendered images are
# byte-identical to what a fresh run would have produced.
from sweep_m_K import (                                  # noqa: E402
    _render_run_classmap, _extract_metrics,
    _register_sample,
    SAMPLES, SAMPLE_SPECS,
)


def _parse_run_name(name: str) -> "tuple[float, int, int] | None":
    """e.g. ``m0.5000_seed42_K30`` -> (0.5, 42, 30).  Returns None if
    the name doesn't match the sweep's naming convention."""
    try:
        parts = name.split("_")
        m = float(parts[0].replace("m", ""))
        seed = int(parts[1].replace("seed", ""))
        K = int(parts[2].replace("K", ""))
        return m, seed, K
    except Exception:
        return None


def _write_gallery_html(sample_dir: str, sample: str, rows: list):
    """rows: list of dicts {stage, m, K, seed, png_rel, K_eff_smooth,
       n_live, avg_conf_e5, loss_final, ok, error}.
    """
    out_path = os.path.join(sample_dir, "run_gallery.html")
    stamp = datetime.now().isoformat(timespec="seconds")
    sb = []
    sb.append("<!doctype html><html><head><meta charset=utf-8>")
    sb.append(f"<title>{html.escape(sample)} -- run gallery</title>")
    sb.append("<style>"
              "body{font-family:Georgia,serif;max-width:1400px;"
              "margin:24px auto;padding:0 16px;color:#222;line-height:1.5}"
              "h1{font-size:22px;border-bottom:2px solid #444;padding-bottom:6px}"
              "h2{font-size:16px;margin-top:24px;border-bottom:1px solid #ccc}"
              ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));"
              "gap:14px;margin:12px 0}"
              ".card{border:1px solid #ccc;padding:6px;background:#fafafa}"
              ".card img{width:100%;display:block;background:#fff;border:1px solid #ddd}"
              ".card .meta{font-size:11px;color:#333;margin-top:4px;line-height:1.35;"
              "font-family:Consolas,monospace}"
              ".muted{color:#666;font-size:13px}"
              ".bad{border-color:#a33;background:#fbecec}"
              "</style></head><body>")
    sb.append(f"<h1>{html.escape(sample)} -- per-run class maps</h1>")
    sb.append(f"<p class=muted>Generated {stamp}.  "
              f"{len([r for r in rows if r.get('ok')])} successful runs "
              f"of {len(rows)} total.</p>")

    by_stage = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(r)
    for stage in sorted(by_stage):
        sb.append(f"<h2>{html.escape(stage)}</h2>")
        sb.append("<div class=grid>")
        # Sort: first by m ascending, then K, then seed.
        runs = sorted(by_stage[stage],
                      key=lambda r: (r["m"], r["K"], r["seed"]))
        for r in runs:
            cls = "card" + ("" if r.get("ok") else " bad")
            sb.append(f'<div class="{cls}">')
            if r.get("png_rel"):
                sb.append(f'<img src="{html.escape(r["png_rel"])}" '
                          f'alt="class map" />')
            else:
                sb.append('<div style="height:200px;display:flex;'
                          'align-items:center;justify-content:center;'
                          'color:#888;background:#eee">(no class map)</div>')
            meta = (f"m={r['m']:g}&nbsp;&nbsp;K={r['K']}&nbsp;&nbsp;seed={r['seed']}<br>"
                    f"K_eff={r.get('K_eff_end_smooth', float('nan')):.2f}"
                    f"&nbsp;&nbsp;n_live={r.get('n_live_end', 0)}<br>"
                    f"avg_conf<sub>5</sub>={r.get('avg_conf_e5', float('nan')):.2f}"
                    f"&nbsp;&nbsp;loss={r.get('loss_final', float('nan')):.3f}")
            if not r.get("ok"):
                err = (r.get("error") or "(failed)")[:120]
                meta += f"<br><span style='color:#a33'>error: {html.escape(err)}</span>"
            sb.append(f'<div class=meta>{meta}</div>')
            sb.append('</div>')
        sb.append('</div>')
    sb.append('</body></html>')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sb))
    print(f"[gallery] {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Sweep root (e.g. runs/_sweep_m_K_<ts>) or "
                         "single sample dir.")
    ap.add_argument("--force", action="store_true",
                    help="Re-render even if class_map.png already exists.")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")

    # If the root is itself a <sample>/ dir, handle that single sample.
    samples_to_walk = []
    base = os.path.basename(root)
    if base in SAMPLE_SPECS:
        samples_to_walk = [(base, root)]
    else:
        for entry in sorted(os.listdir(root)):
            if entry in SAMPLE_SPECS:
                samples_to_walk.append((entry, os.path.join(root, entry)))

    if not samples_to_walk:
        sys.exit(f"no recognized sample subdirs in {root}.  "
                 f"Expected one of {list(SAMPLE_SPECS.keys())}.")

    total_rendered = total_skipped = total_failed = 0
    for sample, sample_dir in samples_to_walk:
        print(f"\n[walk] {sample} :: {sample_dir}", flush=True)
        # Register sample once -- needed for SAMPLES[sample]['scan_shape'].
        try:
            _register_sample(sample)
        except Exception as e:
            print(f"[skip-sample] {sample}: register failed: {e!r}",
                  flush=True)
            continue
        gallery_rows = []
        for stage in ("stage1", "stage2"):
            stage_dir = os.path.join(sample_dir, stage)
            if not os.path.isdir(stage_dir):
                continue
            for run_name in sorted(os.listdir(stage_dir)):
                run_dir = os.path.join(stage_dir, run_name)
                if not os.path.isdir(run_dir):
                    continue
                parsed = _parse_run_name(run_name)
                if parsed is None:
                    print(f"[skip-bad-name] {run_dir}", flush=True)
                    continue
                m, seed, K = parsed
                inf = os.path.join(run_dir, "eval", "inference.npz")
                cm_png = os.path.join(run_dir, "class_map.png")
                row = {"stage": stage, "m": m, "K": K, "seed": seed,
                       "png_rel": None, "ok": False, "error": ""}
                if not os.path.exists(inf):
                    row["error"] = "no inference.npz (run incomplete)"
                    gallery_rows.append(row)
                    print(f"[no-inf]  {run_dir}", flush=True)
                    total_failed += 1
                    continue
                metrics = _extract_metrics(run_dir, sample)
                row.update({k: metrics.get(k) for k in (
                    "K_eff_end_smooth", "n_live_end",
                    "avg_conf_e5", "loss_final", "has_nan")})
                if os.path.exists(cm_png) and not args.force:
                    row["png_rel"] = f"{stage}/{run_name}/class_map.png"
                    row["ok"] = True
                    gallery_rows.append(row)
                    print(f"[skip-existing] {cm_png}", flush=True)
                    total_skipped += 1
                    continue
                try:
                    _render_run_classmap(run_dir, sample, m, K, seed,
                                            metrics)
                    row["png_rel"] = f"{stage}/{run_name}/class_map.png"
                    row["ok"] = True
                    total_rendered += 1
                    print(f"[ok] {cm_png}", flush=True)
                except Exception as e:
                    row["error"] = repr(e)[:200]
                    total_failed += 1
                    print(f"[fail] {run_dir}: {e!r}", flush=True)
                gallery_rows.append(row)
        if gallery_rows:
            _write_gallery_html(sample_dir, sample, gallery_rows)

    print(f"\n[done] rendered={total_rendered}  "
          f"skipped={total_skipped}  failed={total_failed}",
          flush=True)


if __name__ == "__main__":
    main()
