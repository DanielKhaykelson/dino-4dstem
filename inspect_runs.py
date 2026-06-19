"""inspect_runs.py -- generate a self-contained HTML index of all runs.

Walks `runs/` (or any directory passed via --root) and emits a single
HTML file (default: `runs/_inspect.html`) that lets you browse every
training run in one page:

  * sidebar with a tree of runs grouped by parent directory
  * per-run accordion that lazily renders:
      - the cfg dict from run_summary.json
      - the loss curves from training_log.csv (Chart.js)
      - thumbnails of every PNG / JPG / PDF figure in the run dir
        and its `eval/` subdir
      - links to ckpt files / log files
  * search box that filters runs by name in real time

Run:
    python inspect_runs.py
    # then open D:\\DINOSR\\Claude\\PaperRun_claude\\dino_sr_contrastive\\runs\\_inspect.html

JSON / CSV are inlined as JS data so the HTML works from a `file://`
URL without any local server. Image references are RELATIVE paths
from the HTML's location, which is why the HTML lives at the runs/
root by default.
"""
from __future__ import annotations
import os
import sys
import json
import csv
import argparse
import html
from datetime import datetime


def _is_run_dir(d: str) -> bool:
    """A 'run dir' has any of: run_summary.json, training_log.csv,
    _train_kwargs.json (GUI-launched runs)."""
    for fn in ("run_summary.json", "training_log.csv",
                  "_train_kwargs.json"):
        if os.path.isfile(os.path.join(d, fn)):
            return True
    return False


def _list_run_dirs(root: str):
    """Yield every directory under `root` that looks like a run dir."""
    for dirpath, dirnames, _fns in os.walk(root):
        if _is_run_dir(dirpath):
            yield dirpath
            # don't descend further: nested runs would be peers, not children
            dirnames[:] = []


def _read_summary(d: str):
    """Prefer run_summary.json (manual runs); fall back to
    _train_kwargs.json (GUI-launched runs).  When both exist, merge —
    _train_kwargs contributes a 'gui_train_kwargs' sub-tree so the
    user can see exactly what the GUI passed."""
    out = None
    p = os.path.join(d, "run_summary.json")
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                out = json.load(f)
        except Exception as e:
            out = {"_error": f"could not read {p}: {e}"}
    p2 = os.path.join(d, "_train_kwargs.json")
    if os.path.isfile(p2):
        try:
            with open(p2, "r", encoding="utf-8") as f:
                tk = json.load(f)
            if out is None:
                out = {"gui_train_kwargs": tk}
            else:
                out.setdefault("gui_train_kwargs", tk)
        except Exception as e:
            (out or {}).setdefault(
                "_error_train_kwargs",
                f"could not read {p2}: {e}")
    return out


def _read_training_log(d: str):
    """Read training_log.csv -> list of {col: value} dicts. Returns
    None if missing. Numeric cells are converted to float; strings
    pass through."""
    p = os.path.join(d, "training_log.csv")
    if not os.path.isfile(p): return None
    try:
        with open(p, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    out = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            try: rr[k] = float(v)
            except (ValueError, TypeError): rr[k] = v
        out.append(rr)
    return out


_FIG_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".webp")


def _list_figures(d: str, root: str):
    """Return list of (rel_path_from_root, ext, parent_subdir) for every
    figure/file we want to surface in the HTML."""
    items = []
    for sub in (".", "eval", "transfer", "layer_splits", "gradcam",
                  "class_quality", "report/png", "nmf", "nmf_report",
                  "dino_cluster", "blob", "sam", "strain",
                  "fluct", "symm", "ordering", "radial_1d_advanced",
                  "nmf_class_averages"):
        full = os.path.join(d, sub) if sub != "." else d
        if not os.path.isdir(full):
            continue
        try:
            for fn in sorted(os.listdir(full)):
                fpath = os.path.join(full, fn)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext in _FIG_EXTS:
                    rel = os.path.relpath(fpath, root).replace("\\", "/")
                    items.append((rel, ext, sub))
        except OSError:
            continue
    return items


def _list_other_files(d: str, root: str):
    """Notable non-figure files: checkpoints, logs, summaries."""
    interesting = (".pth", ".log", ".csv", ".json", ".md", ".txt", ".npz")
    items = []
    for sub in (".", "eval", "transfer"):
        full = os.path.join(d, sub) if sub != "." else d
        if not os.path.isdir(full): continue
        try:
            for fn in sorted(os.listdir(full)):
                fpath = os.path.join(full, fn)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext in interesting:
                    sz = os.path.getsize(fpath)
                    rel = os.path.relpath(fpath, root).replace("\\", "/")
                    items.append({
                        "rel":  rel,
                        "name": fn,
                        "ext":  ext,
                        "size": sz,
                        "sub":  sub,
                    })
        except OSError:
            continue
    return items


def _human_size(b):
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"


def gather(root: str):
    """Crawl `root` and return the full data payload for the HTML."""
    runs = []
    for d in sorted(_list_run_dirs(root)):
        rel = os.path.relpath(d, root).replace("\\", "/")
        summary = _read_summary(d)
        log = _read_training_log(d)
        figs = _list_figures(d, root)
        files = _list_other_files(d, root)
        runs.append({
            "rel":      rel,
            "name":     os.path.basename(d),
            "parent":   os.path.dirname(rel) or "(root)",
            "summary":  summary,
            "log":      log,
            "figures":  [{"rel": r, "ext": e, "sub": s}
                         for (r, e, s) in figs],
            "files":    files,
            "n_log":    len(log) if log else 0,
            "n_figs":   len(figs),
            "n_files":  len(files),
        })
    return runs


# ------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DINO-SR runs inspector</title>
<style>
:root {
  --bg: #fafafa;
  --fg: #222;
  --muted: #888;
  --card: #fff;
  --border: #ddd;
  --accent: #1f77b4;
  --code-bg: #f4f4f4;
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
  background: var(--bg); color: var(--fg); display: flex;
  height: 100vh; overflow: hidden;
}
#sidebar {
  width: 320px; min-width: 240px; background: #fff;
  border-right: 1px solid var(--border); overflow-y: auto;
  padding: 12px;
}
#sidebar h1 { font-size: 14px; margin: 0 0 8px 0; }
#sidebar .meta { font-size: 11px; color: var(--muted); margin-bottom: 8px; }
#search {
  width: 100%; padding: 6px 8px; border: 1px solid var(--border);
  border-radius: 4px; font-size: 13px;
}
#tree {
  font-size: 12px; margin-top: 8px; padding-left: 0; list-style: none;
}
#tree li.group {
  font-weight: 600; color: var(--muted); margin-top: 6px;
  text-transform: uppercase; font-size: 10px; letter-spacing: .04em;
}
#tree li.run {
  padding: 3px 6px; border-radius: 3px; cursor: pointer;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
#tree li.run:hover { background: #eef; }
#tree li.run.active { background: var(--accent); color: white; }
#tree li.run .badges { float: right; font-size: 10px; opacity: .7; }
#main {
  flex: 1; overflow-y: auto; padding: 18px 24px;
}
.run-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 6px; padding: 14px 18px; margin-bottom: 16px;
  scroll-margin-top: 12px;
}
.run-card h2 {
  margin: 0 0 4px 0; font-size: 18px;
}
.run-card .path {
  font-family: Consolas, monospace; font-size: 11px; color: var(--muted);
  margin-bottom: 12px;
}
.run-card details {
  margin: 8px 0; border-top: 1px solid var(--border); padding-top: 8px;
}
.run-card details summary {
  cursor: pointer; font-weight: 600; font-size: 13px; user-select: none;
}
.run-card details summary::marker { color: var(--accent); }
.cfg-grid {
  display: grid; grid-template-columns: max-content 1fr;
  gap: 2px 14px; font-size: 12px;
  font-family: Consolas, monospace; margin-top: 6px;
  max-height: 360px; overflow-y: auto;
}
.cfg-grid .k { color: var(--muted); padding-right: 8px; }
.cfg-grid .v { word-break: break-all; }
.figures {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 10px; margin-top: 8px;
}
.figures figure {
  margin: 0; background: #fff; border: 1px solid var(--border);
  border-radius: 4px; padding: 4px;
}
.figures figure img {
  width: 100%; height: auto; display: block;
}
.figures figure figcaption {
  font-size: 11px; color: var(--muted); padding: 4px 4px 0;
  word-break: break-all;
}
.files-table {
  width: 100%; font-size: 11px; font-family: Consolas, monospace;
  border-collapse: collapse;
}
.files-table th, .files-table td {
  text-align: left; padding: 2px 8px; border-bottom: 1px solid var(--border);
}
.files-table .size { text-align: right; color: var(--muted); }
canvas.loss-canvas { width: 100% !important; max-height: 300px; }
.no-data { color: var(--muted); font-style: italic; font-size: 12px; }
.section-empty { color: var(--muted); font-size: 12px; padding: 4px 0; }
.bug-banner {
  background: #ffe4b5; padding: 6px 10px; border-radius: 4px;
  font-size: 12px; margin-bottom: 12px;
}
</style>
</head>
<body>

<aside id="sidebar">
  <h1>DINO-SR runs</h1>
  <div class="meta">
    root: <code id="meta-root">__ROOT__</code><br>
    runs: <span id="meta-count">__COUNT__</span><br>
    generated: <span id="meta-time">__TIME__</span>
  </div>
  <input id="search" placeholder="filter runs by name…" autocomplete="off">
  <ul id="tree"></ul>
</aside>

<main id="main">
  <p id="hint" class="no-data">click a run on the left to expand it,
  or scroll through them all below.</p>
  <div id="cards"></div>
</main>

<!-- Chart.js (CDN). If you're offline, replace with a local copy. -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

<script>
const RUNS = __RUNS_JSON__;

// ----- Sidebar tree -----
function buildTree() {
  const tree = document.getElementById('tree');
  const groups = {};
  for (const r of RUNS) {
    const g = r.parent || '(root)';
    (groups[g] ??= []).push(r);
  }
  const orderedGroups = Object.keys(groups).sort();
  for (const g of orderedGroups) {
    const li = document.createElement('li');
    li.className = 'group';
    li.textContent = g;
    tree.appendChild(li);
    for (const r of groups[g]) {
      const ri = document.createElement('li');
      ri.className = 'run';
      ri.dataset.rel = r.rel;
      const badges = [];
      if (r.n_log)   badges.push(`${r.n_log}ep`);
      if (r.n_figs)  badges.push(`${r.n_figs}fig`);
      ri.innerHTML = `<span class="name">${escapeHtml(r.name)}</span>` +
                     `<span class="badges">${badges.join(' · ')}</span>`;
      ri.addEventListener('click', () => scrollToRun(r.rel));
      tree.appendChild(ri);
    }
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
}

function scrollToRun(rel) {
  const el = document.getElementById('run-' + cssId(rel));
  if (el) {
    el.scrollIntoView({behavior: 'smooth', block: 'start'});
    document.querySelectorAll('#tree li.run').forEach(
      x => x.classList.toggle('active', x.dataset.rel === rel));
    // expand all details
    el.querySelectorAll('details').forEach(d => d.open = true);
    // render charts
    const c = el.querySelector('canvas.loss-canvas');
    if (c && !c.dataset.rendered) renderLossChart(c);
  }
}

function cssId(s) { return s.replace(/[^a-zA-Z0-9_-]+/g, '_'); }

// ----- Per-run cards -----
function buildCards() {
  const cards = document.getElementById('cards');
  for (const r of RUNS) {
    cards.appendChild(buildCard(r));
  }
}

function buildCard(r) {
  const div = document.createElement('div');
  div.className = 'run-card';
  div.id = 'run-' + cssId(r.rel);
  div.dataset.rel = r.rel;
  div.dataset.name = (r.name + ' ' + r.rel).toLowerCase();

  const sample = (r.summary && r.summary.sample) || '';
  const ts     = (r.summary && r.summary.timestamp) || '';

  div.innerHTML = `
    <h2>${escapeHtml(r.name)}</h2>
    <div class="path">${escapeHtml(r.rel)} · ${escapeHtml(sample)}
      ${ts ? '· ' + escapeHtml(ts) : ''}</div>
    <details>
      <summary>config (${r.summary ? 'run_summary.json' : 'no summary'})</summary>
      <div class="cfg-grid">${renderCfg(r.summary)}</div>
    </details>
    <details>
      <summary>loss curves (${r.n_log} epoch${r.n_log === 1 ? '' : 's'})</summary>
      ${r.n_log ? '<canvas class="loss-canvas"></canvas>' :
                  '<div class="section-empty">no training_log.csv</div>'}
    </details>
    <details>
      <summary>figures (${r.n_figs})</summary>
      ${renderFigures(r)}
    </details>
    <details>
      <summary>files (${r.n_files})</summary>
      ${renderFilesTable(r)}
    </details>
  `;
  // store the row data for later chart rendering
  if (r.log) div._logData = r.log;
  return div;
}

function renderCfg(summary) {
  if (!summary) return '<div class="no-data">no run_summary.json</div>';
  const cfg = summary.cfg || {};
  const meta = {
    sample:    summary.sample,
    timestamp: summary.timestamp,
    outdir:    summary.outdir,
  };
  const all = {...meta, ...cfg};
  const rows = [];
  for (const k of Object.keys(all)) {
    const v = all[k];
    let vs;
    if (v === null || v === undefined) vs = String(v);
    else if (typeof v === 'object') vs = JSON.stringify(v);
    else vs = String(v);
    rows.push(
      `<div class="k">${escapeHtml(k)}</div>` +
      `<div class="v">${escapeHtml(vs)}</div>`);
  }
  if (summary.timing) {
    rows.push('<div class="k" style="margin-top:8px">timing</div>' +
              '<div class="v" style="margin-top:8px">' +
              escapeHtml(JSON.stringify(summary.timing)) + '</div>');
  }
  return rows.join('');
}

function renderFigures(r) {
  if (!r.figures.length) return '<div class="section-empty">no figures</div>';
  const items = r.figures.map(f => {
    const isPdf = f.ext === '.pdf';
    if (isPdf) {
      return `<figure><a href="${escapeHtml(f.rel)}" target="_blank">
              [open PDF: ${escapeHtml(f.rel.split('/').pop())}]</a>
              <figcaption>${escapeHtml(f.sub)}/${escapeHtml(f.rel.split('/').pop())}</figcaption>
              </figure>`;
    }
    return `<figure>
            <a href="${escapeHtml(f.rel)}" target="_blank">
              <img src="${escapeHtml(f.rel)}" loading="lazy" alt="">
            </a>
            <figcaption>${escapeHtml(f.sub)}/${escapeHtml(f.rel.split('/').pop())}</figcaption>
            </figure>`;
  });
  return `<div class="figures">${items.join('')}</div>`;
}

function renderFilesTable(r) {
  if (!r.files.length) return '<div class="section-empty">no files</div>';
  const rows = r.files.map(f =>
    `<tr><td>${escapeHtml(f.sub)}/</td>
         <td><a href="${escapeHtml(f.rel)}" target="_blank">${escapeHtml(f.name)}</a></td>
         <td class="size">${escapeHtml(humanSize(f.size))}</td></tr>`);
  return `<table class="files-table">
          <thead><tr><th>dir</th><th>file</th><th>size</th></tr></thead>
          <tbody>${rows.join('')}</tbody>
          </table>`;
}

function humanSize(b) {
  const u = ['B','KB','MB','GB','TB']; let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return b.toFixed(1) + ' ' + u[i];
}

// ----- Loss chart -----
function renderLossChart(canvas) {
  const card = canvas.closest('.run-card');
  const data = card._logData;
  if (!data || !data.length) return;
  // pick numeric loss-ish columns
  const candidates = [
    {key:'avg_loss',                   label:'total',         color:'#000'},
    {key:'avg_loss_dino',              label:'L_DINO',        color:'#1f77b4'},
    {key:'avg_loss_supcon',            label:'L_supcon',      color:'#2ca02c'},
    {key:'avg_loss_cluster1d_intra',   label:'L_1d intra',    color:'#d62728'},
    {key:'avg_loss_cluster1d_inter',   label:'L_1d inter',    color:'#9467bd'},
    {key:'avg_loss_centroid_intra',    label:'L_cen intra',   color:'#8c564b'},
    {key:'avg_loss_centroid_inter',    label:'L_cen inter',   color:'#e377c2'},
    {key:'avg_loss_repel',             label:'L_repel',       color:'#7f7f7f'},
    {key:'avg_loss_pair',              label:'L_pair',        color:'#bcbd22'},
  ];
  const epochs = data.map(r => r.epoch);
  const datasets = [];
  for (const c of candidates) {
    if (!data[0] || !(c.key in data[0])) continue;
    const vals = data.map(r => Number(r[c.key]));
    if (vals.every(v => v === 0 || !Number.isFinite(v))) continue;
    datasets.push({
      label: c.label, data: vals, borderColor: c.color,
      backgroundColor: c.color, tension: 0.2, pointRadius: 2,
      borderWidth: c.key === 'avg_loss' ? 1.8 : 1.0,
    });
  }
  new Chart(canvas, {
    type: 'line',
    data: {labels: epochs, datasets},
    options: {
      animation: false,
      plugins: {legend: {position: 'bottom', labels: {boxWidth: 12,
                                                        font: {size: 10}}}},
      scales: {y: {beginAtZero: false}, x: {title: {display: true,
                                                       text: 'epoch'}}}
    }
  });
  canvas.dataset.rendered = '1';
}

// Render charts when their <details> is opened
document.addEventListener('toggle', e => {
  if (e.target.tagName !== 'DETAILS') return;
  if (!e.target.open) return;
  const c = e.target.querySelector('canvas.loss-canvas');
  if (c && !c.dataset.rendered) renderLossChart(c);
}, true);

// ----- Search -----
function applySearch() {
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('#tree li.run').forEach(li => {
    li.style.display = li.dataset.rel.toLowerCase().includes(q)
                          ? '' : 'none';
  });
  document.querySelectorAll('.run-card').forEach(c => {
    c.style.display = c.dataset.name.includes(q) ? '' : 'none';
  });
}

document.addEventListener('DOMContentLoaded', () => {
  buildTree();
  buildCards();
  document.getElementById('search').addEventListener('input', applySearch);
  document.getElementById('hint').remove();
});
</script>
</body>
</html>
"""


def emit_html(runs, root: str, out_path: str):
    payload = json.dumps(runs, default=str)
    html_str = (HTML_TEMPLATE
                  .replace("__ROOT__", html.escape(root))
                  .replace("__COUNT__", str(len(runs)))
                  .replace("__TIME__", datetime.now()
                                          .strftime("%Y-%m-%d %H:%M:%S"))
                  .replace("__RUNS_JSON__", payload))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "runs"),
        help="root directory to crawl. Default: ./runs/")
    ap.add_argument(
        "--out",
        default=None,
        help="output HTML path. Default: <root>/_inspect.html")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"[inspect_runs] root {root!r} is not a directory.",
              file=sys.stderr)
        sys.exit(1)

    out = args.out or os.path.join(root, "_inspect.html")

    print(f"[inspect_runs] crawling {root} …")
    runs = gather(root)
    print(f"[inspect_runs] found {len(runs)} run dirs")
    emit_html(runs, root, out)
    print(f"[inspect_runs] wrote {out}")
    print(f"[inspect_runs] open in browser: file:///{out.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
