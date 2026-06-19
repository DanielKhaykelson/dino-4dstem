"""gui_line_labeler.py -- manually label each (sample, prototype) pair
as Lines / NoLines / Partial(%) for a per-family transfer run.

Inspired by prz_labeler.py.

Workflow:
    1. Browse to a source folder, e.g.
        D:\\...\\runs\\_per_family_v5\\NaPHI_combined_K8_30ep
    2. The GUI walks <source>/transfer/<sample>/eval/ for each sample,
        reads inference.npz to get per-class counts, and shows each
        class-average PNG (eval/class_averages/p{c}.png) plus a strip of
        6 random example patterns from class_examples_200/p{c}/.
    3. For each (sample, prototype): click Lines / NoLines / Partial.
        For Partial, type the approximate % of frames that are lines.
    4. As you go, the right-hand panel updates per-sample counts.
    5. Press "Done & Export CSV" or finish the last item and the GUI
        writes:
            <source>/line_labels.json     (raw labels)
            <source>/line_labels_per_class.csv
            <source>/line_labels_summary.csv

Resume: relaunching with the same source folder reloads the JSON if
present so you can pick up where you left off. Each label is autosaved.

Keyboard shortcuts: L = Lines, N = NoLines, P = Partial (focus % entry).
Right arrow = Next, Left arrow = Prev, Enter = confirm Partial %.
"""
from __future__ import annotations
import os, sys, json, csv, glob, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def _list_samples(transfer_root):
    if not os.path.isdir(transfer_root):
        return []
    out = []
    for name in sorted(os.listdir(transfer_root)):
        d = os.path.join(transfer_root, name)
        if (os.path.isdir(d)
                and os.path.exists(os.path.join(d, "eval", "inference.npz"))):
            out.append(name)
    return out


def _class_count_and_avg(sample_dir, c):
    inf_path = os.path.join(sample_dir, "eval", "inference.npz")
    inf = np.load(inf_path)
    assigns = inf["assigns"]
    K = int(np.max(assigns)) + 1 if assigns.size else 0
    if c >= K:
        return None, None, K
    count = int((assigns == c).sum())
    total = int(assigns.size)
    avg_path = os.path.join(sample_dir, "eval", "class_averages", f"p{c}.png")
    return count, total, K if K else c + 1


class LineLabelerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Line phase labeler")
        self.geometry("1240x780")
        self.source = ""
        self.samples = []          # list[str]
        self.K = 0
        self.tasks = []            # list[(sample, c)]
        self.cursor = 0
        self.labels = {}           # {(sample, c): {"label": ..., "pct": ..., "count": int, "total": int}}
        self._build_ui()

    # ---- UI ----
    def _build_ui(self):
        tb = tk.Frame(self, bd=1, relief=tk.RIDGE, pady=3)
        tb.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(tb, text="Source folder:").pack(side=tk.LEFT)
        self._src_var = tk.StringVar()
        tk.Entry(tb, textvariable=self._src_var, width=70).pack(side=tk.LEFT, padx=2)
        tk.Button(tb, text="Browse", command=self._browse).pack(side=tk.LEFT)
        tk.Button(tb, text="Load", command=self._load_source).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(tb, text="Done & Export CSV",
                   command=self._done_and_export).pack(side=tk.RIGHT, padx=4)

        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # left: class average + examples + buttons
        left = tk.Frame(body, bd=1, relief=tk.GROOVE)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        self._title_label = tk.Label(left,
            text="Browse a source folder...", font=("Arial", 11, "bold"))
        self._title_label.pack(pady=4)

        # main class avg
        avg_frame = tk.Frame(left)
        avg_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        fig_avg = Figure(figsize=(4.6, 4.6), dpi=95)
        self._ax_avg = fig_avg.add_subplot(111)
        self._ax_avg.axis('off')
        fig_avg.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._fig_avg = fig_avg
        self._canvas_avg = FigureCanvasTkAgg(fig_avg, master=avg_frame)
        self._canvas_avg.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close('all')

        # 10 example patterns (2 rows of 5)
        ex_frame = tk.Frame(left)
        ex_frame.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(ex_frame, text="10 example patterns:",
                  font=("Arial", 9, "italic")).pack(anchor='w')
        thumbs = tk.Frame(ex_frame)
        thumbs.pack(fill=tk.X)
        self._ex_axes = []
        for i in range(10):
            r, c = divmod(i, 5)
            f = Figure(figsize=(1.4, 1.4), dpi=80)
            ax = f.add_subplot(111); ax.axis('off')
            f.subplots_adjust(left=0, right=1, top=1, bottom=0)
            cv = FigureCanvasTkAgg(f, master=thumbs)
            cv.get_tk_widget().grid(row=r, column=c, padx=1, pady=1)
            self._ex_axes.append((f, ax, cv))
        plt.close('all')

        # decision buttons
        decisions = tk.Frame(left, pady=4)
        decisions.pack(fill=tk.X, padx=4)
        self._btn_lines = tk.Button(decisions, text="Lines (L)", width=12,
                                      bg="#9DD09D",
                                      command=lambda: self._label("lines"))
        self._btn_lines.pack(side=tk.LEFT, padx=2)
        self._btn_nolines = tk.Button(decisions, text="NoLines (N)", width=12,
                                        bg="#E69191",
                                        command=lambda: self._label("nolines"))
        self._btn_nolines.pack(side=tk.LEFT, padx=2)

        # Partial: quick-pick buttons (10/30/50/70/90) + manual entry
        partial_row = tk.Frame(left, pady=2)
        partial_row.pack(fill=tk.X, padx=4)
        tk.Label(partial_row, text="Partial %:",
                  font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        self._partial_pct = tk.StringVar(value="50")
        for q in (10, 30, 50, 70, 90):
            tk.Button(partial_row, text=f"{q}%", width=4, bg="#E0D080",
                       command=lambda v=q: self._label_partial_pct(v)
                       ).pack(side=tk.LEFT, padx=1)
        tk.Label(partial_row, text="  custom:").pack(side=tk.LEFT,
                                                       padx=(8, 2))
        e = tk.Entry(partial_row, textvariable=self._partial_pct, width=5)
        e.pack(side=tk.LEFT)
        e.bind('<Return>', lambda _e: self._label("partial"))
        tk.Button(partial_row, text="Apply (P)", width=10, bg="#E0D080",
                   command=lambda: self._label("partial")
                   ).pack(side=tk.LEFT, padx=2)

        # nav
        nav = tk.Frame(left, pady=4); nav.pack(fill=tk.X, padx=4)
        tk.Button(nav, text="< Prev", command=self._prev).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="Skip", command=self._skip).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="Next >", command=self._next).pack(side=tk.LEFT, padx=2)
        self._progress_label = tk.Label(nav, text="(no source loaded)")
        self._progress_label.pack(side=tk.LEFT, padx=10)

        # right: per-sample summary
        right = tk.Frame(body, bd=1, relief=tk.GROOVE, width=480)
        right.pack(side=tk.LEFT, fill=tk.BOTH)
        right.pack_propagate(False)
        tk.Label(right, text="Per-sample running summary",
                  font=("Arial", 10, "bold")).pack(pady=4)
        self._summary_text = tk.Text(right, font=("Consolas", 9),
                                       wrap=tk.NONE)
        self._summary_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # keyboard shortcuts
        self.bind_all('l', lambda _e: self._label("lines"))
        self.bind_all('L', lambda _e: self._label("lines"))
        self.bind_all('n', lambda _e: self._label("nolines"))
        self.bind_all('N', lambda _e: self._label("nolines"))
        self.bind_all('p', lambda _e: self._label("partial"))
        self.bind_all('P', lambda _e: self._label("partial"))
        self.bind_all('<Right>', lambda _e: self._next())
        self.bind_all('<Left>',  lambda _e: self._prev())

    # ---- file ----
    def _browse(self):
        d = filedialog.askdirectory(title="Pick source-model folder "
                                      "(contains transfer/<sample>/eval/)")
        if d:
            self._src_var.set(d)
            self._load_source()

    def _load_source(self):
        src = self._src_var.get().strip()
        if not src or not os.path.isdir(src):
            messagebox.showerror("Error", f"Not a folder: {src}")
            return
        transfer_root = os.path.join(src, "transfer")
        if not os.path.isdir(transfer_root):
            messagebox.showerror("Error",
                f"No 'transfer/' subfolder in:\n{src}")
            return
        self.source = src
        self.samples = _list_samples(transfer_root)
        if not self.samples:
            messagebox.showerror("Error", "no test samples found under transfer/")
            return
        # determine K from first sample
        first = os.path.join(transfer_root, self.samples[0])
        inf = np.load(os.path.join(first, "eval", "inference.npz"))
        self.K = int(np.max(inf["assigns"]) + 1)
        # build task list
        self.tasks = [(s, c) for s in self.samples for c in range(self.K)]
        self.cursor = 0
        self.labels = self._load_session()
        self._refresh_summary()
        self._show()

    # ---- session save/load ----
    def _session_path(self):
        return os.path.join(self.source, "line_labels.json")

    def _load_session(self):
        p = self._session_path()
        if not os.path.exists(p):
            return {}
        try:
            with open(p) as f:
                d = json.load(f)
            out = {}
            for row in d.get("labels", []):
                out[(row["sample"], int(row["proto"]))] = {
                    "label": row["label"], "pct": float(row.get("pct", 0)),
                    "count": int(row.get("count", 0)),
                    "total": int(row.get("total", 0)),
                }
            return out
        except Exception:
            return {}

    def _autosave(self):
        if not self.source:
            return
        rows = []
        for (s, c), v in self.labels.items():
            rows.append({"sample": s, "proto": c, **v})
        with open(self._session_path(), "w") as f:
            json.dump({"labels": rows}, f, indent=2)

    # ---- show / label ----
    def _show(self):
        if not self.tasks:
            return
        sample, c = self.tasks[self.cursor]
        sample_dir = os.path.join(self.source, "transfer", sample)
        count, total, K = _class_count_and_avg(sample_dir, c)
        if count is None:
            # class out-of-range for this sample's K, skip
            self._next()
            return
        avg_path = os.path.join(sample_dir, "eval",
                                  "class_averages", f"p{c}.png")
        ex_dir = os.path.join(sample_dir, "eval",
                                "class_examples_200", f"p{c}")
        # save count/total in label dict so summary uses correct numbers
        self.labels.setdefault((sample, c),
                                 {"label": None, "pct": 0,
                                  "count": count, "total": total})
        self.labels[(sample, c)]["count"] = count
        self.labels[(sample, c)]["total"] = total

        existing = self.labels[(sample, c)]["label"]
        existing_pct = self.labels[(sample, c)]["pct"]
        title = (f"Sample: {sample}    Class p{c}    N={count} / "
                  f"{total}  ({100.0 * count / max(total, 1):.1f}%)")
        if existing:
            title += f"   [labeled: {existing}"
            if existing == "partial":
                title += f" {existing_pct}%"
            title += "]"
        self._title_label.config(text=title)

        # main avg
        self._ax_avg.clear(); self._ax_avg.axis('off')
        if os.path.exists(avg_path):
            try:
                img = mpimg.imread(avg_path)
                self._ax_avg.imshow(img, aspect='equal')
            except Exception as e:
                self._ax_avg.text(0.5, 0.5, f"avg load fail: {e}",
                                    ha='center', va='center')
        else:
            self._ax_avg.text(0.5, 0.5, "no class average",
                                ha='center', va='center')
        self._canvas_avg.draw()

        # examples
        ex_files = sorted(glob.glob(os.path.join(ex_dir, "*.png")))
        n_ex = len(self._ex_axes)
        if len(ex_files) >= n_ex:
            picks = random.sample(ex_files, n_ex)
        else:
            picks = ex_files + [None] * (n_ex - len(ex_files))
        for (f, ax, canvas), p in zip(self._ex_axes, picks):
            ax.clear(); ax.axis('off')
            if p is not None:
                try:
                    ax.imshow(mpimg.imread(p), aspect='equal')
                except Exception:
                    pass
            canvas.draw()

        # progress
        n = len(self.tasks); done = sum(1 for v in self.labels.values()
                                          if v.get("label"))
        self._progress_label.config(
            text=f"{self.cursor + 1}/{n}    labeled: {done}")

    def _label_partial_pct(self, pct):
        """Quick-pick: set the % then label as partial."""
        self._partial_pct.set(str(pct))
        self._label("partial")

    def _label(self, kind):
        if not self.tasks:
            return
        sample, c = self.tasks[self.cursor]
        if kind == "partial":
            try:
                pct = float(self._partial_pct.get())
                if not (0 <= pct <= 100):
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error",
                    "Partial % must be a number 0-100")
                return
        else:
            pct = 0.0
        # preserve count/total
        prev = self.labels.get((sample, c), {})
        self.labels[(sample, c)] = {
            "label": kind, "pct": pct,
            "count": prev.get("count", 0),
            "total": prev.get("total", 0),
        }
        self._autosave()
        self._refresh_summary()
        # Check completion BEFORE advancing -- if everything is now labeled,
        # offer to export and stop.
        if self._n_labeled() >= len(self.tasks):
            yes = messagebox.askyesno(
                "All labeled",
                f"All {len(self.tasks)} (sample, prototype) pairs are "
                f"labeled. Export CSVs now?")
            if yes:
                self._done_and_export()
            return
        self._next_unlabeled()

    def _n_labeled(self):
        return sum(1 for v in self.labels.values() if v.get("label"))

    def _next_unlabeled(self):
        """Go to the next task that doesn't have a label yet. If everything
        is labeled, just go to the next index (wrap)."""
        if not self.tasks:
            return
        n = len(self.tasks)
        for step in range(1, n + 1):
            idx = (self.cursor + step) % n
            s, c = self.tasks[idx]
            v = self.labels.get((s, c))
            if v is None or not v.get("label"):
                self.cursor = idx
                self._show()
                return
        # all labeled -- just step
        self._next()

    def _next(self):
        if not self.tasks:
            return
        self.cursor = (self.cursor + 1) % len(self.tasks)
        self._show()

    def _prev(self):
        if not self.tasks:
            return
        self.cursor = (self.cursor - 1) % len(self.tasks)
        self._show()

    def _skip(self):
        self._next()

    # ---- summary ----
    def _per_sample_summary(self):
        out = {}
        for s in self.samples:
            line_frames = 0.0
            nonline_frames = 0.0
            unlabeled = 0
            total = 0
            partial_frames = 0.0
            for c in range(self.K):
                v = self.labels.get((s, c))
                if v is None:
                    continue
                cnt = v.get("count", 0)
                if total == 0:
                    total = v.get("total", 0)
                lab = v.get("label")
                if lab is None:
                    unlabeled += cnt
                elif lab == "lines":
                    line_frames += cnt
                elif lab == "nolines":
                    nonline_frames += cnt
                elif lab == "partial":
                    pct = v.get("pct", 0) / 100.0
                    line_frames += cnt * pct
                    nonline_frames += cnt * (1.0 - pct)
                    partial_frames += cnt
            out[s] = {
                "total": total, "line_frames": line_frames,
                "nonline_frames": nonline_frames,
                "partial_frames": partial_frames,
                "unlabeled_count": unlabeled,
                "coverage": (line_frames / total) if total else 0.0,
            }
        return out

    def _refresh_summary(self):
        s = self._per_sample_summary()
        lines = []
        for samp, d in s.items():
            lines.append(f"{samp}")
            lines.append(f"  total={d['total']:>6}  line~={d['line_frames']:>7.0f}  "
                          f"nonline~={d['nonline_frames']:>7.0f}  "
                          f"unlabeled={d['unlabeled_count']:>5}")
            lines.append(f"  partial_frames={d['partial_frames']:>6.0f}    "
                          f"coverage = {d['coverage']:.4f}")
            lines.append("")
        self._summary_text.delete("1.0", tk.END)
        self._summary_text.insert(tk.END, "\n".join(lines))

    # ---- export ----
    def _done_and_export(self):
        if not self.source:
            return
        per_class_csv = os.path.join(self.source, "line_labels_per_class.csv")
        summary_csv = os.path.join(self.source, "line_labels_summary.csv")
        with open(per_class_csv, "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["sample", "prototype", "count", "total",
                         "label", "partial_pct", "est_line_count"])
            for (s, c), v in sorted(self.labels.items()):
                cnt = v.get("count", 0); tot = v.get("total", 0)
                lab = v.get("label") or "unlabeled"
                pct = v.get("pct", 0)
                est = (cnt if lab == "lines" else
                        0.0 if lab == "nolines" else
                        cnt * pct / 100.0 if lab == "partial" else 0.0)
                w.writerow([s, c, cnt, tot, lab, pct, f"{est:.2f}"])
        sum_data = self._per_sample_summary()
        with open(summary_csv, "w", newline='') as f:
            w = csv.writer(f)
            w.writerow(["sample", "total", "line_frames", "nonline_frames",
                         "partial_frames", "unlabeled_count", "coverage"])
            for s, d in sum_data.items():
                w.writerow([s, d["total"], f"{d['line_frames']:.2f}",
                             f"{d['nonline_frames']:.2f}",
                             f"{d['partial_frames']:.2f}",
                             d["unlabeled_count"], f"{d['coverage']:.4f}"])
        self._autosave()
        messagebox.showinfo("Exported",
            f"Wrote:\n{per_class_csv}\n{summary_csv}")


if __name__ == "__main__":
    LineLabelerApp().mainloop()
