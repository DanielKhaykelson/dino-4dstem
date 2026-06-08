"""collect_interpretation_reports.py — gather every per-run interpretation
report + its figures into one browsable folder (docs/interpretation_reports/),
copy the paper/explainer documents, and write an INDEX.md.

Re-run any time after analysing more runs:  python tools/collect_interpretation_reports.py
"""
import os
import re
import glob
import json
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "docs", "interpretation_reports")
GENERIC = {"_gui", "stage2", "runs"}


def label_for(run_rel):
    parts = [p for p in run_rel.replace("\\", "/").split("/")
             if p not in GENERIC and not p.startswith("_sweep")]
    return "__".join(parts) or "run"


_FRIENDLY = {
    "Survey_CH2_1": "IMC SI3", "Survey_CH2_0_1": "IMC SI4",
    "Na007b": "NaPHI Na007b", "EuInAs_B100": "EuInAs B100",
    "IMC_SI5": "IMC SI5",
}


def sample_of(run):
    raw = "?"
    for fn, key in (("_train_kwargs.json", "sample"),
                    ("run_summary.json", "sample")):
        p = os.path.join(run, fn)
        if os.path.exists(p):
            try:
                raw = json.load(open(p)).get(key, "?"); break
            except Exception:
                pass
    s = raw
    if s.startswith("loaded__"):
        s = s[len("loaded__"):]
    for suf in ("_nbed.cube", ".cube", ".prz"):
        s = s.replace(suf, "")
    return _FRIENDLY.get(s, s)


def parse_report(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"K_active\s*=\s*(\d+)", txt)
    K = m.group(1) if m else "?"
    m = re.search(r"N\s*=\s*(\d+)", txt)
    N = m.group(1) if m else "?"
    uniq = ""
    for ln in txt.splitlines():
        if ln.startswith("**Uniqueness:**"):
            uniq = re.sub(r"\*\*|`", "", ln.replace("**Uniqueness:**", "")).strip()
            break
    if not uniq:                      # curated report → grab TL;DR sentence
        m = re.search(r"clusters on \*\*(.+?)\*\*", txt, re.S)
        uniq = "see report" if not m else ""
    return K, N, uniq


def main():
    # clear prior contents but tolerate Windows file locks on the top dir
    if os.path.isdir(DEST):
        for child in os.listdir(DEST):
            p = os.path.join(DEST, child)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) \
                else os.remove(p)
    os.makedirs(DEST, exist_ok=True)
    rows = []
    for idir in sorted(glob.glob(os.path.join(ROOT, "runs", "**",
                                              "_interpretability"),
                                 recursive=True)):
        run = os.path.dirname(idir)
        rel = os.path.relpath(run, os.path.join(ROOT, "runs"))
        label = label_for(rel)
        sub = os.path.join(DEST, label)
        os.makedirs(sub, exist_ok=True)
        # copy reports + figures + json (skip the bulky paper/ subdir)
        for f in os.listdir(idir):
            src = os.path.join(idir, f)
            if os.path.isfile(src) and f.lower().endswith(
                    (".md", ".png", ".json", ".npy")):
                shutil.copy2(src, os.path.join(sub, f))
        rep = (os.path.join(sub, "report_auto.md")
               if os.path.exists(os.path.join(sub, "report_auto.md"))
               else os.path.join(sub, "report.md"))
        K = N = "?"; uniq = ""
        if os.path.exists(rep):
            K, N, uniq = parse_report(rep)
        rows.append(dict(sample=sample_of(run), label=label, K=K, N=N,
                         uniq=uniq, rel=rel,
                         report=os.path.basename(rep) if os.path.exists(rep)
                         else "(none)"))
        print(f"  {label}: copied {len(os.listdir(sub))} files")

    # copy paper / explainer documents
    papers = os.path.join(DEST, "_documents")
    os.makedirs(papers, exist_ok=True)
    doc_srcs = [
        ("runs/_sweep_m_K_20260525_213539/IMC_SI5/stage2/"
         "m0.9700_seed42_K60/_interpretability/paper/interpretability_acs.pdf",
         "IMC_SI5_interpretability_manuscript.pdf"),
        ("runs/_gui/_interp_summary/paper/interp_summary_acs.pdf",
         "cross_sample_summary.pdf"),
        ("docs/explainer/interpretation_explainer_doc.pdf",
         "explainer_for_PI.pdf"),
        ("docs/explainer/interpretation_explainer.pptx",
         "explainer_for_PI.pptx"),
    ]
    docs_copied = []
    for src, name in doc_srcs:
        sp = os.path.join(ROOT, src)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(papers, name))
            docs_copied.append(name)

    # INDEX.md
    L = ["# Interpretation reports — all runs", "",
         "Auto-collected by `tools/collect_interpretation_reports.py`. "
         "Each subfolder mirrors that run's `_interpretability/` "
         "(report + figures).", "",
         "| sample | run | K | N | classical-uniqueness verdict |",
         "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["sample"]):
        link = f"[{r['label']}]({r['label']}/{r['report']})"
        L.append(f"| {r['sample']} | {link} | {r['K']} | {r['N']} | "
                 f"{r['uniq'] or '—'} |")
    L += ["", "## Documents (`_documents/`)", ""]
    for n in docs_copied:
        L.append(f"- [{n}](_documents/{n})")
    L += ["", "## Notes", "",
          "- *distinctive* = classical methods (virtual-DF / radial / PCA / "
          "NMF) do **not** reproduce the DINO map; *substantially captured* "
          "= they get most of the way.",
          "- All runs share the same physical basis: scattered intensity + "
          "low-q 2-D diffraction structure; not orientation.",
          "- IMC_SI5 carries a hand-curated `report.md` (+ manuscript) rather "
          "than the auto report."]
    open(os.path.join(DEST, "INDEX.md"), "w", encoding="utf-8").write(
        "\n".join(L))
    print(f"\nwrote {os.path.join(DEST, 'INDEX.md')}  "
          f"({len(rows)} runs, {len(docs_copied)} documents)")


if __name__ == "__main__":
    main()
