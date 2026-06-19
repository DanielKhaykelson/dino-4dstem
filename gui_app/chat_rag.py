"""chat_rag.py -- semantic retrieval over the project's documents.

Powers the Assistant's `answer_from_docs` tool: chunk the manuscript /
Methods / notes, embed them with a LOCAL Ollama embedding model
(`nomic-embed-text`), and return the most semantically-relevant excerpts
for a question — so users can ask in their OWN words (not the doc's) and
the model answers grounded in your writing.

Embeddings are cached to disk (keyed by a corpus signature + model), so
they're computed once. Falls back to TF-IDF (sklearn, no model) if the
embedding model/server isn't available, so the tool always works.

Distinct from chat_kb.py (what you TEACH it live); this retrieves from
documents you ALREADY wrote.
"""
from __future__ import annotations
import hashlib
import json
import os

EMBED_MODEL = "nomic-embed-text"
_EXTS = (".md", ".tex", ".txt", ".rst")
_EXCLUDE_DIRS = {"runs", "env_snapshot", "__pycache__", "paper_figures",
                 "pretrained", ".git", "nmf"}
_MAX_FILE_BYTES = 400_000
_CHUNK_CHARS = 900
_CHUNK_OVERLAP = 150

_EMB_PATH = os.path.join("runs", "_gui", "_rag_emb.npy")
_META_PATH = os.path.join("runs", "_gui", "_rag_meta.json")

# cached index
_MODE = None          # "embed" | "tfidf" | None
_CHUNKS = None        # list of (source_path, text)
_EMB = None           # np.ndarray (n, d), L2-normalized  (embed mode)
_VEC = _MAT = None    # TF-IDF fallback


# ----------------------------------------------------------------------
# Corpus
# ----------------------------------------------------------------------
def _gather_files(root="."):
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in _EXCLUDE_DIRS
                 and not d.startswith(".")]
        for f in fn:
            if f.lower().endswith(_EXTS):
                p = os.path.join(dp, f)
                try:
                    if os.path.getsize(p) <= _MAX_FILE_BYTES:
                        out.append(p)
                except OSError:
                    pass
    return sorted(set(out))


def _chunk_text(text):
    text = text.replace("\r", "")
    chunks, i, n = [], 0, len(text)
    while i < n:
        c = text[i:i + _CHUNK_CHARS]
        if c.strip():
            chunks.append(c)
        i += _CHUNK_CHARS - _CHUNK_OVERLAP
    return chunks


def _strip_latex(t):
    """Light LaTeX → plain text so .tex chunks (manuals, paper) read
    cleanly for the model and embed better."""
    import re
    t = re.sub(r"(?m)%.*$", "", t)                       # comments
    t = re.sub(r"\\(label|ref|eqref|cite[a-z]*|usepackage|documentclass|"
               r"input|include|includegraphics|bibliography|"
               r"bibliographystyle|vspace|hspace|newcommand)\b"
               r"\s*(\[[^\]]*\])?\s*(\{[^{}]*\})?", " ", t)
    t = re.sub(r"\\(begin|end)\s*\{[^}]*\}", " ", t)
    t = re.sub(r"\\(section|subsection|subsubsection|paragraph|chapter|"
               r"title|textbf|textit|emph|texttt|textsc|underline)\*?\s*"
               r"\{([^{}]*)\}", r" \2 ", t)
    t = re.sub(r"\\item\b", "- ", t)
    t = re.sub(r"\\[a-zA-Z@]+\*?", " ", t)               # remaining commands
    for ch in ("{", "}", "~", "\\", "&"):
        t = t.replace(ch, " ")
    return t


def _read_doc(p):
    try:
        with open(p, encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return ""
    if p.lower().endswith(".tex"):
        txt = _strip_latex(txt)
    return txt


def _collect_chunks(root="."):
    chunks = []
    for p in _gather_files(root):
        txt = _read_doc(p)
        for c in _chunk_text(txt):
            chunks.append((p, c))
    return chunks


def _signature(chunks):
    h = hashlib.sha1()
    h.update(EMBED_MODEL.encode())
    for p, c in chunks:
        h.update(p.encode("utf-8", "ignore"))
        h.update(str(len(c)).encode())
    return h.hexdigest()


# ----------------------------------------------------------------------
# Embeddings (local Ollama)
# ----------------------------------------------------------------------
def _embed_texts(texts, on_progress=None, cancel=None, batch=64):
    """Embed via Ollama /api/embed (batched).  Returns L2-normalized
    np.ndarray, or None on any failure (caller falls back to TF-IDF)."""
    import numpy as np
    import requests
    from gui_app.chat_backends import DEFAULT_HOST
    host = DEFAULT_HOST
    embs = []
    for i in range(0, len(texts), batch):
        if cancel and cancel():
            return None
        part = texts[i:i + batch]
        try:
            r = requests.post(host + "/api/embed",
                              json={"model": EMBED_MODEL, "input": part},
                              timeout=(10, 300))
            if r.status_code != 200:
                return None
            data = r.json().get("embeddings")
            if not data:
                return None
            embs.extend(data)
        except Exception:
            return None
        if on_progress:
            on_progress(f"embedding docs {min(i + batch, len(texts))}/"
                        f"{len(texts)}")
    arr = np.asarray(embs, dtype=np.float32)
    nrm = np.linalg.norm(arr, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return arr / nrm


def _ensure_embed_model(on_progress=None, cancel=None):
    """Make sure the local embedding model is available (pull if needed).
    Returns True if usable."""
    try:
        from gui_app.chat_backends import OllamaBackend
        if not OllamaBackend.server_up():
            return False
        b = OllamaBackend(model=EMBED_MODEL)
        models = b.list_models()
        if b._model_present(models, EMBED_MODEL):
            return True
        if on_progress:
            on_progress(f"downloading embedding model {EMBED_MODEL}…")
        ok, _ = b.pull_model(EMBED_MODEL, on_progress=on_progress,
                             cancel=cancel)
        return ok
    except Exception:
        return False


# ----------------------------------------------------------------------
# Index build / search
# ----------------------------------------------------------------------
def _build_tfidf(chunks):
    global _MODE, _VEC, _MAT, _CHUNKS
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=1,
                          ngram_range=(1, 2), max_features=50_000)
    _MAT = vec.fit_transform([c for _, c in chunks])
    _VEC = vec
    _CHUNKS = chunks
    _MODE = "tfidf"


def build_index(root=".", on_progress=None, cancel=None):
    """Build (and cache) the semantic index; fall back to TF-IDF."""
    global _MODE, _CHUNKS, _EMB
    import numpy as np
    chunks = _collect_chunks(root)
    if not chunks:
        _MODE, _CHUNKS = None, []
        return 0
    sig = _signature(chunks)
    # 1) try the on-disk embedding cache.
    try:
        if os.path.exists(_EMB_PATH) and os.path.exists(_META_PATH):
            meta = json.load(open(_META_PATH, encoding="utf-8"))
            if meta.get("sig") == sig and meta.get("model") == EMBED_MODEL:
                emb = np.load(_EMB_PATH)
                if emb.shape[0] == len(meta.get("chunks", [])):
                    _EMB = emb
                    _CHUNKS = [tuple(c) for c in meta["chunks"]]
                    _MODE = "embed"
                    return len(_CHUNKS)
    except Exception:
        pass
    # 2) compute embeddings if the model is available.
    if _ensure_embed_model(on_progress=on_progress, cancel=cancel):
        if on_progress:
            on_progress("embedding documents (one-time)…")
        emb = _embed_texts([c for _, c in chunks],
                           on_progress=on_progress, cancel=cancel)
        if emb is not None and emb.shape[0] == len(chunks):
            _EMB, _CHUNKS, _MODE = emb, chunks, "embed"
            try:
                os.makedirs(os.path.dirname(_EMB_PATH), exist_ok=True)
                np.save(_EMB_PATH, emb)
                json.dump({"sig": sig, "model": EMBED_MODEL,
                           "chunks": [[p, c] for p, c in chunks]},
                          open(_META_PATH, "w", encoding="utf-8"))
            except Exception:
                pass
            return len(chunks)
    # 3) fallback: TF-IDF (keyword) so the tool still works.
    _build_tfidf(chunks)
    return len(chunks)


def ensure_index(on_progress=None, cancel=None):
    if _MODE is None:
        build_index(on_progress=on_progress, cancel=cancel)
    return _CHUNKS or []


def search(query, k=4, on_progress=None, cancel=None):
    """Return top-k (score, source_path, text) for the query."""
    ensure_index(on_progress=on_progress, cancel=cancel)
    if not _CHUNKS:
        return []
    if _MODE == "embed":
        import numpy as np
        qe = _embed_texts([query])
        if qe is None:
            return []
        sims = (_EMB @ qe[0])
        order = np.argsort(sims)[::-1][:k]
        return [(float(sims[i]), _CHUNKS[i][0], _CHUNKS[i][1])
                for i in order if sims[i] > 0.05]
    # tfidf
    from sklearn.metrics.pairwise import linear_kernel
    qv = _VEC.transform([query])
    sims = linear_kernel(qv, _MAT).ravel()
    order = sims.argsort()[::-1][:k]
    return [(float(sims[i]), _CHUNKS[i][0], _CHUNKS[i][1])
            for i in order if sims[i] > 0.0]


def index_mode():
    return _MODE
