"""chat_kb.py -- persistent 'learned knowledge' for the Assistant.

A tiny, growing store of things the USER teaches the assistant — facts,
corrections, preferences, and few-shot examples — saved to disk and
injected into the system prompt every turn (in-context learning; no
fine-tuning).  This is how a small local model (qwen2.5) is made reliable
on THIS app/domain: it stops repeating mistakes and honours your prefs.

Distinct from RAG (chat_rag.py): RAG retrieves chunks from your existing
documents; this stores what you tell it live and injects it wholesale.
"""
from __future__ import annotations
import json
import os

KB_PATH = os.path.join("runs", "_gui", "_assistant_kb.json")
_KINDS = ("fact", "correction", "preference", "example")
_MAX_NOTES = 200


def load_kb() -> dict:
    try:
        with open(KB_PATH, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("notes"), list):
            return d
    except Exception:
        pass
    return {"notes": []}


def add_note(kind: str, text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    kind = (kind or "fact").strip().lower()
    if kind not in _KINDS:
        kind = "fact"
    kb = load_kb()
    for n in kb["notes"]:                       # dedup identical text
        if n.get("text", "").strip().lower() == text.lower():
            n["kind"] = kind
            _save(kb)
            return True
    kb["notes"].append({"kind": kind, "text": text})
    kb["notes"] = kb["notes"][-_MAX_NOTES:]
    _save(kb)
    return True


def list_notes() -> list:
    return load_kb().get("notes", [])


def forget(substring: str) -> int:
    """Remove notes whose text contains `substring` (case-insensitive).
    Returns how many were removed."""
    sub = (substring or "").strip().lower()
    if not sub:
        return 0
    kb = load_kb()
    before = len(kb["notes"])
    kb["notes"] = [n for n in kb["notes"]
                   if sub not in n.get("text", "").lower()]
    removed = before - len(kb["notes"])
    if removed:
        _save(kb)
    return removed


def render_for_prompt(max_chars: int = 2500) -> str:
    """Bulleted text of the learned knowledge for the system prompt."""
    notes = load_kb().get("notes", [])
    if not notes:
        return ""
    lines = [f"- [{n.get('kind', 'fact')}] {n.get('text', '')}"
             for n in notes if n.get("text")]
    body = "\n".join(lines)
    if len(body) > max_chars:                   # keep the most recent
        body = body[-max_chars:]
    return body


def _save(kb: dict) -> None:
    try:
        os.makedirs(os.path.dirname(KB_PATH), exist_ok=True)
        with open(KB_PATH, "w", encoding="utf-8") as f:
            json.dump(kb, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[chat_kb] save failed: {e!r}", flush=True)
