"""Shared evidence helpers for `critic` and `citation_checker`.

Everything here MIRRORS the frozen scorer instead of approximating it:
the same normalisation (`arena.scorer._norm`), the same one-LINE support
rule (`_supports` over `_norm_lines`), and the same definition of what a
run "retrieved" (every `fetch_doc` target plus a replay of every `search`
event at its recorded `k`, read off the run's own trace). A layer that
judges claims with a looser rule than the grader keeps claims the grader
then calls `HALLUCINATED` or `UNRETRIEVED`.

Read-only: nothing in this module edits a claim's text. The only text it
ever produces is a SUBSTRING of a claim (see `supported_segments`), which
is a legal trim — the scorer still finds it in the model's own payload.
"""

from __future__ import annotations

import json
import re

from arena.scorer import (
    MAX_CLAIM_CHARS,
    MAX_CLAIMS_PER_DOC,
    MAX_REPLAYED_SEARCHES,
    MAX_SCORED_CLAIMS,
    MIN_SUPPORT_CHARS,
    _norm,
    _normalised_bodies,
)

__all__ = [
    "MAX_CLAIM_CHARS",
    "MAX_CLAIMS_PER_DOC",
    "MAX_SCORED_CLAIMS",
    "MIN_SUPPORT_CHARS",
    "norm",
    "doc_lines",
    "supports",
    "retrieved_doc_ids",
    "fully_read_doc_ids",
    "supporting_doc_ids",
    "supported_segments",
]

_DOC_ID_RE = re.compile(r"doc-\d{4}")

#: Words and single punctuation marks. Segment boundaries fall between
#: tokens, so a trim can drop a trailing "." the document line lacks.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def norm(text) -> str:
    """The scorer's own normalisation: NFC, casefold, whitespace collapsed."""
    return _norm(text)


def doc_lines(corpus) -> dict:
    """`{doc_id: (normalised line, ...)}` — cached by the scorer per corpus."""
    if corpus is None or not isinstance(getattr(corpus, "docs", None), list):
        return {}
    return _normalised_bodies(corpus)


def supports(lines, normalised_text: str) -> bool:
    """Exactly `arena.scorer._supports`: verbatim inside ONE line."""
    if len(normalised_text) < MIN_SUPPORT_CHARS:
        return False
    return any(normalised_text in line for line in lines)


def _as_k(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if value != value:  # NaN
        return 0
    return int(value)


def retrieved_doc_ids(ctx) -> set:
    """Doc ids the SCORER will accept as retrieved for this run.

    Read off the run's own trace and replayed the way
    `arena.scorer._read_trace` does it. Falls back to the doc ids visible
    in the observations if the trace cannot be read.
    """
    corpus = getattr(ctx, "corpus", None)
    if corpus is None or not isinstance(getattr(corpus, "docs", None), list):
        return set()
    known = {doc.doc_id for doc in corpus.docs}
    trace = getattr(ctx, "trace", None)
    to_jsonl = getattr(trace, "to_jsonl", None)
    if not callable(to_jsonl):
        return set(_DOC_ID_RE.findall(ctx.observed_text)) & known

    jsonl = to_jsonl()
    jsonl = jsonl if isinstance(jsonl, str) else ""
    retrieved: set = set()
    seen_queries: list = []
    max_k = max(1, len(corpus.docs))
    for line in jsonl.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or record.get("event") != "tool_call":
            continue
        name = record.get("name")
        if name == "fetch_doc":
            doc_id = record.get("doc_id")
            if isinstance(doc_id, str) and doc_id in known:
                retrieved.add(doc_id)
        elif name == "search":
            query = record.get("query")
            if not isinstance(query, str) or not query:
                continue
            k = min(max(1, _as_k(record.get("k")) or 5), max_k)
            if (query, k) in seen_queries or len(seen_queries) >= MAX_REPLAYED_SEARCHES:
                continue
            seen_queries.append((query, k))
            for doc in corpus.search(query, k=k):
                retrieved.add(doc.doc_id)
    return retrieved


def fully_read_doc_ids(ctx, candidates) -> set:
    """Candidates whose WHOLE body came back in one clean observation."""
    corpus = getattr(ctx, "corpus", None)
    if corpus is None:
        return set()
    observed = ctx.observed_text
    out = set()
    for doc_id in candidates:
        doc = corpus.get(doc_id)
        if doc is not None and doc.body and doc.body in observed:
            out.add(doc_id)
    return out


def supporting_doc_ids(ctx, normalised_text: str, candidates, lines=None) -> list:
    """Candidates with a line that quotes the text, best source first:
    fully-read documents before ones only seen through a search hit,
    then corpus order (deterministic)."""
    corpus = getattr(ctx, "corpus", None)
    if corpus is None or len(normalised_text) < MIN_SUPPORT_CHARS:
        return []
    lines = doc_lines(corpus) if lines is None else lines
    hits = [
        doc.doc_id
        for doc in corpus.docs
        if doc.doc_id in candidates and supports(lines.get(doc.doc_id, ()), normalised_text)
    ]
    full = fully_read_doc_ids(ctx, hits)
    return [d for d in hits if d in full] + [d for d in hits if d not in full]


def supported_segments(text: str, haystack: str) -> list:
    """Maximal token-aligned SUBSTRINGS of `text` that occur, normalised,
    inside `haystack` (normalised document lines joined by newlines, so a
    match can never span two lines).

    Returns `(start, end)` character spans into `text`. Two pointers: if
    tokens [i, j) are found then so are [i+1, j), so the scan is linear in
    the number of tokens.
    """
    tokens = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    n = len(tokens)
    spans: list = []
    j = 0
    for i in range(n):
        if j < i:
            j = i
        while j < n and _norm(text[tokens[i][0]:tokens[j][1]]) in haystack:
            j += 1
        if j > i and (not spans or spans[-1][1] < j):
            spans.append((i, j))
    return [(tokens[i][0], tokens[j - 1][1]) for i, j in spans]
