"""Citation handling: number the context, then hold the model to it.

The answer may only cite sources that were actually retrieved. References to
numbers outside the context are stripped, and an answer with no valid citation
is marked ``grounded=False`` so callers can treat it with suspicion.
"""

from __future__ import annotations

import re

from earningsiq.models import Citation, ScoredChunk

_REF = re.compile(r"\[(\d{1,2})\]")
_SNIPPET_WORDS = 40


def source_label(number: int, hit: ScoredChunk) -> str:
    c = hit.chunk
    who = f"{c.speaker} ({c.role})" if c.role else c.speaker
    section = "Q&A" if c.section.value == "qa" else "Prepared remarks"
    return f"[{number}] {c.ticker} {c.period} · {who} · {section}"


def build_context(hits: list[ScoredChunk]) -> str:
    return "\n\n".join(f"{source_label(i, h)}\n{h.chunk.text}" for i, h in enumerate(hits, 1))


def cited_numbers(text: str) -> list[int]:
    seen: dict[int, None] = {}
    for m in _REF.finditer(text):
        seen.setdefault(int(m.group(1)), None)
    return list(seen)


def _snippet(text: str) -> str:
    words = text.split()
    return " ".join(words[:_SNIPPET_WORDS]) + (" …" if len(words) > _SNIPPET_WORDS else "")


def resolve_citations(text: str, hits: list[ScoredChunk]) -> tuple[str, list[Citation], bool]:
    """Drop invalid references and return (clean_text, citations, grounded)."""
    valid = set(range(1, len(hits) + 1))
    clean = _REF.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", text)
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    citations = []
    for n in cited_numbers(clean):
        c = hits[n - 1].chunk
        citations.append(
            Citation(
                number=n,
                chunk_id=c.chunk_id,
                ticker=c.ticker,
                period=c.period,
                speaker=c.speaker,
                section=c.section,
                snippet=_snippet(c.text),
            )
        )
    return clean, citations, bool(citations)
