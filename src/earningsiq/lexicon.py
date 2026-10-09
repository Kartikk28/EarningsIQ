"""Finance vocabulary for query expansion.

Analysts and executives describe the same thing with different words
("buyback" vs "share repurchase", "guidance" vs "outlook"). Expanding the
*query* side keeps keyword search and extractive answers from missing them.
"""

from __future__ import annotations

from earningsiq.embeddings import tokenize

_GROUPS: list[tuple[str, ...]] = [
    ("buyback", "repurchase", "repurchases"),
    ("guidance", "guide", "outlook", "expect", "forecast"),
    ("revenue", "sales", "topline"),
    ("profit", "income", "earnings"),
    ("margin", "margins"),
    ("capex", "capital expenditure"),
    ("headcount", "employees", "hiring"),
    ("arr", "annual recurring revenue"),
    ("fcf", "free cash flow"),
    ("opex", "operating expenses"),
    ("r&d", "research and development"),
    ("competition", "competitor", "competitive", "rivals"),
    ("demand", "orders", "bookings", "pipeline"),
    ("increase", "grew", "growth", "up", "rose", "expanded", "expansion", "raised"),
    ("decrease", "declined", "fell", "down", "lower", "contraction"),
]


def _build_index() -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for group in _GROUPS:
        stems = {t for phrase in group for t in tokenize(phrase)}
        for s in stems:
            index.setdefault(s, set()).update(stems)
    return index


_INDEX = _build_index()


def expand_terms(terms: set[str]) -> set[str]:
    """Return the query terms plus their finance synonyms (all stemmed)."""
    expanded = set(terms)
    for t in terms:
        expanded |= _INDEX.get(t, set())
    return expanded
