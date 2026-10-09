"""In-process store: exact cosine search plus Okapi BM25 for keywords.

Good for tests, demos and small corpora. Re-ingesting a transcript replaces
its previous chunks, so ingestion is idempotent.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from earningsiq.embeddings import tokenize
from earningsiq.lexicon import expand_terms
from earningsiq.models import Chunk, ScoredChunk
from earningsiq.store.base import SearchFilters, TranscriptSummary


class InMemoryStore:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._tokens: list[list[str]] = []
        self.k1, self.b = k1, b

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        for tid in {c.transcript_id for c in chunks}:
            self.delete_transcript(tid)
        vectors = np.asarray(vectors, dtype=np.float32)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)
        self._tokens.extend(tokenize(c.text) for c in chunks)

    def _candidates(self, filters: SearchFilters | None) -> list[int]:
        if filters is None:
            return list(range(len(self._chunks)))
        return [i for i, c in enumerate(self._chunks) if filters.matches(c)]

    def vector_search(
        self, query_vector: np.ndarray, k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        idx = self._candidates(filters)
        if not idx or self._vectors is None:
            return []
        sims = self._vectors[idx] @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-sims)[:k]
        return [
            ScoredChunk(self._chunks[idx[i]], float(sims[i]), vector_score=float(sims[i]))
            for i in order
        ]

    def keyword_search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        idx = self._candidates(filters)
        terms = expand_terms(set(tokenize(query)))
        if not idx or not terms:
            return []
        docs = [self._tokens[i] for i in idx]
        n = len(docs)
        avg_len = sum(map(len, docs)) / n or 1.0
        df = Counter(t for d in docs for t in set(d) if t in terms)
        scored: list[tuple[float, int]] = []
        for pos, doc in enumerate(docs):
            tf = Counter(t for t in doc if t in terms)
            if not tf:
                continue
            score = 0.0
            for term, freq in tf.items():
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * len(doc) / avg_len)
                score += idf * freq * (self.k1 + 1) / denom
            scored.append((score, pos))
        scored.sort(reverse=True)
        return [ScoredChunk(self._chunks[idx[pos]], s, keyword_score=s) for s, pos in scored[:k]]

    def delete_transcript(self, transcript_id: str) -> int:
        keep = [i for i, c in enumerate(self._chunks) if c.transcript_id != transcript_id]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._tokens = [self._tokens[i] for i in keep]
            self._vectors = self._vectors[keep] if keep and self._vectors is not None else None
        return removed

    def list_transcripts(self) -> list[TranscriptSummary]:
        counts: dict[str, list] = {}
        for c in self._chunks:
            entry = counts.setdefault(c.transcript_id, [c, 0])
            entry[1] += 1
        return [
            TranscriptSummary(tid, c.ticker, c.fiscal_year, c.fiscal_quarter, n)
            for tid, (c, n) in sorted(counts.items())
        ]
