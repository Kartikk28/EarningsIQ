"""Hybrid retrieval: dense vectors + keyword search fused with Reciprocal Rank Fusion.

Dense search catches paraphrases ("how did margins do" ~ "gross margin expanded");
keyword search catches exact figures, product names and tickers that embeddings
blur. RRF combines the two rankings without having to calibrate their scores.
"""

from __future__ import annotations

from dataclasses import dataclass

from earningsiq.embeddings import Embedder
from earningsiq.models import ScoredChunk
from earningsiq.store.base import SearchFilters, VectorStore


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]], weights: list[float], rrf_k: int = 60
) -> list[ScoredChunk]:
    fused: dict[str, ScoredChunk] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking):
            cid = hit.chunk.chunk_id
            entry = fused.get(cid)
            if entry is None:
                entry = ScoredChunk(hit.chunk, 0.0)
                fused[cid] = entry
            entry.score += weight / (rrf_k + rank + 1)
            if hit.vector_score is not None:
                entry.vector_score = hit.vector_score
            if hit.keyword_score is not None:
                entry.keyword_score = hit.keyword_score
    return sorted(fused.values(), key=lambda h: h.score, reverse=True)


@dataclass
class Retriever:
    store: VectorStore
    embedder: Embedder
    top_k: int = 6
    candidate_k: int = 30
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    max_per_turn: int = 2

    def retrieve(
        self, query: str, filters: SearchFilters | None = None, top_k: int | None = None
    ) -> list[ScoredChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        k = top_k or self.top_k
        qvec = self.embedder.embed([query])[0]
        dense = self.store.vector_search(qvec, self.candidate_k, filters)
        sparse = self.store.keyword_search(query, self.candidate_k, filters)
        fused = reciprocal_rank_fusion([dense, sparse], [self.vector_weight, self.keyword_weight])
        # Overlapping windows from one speaker turn say nearly the same thing;
        # cap them so the context covers more distinct evidence.
        per_turn: dict[tuple[str, int], int] = {}
        results: list[ScoredChunk] = []
        for hit in fused:
            key = (hit.chunk.transcript_id, hit.chunk.turn_index)
            if per_turn.get(key, 0) >= self.max_per_turn:
                continue
            per_turn[key] = per_turn.get(key, 0) + 1
            results.append(hit)
            if len(results) == k:
                break
        return results
