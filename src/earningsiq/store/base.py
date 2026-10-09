from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from earningsiq.models import Chunk, ScoredChunk, Section


@dataclass(frozen=True)
class SearchFilters:
    """Optional metadata constraints applied before ranking."""

    ticker: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    section: Section | None = None
    speaker: str | None = None

    def matches(self, chunk: Chunk) -> bool:
        if self.ticker and chunk.ticker != self.ticker.upper():
            return False
        if self.fiscal_year is not None and chunk.fiscal_year != self.fiscal_year:
            return False
        if self.fiscal_quarter is not None and chunk.fiscal_quarter != self.fiscal_quarter:
            return False
        if self.section is not None and chunk.section != self.section:
            return False
        if self.speaker and self.speaker.lower() not in chunk.speaker.lower():
            return False
        return True


@dataclass(frozen=True)
class TranscriptSummary:
    transcript_id: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    chunk_count: int


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def vector_search(
        self, query_vector: np.ndarray, k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]: ...

    def keyword_search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]: ...

    def delete_transcript(self, transcript_id: str) -> int: ...

    def list_transcripts(self) -> list[TranscriptSummary]: ...
