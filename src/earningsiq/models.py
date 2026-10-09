"""Core domain objects shared across ingestion, retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Section(str, Enum):
    """Which part of the call a passage came from."""

    PREPARED_REMARKS = "prepared_remarks"
    QA = "qa"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TranscriptMeta:
    """Identifies one earnings call."""

    ticker: str
    company: str
    fiscal_year: int
    fiscal_quarter: int  # 1-4
    call_date: str | None = None  # ISO date, optional

    @property
    def transcript_id(self) -> str:
        return f"{self.ticker.upper()}-{self.fiscal_year}Q{self.fiscal_quarter}"

    @property
    def period(self) -> str:
        return f"Q{self.fiscal_quarter} FY{self.fiscal_year}"


@dataclass(frozen=True)
class SpeakerTurn:
    """One uninterrupted block of speech by a single speaker."""

    speaker: str
    role: str | None
    section: Section
    text: str
    index: int


@dataclass
class Transcript:
    meta: TranscriptMeta
    turns: list[SpeakerTurn]

    @property
    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for t in self.turns:
            seen.setdefault(t.speaker, None)
        return list(seen)


@dataclass
class Chunk:
    """A retrievable passage. Carries enough metadata to cite it precisely."""

    chunk_id: str
    transcript_id: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    speaker: str
    role: str | None
    section: Section
    turn_index: int
    text: str
    embedding: list[float] | None = None

    @property
    def period(self) -> str:
        return f"Q{self.fiscal_quarter} FY{self.fiscal_year}"


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None


@dataclass
class Citation:
    """A numbered source the answer points to, e.g. [2]."""

    number: int
    chunk_id: str
    ticker: str
    period: str
    speaker: str
    section: Section
    snippet: str


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    model: str = "extractive"
