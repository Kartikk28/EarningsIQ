"""Speaker-aware chunking.

Chunks never cross a speaker boundary, so every retrieved passage can be cited
to exactly one person ("CFO, Q3 FY2026"). Long turns are split on sentence
boundaries into windows of roughly ``max_words`` with ``overlap_sentences`` of
context carried over between windows.
"""

from __future__ import annotations

import re

from earningsiq.models import Chunk, Transcript

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_SKIP_SPEAKERS = {"operator"}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _windows(sentences: list[str], max_words: int, overlap: int) -> list[list[str]]:
    windows: list[list[str]] = []
    current: list[str] = []
    words = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and words + n > max_words:
            windows.append(current)
            current = current[-overlap:] if overlap else []
            words = sum(len(s.split()) for s in current)
        current.append(sentence)
        words += n
    if current:
        windows.append(current)
    return windows


def chunk_transcript(
    transcript: Transcript,
    max_words: int = 180,
    overlap_sentences: int = 1,
    min_words: int = 8,
) -> list[Chunk]:
    if max_words < 20:
        raise ValueError("max_words must be at least 20")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    meta = transcript.meta
    chunks: list[Chunk] = []
    for turn in transcript.turns:
        if turn.speaker.lower() in _SKIP_SPEAKERS:
            continue
        sentences = split_sentences(turn.text)
        for n, window in enumerate(_windows(sentences, max_words, overlap_sentences)):
            text = " ".join(window)
            if len(text.split()) < min_words:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{meta.transcript_id}:{turn.index}:{n}",
                    transcript_id=meta.transcript_id,
                    ticker=meta.ticker.upper(),
                    fiscal_year=meta.fiscal_year,
                    fiscal_quarter=meta.fiscal_quarter,
                    speaker=turn.speaker,
                    role=turn.role,
                    section=turn.section,
                    turn_index=turn.index,
                    text=text,
                )
            )
    return chunks
