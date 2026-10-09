"""The EarningsIQ service: ingest transcripts, answer questions with citations."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from earningsiq.config import Settings
from earningsiq.embeddings import Embedder, build_embedder
from earningsiq.generation import NOT_FOUND, Generator, build_generator, resolve_citations
from earningsiq.ingest import chunk_transcript, parse_transcript
from earningsiq.models import Answer, TranscriptMeta
from earningsiq.retrieval import Retriever
from earningsiq.store import SearchFilters, VectorStore, build_store

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    transcript_id: str
    turns: int
    chunks: int
    speakers: list[str]


class EarningsIQ:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        generator: Generator,
        top_k: int = 6,
        chunk_max_words: int = 180,
    ):
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.retriever = Retriever(store, embedder, top_k=top_k)
        self.chunk_max_words = chunk_max_words

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> EarningsIQ:
        s = settings or Settings.from_env()
        embedder = build_embedder(s.embedder, s.embedding_dim)
        return cls(
            store=build_store(s.database_url, embedder.dim),
            embedder=embedder,
            generator=build_generator(s.generator, s.generator_model),
            top_k=s.top_k,
            chunk_max_words=s.chunk_max_words,
        )

    def ingest(self, raw_text: str, meta: TranscriptMeta) -> IngestResult:
        transcript = parse_transcript(raw_text, meta)
        chunks = chunk_transcript(transcript, max_words=self.chunk_max_words)
        if not chunks:
            raise ValueError(f"Transcript {meta.transcript_id} produced no chunks")
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        log.info(
            "ingested %s: %d turns, %d chunks",
            meta.transcript_id,
            len(transcript.turns),
            len(chunks),
        )
        return IngestResult(
            meta.transcript_id, len(transcript.turns), len(chunks), transcript.speakers
        )

    def ask(
        self, question: str, filters: SearchFilters | None = None, top_k: int | None = None
    ) -> Answer:
        started = time.perf_counter()
        hits = self.retriever.retrieve(question, filters, top_k)
        raw = self.generator.generate(question, hits) if hits else NOT_FOUND
        if raw.strip() == NOT_FOUND:
            return Answer(question, NOT_FOUND, [], grounded=False, model=self.generator.name)
        text, citations, grounded = resolve_citations(raw, hits)
        log.info(
            "answered in %.0f ms (%d hits, %d citations, grounded=%s)",
            (time.perf_counter() - started) * 1000,
            len(hits),
            len(citations),
            grounded,
        )
        return Answer(question, text, citations, grounded=grounded, model=self.generator.name)
