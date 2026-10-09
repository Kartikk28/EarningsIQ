"""Postgres + pgvector store (requires ``pip install earningsiq[pg]``).

Vector search uses an HNSW cosine index; keyword search uses Postgres full-text
search (``ts_rank_cd`` over a generated ``tsvector`` column). Metadata filters
are pushed down into SQL so ranking only sees eligible rows.
"""

from __future__ import annotations

import numpy as np

from earningsiq.embeddings import raw_terms
from earningsiq.models import Chunk, ScoredChunk, Section
from earningsiq.store.base import SearchFilters, TranscriptSummary

_COLUMNS = (
    "chunk_id, transcript_id, ticker, fiscal_year, fiscal_quarter, speaker, role, section, "
    "turn_index, text"
)


def _row_to_chunk(row: tuple) -> Chunk:
    return Chunk(
        chunk_id=row[0],
        transcript_id=row[1],
        ticker=row[2],
        fiscal_year=row[3],
        fiscal_quarter=row[4],
        speaker=row[5],
        role=row[6],
        section=Section(row[7]),
        turn_index=row[8],
        text=row[9],
    )


def _where(filters: SearchFilters | None) -> tuple[str, list]:
    if filters is None:
        return "", []
    clauses, params = [], []
    if filters.ticker:
        clauses.append("ticker = %s")
        params.append(filters.ticker.upper())
    if filters.fiscal_year is not None:
        clauses.append("fiscal_year = %s")
        params.append(filters.fiscal_year)
    if filters.fiscal_quarter is not None:
        clauses.append("fiscal_quarter = %s")
        params.append(filters.fiscal_quarter)
    if filters.section is not None:
        clauses.append("section = %s")
        params.append(filters.section.value)
    if filters.speaker:
        clauses.append("speaker ILIKE %s")
        params.append(f"%{filters.speaker}%")
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


class PgVectorStore:
    def __init__(self, dsn: str, dim: int, table: str = "earnings_chunks"):
        import psycopg
        from pgvector.psycopg import register_vector

        if not table.isidentifier():
            raise ValueError("table must be a plain SQL identifier")
        self.table, self.dim = table, dim
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._migrate()

    def _migrate(self) -> None:
        t = self.table
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t} (
                chunk_id TEXT PRIMARY KEY,
                transcript_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                fiscal_year INT NOT NULL,
                fiscal_quarter INT NOT NULL,
                speaker TEXT NOT NULL,
                role TEXT,
                section TEXT NOT NULL,
                turn_index INT NOT NULL,
                text TEXT NOT NULL,
                embedding vector({self.dim}) NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            )"""
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {t}_hnsw ON {t} USING hnsw (embedding vector_cosine_ops)"
        )
        self._conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_tsv ON {t} USING gin (tsv)")
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {t}_period ON {t} (ticker, fiscal_year, fiscal_quarter)"
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._conn.transaction():
            for tid in {c.transcript_id for c in chunks}:
                self.delete_transcript(tid)
            with self._conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self.table} ({_COLUMNS}, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        (
                            c.chunk_id,
                            c.transcript_id,
                            c.ticker,
                            c.fiscal_year,
                            c.fiscal_quarter,
                            c.speaker,
                            c.role,
                            c.section.value,
                            c.turn_index,
                            c.text,
                            np.asarray(v, dtype=np.float32),
                        )
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )

    def vector_search(
        self, query_vector: np.ndarray, k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        where, params = _where(filters)
        q = np.asarray(query_vector, dtype=np.float32)
        rows = self._conn.execute(
            f"SELECT {_COLUMNS}, 1 - (embedding <=> %s) AS sim FROM {self.table} "
            f"WHERE TRUE{where} ORDER BY embedding <=> %s LIMIT %s",
            [q, *params, q, k],
        ).fetchall()
        return [
            ScoredChunk(_row_to_chunk(r), float(r[10]), vector_score=float(r[10])) for r in rows
        ]

    def keyword_search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        # OR the terms together: a question rarely has every word in one passage.
        # Only [a-z0-9.] survive raw_terms, so the tsquery string is injection-safe.
        terms = [t.replace("%", "") for t in raw_terms(query)]
        terms = [t for t in terms if t]
        if not terms:
            return []
        where, params = _where(filters)
        rows = self._conn.execute(
            f"SELECT {_COLUMNS}, ts_rank_cd(tsv, q) AS rank "
            f"FROM {self.table}, to_tsquery('english', %s) q "
            f"WHERE tsv @@ q{where} ORDER BY rank DESC LIMIT %s",
            [" | ".join(terms), *params, k],
        ).fetchall()
        return [
            ScoredChunk(_row_to_chunk(r), float(r[10]), keyword_score=float(r[10])) for r in rows
        ]

    def delete_transcript(self, transcript_id: str) -> int:
        cur = self._conn.execute(
            f"DELETE FROM {self.table} WHERE transcript_id = %s", [transcript_id]
        )
        return cur.rowcount or 0

    def list_transcripts(self) -> list[TranscriptSummary]:
        rows = self._conn.execute(
            f"SELECT transcript_id, ticker, fiscal_year, fiscal_quarter, COUNT(*) "
            f"FROM {self.table} GROUP BY 1, 2, 3, 4 ORDER BY 1"
        ).fetchall()
        return [TranscriptSummary(*r) for r in rows]
