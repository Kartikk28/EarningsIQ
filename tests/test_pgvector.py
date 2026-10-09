"""Integration test for the pgvector store.

Runs only when EARNINGSIQ_TEST_DATABASE_URL points at a Postgres with pgvector,
e.g. in CI or via `docker compose up db`.
"""

import os
import uuid

import pytest

from earningsiq.embeddings import HashingEmbedder
from earningsiq.generation import ExtractiveGenerator
from earningsiq.pipeline import EarningsIQ
from earningsiq.store import SearchFilters

DSN = os.getenv("EARNINGSIQ_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="EARNINGSIQ_TEST_DATABASE_URL not set")


@pytest.fixture
def pg_engine():
    from earningsiq.store.pgvector import PgVectorStore

    table = f"test_chunks_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(DSN, dim=256, table=table)
    yield EarningsIQ(store, HashingEmbedder(256), ExtractiveGenerator())
    store._conn.execute(f"DROP TABLE IF EXISTS {table}")


def test_pgvector_round_trip(pg_engine, q2_text, q3_text, q2_meta, q3_meta):
    pg_engine.ingest(q2_text, q2_meta)
    pg_engine.ingest(q3_text, q3_meta)
    pg_engine.ingest(q3_text, q3_meta)  # idempotent re-ingest

    listed = {t.transcript_id: t.chunk_count for t in pg_engine.store.list_transcripts()}
    assert set(listed) == {"ACME-2026Q2", "ACME-2026Q3"}

    ans = pg_engine.ask("What was gross margin?", SearchFilters(fiscal_quarter=3))
    assert ans.grounded and "46.2%" in ans.text

    assert pg_engine.store.delete_transcript("ACME-2026Q2") == listed["ACME-2026Q2"]
