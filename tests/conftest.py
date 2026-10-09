from pathlib import Path

import pytest

from earningsiq.embeddings import HashingEmbedder
from earningsiq.generation import ExtractiveGenerator
from earningsiq.models import TranscriptMeta
from earningsiq.pipeline import EarningsIQ
from earningsiq.store import InMemoryStore

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture
def q3_text() -> str:
    return (SAMPLES / "ACME_2026Q3.txt").read_text()


@pytest.fixture
def q2_text() -> str:
    return (SAMPLES / "ACME_2026Q2.txt").read_text()


@pytest.fixture
def q3_meta() -> TranscriptMeta:
    return TranscriptMeta("ACME", "Acme Robotics", 2026, 3)


@pytest.fixture
def q2_meta() -> TranscriptMeta:
    return TranscriptMeta("ACME", "Acme Robotics", 2026, 2)


@pytest.fixture
def engine(q2_text, q3_text, q2_meta, q3_meta) -> EarningsIQ:
    e = EarningsIQ(InMemoryStore(), HashingEmbedder(512), ExtractiveGenerator())
    e.ingest(q2_text, q2_meta)
    e.ingest(q3_text, q3_meta)
    return e
