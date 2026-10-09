import pytest
from fastapi.testclient import TestClient

from earningsiq.api import create_app
from earningsiq.embeddings import HashingEmbedder
from earningsiq.generation import ExtractiveGenerator
from earningsiq.pipeline import EarningsIQ
from earningsiq.store import InMemoryStore


@pytest.fixture
def client():
    engine = EarningsIQ(InMemoryStore(), HashingEmbedder(256), ExtractiveGenerator())
    return TestClient(create_app(engine))


def _ingest(client, text, quarter=3):
    return client.post(
        "/transcripts",
        json={
            "ticker": "acme",
            "company": "Acme Robotics",
            "fiscal_year": 2026,
            "fiscal_quarter": quarter,
            "text": text,
        },
    )


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["transcripts"] == 0


def test_ingest_list_ask_delete_flow(client, q3_text):
    r = _ingest(client, q3_text)
    assert r.status_code == 201
    assert r.json()["transcript_id"] == "ACME-2026Q3"

    listed = client.get("/transcripts").json()
    assert listed[0]["ticker"] == "ACME" and listed[0]["chunk_count"] > 0

    ans = client.post("/ask", json={"question": "What is the Q4 revenue guidance?"}).json()
    assert ans["grounded"]
    assert "$440 million" in ans["answer"]
    assert ans["citations"][0]["ticker"] == "ACME"

    assert client.delete("/transcripts/ACME-2026Q3").status_code == 200
    assert client.delete("/transcripts/ACME-2026Q3").status_code == 404


def test_ingest_rejects_unparseable_text(client):
    r = _ingest(client, "this is just a long paragraph with no speakers anywhere in it at all ok")
    assert r.status_code == 422


def test_ask_validates_quarter(client):
    r = client.post("/ask", json={"question": "revenue?", "fiscal_quarter": 7})
    assert r.status_code == 422
