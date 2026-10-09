import numpy as np

from earningsiq.embeddings import HashingEmbedder
from earningsiq.models import Chunk, ScoredChunk, Section
from earningsiq.retrieval import reciprocal_rank_fusion
from earningsiq.store import InMemoryStore, SearchFilters


def _chunk(cid: str, text: str, quarter: int = 3) -> Chunk:
    return Chunk(
        cid,
        f"ACME-2026Q{quarter}",
        "ACME",
        2026,
        quarter,
        "Laura Chen",
        "CFO",
        Section.PREPARED_REMARKS,
        int(cid[-1]),
        text,
    )


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(256)
    a, b = e.embed(["gross margin expanded"]), e.embed(["gross margin expanded"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_similar_text_scores_higher():
    e = HashingEmbedder(512)
    q, near, far = e.embed(
        ["gross margin guidance", "gross margin guidance for Q4", "share buyback"]
    )
    assert q @ near > q @ far


def test_bm25_prefers_exact_terms():
    store = InMemoryStore()
    chunks = [
        _chunk("c0", "Revenue grew strongly this quarter."),
        _chunk("c1", "Free cash flow was $27 million and we have no debt."),
    ]
    store.add(chunks, HashingEmbedder(64).embed([c.text for c in chunks]))
    hits = store.keyword_search("free cash flow", 2)
    assert hits[0].chunk.chunk_id == "c1"


def test_filters_restrict_results():
    store = InMemoryStore()
    chunks = [
        _chunk("c0", "Revenue was $371 million.", quarter=2),
        _chunk("c1", "Revenue was $412 million.", quarter=3),
    ]
    store.add(chunks, HashingEmbedder(64).embed([c.text for c in chunks]))
    hits = store.keyword_search("revenue", 5, SearchFilters(fiscal_quarter=2))
    assert [h.chunk.chunk_id for h in hits] == ["c0"]


def test_reingesting_replaces_previous_chunks():
    store = InMemoryStore()
    emb = HashingEmbedder(64)
    first = [_chunk("c0", "old text about revenue")]
    store.add(first, emb.embed([c.text for c in first]))
    second = [_chunk("c1", "new text about margins")]
    store.add(second, emb.embed([c.text for c in second]))
    assert len(store) == 1
    assert store.list_transcripts()[0].chunk_count == 1


def test_rrf_rewards_agreement():
    a, b, c = (_chunk(f"c{i}", f"text {i}") for i in range(3))
    dense = [ScoredChunk(a, 0.9), ScoredChunk(b, 0.8)]
    sparse = [ScoredChunk(b, 5.0), ScoredChunk(c, 4.0)]
    fused = reciprocal_rank_fusion([dense, sparse], [1.0, 1.0])
    assert fused[0].chunk.chunk_id == "c1"
