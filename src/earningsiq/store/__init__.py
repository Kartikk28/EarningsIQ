from earningsiq.store.base import SearchFilters, TranscriptSummary, VectorStore
from earningsiq.store.memory import InMemoryStore


def build_store(database_url: str | None, dim: int) -> VectorStore:
    """Use pgvector when a database URL is configured, otherwise in-memory."""
    if database_url:
        from earningsiq.store.pgvector import PgVectorStore

        return PgVectorStore(database_url, dim=dim)
    return InMemoryStore()


__all__ = ["InMemoryStore", "SearchFilters", "TranscriptSummary", "VectorStore", "build_store"]
