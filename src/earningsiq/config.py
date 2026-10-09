"""Settings read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    embedder: str
    embedding_dim: int
    generator: str
    generator_model: str | None
    top_k: int
    chunk_max_words: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=os.getenv("EARNINGSIQ_DATABASE_URL") or None,
            embedder=os.getenv("EARNINGSIQ_EMBEDDER", "hashing"),
            embedding_dim=_int("EARNINGSIQ_EMBEDDING_DIM", 512),
            generator=os.getenv("EARNINGSIQ_GENERATOR", "extractive"),
            generator_model=os.getenv("EARNINGSIQ_GENERATOR_MODEL") or None,
            top_k=_int("EARNINGSIQ_TOP_K", 6),
            chunk_max_words=_int("EARNINGSIQ_CHUNK_MAX_WORDS", 180),
        )
