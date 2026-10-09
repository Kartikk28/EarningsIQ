"""Embedding backends.

``HashingEmbedder`` is deterministic, dependency-free and fast. It lets the whole
pipeline (and CI) run offline. ``OpenAIEmbedder`` gives real semantic vectors
when an API key is configured.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were will with we our you your they their i what did does do how which why when who "
    "say said about can could would should there them".split()
)
# Longest first. Keeps a stem of at least 4 characters so short words survive.
_SUFFIXES = (
    "ations",
    "ation",
    "itive",
    "itions",
    "ition",
    "ions",
    "ion",
    "ings",
    "ing",
    "ive",
    "ed",
    "es",
    "s",
)


def stem(token: str) -> str:
    """Tiny suffix stripper: 'competition', 'competitive' -> 'compet'."""
    if not token.isalpha():
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def raw_terms(text: str) -> list[str]:
    """Lower-cased tokens without stopwords, unstemmed (for engines that stem themselves)."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in raw_terms(text)]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""
        ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing over unigrams and bigrams with log term frequency."""

    def __init__(self, dim: int = 512):
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, (1.0 if (value >> 63) & 1 else -1.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            counts: dict[str, int] = {}
            for f in features:
                counts[f] = counts.get(f, 0) + 1
            for feature, count in counts.items():
                idx, sign = self._bucket(feature)
                out[row, idx] += sign * (1.0 + math.log(count))
        return _normalise(out)


class OpenAIEmbedder:
    """OpenAI embeddings (requires ``pip install earningsiq[openai]``)."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536, batch: int = 96):
        from openai import OpenAI  # imported lazily so the extra stays optional

        self._client = OpenAI()
        self.model = model
        self.dim = dim
        self.batch = batch
        self.name = f"openai-{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            resp = self._client.embeddings.create(
                model=self.model, input=texts[start : start + self.batch], dimensions=self.dim
            )
            vectors.extend(item.embedding for item in resp.data)
        return _normalise(np.asarray(vectors, dtype=np.float32).reshape(len(texts), self.dim))


def build_embedder(kind: str, dim: int | None = None) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder(dim or 512)
    if kind == "openai":
        return OpenAIEmbedder(dim=dim or 1536)
    raise ValueError(f"Unknown embedder '{kind}' (expected 'hashing' or 'openai')")
