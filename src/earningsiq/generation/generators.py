"""Answer generators.

All generators share one contract: given a question and numbered sources, return
text whose factual claims carry ``[n]`` references to those sources.
"""

from __future__ import annotations

import re
from typing import Protocol

from earningsiq.embeddings import tokenize
from earningsiq.generation.citations import build_context
from earningsiq.ingest.chunking import split_sentences
from earningsiq.lexicon import expand_terms
from earningsiq.models import ScoredChunk

NOT_FOUND = "I couldn't find this in the ingested earnings calls."
_FIGURE = re.compile(r"\$\d|\d[\d,.]*\s?(%|million|billion|basis points|bps|units|weeks)")

SYSTEM_PROMPT = f"""You are EarningsIQ, an analyst assistant that answers questions about \
earnings calls using ONLY the numbered sources provided.

Rules:
- Every factual sentence must end with one or more source references like [1] or [2][3].
- Quote figures exactly as stated in the sources (units, currency, period). Never compute \
or estimate numbers that are not in the sources.
- Attribute forward-looking statements to the speaker ("the CFO guided ...").
- If the sources don't answer the question, reply exactly: "{NOT_FOUND}"
- Be concise: 2-5 sentences unless the question asks for a list."""


def user_prompt(question: str, hits: list[ScoredChunk]) -> str:
    return f"Sources:\n\n{build_context(hits)}\n\nQuestion: {question}"


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[ScoredChunk]) -> str: ...


class ExtractiveGenerator:
    """No-LLM fallback: picks the source sentences that best overlap the question.

    Always grounded by construction, so the API is useful (and testable) without
    any API key.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_overlap: int = 1):
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, question: str, hits: list[ScoredChunk]) -> str:
        q_terms = set(tokenize(question))
        if not hits or not q_terms:
            return NOT_FOUND
        synonyms = expand_terms(q_terms) - q_terms
        candidates: list[tuple[float, int, int, str]] = []
        for n, hit in enumerate(hits, 1):
            for pos, sentence in enumerate(split_sentences(hit.chunk.text)):
                if sentence.rstrip().endswith("?"):
                    continue  # an analyst's question is not evidence
                terms = set(tokenize(sentence))
                exact, related = len(q_terms & terms), len(synonyms & terms)
                if exact + related < self.min_overlap:
                    continue
                # Exact matches beat synonyms; sentences with figures are usually
                # what an earnings question is after.
                score = exact + 0.5 * related + (0.3 if _FIGURE.search(sentence) else 0.0)
                candidates.append((score + hit.score, -n, -pos, sentence))
        if not candidates:
            return NOT_FOUND
        candidates.sort(reverse=True)
        picked = []
        seen: set[str] = set()
        for _, neg_n, _, sentence in candidates:
            if sentence in seen:
                continue
            seen.add(sentence)
            picked.append(f"{sentence} [{-neg_n}]")
            if len(picked) == self.max_sentences:
                break
        return " ".join(picked)


class AnthropicGenerator:
    """Claude via the Anthropic API (``pip install earningsiq[anthropic]``)."""

    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 600):
        import anthropic

        self._client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens
        self.name = f"anthropic:{model}"

    def generate(self, question: str, hits: list[ScoredChunk]) -> str:
        if not hits:
            return NOT_FOUND
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt(question, hits)}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


class OpenAIGenerator:
    """OpenAI chat models (``pip install earningsiq[openai]``)."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self._client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def generate(self, question: str, hits: list[ScoredChunk]) -> str:
        if not hits:
            return NOT_FOUND
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, hits)},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def build_generator(kind: str, model: str | None = None) -> Generator:
    if kind == "extractive":
        return ExtractiveGenerator()
    if kind == "anthropic":
        return AnthropicGenerator(model) if model else AnthropicGenerator()
    if kind == "openai":
        return OpenAIGenerator(model) if model else OpenAIGenerator()
    raise ValueError(f"Unknown generator '{kind}' (expected extractive, anthropic or openai)")
