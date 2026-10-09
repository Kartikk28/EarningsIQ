from earningsiq.generation import NOT_FOUND, resolve_citations
from earningsiq.models import Chunk, ScoredChunk, Section
from earningsiq.store import SearchFilters


def test_answer_cites_the_right_figures(engine):
    ans = engine.ask("What was gross margin in Q3?", SearchFilters(fiscal_quarter=3))
    assert ans.grounded
    assert "46.2%" in ans.text
    assert ans.citations and all(c.period == "Q3 FY2026" for c in ans.citations)


def test_quarter_filter_changes_the_answer(engine):
    q2 = engine.ask("What was total revenue?", SearchFilters(fiscal_quarter=2))
    q3 = engine.ask("What was total revenue?", SearchFilters(fiscal_quarter=3))
    assert "$371 million" in q2.text
    assert "$412 million" in q3.text


def test_speaker_filter(engine):
    ans = engine.ask("What did they say about competition?", SearchFilters(speaker="Okafor"))
    assert ans.citations
    assert {c.speaker for c in ans.citations} == {"Daniel Okafor"}


def test_qa_section_filter(engine):
    ans = engine.ask("share repurchase program", SearchFilters(section=Section.QA))
    assert "$200 million" in ans.text
    assert all(c.section is Section.QA for c in ans.citations)


def test_unknown_ticker_returns_not_found(engine):
    ans = engine.ask("What was revenue?", SearchFilters(ticker="ZZZZ"))
    assert ans.text == NOT_FOUND
    assert not ans.grounded and not ans.citations


def _hit(cid: str) -> ScoredChunk:
    return ScoredChunk(
        Chunk(
            cid,
            "ACME-2026Q3",
            "ACME",
            2026,
            3,
            "Laura Chen",
            "CFO",
            Section.QA,
            0,
            "Gross margin was 46.2%.",
        ),
        1.0,
    )


def test_invalid_citation_numbers_are_stripped():
    text, cites, grounded = resolve_citations("Margin was 46.2% [1] and rising [7].", [_hit("a")])
    assert "[7]" not in text
    assert [c.number for c in cites] == [1]
    assert grounded


def test_answer_without_citations_is_not_grounded():
    _, cites, grounded = resolve_citations("Margins look great.", [_hit("a")])
    assert not cites and not grounded


def test_synonyms_bridge_vocabulary(engine):
    ans = engine.ask("Is there a buyback?")
    assert ans.grounded
    assert "$200 million" in ans.text


def test_analyst_questions_are_not_used_as_evidence(engine):
    ans = engine.ask("How did gross margin change?", SearchFilters(fiscal_quarter=3))
    assert "?" not in ans.text
    assert "46.2%" in ans.text
