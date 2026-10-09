import pytest

from earningsiq.ingest import TranscriptParseError, chunk_transcript, parse_transcript
from earningsiq.models import Section, TranscriptMeta


def test_parses_speakers_roles_and_sections(q3_text, q3_meta):
    t = parse_transcript(q3_text, q3_meta)
    speakers = t.speakers
    assert {"Daniel Okafor", "Laura Chen", "Michael Rossi", "Sarah Kim"} <= set(speakers)

    cfo_turns = [x for x in t.turns if x.speaker == "Laura Chen"]
    assert cfo_turns[0].role == "Chief Financial Officer"
    assert cfo_turns[0].section is Section.PREPARED_REMARKS
    assert any(x.section is Section.QA for x in cfo_turns)


def test_text_before_first_speaker_is_ignored(q3_text, q3_meta):
    t = parse_transcript(q3_text, q3_meta)
    assert all("Fictional sample transcript" not in x.text for x in t.turns)


def test_inline_speaker_format():
    raw = "Operator: Welcome everyone.\nJane Roe: Revenue grew ten percent this quarter."
    t = parse_transcript(raw, TranscriptMeta("XYZ", "Xyz", 2025, 1))
    assert [x.speaker for x in t.turns] == ["Operator", "Jane Roe"]
    assert t.turns[0].role == "Operator"


def test_unparseable_transcript_raises():
    with pytest.raises(TranscriptParseError):
        parse_transcript(
            "just some text with no speakers at all", TranscriptMeta("X", "X", 2025, 1)
        )


def test_chunks_never_cross_speakers_and_skip_operator(q3_text, q3_meta):
    t = parse_transcript(q3_text, q3_meta)
    chunks = chunk_transcript(t, max_words=60)
    turn_speaker = {x.index: x.speaker for x in t.turns}
    assert chunks
    for c in chunks:
        assert c.speaker == turn_speaker[c.turn_index]
        assert c.speaker != "Operator"
        assert c.transcript_id == "ACME-2026Q3"


def test_long_turns_split_with_overlap(q3_text, q3_meta):
    t = parse_transcript(q3_text, q3_meta)
    chunks = [c for c in chunk_transcript(t, max_words=40) if c.speaker == "Laura Chen"]
    first_turn = [c for c in chunks if c.turn_index == chunks[0].turn_index]
    assert len(first_turn) > 1
    last_sentence = first_turn[0].text.split(". ")[-1]
    assert last_sentence in first_turn[1].text


def test_chunk_ids_are_unique_and_stable(q3_text, q3_meta):
    t = parse_transcript(q3_text, q3_meta)
    a = [c.chunk_id for c in chunk_transcript(t)]
    b = [c.chunk_id for c in chunk_transcript(t)]
    assert a == b
    assert len(set(a)) == len(a)


def test_rejects_bad_chunk_settings(q3_text, q3_meta):
    t = parse_transcript(q3_text, q3_meta)
    with pytest.raises(ValueError):
        chunk_transcript(t, max_words=5)
