"""Parse raw earnings call transcripts into speaker turns.

Supported speaker formats (one per line):

    Jane Doe -- Chief Executive Officer        (header line, speech follows)
    Jane Doe - CFO                             (header line, speech follows)
    Jane Doe: Thanks everyone for joining...   (inline speech)

Section markers such as "Prepared Remarks" and "Questions and Answers" switch the
section that subsequent turns are tagged with.
"""

from __future__ import annotations

import re

from earningsiq.models import Section, SpeakerTurn, Transcript, TranscriptMeta

_QA_MARKERS = re.compile(
    r"^\s*(questions?\s*(and|&)\s*answers?|question-and-answer session|q\s*&\s*a)\s*:?\s*$",
    re.IGNORECASE,
)
_PREPARED_MARKERS = re.compile(
    r"^\s*(prepared remarks|presentation|opening remarks)\s*:?\s*$", re.IGNORECASE
)
# "Name -- Role" or "Name - Role" with nothing else on the line.
_HEADER = re.compile(r"^\s*([A-Z][\w.'\- ]{1,60}?)\s+-{1,2}\s+([A-Za-z][\w,&.'()/\- ]{1,80})\s*$")
# "Name: speech" where Name is 1-5 capitalised words (or "Operator").
_INLINE = re.compile(r"^\s*((?:[A-Z][\w.'\-]*\s?){1,5}):\s+(.+)$")

_ROLE_HINTS = (
    "officer",
    "president",
    "ceo",
    "cfo",
    "coo",
    "cto",
    "director",
    "analyst",
    "investor relations",
    "chairman",
    "vp",
    "head of",
    "treasurer",
    "operator",
)


class TranscriptParseError(ValueError):
    pass


def _looks_like_role(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _ROLE_HINTS) or "," in text


def parse_transcript(raw: str, meta: TranscriptMeta) -> Transcript:
    """Split a transcript into ordered speaker turns.

    Raises TranscriptParseError if no speaker turns can be found, since a
    transcript we can't attribute is not useful for cited answers.
    """
    section = Section.UNKNOWN
    turns: list[SpeakerTurn] = []
    speaker: str | None = None
    role: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = " ".join(" ".join(buffer).split())
        if speaker and text:
            turns.append(
                SpeakerTurn(
                    speaker=speaker, role=role, section=section, text=text, index=len(turns)
                )
            )
        buffer = []

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if _QA_MARKERS.match(stripped):
            flush()
            section = Section.QA
            continue
        if _PREPARED_MARKERS.match(stripped):
            flush()
            section = Section.PREPARED_REMARKS
            continue

        header = _HEADER.match(stripped)
        if header and _looks_like_role(header.group(2)):
            flush()
            speaker, role = header.group(1).strip(), header.group(2).strip()
            if section is Section.UNKNOWN:
                section = Section.PREPARED_REMARKS
            continue

        inline = _INLINE.match(stripped)
        if inline:
            flush()
            speaker = inline.group(1).strip()
            role = "Operator" if speaker.lower() == "operator" else None
            if section is Section.UNKNOWN:
                section = Section.PREPARED_REMARKS
            buffer.append(inline.group(2))
            continue

        buffer.append(stripped)

    flush()

    if not turns:
        raise TranscriptParseError(
            f"No speaker turns found in transcript {meta.transcript_id}; "
            "expected lines like 'Name -- Role' or 'Name: text'."
        )
    return Transcript(meta=meta, turns=turns)
