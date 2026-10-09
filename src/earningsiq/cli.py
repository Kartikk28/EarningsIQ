"""Command line interface.

    earningsiq ingest data/samples            # files named TICKER_YYYYQn.txt
    earningsiq ask "What did the CFO guide for Q4 revenue?" --ticker ACME
    earningsiq demo                            # ingest samples + interactive Q&A

With no EARNINGSIQ_DATABASE_URL set, storage is in-memory, so ``ingest`` and
``ask`` only share data inside ``demo`` (or ``ask --ingest DIR``).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from earningsiq.models import Answer, TranscriptMeta
from earningsiq.pipeline import EarningsIQ
from earningsiq.store import SearchFilters

_FILENAME = re.compile(r"^(?P<ticker>[A-Za-z.]{1,10})_(?P<year>\d{4})Q(?P<q>[1-4])$")
SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


def meta_from_path(path: Path) -> TranscriptMeta:
    m = _FILENAME.match(path.stem)
    if not m:
        raise ValueError(f"{path.name}: expected a name like ACME_2026Q3.txt")
    ticker = m["ticker"].upper()
    return TranscriptMeta(ticker, ticker, int(m["year"]), int(m["q"]))


def ingest_dir(engine: EarningsIQ, directory: Path) -> int:
    files = sorted(directory.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt transcripts found in {directory}")
    for f in files:
        result = engine.ingest(f.read_text(encoding="utf-8"), meta_from_path(f))
        print(
            f"  ✓ {result.transcript_id}: {result.chunks} chunks, {len(result.speakers)} speakers"
        )
    return len(files)


def print_answer(answer: Answer) -> None:
    print(f"\n{answer.text}\n")
    for c in answer.citations:
        print(f"  [{c.number}] {c.ticker} {c.period} · {c.speaker} · {c.section.value}")
    if not answer.grounded:
        print("  (no supporting source found)")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="earningsiq", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a directory of transcripts")
    p_ingest.add_argument("directory", type=Path)

    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--ticker")
    p_ask.add_argument("--year", type=int)
    p_ask.add_argument("--quarter", type=int, choices=[1, 2, 3, 4])
    p_ask.add_argument("--ingest", type=Path, help="Ingest this directory first")

    p_demo = sub.add_parser("demo", help="Ingest sample calls and start interactive Q&A")
    p_demo.add_argument("--directory", type=Path, default=SAMPLES)

    args = parser.parse_args(argv)
    engine = EarningsIQ.from_settings()

    if args.command == "ingest":
        ingest_dir(engine, args.directory)
        return 0

    if args.command == "ask":
        if args.ingest:
            ingest_dir(engine, args.ingest)
        filters = SearchFilters(
            ticker=args.ticker, fiscal_year=args.year, fiscal_quarter=args.quarter
        )
        print_answer(engine.ask(args.question, filters))
        return 0

    print(f"Ingesting sample calls from {args.directory} …")
    ingest_dir(engine, args.directory)
    print("Ask a question (blank line to quit).")
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("> "), ""):
        question = line.strip()
        if not question:
            break
        print_answer(engine.ask(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
