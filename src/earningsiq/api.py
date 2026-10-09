"""FastAPI service.

uvicorn earningsiq.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from earningsiq import __version__
from earningsiq.ingest import TranscriptParseError
from earningsiq.models import Section, TranscriptMeta
from earningsiq.pipeline import EarningsIQ
from earningsiq.store import SearchFilters


class IngestRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=10, examples=["ACME"])
    company: str = Field(min_length=1, examples=["Acme Robotics"])
    fiscal_year: int = Field(ge=1990, le=2100, examples=[2026])
    fiscal_quarter: int = Field(ge=1, le=4, examples=[3])
    call_date: str | None = Field(default=None, examples=["2026-10-28"])
    text: str = Field(min_length=50, description="Raw transcript text")


class IngestResponse(BaseModel):
    transcript_id: str
    turns: int
    chunks: int
    speakers: list[str]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    ticker: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = Field(default=None, ge=1, le=4)
    section: Section | None = None
    speaker: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    number: int
    chunk_id: str
    ticker: str
    period: str
    speaker: str
    section: Section
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    model: str
    citations: list[CitationOut]


class TranscriptOut(BaseModel):
    transcript_id: str
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    chunk_count: int


def create_app(engine: EarningsIQ | None = None) -> FastAPI:
    app = FastAPI(
        title="EarningsIQ",
        version=__version__,
        description="Ask questions about earnings calls and get answers with exact citations.",
    )
    app.state.engine = engine or EarningsIQ.from_settings()

    def get_engine(request: Request) -> EarningsIQ:
        return request.app.state.engine

    @app.get("/health")
    def health(request: Request) -> dict:
        engine = get_engine(request)
        return {
            "status": "ok",
            "version": __version__,
            "embedder": engine.embedder.name,
            "generator": engine.generator.name,
            "transcripts": len(engine.store.list_transcripts()),
        }

    @app.post("/transcripts", response_model=IngestResponse, status_code=201)
    def ingest(body: IngestRequest, request: Request) -> IngestResponse:
        meta = TranscriptMeta(
            ticker=body.ticker.upper(),
            company=body.company,
            fiscal_year=body.fiscal_year,
            fiscal_quarter=body.fiscal_quarter,
            call_date=body.call_date,
        )
        try:
            result = get_engine(request).ingest(body.text, meta)
        except (TranscriptParseError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestResponse(**result.__dict__)

    @app.get("/transcripts", response_model=list[TranscriptOut])
    def list_transcripts(request: Request) -> list[TranscriptOut]:
        return [TranscriptOut(**t.__dict__) for t in get_engine(request).store.list_transcripts()]

    @app.delete("/transcripts/{transcript_id}")
    def delete_transcript(transcript_id: str, request: Request) -> dict:
        removed = get_engine(request).store.delete_transcript(transcript_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"{transcript_id} not found")
        return {"transcript_id": transcript_id, "chunks_removed": removed}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        filters = SearchFilters(
            ticker=body.ticker,
            fiscal_year=body.fiscal_year,
            fiscal_quarter=body.fiscal_quarter,
            section=body.section,
            speaker=body.speaker,
        )
        answer = get_engine(request).ask(body.question, filters, body.top_k)
        return AskResponse(
            question=answer.question,
            answer=answer.text,
            grounded=answer.grounded,
            model=answer.model,
            citations=[CitationOut(**c.__dict__) for c in answer.citations],
        )

    return app


app = create_app()
