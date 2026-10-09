# EarningsIQ Roadmap

Each item is sized to be one solid day of work: code + tests + docs, shipped to `main` green.
Check an item off in the same commit that completes it, and add a one-line note under
**Changelog** at the bottom.

## v0.1 — Foundation ✅
- [x] Speaker-aware transcript parser (prepared remarks vs Q&A, speaker roles)
- [x] Sentence-window chunking that never crosses a speaker boundary
- [x] Hashing embedder (offline) + OpenAI embedder
- [x] In-memory store (cosine + BM25) and Postgres/pgvector store (HNSW + full-text)
- [x] Hybrid retrieval with Reciprocal Rank Fusion and metadata filters
- [x] Citation enforcement: invalid refs stripped, ungrounded answers flagged
- [x] Extractive, Claude and OpenAI answer generators
- [x] FastAPI service, CLI, Docker Compose, CI with a real pgvector service

## v0.2 — Trustworthy answers
- [ ] **Evaluation harness**: golden Q&A set over the sample calls; metrics for retrieval recall@k, citation precision and exact figure match; `make eval`; CI fails below a threshold
- [ ] **Number verification**: every figure in a generated answer must appear in a cited source; unverified figures are flagged in the response
- [ ] **Query understanding**: turn "Q3 2026", "last quarter", "the CFO", "$ACME" in the question into filters automatically
- [ ] **Reranking**: optional cross-encoder / LLM reranker stage after fusion, measured with the eval harness
- [ ] **Persistent local store**: save/load the in-memory index (`.npz` + JSON) so `ingest` and `ask` share data without Postgres

## v0.3 — Real data
- [ ] **SEC EDGAR ingestion**: fetch 8-K earnings press releases (Exhibit 99.1) by ticker and index them alongside transcripts
- [ ] **More input formats**: PDF and HTML transcript parsing with the same speaker model
- [ ] **Async ingestion jobs**: `POST /transcripts` returns a job id; background worker; `GET /jobs/{id}`

## v0.4 — Analyst features
- [ ] **KPI extraction**: pull revenue, gross margin, operating margin, ARR, guidance ranges into a structured `metrics` table with source citations
- [ ] **Quarter-over-quarter compare**: `POST /compare` returns a metric across periods as a table plus a cited narrative
- [ ] **Guidance tracker**: guidance given in quarter N vs actuals reported in N+1 (beat / in-line / miss)
- [ ] **Call summaries**: per-call bullet summary (results, guidance, risks, Q&A themes), every bullet cited
- [ ] **Tone and hedging signals**: flag hedging language and tone shifts between prepared remarks and Q&A

## v0.5 — Product
- [ ] **Streaming answers**: Server-Sent Events endpoint for token streaming with citations at the end
- [ ] **Web UI**: chat with a citations side panel that highlights the source passage
- [ ] **API keys and rate limiting**
- [ ] **Caching**: embedding cache and answer cache keyed by (question, filters, corpus version)
- [ ] **Observability**: structured JSON logs, Prometheus metrics (latency, grounded rate), OpenTelemetry traces
- [ ] **Deploy**: publish the image to GHCR and add a one-click deploy config

## Changelog
- 2026-10-09 — v0.1.0: first public version (parser, chunking, hybrid retrieval, citations, API, CLI, CI)
