# EarningsIQ Roadmap

Where this is going: from a cited Q&A tool into a **research platform that turns earnings
calls into measured, point-in-time signals**, with the engineering discipline of a production
system (benchmarks, typed code, reproducible evaluation).

Each item is one solid day of work: code + tests + docs, shipped to `main` green. Every item
that touches quality or speed reports a number (before → after) in its commit and in
`docs/RESULTS.md`. Check an item off in the same commit that completes it and add a line under
**Changelog**.

## v0.1 — Foundation ✅
- [x] Speaker-aware transcript parser (prepared remarks vs Q&A, speaker roles)
- [x] Sentence-window chunking that never crosses a speaker boundary
- [x] Hashing embedder (offline) + OpenAI embedder
- [x] In-memory store (cosine + BM25) and Postgres/pgvector store (HNSW + full-text)
- [x] Hybrid retrieval with Reciprocal Rank Fusion and metadata filters
- [x] Citation enforcement: invalid refs stripped, ungrounded answers flagged
- [x] Extractive, Claude and OpenAI answer generators
- [x] FastAPI service, CLI, Docker Compose, CI with a real pgvector service

## v0.2 — Measured quality
- [ ] **Evaluation harness**: golden Q&A set (30+ questions over the sample calls, incl. unanswerable ones); retrieval recall@k and MRR, citation precision, exact-figure match, refusal accuracy; `make eval` writes `docs/RESULTS.md`; CI fails on regression
- [ ] **Number verification**: every figure in an answer must appear in a cited source (normalising $412M / $412 million / 412.0); unverified figures flagged; hallucinated-number rate added to eval
- [ ] **Engineering bar**: mypy `--strict` clean, property-based tests (Hypothesis) for parser/chunker invariants, coverage gate in CI
- [ ] **Query understanding**: parse "Q3 2026", "last quarter", "the CFO", "$ACME" into filters; measured on the eval set
- [ ] **Reranking**: pluggable reranker after fusion (cross-encoder or LLM), with an ablation table (BM25 / dense / hybrid / hybrid+rerank)

## v0.3 — Performance
- [ ] **Benchmark suite**: synthetic corpus generator (10k → 1M chunks); ingest throughput and query p50/p95/p99 for each store; `make bench` publishes a table
- [ ] **Vectorised hot paths**: batched BM25 with sparse matrices, top-k via `argpartition`, float16 storage option; report speedups
- [ ] **ANN tuning study**: HNSW `m` / `ef_search` sweep on pgvector, recall vs latency curve in `docs/RESULTS.md`
- [ ] **Persistent local index**: memory-mapped `.npy` + JSON so `ingest` and `ask` share data without Postgres; zero-copy load
- [ ] **Async ingestion**: job queue + worker, idempotent re-ingest, `GET /jobs/{id}`

## v0.4 — Real data, point-in-time
- [ ] **SEC EDGAR ingestion**: 8-K earnings releases (Exhibit 99.1) by ticker with acceptance timestamps; respectful rate limiting and caching
- [ ] **Point-in-time corpus**: every document carries `available_at`; queries and signals take an `as_of` time and can never see the future (tested)
- [ ] **KPI extraction**: revenue, margins, ARR, EPS and guidance ranges into a typed `metrics` table, each value with its source citation
- [ ] **Guidance tracker**: guidance in quarter N vs actuals in N+1 → beat / in-line / miss, plus guidance revisions between calls

## v0.5 — Signal research
- [ ] **Text signals**: guidance-revision direction, hedging/uncertainty density, Q&A evasiveness (answer–question similarity), prepared-vs-Q&A tone gap; each documented with its definition
- [ ] **Event-study engine**: abnormal returns vs a market model around call dates (CAR[0,+1], [+2,+20]); price data behind an interface with a synthetic provider for tests
- [ ] **Signal evaluation**: information coefficient, quantile spreads, t-stats with Newey-West errors, decay curves; strict no-lookahead checks
- [ ] **Research report**: `earningsiq research` produces a reproducible notebook/HTML report of signal performance

## v0.6 — Product
- [ ] **Streaming answers** over Server-Sent Events, citations at the end
- [ ] **Web UI**: chat with a citations panel that highlights the source passage
- [ ] **Observability**: structured logs, Prometheus metrics (latency, grounded rate), OpenTelemetry traces
- [ ] **Caching**: embedding and answer caches keyed by (question, filters, corpus version)
- [ ] **Deploy**: image on GHCR and a one-click deploy config

## Changelog
- 2026-10-09 — v0.1.0: first public version (parser, chunking, hybrid retrieval, citations, API, CLI, CI)
