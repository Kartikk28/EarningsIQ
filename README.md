# EarningsIQ

Ask questions about earnings calls and get answers you can check.

Every sentence in an EarningsIQ answer points back to the exact passage it came from: which
company, which quarter, who said it, and whether it was in the prepared remarks or the Q&A.
If the calls don't contain the answer, it says so instead of guessing.

```text
$ earningsiq demo
> What did the CFO guide for Q4 revenue?

For Q4 fiscal 2026, we expect revenue between $440 million and $455 million and gross
margin between 46% and 47%. [1]

  [1] ACME Q3 FY2026 · Laura Chen · prepared_remarks
```

## How it works

```mermaid
flowchart LR
    subgraph Ingest
        A[Raw transcript] --> B[Speaker-aware parser]
        B --> C[Sentence-window chunker<br/>one speaker per chunk]
        C --> D[Embedder]
    end
    D --> E[(Store<br/>pgvector or in-memory)]
    subgraph Ask
        Q[Question + filters] --> V[Vector search]
        Q --> K[Keyword search<br/>BM25 / Postgres FTS]
        E --> V
        E --> K
        V --> F[Reciprocal Rank Fusion]
        K --> F
        F --> G[Generator<br/>Claude · OpenAI · extractive]
        G --> H[Citation check<br/>drop invalid refs, flag ungrounded]
    end
    H --> R[Answer + numbered sources]
```

A few decisions that matter:

- **Chunks never cross a speaker.** A passage is always attributable to one person, so a
  citation can say "the CFO said this in Q3", not "somewhere in this call".
- **Hybrid retrieval.** Embeddings catch paraphrases ("how did margins do" vs "gross margin
  expanded"); keyword search catches exact figures and product names that embeddings blur.
  Reciprocal Rank Fusion merges the two without tuning score scales.
- **Citations are enforced, not requested.** The model is told to cite, and then the output
  is checked: references to sources that weren't retrieved are removed, and an answer with no
  valid citation comes back with `grounded: false`.
- **Runs with zero API keys.** The default hashing embedder and extractive answerer make the
  full pipeline, tests and CI work offline. Switch to OpenAI embeddings and Claude or OpenAI
  generation with environment variables.

## Quickstart

```bash
pip install -e ".[dev]"
earningsiq demo                 # ingests the sample calls, then asks you for questions
make test
```

With Postgres + pgvector:

```bash
docker compose up --build      # API on http://localhost:8000, docs at /docs
```

## API

```bash
# Ingest a transcript
curl -X POST localhost:8000/transcripts -H 'content-type: application/json' -d @- <<'JSON'
{"ticker": "ACME", "company": "Acme Robotics", "fiscal_year": 2026, "fiscal_quarter": 3,
 "text": "Laura Chen -- Chief Financial Officer\nGross margin was 46.2% ..."}
JSON

# Ask, optionally filtered by ticker, year, quarter, section (prepared_remarks | qa) or speaker
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What was gross margin?", "ticker": "ACME", "fiscal_quarter": 3}'
```

| Endpoint | What it does |
| --- | --- |
| `POST /transcripts` | Parse, chunk, embed and index a transcript (re-ingesting replaces it) |
| `GET /transcripts` | List indexed calls with chunk counts |
| `DELETE /transcripts/{id}` | Remove a call, e.g. `ACME-2026Q3` |
| `POST /ask` | Answer a question with numbered citations |
| `GET /health` | Status plus active embedder and generator |

## Configuration

| Variable | Default | Options |
| --- | --- | --- |
| `EARNINGSIQ_DATABASE_URL` | unset (in-memory) | `postgresql://user:pass@host:5432/db` |
| `EARNINGSIQ_EMBEDDER` | `hashing` | `hashing`, `openai` |
| `EARNINGSIQ_EMBEDDING_DIM` | `512` | any; must match the table when using pgvector |
| `EARNINGSIQ_GENERATOR` | `extractive` | `extractive`, `anthropic`, `openai` |
| `EARNINGSIQ_GENERATOR_MODEL` | provider default | e.g. `claude-sonnet-5-5`, `gpt-4o-mini` |
| `EARNINGSIQ_TOP_K` | `6` | sources passed to the generator |

`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` are read by the provider SDKs. See `.env.example`.

## Project layout

```text
src/earningsiq/
  ingest/        parser.py (speaker turns, sections) · chunking.py
  store/         memory.py (cosine + BM25) · pgvector.py (HNSW + full-text)
  generation/    generators.py (extractive, Claude, OpenAI) · citations.py
  embeddings.py  lexicon.py  retrieval.py  pipeline.py  api.py  cli.py
data/samples/    fictional sample calls used by the demo and tests
```

## Roadmap

Next up: an evaluation harness, number verification for generated answers, automatic
filters from the question ("last quarter", "the CFO"), and SEC EDGAR ingestion.
See [ROADMAP.md](ROADMAP.md).

The sample transcripts in `data/samples/` are fictional; the company, people and figures are
invented for testing.

## License

MIT
