# Personalized Weekly arXiv Reading List

`arxiv-digest` is a Python 3.12 application for collecting, ranking, and presenting a
traceable weekly reading list from the official arXiv API. This repository implements
Milestones 1–4 plus the local Milestone 4.5 dashboard: validated configuration and
models, polite retrieval, normalization, deduplication, deterministic hybrid ranking,
history filtering, diversity selection, abstract-grounded summaries, canonical JSON,
Markdown/HTML reports, and a read-only local viewer.

The default workflow is fully offline after candidate retrieval and needs no API key or
paid service. An optional OpenAI provider is available for summaries, with strict
structured-output validation and deterministic per-paper fallback.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run arxiv-digest doctor
uv run arxiv-digest show-config
```

The dashboard is an optional runtime dependency. Install it when you want the local UI:

```bash
uv sync --extra dashboard
```

The normal development environment includes Streamlit so dashboard tests run with the
standard `uv sync` command.

The default configuration uses `researcher@example.org` in the arXiv `User-Agent`.
Before a live fetch, replace it with a monitored contact address in `config/app.yaml`
or set:

```text
ARXIV_DIGEST_APP__ARXIV__CONTACT_EMAIL=you@example.org
```

Environment variables use the `ARXIV_DIGEST_` prefix and double underscores for
nested fields. They override YAML; command-line options select the YAML files. The
precedence is:

1. nested environment variables;
2. the selected YAML files;
3. schema defaults for omitted optional settings.

`.env.example` documents safe examples. The application does not load `.env` files
itself, and generated `.env` files are ignored by Git.

## Configuration

- `config/app.yaml` controls paths, explicit HTTP timeouts, page size, bounded retries,
  exponential backoff, the minimum three-second request interval, overlap days, summary
  provider limits, report timezone, and near-miss count.
- `config/research_profile.yaml` contains six broad editable queries plus the research
  interests and future ranking settings. Category membership is retained as metadata
  and is not a retrieval hard filter.

The default query groups cover spin ice and pyrochlores, neutron scattering, spin-wave
fitting, frustrated spinels, spin caloritronics, and relevant computational methods.
Phrases are quoted and grouped. Each topic uses a documented `submittedDate`-bounded
pass for new records and a `lastUpdatedDate`-sorted pass for revisions; the latter is
stopped and filtered using parsed UTC timestamps because arXiv exposes
`lastUpdatedDate` as a sort mode, not a search field.

## Fetching candidates

Retrieve the previous seven complete UTC days plus the configured overlap:

```bash
uv run arxiv-digest fetch --days 7
```

Or use an explicit inclusive UTC date range and output file:

```bash
uv run arxiv-digest fetch \
  --start 2026-07-20 \
  --end 2026-07-27 \
  --output data/candidates.json
```

PowerShell users can put that command on one line or use PowerShell backticks instead
of the displayed POSIX continuations.

Snapshots contain a deterministic run ID, generation timestamp, half-open UTC
retrieval window, per-query retrieval counts, raw and deduplicated counts, and the
validated paper metadata. The default filename is stable for a window, so repeating a
fetch replaces that snapshot atomically instead of accumulating duplicates.

arXiv identities are stored without the version suffix; the retrieved version remains
a separate integer. Results are filtered by explicit published/updated timestamps,
then deduplicated across overlapping queries, versions, and punctuation-only title
variants. The newest version wins and category metadata is merged.

## Ranking an existing snapshot

`rank` reads the default same-window candidate snapshot and never retrieves implicitly:

```bash
uv run arxiv-digest rank --days 7
```

Use an explicit snapshot or deliberately permit retrieval when the default is absent:

```bash
uv run arxiv-digest rank --snapshot data/candidates.json --output data/ranked.json
uv run arxiv-digest rank --days 7 --fetch-missing
```

Ranking combines corpus TF-IDF cosine similarity, weighted whole-token profile terms,
category preference, UTC recency, and novelty. Title matches outweigh abstract matches;
negative terms apply a bounded penalty; author boosts are supported. All components and
the final score are in `0..1`, and every score records its strongest profile matches.
Weights are normalized from `config/research_profile.yaml`.

## Weekly report run

Run retrieval when needed, ranking, top-ten selection, deterministic summaries, and
report rendering without paid APIs:

```bash
uv run arxiv-digest run --days 7 --limit 10 --dry-run
```

For a network-free run, supply an existing candidate snapshot:

```bash
uv run arxiv-digest run \
  --snapshot data/candidates.json \
  --limit 10 \
  --dry-run
```

The run writes stable ranked and selection JSON artifacts plus:

- `reports/YYYY-MM-DD-weekly-arxiv-digest.md`;
- `reports/YYYY-MM-DD-weekly-arxiv-digest.html`;
- `reports/YYYY-MM-DD-weekly-arxiv-digest.json`;
- `reports/latest.md`;
- `reports/latest.html`;
- `reports/latest.json`.

The versioned digest JSON is the canonical presentation artifact. It contains report
metadata, the full ranked candidate pool, selections with validated summaries, and up
to five near misses. Markdown, HTML, and the dashboard all consume that same validated
model; the dashboard never parses rendered report text.

Every report prominently labels summaries as based only on the title, abstract, and
metadata. `--dry-run` always forces the offline summary provider and leaves history
unchanged. Without it, history is appended only after all reports are written
successfully. Repeating the same source/profile/limit reproduces the same selection and
does not duplicate history.

Selection uses deterministic maximal marginal relevance over title-and-abstract term
vectors, configurable direct/adjacent/wildcard targets, a minimum score for every
bucket, and a per-topic cap. Targets are relaxed when a bucket is unavailable, but the
minimum wildcard threshold is never relaxed merely to fill the requested limit.

Recent recommendations are excluded for 90 days by default. A recent paper can
resurface only when resurfacing is enabled, its version differs, and its update
timestamp is later than the prior recommendation timestamp.

## Local dashboard

After a completed run has written `reports/latest.json`, start the local read-only
dashboard:

```bash
uv run arxiv-digest dashboard
```

Choose a different canonical report or history directory when needed:

```bash
uv run arxiv-digest dashboard --report reports/2026-07-28-weekly-arxiv-digest.json --reports-dir reports
```

The launcher uses the active Python interpreter directly (not a shell), binds the
server to `127.0.0.1`, and disables Streamlit usage telemetry. Opening the dashboard
does not run retrieval, ranking, summarization, history updates, email, or any other
pipeline stage. It needs no API key and performs no network access; the arXiv abstract
and PDF buttons navigate only when the user clicks them.

The four views are:

- **Current Digest** — report metadata, abstract-only notice, recommendation cards,
  score components, links, and near misses;
- **Candidate Explorer** — the complete ranked pool with selected/near-miss status,
  text/category/type/score/date filters, and deterministic sorting;
- **History** — dated canonical JSON reports, prior recommendations, repeated papers,
  and updated-version resurfacing where local report history demonstrates it;
- **Paper Detail** — complete stored metadata, abstract, selection status, score
  explanation, links, and a validated summary when one was generated.

Filter and view choices live only in Streamlit session state and are discarded when the
session ends. The dashboard does not write notes, reading status, feedback, report
files, or history. Missing, malformed, incompatible, or individually vanished history
files produce visible actionable messages while valid local reports remain usable.

## Summary providers

The default `summarization.provider: offline` uses a deterministic extractive provider.
It never reads PDFs and works without network access. Summary validation enforces the
35-word takeaway ceiling, concise field limits, normalized method labels, confidence in
`0..1`, and the exact abstract-only basis statement.

To enable optional OpenAI summaries, set the model at runtime and select the provider:

```text
OPENAI_API_KEY=...
OPENAI_SUMMARY_MODEL=...
ARXIV_DIGEST_APP__SUMMARIZATION__PROVIDER=openai
```

The provider uses the Responses API with Pydantic structured output. It sends only the
paper title, abstract, authors, categories, submission/update dates, and relevant profile
context—never PDF content or PDF URLs. Invalid or failed responses are retried within the
configured bound, then replaced by the deterministic offline summary without dropping
the paper. `OPENAI_EMBEDDING_MODEL` configures the optional embedding-provider interface;
Milestone 4 does not replace the deterministic TF-IDF ranker with paid embeddings.

Model names intentionally have no repository default. Environment variables
`OPENAI_SUMMARY_MODEL` and `OPENAI_EMBEDDING_MODEL` override the corresponding YAML
fields directly. Nested `ARXIV_DIGEST_...` overrides remain available for every setting.

## arXiv API behavior

The client uses one persistent `httpx` connection per run and a descriptive contact
`User-Agent`. It requests modest pages, waits at least three seconds between network
requests, retries only transient transport/status failures with bounded exponential
backoff, caches identical raw pages during a run, and never retrieves metadata one
paper at a time. HTTP and malformed-feed failures produce concise nonzero CLI errors.
Normal logs include query parameters and counts, never abstracts or secrets.

## Development and tests

Normal tests use checked-in Atom fixtures and HTTP mocks; they require neither network
access nor credentials:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=src/arxiv_digest --cov-report=term-missing
```

An optional live smoke test is excluded by default. Run it only after setting a real
contact address:

```bash
uv run pytest -m live tests/test_live_arxiv.py
```

## Current limitations

- Feedback affinity is reported as a neutral component; user feedback adaptation is
  deferred to Milestone 6 and does not affect ranking yet.
- Recommendation history is local append-only JSONL. There is no database or hosted
  persistence.
- Offline summaries are extractive and can be terse when an arXiv abstract is short.
- OpenAI behavior is covered with injected fakes in normal tests; live paid API tests are
  intentionally not part of the suite.
- The dashboard is local and read-only; there is no hosted frontend, authentication,
  multi-user state, feedback editing, or mobile-specific design.
- No email delivery, PDF parsing, database framework, or GitHub Actions workflow is
  included yet.

The recommended next task is Milestone 5: optional SMTP delivery and the weekly GitHub
Actions workflow, with an honest persistence strategy for ephemeral hosted runners.
