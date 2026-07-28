# Personalized Weekly arXiv Reading List

`arxiv-digest` is a Python 3.12 application for collecting and ranking a traceable
candidate pool from the official arXiv API. This repository currently implements
Milestones 1–3: configuration, validated models, polite retrieval, normalization,
deduplication, deterministic offline hybrid ranking, recent-history filtering, updated
version resurfacing, MMR diversity selection, and machine-readable JSON artifacts.

No API key or paid service is needed. Summaries, reader-facing reports, feedback,
email, and scheduling deliberately remain for later milestones.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run arxiv-digest doctor
uv run arxiv-digest show-config
```

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
  exponential backoff, the minimum three-second request interval, and overlap days.
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

## Diversity-aware run

Run retrieval when needed, ranking, and top-ten selection without paid APIs:

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

The run writes stable ranked and selection JSON artifacts. `--dry-run` leaves history
unchanged. Without it, selected papers are appended to `data/history.jsonl`. Repeating
the same source/profile/limit reproduces the same selection and does not duplicate
history.

Selection uses deterministic maximal marginal relevance over title-and-abstract term
vectors, configurable direct/adjacent/wildcard targets, a minimum score for every
bucket, and a per-topic cap. Targets are relaxed when a bucket is unavailable, but the
minimum wildcard threshold is never relaxed merely to fill the requested limit.

Recent recommendations are excluded for 90 days by default. A recent paper can
resurface only when resurfacing is enabled, its version differs, and its update
timestamp is later than the prior recommendation timestamp.

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

- Milestone 3 produces machine-readable ranked and selection JSON, not the final weekly
  Markdown/HTML report.
- Feedback affinity is reported as a neutral component; user feedback adaptation is
  deferred to Milestone 6 and does not affect ranking yet.
- Recommendation history is local append-only JSONL. There is no database or hosted
  persistence.
- No summaries, report templates, OpenAI integration, email, PDFs, frontend, or GitHub
  Actions workflow are included yet.

The next task should implement Milestone 4 separately: summary-provider interfaces,
deterministic abstract-grounded fallbacks, optional OpenAI summaries, strict validation,
and Markdown/HTML reports clearly labeled as abstract-based.
