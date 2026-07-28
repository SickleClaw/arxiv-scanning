# AGENTS.md

## Project: Personalized Weekly arXiv Reading List

### Mission

Build a reliable Python application that searches arXiv once per week, ranks newly submitted or updated papers against the user's research interests, selects ten papers that are both relevant and varied, generates brief grounded summaries, and produces a readable weekly report.

The application is a research-assistance tool, not a general news aggregator. Optimize for recommendation quality, traceability, low maintenance, and reproducible behavior.

## Current target

Implement a complete minimum viable product that:

1. Retrieves candidate papers from arXiv for a configurable date window.
2. Deduplicates and normalizes arXiv records.
3. Scores candidates against the research profile in `config/research_profile.yaml`.
4. Selects ten papers using relevance plus diversity.
5. Generates abstract-grounded summaries through a provider interface.
6. Writes Markdown and HTML reports.
7. Tracks previously recommended papers.
8. Can run locally and on a weekly GitHub Actions schedule.
9. Has unit and integration tests that do not require network access or paid API calls.

Do not add full-PDF analysis, a web application, a vector database, or a complex frontend during the MVP unless explicitly requested.

## User research profile

The default profile should be tailored to these interests:

### Highest-priority topics

- Spin ice and frustrated magnetism
- Ho2Ti2O7 and related pyrochlores
- Magnetic monopoles, field-driven dynamics, deconfinement, and nonequilibrium dynamics
- Kinetic Monte Carlo, loop and cluster algorithms, and spin-dynamics simulations
- Neutron scattering, neutron diffuse scattering, time-resolved scattering, and dynamical structure factors
- Spectral-weight and momentum-sum-rule analyses
- ZnFe2O4 and frustrated or disordered spinels
- Insulating spin glasses
- Linear spin-wave theory, exchange-parameter fitting, domain averaging, and magnetic excitation spectra

### Secondary topics

- Spin Seebeck effect and spin caloritronics
- Polarized neutron reflectometry
- Magnetic-insulator/Pt heterostructures
- Scientific machine learning, inverse problems, parameter fitting, and research-workflow automation when relevant to condensed-matter physics

### Preferred arXiv categories

- `cond-mat.str-el`
- `cond-mat.mtrl-sci`
- `cond-mat.stat-mech`
- `cond-mat.mes-hall`
- `physics.comp-ph`
- `physics.ins-det`

Category membership is a useful signal, not a hard requirement.

### Recommendation mix

For a normal week, aim for:

- 5–6 directly relevant papers
- 2–3 adjacent methodological or conceptual papers
- 1 wildcard paper with weaker direct overlap but strong potential interest

Do not force this mix when the candidate pool is too small. Never recommend a weak paper solely to fill a bucket.

## Required repository layout

Use this layout unless the existing repository already has a clear equivalent:

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── config/
│   ├── research_profile.yaml
│   └── app.yaml
├── src/
│   └── arxiv_digest/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── arxiv_client.py
│       ├── normalization.py
│       ├── ranking.py
│       ├── selection.py
│       ├── summarization.py
│       ├── history.py
│       ├── reporting.py
│       ├── delivery.py
│       └── pipeline.py
├── templates/
│   ├── weekly_report.md.j2
│   └── weekly_report.html.j2
├── tests/
│   ├── fixtures/
│   ├── test_arxiv_client.py
│   ├── test_normalization.py
│   ├── test_ranking.py
│   ├── test_selection.py
│   ├── test_summarization.py
│   ├── test_history.py
│   └── test_pipeline.py
├── data/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
└── .github/
    └── workflows/
        └── weekly_digest.yml
```

## Technology choices

Use:

- Python 3.12 or newer
- `uv` for environment and dependency management
- `httpx` for HTTP
- `feedparser` for Atom parsing
- `pydantic` and `pydantic-settings` for models and configuration
- `PyYAML` for user-editable profiles
- `typer` for the CLI
- `jinja2` for reports
- `numpy` for vector calculations and maximal marginal relevance
- the current official OpenAI Python SDK for optional LLM summaries and embeddings
- `pytest`, `respx`, and `pytest-cov` for tests
- `ruff` for formatting and linting
- `mypy` for static type checking

Avoid adding a database framework for the MVP. Store recommendation history as append-only JSON Lines in `data/history.jsonl`. Keep feedback in `data/feedback.yaml`.

Do not add dependencies merely for convenience when the standard library is adequate. Use `smtplib` for optional SMTP delivery.

## External-service rules

### arXiv

Use the official arXiv API rather than scraping search-result pages.

The client must:

- use a descriptive `User-Agent` containing the application name and a configurable contact email;
- maintain a single connection;
- enforce at least three seconds between arXiv API requests;
- use bounded retries with exponential backoff for transient errors;
- set explicit connect and read timeouts;
- support pagination;
- cache raw responses during a run;
- never issue one request per paper when metadata is already available in the Atom feed;
- sort and filter using explicit timestamps rather than assuming result order;
- log query parameters but not secrets;
- provide clear errors for malformed feeds and HTTP failures.

Treat the arXiv identifier without its version suffix as the canonical paper identity. Preserve the retrieved version separately.

Use UTC internally. The default retrieval window is the previous seven complete days, with configurable overlap to avoid papers being missed around scheduler boundaries.

### OpenAI

All OpenAI use must be behind interfaces so the application can run without an API key.

Provide:

- `SummaryProvider`
- `EmbeddingProvider`

Implement:

- an OpenAI-backed provider;
- a deterministic offline summary provider for tests and local dry runs;
- a TF-IDF or keyword-based offline relevance fallback when embeddings are unavailable.

Do not hardcode a model name deep in the code. Read model names from environment variables or configuration:

- `OPENAI_SUMMARY_MODEL`
- `OPENAI_EMBEDDING_MODEL`

Use structured outputs or strict JSON validation where supported. Validate every model response with Pydantic and retry a small, bounded number of times when validation fails.

Never send PDFs in the MVP. Send only the paper title, abstract, authors, categories, and dates required for the requested summary.

### Secrets

Never commit:

- API keys
- SMTP passwords
- personal email credentials
- generated `.env` files

Provide `.env.example` with names and descriptions but no real values.

## Core data models

Define explicit Pydantic models with timezone-aware datetimes.

### `Paper`

Required fields:

- `arxiv_id`
- `version`
- `title`
- `authors`
- `abstract`
- `primary_category`
- `categories`
- `published_at`
- `updated_at`
- `abstract_url`
- `pdf_url`
- `doi`, optional
- `journal_reference`, optional

Normalize whitespace in titles and abstracts. Preserve Unicode.

### `ScoreBreakdown`

Include:

- semantic relevance
- keyword relevance
- category relevance
- recency
- novelty
- feedback affinity
- final preselection score
- an explanation containing the strongest matched profile terms

All component scores must have documented ranges.

### `PaperSummary`

Include:

- `one_sentence_takeaway`
- `brief_summary`
- `why_relevant`
- `methods_or_systems`
- `limitations`
- `summary_basis`
- `confidence`

For the MVP, `summary_basis` must explicitly say that the summary is based on the title, abstract, and metadata. Do not imply that the full paper was read.

### `Recommendation`

Combine the paper, score breakdown, summary, rank, and recommendation type:

- `direct`
- `adjacent`
- `wildcard`

## Retrieval strategy

Build a small number of broad arXiv queries rather than one request for every keyword.

The default profile should generate candidate queries covering:

1. Spin ice, pyrochlores, magnetic monopoles, and nonequilibrium magnetic dynamics.
2. Neutron diffuse or inelastic scattering and magnetic structure factors.
3. Linear spin-wave theory and exchange-model fitting.
4. Frustrated spinels, ZnFe2O4, and magnetic disorder.
5. Spin Seebeck effect, spin caloritronics, and polarized neutron reflectometry.
6. Relevant computational or machine-learning methods for condensed-matter research.

The query builder must correctly quote phrases, group Boolean expressions, URL-encode parameters, and constrain the submission date window. Keep query definitions editable in YAML.

Retrieve a configurable candidate pool, with a default target of 100–250 unique papers. If a query returns many results, paginate in modest pages rather than requesting a huge response.

## Deduplication and history

Deduplicate across:

- overlapping queries;
- arXiv versions;
- repeated weekly runs;
- title variants with trivial whitespace or punctuation differences.

Recommendation history must record:

- run ID
- run timestamp
- retrieval window
- arXiv ID and version
- rank
- recommendation type
- final score
- report path

By default, exclude a paper recommended during the previous 90 days. A substantially updated version may be reconsidered only when:

- its `updated_at` timestamp is newer than the stored recommendation;
- the new version number differs; and
- configuration allows updated-paper resurfacing.

All runs must be idempotent for the same run ID and date window.

## Ranking

Implement hybrid ranking in two passes.

### Pass 1: inexpensive preselection

Use title, abstract, categories, and profile configuration.

A reasonable initial weighting is:

```text
0.45 semantic or TF-IDF similarity
0.25 weighted keyword relevance
0.10 category relevance
0.10 recency
0.10 feedback affinity or novelty
```

Weights must be configurable and normalized. Tests should verify that changing weights changes ranking predictably.

Give title matches more weight than abstract matches. Support:

- exact phrases;
- positive weighted terms;
- negative terms;
- category weights;
- material names;
- method names;
- author boosts, optional.

Avoid naive substring matching that produces false positives inside unrelated words.

### Pass 2: diversity-aware selection

Select ten papers using maximal marginal relevance or an equivalent deterministic method.

The selection method must balance:

- relevance to the profile;
- novelty relative to previously recommended papers;
- diversity within the current list;
- the direct/adjacent/wildcard mix.

Do not allow one narrow topic to occupy the entire report when other strong candidates exist. Include a configurable maximum number of near-duplicate-topic papers.

Use a stable tie-break order, such as final score, updated time, and canonical arXiv ID.

## Summarization rules

Generate summaries only after ranking has reduced the pool. The default maximum number of papers sent to the summary provider should be configurable and should not greatly exceed the final ten.

Every generated summary must:

- be grounded only in supplied metadata and abstract text;
- distinguish the authors' claims from established facts;
- avoid claiming that figures, equations, appendices, or results were inspected;
- avoid inventing sample sizes, temperatures, fields, materials, instruments, or numerical results;
- state when the abstract does not provide enough information;
- use language appropriate for a condensed-matter-physics PhD researcher;
- explain why the paper may matter to the user's current projects;
- remain concise.

Target length per paper:

- one-sentence takeaway: no more than 35 words;
- brief summary: approximately 80–130 words;
- why relevant: approximately 30–70 words;
- limitations: one concise sentence.

If LLM summarization fails, include a safe extractive fallback rather than dropping the paper.

## Weekly report

Produce:

- `reports/YYYY-MM-DD-weekly-arxiv-digest.md`
- `reports/YYYY-MM-DD-weekly-arxiv-digest.html`
- `reports/latest.md`
- `reports/latest.html`

The report header must show:

- generation date and timezone;
- paper retrieval window;
- number of candidates retrieved;
- number remaining after deduplication;
- summary basis;
- profile version or hash.

For each paper show:

1. rank and title;
2. authors;
3. submitted and updated dates;
4. primary category;
5. recommendation type;
6. one-sentence takeaway;
7. brief summary;
8. why it was selected;
9. limitations of the available metadata;
10. score breakdown;
11. links to the arXiv abstract and PDF.

Add a final section listing strong candidates that narrowly missed the top ten. Limit this to five.

The report must clearly label summaries as abstract-based.

## Feedback

Support these feedback values:

- `must_read`
- `relevant`
- `interesting_but_peripheral`
- `not_relevant`
- `already_known`
- `poor_recommendation`

Expose a CLI command:

```bash
arxiv-digest feedback ARXIV_ID must_read
```

Feedback should affect future ranking through transparent, bounded adjustments. It must not permanently suppress an entire broad field based on a single negative rating.

## CLI

Provide these commands:

```bash
arxiv-digest doctor
arxiv-digest fetch --days 7
arxiv-digest rank --days 7
arxiv-digest run --days 7 --limit 10
arxiv-digest run --start YYYY-MM-DD --end YYYY-MM-DD --dry-run
arxiv-digest feedback ARXIV_ID VALUE
arxiv-digest show-config
```

Behavior:

- `doctor` validates configuration, writable paths, network settings, and optional credentials without exposing secrets.
- `fetch` retrieves and writes a machine-readable candidate snapshot.
- `rank` ranks an existing snapshot or retrieves one when explicitly requested.
- `run` executes the full pipeline.
- `--dry-run` must not call paid APIs or send email.
- commands must return nonzero exit codes on failures.
- console output should be concise and actionable.

## Email delivery

Email is optional. The application must still be fully useful by writing reports locally.

When SMTP variables are configured, allow the HTML report to be sent with a plain-text alternative.

Expected variables:

```text
SMTP_HOST
SMTP_PORT
SMTP_USERNAME
SMTP_PASSWORD
SMTP_USE_TLS
DIGEST_FROM_EMAIL
DIGEST_TO_EMAIL
```

Do not print credentials. A failed email should not delete or invalidate a successfully generated report.

## GitHub Actions

Create `.github/workflows/weekly_digest.yml`.

Requirements:

- run weekly on Monday at `13:00 UTC`;
- support `workflow_dispatch`;
- install dependencies with `uv`;
- run lint, type checks, and tests before the digest;
- run the digest only if validation passes;
- upload the generated reports as workflow artifacts;
- send email only when required secrets exist;
- use least-privilege permissions;
- use dependency caching;
- avoid exposing secrets in logs.

Do not automatically commit generated reports or history to the repository in the first implementation. Document that GitHub-hosted runners are ephemeral and provide a clearly isolated future option for persistent history. For the initial workflow, history-based filtering is guaranteed for local runs; scheduled runs should use an uploaded/downloaded artifact strategy only if it can be implemented robustly and simply. Do not pretend ephemeral storage is persistent.

## Configuration

Create `config/research_profile.yaml` with editable sections for:

- profile name and version;
- profile description;
- priority topics;
- exact phrases;
- materials;
- methods;
- categories and weights;
- negative terms;
- author boosts;
- ranking weights;
- direct/adjacent/wildcard targets;
- history exclusion period;
- maximum candidate count;
- maximum papers per near-duplicate topic.

Create `config/app.yaml` for operational settings such as paths, timeouts, page size, retry count, rate limiting, report format, and delivery.

Environment variables override YAML values. Clearly document precedence.

## Logging and observability

Use the standard `logging` package.

Include:

- run ID;
- pipeline stage;
- counts before and after filters;
- elapsed time per stage;
- retry notices;
- report paths.

Do not log full abstracts at normal verbosity. Do not log model prompts or secrets by default.

At the end of a successful run, print a compact summary such as:

```text
Retrieved 184 records, deduplicated to 137, ranked 137, summarized 12,
selected 10, wrote reports/...md and reports/...html.
```

## Testing requirements

Tests must run without live network access and without API keys.

Required coverage:

- arXiv query encoding;
- date-window handling and UTC boundaries;
- Atom parsing using checked-in fixtures;
- canonical ID and version extraction;
- whitespace normalization;
- cross-query deduplication;
- history exclusion and resurfacing rules;
- deterministic keyword scoring;
- category weighting;
- ranking-weight normalization;
- maximal marginal relevance behavior;
- direct/adjacent/wildcard allocation;
- summary validation and fallback behavior;
- report rendering;
- pipeline idempotency;
- CLI exit codes;
- secret redaction.

Use `respx` for HTTP mocks. Add one optional live arXiv smoke test marked `live`, excluded from the normal test run.

Aim for at least 85% line coverage in project code, but prioritize meaningful tests over gaming the metric.

## Engineering standards

- Use type annotations for public and internal functions.
- Prefer small pure functions for normalization, scoring, and selection.
- Use dependency injection for the arXiv client, time source, summary provider, embedding provider, and delivery provider.
- Keep network I/O separate from ranking logic.
- Make randomness explicit and seeded; prefer deterministic algorithms.
- Use timezone-aware datetimes only.
- Raise domain-specific exceptions with actionable messages.
- Keep public functions documented.
- Do not suppress exceptions broadly.
- Do not use bare `except`.
- Do not leave placeholder implementations, silent `pass` blocks, or commented-out abandoned code.
- Do not over-engineer abstractions before a second implementation requires them.
- Preserve backward compatibility for config files once the MVP is released.

## Required commands

Configure these commands through `pyproject.toml` or a small `Makefile`:

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=src/arxiv_digest --cov-report=term-missing
uv run arxiv-digest doctor
uv run arxiv-digest run --days 7 --limit 10 --dry-run
```

Before considering a task complete, run every relevant command and report the exact results.

## Definition of done for the MVP

The MVP is complete only when:

- a new user can clone the repository and follow `README.md`;
- `uv sync` succeeds;
- all lint, type, and test checks pass;
- the dry-run pipeline works without credentials;
- a live run can retrieve arXiv data;
- an OpenAI-backed run works when credentials and model names are configured;
- ten recommendations are selected when at least ten viable candidates exist;
- reports are readable and clearly marked as abstract-based;
- rerunning the same window does not duplicate history;
- failures produce actionable messages;
- the GitHub Actions workflow validates and can be manually triggered;
- no secrets or personal credentials are committed.

## Codex working protocol

When implementing changes:

1. Inspect the repository and this file before editing.
2. For a multi-file change, produce a concise implementation plan first.
3. Make the smallest coherent change that advances the current milestone.
4. Add or update tests with the implementation.
5. Run formatting, linting, type checking, and relevant tests.
6. Review the diff for correctness, secret leakage, dead code, and accidental scope expansion.
7. Summarize:
   - files changed;
   - behavior added or changed;
   - commands run and their results;
   - remaining limitations;
   - the next logical task.

Do not claim a command passed unless it was actually run. Do not replace a failed test with a weaker test merely to make CI green.

When requirements are ambiguous, prefer a documented, reversible choice that preserves user control. Ask a question only when the ambiguity blocks a safe implementation.

## Implementation sequence

Work through these milestones in order.

### Milestone 1: repository scaffold

- Create the package layout, configuration loading, models, CLI shell, tooling, and tests.
- Add a complete README and `.env.example`.
- Make `doctor` and `show-config` work.

### Milestone 2: arXiv retrieval

- Implement query construction, rate limiting, retries, pagination, parsing, normalization, deduplication, and fixtures.
- Make `fetch` produce a candidate JSON snapshot.

### Milestone 3: ranking and selection

- Implement offline hybrid scoring, score explanations, history filtering, and diversity-aware top-ten selection.
- Make `rank` and dry-run `run` work without OpenAI.

### Milestone 4: summaries and reports

- Add summary-provider interfaces, deterministic fallback, OpenAI provider, validation, Markdown/HTML reports, and summary tests.
- Ensure every report states that it is abstract-based.

### Milestone 5: delivery and scheduling

- Add optional SMTP delivery.
- Add the weekly GitHub Actions workflow.
- Document persistence limitations of hosted runners honestly.

### Milestone 6: feedback adaptation

- Add the feedback CLI and bounded ranking adjustments.
- Add tests showing positive and negative feedback influence without dominating relevance.

## First Codex task

When asked to begin implementation, do the following:

> Implement Milestones 1 and 2. First inspect the repository and present a concise plan. Then create the Python project, configuration system, data models, CLI, arXiv client, normalization and deduplication logic, candidate snapshot output, tests, and documentation. Use mocked arXiv fixtures for normal tests. Run all relevant checks. Do not implement LLM calls, email, full-PDF parsing, or a frontend in this task.

After Milestones 1 and 2 pass, proceed to Milestone 3 only in a separate task or clearly separated change.
