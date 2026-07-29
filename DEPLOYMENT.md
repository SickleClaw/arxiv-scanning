# Production deployment and weekly operations

The production design has two deliberately separate systems:

1. GitHub Actions validates and generates the weekly digest, uploads the complete report
   bundle, commits the minimum canonical JSON/history state, and optionally sends email.
2. Streamlit Community Cloud reads the committed JSON state. It never runs the pipeline,
   uses OpenAI, sends email, or writes repository data.

## GitHub repository configuration

Configure these under **Settings > Secrets and variables > Actions**. Values marked
optional are needed only for their corresponding mode.

| Name | GitHub type | Required when | Purpose |
| --- | --- | --- | --- |
| `ARXIV_CONTACT_EMAIL` | Variable | Always | Monitored address in the arXiv User-Agent; placeholders fail production validation. |
| `OPENAI_SUMMARY_MODEL` | Variable | `openai`; optional for `auto` | Runtime summary model name. |
| `OPENAI_API_KEY` | Secret | `openai`; optional for `auto` | Paid summary-provider credential. |
| `SMTP_HOST` | Variable | Email enabled | SMTP hostname. |
| `SMTP_PORT` | Variable | Email enabled | SMTP port, normally `587` for STARTTLS or `465` for implicit SSL. |
| `SMTP_USE_TLS` | Variable | Email enabled | `true` for STARTTLS, otherwise `false`. |
| `SMTP_USE_SSL` | Variable | Email enabled | `true` for implicit SSL, otherwise `false`; cannot be true with TLS. |
| `SMTP_TIMEOUT_SECONDS` | Variable | Optional email override | Blocking SMTP timeout; defaults to 30 seconds. |
| `SMTP_USERNAME` | Secret | Authenticated SMTP | Username; configure together with the password. |
| `SMTP_PASSWORD` | Secret | Authenticated SMTP | Password or provider-issued app password. |
| `DIGEST_FROM_EMAIL` | Secret | Email enabled | From address; secret classification avoids exposing personal addresses. |
| `DIGEST_TO_EMAIL` | Secret | Email enabled | One address or a comma-separated recipient list. |

Do not add a personal access token. The persistence job uses the built-in
`GITHUB_TOKEN`, scoped to `contents: write` only for that job. All other jobs use
`contents: read`.

## Schedule and manual runs

The workflow runs from the default branch on Monday at `13:00 UTC` using seven complete
days, a limit of ten, `auto` summaries, repository persistence enabled, and email
disabled.

To run manually in GitHub:

1. Open **Actions > Weekly arXiv digest**.
2. Select **Run workflow** and the intended branch.
3. Enter `days` from 1–31 and `limit` from 1–50.
4. Choose `auto`, `offline`, or `openai`.
5. Choose whether to send email and persist canonical state.
6. Select **Run workflow**.

Equivalent GitHub CLI invocation:

```bash
gh workflow run weekly_digest.yml \
  -f days=7 \
  -f limit=10 \
  -f summary_mode=offline \
  -f send_email=false \
  -f persist_reports=true
```

PowerShell users can use one line or replace the displayed POSIX continuations with
PowerShell backticks.

## Workflow behavior and failure boundaries

- **Validate** checks inputs and the required non-placeholder contact, installs Python
  3.12 and locked uv dependencies with caching, then runs every quality gate and
  `doctor --production`.
- **Generate** retrieves from the official arXiv API, ranks, summarizes in the selected
  mode, writes reports, appends idempotent history, and uploads two artifacts.
- **Persist** downloads only dated/latest canonical JSON and `data/history.jsonl`, rejects
  unexpected paths, stages the same allowlist, skips unchanged state, and pushes a bot
  commit without force. A genuine branch race fails visibly instead of overwriting work.
- **Email** is a separate optional job. It downloads the already-uploaded report bundle
  and calls `email-report`. Therefore an SMTP failure fails the email job while the
  complete reports remain downloadable and persistence proceeds independently.

The 30-day report artifact contains dated/latest JSON, Markdown, and HTML plus
`run-summary.json`. A two-day internal state artifact passes canonical JSON and history
between jobs. Artifacts are not durable application state. The committed dated JSON,
`latest.json`, and append-only history are the source for subsequent runs and the hosted
dashboard.

## Streamlit Community Cloud

1. Push the repository and ensure the canonical JSON/history files are present.
2. Sign in at <https://share.streamlit.io/> and choose **Create app**.
3. Select the same repository and branch used for workflow persistence.
4. Set **Main file path** to `streamlit_app.py`.
5. In **Advanced settings**, select Python 3.12.
6. Add no Streamlit secrets; the viewer has no service credentials.
7. Deploy and verify the current digest, dated history, and read-only workflow notice.
8. Choose public or restricted viewer access in the app's Sharing settings.

Community Cloud detects the root `uv.lock`, and GitHub is the app's data source. A
successful persistence commit should update the app automatically. Repository writers
can inspect cloud build/runtime logs; viewer access alone does not grant those logs.

## Local production checks

```bash
uv sync
uv run arxiv-digest doctor
uv run arxiv-digest doctor --production
uv run arxiv-digest run --days 7 --limit 10 --dry-run
uv run streamlit run streamlit_app.py
```

The normal test suite uses mocked SMTP and arXiv fixtures. It neither sends mail nor
uses paid APIs. Live arXiv, OpenAI, SMTP, Streamlit Community Cloud, and GitHub-hosted
workflow execution require external configuration and must be verified in their
respective environments.
