"""Typer command-line interface through abstract-grounded Milestone 4 reports."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn
from zoneinfo import ZoneInfo

import typer

from arxiv_digest.arxiv_client import ArxivClient
from arxiv_digest.config import Settings, load_settings, settings_as_json
from arxiv_digest.dashboard.launcher import launch_dashboard
from arxiv_digest.dashboard.paths import find_repository_root
from arxiv_digest.delivery import SMTPDeliveryProvider, validate_delivery_ready
from arxiv_digest.exceptions import ArxivDigestError, ConfigurationError
from arxiv_digest.history import load_history
from arxiv_digest.models import CandidateSnapshot, DateWindow, DigestArtifact
from arxiv_digest.pipeline import (
    default_ranked_path,
    default_selection_path,
    load_model,
    rank_snapshot,
    retrieve_candidates,
    run_milestone4,
    write_model,
)
from arxiv_digest.snapshot import default_snapshot_path, write_snapshot
from arxiv_digest.summarization import DeterministicSummaryProvider, OpenAISummaryProvider

app = typer.Typer(
    name="arxiv-digest",
    help="Retrieve, rank, and diversify arXiv candidates for a weekly digest.",
    no_args_is_help=True,
)

AppConfigOption = Annotated[
    Path, typer.Option("--app-config", help="Path to operational YAML configuration.")
]
ProfileConfigOption = Annotated[
    Path, typer.Option("--profile-config", help="Path to research profile YAML.")
]
GroupConfigOption = Annotated[
    Path, typer.Option("--group-config", help="Path to the lab group profile YAML.")
]


class SummaryMode(StrEnum):
    """CLI-selectable summary behavior for unattended and local runs."""

    AUTO = "auto"
    OFFLINE = "offline"
    OPENAI = "openai"


def _load(app_config: Path, profile_config: Path, group_config: Path) -> Settings:
    return load_settings(app_config, profile_config, group_config)


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.app.logging.level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _fail(exc: Exception) -> NoReturn:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=1)


def _writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".doctor-", delete=True):
            return
    except OSError as exc:
        raise ConfigurationError(f"Directory is not writable: {directory} ({exc})") from exc


def _summary_provider(
    settings: Settings,
    *,
    dry_run: bool,
    summary_mode: SummaryMode | None = None,
) -> DeterministicSummaryProvider | OpenAISummaryProvider:
    """Resolve the configured provider while guaranteeing that dry runs stay offline."""
    if dry_run:
        return DeterministicSummaryProvider()
    effective_mode = summary_mode or SummaryMode(settings.app.summarization.provider)
    if effective_mode == SummaryMode.OFFLINE:
        return DeterministicSummaryProvider()
    model = settings.app.summarization.openai_summary_model
    api_key_available = bool(os.environ.get("OPENAI_API_KEY"))
    if effective_mode == SummaryMode.AUTO and (model is None or not api_key_available):
        return DeterministicSummaryProvider()
    if model is None:
        raise ConfigurationError("OPENAI_SUMMARY_MODEL is required for OpenAI summaries")
    if not api_key_available:
        raise ConfigurationError("OPENAI_API_KEY is required for OpenAI summaries")
    return OpenAISummaryProvider(
        model=model,
        validation_retries=settings.app.summarization.validation_retries,
    )


def _window(
    *,
    days: int,
    start: date | None,
    end: date | None,
    overlap_days: int,
    now: datetime,
) -> DateWindow:
    if (start is None) != (end is None):
        raise ConfigurationError("--start and --end must be provided together")
    if start is not None and end is not None:
        if end < start:
            raise ConfigurationError("--end must be on or after --start")
        return DateWindow(
            start=datetime.combine(start, time.min, tzinfo=UTC),
            end=datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC),
        )
    if days < 1:
        raise ConfigurationError("--days must be at least 1")
    complete_day_end = datetime.combine(now.astimezone(UTC).date(), time.min, tzinfo=UTC)
    return DateWindow(
        start=complete_day_end - timedelta(days=days + overlap_days),
        end=complete_day_end,
    )


def _optional_date(value: str | None, option: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigurationError(f"{option} must use YYYY-MM-DD format") from exc


def _requested_window(
    settings: Settings,
    *,
    days: int,
    start: str | None,
    end: str | None,
    now: datetime,
) -> DateWindow:
    return _window(
        days=days,
        start=_optional_date(start, "--start"),
        end=_optional_date(end, "--end"),
        overlap_days=settings.app.arxiv.overlap_days,
        now=now,
    )


def _retrieve(settings: Settings, window: DateWindow, now: datetime) -> CandidateSnapshot:
    with ArxivClient(settings.app.arxiv) as client:
        return retrieve_candidates(settings, window, now=now, client=client)


def _candidate_for_command(
    settings: Settings,
    *,
    window: DateWindow,
    snapshot_path: Path | None,
    fetch_missing: bool,
    now: datetime,
) -> tuple[CandidateSnapshot, Path]:
    path = snapshot_path or default_snapshot_path(settings.app.paths.data_dir, window)
    if path.exists():
        return load_model(path, CandidateSnapshot), path
    if not fetch_missing:
        raise ConfigurationError(
            f"Candidate snapshot does not exist: {path}. Run fetch first or pass --fetch-missing."
        )
    snapshot = _retrieve(settings, window, now)
    write_snapshot(snapshot, path)
    return snapshot, path


@app.command("doctor")
def doctor(
    production: Annotated[
        bool,
        typer.Option(
            "--production",
            help="Require production contact and cloud dashboard prerequisites.",
        ),
    ] = False,
    app_config: AppConfigOption = Path("config/app.yaml"),
    profile_config: ProfileConfigOption = Path("config/research_profile.yaml"),
    group_config: GroupConfigOption = Path("profiles/group.yaml"),
) -> None:
    """Validate configuration, writable paths, and arXiv connection settings."""
    try:
        settings = _load(app_config, profile_config, group_config)
        _writable(settings.app.paths.data_dir)
        _writable(settings.app.paths.reports_dir)
        _writable(settings.app.paths.history_file.parent)
        for template_name in ("weekly_report.md.j2", "weekly_report.html.j2"):
            template_path = settings.app.paths.templates_dir / template_name
            if not template_path.is_file():
                raise ConfigurationError(f"Report template does not exist: {template_path}")
        repository_root = find_repository_root(Path(__file__))
        if not (repository_root / "streamlit_app.py").is_file():
            raise ConfigurationError("Cloud dashboard entrypoint streamlit_app.py is missing")
        if production and not (repository_root / "reports" / "latest.json").is_file():
            raise ConfigurationError("Production dashboard report reports/latest.json is missing")
        placeholder_contact = settings.app.arxiv.contact_email.lower().endswith(
            ("@example.org", "@example.com", ".invalid")
        )
        if production and placeholder_contact:
            raise ConfigurationError(
                "ARXIV_CONTACT_EMAIL must be a monitored non-placeholder address in production"
            )
        provider_status = settings.app.summarization.provider
        if provider_status == "openai":
            _summary_provider(settings, dry_run=False)
        smtp_status = "disabled"
        if settings.app.delivery.enabled:
            validate_delivery_ready(settings.app.delivery)
            smtp_status = "ready"
    except ArxivDigestError as exc:
        _fail(exc)
    contact_note = (
        " (placeholder arXiv contact; replace before production)" if placeholder_contact else ""
    )
    typer.echo(
        "Configuration valid; data, report, and history paths are writable; "
        "arXiv timeouts, pacing, retries, contact User-Agent, report templates, and "
        f"{provider_status} summaries, cloud dashboard entrypoint, and SMTP {smtp_status} are "
        f"configured{contact_note}."
    )


@app.command("show-config")
def show_config(
    app_config: AppConfigOption = Path("config/app.yaml"),
    profile_config: ProfileConfigOption = Path("config/research_profile.yaml"),
    group_config: GroupConfigOption = Path("profiles/group.yaml"),
) -> None:
    """Print the effective validated configuration with secret-like values redacted."""
    try:
        settings = _load(app_config, profile_config, group_config)
    except ArxivDigestError as exc:
        _fail(exc)
    typer.echo(settings_as_json(settings))


@app.command("dashboard")
def dashboard_command(
    report: Annotated[
        Path | None,
        typer.Option(help="Canonical digest JSON to open (defaults to REPORTS_DIR/latest.json)."),
    ] = None,
    reports_dir: Annotated[
        Path,
        typer.Option(help="Directory scanned for dated digest history."),
    ] = Path("reports"),
) -> None:
    """Open the read-only local dashboard without running the pipeline."""
    resolved_report = report or reports_dir / "latest.json"
    try:
        exit_code = launch_dashboard(resolved_report, reports_dir)
    except ArxivDigestError as exc:
        _fail(exc)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("email-report")
def email_report(
    markdown: Annotated[
        Path | None, typer.Option(help="Rendered Markdown report (defaults to latest.md).")
    ] = None,
    html: Annotated[
        Path | None, typer.Option(help="Rendered HTML report (defaults to latest.html).")
    ] = None,
    digest_json: Annotated[
        Path | None, typer.Option(help="Canonical JSON used to determine the report date.")
    ] = None,
    app_config: AppConfigOption = Path("config/app.yaml"),
    profile_config: ProfileConfigOption = Path("config/research_profile.yaml"),
    group_config: GroupConfigOption = Path("profiles/group.yaml"),
) -> None:
    """Explicitly email an already-generated report without rerunning the pipeline."""
    try:
        settings = _load(app_config, profile_config, group_config)
        reports_dir = settings.app.paths.reports_dir
        markdown_path = markdown or reports_dir / "latest.md"
        html_path = html or reports_dir / "latest.html"
        digest_path = digest_json or reports_dir / "latest.json"
        digest = load_model(digest_path, DigestArtifact)
        provider = SMTPDeliveryProvider(settings.app.delivery)
        report_date = digest.generated_at.astimezone(ZoneInfo(digest.timezone)).date()
        result = provider.send(markdown_path, html_path, report_date=report_date)
    except (ArxivDigestError, OSError, ValueError) as exc:
        _fail(exc)
    typer.echo(f"Email sent to {result.recipient_count} configured recipient(s).")


@app.command("fetch")
def fetch(
    days: Annotated[int, typer.Option(min=1, help="Complete UTC days to retrieve.")] = 7,
    start: Annotated[
        str | None, typer.Option(help="First UTC date, YYYY-MM-DD (inclusive).")
    ] = None,
    end: Annotated[str | None, typer.Option(help="Last UTC date, YYYY-MM-DD (inclusive).")] = None,
    output: Annotated[Path | None, typer.Option(help="Candidate JSON output path.")] = None,
    app_config: AppConfigOption = Path("config/app.yaml"),
    profile_config: ProfileConfigOption = Path("config/research_profile.yaml"),
    group_config: GroupConfigOption = Path("profiles/group.yaml"),
) -> None:
    """Fetch, normalize, deduplicate, and write an arXiv candidate snapshot."""
    try:
        settings = _load(app_config, profile_config, group_config)
        _configure_logging(settings)
        now = datetime.now(UTC)
        window = _requested_window(
            settings,
            days=days,
            start=start,
            end=end,
            now=now,
        )
        snapshot = _retrieve(settings, window, now)
        destination = output or default_snapshot_path(settings.app.paths.data_dir, window)
        write_snapshot(snapshot, destination)
    except (ArxivDigestError, OSError, ValueError) as exc:
        _fail(exc)
    typer.echo(
        f"Retrieved {snapshot.records_retrieved} records, deduplicated to "
        f"{snapshot.records_after_deduplication}, wrote {destination}."
    )


@app.command("rank")
def rank(
    days: Annotated[int, typer.Option(min=1, help="Complete UTC days in the snapshot.")] = 7,
    start: Annotated[
        str | None, typer.Option(help="First UTC date, YYYY-MM-DD (inclusive).")
    ] = None,
    end: Annotated[str | None, typer.Option(help="Last UTC date, YYYY-MM-DD (inclusive).")] = None,
    snapshot_path: Annotated[
        Path | None, typer.Option("--snapshot", help="Existing candidate JSON snapshot.")
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Ranked JSON output path.")] = None,
    fetch_missing: Annotated[
        bool,
        typer.Option(help="Retrieve candidates only when the selected snapshot is missing."),
    ] = False,
    app_config: AppConfigOption = Path("config/app.yaml"),
    profile_config: ProfileConfigOption = Path("config/research_profile.yaml"),
    group_config: GroupConfigOption = Path("profiles/group.yaml"),
) -> None:
    """Rank an existing snapshot, retrieving only when explicitly requested."""
    try:
        settings = _load(app_config, profile_config, group_config)
        _configure_logging(settings)
        now = datetime.now(UTC)
        window = _requested_window(settings, days=days, start=start, end=end, now=now)
        snapshot, source_path = _candidate_for_command(
            settings,
            window=window,
            snapshot_path=snapshot_path,
            fetch_missing=fetch_missing,
            now=now,
        )
        history = load_history(settings.app.paths.history_file)
        ranked, excluded = rank_snapshot(snapshot, settings, history, now=now)
        destination = output or default_ranked_path(
            settings.app.paths.data_dir, snapshot.retrieval_window
        )
        write_model(ranked, destination)
    except (ArxivDigestError, OSError, ValueError) as exc:
        _fail(exc)
    typer.echo(
        f"Loaded {len(snapshot.papers)} candidates from {source_path}, excluded {excluded} "
        f"by history, ranked {len(ranked.ranked_papers)}, wrote {destination}."
    )


@app.command("run")
def run(
    days: Annotated[int, typer.Option(min=1, help="Complete UTC days to process.")] = 7,
    start: Annotated[
        str | None, typer.Option(help="First UTC date, YYYY-MM-DD (inclusive).")
    ] = None,
    end: Annotated[str | None, typer.Option(help="Last UTC date, YYYY-MM-DD (inclusive).")] = None,
    limit: Annotated[int, typer.Option(min=1, help="Maximum papers to select.")] = 10,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Use offline ranking and do not write history."),
    ] = False,
    summary_mode: Annotated[
        SummaryMode | None,
        typer.Option(
            help="Summary provider mode: auto uses OpenAI only when key and model are available."
        ),
    ] = None,
    send_email: Annotated[
        bool,
        typer.Option("--send-email", help="Email completed reports after successful generation."),
    ] = False,
    snapshot_path: Annotated[
        Path | None, typer.Option("--snapshot", help="Existing candidate JSON snapshot.")
    ] = None,
    ranked_output: Annotated[Path | None, typer.Option(help="Ranked JSON output path.")] = None,
    selection_output: Annotated[
        Path | None, typer.Option(help="Selection JSON output path.")
    ] = None,
    app_config: AppConfigOption = Path("config/app.yaml"),
    profile_config: ProfileConfigOption = Path("config/research_profile.yaml"),
    group_config: GroupConfigOption = Path("profiles/group.yaml"),
) -> None:
    """Retrieve or load candidates, then rank, summarize, and write weekly reports."""
    try:
        if dry_run and send_email:
            raise ConfigurationError("--send-email cannot be combined with --dry-run")
        settings = _load(app_config, profile_config, group_config)
        _configure_logging(settings)
        now = datetime.now(UTC)
        window = _requested_window(settings, days=days, start=start, end=end, now=now)
        snapshot, source_path = _candidate_for_command(
            settings,
            window=window,
            snapshot_path=snapshot_path,
            fetch_missing=True,
            now=now,
        )
        history = load_history(settings.app.paths.history_file)
        provider = _summary_provider(settings, dry_run=dry_run, summary_mode=summary_mode)
        result = run_milestone4(
            snapshot,
            settings,
            history,
            provider,
            now=now,
            limit=limit,
            persist_history=not dry_run,
        )
        ranked_destination = ranked_output or default_ranked_path(
            settings.app.paths.data_dir, snapshot.retrieval_window
        )
        selection_destination = selection_output or default_selection_path(
            settings.app.paths.data_dir, snapshot.retrieval_window
        )
        write_model(result.ranked, ranked_destination)
        write_model(result.selection, selection_destination)
        email_recipient_count = 0
        if send_email:
            email_provider = SMTPDeliveryProvider(settings.app.delivery)
            report_date = result.digest.generated_at.astimezone(
                ZoneInfo(result.digest.timezone)
            ).date()
            email_result = email_provider.send(
                result.reports.markdown,
                result.reports.html,
                report_date=report_date,
            )
            email_recipient_count = email_result.recipient_count
    except (ArxivDigestError, OSError, ValueError) as exc:
        _fail(exc)
    history_status = (
        "dry-run history unchanged"
        if dry_run
        else f"appended {result.history_appended} history records"
    )
    email_status = (
        f"emailed {email_recipient_count} recipient(s)" if send_email else "email not requested"
    )
    typer.echo(
        f"Loaded {len(snapshot.papers)} candidates from {source_path}, excluded "
        f"{result.history_excluded} by history, ranked {len(result.ranked.ranked_papers)}, "
        f"summarized and selected {len(result.recommendations)}, used "
        f"{result.summary_fallbacks} summary fallbacks, {history_status}; wrote "
        f"{ranked_destination}, {selection_destination}, {result.reports.markdown}, and "
        f"{result.reports.html}, and {result.reports.json} "
        f"(plus latest aliases and run-summary.json); {email_status}."
    )


if __name__ == "__main__":
    app()
