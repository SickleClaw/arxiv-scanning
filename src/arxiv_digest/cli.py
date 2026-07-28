"""Typer command-line interface through deterministic offline Milestone 3."""

from __future__ import annotations

import logging
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from arxiv_digest.arxiv_client import ArxivClient
from arxiv_digest.config import Settings, load_settings, settings_as_json
from arxiv_digest.exceptions import ArxivDigestError, ConfigurationError
from arxiv_digest.history import load_history
from arxiv_digest.models import CandidateSnapshot, DateWindow
from arxiv_digest.pipeline import (
    default_ranked_path,
    default_selection_path,
    load_model,
    rank_snapshot,
    retrieve_candidates,
    run_milestone3,
    write_model,
)
from arxiv_digest.snapshot import default_snapshot_path, write_snapshot

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


def _load(app_config: Path, profile_config: Path) -> Settings:
    return load_settings(app_config, profile_config)


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
    app_config: AppConfigOption = Path("config/app.yaml"),
    profile_config: ProfileConfigOption = Path("config/research_profile.yaml"),
) -> None:
    """Validate configuration, writable paths, and arXiv connection settings."""
    try:
        settings = _load(app_config, profile_config)
        _writable(settings.app.paths.data_dir)
        _writable(settings.app.paths.reports_dir)
        _writable(settings.app.paths.history_file.parent)
    except ArxivDigestError as exc:
        _fail(exc)
    typer.echo(
        "Configuration valid; data, report, and history paths are writable; "
        "arXiv timeouts, pacing, retries, and contact User-Agent are configured."
    )


@app.command("show-config")
def show_config(
    app_config: AppConfigOption = Path("config/app.yaml"),
    profile_config: ProfileConfigOption = Path("config/research_profile.yaml"),
) -> None:
    """Print the effective validated configuration with secret-like values redacted."""
    try:
        settings = _load(app_config, profile_config)
    except ArxivDigestError as exc:
        _fail(exc)
    typer.echo(settings_as_json(settings))


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
) -> None:
    """Fetch, normalize, deduplicate, and write an arXiv candidate snapshot."""
    try:
        settings = _load(app_config, profile_config)
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
) -> None:
    """Rank an existing snapshot, retrieving only when explicitly requested."""
    try:
        settings = _load(app_config, profile_config)
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
    snapshot_path: Annotated[
        Path | None, typer.Option("--snapshot", help="Existing candidate JSON snapshot.")
    ] = None,
    ranked_output: Annotated[Path | None, typer.Option(help="Ranked JSON output path.")] = None,
    selection_output: Annotated[
        Path | None, typer.Option(help="Selection JSON output path.")
    ] = None,
    app_config: AppConfigOption = Path("config/app.yaml"),
    profile_config: ProfileConfigOption = Path("config/research_profile.yaml"),
) -> None:
    """Retrieve or load candidates, then rank and select without paid services."""
    try:
        settings = _load(app_config, profile_config)
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
        result = run_milestone3(
            snapshot,
            settings,
            history,
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
    except (ArxivDigestError, OSError, ValueError) as exc:
        _fail(exc)
    history_status = (
        "dry-run history unchanged"
        if dry_run
        else f"appended {result.history_appended} history records"
    )
    typer.echo(
        f"Loaded {len(snapshot.papers)} candidates from {source_path}, excluded "
        f"{result.history_excluded} by history, ranked {len(result.ranked.ranked_papers)}, "
        f"selected {len(result.selection.selected)}, {history_status}; wrote "
        f"{ranked_destination} and {selection_destination}."
    )


if __name__ == "__main__":
    app()
