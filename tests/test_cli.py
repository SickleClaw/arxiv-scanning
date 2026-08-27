"""Offline CLI integration tests for Milestones 1 and 2."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from arxiv_digest.arxiv_client import RetrievalResult
from arxiv_digest.cli import SummaryMode, _summary_provider, _window, app
from arxiv_digest.config import load_settings
from arxiv_digest.delivery import DeliveryResult
from arxiv_digest.exceptions import ConfigurationError, DeliveryError
from arxiv_digest.models import (
    CandidateSnapshot,
    DateWindow,
    QueryResult,
    RankedSnapshot,
    SelectionSnapshot,
)
from arxiv_digest.snapshot import create_snapshot
from arxiv_digest.summarization import DeterministicSummaryProvider

runner = CliRunner()


def test_default_window_uses_previous_complete_utc_days_and_overlap() -> None:
    result = _window(
        days=7,
        start=None,
        end=None,
        overlap_days=1,
        now=datetime(2026, 7, 28, 23, 59, tzinfo=UTC),
    )
    assert result.start == datetime(2026, 7, 20, tzinfo=UTC)
    assert result.end == datetime(2026, 7, 28, tzinfo=UTC)


def test_explicit_window_rejects_reversed_dates() -> None:
    with pytest.raises(ConfigurationError, match="--end must be on or after --start"):
        _window(
            days=7,
            start=date(2026, 7, 28),
            end=date(2026, 7, 27),
            overlap_days=1,
            now=datetime(2026, 7, 28, tzinfo=UTC),
        )


def test_doctor_and_show_config_succeed() -> None:
    doctor = runner.invoke(app, ["doctor"])
    assert doctor.exit_code == 0
    assert "Configuration valid" in doctor.stdout
    shown = runner.invoke(app, ["show-config"])
    assert shown.exit_code == 0
    assert '"max_candidate_count": 200' in shown.stdout


def test_production_doctor_rejects_placeholder_contact() -> None:
    result = runner.invoke(app, ["doctor", "--production"])
    assert result.exit_code == 1
    assert "ARXIV_CONTACT_EMAIL" in result.stderr


def test_doctor_returns_nonzero_for_missing_config(tmp_path: Path) -> None:
    result = runner.invoke(app, ["doctor", "--app-config", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 1
    assert "Cannot read configuration" in result.stderr


def test_fetch_writes_snapshot_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    paper_factory,
) -> None:  # type: ignore[no-untyped-def]
    paper = paper_factory()

    class FakeClient:
        def __init__(self, _config: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def fetch(
            self, _queries: object, _window: object, _maximum: int, _domain: object
        ) -> RetrievalResult:
            return RetrievalResult(
                papers=[paper, paper],
                query_results=[QueryResult(name="fixture", records_received=2)],
            )

    monkeypatch.setattr("arxiv_digest.cli.ArxivClient", FakeClient)
    output = tmp_path / "snapshot.json"
    result = runner.invoke(
        app,
        [
            "fetch",
            "--start",
            "2026-07-20",
            "--end",
            "2026-07-27",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Retrieved 2 records, deduplicated to 1" in result.stdout
    snapshot = CandidateSnapshot.model_validate_json(output.read_text(encoding="utf-8"))
    assert snapshot.retrieval_window.start == datetime(2026, 7, 20, tzinfo=UTC)
    assert snapshot.retrieval_window.end == datetime(2026, 7, 28, tzinfo=UTC)
    assert len(snapshot.papers) == 1


def test_fetch_rejects_incomplete_explicit_window() -> None:
    result = runner.invoke(app, ["fetch", "--start", "2026-07-20"])
    assert result.exit_code == 1
    assert "--start and --end must be provided together" in result.stderr


def _write_candidate_snapshot(path: Path, paper_factory) -> CandidateSnapshot:  # type: ignore[no-untyped-def]
    window = DateWindow(
        start=datetime(2026, 7, 20, tzinfo=UTC),
        end=datetime(2026, 7, 28, tzinfo=UTC),
    )
    titles = [
        "Spin ice magnetic monopole dynamics",
        "Neutron diffuse scattering structure factor",
        "Linear spin-wave theory exchange fitting",
        "Frustrated spinel magnetic disorder",
    ]
    papers = [
        paper_factory(arxiv_id=f"2607.40{index:03d}", title=title)
        for index, title in enumerate(titles, start=1)
    ]
    snapshot = create_snapshot(
        window=window,
        profile_version="1.0",
        generated_at=datetime(2026, 7, 28, 12, tzinfo=UTC),
        query_results=[],
        raw_papers=papers,
        deduplicated_papers=papers,
    )
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    return snapshot


def test_rank_uses_existing_snapshot_without_network(tmp_path: Path, paper_factory) -> None:  # type: ignore[no-untyped-def]
    snapshot_path = tmp_path / "candidates.json"
    snapshot = _write_candidate_snapshot(snapshot_path, paper_factory)
    output = tmp_path / "ranked.json"
    result = runner.invoke(
        app,
        ["rank", "--snapshot", str(snapshot_path), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    ranked = RankedSnapshot.model_validate_json(output.read_text(encoding="utf-8"))
    assert ranked.source_run_id == snapshot.run_id
    assert len(ranked.ranked_papers) == 4
    assert "ranked 4" in result.stdout


def test_rank_missing_snapshot_returns_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["rank", "--snapshot", str(tmp_path / "missing.json")],
    )
    assert result.exit_code == 1
    assert "Run fetch first or pass --fetch-missing" in result.stderr


def test_run_dry_run_writes_artifacts_but_not_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    paper_factory,
) -> None:  # type: ignore[no-untyped-def]
    snapshot_path = tmp_path / "candidates-2026-07-20-2026-07-27.json"
    _write_candidate_snapshot(snapshot_path, paper_factory)
    ranked_path = tmp_path / "ranked.json"
    selection_path = tmp_path / "selection.json"
    history_path = tmp_path / "history.jsonl"
    reports_path = tmp_path / "reports"
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__HISTORY_FILE", str(history_path))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__REPORTS_DIR", str(reports_path))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__TEMPLATES_DIR", str(Path.cwd() / "templates"))
    result = runner.invoke(
        app,
        [
            "run",
            "--start",
            "2026-07-20",
            "--end",
            "2026-07-27",
            "--limit",
            "4",
            "--dry-run",
            "--ranked-output",
            str(ranked_path),
            "--selection-output",
            str(selection_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(RankedSnapshot.model_validate_json(ranked_path.read_text()).ranked_papers) == 4
    selection = SelectionSnapshot.model_validate_json(selection_path.read_text())
    assert 1 <= len(selection.selected) <= 4
    assert "dry-run history unchanged" in result.stdout
    assert "summarized and selected" in result.stdout
    assert (reports_path / "latest.md").exists()
    assert (reports_path / "latest.html").exists()
    assert (reports_path / "latest.json").exists()
    assert not history_path.exists()


def test_dry_run_forces_offline_provider_when_openai_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    paper_factory,
) -> None:  # type: ignore[no-untyped-def]
    snapshot_path = tmp_path / "candidates.json"
    _write_candidate_snapshot(snapshot_path, paper_factory)
    monkeypatch.setenv("ARXIV_DIGEST_APP__SUMMARIZATION__PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_SUMMARY_MODEL", raising=False)
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__TEMPLATES_DIR", str(Path.cwd() / "templates"))
    result = runner.invoke(
        app,
        ["run", "--snapshot", str(snapshot_path), "--limit", "2", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "0 summary fallbacks" in result.stdout


def test_summary_mode_auto_falls_back_offline_without_paid_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_SUMMARY_MODEL", raising=False)
    provider = _summary_provider(load_settings(), dry_run=False, summary_mode=SummaryMode.AUTO)
    assert isinstance(provider, DeterministicSummaryProvider)
    with pytest.raises(ConfigurationError, match="OPENAI_SUMMARY_MODEL"):
        _summary_provider(load_settings(), dry_run=False, summary_mode=SummaryMode.OPENAI)


def test_run_send_email_uses_completed_reports_and_explicit_offline_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    paper_factory,
) -> None:  # type: ignore[no-untyped-def]
    snapshot_path = tmp_path / "candidates.json"
    _write_candidate_snapshot(snapshot_path, paper_factory)
    reports_path = tmp_path / "reports"
    history_path = tmp_path / "history.jsonl"
    received: list[tuple[Path, Path]] = []

    class FakeDelivery:
        def __init__(self, _config: object) -> None:
            pass

        def send(self, markdown: Path, html: Path, *, report_date: date) -> DeliveryResult:
            assert report_date.isoformat()
            received.append((markdown, html))
            return DeliveryResult(recipient_count=2)

    monkeypatch.setattr("arxiv_digest.cli.SMTPDeliveryProvider", FakeDelivery)
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__REPORTS_DIR", str(reports_path))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__HISTORY_FILE", str(history_path))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__TEMPLATES_DIR", str(Path.cwd() / "templates"))
    result = runner.invoke(
        app,
        [
            "run",
            "--snapshot",
            str(snapshot_path),
            "--limit",
            "2",
            "--summary-mode",
            "offline",
            "--send-email",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "emailed 2 recipient(s)" in result.stdout
    assert received == [
        (
            next(reports_path.glob("*-weekly-arxiv-digest.md")),
            next(reports_path.glob("*-weekly-arxiv-digest.html")),
        )
    ]
    assert history_path.exists()


def test_failed_email_returns_nonzero_after_preserving_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    paper_factory,
) -> None:  # type: ignore[no-untyped-def]
    snapshot_path = tmp_path / "candidates.json"
    _write_candidate_snapshot(snapshot_path, paper_factory)
    reports_path = tmp_path / "reports"
    history_path = tmp_path / "history.jsonl"

    class FailingDelivery:
        def __init__(self, _config: object) -> None:
            pass

        def send(self, _markdown: Path, _html: Path, *, report_date: date) -> DeliveryResult:
            assert report_date.isoformat()
            raise DeliveryError("SMTP delivery failed; completed reports remain available.")

    monkeypatch.setattr("arxiv_digest.cli.SMTPDeliveryProvider", FailingDelivery)
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__REPORTS_DIR", str(reports_path))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__HISTORY_FILE", str(history_path))
    monkeypatch.setenv("ARXIV_DIGEST_APP__PATHS__TEMPLATES_DIR", str(Path.cwd() / "templates"))
    result = runner.invoke(
        app,
        [
            "run",
            "--snapshot",
            str(snapshot_path),
            "--limit",
            "2",
            "--summary-mode",
            "offline",
            "--send-email",
        ],
    )
    assert result.exit_code == 1
    assert "completed reports remain available" in result.stderr
    assert (reports_path / "latest.json").exists()
    assert (reports_path / "latest.md").exists()
    assert (reports_path / "latest.html").exists()
    assert history_path.exists()


def test_dry_run_rejects_email_before_pipeline() -> None:
    result = runner.invoke(app, ["run", "--dry-run", "--send-email"])
    assert result.exit_code == 1
    assert "cannot be combined" in result.stderr
