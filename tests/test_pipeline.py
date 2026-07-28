"""Milestone 3 pipeline integration and idempotency tests."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from arxiv_digest.config import RecommendationMix, SelectionConfig, load_settings
from arxiv_digest.exceptions import ReportingError
from arxiv_digest.history import load_history
from arxiv_digest.models import CandidateSnapshot, DateWindow
from arxiv_digest.pipeline import load_model, run_milestone3, run_milestone4, write_model
from arxiv_digest.snapshot import create_snapshot
from arxiv_digest.summarization import DeterministicSummaryProvider

NOW = datetime(2026, 7, 28, 12, tzinfo=UTC)
WINDOW = DateWindow(
    start=datetime(2026, 7, 20, tzinfo=UTC),
    end=datetime(2026, 7, 28, tzinfo=UTC),
)


def candidate_snapshot(paper_factory) -> CandidateSnapshot:  # type: ignore[no-untyped-def]
    titles = [
        "Spin ice magnetic monopole dynamics",
        "Neutron diffuse scattering structure factor",
        "Linear spin-wave theory exchange fitting",
        "Frustrated spinel magnetic disorder",
        "Spin Seebeck effect in a magnetic insulator",
        "Kinetic Monte Carlo magnetism",
    ]
    papers = [
        paper_factory(
            arxiv_id=f"2607.30{index:03d}",
            title=title,
            published_at=WINDOW.start + timedelta(days=index),
            updated_at=WINDOW.start + timedelta(days=index),
        )
        for index, title in enumerate(titles, start=1)
    ]
    return create_snapshot(
        window=WINDOW,
        profile_version="1.0",
        generated_at=NOW,
        query_results=[],
        raw_papers=papers,
        deduplicated_papers=papers,
    )


def test_pipeline_persists_history_idempotently_and_artifacts_round_trip(
    tmp_path: Path, paper_factory
) -> None:  # type: ignore[no-untyped-def]
    settings = load_settings()
    paths = settings.app.paths.model_copy(update={"history_file": tmp_path / "history.jsonl"})
    app_config = settings.app.model_copy(update={"paths": paths})
    profile = settings.profile.model_copy(
        update={
            "recommendation_mix": RecommendationMix(direct=3, adjacent=2, wildcard=1),
            "selection": SelectionConfig(
                mmr_lambda=0.75,
                direct_min_score=0.20,
                adjacent_min_score=0.10,
                wildcard_min_score=0.05,
            ),
        }
    )
    settings = settings.model_copy(update={"app": app_config, "profile": profile})
    snapshot = candidate_snapshot(paper_factory)

    first = run_milestone3(
        snapshot,
        settings,
        [],
        now=NOW,
        limit=6,
        persist_history=True,
    )
    second = run_milestone3(
        snapshot,
        settings,
        load_history(paths.history_file),
        now=NOW,
        limit=6,
        persist_history=True,
    )
    assert len(first.selection.selected) == 6
    assert first.history_appended == 6
    assert second.history_appended == 0
    assert second.selection == first.selection
    assert len(load_history(paths.history_file)) == 6

    path = write_model(first.ranked, tmp_path / "ranked.json")
    assert load_model(path, type(first.ranked)) == first.ranked


def test_dry_run_does_not_create_history(tmp_path: Path, paper_factory) -> None:  # type: ignore[no-untyped-def]
    settings = load_settings()
    history_path = tmp_path / "history.jsonl"
    paths = settings.app.paths.model_copy(update={"history_file": history_path})
    settings = settings.model_copy(update={"app": settings.app.model_copy(update={"paths": paths})})
    result = run_milestone3(
        candidate_snapshot(paper_factory),
        settings,
        [],
        now=NOW,
        limit=3,
        persist_history=False,
    )
    assert result.history_appended == 0
    assert not history_path.exists()


def _milestone4_settings(tmp_path: Path):  # type: ignore[no-untyped-def]
    settings = load_settings()
    paths = settings.app.paths.model_copy(
        update={
            "reports_dir": tmp_path / "reports",
            "history_file": tmp_path / "history.jsonl",
            "templates_dir": Path.cwd() / "templates",
        }
    )
    app_config = settings.app.model_copy(update={"paths": paths})
    profile = settings.profile.model_copy(
        update={
            "selection": SelectionConfig(
                mmr_lambda=0.75,
                direct_min_score=0.10,
                adjacent_min_score=0.05,
                wildcard_min_score=0.01,
            )
        }
    )
    return settings.model_copy(update={"app": app_config, "profile": profile})


def test_milestone4_writes_reports_then_history_idempotently(tmp_path: Path, paper_factory) -> None:  # type: ignore[no-untyped-def]
    settings = _milestone4_settings(tmp_path)
    snapshot = candidate_snapshot(paper_factory)
    provider = DeterministicSummaryProvider()
    first = run_milestone4(
        snapshot,
        settings,
        [],
        provider,
        now=NOW,
        limit=4,
        persist_history=True,
    )
    second = run_milestone4(
        snapshot,
        settings,
        load_history(settings.app.paths.history_file),
        provider,
        now=NOW,
        limit=4,
        persist_history=True,
    )
    assert len(first.recommendations) == 4
    assert first.reports.markdown.exists()
    assert first.reports.html.exists()
    assert first.history_appended == 4
    assert second.history_appended == 0
    assert all(
        record.report_path == str(first.reports.markdown)
        for record in load_history(settings.app.paths.history_file)
    )


def test_report_failure_does_not_write_history(tmp_path: Path, paper_factory) -> None:  # type: ignore[no-untyped-def]
    settings = _milestone4_settings(tmp_path)
    missing_paths = settings.app.paths.model_copy(
        update={"templates_dir": tmp_path / "missing-templates"}
    )
    settings = settings.model_copy(
        update={"app": settings.app.model_copy(update={"paths": missing_paths})}
    )
    with pytest.raises(ReportingError, match="Cannot render"):
        run_milestone4(
            candidate_snapshot(paper_factory),
            settings,
            [],
            DeterministicSummaryProvider(),
            now=NOW,
            limit=3,
            persist_history=True,
        )
    assert not settings.app.paths.history_file.exists()
