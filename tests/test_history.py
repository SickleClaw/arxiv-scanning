"""Tests for history exclusion, resurfacing, and append-only idempotency."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from arxiv_digest.exceptions import HistoryError
from arxiv_digest.history import append_history, filter_recent_history, load_history
from arxiv_digest.models import (
    DateWindow,
    HistoryRecord,
    RecommendationType,
)

NOW = datetime(2026, 7, 28, 12, tzinfo=UTC)
WINDOW = DateWindow(
    start=datetime(2026, 7, 20, tzinfo=UTC),
    end=datetime(2026, 7, 28, tzinfo=UTC),
)


def history_record(**overrides: object) -> HistoryRecord:
    values: dict[str, object] = {
        "run_id": "select-run",
        "run_timestamp": NOW - timedelta(days=3),
        "retrieval_window": WINDOW,
        "arxiv_id": "2607.12345",
        "version": 1,
        "paper_updated_at": NOW - timedelta(days=4),
        "rank": 1,
        "recommendation_type": RecommendationType.DIRECT,
        "final_score": 0.8,
        "report_path": None,
    }
    values.update(overrides)
    return HistoryRecord.model_validate(values)


def test_recent_same_version_is_excluded(paper_factory) -> None:  # type: ignore[no-untyped-def]
    paper = paper_factory()
    included, excluded = filter_recent_history(
        [paper],
        [history_record()],
        now=NOW,
        exclusion_days=90,
        allow_updated_resurfacing=True,
    )
    assert included == []
    assert excluded == 1


def test_updated_version_resurfaces_only_when_all_conditions_hold(paper_factory) -> None:  # type: ignore[no-untyped-def]
    updated = paper_factory(version=2, updated_at=NOW - timedelta(days=1))
    included, excluded = filter_recent_history(
        [updated],
        [history_record()],
        now=NOW,
        exclusion_days=90,
        allow_updated_resurfacing=True,
    )
    assert included == [updated]
    assert excluded == 0

    stale_update = paper_factory(
        version=2,
        published_at=NOW - timedelta(days=6),
        updated_at=NOW - timedelta(days=5),
    )
    included, excluded = filter_recent_history(
        [stale_update],
        [history_record()],
        now=NOW,
        exclusion_days=90,
        allow_updated_resurfacing=True,
    )
    assert included == []
    assert excluded == 1

    included, excluded = filter_recent_history(
        [updated],
        [history_record()],
        now=NOW,
        exclusion_days=90,
        allow_updated_resurfacing=False,
    )
    assert included == []
    assert excluded == 1


def test_old_history_does_not_exclude(paper_factory) -> None:  # type: ignore[no-untyped-def]
    old = history_record(run_timestamp=NOW - timedelta(days=91))
    paper = paper_factory()
    included, excluded = filter_recent_history(
        [paper],
        [old],
        now=NOW,
        exclusion_days=90,
        allow_updated_resurfacing=True,
    )
    assert included == [paper]
    assert excluded == 0


def test_append_history_is_idempotent_and_rejects_window_collision(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    records = [history_record(), history_record(arxiv_id="2607.99999", rank=2)]
    assert append_history(path, records) == 2
    assert append_history(path, records) == 0
    assert load_history(path) == records

    other_window = DateWindow(
        start=datetime(2026, 7, 1, tzinfo=UTC),
        end=datetime(2026, 7, 8, tzinfo=UTC),
    )
    with pytest.raises(HistoryError, match="different retrieval window"):
        append_history(path, [history_record(retrieval_window=other_window)])


def test_invalid_history_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text("{}\nnot-json\n", encoding="utf-8")
    with pytest.raises(HistoryError, match=r"history\.jsonl:1"):
        load_history(path)
