"""Tests for validated dashboard loading and local report history."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arxiv_digest.dashboard.data import (
    DashboardDataError,
    discover_digest_paths,
    load_digest,
    load_digest_history,
    repeated_papers,
)
from arxiv_digest.models import DigestArtifact


def _write(path: Path, digest: DigestArtifact) -> None:
    path.write_text(digest.model_dump_json(indent=2), encoding="utf-8")


def test_load_digest_accepts_valid_artifact(tmp_path: Path, digest_factory) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "latest.json"
    _write(path, digest_factory())
    loaded = load_digest(path)
    assert loaded.records_selected == 2
    assert loaded.ranked_candidates[0].paper.arxiv_id == "2607.10001"


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not json", "malformed JSON"),
        ("[]", "must contain a JSON object"),
        ("{}", "has no schema_version"),
        ('{"schema_version": "99"}', "unsupported schema"),
    ],
)
def test_load_digest_reports_safe_actionable_errors(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(DashboardDataError, match=message):
        load_digest(path)


def test_load_digest_reports_missing_and_invalid_counts(tmp_path: Path, digest_factory) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(DashboardDataError, match="Run 'arxiv-digest run --dry-run'"):
        load_digest(tmp_path / "missing.json")
    payload = digest_factory().model_dump(mode="json")
    payload["records_ranked"] = 99
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DashboardDataError, match="records_ranked"):
        load_digest(path)


def test_history_discovery_is_newest_first_and_isolates_bad_files(
    tmp_path: Path, digest_factory
) -> None:  # type: ignore[no-untyped-def]
    digest = digest_factory()
    old = tmp_path / "2026-07-21-weekly-arxiv-digest.json"
    new = tmp_path / "2026-07-28-weekly-arxiv-digest.json"
    _write(old, digest)
    _write(new, digest)
    _write(tmp_path / "latest.json", digest)
    (tmp_path / "2026-07-27-weekly-arxiv-digest.json").write_text("bad", encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    assert [value[0].isoformat() for value in discover_digest_paths(tmp_path)] == [
        "2026-07-28",
        "2026-07-27",
        "2026-07-21",
    ]
    history, errors = load_digest_history(tmp_path)
    assert [entry.report_date.isoformat() for entry, _digest in history] == [
        "2026-07-28",
        "2026-07-21",
    ]
    assert len(errors) == 1
    assert "malformed JSON" in errors[0]


def test_empty_history_and_updated_version_resurfacing(tmp_path: Path, digest_factory) -> None:  # type: ignore[no-untyped-def]
    assert discover_digest_paths(tmp_path) == []
    assert load_digest_history(tmp_path) == ([], [])
    first = digest_factory(generated_at=datetime(2026, 7, 21, 12, tzinfo=UTC))
    payload = digest_factory(
        generated_at=datetime(2026, 7, 28, 12, tzinfo=UTC),
        run_id="selection-second",
    ).model_dump(mode="json")
    payload["recommendations"][0]["paper"]["version"] = 2
    payload["recommendations"][0]["paper"]["updated_at"] = "2026-07-28T10:00:00Z"
    payload["ranked_candidates"][0]["paper"]["version"] = 2
    payload["ranked_candidates"][0]["paper"]["updated_at"] = "2026-07-28T10:00:00Z"
    second = DigestArtifact.model_validate(payload)
    _write(tmp_path / "2026-07-21-weekly-arxiv-digest.json", first)
    _write(tmp_path / "2026-07-28-weekly-arxiv-digest.json", second)
    history, errors = load_digest_history(tmp_path)
    repeated = repeated_papers(history)
    spin_ice = next(item for item in repeated if item.arxiv_id == "2607.10001")
    assert errors == []
    assert spin_ice.versions == (1, 2)
    assert spin_ice.updated_version_resurfaced is True
