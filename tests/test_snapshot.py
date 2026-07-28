"""Tests for deterministic candidate snapshot output."""

from datetime import UTC, datetime
from pathlib import Path

from arxiv_digest.models import CandidateSnapshot, DateWindow, QueryResult
from arxiv_digest.snapshot import create_snapshot, make_run_id, write_snapshot


def test_snapshot_run_id_and_atomic_overwrite(tmp_path: Path, paper_factory) -> None:  # type: ignore[no-untyped-def]
    retrieval_window = DateWindow(
        start=datetime(2026, 7, 20, tzinfo=UTC),
        end=datetime(2026, 7, 28, tzinfo=UTC),
    )
    paper = paper_factory()
    snapshot = create_snapshot(
        window=retrieval_window,
        profile_version="1.0",
        generated_at=datetime(2026, 7, 28, 12, tzinfo=UTC),
        query_results=[QueryResult(name="test", records_received=2)],
        raw_papers=[paper, paper],
        deduplicated_papers=[paper],
    )
    assert snapshot.run_id == make_run_id(retrieval_window, "1.0")
    assert snapshot.records_retrieved == 2
    path = write_snapshot(snapshot, tmp_path / "candidates.json")
    write_snapshot(snapshot, path)
    loaded = CandidateSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded == snapshot
    assert not (tmp_path / "candidates.json.tmp").exists()
