"""Candidate snapshot construction and atomic JSON persistence."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from arxiv_digest.models import CandidateSnapshot, DateWindow, Paper, QueryResult


def make_run_id(window: DateWindow, profile_version: str) -> str:
    """Create a deterministic identifier for a profile and retrieval window."""
    source = f"{window.start.isoformat()}|{window.end.isoformat()}|{profile_version}"
    suffix = hashlib.sha256(source.encode()).hexdigest()[:10]
    return f"fetch-{window.start:%Y%m%d}-{window.end:%Y%m%d}-{suffix}"


def create_snapshot(
    *,
    window: DateWindow,
    profile_version: str,
    generated_at: datetime,
    query_results: list[QueryResult],
    raw_papers: list[Paper],
    deduplicated_papers: list[Paper],
) -> CandidateSnapshot:
    """Build a validated machine-readable snapshot."""
    return CandidateSnapshot(
        run_id=make_run_id(window, profile_version),
        generated_at=generated_at.astimezone(UTC),
        retrieval_window=window,
        queries=query_results,
        records_retrieved=len(raw_papers),
        records_after_deduplication=len(deduplicated_papers),
        papers=deduplicated_papers,
    )


def default_snapshot_path(data_dir: Path, window: DateWindow) -> Path:
    """Return the stable output path used for repeated fetches of one window."""
    inclusive_end = (window.end - timedelta(microseconds=1)).date().isoformat()
    return data_dir / f"candidates-{window.start.date().isoformat()}-{inclusive_end}.json"


def write_snapshot(snapshot: CandidateSnapshot, path: Path) -> Path:
    """Atomically write a UTF-8 snapshot, replacing the same-window output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return path
