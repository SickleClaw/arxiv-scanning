"""Append-only JSONL recommendation history and resurfacing rules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from arxiv_digest.exceptions import HistoryError
from arxiv_digest.models import (
    DateWindow,
    HistoryRecord,
    Paper,
    SelectedPaper,
)


def load_history(path: Path) -> list[HistoryRecord]:
    """Load and validate an append-only history file; a missing file is empty history."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HistoryError(f"Cannot read recommendation history {path}: {exc}") from exc
    records: list[HistoryRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(HistoryRecord.model_validate_json(line))
        except (ValidationError, ValueError) as exc:
            raise HistoryError(
                f"Invalid recommendation history at {path}:{line_number}: {exc}"
            ) from exc
    return records


def filter_recent_history(
    papers: Iterable[Paper],
    history: Sequence[HistoryRecord],
    *,
    now: datetime,
    exclusion_days: int,
    allow_updated_resurfacing: bool,
) -> tuple[list[Paper], int]:
    """Exclude recent recommendations, allowing only qualifying new versions to resurface."""
    cutoff = now - timedelta(days=exclusion_days)
    latest_by_id: dict[str, HistoryRecord] = {}
    for record in history:
        current = latest_by_id.get(record.arxiv_id)
        if current is None or record.run_timestamp > current.run_timestamp:
            latest_by_id[record.arxiv_id] = record

    included: list[Paper] = []
    excluded = 0
    for paper in papers:
        previous = latest_by_id.get(paper.arxiv_id)
        if previous is None or previous.run_timestamp < cutoff:
            included.append(paper)
            continue
        is_qualifying_update = (
            allow_updated_resurfacing
            and paper.version != previous.version
            and paper.updated_at > previous.run_timestamp
        )
        if is_qualifying_update:
            included.append(paper)
        else:
            excluded += 1
    return included, excluded


def recommended_ids(history: Sequence[HistoryRecord]) -> set[str]:
    """Return canonical identifiers seen at any point in history for novelty scoring."""
    return {record.arxiv_id for record in history}


def records_for_selection(
    *,
    run_id: str,
    run_timestamp: datetime,
    window: DateWindow,
    selected: Sequence[SelectedPaper],
    report_path: str | None = None,
) -> list[HistoryRecord]:
    """Convert selected papers into validated append-only history records."""
    return [
        HistoryRecord(
            run_id=run_id,
            run_timestamp=run_timestamp,
            retrieval_window=window,
            arxiv_id=item.paper.arxiv_id,
            version=item.paper.version,
            paper_updated_at=item.paper.updated_at,
            rank=item.rank,
            recommendation_type=item.recommendation_type,
            final_score=item.score.final_preselection_score,
            report_path=report_path,
        )
        for item in selected
    ]


def append_history(path: Path, records: Sequence[HistoryRecord]) -> int:
    """Append only missing run/paper records and reject run-ID window collisions."""
    if not records:
        return 0
    intended_run_id = records[0].run_id
    intended_window = records[0].retrieval_window
    if any(
        record.run_id != intended_run_id
        or record.retrieval_window.start != intended_window.start
        or record.retrieval_window.end != intended_window.end
        for record in records
    ):
        raise HistoryError("A history append batch must contain one run ID and retrieval window")
    existing = load_history(path)
    existing_windows = {
        (record.retrieval_window.start, record.retrieval_window.end)
        for record in existing
        if record.run_id == intended_run_id
    }
    intended_key = (intended_window.start, intended_window.end)
    if existing_windows and existing_windows != {intended_key}:
        raise HistoryError(
            f"Run ID {intended_run_id!r} already exists for a different retrieval window"
        )
    if existing_windows:
        return 0
    existing_keys = {(record.run_id, record.arxiv_id) for record in existing}
    missing = [
        record for record in records if (record.run_id, record.arxiv_id) not in existing_keys
    ]
    if not missing:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            for record in missing:
                stream.write(record.model_dump_json())
                stream.write("\n")
    except OSError as exc:
        raise HistoryError(f"Cannot append recommendation history {path}: {exc}") from exc
    return len(missing)
