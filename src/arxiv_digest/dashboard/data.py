"""Validated, read-only loading and history discovery for digest artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from arxiv_digest.models import DIGEST_SCHEMA_VERSION, DigestArtifact

_DATED_REPORT = re.compile(r"^(\d{4}-\d{2}-\d{2})-weekly-arxiv-digest\.json$")


class DashboardDataError(Exception):
    """A safe, actionable error while loading local dashboard data."""


@dataclass(frozen=True, slots=True)
class DigestIndexEntry:
    """Lightweight metadata for one dated digest report."""

    report_date: date
    path: Path
    generated_at: str
    recommendation_count: int


@dataclass(frozen=True, slots=True)
class RepeatedPaper:
    """A paper recommended in more than one locally available digest."""

    arxiv_id: str
    title: str
    report_dates: tuple[date, ...]
    versions: tuple[int, ...]

    @property
    def updated_version_resurfaced(self) -> bool:
        """Whether recommendation history contains more than one arXiv version."""
        return len(set(self.versions)) > 1


def load_digest(path: Path) -> DigestArtifact:
    """Load one supported canonical digest without performing network or write I/O."""
    label = path.name or "configured digest report"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DashboardDataError(
            f"Digest report not found: {label}. Run 'arxiv-digest run --dry-run' first, "
            "or choose an existing JSON report."
        ) from exc
    except OSError as exc:
        raise DashboardDataError(
            f"Cannot read digest report {label} ({type(exc).__name__})."
        ) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DashboardDataError(
            f"Digest report {label} is malformed JSON (line {exc.lineno}, column {exc.colno})."
        ) from exc
    if not isinstance(value, dict):
        raise DashboardDataError(f"Digest report {label} must contain a JSON object.")
    version = value.get("schema_version")
    if version is None:
        raise DashboardDataError(
            f"Digest report {label} has no schema_version and is not dashboard-compatible."
        )
    if version != DIGEST_SCHEMA_VERSION:
        raise DashboardDataError(
            f"Digest report {label} uses unsupported schema {version!r}; "
            f"this dashboard supports {DIGEST_SCHEMA_VERSION!r}."
        )
    try:
        return DigestArtifact.model_validate(value)
    except ValidationError as exc:
        problem = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in problem["loc"])
        raise DashboardDataError(
            f"Digest report {label} failed validation at {location or 'document'}: {problem['msg']}"
        ) from exc


def discover_digest_paths(reports_dir: Path) -> list[tuple[date, Path]]:
    """Find dated canonical reports newest-first, excluding the latest alias."""
    if not reports_dir.exists():
        return []
    if not reports_dir.is_dir():
        raise DashboardDataError("The configured reports path is not a directory.")
    discovered: list[tuple[date, Path]] = []
    try:
        paths = reports_dir.iterdir()
        for path in paths:
            match = _DATED_REPORT.fullmatch(path.name)
            if match and path.is_file():
                discovered.append((date.fromisoformat(match.group(1)), path))
    except OSError as exc:
        raise DashboardDataError(
            f"Cannot inspect the configured reports directory ({type(exc).__name__})."
        ) from exc
    return sorted(discovered, key=lambda item: (item[0], item[1].name), reverse=True)


def load_digest_history(
    reports_dir: Path,
) -> tuple[list[tuple[DigestIndexEntry, DigestArtifact]], list[str]]:
    """Load all valid dated digests while isolating individual stale or corrupt files."""
    loaded: list[tuple[DigestIndexEntry, DigestArtifact]] = []
    errors: list[str] = []
    for report_date, path in discover_digest_paths(reports_dir):
        try:
            digest = load_digest(path)
        except DashboardDataError as exc:
            errors.append(str(exc))
            continue
        entry = DigestIndexEntry(
            report_date=report_date,
            path=path,
            generated_at=digest.generated_at.astimezone(ZoneInfo(digest.timezone)).isoformat(),
            recommendation_count=len(digest.recommendations),
        )
        loaded.append((entry, digest))
    return loaded, errors


def repeated_papers(
    history: list[tuple[DigestIndexEntry, DigestArtifact]],
) -> list[RepeatedPaper]:
    """Summarize papers recommended across multiple available report dates."""
    occurrences: dict[str, list[tuple[date, str, int]]] = {}
    for entry, digest in history:
        for recommendation in digest.recommendations:
            occurrences.setdefault(recommendation.paper.arxiv_id, []).append(
                (entry.report_date, recommendation.paper.title, recommendation.paper.version)
            )
    result: list[RepeatedPaper] = []
    for arxiv_id, values in occurrences.items():
        if len(values) < 2:
            continue
        ordered = sorted(values, key=lambda value: value[0])
        result.append(
            RepeatedPaper(
                arxiv_id=arxiv_id,
                title=ordered[-1][1],
                report_dates=tuple(value[0] for value in ordered),
                versions=tuple(value[2] for value in ordered),
            )
        )
    return sorted(result, key=lambda item: (-len(item.report_dates), item.arxiv_id))
