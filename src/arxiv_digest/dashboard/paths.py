"""Repository-root-relative paths for local and hosted dashboard execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from arxiv_digest.dashboard.data import DashboardDataError


@dataclass(frozen=True, slots=True)
class DashboardPaths:
    """Resolved read-only paths used by the dashboard."""

    repository_root: Path
    report: Path
    reports_dir: Path


def find_repository_root(start: Path | None = None) -> Path:
    """Find the checkout root without relying on the process working directory."""
    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "src" / "arxiv_digest"
        ).is_dir():
            return directory
    raise DashboardDataError(
        "Cannot locate the application repository root. Start the dashboard from a complete "
        "checkout containing pyproject.toml and src/arxiv_digest."
    )


def _from_root(path: Path, repository_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def resolve_dashboard_paths(
    *,
    report: Path | None = None,
    reports_dir: Path | None = None,
    repository_root: Path | None = None,
) -> DashboardPaths:
    """Resolve defaults and relative overrides against the checkout root."""
    root = (repository_root or find_repository_root()).resolve()
    resolved_reports = _from_root(reports_dir or Path("reports"), root)
    resolved_report = (
        _from_root(report, root) if report is not None else resolved_reports / "latest.json"
    )
    return DashboardPaths(
        repository_root=root,
        report=resolved_report,
        reports_dir=resolved_reports,
    )
