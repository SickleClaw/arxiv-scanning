"""Jinja-based Markdown and HTML weekly report rendering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError, select_autoescape

from arxiv_digest.config import ResearchProfile
from arxiv_digest.exceptions import ReportingError
from arxiv_digest.models import (
    ABSTRACT_SUMMARY_BASIS,
    CandidateSnapshot,
    RankedPaper,
    RankedSnapshot,
    Recommendation,
)


@dataclass(frozen=True, slots=True)
class ReportPaths:
    """Dated and latest report artifacts written together."""

    markdown: Path
    html: Path
    latest_markdown: Path
    latest_html: Path


def profile_hash(profile: ResearchProfile) -> str:
    """Return a short stable hash of the effective research profile."""
    return hashlib.sha256(profile.model_dump_json().encode()).hexdigest()[:12]


def near_misses(
    ranked: RankedSnapshot,
    recommendations: list[Recommendation],
    *,
    limit: int,
) -> list[RankedPaper]:
    """Return the strongest candidates not selected for the final list."""
    selected_ids = {item.paper.arxiv_id for item in recommendations}
    return [item for item in ranked.ranked_papers if item.paper.arxiv_id not in selected_ids][
        :limit
    ]


def _write_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(contents, encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise ReportingError(f"Cannot write report {path}: {exc}") from exc


def render_reports(
    *,
    snapshot: CandidateSnapshot,
    ranked: RankedSnapshot,
    recommendations: list[Recommendation],
    profile: ResearchProfile,
    generated_at: datetime,
    timezone: str,
    templates_dir: Path,
    reports_dir: Path,
    near_miss_limit: int,
) -> ReportPaths:
    """Render and atomically write dated/latest Markdown and HTML reports."""
    local_generated_at = generated_at.astimezone(ZoneInfo(timezone))
    report_date = local_generated_at.date().isoformat()
    inclusive_end = (snapshot.retrieval_window.end - timedelta(microseconds=1)).date()
    context = {
        "generated_at": local_generated_at,
        "timezone": timezone,
        "retrieval_start": snapshot.retrieval_window.start.date(),
        "retrieval_end": inclusive_end,
        "records_retrieved": snapshot.records_retrieved,
        "records_after_deduplication": snapshot.records_after_deduplication,
        "records_ranked": len(ranked.ranked_papers),
        "summary_basis": ABSTRACT_SUMMARY_BASIS,
        "profile_name": profile.name,
        "profile_version": profile.version,
        "profile_hash": profile_hash(profile),
        "recommendations": recommendations,
        "near_misses": near_misses(
            ranked,
            recommendations,
            limit=near_miss_limit,
        ),
    }
    environment = Environment(
        loader=FileSystemLoader(templates_dir),
        undefined=StrictUndefined,
        autoescape=select_autoescape(enabled_extensions=("html.j2",)),
        keep_trailing_newline=True,
    )
    try:
        markdown = environment.get_template("weekly_report.md.j2").render(context)
        html = environment.get_template("weekly_report.html.j2").render(context)
    except (OSError, TemplateError) as exc:
        raise ReportingError(
            f"Cannot render weekly report templates from {templates_dir}: {exc}"
        ) from exc

    paths = ReportPaths(
        markdown=reports_dir / f"{report_date}-weekly-arxiv-digest.md",
        html=reports_dir / f"{report_date}-weekly-arxiv-digest.html",
        latest_markdown=reports_dir / "latest.md",
        latest_html=reports_dir / "latest.html",
    )
    _write_atomic(paths.markdown, markdown)
    _write_atomic(paths.html, html)
    _write_atomic(paths.latest_markdown, markdown)
    _write_atomic(paths.latest_html, html)
    return paths
