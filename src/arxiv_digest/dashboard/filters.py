"""Pure candidate-explorer projections, filtering, and deterministic sorting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from arxiv_digest.models import (
    DigestArtifact,
    PaperSummary,
    RankedPaper,
    RecommendationType,
)

DateField = Literal["published", "updated"]
SortField = Literal["rank", "score", "publication_date", "update_date"]


@dataclass(frozen=True, slots=True)
class ExplorerPaper:
    """Dashboard status layered onto one canonical ranked candidate."""

    ranked: RankedPaper
    ranking_position: int
    selected: bool
    near_miss: bool
    recommendation_rank: int | None
    recommendation_type: RecommendationType | None
    summary: PaperSummary | None


def build_explorer_papers(digest: DigestArtifact) -> list[ExplorerPaper]:
    """Join ranked candidates to selection, summary, and near-miss state."""
    selected = {item.paper.arxiv_id: item for item in digest.recommendations}
    near_miss_ids = {item.paper.arxiv_id for item in digest.near_misses}
    return [
        ExplorerPaper(
            ranked=item,
            ranking_position=index,
            selected=item.paper.arxiv_id in selected,
            near_miss=item.paper.arxiv_id in near_miss_ids,
            recommendation_rank=(
                selected[item.paper.arxiv_id].rank if item.paper.arxiv_id in selected else None
            ),
            recommendation_type=(
                selected[item.paper.arxiv_id].recommendation_type
                if item.paper.arxiv_id in selected
                else None
            ),
            summary=(
                selected[item.paper.arxiv_id].summary if item.paper.arxiv_id in selected else None
            ),
        )
        for index, item in enumerate(digest.ranked_candidates, start=1)
    ]


def filter_papers(
    papers: list[ExplorerPaper],
    *,
    search: str = "",
    primary_categories: set[str] | None = None,
    recommendation_types: set[RecommendationType] | None = None,
    minimum_score: float = 0.0,
    start_date: date | None = None,
    end_date: date | None = None,
    date_field: DateField = "updated",
    selected_only: bool = False,
    include_near_misses: bool = False,
) -> list[ExplorerPaper]:
    """Apply composable explorer filters without mutating canonical data."""
    categories = primary_categories or set()
    types = recommendation_types or set()
    words = search.casefold().split()
    result: list[ExplorerPaper] = []
    for item in papers:
        paper = item.ranked.paper
        searchable = " ".join((paper.title, " ".join(paper.authors), paper.abstract)).casefold()
        if words and not all(word in searchable for word in words):
            continue
        if categories and paper.primary_category not in categories:
            continue
        if types and item.recommendation_type not in types:
            continue
        if item.ranked.score.final_preselection_score < minimum_score:
            continue
        relevant_date = (
            paper.published_at.date() if date_field == "published" else paper.updated_at.date()
        )
        if start_date is not None and relevant_date < start_date:
            continue
        if end_date is not None and relevant_date > end_date:
            continue
        if selected_only and not (item.selected or (include_near_misses and item.near_miss)):
            continue
        result.append(item)
    return result


def sort_papers(
    papers: list[ExplorerPaper],
    *,
    sort_by: SortField = "rank",
    descending: bool | None = None,
) -> list[ExplorerPaper]:
    """Sort candidates with canonical arXiv ID as a stable final tie-break."""
    reverse = (sort_by != "rank") if descending is None else descending
    if sort_by == "rank":
        return sorted(
            papers,
            key=lambda item: (item.ranking_position, item.ranked.paper.arxiv_id),
            reverse=reverse,
        )
    if sort_by == "score":
        return sorted(
            papers,
            key=lambda item: (
                item.ranked.score.final_preselection_score,
                item.ranked.paper.updated_at,
                item.ranked.paper.arxiv_id,
            ),
            reverse=reverse,
        )
    if sort_by == "publication_date":
        return sorted(
            papers,
            key=lambda item: (
                item.ranked.paper.published_at,
                item.ranked.score.final_preselection_score,
                item.ranked.paper.arxiv_id,
            ),
            reverse=reverse,
        )
    return sorted(
        papers,
        key=lambda item: (
            item.ranked.paper.updated_at,
            item.ranked.score.final_preselection_score,
            item.ranked.paper.arxiv_id,
        ),
        reverse=reverse,
    )


def strongest_profile_terms(explanation: str) -> list[str]:
    """Extract the transparent strongest-term list from a score explanation."""
    match = re.search(
        r"strongest matched profile terms:\s*([^.;]+)",
        explanation,
        flags=re.IGNORECASE,
    )
    if match is None:
        return []
    return [term.strip() for term in match.group(1).split(",") if term.strip()]
