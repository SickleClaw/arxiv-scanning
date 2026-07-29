"""Deterministic tests for dashboard candidate filtering and sorting."""

from __future__ import annotations

from datetime import date

from arxiv_digest.dashboard.filters import (
    build_explorer_papers,
    filter_papers,
    sort_papers,
    strongest_profile_terms,
)
from arxiv_digest.models import RecommendationType


def test_build_explorer_papers_assigns_all_statuses(digest_factory) -> None:  # type: ignore[no-untyped-def]
    papers = build_explorer_papers(digest_factory())
    assert [(item.selected, item.near_miss) for item in papers] == [
        (True, False),
        (True, False),
        (False, True),
        (False, False),
    ]
    assert papers[0].summary is not None
    assert papers[2].summary is None


def test_text_category_type_and_score_filters_compose(digest_factory) -> None:  # type: ignore[no-untyped-def]
    papers = build_explorer_papers(digest_factory())
    neutron = filter_papers(
        papers,
        search="nia neutron",
        primary_categories={"cond-mat.mtrl-sci"},
        recommendation_types={RecommendationType.ADJACENT},
        minimum_score=0.8,
    )
    assert [item.ranked.paper.arxiv_id for item in neutron] == ["2607.10002"]
    assert filter_papers(papers, search="no matching phrase") == []


def test_date_selected_and_near_miss_filters(digest_factory) -> None:  # type: ignore[no-untyped-def]
    papers = build_explorer_papers(digest_factory())
    selected = filter_papers(papers, selected_only=True)
    selected_and_near = filter_papers(
        papers,
        selected_only=True,
        include_near_misses=True,
    )
    published = filter_papers(
        papers,
        start_date=date(2026, 7, 24),
        end_date=date(2026, 7, 24),
        date_field="published",
    )
    assert len(selected) == 2
    assert len(selected_and_near) == 3
    assert [item.ranked.paper.arxiv_id for item in published] == ["2607.10002"]


def test_sorting_is_deterministic_for_all_supported_fields(digest_factory) -> None:  # type: ignore[no-untyped-def]
    papers = build_explorer_papers(digest_factory())
    assert [item.ranking_position for item in sort_papers(papers)] == [1, 2, 3, 4]
    assert [item.ranking_position for item in sort_papers(papers, sort_by="score")] == [
        1,
        2,
        3,
        4,
    ]
    assert [
        item.ranking_position
        for item in sort_papers(papers, sort_by="publication_date", descending=False)
    ] == [4, 1, 2, 3]
    assert [
        item.ranking_position
        for item in sort_papers(papers, sort_by="update_date", descending=True)
    ] == [1, 2, 3, 4]


def test_strongest_terms_parse_or_return_empty() -> None:
    assert strongest_profile_terms(
        "Strongest matched profile terms: spin ice, neutron scattering. Category overlap."
    ) == ["spin ice", "neutron scattering"]
    assert strongest_profile_terms("No exact configured term matches.") == []
