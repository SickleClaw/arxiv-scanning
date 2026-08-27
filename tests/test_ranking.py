"""Tests for deterministic offline hybrid ranking."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arxiv_digest.config import RankingWeights, load_settings
from arxiv_digest.models import CandidateSnapshot, DateWindow
from arxiv_digest.ranking import (
    category_relevance,
    keyword_relevance,
    rank_papers,
)

FIXTURES = Path(__file__).parent / "fixtures"
CANDIDATES_PATH = FIXTURES / "candidates_2026-07-21.json"
BASELINE_PATH = FIXTURES / "baseline_2026-07-21.json"


def ranking_window() -> DateWindow:
    return DateWindow(
        start=datetime(2026, 7, 20, tzinfo=UTC),
        end=datetime(2026, 7, 28, tzinfo=UTC),
    )


def test_title_match_outweighs_abstract_match(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile
    title_match = paper_factory(
        arxiv_id="2607.10001",
        title="Spin ice dynamics",
        abstract="A general magnetic study.",
    )
    abstract_match = paper_factory(
        arxiv_id="2607.10002",
        title="A general magnetic study",
        abstract="We investigate spin ice dynamics.",
    )
    title_score, title_terms, _negative = keyword_relevance(title_match, profile)
    abstract_score, abstract_terms, _negative = keyword_relevance(abstract_match, profile)
    assert title_score > abstract_score
    assert title_terms[0][0] == abstract_terms[0][0] == "spin ice"


def test_whole_term_matching_avoids_substring_false_positive(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile
    paper = paper_factory(
        title="Spineless optimization",
        abstract="A spineless model for a social network.",
    )
    score, matched, negative = keyword_relevance(paper, profile)
    assert score == 0.0
    assert all(term != "spinel" for term, _contribution in matched)
    assert "social network" in negative


def test_category_weight_is_normalized(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile
    strongest = paper_factory(categories=["cond-mat.str-el"])
    secondary = paper_factory(
        arxiv_id="2607.10003",
        primary_category="cond-mat.mes-hall",
        categories=["cond-mat.mes-hall"],
    )
    unknown = paper_factory(
        arxiv_id="2607.10004",
        primary_category="math.AG",
        categories=["math.AG"],
    )
    assert category_relevance(strongest, profile) == 1.0
    assert category_relevance(secondary, profile) == pytest.approx(0.5)
    assert category_relevance(unknown, profile) == 0.0


def test_changing_weights_changes_ranking_predictably(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile
    keyword_paper = paper_factory(
        arxiv_id="2607.11001",
        title="Spin ice magnetic monopole neutron diffuse scattering",
        primary_category="math.AG",
        categories=["math.AG"],
    )
    category_paper = paper_factory(
        arxiv_id="2607.11002",
        title="Unrelated topology",
        abstract="An unrelated abstract.",
        primary_category="cond-mat.str-el",
        categories=["cond-mat.str-el"],
    )
    keyword_profile = profile.model_copy(
        update={
            "ranking_weights": RankingWeights(
                semantic_relevance=0,
                keyword_relevance=1,
                category_relevance=0,
                recency=0,
                feedback_or_novelty=0,
            )
        }
    )
    category_profile = profile.model_copy(
        update={
            "ranking_weights": RankingWeights(
                semantic_relevance=0,
                keyword_relevance=0,
                category_relevance=1,
                recency=0,
                feedback_or_novelty=0,
            )
        }
    )
    assert (
        rank_papers([category_paper, keyword_paper], keyword_profile, ranking_window())[
            0
        ].paper.arxiv_id
        == "2607.11001"
    )
    assert (
        rank_papers([keyword_paper, category_paper], category_profile, ranking_window())[
            0
        ].paper.arxiv_id
        == "2607.11002"
    )


def test_ranking_is_deterministic_and_explains_strongest_terms(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile
    papers = [
        paper_factory(arxiv_id="2607.12002", title="Spin ice dynamics"),
        paper_factory(arxiv_id="2607.12001", title="Neutron diffuse scattering"),
    ]
    first = rank_papers(papers, profile, ranking_window())
    second = rank_papers(reversed(papers), profile, ranking_window())
    assert [item.paper.arxiv_id for item in first] == [item.paper.arxiv_id for item in second]
    assert "Strongest matched profile terms:" in first[0].score.explanation
    assert all(0.0 <= item.score.final_preselection_score <= 1.0 for item in first)


def test_author_boost_and_negative_penalty_are_bounded(paper_factory) -> None:  # type: ignore[no-untyped-def]
    base = load_settings().profile
    boosted_profile = base.model_copy(update={"author_boosts": {"Ada Curie": 2.0}})
    paper = paper_factory(title="A general magnetic study")
    base_score, _matched, _negative = keyword_relevance(paper, base)
    boosted_score, _matched, _negative = keyword_relevance(paper, boosted_profile)
    assert base_score < boosted_score <= base_score + 0.15

    novelty_profile = base.model_copy(
        update={
            "ranking_weights": RankingWeights(
                semantic_relevance=0,
                keyword_relevance=0,
                category_relevance=0,
                recency=0,
                feedback_or_novelty=1,
            )
        }
    )
    clean = paper_factory(arxiv_id="2607.13001", abstract="A general magnetic study.")
    penalized = paper_factory(
        arxiv_id="2607.13002",
        abstract="A general magnetic study of a financial market.",
    )
    scores = {
        item.paper.arxiv_id: item.score.final_preselection_score
        for item in rank_papers([clean, penalized], novelty_profile, ranking_window())
    }
    assert scores["2607.13002"] == pytest.approx(scores["2607.13001"] - 0.20)


def test_frozen_baseline_matches_current_ranking() -> None:
    """Guard the whole scoring chain against unintended drift.

    Any change to retrieval-independent scoring shows up here as an explicit
    diff. When a change is intentional, regenerate with
    ``python tests/tools/regenerate_baseline.py`` and review the result.
    """
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    settings = load_settings()
    assert baseline["profile_name"] == settings.profile.name
    assert baseline["profile_version"] == settings.profile.version

    snapshot = CandidateSnapshot.model_validate_json(CANDIDATES_PATH.read_text(encoding="utf-8"))
    ranked = rank_papers(snapshot.papers, settings.profile, snapshot.retrieval_window, set())
    assert len(ranked) == len(baseline["ranking"])
    for expected, actual in zip(baseline["ranking"], ranked, strict=True):
        assert actual.paper.arxiv_id == expected["arxiv_id"]
        assert actual.score.final_preselection_score == pytest.approx(
            expected["final_preselection_score"], abs=1e-6
        )
        components = expected["components"]
        assert actual.score.semantic_relevance == pytest.approx(
            components["semantic_relevance"], abs=1e-6
        )
        assert actual.score.keyword_relevance == pytest.approx(
            components["keyword_relevance"], abs=1e-6
        )
        assert actual.score.category_relevance == pytest.approx(
            components["category_relevance"], abs=1e-6
        )
        assert actual.score.recency == pytest.approx(components["recency"], abs=1e-6)
