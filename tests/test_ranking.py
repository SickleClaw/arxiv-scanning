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
    title_score, title_terms = keyword_relevance(title_match, profile)
    abstract_score, abstract_terms = keyword_relevance(abstract_match, profile)
    assert title_score > abstract_score
    assert title_terms[0][0] == abstract_terms[0][0] == "spin ice"


def test_whole_term_matching_avoids_substring_false_positive(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile
    paper = paper_factory(
        title="Spineless optimization",
        abstract="A spineless model for a social network.",
    )
    score, matched = keyword_relevance(paper, profile)
    assert score == 0.0
    assert all(term != "spinel" for term, _contribution in matched)


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
            )
        }
    )
    category_profile = profile.model_copy(
        update={
            "ranking_weights": RankingWeights(
                semantic_relevance=0,
                keyword_relevance=0,
                category_relevance=1,
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


def test_author_boost_is_bounded(paper_factory) -> None:  # type: ignore[no-untyped-def]
    base = load_settings().profile
    boosted_profile = base.model_copy(update={"author_boosts": {"Ada Curie": 2.0}})
    paper = paper_factory(title="A general magnetic study")
    base_score, _matched = keyword_relevance(paper, base)
    boosted_score, _matched = keyword_relevance(paper, boosted_profile)
    assert base_score < boosted_score <= base_score + 0.15


def test_ranking_no_longer_penalizes_negative_terms(paper_factory) -> None:  # type: ignore[no-untyped-def]
    """Negative terms are a gate, not a penalty a strong score could outvote.

    A paper matching one never reaches ranking; domain_filter.gate_papers
    removes it with a logged reason. Scoring must therefore treat the two
    papers below identically.
    """
    profile = load_settings().profile
    clean = paper_factory(arxiv_id="2607.13001", abstract="A general magnetic study.")
    formerly_penalized = paper_factory(
        arxiv_id="2607.13002",
        abstract="A general magnetic study of a financial market.",
    )
    scores = {
        item.paper.arxiv_id: item.score.final_preselection_score
        for item in rank_papers([clean, formerly_penalized], profile, ranking_window())
    }
    # The old penalty was a flat 0.20. What remains is only the incidental
    # TF-IDF difference between two slightly different abstracts.
    assert abs(scores["2607.13002"] - scores["2607.13001"]) < 0.05


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


def test_keyword_evidence_saturates_instead_of_dividing_by_every_weight(paper_factory) -> None:  # type: ignore[no-untyped-def]
    """Three strong title matches should score around 0.70, not 0.09."""
    profile = load_settings().profile
    strong = paper_factory(
        title="Spin ice, pyrochlore, and neutron scattering",
        abstract="A study.",
    )
    score, matched = keyword_relevance(strong, profile)
    assert len(matched) >= 3
    assert 0.55 < score < 0.85


def test_adding_interests_does_not_dilute_existing_matches(paper_factory) -> None:  # type: ignore[no-untyped-def]
    """The old denominator was the sum of every configured weight.

    Under it, adding an unrelated interest lowered the score of every paper
    matching the interests already there.
    """
    profile = load_settings().profile
    paper = paper_factory(title="A spin ice study", abstract="Spin ice dynamics.")
    before, _matched = keyword_relevance(paper, profile)
    widened = profile.model_copy(
        update={
            "exact_phrases": {
                **profile.exact_phrases,
                **{f"unrelated topic {index}": 2.0 for index in range(20)},
            }
        }
    )
    after, _matched = keyword_relevance(paper, widened)
    assert after == pytest.approx(before)


def test_keyword_score_is_monotone_in_evidence(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile
    one = paper_factory(title="Spin ice", abstract="A study.")
    two = paper_factory(title="Spin ice and pyrochlore", abstract="A study.")
    three = paper_factory(title="Spin ice, pyrochlore, neutron scattering", abstract="A study.")
    scores = [keyword_relevance(item, profile)[0] for item in (one, two, three)]
    assert scores == sorted(scores)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_facet_weighting_makes_a_method_match_weaker_than_a_topic_match(paper_factory) -> None:  # type: ignore[no-untyped-def]
    """A technique match alone is weaker evidence than a topic or material."""
    profile = load_settings().profile
    equal_weights = profile.model_copy(
        update={
            "exact_phrases": {"spin ice": 2.0},
            "materials": {},
            "methods": {"spin dynamics": 2.0},
        }
    )
    topic = paper_factory(title="A spin ice paper", abstract="A study.")
    method = paper_factory(title="A spin dynamics paper", abstract="A study.")
    topic_score, _matched = keyword_relevance(topic, equal_weights)
    method_score, _matched = keyword_relevance(method, equal_weights)
    assert method_score < topic_score


def test_keyword_scores_span_a_real_range_on_the_committed_window() -> None:
    """Observed range was 0.00-0.09; the signal was too weak to matter."""
    snapshot = CandidateSnapshot.model_validate_json(CANDIDATES_PATH.read_text(encoding="utf-8"))
    profile = load_settings().profile
    scores = [keyword_relevance(item, profile)[0] for item in snapshot.papers]
    assert max(scores) > 0.25


def test_an_irrelevant_paper_scores_near_zero_whenever_it_was_submitted(paper_factory) -> None:  # type: ignore[no-untyped-def]
    """Finding 2, asserted directly.

    Under the old formula a paper matching nothing scored at least 0.20 from
    novelty and recency alone — equal to the adjacent threshold. Submission date
    must no longer be able to manufacture a score.
    """
    profile = load_settings().profile
    window = ranking_window()
    early = paper_factory(
        arxiv_id="2607.40001",
        title="An entirely unrelated result",
        abstract="This concerns something with no bearing on the profile at all.",
        primary_category="physics.flu-dyn",
        categories=["physics.flu-dyn"],
        published_at=datetime(2026, 7, 20, 1, tzinfo=UTC),
        updated_at=datetime(2026, 7, 20, 1, tzinfo=UTC),
    )
    late = early.model_copy(
        update={
            "arxiv_id": "2607.40002",
            "published_at": datetime(2026, 7, 27, 23, tzinfo=UTC),
            "updated_at": datetime(2026, 7, 27, 23, tzinfo=UTC),
        }
    )
    scores = {
        item.paper.arxiv_id: item.score.final_preselection_score
        for item in rank_papers([early, late], profile, window)
    }
    assert max(scores.values()) < profile.selection.wildcard_min_score
    # Recency may separate them, but only marginally.
    assert abs(scores["2607.40002"] - scores["2607.40001"]) < 0.02


def test_recency_modulates_but_cannot_manufacture_a_score(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile
    window = ranking_window()
    relevant = paper_factory(
        arxiv_id="2607.41001",
        title="Spin ice and pyrochlore neutron scattering",
        abstract="Magnetic monopole dynamics in spin ice.",
        published_at=datetime(2026, 7, 21, tzinfo=UTC),
        updated_at=datetime(2026, 7, 21, tzinfo=UTC),
    )
    irrelevant_but_newer = paper_factory(
        arxiv_id="2607.41002",
        title="An entirely unrelated result",
        abstract="Nothing in the profile appears here.",
        primary_category="physics.flu-dyn",
        categories=["physics.flu-dyn"],
        published_at=datetime(2026, 7, 27, 23, tzinfo=UTC),
        updated_at=datetime(2026, 7, 27, 23, tzinfo=UTC),
    )
    ordered = rank_papers([irrelevant_but_newer, relevant], profile, window)
    assert ordered[0].paper.arxiv_id == "2607.41001"


def test_an_in_field_paper_is_not_scored_zero_on_an_unlisted_subcategory(paper_factory) -> None:  # type: ignore[no-untyped-def]
    """Unlisted is not the same as off-topic."""
    settings = load_settings()
    unlisted = paper_factory(
        primary_category="cond-mat.dis-nn",
        categories=["cond-mat.dis-nn"],
    )
    assert category_relevance(unlisted, settings.profile) == 0.0
    floored = category_relevance(unlisted, settings.profile, settings.group.domain)
    assert floored == pytest.approx(settings.profile.scoring.in_field_category_floor)
    # An out-of-field paper gets no floor.
    off_domain = paper_factory(primary_category="hep-ph", categories=["hep-ph"])
    assert category_relevance(off_domain, settings.profile, settings.group.domain) == 0.0


def test_committed_thresholds_reject_the_reported_plasma_paper() -> None:
    """The second paper the audit named, removed by score rather than by gate."""
    settings = load_settings()
    snapshot = CandidateSnapshot.model_validate_json(CANDIDATES_PATH.read_text(encoding="utf-8"))
    ranked = rank_papers(
        snapshot.papers,
        settings.profile,
        snapshot.retrieval_window,
        None,
        settings.group.domain,
    )
    plasma = next(item for item in ranked if item.paper.arxiv_id == "2607.25481")
    assert plasma.score.final_preselection_score < settings.profile.selection.wildcard_min_score
