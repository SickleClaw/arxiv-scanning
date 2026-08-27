"""Tests for MMR diversity, topic caps, strength thresholds, and mix allocation."""

from arxiv_digest.config import RecommendationMix, SelectionConfig, load_settings
from arxiv_digest.models import RankedPaper, RecommendationType, ScoreBreakdown
from arxiv_digest.selection import UNMATCHED_TOPIC, select_diverse, topic_key


def ranked(paper, score: float) -> RankedPaper:  # type: ignore[no-untyped-def]
    return RankedPaper(
        paper=paper,
        score=ScoreBreakdown(
            semantic_relevance=score,
            keyword_relevance=score,
            category_relevance=score,
            recency=score,
            relevance=score,
            final_preselection_score=score,
            explanation="fixture",
        ),
    )


def test_topic_cap_prevents_one_narrow_topic_from_dominating(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile.model_copy(update={"max_papers_per_topic": 1})
    candidates = [
        ranked(
            paper_factory(
                arxiv_id="2607.20001",
                title="Spin ice magnetic monopole dynamics",
            ),
            0.90,
        ),
        ranked(
            paper_factory(
                arxiv_id="2607.20002",
                title="Spin ice magnetic monopole transport",
            ),
            0.89,
        ),
        ranked(
            paper_factory(
                arxiv_id="2607.20003",
                title="Neutron diffuse scattering structure factor",
            ),
            0.75,
        ),
    ]
    selected = select_diverse(candidates, profile, limit=3)
    assert [item.paper.arxiv_id for item in selected] == ["2607.20001", "2607.20003"]
    assert len({item.topic_key for item in selected}) == 2


def test_mmr_prefers_less_redundant_candidate(paper_factory) -> None:  # type: ignore[no-untyped-def]
    base = load_settings().profile
    profile = base.model_copy(
        update={
            "selection": base.selection.model_copy(update={"mmr_lambda": 0.50}),
            "max_papers_per_topic": 3,
        }
    )
    candidates = [
        ranked(
            paper_factory(
                arxiv_id="2607.20501",
                title="Spin ice magnetic monopole dynamics",
                abstract="Spin ice magnetic monopole field dynamics.",
            ),
            0.90,
        ),
        ranked(
            paper_factory(
                arxiv_id="2607.20502",
                title="Spin ice magnetic monopole field dynamics",
                abstract="Spin ice magnetic monopole dynamics in a field.",
            ),
            0.89,
        ),
        ranked(
            paper_factory(
                arxiv_id="2607.20503",
                title="Neutron diffuse scattering in a frustrated spinel",
                abstract="A structure factor study of magnetic disorder.",
            ),
            0.84,
        ),
    ]
    selected = select_diverse(candidates, profile, limit=2)
    assert [item.paper.arxiv_id for item in selected] == ["2607.20501", "2607.20503"]


def test_direct_adjacent_wildcard_targets_and_weak_rejection(paper_factory) -> None:  # type: ignore[no-untyped-def]
    base = load_settings().profile
    profile = base.model_copy(
        update={
            "recommendation_mix": RecommendationMix(direct=2, adjacent=1, wildcard=1),
            "selection": SelectionConfig(
                mmr_lambda=1.0,
                direct_min_score=0.7,
                adjacent_min_score=0.4,
                wildcard_min_score=0.2,
            ),
        }
    )
    candidates = [
        ranked(paper_factory(arxiv_id="2607.21001", title="Spin ice"), 0.90),
        ranked(paper_factory(arxiv_id="2607.21002", title="Neutron diffuse scattering"), 0.80),
        ranked(paper_factory(arxiv_id="2607.21003", title="Linear spin-wave theory"), 0.55),
        ranked(paper_factory(arxiv_id="2607.21004", title="Spin Seebeck effect"), 0.25),
        ranked(paper_factory(arxiv_id="2607.21005", title="Unrelated weak paper"), 0.10),
    ]
    selected = select_diverse(candidates, profile, limit=5)
    assert [item.recommendation_type for item in selected] == [
        RecommendationType.DIRECT,
        RecommendationType.DIRECT,
        RecommendationType.ADJACENT,
        RecommendationType.WILDCARD,
    ]
    assert all(item.paper.arxiv_id != "2607.21005" for item in selected)


def test_mmr_is_deterministic_under_input_reordering(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile
    candidates = [
        ranked(paper_factory(arxiv_id=f"2607.2200{index}", title=title), score)
        for index, (title, score) in enumerate(
            [
                ("Spin ice dynamics", 0.8),
                ("Neutron diffuse scattering", 0.7),
                ("Linear spin-wave theory", 0.6),
            ],
            start=1,
        )
    ]
    first = select_diverse(candidates, profile, limit=3)
    second = select_diverse(list(reversed(candidates)), profile, limit=3)
    assert [item.paper.arxiv_id for item in first] == [item.paper.arxiv_id for item in second]


def test_selects_ten_when_ten_viable_diverse_candidates_exist(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile
    topics = [
        "Spin ice magnetic monopole dynamics",
        "Neutron diffuse scattering structure factor",
        "Linear spin-wave theory exchange fitting",
        "Frustrated spinel magnetic disorder",
    ]
    candidates = [
        ranked(
            paper_factory(arxiv_id=f"2607.230{index:02d}", title=topics[index % 4]),
            0.90 - (index / 100),
        )
        for index in range(12)
    ]
    selected = select_diverse(candidates, profile, limit=10)
    assert len(selected) == 10
    assert (
        max(
            sum(item.topic_key == topic for item in selected)
            for topic in {item.topic_key for item in selected}
        )
        <= profile.max_papers_per_topic
    )


def test_unmatched_papers_share_one_topic_bucket(paper_factory) -> None:  # type: ignore[no-untyped-def]
    """The diversity cap must not shelter papers that match no configured topic.

    The old fallback gave each unmatched paper a private ``category:{primary}``
    bucket, so max_papers_per_topic protected it from being crowded out by
    better ones — the diversity mechanism was amplifying the filtering bug.
    """
    profile = load_settings().profile
    first = ranked(
        paper_factory(
            arxiv_id="2607.30001",
            title="An unrelated result",
            abstract="Nothing configured appears here.",
            primary_category="cond-mat.dis-nn",
            categories=["cond-mat.dis-nn"],
        ),
        0.5,
    )
    second = ranked(
        paper_factory(
            arxiv_id="2607.30002",
            title="Another unrelated result",
            abstract="Also nothing configured.",
            primary_category="cond-mat.soft",
            categories=["cond-mat.soft"],
        ),
        0.5,
    )
    assert topic_key(first, profile) == topic_key(second, profile) == UNMATCHED_TOPIC


def test_unmatched_papers_compete_for_a_single_quota_slot(paper_factory) -> None:  # type: ignore[no-untyped-def]
    profile = load_settings().profile.model_copy(update={"max_papers_per_topic": 1})
    candidates = [
        ranked(
            paper_factory(
                arxiv_id=f"2607.3010{index}",
                title=f"Unrelated result {index}",
                abstract="Nothing configured appears here.",
                primary_category=f"cond-mat.other{index}",
                categories=[f"cond-mat.other{index}"],
            ),
            0.5 - (index * 0.01),
        )
        for index in range(4)
    ]
    selected = select_diverse(candidates, profile, limit=10)
    assert len(selected) == 1
