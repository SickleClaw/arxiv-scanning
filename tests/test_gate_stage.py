"""Tests for the explicit gate stage that runs before any scoring."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from arxiv_digest.config import load_settings
from arxiv_digest.domain_filter import gate_papers, load_disambiguation
from arxiv_digest.models import CandidateSnapshot, GateStage, Paper

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "disambiguation.yaml"


def paper(arxiv_id: str, title: str, abstract: str, categories: list[str]) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        version=1,
        title=title,
        authors=["A Researcher"],
        abstract=abstract,
        primary_category=categories[0],
        categories=categories,
        published_at=datetime(2026, 7, 24, tzinfo=UTC),
        updated_at=datetime(2026, 7, 24, tzinfo=UTC),
        abstract_url=f"https://arxiv.org/abs/{arxiv_id}v1",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}v1",
    )


def gate(papers: list[Paper], negative_terms: list[str] | None = None):  # type: ignore[no-untyped-def]
    settings = load_settings()
    return gate_papers(
        papers,
        domain=settings.group.domain,
        disambiguation=load_disambiguation(CONFIG_PATH),
        negative_terms=negative_terms if negative_terms is not None else [],
    )


def test_out_of_field_categories_are_rejected_with_a_reason() -> None:
    result = gate([paper("2607.00001", "A collider result", "Nothing here.", ["hep-ph"])])
    assert result.kept == []
    assert len(result.rejections) == 1
    rejection = result.rejections[0]
    assert rejection.stage is GateStage.CATEGORY
    assert "out of field" in rejection.reason
    assert rejection.title == "A collider result"


def test_categories_that_are_neither_in_field_nor_adjacent_are_rejected() -> None:
    result = gate([paper("2607.00002", "Fluid turbulence", "Nothing here.", ["physics.flu-dyn"])])
    assert result.kept == []
    assert result.rejections[0].stage is GateStage.CATEGORY
    assert "neither in-field nor adjacent" in result.rejections[0].reason


def test_in_field_and_adjacent_papers_survive_the_category_gate() -> None:
    papers = [
        paper("2607.00003", "Spin ice dynamics", "Monopoles in pyrochlore.", ["cond-mat.str-el"]),
        paper("2607.00004", "ML potentials", "A neural potential.", ["cs.LG"]),
    ]
    result = gate(papers)
    assert [item.arxiv_id for item in result.kept] == ["2607.00003", "2607.00004"]
    assert result.rejections == []


def test_hard_rules_run_after_the_category_gate_and_name_their_rule() -> None:
    result = gate(
        [
            paper(
                "2607.00005",
                "Primordial monopoles",
                "Cosmological inflation in the early universe produces monopoles.",
                ["quant-ph"],
            )
        ]
    )
    assert result.kept == []
    assert result.rejections[0].stage is GateStage.HARD_RULE
    assert result.rejections[0].rule_id == "cosmological-monopole"


def test_negative_profile_terms_reject_rather_than_penalize() -> None:
    """Negative terms are a gate now, not a score penalty a good score could outvote."""
    papers = [
        paper(
            "2607.00006",
            "Correlations in a social network",
            "We study a social network of agents.",
            ["cond-mat.stat-mech"],
        )
    ]
    result = gate(papers, negative_terms=["social network"])
    assert result.kept == []
    assert result.rejections[0].stage is GateStage.NEGATIVE_TERM
    assert "social network" in result.rejections[0].reason
    # Without the configured term the same paper is kept.
    assert len(gate(papers, negative_terms=[]).kept) == 1


def test_kept_papers_carry_their_unsupported_ambiguous_terms_as_flags() -> None:
    result = gate(
        [
            paper(
                "2607.00007",
                "On the monopole",
                "A monopole is discussed with no further context.",
                ["cond-mat.str-el"],
            )
        ]
    )
    assert len(result.kept) == 1
    assert [(flag.arxiv_id, flag.term) for flag in result.context_flags] == [
        ("2607.00007", "monopole")
    ]


def test_gating_the_real_window_removes_the_off_domain_papers_only() -> None:
    """The audit's window, gated: four off-domain papers out, every cond-mat one in."""
    snapshot = CandidateSnapshot.model_validate_json(
        (FIXTURES / "candidates_2026-07-21.json").read_text(encoding="utf-8")
    )
    settings = load_settings()
    result = gate_papers(
        snapshot.papers,
        domain=settings.group.domain,
        disambiguation=load_disambiguation(CONFIG_PATH),
        negative_terms=settings.profile.negative_terms,
    )
    rejected_ids = {rejection.arxiv_id for rejection in result.rejections}
    # The hep-ph dark-monopole paper the report named.
    assert "2607.20843" in rejected_ids
    # Everything in-field survives; the gates decide domain, not relevance.
    for item in snapshot.papers:
        if settings.group.domain.included(item.categories):
            assert item.arxiv_id not in rejected_ids, item.arxiv_id
    assert len(result.kept) + len(result.rejections) == len(snapshot.papers)


def test_every_rejection_is_recorded_exactly_once() -> None:
    snapshot = CandidateSnapshot.model_validate_json(
        (FIXTURES / "candidates_2026-07-21.json").read_text(encoding="utf-8")
    )
    settings = load_settings()
    result = gate_papers(
        snapshot.papers,
        domain=settings.group.domain,
        disambiguation=load_disambiguation(CONFIG_PATH),
        negative_terms=settings.profile.negative_terms,
    )
    identifiers = [rejection.arxiv_id for rejection in result.rejections]
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers).isdisjoint({item.arxiv_id for item in result.kept})
