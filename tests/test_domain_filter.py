"""Tests for hard exclusion rules and their unconditional cross-list safety catch."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from arxiv_digest.config import DomainConfig, load_settings
from arxiv_digest.domain_filter import (
    DisambiguationConfig,
    HardExclusionRule,
    RuleCondition,
    evaluate_hard_exclusions,
    load_disambiguation,
)
from arxiv_digest.models import CandidateSnapshot, Paper

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "disambiguation.yaml"
FIXTURES = Path(__file__).parent / "fixtures"


def domain() -> DomainConfig:
    return DomainConfig(
        include_categories=["cond-mat.*", "physics.ins-det"],
        soft_categories=["quant-ph", "cs.LG"],
        exclude_categories=["astro-ph.*", "hep-ph", "hep-th", "gr-qc"],
    )


def paper(title: str, abstract: str, categories: list[str]) -> Paper:
    return Paper(
        arxiv_id="2607.00001",
        version=1,
        title=title,
        authors=["A Researcher"],
        abstract=abstract,
        primary_category=categories[0],
        categories=categories,
        published_at=datetime(2026, 7, 24, tzinfo=UTC),
        updated_at=datetime(2026, 7, 24, tzinfo=UTC),
        abstract_url="https://arxiv.org/abs/2607.00001v1",
        pdf_url="https://arxiv.org/pdf/2607.00001v1",
    )


def rules() -> list[HardExclusionRule]:
    return load_disambiguation(CONFIG_PATH).hard_exclusions


# The worked examples from the design report, plus the cases they bracket.
WORKED_EXAMPLES = [
    (
        "emergent monopoles are condensed matter",
        "Magnetic monopole excitations in the spin ice Ho2Ti2O7 probed by neutron scattering",
        "Emergent magnetic monopoles propagate through the pyrochlore lattice.",
        ["cond-mat.str-el"],
        None,
    ),
    (
        "cosmological monopoles are not",
        "Primordial magnetic monopoles and the cosmological constant problem",
        "Monopoles produced during inflation in the early universe are considered.",
        ["hep-ph", "astro-ph.CO"],
        "cosmological-monopole",
    ),
    (
        "ML for materials survives on its cond-mat cross-list",
        "Machine learning interatomic potentials for magnetic materials",
        "A neural network potential reproduces magnetic exchange in solids.",
        ["cs.LG", "cond-mat.mtrl-sci"],
        None,
    ),
    (
        "collider physics is rejected",
        "Search for magnetic monopoles at the LHC with the ATLAS detector",
        "We report a search using collider data at 13 TeV.",
        ["hep-ex"],
        "collider-physics",
    ),
    (
        "pure astrophysics is rejected on its primary category",
        "Galactic magnetic field structure from polarization maps",
        "We map the field of the galaxy using dust polarization.",
        ["astro-ph.GA"],
        "pure-astro",
    ),
]


@pytest.mark.parametrize(
    ("label", "title", "abstract", "categories", "expected_rule"),
    WORKED_EXAMPLES,
    ids=[case[0] for case in WORKED_EXAMPLES],
)
def test_worked_examples(
    label: str, title: str, abstract: str, categories: list[str], expected_rule: str | None
) -> None:
    verdict = evaluate_hard_exclusions(paper(title, abstract, categories), rules(), domain())
    if expected_rule is None:
        assert verdict is None, f"{label}: unexpectedly rejected by {verdict}"
    else:
        assert verdict is not None, f"{label}: expected rejection by {expected_rule}"
        assert verdict.rule_id == expected_rule


def test_every_rejection_carries_a_rule_id_and_a_reason() -> None:
    verdict = evaluate_hard_exclusions(
        paper(
            "Primordial monopoles in the early universe",
            "Cosmological defects from inflation.",
            ["hep-ph"],
        ),
        rules(),
        domain(),
    )
    assert verdict is not None
    assert verdict.rule_id == "cosmological-monopole"
    assert verdict.reason.strip()
    assert verdict.reason != verdict.rule_id


# The property that matters most: no rule, present or future, can reject a paper
# cross-listed into the group's own field. The guard is in the engine, so this
# holds for any rule file, not just the committed one.
OFF_DOMAIN_TEXT = [
    "Primordial magnetic monopoles from inflation in the early universe",
    "Search for monopoles at the LHC: collider limits, luminosity, and TeV scales",
    "Cosmological domain walls, relic abundance, and dark matter",
    "GUT monopoles, cosmic ray flux, and the Parker bound",
]
IN_FIELD_CATEGORIES = [
    ["cond-mat.str-el"],
    ["cond-mat"],
    ["cond-mat.stat-mech", "hep-th"],
    ["physics.ins-det", "astro-ph.CO"],
]


@pytest.mark.parametrize(
    ("text", "categories"), list(itertools.product(OFF_DOMAIN_TEXT, IN_FIELD_CATEGORIES))
)
def test_no_rule_can_fire_on_an_in_field_cross_list(text: str, categories: list[str]) -> None:
    for rule in rules():
        verdict = evaluate_hard_exclusions(
            paper(text, text, [*categories, "hep-ph", "astro-ph.CO"]), [rule], domain()
        )
        assert verdict is None, f"rule {rule.id} fired on an in-field cross-list"


def test_the_guard_holds_for_a_rule_that_tries_to_match_everything() -> None:
    """Even a maximally broad rule cannot reach in-field work."""
    catch_all = HardExclusionRule(
        id="catch-all",
        reason="a rule that matches any paper at all",
        when_all=[RuleCondition(field="categories", none_of=["definitely-not-a-category"])],
    )
    in_field = paper("Anything at all", "Any words whatsoever.", ["cond-mat.mtrl-sci"])
    assert evaluate_hard_exclusions(in_field, [catch_all], domain()) is None
    off_domain = paper("Anything at all", "Any words whatsoever.", ["hep-ph"])
    assert evaluate_hard_exclusions(off_domain, [catch_all], domain()) is not None


def test_soft_categories_do_not_get_the_cross_list_protection() -> None:
    """Adjacent categories must earn their place; only in-field work is shielded."""
    quant_ph = paper(
        "Primordial monopoles in the early universe",
        "Cosmological inflation produces monopoles.",
        ["quant-ph"],
    )
    assert evaluate_hard_exclusions(quant_ph, rules(), domain()) is not None


def test_text_matching_is_whole_term_not_substring() -> None:
    """A rule term must not fire on a longer word that merely contains it."""
    survivor = paper(
        "Gevrey regularity of a lattice model",
        "We study Gevrey classes, with no high-energy content whatsoever.",
        ["math-ph"],
    )
    assert evaluate_hard_exclusions(survivor, rules(), domain()) is None


def test_conditions_require_exactly_one_operator() -> None:
    with pytest.raises(ValidationError):
        RuleCondition(field="text", any_of=["a"], none_of=["b"])
    with pytest.raises(ValidationError):
        RuleCondition(field="text")
    with pytest.raises(ValidationError):
        RuleCondition(field="text", any_of=["  "])


def test_duplicate_rule_ids_are_rejected() -> None:
    rule = {"id": "duplicated", "reason": "r", "when_all": [{"field": "text", "any_of": ["x"]}]}
    with pytest.raises(ValidationError):
        DisambiguationConfig.model_validate({"hard_exclusions": [rule, rule]})


def test_committed_rules_load_and_stay_narrow() -> None:
    configured = load_disambiguation(CONFIG_PATH)
    assert [rule.id for rule in configured.hard_exclusions] == [
        "cosmological-monopole",
        "collider-physics",
        "pure-astro",
    ]
    # Hard rules cause silent recall loss when they multiply. Keep the list short.
    assert len(configured.hard_exclusions) <= 5


def test_committed_rules_reject_the_reported_hep_ph_paper() -> None:
    """The paper the audit named must be rejected, by name."""
    snapshot = CandidateSnapshot.model_validate_json(
        (FIXTURES / "candidates_2026-07-21.json").read_text(encoding="utf-8")
    )
    configured = load_disambiguation(CONFIG_PATH)
    live_domain = load_settings().group.domain
    verdicts = {
        item.arxiv_id: evaluate_hard_exclusions(item, configured.hard_exclusions, live_domain)
        for item in snapshot.papers
    }
    dark_monopoles = verdicts["2607.20843"]
    assert dark_monopoles is not None
    assert dark_monopoles.rule_id == "cosmological-monopole"
    # No in-field paper in the real window is touched by a hard rule.
    for item in snapshot.papers:
        if live_domain.included(item.categories):
            assert verdicts[item.arxiv_id] is None, item.arxiv_id
