"""Tests for the arXiv category matcher and the group domain gate."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arxiv_digest.config import (
    DomainClass,
    DomainConfig,
    category_matches,
    load_settings,
    normalize_category,
)


def domain() -> DomainConfig:
    return DomainConfig(
        include_categories=["cond-mat.*", "physics.ins-det"],
        soft_categories=["quant-ph", "cs.LG"],
        exclude_categories=["astro-ph.*", "hep-ph", "gr-qc"],
    )


@pytest.mark.parametrize(
    ("pattern", "category", "expected"),
    [
        # Exact archives and subjects.
        ("hep-ph", "hep-ph", True),
        ("hep-ph", "hep-th", False),
        ("cond-mat.str-el", "cond-mat.str-el", True),
        ("cond-mat.str-el", "cond-mat.mtrl-sci", False),
        # Wildcards over an archive.
        ("cond-mat.*", "cond-mat.str-el", True),
        ("cond-mat.*", "cond-mat.quant-gas", True),
        ("astro-ph.*", "astro-ph.CO", True),
        ("cond-mat.*", "hep-ph", False),
        # The bare legacy archive that old cross-lists still carry.
        ("cond-mat.*", "cond-mat", True),
        ("astro-ph.*", "astro-ph", True),
        # A wildcard must not match a different archive sharing a prefix.
        ("cond-mat.*", "cond-matter.foo", False),
        ("hep-ph", "hep-ph.something", False),
        # Matching is case-insensitive and whitespace-tolerant.
        ("COND-MAT.*", "cond-mat.str-el", True),
        ("cond-mat.*", " Cond-Mat.Str-El ", True),
    ],
)
def test_category_matches_handles_every_real_arxiv_shape(
    pattern: str, category: str, expected: bool
) -> None:
    assert category_matches(pattern, category) is expected


def test_normalize_category_casefolds_and_strips() -> None:
    assert normalize_category("  Cond-Mat.Str-El \n") == "cond-mat.str-el"


def test_in_field_papers_classify_as_include() -> None:
    assert domain().classify(["cond-mat.str-el"]) is DomainClass.INCLUDE
    assert domain().classify(["physics.ins-det"]) is DomainClass.INCLUDE
    assert domain().classify(["cond-mat"]) is DomainClass.INCLUDE


def test_cross_listing_into_the_field_beats_an_excluded_primary() -> None:
    """The safety catch: inclusion outranks exclusion, always.

    This is what preserves emergent magnetic monopoles in spin ice while still
    rejecting the cosmological kind.
    """
    assert domain().classify(["hep-ph", "cond-mat.str-el"]) is DomainClass.INCLUDE
    assert domain().classify(["astro-ph.CO", "cond-mat.stat-mech"]) is DomainClass.INCLUDE
    assert domain().classify(["cs.LG", "cond-mat.mtrl-sci"]) is DomainClass.INCLUDE


def test_excluded_papers_without_an_in_field_cross_list_are_excluded() -> None:
    assert domain().classify(["hep-ph", "astro-ph.CO", "gr-qc"]) is DomainClass.EXCLUDE
    assert domain().classify(["astro-ph.HE"]) is DomainClass.EXCLUDE


def test_soft_categories_rank_below_include_and_above_exclude() -> None:
    assert domain().classify(["quant-ph"]) is DomainClass.SOFT
    assert domain().classify(["quant-ph", "hep-th"]) is DomainClass.SOFT
    assert domain().classify(["quant-ph", "cond-mat.mes-hall"]) is DomainClass.INCLUDE


def test_unconfigured_categories_classify_as_unknown() -> None:
    assert domain().classify(["physics.plasm-ph"]) is DomainClass.UNKNOWN
    assert domain().classify(["physics.flu-dyn", "physics.data-an"]) is DomainClass.UNKNOWN


def test_matched_categories_are_reported_for_explainability() -> None:
    matched = domain().included(["hep-ph", "cond-mat.str-el", "cond-mat.stat-mech"])
    assert matched == ["cond-mat.str-el", "cond-mat.stat-mech"]
    assert domain().excluded(["hep-ph", "cond-mat.str-el"]) == ["hep-ph"]


def test_malformed_patterns_are_rejected_at_load_time() -> None:
    with pytest.raises(ValidationError):
        DomainConfig(include_categories=["cond mat"])
    with pytest.raises(ValidationError):
        DomainConfig(include_categories=["cond-mat.*.*"])
    with pytest.raises(ValidationError):
        DomainConfig(include_categories=["  "])


def test_duplicate_and_overlapping_patterns_are_rejected() -> None:
    with pytest.raises(ValidationError):
        DomainConfig(include_categories=["cond-mat.*", "cond-mat.*"])
    with pytest.raises(ValidationError):
        DomainConfig(include_categories=["quant-ph"], soft_categories=["quant-ph"])


def test_committed_group_profile_gates_the_reported_failure_cases() -> None:
    """The committed profile must classify the papers the audit named."""
    configured = load_settings().group.domain
    # The two papers Phase 1 exists to remove.
    assert configured.classify(["hep-ph", "astro-ph.CO"]) is DomainClass.EXCLUDE
    assert configured.classify(["physics.plasm-ph", "physics.atom-ph"]) is DomainClass.UNKNOWN
    # The work it must keep.
    assert configured.classify(["cond-mat.str-el"]) is DomainClass.INCLUDE
    assert configured.classify(["cs.LG", "cond-mat.mtrl-sci"]) is DomainClass.INCLUDE
    assert configured.classify(["quant-ph", "cond-mat.quant-gas", "hep-th"]) is DomainClass.INCLUDE
