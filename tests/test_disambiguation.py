"""Tests for measured term ambiguity and its cached table."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from arxiv_digest.config import DomainConfig, load_settings
from arxiv_digest.disambiguation import (
    GUARD_ABOVE,
    AmbiguityRecord,
    AmbiguityTable,
    domain_signature,
    in_field_query,
    load_table,
    measure_terms,
    unguarded_risky_terms,
)
from arxiv_digest.domain_filter import load_disambiguation

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "data" / "term_ambiguity.json"
CONFIG_PATH = ROOT / "config" / "disambiguation.yaml"


class ForbiddenClient:
    """A client that fails the test if anything tries to reach the network."""

    def count(self, search_query: str) -> int:
        raise AssertionError(f"unexpected network request: {search_query}")


def record(term: str, total: int, in_field: int) -> AmbiguityRecord:
    return AmbiguityRecord(
        term=term,
        total_results=total,
        in_field_results=in_field,
        measured_at=datetime(2026, 8, 27, tzinfo=UTC),
    )


def test_ambiguity_is_the_out_of_field_share() -> None:
    assert record("spin ice", 931, 921).ambiguity == pytest.approx(0.0107, abs=1e-3)
    assert record("magnetic monopole", 1830, 448).ambiguity == pytest.approx(0.755, abs=1e-3)
    # A term nobody has written about is not evidence of anything.
    assert record("nothing at all", 0, 0).ambiguity == 0.0


def test_committed_table_reproduces_the_two_reported_measurements() -> None:
    """The numbers the design report rests on, from cache and without network."""
    table = load_table(TABLE_PATH)
    assert table is not None
    assert table.records["spin ice"].ambiguity == pytest.approx(0.01, abs=0.01)
    assert table.records["magnetic monopole"].ambiguity == pytest.approx(0.76, abs=0.01)
    assert table.records["spin ice"].total_results == 931
    assert table.records["magnetic monopole"].total_results == 1830


def test_a_fully_cached_measurement_makes_no_requests() -> None:
    """Two requests per term, once, then cached indefinitely."""
    domain = load_settings().group.domain
    cached = AmbiguityTable(
        domain_signature=domain_signature(domain),
        records={"spin ice": record("spin ice", 931, 921)},
    )
    table = measure_terms(
        ForbiddenClient(),  # type: ignore[arg-type]
        ["spin ice"],
        domain,
        cached=cached,
    )
    assert table.records["spin ice"].total_results == 931


def test_a_changed_domain_invalidates_the_cache() -> None:
    """These numbers are only meaningful against the field they were measured for."""
    narrow = DomainConfig(include_categories=["cond-mat.*"])
    wide = DomainConfig(include_categories=["cond-mat.*", "physics.ins-det"])
    cached = AmbiguityTable(
        domain_signature=domain_signature(narrow),
        records={"spin ice": record("spin ice", 931, 921)},
    )
    assert cached.stale_for(wide)
    assert not cached.stale_for(narrow)
    with pytest.raises(AssertionError, match="unexpected network request"):
        measure_terms(ForbiddenClient(), ["spin ice"], wide, cached=cached)  # type: ignore[arg-type]


def test_refresh_ignores_the_cache() -> None:
    domain = load_settings().group.domain
    cached = AmbiguityTable(
        domain_signature=domain_signature(domain),
        records={"spin ice": record("spin ice", 931, 921)},
    )
    with pytest.raises(AssertionError, match="unexpected network request"):
        measure_terms(
            ForbiddenClient(),  # type: ignore[arg-type]
            ["spin ice"],
            domain,
            cached=cached,
            refresh=True,
        )


def test_in_field_query_counts_only_in_field_categories() -> None:
    """Adjacent categories are broad; counting them would flatter every term."""
    domain = DomainConfig(
        include_categories=["cond-mat.*"],
        soft_categories=["cs.LG"],
    )
    query = in_field_query("spin ice", domain)
    assert "cat:cond-mat.*" in query
    assert "cs.LG" not in query


def test_a_guard_covers_the_terms_that_contain_it() -> None:
    """The guard on 'monopole' covers the profile's 'magnetic monopole'."""
    table = AmbiguityTable(
        domain_signature="x",
        records={"magnetic monopole": record("magnetic monopole", 1830, 448)},
    )
    assert unguarded_risky_terms(table, ["monopole"]) == []
    assert len(unguarded_risky_terms(table, ["frustration"])) == 1


def test_unambiguous_terms_are_never_flagged() -> None:
    table = AmbiguityTable(domain_signature="x", records={"spin ice": record("spin ice", 931, 921)})
    assert unguarded_risky_terms(table, []) == []


def test_every_measured_risky_term_in_this_profile_has_a_guard() -> None:
    """Self-policing config: adding a risky term without a guard fails here.

    'inverse problems' (4% in field), 'cluster algorithm' (8%), and
    'loop algorithm' (37%) were all found this way rather than anticipated.
    """
    table = load_table(TABLE_PATH)
    assert table is not None
    guards = load_disambiguation(CONFIG_PATH).ambiguous_terms
    unguarded = unguarded_risky_terms(table, list(guards))
    assert unguarded == [], [item.describe() for item in unguarded]


def test_the_guard_threshold_is_where_the_design_put_it() -> None:
    assert GUARD_ABOVE == 0.40
    assert record("magnetic monopole", 1830, 448).needs_context_guard
    assert not record("spin ice", 931, 921).needs_context_guard
