"""Optional live API tests, excluded from the default test run.

Run with ``pytest -m live``. These make real, paced requests to arXiv and
assert on properties that survive the corpus growing, not on fixed counts.
"""

from datetime import UTC, datetime, timedelta

import pytest

from arxiv_digest.arxiv_client import ArxivClient, build_category_guard
from arxiv_digest.config import ArxivConfig, DomainConfig, QueryConfig
from arxiv_digest.models import DateWindow


def live_domain() -> DomainConfig:
    return DomainConfig(
        include_categories=["cond-mat.*"],
        exclude_categories=["hep-ph", "astro-ph.*"],
    )


@pytest.mark.live
def test_live_arxiv_api_returns_a_valid_feed() -> None:
    now = datetime.now(UTC)
    window = DateWindow(start=now - timedelta(days=30), end=now)
    with ArxivClient(ArxivConfig(page_size=1)) as client:
        result = client.fetch(
            [QueryConfig(name="smoke", terms=["condensed matter"])],
            window,
            1,
            live_domain(),
        )
    assert result.query_results[0].records_received >= 0


@pytest.mark.live
def test_category_guard_removes_most_off_domain_monopole_results() -> None:
    """The measurement Phase 1 rests on: the positive guard is the big win.

    Measured 2026-08: 1,830 results for the bare term, 448 with the guard —
    a 76% reduction before anything is scored.
    """
    guard = build_category_guard(live_domain())
    with ArxivClient(ArxivConfig(page_size=1)) as client:
        baseline = client.count('all:"magnetic monopole"')
        guarded = client.count(f'all:"magnetic monopole" AND ({guard})')
        unambiguous = client.count('all:"spin ice"')
        unambiguous_guarded = client.count(f'all:"spin ice" AND ({guard})')

    assert baseline > 0
    assert guarded < baseline * 0.5, (
        f"expected the cond-mat guard to remove most monopole results; "
        f"{guarded} of {baseline} survived"
    )
    # Ambiguity is a per-term property, not a global one: spin ice barely moves.
    assert unambiguous_guarded > unambiguous * 0.9


@pytest.mark.live
def test_chained_andnot_does_not_compose_but_grouped_negation_does() -> None:
    """Record the arXiv query-parser behaviour that keeps ANDNOT out of the builders.

    Chained ``ANDNOT`` clauses do not associate as written — measured 2026-08,
    adding a second one *increased* the result count from 1,372 to 1,424. Only
    the grouped form behaves monotonically. The builders emit no negation at all
    (see test_arxiv_client.test_queries_never_emit_andnot); this test exists so
    the finding stays measured rather than remembered.
    """
    with ArxivClient(ArxivConfig(page_size=1)) as client:
        baseline = client.count('all:"magnetic monopole"')
        single = client.count('all:"magnetic monopole" ANDNOT cat:hep-ph')
        chained = client.count('all:"magnetic monopole" ANDNOT cat:hep-ph ANDNOT cat:astro-ph.CO')
        grouped = client.count('all:"magnetic monopole" ANDNOT (cat:hep-ph OR cat:astro-ph.CO)')

    # Grouping is the form that behaves: each added exclusion can only narrow.
    assert grouped <= single <= baseline
    if chained > single:
        pytest.skip(
            f"chained ANDNOT still does not compose: {single} -> {chained} "
            f"(grouped: {grouped}). The builders emit no ANDNOT, so nothing depends on it."
        )
