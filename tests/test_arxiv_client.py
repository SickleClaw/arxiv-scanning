"""Offline tests for arXiv queries, Atom parsing, retries, caching, and pagination."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx

from arxiv_digest import __version__
from arxiv_digest.arxiv_client import (
    ArxivClient,
    build_category_guard,
    build_search_query,
    build_term_query,
    parse_atom_feed,
)
from arxiv_digest.config import ArxivConfig, DomainConfig, QueryConfig
from arxiv_digest.exceptions import ArxivFeedError, ArxivHTTPError
from arxiv_digest.models import DateWindow


class FakeClock:
    """Deterministic monotonic clock whose sleep advances time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def window() -> DateWindow:
    return DateWindow(
        start=datetime(2026, 7, 24, tzinfo=UTC),
        end=datetime(2026, 7, 28, tzinfo=UTC),
    )


def domain() -> DomainConfig:
    return DomainConfig(
        include_categories=["cond-mat.*", "physics.ins-det"],
        soft_categories=["cs.LG"],
        exclude_categories=["hep-ph", "astro-ph.*"],
    )


def test_query_groups_quotes_and_documented_submission_date_filter() -> None:
    query = QueryConfig(name="quoted", terms=["spin ice", 'moment "sum" rule'])
    result = build_search_query(query, window(), domain())
    assert result.startswith('(all:"spin ice" OR all:"moment \\"sum\\" rule") AND ')
    assert "submittedDate:[202607240000 TO 202607272359]" in result
    assert "lastUpdatedDate" not in result
    assert build_term_query(query, domain()).startswith('(all:"spin ice" OR ')


def test_category_guard_admits_in_field_and_adjacent_categories_verbatim() -> None:
    guard = build_category_guard(domain())
    assert guard == "cat:cond-mat.* OR cat:physics.ins-det OR cat:cs.LG"
    # arXiv writes subject classes in mixed case; the guard must not fold them.
    assert "cat:cs.lg" not in guard


def test_queries_carry_exactly_one_parenthesised_category_group() -> None:
    query = QueryConfig(name="guarded", terms=["spin ice"])
    for result in (
        build_search_query(query, window(), domain()),
        build_term_query(query, domain()),
    ):
        assert f"({build_category_guard(domain())})" in result
        assert result.count("cat:") == 3
        assert result.count("(") == result.count(")")


def test_queries_never_emit_andnot() -> None:
    """Negation at the source could only remove cross-listed in-field work.

    The positive guard already requires an in-field or adjacent category, so any
    paper an ``ANDNOT`` clause would additionally drop carries both an in-field
    and an excluded category — exactly the work the domain gate protects. Such a
    rejection would also be invisible, since nothing is retrieved to log it.
    Exclusion belongs in domain_filter, where inclusion still outranks it.

    Grouping is the second reason to keep negation out of query strings:
    measured against the live API, chained ``ANDNOT`` clauses do not compose.
    ``all:"magnetic monopole" ANDNOT cat:hep-ph`` returned 1372 results and
    adding ``ANDNOT cat:astro-ph.CO`` returned 1424 — more, not fewer. See
    test_live_arxiv.py for the opt-in measurement.
    """
    query = QueryConfig(name="guarded", terms=["magnetic monopole"])
    for result in (
        build_search_query(query, window(), domain()),
        build_term_query(query, domain()),
    ):
        assert "ANDNOT" not in result
        assert "hep-ph" not in result
        assert "astro-ph" not in result


def test_atom_fixture_parses_normalized_metadata(fixture_dir: Path) -> None:
    papers, total = parse_atom_feed((fixture_dir / "arxiv_page_1.xml").read_bytes())
    assert total == 2
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2607.12345"
    assert paper.version == 1
    assert paper.title == "Nonequilibrium dynamics in Ho₂Ti₂O₇ spin ice"
    assert paper.authors == ["Ada Curie", "Niels Raman"]
    assert paper.doi == "10.0000/example.1"
    assert paper.categories == ["cond-mat.str-el", "cond-mat.stat-mech"]


@pytest.mark.parametrize("content", [b"", b"<not-atom>"])
def test_malformed_atom_feed_has_clear_error(content: bytes) -> None:
    with pytest.raises(ArxivFeedError):
        parse_atom_feed(content)


@respx.mock
def test_client_url_encodes_and_paginates_with_one_connection(fixture_dir: Path) -> None:
    route = respx.get("https://export.arxiv.org/api/query").mock(
        side_effect=[
            httpx.Response(200, content=(fixture_dir / "arxiv_page_1.xml").read_bytes()),
            httpx.Response(200, content=(fixture_dir / "arxiv_page_2.xml").read_bytes()),
            httpx.Response(200, content=(fixture_dir / "arxiv_empty.xml").read_bytes()),
        ]
    )
    clock = FakeClock()
    config = ArxivConfig(page_size=1, retry_count=0)
    query = QueryConfig(name="spin ice", terms=["spin ice"])
    with httpx.Client(headers={"User-Agent": "test"}) as http_client:
        client = ArxivClient(
            config,
            client=http_client,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        result = client.fetch([query], window(), 2, domain())
    assert len(result.papers) == 2
    assert result.papers[0].arxiv_id == "2607.12345"
    assert result.papers[1].arxiv_id == "cond-mat/0501234"
    assert route.call_count == 3
    first_url = str(route.calls[0].request.url)
    query_string = parse_qs(urlparse(first_url).query)
    assert query_string["start"] == ["0"]
    assert query_string["max_results"] == ["1"]
    assert query_string["search_query"][0].startswith('(all:"spin ice")')
    assert "%22spin+ice%22" in first_url
    assert clock.sleeps == [3.0, 3.0]


@respx.mock
def test_transient_status_retries_and_response_cache(fixture_dir: Path) -> None:
    route = respx.get("https://export.arxiv.org/api/query").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, content=(fixture_dir / "arxiv_page_1.xml").read_bytes()),
            httpx.Response(200, content=(fixture_dir / "arxiv_empty.xml").read_bytes()),
        ]
    )
    clock = FakeClock()
    config = ArxivConfig(page_size=1, retry_count=1, retry_backoff_seconds=2.0)
    query = QueryConfig(name="spin ice", terms=["spin ice"])
    with httpx.Client() as http_client:
        client = ArxivClient(
            config,
            client=http_client,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        first = client.fetch([query], window(), 1, domain())
        second = client.fetch([query], window(), 1, domain())
    assert len(first.papers) == len(second.papers) == 1
    assert route.call_count == 3
    assert clock.sleeps == [2.0, 1.0, 3.0]


@respx.mock
def test_update_sorted_pass_retrieves_revised_old_submission(fixture_dir: Path) -> None:
    route = respx.get("https://export.arxiv.org/api/query").mock(
        side_effect=[
            httpx.Response(200, content=(fixture_dir / "arxiv_empty.xml").read_bytes()),
            httpx.Response(200, content=(fixture_dir / "arxiv_page_2.xml").read_bytes()),
        ]
    )
    clock = FakeClock()
    with httpx.Client() as http_client:
        client = ArxivClient(
            ArxivConfig(page_size=1, retry_count=0),
            client=http_client,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        result = client.fetch(
            [QueryConfig(name="revisions", terms=["neutron scattering"])],
            window(),
            1,
            domain(),
        )
    assert [paper.arxiv_id for paper in result.papers] == ["cond-mat/0501234"]
    assert route.calls[1].request.url.params["sortBy"] == "lastUpdatedDate"
    assert "submittedDate" not in route.calls[1].request.url.params["search_query"]


@respx.mock
def test_nonretryable_http_error_is_actionable() -> None:
    respx.get("https://export.arxiv.org/api/query").mock(return_value=httpx.Response(400))
    with httpx.Client() as http_client:
        client = ArxivClient(ArxivConfig(retry_count=0), client=http_client)
        with pytest.raises(ArxivHTTPError, match="non-retryable HTTP 400"):
            client.fetch([QueryConfig(name="x", terms=["spin ice"])], window(), 1, domain())


@respx.mock
def test_internal_client_sends_descriptive_user_agent(fixture_dir: Path) -> None:
    route = respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(
            200,
            content=(fixture_dir / "arxiv_empty.xml").read_bytes(),
        )
    )
    config = ArxivConfig(contact_email="physics@example.org", retry_count=0)
    clock = FakeClock()
    with ArxivClient(config, monotonic=clock.monotonic, sleep=clock.sleep) as client:
        client.fetch([QueryConfig(name="x", terms=["spin ice"])], window(), 1, domain())
    assert route.calls[0].request.headers["User-Agent"] == (
        f"arxiv-digest/{__version__} (physics@example.org)"
    )
