"""Official arXiv API query construction, retrieval, and Atom parsing."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import feedparser
import httpx

from arxiv_digest import __version__
from arxiv_digest.config import ArxivConfig, DomainConfig, QueryConfig
from arxiv_digest.exceptions import ArxivFeedError, ArxivHTTPError
from arxiv_digest.models import DateWindow, Paper, QueryResult
from arxiv_digest.normalization import normalize_whitespace, parse_arxiv_identity

LOGGER = logging.getLogger(__name__)
_TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Raw cross-query papers and transparent per-query counts."""

    papers: list[Paper]
    query_results: list[QueryResult]


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def quote_term(value: str) -> str:
    """Quote and escape one search term for an arXiv ``all:`` clause."""
    escaped = normalize_whitespace(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'all:"{escaped}"'


def build_category_guard(domain: DomainConfig) -> str:
    """Build the positive ``cat:`` guard that keeps retrieval inside the field.

    Both in-field and adjacent categories are admitted here; adjacent ones must
    still earn their place at the local gate. Measured against the live API,
    adding this guard to ``all:"magnetic monopole"`` removes 76% of the result
    set before a single byte is scored.

    No ``ANDNOT`` clause is emitted, deliberately. Under a positive guard every
    paper an exclusion would additionally remove is one carrying *both* an
    in-field and an excluded category — precisely the cross-listed work the
    domain gate exists to protect. At the source such a rejection would also be
    invisible, since nothing is retrieved to log. Exclusion is therefore applied
    locally, in ``domain_filter``, where inclusion still outranks it and every
    rejection is recorded with a reason.
    """
    patterns = [*domain.include_categories, *domain.soft_categories]
    return " OR ".join(f"cat:{pattern}" for pattern in patterns)


def build_search_query(query: QueryConfig, window: DateWindow, domain: DomainConfig) -> str:
    """Build a category-guarded expression bounded by arXiv's submission filter."""
    term_group = " OR ".join(quote_term(term) for term in query.terms)
    inclusive_end = _utc(window.end) - timedelta(minutes=1)
    start = _utc(window.start).strftime("%Y%m%d%H%M")
    end = inclusive_end.strftime("%Y%m%d%H%M")
    return (
        f"({term_group}) AND ({build_category_guard(domain)}) AND submittedDate:[{start} TO {end}]"
    )


def build_term_query(query: QueryConfig, domain: DomainConfig) -> str:
    """Build an unbounded category-guarded query for the update-sorted pass."""
    term_group = " OR ".join(quote_term(term) for term in query.terms)
    return f"({term_group}) AND ({build_category_guard(domain)})"


def _parse_datetime(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArxivFeedError(f"Malformed arXiv {field} timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ArxivFeedError(f"arXiv {field} timestamp lacks a timezone: {value!r}")
    return parsed.astimezone(UTC)


def _required(entry: Mapping[str, Any], key: str) -> Any:
    value = entry.get(key)
    if value is None or value == "":
        raise ArxivFeedError(f"arXiv entry is missing required field {key!r}")
    return value


def _parse_entry(entry: Mapping[str, Any]) -> Paper:
    identifier, version = parse_arxiv_identity(str(_required(entry, "id")))
    author_records = _required(entry, "authors")
    authors = [str(author["name"]) for author in author_records if author.get("name")]
    tags = entry.get("tags", [])
    categories = [str(tag["term"]) for tag in tags if tag.get("term")]
    primary_record = entry.get("arxiv_primary_category", {})
    primary_category = str(primary_record.get("term") or (categories[0] if categories else ""))
    if not primary_category:
        raise ArxivFeedError("arXiv entry is missing a primary category")

    links = entry.get("links", [])
    abstract_url = next(
        (str(link["href"]) for link in links if link.get("rel") == "alternate"),
        f"https://arxiv.org/abs/{identifier}",
    )
    pdf_url = next(
        (
            str(link["href"])
            for link in links
            if link.get("title") == "pdf" or link.get("type") == "application/pdf"
        ),
        f"https://arxiv.org/pdf/{identifier}",
    )
    doi_value = entry.get("arxiv_doi")
    journal_value = entry.get("arxiv_journal_ref")
    return Paper(
        arxiv_id=identifier,
        version=version,
        title=str(_required(entry, "title")),
        authors=authors,
        abstract=str(_required(entry, "summary")),
        primary_category=primary_category,
        categories=categories or [primary_category],
        published_at=_parse_datetime(str(_required(entry, "published")), "published"),
        updated_at=_parse_datetime(str(_required(entry, "updated")), "updated"),
        abstract_url=abstract_url,
        pdf_url=pdf_url,
        doi=str(doi_value) if doi_value else None,
        journal_reference=str(journal_value) if journal_value else None,
    )


def parse_atom_feed(content: bytes) -> tuple[list[Paper], int]:
    """Parse one arXiv Atom page and return papers plus the advertised total count."""
    if not content.strip():
        raise ArxivFeedError("arXiv returned an empty Atom response")
    parsed = feedparser.parse(content)
    if parsed.get("bozo"):
        detail = parsed.get("bozo_exception")
        raise ArxivFeedError(f"Malformed arXiv Atom feed: {detail}")
    feed = parsed.get("feed")
    if not feed or "title" not in feed:
        raise ArxivFeedError("arXiv response is not a recognizable Atom feed")
    entries = parsed.get("entries", [])
    try:
        total = int(feed.get("opensearch_totalresults", len(entries)))
        papers = [_parse_entry(entry) for entry in entries]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArxivFeedError(f"Malformed metadata in arXiv Atom feed: {exc}") from exc
    return papers, total


class ArxivClient:
    """Synchronous single-connection client with caching, pacing, and bounded retries."""

    def __init__(
        self,
        config: ArxivConfig,
        *,
        client: httpx.Client | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_at: float | None = None
        self._response_cache: dict[tuple[tuple[str, str], ...], bytes] = {}
        self._owns_client = client is None
        timeout = httpx.Timeout(
            connect=config.connect_timeout_seconds,
            read=config.read_timeout_seconds,
            write=config.read_timeout_seconds,
            pool=config.connect_timeout_seconds,
        )
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": f"arxiv-digest/{__version__} ({config.contact_email})",
                "Accept": "application/atom+xml",
            },
        )

    def __enter__(self) -> ArxivClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the internally created persistent HTTP connection."""
        if self._owns_client:
            self._client.close()

    def _pace(self) -> None:
        if self._last_request_at is not None:
            remaining = self._config.rate_limit_seconds - (
                self._monotonic() - self._last_request_at
            )
            if remaining > 0:
                self._sleep(remaining)

    def _request_page(self, params: Mapping[str, str | int]) -> bytes:
        key = tuple(sorted((name, str(value)) for name, value in params.items()))
        cached = self._response_cache.get(key)
        if cached is not None:
            return cached

        for attempt in range(self._config.retry_count + 1):
            self._pace()
            LOGGER.info("Requesting arXiv API params=%s", dict(params))
            try:
                response = self._client.get(self._config.api_url, params=params)
                self._last_request_at = self._monotonic()
            except httpx.TransportError as exc:
                self._last_request_at = self._monotonic()
                if attempt >= self._config.retry_count:
                    raise ArxivHTTPError(
                        f"arXiv request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                delay = self._config.retry_backoff_seconds * (2**attempt)
                LOGGER.warning("Transient arXiv transport error; retrying in %.1fs", delay)
                self._sleep(delay)
                continue

            if response.status_code in _TRANSIENT_STATUS_CODES:
                if attempt >= self._config.retry_count:
                    raise ArxivHTTPError(
                        f"arXiv returned HTTP {response.status_code} after {attempt + 1} attempts"
                    )
                delay = self._config.retry_backoff_seconds * (2**attempt)
                LOGGER.warning(
                    "arXiv returned HTTP %d; retrying in %.1fs", response.status_code, delay
                )
                self._sleep(delay)
                continue
            if response.is_error:
                raise ArxivHTTPError(f"arXiv returned non-retryable HTTP {response.status_code}")
            self._response_cache[key] = response.content
            return response.content

        raise ArxivHTTPError("arXiv request exhausted retries unexpectedly")

    def count(self, search_query: str) -> int:
        """Return how many results arXiv advertises for a query, fetching one page.

        Used to measure a term's ambiguity: the fraction of its hits that fall
        outside the group's field. Responses are cached like any other page, so
        repeating a measurement within a run is free.
        """
        content = self._request_page({"search_query": search_query, "start": 0, "max_results": 1})
        _papers, total = parse_atom_feed(content)
        return total

    def fetch(
        self,
        queries: Sequence[QueryConfig],
        window: DateWindow,
        maximum_candidates: int,
        domain: DomainConfig,
    ) -> RetrievalResult:
        """Fetch fair new/update shares for every query and filter explicit timestamps."""
        if not queries:
            return RetrievalResult(papers=[], query_results=[])
        per_query_limit = max(1, math.ceil(maximum_candidates / len(queries)))
        all_papers: list[Paper] = []
        query_results: list[QueryResult] = []

        for query in queries:
            received = 0
            streams = (
                (build_search_query(query, window, domain), "submittedDate", False),
                (build_term_query(query, domain), "lastUpdatedDate", True),
            )
            for search_query, sort_by, stop_after_old_page in streams:
                accepted = 0
                scanned = 0
                start = 0
                advertised_total: int | None = None
                while (
                    accepted < per_query_limit
                    and (advertised_total is None or start < advertised_total)
                    and scanned < per_query_limit + (4 * self._config.page_size)
                ):
                    page_size = min(self._config.page_size, per_query_limit - accepted)
                    params: dict[str, str | int] = {
                        "search_query": search_query,
                        "start": start,
                        "max_results": page_size,
                        "sortBy": sort_by,
                        "sortOrder": "descending",
                    }
                    page, advertised_total = parse_atom_feed(self._request_page(params))
                    if not page:
                        break
                    scanned += len(page)
                    received += len(page)
                    in_window = [
                        paper
                        for paper in page
                        if (
                            window.start <= paper.published_at < window.end
                            or window.start <= paper.updated_at < window.end
                        )
                    ]
                    all_papers.extend(in_window)
                    accepted += len(in_window)
                    start += len(page)
                    if stop_after_old_page and all(
                        paper.updated_at < window.start for paper in page
                    ):
                        break
                    if len(page) < page_size:
                        break
            query_results.append(QueryResult(name=query.name, records_received=received))

        all_papers.sort(
            key=lambda paper: (paper.updated_at, paper.published_at, paper.arxiv_id),
            reverse=True,
        )
        return RetrievalResult(papers=all_papers, query_results=query_results)
