"""Pure normalization and deterministic de-duplication helpers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from urllib.parse import unquote, urlparse

from arxiv_digest.models import Paper

_ARXIV_ID_PATTERN = re.compile(
    r"^(?P<identifier>(?:\d{4}\.\d{4,5}|[a-z0-9.-]+/\d{7}))v(?P<version>\d+)$",
    re.IGNORECASE,
)


def normalize_whitespace(value: str) -> str:
    """Collapse all Unicode whitespace runs without altering other characters."""
    return re.sub(r"\s+", " ", value).strip()


def parse_arxiv_identity(value: str) -> tuple[str, int]:
    """Extract the canonical identifier and integer version from an arXiv URL or ID."""
    parsed = urlparse(value)
    candidate = unquote(parsed.path if parsed.scheme else value).strip("/")
    if candidate.startswith("abs/"):
        candidate = candidate[4:]
    match = _ARXIV_ID_PATTERN.fullmatch(candidate)
    if match is None:
        raise ValueError(f"Invalid versioned arXiv identifier: {value}")
    return match.group("identifier"), int(match.group("version"))


def normalized_title_key(title: str) -> str:
    """Create a punctuation-insensitive key for trivial title variants."""
    folded = normalize_whitespace(title).casefold()
    characters = (
        " " if unicodedata.category(character).startswith(("P", "Z")) else character
        for character in folded
    )
    return normalize_whitespace("".join(characters))


def _preferred_paper(left: Paper, right: Paper) -> Paper:
    if left.arxiv_id == right.arxiv_id:
        winner = max(
            (left, right),
            key=lambda paper: (
                paper.version,
                paper.updated_at,
                paper.published_at,
                paper.arxiv_id,
            ),
        )
    else:
        winner = max(
            (left, right),
            key=lambda paper: (
                paper.updated_at,
                paper.published_at,
                paper.version,
                paper.arxiv_id,
            ),
        )
    other = right if winner is left else left
    categories = list(dict.fromkeys([*winner.categories, *other.categories]))
    return winner.model_copy(update={"categories": categories})


def deduplicate_papers(papers: Iterable[Paper]) -> list[Paper]:
    """Deduplicate versions and trivial title variants with deterministic winners."""
    by_identifier: dict[str, Paper] = {}
    for paper in papers:
        current = by_identifier.get(paper.arxiv_id)
        by_identifier[paper.arxiv_id] = (
            paper if current is None else _preferred_paper(current, paper)
        )

    by_title: dict[str, Paper] = {}
    for paper in by_identifier.values():
        key = normalized_title_key(paper.title)
        current = by_title.get(key)
        by_title[key] = paper if current is None else _preferred_paper(current, paper)

    return sorted(
        by_title.values(),
        key=lambda paper: (paper.updated_at, paper.published_at, paper.arxiv_id),
        reverse=True,
    )
