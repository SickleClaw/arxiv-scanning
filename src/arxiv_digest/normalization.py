"""Pure normalization, tokenization, and deterministic de-duplication helpers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from urllib.parse import unquote, urlparse

from arxiv_digest.models import Paper

_ARXIV_ID_PATTERN = re.compile(
    r"^(?P<identifier>(?:\d{4}\.\d{4,5}|[a-z0-9.-]+/\d{7}))v(?P<version>\d+)$",
    re.IGNORECASE,
)


_SUBSCRIPT_DIGITS = str.maketrans(
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789"
)
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+")
_MATH_MARKUP = re.compile("[${}_^]")


def normalize_formulae(value: str) -> str:
    """Fold the ways arXiv writes chemical formulae into one plain form.

    ``Ho$_2$Ti$_2$O$_7$``, ``Ho\u2082Ti\u2082O\u2087`` and ``Ho2Ti2O7`` all become
    ``Ho2Ti2O7``. Normalizing the text is strictly better than generating the
    variants of each configured term and matching them one by one: it costs one
    pass instead of one per term, and it also folds spellings nobody thought to
    enumerate.

    Without this, ``ranking.tokenize`` split ``Ho$_2$Ti$_2$O$_7$`` into
    ``('ho', '2', 'ti', '2', 'o', '7')``, which never matched the configured
    ``Ho2Ti2O7``. The two highest-weighted materials in the profile were dead
    weight that also inflated the keyword denominator, suppressing every other
    term along with themselves.
    """
    folded = value.translate(_SUBSCRIPT_DIGITS)
    folded = _LATEX_COMMAND.sub("", folded)
    folded = folded.replace("\\", "")
    return _MATH_MARKUP.sub("", folded)


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(value: str) -> tuple[str, ...]:
    """Return case-folded whole-word tokens with chemical formulae folded together.

    Formula normalization happens here rather than at each call site so that
    every consumer — keyword scoring, hard rules, context requirements, topic
    assignment — matches the same text.
    """
    return tuple(token.casefold() for token in _TOKEN_PATTERN.findall(normalize_formulae(value)))


def count_term(tokens: Sequence[str], term: str) -> int:
    """Count exact token-sequence occurrences without substring false positives."""
    needle = tokenize(term)
    if not needle or len(needle) > len(tokens):
        return 0
    width = len(needle)
    return sum(
        tuple(tokens[index : index + width]) == needle for index in range(len(tokens) - width + 1)
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
