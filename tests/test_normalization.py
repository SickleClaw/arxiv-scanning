"""Tests for identity, title normalization, and cross-query de-duplication."""

from datetime import UTC, datetime

import pytest

from arxiv_digest.normalization import (
    deduplicate_papers,
    normalized_title_key,
    parse_arxiv_identity,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://arxiv.org/abs/2607.12345v2", ("2607.12345", 2)),
        ("cond-mat/0501234v3", ("cond-mat/0501234", 3)),
    ],
)
def test_parse_arxiv_identity(value: str, expected: tuple[str, int]) -> None:
    assert parse_arxiv_identity(value) == expected


def test_invalid_arxiv_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid versioned"):
        parse_arxiv_identity("https://example.com/not-an-arxiv-id")


def test_title_key_ignores_trivial_punctuation_and_spacing() -> None:
    assert normalized_title_key("Spin-Ice: Dynamics") == normalized_title_key(
        " spin ice — dynamics! "
    )


def test_deduplicate_prefers_new_version_and_merges_categories(paper_factory) -> None:  # type: ignore[no-untyped-def]
    old = paper_factory(categories=["cond-mat.str-el"])
    new = paper_factory(
        version=2,
        updated_at=datetime(2026, 7, 28, tzinfo=UTC),
        categories=["cond-mat.str-el", "physics.comp-ph"],
    )
    title_variant = paper_factory(
        arxiv_id="2607.99999",
        version=5,
        title="A spin-ice paper!",
        published_at=datetime(2026, 7, 24, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, tzinfo=UTC),
    )
    result = deduplicate_papers([old, title_variant, new])
    assert len(result) == 1
    assert result[0].version == 2
    assert result[0].categories == ["cond-mat.str-el", "physics.comp-ph"]
