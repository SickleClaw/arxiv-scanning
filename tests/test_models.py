"""Tests for strict core data models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from arxiv_digest.models import DateWindow


def test_date_window_requires_aware_ordered_datetimes() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        DateWindow(start=datetime(2026, 1, 1), end=datetime(2026, 1, 2))
    with pytest.raises(ValidationError, match="end must be after start"):
        DateWindow(
            start=datetime(2026, 1, 2, tzinfo=UTC),
            end=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_paper_normalizes_whitespace_and_category(paper_factory) -> None:  # type: ignore[no-untyped-def]
    paper = paper_factory(
        title="  Spin\n  ice — μSR  ",
        abstract=" preserve\tUnicode Ω  and spacing ",
        primary_category="cond-mat.str-el",
        categories=["physics.comp-ph", "physics.comp-ph"],
    )
    assert paper.title == "Spin ice — μSR"
    assert paper.abstract == "preserve Unicode Ω and spacing"
    assert paper.categories == ["cond-mat.str-el", "physics.comp-ph"]
