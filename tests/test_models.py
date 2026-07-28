"""Tests for strict core data models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from arxiv_digest.models import ABSTRACT_SUMMARY_BASIS, DateWindow, PaperSummary


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


def test_summary_validation_enforces_length_and_abstract_basis() -> None:
    values = {
        "one_sentence_takeaway": "A concise takeaway.",
        "brief_summary": "A concise abstract-grounded summary.",
        "why_relevant": "It matches spin ice.",
        "methods_or_systems": ["spin ice"],
        "limitations": "The abstract omits experimental details.",
        "summary_basis": ABSTRACT_SUMMARY_BASIS,
        "confidence": 0.7,
    }
    assert PaperSummary.model_validate(values).summary_basis == ABSTRACT_SUMMARY_BASIS
    with pytest.raises(ValidationError, match="no more than 35 words"):
        PaperSummary.model_validate({**values, "one_sentence_takeaway": " ".join(["word"] * 36)})
    with pytest.raises(ValidationError, match="summary_basis must be exactly"):
        PaperSummary.model_validate({**values, "summary_basis": "Based on the paper."})
    with pytest.raises(ValidationError, match="must not imply inspection"):
        PaperSummary.model_validate({**values, "brief_summary": "The figures show the result."})
