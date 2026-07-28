"""Shared test helpers and fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from arxiv_digest.models import Paper


@pytest.fixture
def fixture_dir() -> Path:
    """Return the checked-in Atom fixture directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def paper_factory():  # type: ignore[no-untyped-def]
    """Build normalized papers while allowing concise field overrides."""

    def factory(**overrides: object) -> Paper:
        values: dict[str, object] = {
            "arxiv_id": "2607.12345",
            "version": 1,
            "title": "A spin ice paper",
            "authors": ["Ada Curie"],
            "abstract": "An abstract about magnetic dynamics.",
            "primary_category": "cond-mat.str-el",
            "categories": ["cond-mat.str-el"],
            "published_at": datetime(2026, 7, 26, tzinfo=UTC),
            "updated_at": datetime(2026, 7, 27, tzinfo=UTC),
            "abstract_url": "https://arxiv.org/abs/2607.12345v1",
            "pdf_url": "https://arxiv.org/pdf/2607.12345v1",
        }
        values.update(overrides)
        return Paper.model_validate(values)

    return factory
