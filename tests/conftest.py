"""Shared test helpers and fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from arxiv_digest.models import (
    ABSTRACT_SUMMARY_BASIS,
    DIGEST_SCHEMA_VERSION,
    DateWindow,
    DigestArtifact,
    Paper,
    PaperSummary,
    RankedPaper,
    Recommendation,
    RecommendationType,
    ScoreBreakdown,
)


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


@pytest.fixture
def digest_factory(paper_factory):  # type: ignore[no-untyped-def]
    """Build a small, complete canonical digest for dashboard tests."""

    def score(value: float, term: str) -> ScoreBreakdown:
        return ScoreBreakdown(
            semantic_relevance=value,
            keyword_relevance=value,
            category_relevance=value,
            recency=value,
            novelty=value,
            feedback_affinity=0.5,
            final_preselection_score=value,
            explanation=f"Strongest matched profile terms: {term}. Category overlap.",
        )

    def factory(**overrides: object) -> DigestArtifact:
        papers = [
            paper_factory(
                arxiv_id="2607.10001",
                version=1,
                title="Spin ice monopole dynamics",
                authors=["Ada Curie", "Max Planck"],
                abstract="Magnetic monopoles evolve out of equilibrium in pyrochlore spin ice.",
                primary_category="cond-mat.str-el",
                categories=["cond-mat.str-el", "cond-mat.stat-mech"],
                published_at=datetime(2026, 7, 23, tzinfo=UTC),
                updated_at=datetime(2026, 7, 27, tzinfo=UTC),
            ),
            paper_factory(
                arxiv_id="2607.10002",
                title="Neutron scattering in a frustrated magnet",
                authors=["Nia Raman"],
                abstract="Neutron diffuse scattering resolves a magnetic structure factor.",
                primary_category="cond-mat.mtrl-sci",
                categories=["cond-mat.mtrl-sci"],
                published_at=datetime(2026, 7, 24, tzinfo=UTC),
                updated_at=datetime(2026, 7, 26, tzinfo=UTC),
            ),
            paper_factory(
                arxiv_id="2607.10003",
                title="Cluster algorithms for disordered spinels",
                authors=["Lee Onsager"],
                abstract="A cluster method accelerates simulations of frustrated spinels.",
                primary_category="physics.comp-ph",
                categories=["physics.comp-ph"],
                published_at=datetime(2026, 7, 25, tzinfo=UTC),
                updated_at=datetime(2026, 7, 25, tzinfo=UTC),
            ),
            paper_factory(
                arxiv_id="2607.10004",
                title="Peripheral quantum transport",
                authors=["Sam Bell"],
                abstract="Transport is studied in an adjacent condensed matter system.",
                primary_category="cond-mat.mes-hall",
                categories=["cond-mat.mes-hall"],
                published_at=datetime(2026, 7, 22, tzinfo=UTC),
                updated_at=datetime(2026, 7, 22, tzinfo=UTC),
            ),
        ]
        scores = [
            score(0.91, "spin ice, magnetic monopoles"),
            score(0.82, "neutron scattering"),
            score(0.73, "cluster algorithms, spinels"),
            score(0.41, "condensed matter"),
        ]
        ranked = [
            RankedPaper(paper=paper, score=item_score)
            for paper, item_score in zip(papers, scores, strict=True)
        ]
        summary = PaperSummary(
            one_sentence_takeaway="The abstract reports a relevant condensed-matter result.",
            brief_summary=(
                "The authors describe a result using only the information available in the "
                "supplied abstract and metadata."
            ),
            why_relevant=(
                "The metadata overlaps with configured materials, methods, or concepts in "
                "the research profile."
            ),
            methods_or_systems=["abstract metadata"],
            limitations="The full paper, figures, equations, and appendices were not inspected.",
            summary_basis=ABSTRACT_SUMMARY_BASIS,
            confidence=0.7,
        )
        recommendations = [
            Recommendation(
                paper=papers[0],
                score=scores[0],
                summary=summary,
                rank=1,
                recommendation_type=RecommendationType.DIRECT,
            ),
            Recommendation(
                paper=papers[1],
                score=scores[1],
                summary=summary,
                rank=2,
                recommendation_type=RecommendationType.ADJACENT,
            ),
        ]
        values: dict[str, object] = {
            "schema_version": DIGEST_SCHEMA_VERSION,
            "run_id": "selection-dashboard-fixture",
            "title": "Personalized Weekly arXiv Reading List",
            "generated_at": datetime(2026, 7, 28, 12, tzinfo=UTC),
            "timezone": "America/New_York",
            "retrieval_window": DateWindow(
                start=datetime(2026, 7, 20, tzinfo=UTC),
                end=datetime(2026, 7, 28, tzinfo=UTC),
            ),
            "records_retrieved": 6,
            "records_after_deduplication": 4,
            "records_ranked": 4,
            "records_selected": 2,
            "summary_mode": "offline",
            "summary_fallbacks": 0,
            "summary_basis": ABSTRACT_SUMMARY_BASIS,
            "profile_name": "fixture profile",
            "profile_version": "1.0",
            "profile_hash": "abc123",
            "recommendations": recommendations,
            "near_misses": [ranked[2]],
            "ranked_candidates": ranked,
        }
        values.update(overrides)
        return DigestArtifact.model_validate(values)

    return factory
