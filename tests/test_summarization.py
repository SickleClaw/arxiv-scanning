"""Tests for deterministic and optional OpenAI provider behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from arxiv_digest.config import load_settings
from arxiv_digest.exceptions import SummaryProviderError
from arxiv_digest.models import (
    ABSTRACT_SUMMARY_BASIS,
    PaperSummary,
    RecommendationType,
    ScoreBreakdown,
    SelectedPaper,
)
from arxiv_digest.summarization import (
    DeterministicSummaryProvider,
    OpenAIEmbeddingProvider,
    OpenAISummaryProvider,
    summarize_selection,
)


def score() -> ScoreBreakdown:
    return ScoreBreakdown(
        semantic_relevance=0.8,
        keyword_relevance=0.7,
        category_relevance=1.0,
        recency=0.9,
        novelty=1.0,
        feedback_affinity=0.5,
        final_preselection_score=0.79,
        explanation="Strongest matched profile terms: spin ice, magnetic monopole.",
    )


def valid_summary() -> PaperSummary:
    return PaperSummary(
        one_sentence_takeaway="The authors report field-driven magnetic dynamics in spin ice.",
        brief_summary=(
            "The abstract describes a study of field-driven magnetic dynamics in a spin-ice "
            "system. The authors report a metadata-grounded comparison of relaxation behavior "
            "and connect it to magnetic monopole motion. The supplied abstract does not establish "
            "details beyond those claims."
        ),
        why_relevant=(
            "The topic overlaps directly with the configured interests in spin ice, magnetic "
            "monopoles, and nonequilibrium magnetic dynamics."
        ),
        methods_or_systems=["spin ice", "magnetic monopole"],
        limitations="The abstract does not provide enough detail to assess the full analysis.",
        summary_basis=ABSTRACT_SUMMARY_BASIS,
        confidence=0.8,
    )


def test_offline_summary_is_deterministic_and_abstract_grounded(paper_factory) -> None:  # type: ignore[no-untyped-def]
    paper = paper_factory(
        title="Field-driven spin ice magnetic monopole dynamics",
        abstract=(
            "We report field-driven relaxation in a pyrochlore spin ice. The results connect "
            "the observed response to magnetic monopole dynamics using neutron scattering."
        ),
    )
    provider = DeterministicSummaryProvider()
    profile = load_settings().profile
    first = provider.summarize(paper, score(), profile)
    second = provider.summarize(paper, score(), profile)
    assert first == second
    assert first.summary_basis == ABSTRACT_SUMMARY_BASIS
    assert len(first.one_sentence_takeaway.split()) <= 35
    assert "spin ice" in first.methods_or_systems
    assert "full derivation" in first.limitations


def test_openai_provider_retries_validation_and_never_sends_pdf(paper_factory) -> None:  # type: ignore[no-untyped-def]
    calls: list[dict[str, object]] = []

    class FakeResponses:
        def parse(self, **kwargs: object) -> object:
            calls.append(kwargs)
            parsed = None if len(calls) == 1 else valid_summary()
            return SimpleNamespace(output_parsed=parsed)

    client = SimpleNamespace(responses=FakeResponses())
    provider = OpenAISummaryProvider(model="configured-model", validation_retries=1, client=client)
    result = provider.summarize(paper_factory(), score(), load_settings().profile)
    assert result == valid_summary()
    assert len(calls) == 2
    assert calls[0]["model"] == "configured-model"
    messages = calls[0]["input"]
    serialized = json.dumps(messages)
    assert "pdf_url" not in serialized
    assert "https://arxiv.org/pdf" not in serialized
    assert "abstract" in serialized


def test_summary_batch_falls_back_without_dropping_paper(paper_factory) -> None:  # type: ignore[no-untyped-def]
    class FailingProvider:
        def summarize(self, *_args: object) -> PaperSummary:
            raise SummaryProviderError("fixture failure")

    selected = SelectedPaper(
        paper=paper_factory(),
        score=score(),
        rank=1,
        recommendation_type=RecommendationType.DIRECT,
        topic_key="spin ice",
    )
    batch = summarize_selection(
        [selected],
        load_settings().profile,
        FailingProvider(),
        max_provider_papers=1,
    )
    assert len(batch.recommendations) == 1
    assert batch.fallback_count == 1
    assert batch.recommendations[0].summary.summary_basis == ABSTRACT_SUMMARY_BASIS


def test_openai_embedding_provider_preserves_order_and_validates() -> None:
    class FakeEmbeddings:
        def create(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 0.0]), SimpleNamespace(embedding=[0.0, 1.0])]
            )

    provider = OpenAIEmbeddingProvider(
        model="configured-embedding-model",
        client=SimpleNamespace(embeddings=FakeEmbeddings()),
    )
    assert provider.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(SummaryProviderError, match="nonblank"):
        provider.embed([""])


def test_openai_provider_exhaustion_has_actionable_error(paper_factory) -> None:  # type: ignore[no-untyped-def]
    class EmptyResponses:
        def parse(self, **_kwargs: object) -> object:
            return SimpleNamespace(output_parsed=None)

    provider = OpenAISummaryProvider(
        model="configured-model",
        validation_retries=1,
        client=SimpleNamespace(responses=EmptyResponses()),
    )
    with pytest.raises(SummaryProviderError, match="after 2 attempts"):
        provider.summarize(paper_factory(), score(), load_settings().profile)
