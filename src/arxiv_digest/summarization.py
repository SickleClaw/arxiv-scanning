"""Abstract-grounded summary and embedding provider interfaces and implementations."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from arxiv_digest.config import ResearchProfile
from arxiv_digest.exceptions import SummaryProviderError
from arxiv_digest.models import (
    ABSTRACT_SUMMARY_BASIS,
    Paper,
    PaperSummary,
    Recommendation,
    ScoreBreakdown,
    SelectedPaper,
)

LOGGER = logging.getLogger(__name__)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class SummaryProvider(Protocol):
    """Interface for metadata-only paper summary providers."""

    def summarize(
        self,
        paper: Paper,
        score: ScoreBreakdown,
        profile: ResearchProfile,
    ) -> PaperSummary:
        """Return a validated summary grounded in the supplied paper metadata."""


class EmbeddingProvider(Protocol):
    """Interface for optional text embedding providers used by future ranking variants."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in input order."""


class _ParsedResponse(Protocol):
    output_parsed: object | None


class _ResponsesAPI(Protocol):
    def parse(
        self,
        *,
        model: str,
        input: list[dict[str, str]],
        text_format: type[PaperSummary],
    ) -> _ParsedResponse: ...


class _EmbeddingDatum(Protocol):
    embedding: list[float]


class _EmbeddingResponse(Protocol):
    data: list[_EmbeddingDatum]


class _EmbeddingsAPI(Protocol):
    def create(
        self,
        *,
        model: str,
        input: list[str],
        encoding_format: str,
    ) -> _EmbeddingResponse: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesAPI
    embeddings: _EmbeddingsAPI


def _truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).rstrip(".,;:") + "…"


def _first_sentence(text: str) -> str:
    sentence = _SENTENCE_BOUNDARY.split(text.strip(), maxsplit=1)[0]
    return sentence.rstrip(".") + "."


def _term_present(term: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", text.casefold()) is not None


def _matched_profile_terms(paper: Paper, profile: ResearchProfile) -> list[str]:
    text = f"{paper.title} {paper.abstract}"
    weighted_terms = {
        **profile.exact_phrases,
        **profile.materials,
        **profile.methods,
    }
    matches = [term for term, _weight in weighted_terms.items() if _term_present(term, text)]
    return sorted(matches, key=lambda term: (-weighted_terms[term], term.casefold()))[:5]


class DeterministicSummaryProvider:
    """Create reproducible extractive summaries without network access or paid APIs."""

    def summarize(
        self,
        paper: Paper,
        score: ScoreBreakdown,
        profile: ResearchProfile,
    ) -> PaperSummary:
        """Build a concise summary from title, abstract, and configured profile terms."""
        matches = _matched_profile_terms(paper, profile)
        methods = matches or [paper.primary_category]
        match_text = ", ".join(matches) if matches else paper.primary_category
        takeaway = _truncate_words(
            f"The abstract reports: {_first_sentence(paper.abstract)}",
            35,
        )
        brief = _truncate_words(
            f"The abstract states: {paper.abstract}",
            125,
        )
        why = _truncate_words(
            "This paper was selected because its metadata and abstract overlap with the "
            f"configured profile through {match_text}. Its preselection score was "
            f"{score.final_preselection_score:.3f}, so the relevance claim reflects only "
            "the available abstract and ranking signals.",
            70,
        )
        limitations = (
            "The abstract and metadata do not provide enough information to assess the full "
            "derivation, experimental details, figures, or robustness of the reported results."
        )
        confidence = min(0.85, max(0.45, len(paper.abstract.split()) / 200.0))
        return PaperSummary(
            one_sentence_takeaway=takeaway,
            brief_summary=brief,
            why_relevant=why,
            methods_or_systems=methods,
            limitations=limitations,
            summary_basis=ABSTRACT_SUMMARY_BASIS,
            confidence=confidence,
        )


class OpenAISummaryProvider:
    """Generate validated metadata-only summaries using OpenAI structured outputs."""

    def __init__(
        self,
        *,
        model: str,
        validation_retries: int = 2,
        client: object | None = None,
    ) -> None:
        if not model.strip():
            raise SummaryProviderError(
                "OPENAI_SUMMARY_MODEL is required when the OpenAI summary provider is enabled"
            )
        self._model = model
        self._validation_retries = validation_retries
        try:
            resolved_client = client if client is not None else OpenAI()
        except OpenAIError as exc:
            raise SummaryProviderError(f"Cannot initialize the OpenAI client: {exc}") from exc
        self._client = cast(_OpenAIClient, resolved_client)

    def summarize(
        self,
        paper: Paper,
        score: ScoreBreakdown,
        profile: ResearchProfile,
    ) -> PaperSummary:
        """Request strict structured output and retry a bounded number of invalid responses."""
        metadata = {
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "primary_category": paper.primary_category,
            "categories": paper.categories,
            "published_at": paper.published_at.isoformat(),
            "updated_at": paper.updated_at.isoformat(),
            "profile_name": profile.name,
            "profile_description": profile.description,
            "strongest_ranking_matches": score.explanation,
        }
        instructions = (
            "Summarize only the supplied arXiv title, abstract, authors, categories, and dates. "
            "Attribute claims to the authors or abstract. Do not imply that the PDF, figures, "
            "equations, appendices, or supporting information were read. Do not invent numbers, "
            "materials, instruments, temperatures, fields, or methods. State when metadata is "
            "insufficient. Keep the takeaway at most 35 words, brief summary about 80-130 words, "
            "why_relevant about 30-70 words, and limitations to one concise sentence. Set "
            f"summary_basis exactly to: {ABSTRACT_SUMMARY_BASIS}"
        )
        last_error: Exception | None = None
        for attempt in range(self._validation_retries + 1):
            try:
                response = self._client.responses.parse(
                    model=self._model,
                    input=[
                        {"role": "developer", "content": instructions},
                        {"role": "user", "content": json.dumps(metadata, ensure_ascii=False)},
                    ],
                    text_format=PaperSummary,
                )
                if response.output_parsed is None:
                    raise SummaryProviderError("OpenAI returned no parsed summary")
                return PaperSummary.model_validate(response.output_parsed)
            except (OpenAIError, ValidationError, ValueError, SummaryProviderError) as exc:
                last_error = exc
                if attempt < self._validation_retries:
                    LOGGER.warning(
                        "stage=summarize provider=openai validation_retry=%d",
                        attempt + 1,
                    )
        raise SummaryProviderError(
            "OpenAI did not return a valid abstract-grounded summary after "
            f"{self._validation_retries + 1} attempts: {last_error}"
        ) from last_error


class OpenAIEmbeddingProvider:
    """Optional OpenAI embedding implementation behind the common provider interface."""

    def __init__(self, *, model: str, client: object | None = None) -> None:
        if not model.strip():
            raise SummaryProviderError("OPENAI_EMBEDDING_MODEL is required for OpenAI embeddings")
        self._model = model
        try:
            resolved_client = client if client is not None else OpenAI()
        except OpenAIError as exc:
            raise SummaryProviderError(f"Cannot initialize the OpenAI client: {exc}") from exc
        self._client = cast(_OpenAIClient, resolved_client)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return finite vectors in the same order as the nonblank input texts."""
        values = list(texts)
        if not values or any(not value.strip() for value in values):
            raise SummaryProviderError("Embedding input must contain nonblank text")
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=values,
                encoding_format="float",
            )
        except OpenAIError as exc:
            raise SummaryProviderError(f"OpenAI embedding request failed: {exc}") from exc
        vectors = [list(item.embedding) for item in response.data]
        if len(vectors) != len(values) or any(
            not vector or any(not math.isfinite(component) for component in vector)
            for vector in vectors
        ):
            raise SummaryProviderError("OpenAI returned invalid embedding vectors")
        return vectors


@dataclass(frozen=True, slots=True)
class SummaryBatch:
    """Recommendations plus observability counts from one summary batch."""

    recommendations: list[Recommendation]
    fallback_count: int


def summarize_selection(
    selected: Sequence[SelectedPaper],
    profile: ResearchProfile,
    provider: SummaryProvider,
    *,
    max_provider_papers: int,
    fallback: SummaryProvider | None = None,
) -> SummaryBatch:
    """Summarize selections in rank order, falling back without dropping any paper."""
    offline = fallback or DeterministicSummaryProvider()
    primary_is_offline = isinstance(provider, DeterministicSummaryProvider)
    recommendations: list[Recommendation] = []
    fallback_count = 0
    for index, item in enumerate(selected):
        active_provider = provider if index < max_provider_papers else offline
        used_fallback = active_provider is offline and not primary_is_offline
        try:
            summary = active_provider.summarize(item.paper, item.score, profile)
        except (SummaryProviderError, ValidationError, ValueError) as exc:
            LOGGER.warning(
                "stage=summarize arxiv_id=%s fallback=offline reason=%s",
                item.paper.arxiv_id,
                type(exc).__name__,
            )
            summary = offline.summarize(item.paper, item.score, profile)
            used_fallback = True
        fallback_count += int(used_fallback)
        recommendations.append(
            Recommendation(
                paper=item.paper,
                score=item.score,
                summary=summary,
                rank=item.rank,
                recommendation_type=item.recommendation_type,
            )
        )
    return SummaryBatch(recommendations=recommendations, fallback_count=fallback_count)
