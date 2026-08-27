"""Deterministic offline hybrid ranking for arXiv candidate metadata."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from arxiv_digest.config import ResearchProfile
from arxiv_digest.models import DateWindow, Paper, RankedPaper, ScoreBreakdown
from arxiv_digest.normalization import count_term, tokenize

_TITLE_MATCH_WEIGHT = 1.0
_ABSTRACT_MATCH_WEIGHT = 0.35
_MAX_AUTHOR_BOOST = 0.15


def _positive_terms(profile: ResearchProfile) -> dict[str, float]:
    terms: dict[str, float] = {}
    for configured in (profile.exact_phrases, profile.materials, profile.methods):
        for term, weight in configured.items():
            terms[term] = max(terms.get(term, 0.0), max(weight, 0.0))
    return terms


def keyword_relevance(
    paper: Paper,
    profile: ResearchProfile,
) -> tuple[float, list[tuple[str, float]]]:
    """Score weighted whole-term matches in 0..1, favoring title over abstract matches.

    Negative terms are not handled here. They are a gate now (see
    ``domain_filter.gate_papers``): a paper matching one is removed with a
    logged reason rather than penalized by an amount other components could
    outvote.
    """
    title_tokens = tokenize(paper.title)
    abstract_tokens = tokenize(paper.abstract)
    configured_terms = _positive_terms(profile)
    total_weight = sum(configured_terms.values())
    matched: list[tuple[str, float]] = []
    weighted_score = 0.0
    for term, weight in configured_terms.items():
        title_count = count_term(title_tokens, term)
        abstract_count = count_term(abstract_tokens, term)
        strength = min(
            1.0,
            (_TITLE_MATCH_WEIGHT * title_count) + (_ABSTRACT_MATCH_WEIGHT * abstract_count),
        )
        if strength > 0:
            contribution = weight * strength
            weighted_score += contribution
            matched.append((term, contribution))

    base_score = weighted_score / total_weight if total_weight > 0 else 0.0
    normalized_authors = {" ".join(tokenize(author)) for author in paper.authors}
    author_weights = [
        weight
        for author, weight in profile.author_boosts.items()
        if " ".join(tokenize(author)) in normalized_authors and weight > 0
    ]
    if author_weights:
        largest_configured = max(profile.author_boosts.values(), default=1.0)
        author_boost = _MAX_AUTHOR_BOOST * min(
            1.0, max(author_weights) / max(largest_configured, 1.0)
        )
        base_score += author_boost

    matched.sort(key=lambda item: (-item[1], item[0].casefold()))
    return min(1.0, base_score), matched


def category_relevance(paper: Paper, profile: ResearchProfile) -> float:
    """Return the strongest configured category weight normalized to 0..1."""
    maximum = max(profile.categories.values(), default=0.0)
    if maximum <= 0:
        return 0.0
    matched = max(
        (profile.categories.get(category, 0.0) for category in paper.categories), default=0.0
    )
    return min(1.0, max(0.0, matched / maximum))


def recency_score(paper: Paper, window: DateWindow) -> float:
    """Map the paper's latest timestamp linearly onto the retrieval window in 0..1."""
    duration = (window.end - window.start).total_seconds()
    if duration <= 0:
        return 0.0
    timestamp = max(paper.published_at, paper.updated_at)
    position = (timestamp - window.start).total_seconds() / duration
    return min(1.0, max(0.0, position))


def _profile_document(profile: ResearchProfile) -> str:
    weighted_terms: list[str] = []
    for term, weight in _positive_terms(profile).items():
        weighted_terms.extend([term] * max(1, math.ceil(weight)))
    return " ".join(
        [
            profile.description,
            *profile.priority_topics,
            *weighted_terms,
            *profile.categories,
        ]
    )


def _tfidf_matrix(documents: Sequence[Sequence[str]]) -> NDArray[np.float64]:
    vocabulary = sorted({token for document in documents for token in document})
    if not vocabulary:
        return np.zeros((len(documents), 0), dtype=np.float64)
    indices = {token: index for index, token in enumerate(vocabulary)}
    document_count = len(documents)
    document_frequency = Counter(token for document in documents for token in set(document))
    matrix = np.zeros((document_count, len(vocabulary)), dtype=np.float64)
    for row, document in enumerate(documents):
        counts = Counter(document)
        token_count = max(1, len(document))
        for token, count in counts.items():
            inverse_frequency = (
                math.log((1 + document_count) / (1 + document_frequency[token])) + 1.0
            )
            matrix[row, indices[token]] = (count / token_count) * inverse_frequency
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def semantic_relevance_scores(papers: Sequence[Paper], profile: ResearchProfile) -> list[float]:
    """Compute deterministic corpus TF-IDF cosine similarities in 0..1."""
    if not papers:
        return []
    paper_documents = [
        tokenize(f"{paper.title} {paper.title} {paper.abstract} {' '.join(paper.categories)}")
        for paper in papers
    ]
    documents = [*paper_documents, tokenize(_profile_document(profile))]
    matrix = _tfidf_matrix(documents)
    profile_vector = matrix[-1]
    similarities = matrix[:-1] @ profile_vector
    return [min(1.0, max(0.0, float(value))) for value in similarities]


def rank_papers(
    papers: Iterable[Paper],
    profile: ResearchProfile,
    window: DateWindow,
    previously_recommended_ids: set[str] | None = None,
) -> list[RankedPaper]:
    """Rank papers with configurable hybrid weights and deterministic tie-breaking."""
    candidates = list(papers)
    semantic_scores = semantic_relevance_scores(candidates, profile)
    previous_ids = previously_recommended_ids or set()
    weights = profile.ranking_weights
    ranked: list[RankedPaper] = []

    for paper, semantic in zip(candidates, semantic_scores, strict=True):
        keyword, matched = keyword_relevance(paper, profile)
        category = category_relevance(paper, profile)
        recency = recency_score(paper, window)
        novelty = 0.5 if paper.arxiv_id in previous_ids else 1.0
        feedback_affinity = 0.5
        raw_score = (
            (weights.semantic_relevance * semantic)
            + (weights.keyword_relevance * keyword)
            + (weights.category_relevance * category)
            + (weights.recency * recency)
            + (weights.feedback_or_novelty * novelty)
        )
        final_score = min(1.0, max(0.0, raw_score))
        strongest = [term for term, _contribution in matched[:5]]
        explanation_parts = [
            f"Strongest matched profile terms: {', '.join(strongest)}."
            if strongest
            else "No configured positive profile terms matched."
        ]
        ranked.append(
            RankedPaper(
                paper=paper,
                score=ScoreBreakdown(
                    semantic_relevance=semantic,
                    keyword_relevance=keyword,
                    category_relevance=category,
                    recency=recency,
                    novelty=novelty,
                    feedback_affinity=feedback_affinity,
                    final_preselection_score=final_score,
                    explanation=" ".join(explanation_parts),
                ),
            )
        )

    return sorted(
        ranked,
        key=lambda item: (
            -item.score.final_preselection_score,
            -item.paper.updated_at.timestamp(),
            item.paper.arxiv_id,
        ),
    )
