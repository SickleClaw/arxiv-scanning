"""Deterministic diversity-aware selection using maximal marginal relevance."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from arxiv_digest.config import ResearchProfile
from arxiv_digest.models import (
    RankedPaper,
    RecommendationType,
    SelectedPaper,
)
from arxiv_digest.normalization import count_term, tokenize


def recommendation_type(
    candidate: RankedPaper, profile: ResearchProfile
) -> RecommendationType | None:
    """Classify a candidate by configurable strength thresholds, rejecting weak papers."""
    score = candidate.score.final_preselection_score
    thresholds = profile.selection
    if score >= thresholds.direct_min_score:
        return RecommendationType.DIRECT
    if score >= thresholds.adjacent_min_score:
        return RecommendationType.ADJACENT
    if score >= thresholds.wildcard_min_score:
        return RecommendationType.WILDCARD
    return None


def topic_key(candidate: RankedPaper, profile: ResearchProfile) -> str:
    """Assign the strongest configured broad-query topic for deterministic topic caps."""
    title_tokens = tokenize(candidate.paper.title)
    abstract_tokens = tokenize(candidate.paper.abstract)
    scored_topics: list[tuple[int, str]] = []
    for query in profile.queries:
        strength = sum(
            (2 * count_term(title_tokens, term)) + count_term(abstract_tokens, term)
            for term in query.terms
        )
        scored_topics.append((strength, query.name))
    strongest = min(scored_topics, key=lambda item: (-item[0], item[1]))
    return strongest[1] if strongest[0] > 0 else f"category:{candidate.paper.primary_category}"


def _normalized_binary_vectors(candidates: Sequence[RankedPaper]) -> NDArray[np.float64]:
    documents = [
        set(tokenize(f"{candidate.paper.title} {candidate.paper.abstract}"))
        for candidate in candidates
    ]
    vocabulary = sorted(set().union(*documents)) if documents else []
    if not vocabulary:
        return np.zeros((len(candidates), 0), dtype=np.float64)
    indices = {token: index for index, token in enumerate(vocabulary)}
    matrix = np.zeros((len(candidates), len(vocabulary)), dtype=np.float64)
    for row, document in enumerate(documents):
        for token in document:
            matrix[row, indices[token]] = 1.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def select_diverse(
    ranked: Sequence[RankedPaper],
    profile: ResearchProfile,
    *,
    limit: int = 10,
) -> list[SelectedPaper]:
    """Select strong, varied candidates with deterministic MMR, mix targets, and caps."""
    if limit < 1:
        return []
    eligible: list[tuple[RankedPaper, RecommendationType, str]] = []
    for candidate in ranked:
        candidate_type = recommendation_type(candidate, profile)
        if candidate_type is not None:
            eligible.append((candidate, candidate_type, topic_key(candidate, profile)))
    if not eligible:
        return []

    candidates = [item[0] for item in eligible]
    vectors = _normalized_binary_vectors(candidates)
    similarities = vectors @ vectors.T
    selected_indices: list[int] = []
    selected_types: Counter[RecommendationType] = Counter()
    selected_topics: Counter[str] = Counter()
    targets = {
        RecommendationType.DIRECT: profile.recommendation_mix.direct,
        RecommendationType.ADJACENT: profile.recommendation_mix.adjacent,
        RecommendationType.WILDCARD: profile.recommendation_mix.wildcard,
    }

    while len(selected_indices) < min(limit, len(eligible)):
        available = [
            index
            for index, (_candidate, _candidate_type, candidate_topic) in enumerate(eligible)
            if index not in selected_indices
            and selected_topics[candidate_topic] < profile.max_papers_per_topic
        ]
        if not available:
            break
        needed = [
            index
            for index in available
            if selected_types[eligible[index][1]] < targets[eligible[index][1]]
        ]
        pool = needed or available

        def mmr_key(index: int) -> tuple[float, float, float, str]:
            candidate = eligible[index][0]
            redundancy = (
                max(float(similarities[index, chosen]) for chosen in selected_indices)
                if selected_indices
                else 0.0
            )
            relevance = candidate.score.final_preselection_score
            mmr = (profile.selection.mmr_lambda * relevance) - (
                (1.0 - profile.selection.mmr_lambda) * redundancy
            )
            return (
                -mmr,
                -relevance,
                -candidate.paper.updated_at.timestamp(),
                candidate.paper.arxiv_id,
            )

        chosen = min(pool, key=mmr_key)
        selected_indices.append(chosen)
        selected_types[eligible[chosen][1]] += 1
        selected_topics[eligible[chosen][2]] += 1

    return [
        SelectedPaper(
            paper=eligible[index][0].paper,
            score=eligible[index][0].score,
            rank=rank,
            recommendation_type=eligible[index][1],
            topic_key=eligible[index][2],
        )
        for rank, index in enumerate(selected_indices, start=1)
    ]
