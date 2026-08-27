"""Replay every gate verdict and score component for one paper.

When someone asks "why did this show up?" - or, more importantly, "why didn't
it?" - a command that prints the whole chain turns a debate into a lookup. It is
also how the rejection log gets audited without reading JSON by hand.
"""

from __future__ import annotations

from arxiv_digest.config import DomainClass, Settings
from arxiv_digest.domain_filter import (
    DisambiguationConfig,
    blocking_finding,
    evaluate_context_requirements,
    evaluate_hard_exclusions,
    paper_tokens,
)
from arxiv_digest.models import (
    CandidateSnapshot,
    DateWindow,
    GateRejection,
    GateStage,
    Paper,
)
from arxiv_digest.normalization import count_term
from arxiv_digest.ranking import rank_papers
from arxiv_digest.selection import recommendation_type, topic_key

_LABEL_WIDTH = 16


def _row(label: str, value: str) -> str:
    return f"  {label.ljust(_LABEL_WIDTH)}{value}"


def _header(paper: Paper, window: DateWindow) -> list[str]:
    return [
        f"{paper.arxiv_id}v{paper.version}  {paper.title}",
        _row("window", f"{window.start.date()} -> {window.end.date()}"),
        _row("categories", f"{', '.join(paper.categories)}  (primary {paper.primary_category})"),
        _row("submitted", f"{paper.published_at.date()}, updated {paper.updated_at.date()}"),
    ]


def _rejection_lines(rejection: GateRejection) -> list[str]:
    return [
        "",
        "GATES",
        _row(rejection.stage.value, rejection.reason),
        _row("rule", rejection.rule_id or "-"),
        "",
        f"REJECTED at the {rejection.stage.value} gate. It never reached ranking.",
    ]


def _gate_lines(
    paper: Paper,
    settings: Settings,
    disambiguation: DisambiguationConfig,
    penalized: bool,
) -> tuple[list[str], str | None]:
    """Return the gate report and the stage that rejected the paper, if any.

    The verdict is recomputed from the current configuration rather than read
    back from the snapshot, so this answers "what would the gates do now?" — the
    question actually being asked when a rule has just been edited.
    """
    domain = settings.group.domain
    rejected_at: str | None = None
    classification = domain.classify(paper.categories)
    if classification is DomainClass.INCLUDE:
        detail = f"in-field via {', '.join(domain.included(paper.categories))}"
    elif classification is DomainClass.SOFT:
        detail = f"adjacent via {', '.join(domain.soft(paper.categories))}"
    elif classification is DomainClass.EXCLUDE:
        detail = f"out of field via {', '.join(domain.excluded(paper.categories))}"
        rejected_at = GateStage.CATEGORY.value
    else:
        detail = "neither in-field nor adjacent"
        rejected_at = GateStage.CATEGORY.value
    lines = ["", "GATES", _row("category", f"{classification.value.upper()}  {detail}")]

    exclusion = evaluate_hard_exclusions(paper, disambiguation.hard_exclusions, domain)
    if exclusion is None:
        shielded = (
            " (in-field papers are never hard-rejected)"
            if domain.included(paper.categories)
            else ""
        )
        lines.append(
            _row(
                "hard rules",
                f"passed; {len(disambiguation.hard_exclusions)} evaluated{shielded}",
            )
        )
    else:
        lines.append(_row("hard rules", f"{exclusion.rule_id}: {exclusion.reason}"))
        rejected_at = rejected_at or GateStage.HARD_RULE.value

    findings = evaluate_context_requirements(paper, disambiguation.ambiguous_terms, domain)
    if not findings:
        lines.append(_row("context", "no ambiguous terms present"))
    for finding in findings:
        lines.append(_row("context", f"{finding.outcome.value}: {finding.describe()}"))
    if blocking_finding(findings) is not None:
        rejected_at = rejected_at or GateStage.CONTEXT.value

    tokens = paper_tokens(paper)
    matched_negative = [
        term for term in settings.profile.negative_terms if count_term(tokens, term)
    ]
    lines.append(
        _row("negative terms", ", ".join(matched_negative) if matched_negative else "none matched")
    )
    if matched_negative:
        rejected_at = rejected_at or GateStage.NEGATIVE_TERM.value
    if penalized:
        lines.append(_row("flagged", "an ambiguous term lacks condensed-matter context"))
    lines.append(_row("verdict", "REJECTED" if rejected_at else "passed"))
    return lines, rejected_at


def _score_lines(paper: Paper, settings: Settings, snapshot: CandidateSnapshot) -> list[str]:
    profile = settings.profile
    ranked = rank_papers(
        snapshot.papers,
        profile,
        snapshot.retrieval_window,
        {flag.arxiv_id for flag in snapshot.context_flags},
        settings.group.domain,
    )
    position = next(
        index for index, item in enumerate(ranked, start=1) if item.paper.arxiv_id == paper.arxiv_id
    )
    entry = ranked[position - 1]
    score = entry.score
    weights = profile.ranking_weights
    tier = recommendation_type(entry, profile)
    thresholds = profile.selection
    lines = [
        "",
        "SCORE",
        _row(
            "semantic",
            f"{score.semantic_relevance:.3f} x {weights.semantic_relevance:.4f}"
            f" = {score.semantic_relevance * weights.semantic_relevance:.3f}",
        ),
        _row(
            "keyword",
            f"{score.keyword_relevance:.3f} x {weights.keyword_relevance:.4f}"
            f" = {score.keyword_relevance * weights.keyword_relevance:.3f}",
        ),
        _row(
            "category",
            f"{score.category_relevance:.3f} x {weights.category_relevance:.4f}"
            f" = {score.category_relevance * weights.category_relevance:.3f}",
        ),
        _row("relevance", f"{score.relevance:.3f}"),
        _row("recency", f"{score.recency:.3f} (multiplier only)"),
    ]
    if score.context_penalty_applied:
        lines.append(_row("context penalty", f"x {profile.scoring.context_penalty:.2f}"))
    lines.extend(
        [
            _row("final", f"{score.final_preselection_score:.3f}"),
            _row("rank", f"{position} of {len(ranked)} ranked"),
            _row(
                "tier",
                f"{tier.value} (direct >= {thresholds.direct_min_score}, "
                f"adjacent >= {thresholds.adjacent_min_score}, "
                f"wildcard >= {thresholds.wildcard_min_score})"
                if tier is not None
                else f"rejected: below wildcard_min_score {thresholds.wildcard_min_score}",
            ),
            _row("topic", topic_key(entry, profile)),
            _row("terms", score.explanation),
        ]
    )
    return lines


def explain_paper(
    arxiv_id: str,
    snapshot: CandidateSnapshot,
    settings: Settings,
    disambiguation: DisambiguationConfig,
) -> list[str]:
    """Return every gate verdict and score component for one paper in one run.

    Raises ``LookupError`` when the identifier is absent from the snapshot
    entirely, which means it was never retrieved rather than rejected — a
    different problem, and worth saying so plainly.
    """
    canonical = arxiv_id.strip()
    rejection = next((item for item in snapshot.rejections if item.arxiv_id == canonical), None)
    if rejection is not None:
        paper = Paper.model_construct(
            arxiv_id=rejection.arxiv_id,
            version=1,
            title=rejection.title,
            authors=["-"],
            abstract="-",
            primary_category=rejection.primary_category,
            categories=list(rejection.categories),
            published_at=snapshot.retrieval_window.start,
            updated_at=snapshot.retrieval_window.start,
            abstract_url="",
            pdf_url="",
        )
        return [
            *_header(paper, snapshot.retrieval_window),
            *_rejection_lines(rejection),
        ]

    kept = next((item for item in snapshot.papers if item.arxiv_id == canonical), None)
    if kept is None:
        raise LookupError(
            f"{canonical} is not in this run: it was never retrieved, so no gate "
            "rejected it. Widen the queries or the window, or check the identifier."
        )
    penalized = any(flag.arxiv_id == canonical for flag in snapshot.context_flags)
    gates, rejected_at = _gate_lines(kept, settings, disambiguation, penalized)
    lines = [*_header(kept, snapshot.retrieval_window), *gates]
    if rejected_at is not None:
        # Scoring a paper the gates reject would invite the reader to weigh a
        # number that no longer decides anything.
        lines.extend(
            [
                "",
                f"REJECTED at the {rejected_at} gate. It does not reach ranking, "
                "so it has no score.",
            ]
        )
        return lines
    lines.extend(_score_lines(kept, settings, snapshot))
    return lines
