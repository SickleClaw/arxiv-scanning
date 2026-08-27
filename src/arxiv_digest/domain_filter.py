"""Deterministic domain gating: hard exclusion rules and ambiguous-term context.

Everything here is auditable by construction. No model, no similarity, no
network: a paper is rejected only by a named rule, and the rule that rejected it
travels with the rejection so ``arxiv-digest explain`` can replay the decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from arxiv_digest.config import (
    DomainClass,
    DomainConfig,
    category_matches,
    read_yaml_mapping,
)
from arxiv_digest.exceptions import ConfigurationError
from arxiv_digest.models import (
    ContextFlag,
    GateRejection,
    GateStage,
    Paper,
    StrictModel,
)
from arxiv_digest.normalization import count_term, tokenize

ConditionField = Literal["text", "categories", "primary_category"]


class RuleCondition(StrictModel):
    """One clause of a hard exclusion rule.

    ``text`` matches whole terms across the title and abstract; the category
    fields match arXiv category patterns, so ``cond-mat.*`` behaves as it does
    everywhere else.
    """

    field: ConditionField
    any_of: list[str] = Field(default_factory=list)
    none_of: list[str] = Field(default_factory=list)

    @field_validator("any_of", "none_of")
    @classmethod
    def reject_blank_values(cls, values: list[str]) -> list[str]:
        """Prevent a blank value from matching everything or nothing silently."""
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("condition values cannot be blank")
        return stripped

    @model_validator(mode="after")
    def require_exactly_one_operator(self) -> RuleCondition:
        """Require each condition to state one test, so intent is never ambiguous."""
        if bool(self.any_of) == bool(self.none_of):
            raise ValueError("each condition must set exactly one of any_of or none_of")
        return self

    def holds(self, paper: Paper, tokens: Sequence[str]) -> bool:
        """Evaluate this condition against one paper."""
        if self.field == "text":
            if self.any_of:
                return any(count_term(tokens, value) for value in self.any_of)
            return not any(count_term(tokens, value) for value in self.none_of)
        categories = paper.categories if self.field == "categories" else [paper.primary_category]
        if self.any_of:
            return any(
                category_matches(pattern, category)
                for pattern in self.any_of
                for category in categories
            )
        return not any(
            category_matches(pattern, category)
            for pattern in self.none_of
            for category in categories
        )


class HardExclusionRule(StrictModel):
    """A narrow, high-precision rejection rule that must be individually defensible."""

    id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    when_all: list[RuleCondition] = Field(min_length=1)

    def holds(self, paper: Paper, tokens: Sequence[str]) -> bool:
        """Return whether every clause of this rule holds for the paper."""
        return all(condition.holds(paper, tokens) for condition in self.when_all)


class ContextDefault(StrEnum):
    """What to do when an ambiguous term appears with neither context nor blocker."""

    ACCEPT = "accept"
    PENALIZE = "penalize"
    REJECT = "reject"


class AmbiguousTerm(StrictModel):
    """Context requirements for a term whose meaning depends on its field."""

    requires_any_of: list[str] = Field(min_length=1)
    blocked_by_any_of: list[str] = Field(default_factory=list)
    default_when_neither: ContextDefault = ContextDefault.PENALIZE

    @field_validator("requires_any_of", "blocked_by_any_of")
    @classmethod
    def reject_blank_values(cls, values: list[str]) -> list[str]:
        """Reject blank context terms, which would match unpredictably."""
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("context terms cannot be blank")
        return stripped


class DisambiguationConfig(StrictModel):
    """Hard exclusion rules and ambiguous-term context requirements."""

    hard_exclusions: list[HardExclusionRule] = Field(default_factory=list)
    ambiguous_terms: dict[str, AmbiguousTerm] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_unique_rule_ids(self) -> DisambiguationConfig:
        """Keep rule ids unique so a logged rejection names exactly one rule."""
        identifiers = [rule.id for rule in self.hard_exclusions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("hard exclusion rule ids must be unique")
        return self


def load_disambiguation(
    path: Path = Path("config/disambiguation.yaml"),
) -> DisambiguationConfig:
    """Load and validate the disambiguation rules."""
    try:
        return DisambiguationConfig.model_validate(read_yaml_mapping(path))
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid disambiguation configuration in {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class HardExclusion:
    """The named rule that rejected a paper."""

    rule_id: str
    reason: str


def paper_tokens(paper: Paper) -> tuple[str, ...]:
    """Tokenize the evidence a rule may look at: the title and the abstract."""
    return tokenize(f"{paper.title} {paper.abstract}")


def evaluate_hard_exclusions(
    paper: Paper,
    rules: Sequence[HardExclusionRule],
    domain: DomainConfig,
) -> HardExclusion | None:
    """Return the first rule that rejects this paper, or None if it survives.

    A paper carrying one of the group's in-field categories is never rejected
    here, whatever it contains. The guard is unconditional and lives in this
    function rather than in the rules, so no edit to the rule file can remove
    it: cross-listed condensed-matter work is safe from every hard rule, present
    and future. Adjacent (soft) categories get no such protection — they have to
    earn their place.
    """
    if domain.included(paper.categories):
        return None
    tokens = paper_tokens(paper)
    for rule in rules:
        if rule.holds(paper, tokens):
            return HardExclusion(rule_id=rule.id, reason=rule.reason)
    return None


class ContextOutcome(StrEnum):
    """The verdict on one ambiguous term found in a paper."""

    SATISFIED = "satisfied"
    """Required condensed-matter context was present."""

    PENALIZED = "penalized"
    """Neither context nor blocker appeared; flagged for scoring to weigh."""

    ACCEPTED = "accepted"
    """Neither appeared, and the term is configured to accept that."""

    BLOCKED = "blocked"
    """A blocking term appeared, and the paper is not in-field."""

    @property
    def rejects(self) -> bool:
        """Whether this outcome removes the paper rather than flagging it."""
        return self is ContextOutcome.BLOCKED


@dataclass(frozen=True, slots=True)
class ContextFinding:
    """One ambiguous term found in a paper, with the evidence for its verdict."""

    term: str
    outcome: ContextOutcome
    matched_context: tuple[str, ...] = ()
    matched_blockers: tuple[str, ...] = ()

    def describe(self) -> str:
        """Render the finding for a rejection log or an explain listing."""
        if self.outcome is ContextOutcome.BLOCKED:
            return f"{self.term!r} appears with {', '.join(self.matched_blockers)}"
        if self.outcome is ContextOutcome.SATISFIED:
            return f"{self.term!r} supported by {', '.join(self.matched_context)}"
        return f"{self.term!r} appears without condensed-matter context"


def evaluate_context_requirements(
    paper: Paper,
    ambiguous_terms: dict[str, AmbiguousTerm],
    domain: DomainConfig,
) -> tuple[ContextFinding, ...]:
    """Judge every configured ambiguous term that actually appears in the paper.

    Absence of positive context is weak evidence, not a verdict: abstracts are
    short and an author may simply not use the expected vocabulary. The default
    is therefore to flag for penalty and let scoring decide.

    Blocking is subject to the same in-field guard as the hard rules. A paper
    cross-listed into the group's field is flagged, never removed, no matter
    which blocking words it happens to contain.
    """
    tokens = paper_tokens(paper)
    in_field = bool(domain.included(paper.categories))
    findings: list[ContextFinding] = []
    for term, configured in ambiguous_terms.items():
        if not count_term(tokens, term):
            continue
        blockers = tuple(
            value for value in configured.blocked_by_any_of if count_term(tokens, value)
        )
        context = tuple(value for value in configured.requires_any_of if count_term(tokens, value))
        if blockers and not in_field:
            outcome = ContextOutcome.BLOCKED
        elif context:
            outcome = ContextOutcome.SATISFIED
        elif blockers:
            # In-field work is never removed here; the conflict is still recorded.
            outcome = ContextOutcome.PENALIZED
        elif configured.default_when_neither is ContextDefault.ACCEPT:
            outcome = ContextOutcome.ACCEPTED
        elif configured.default_when_neither is ContextDefault.REJECT and not in_field:
            outcome = ContextOutcome.BLOCKED
        else:
            outcome = ContextOutcome.PENALIZED
        findings.append(
            ContextFinding(
                term=term,
                outcome=outcome,
                matched_context=context,
                matched_blockers=blockers,
            )
        )
    return tuple(findings)


def blocking_finding(findings: Sequence[ContextFinding]) -> ContextFinding | None:
    """Return the first finding that removes the paper, if any."""
    return next((finding for finding in findings if finding.outcome.rejects), None)


@dataclass(frozen=True, slots=True)
class GateResult:
    """What the deterministic gates kept, removed, and flagged."""

    kept: list[Paper]
    rejections: list[GateRejection]
    context_flags: list[ContextFlag]


def _rejection(
    paper: Paper, stage: GateStage, reason: str, rule_id: str | None = None
) -> GateRejection:
    return GateRejection(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        primary_category=paper.primary_category,
        categories=list(paper.categories),
        stage=stage,
        reason=reason,
        rule_id=rule_id,
    )


def gate_papers(
    papers: Sequence[Paper],
    *,
    domain: DomainConfig,
    disambiguation: DisambiguationConfig,
    negative_terms: Sequence[str] = (),
) -> GateResult:
    """Run the deterministic gates in cost order, recording every rejection.

    Category first because it is free and removes the most, then the hard rules,
    then the ambiguous-term context requirements, then the profile's own
    negative terms. Gating happens before scoring, not as part of it: a soft
    penalty inside a weighted sum can always be outvoted, and that is precisely
    how off-domain papers reached the digest before.

    History exclusion is deliberately not here. It depends on when the run
    happens and on a file that grows every week, and keeping it out leaves the
    retrieval snapshot reproducible.
    """
    kept: list[Paper] = []
    rejections: list[GateRejection] = []
    flags: list[ContextFlag] = []

    for paper in papers:
        classification = domain.classify(paper.categories)
        if classification is DomainClass.EXCLUDE:
            rejections.append(
                _rejection(
                    paper,
                    GateStage.CATEGORY,
                    f"categories {', '.join(domain.excluded(paper.categories))} are out of field, "
                    "with no in-field cross-list",
                )
            )
            continue
        if classification is DomainClass.UNKNOWN:
            rejections.append(
                _rejection(
                    paper,
                    GateStage.CATEGORY,
                    f"categories {', '.join(paper.categories)} are neither in-field nor adjacent",
                )
            )
            continue

        exclusion = evaluate_hard_exclusions(paper, disambiguation.hard_exclusions, domain)
        if exclusion is not None:
            rejections.append(
                _rejection(paper, GateStage.HARD_RULE, exclusion.reason, exclusion.rule_id)
            )
            continue

        findings = evaluate_context_requirements(paper, disambiguation.ambiguous_terms, domain)
        blocked = blocking_finding(findings)
        if blocked is not None:
            rejections.append(
                _rejection(
                    paper,
                    GateStage.CONTEXT,
                    f"ambiguous term {blocked.describe()}",
                    blocked.term,
                )
            )
            continue

        tokens = paper_tokens(paper)
        matched_negative = [term for term in negative_terms if count_term(tokens, term)]
        if matched_negative:
            rejections.append(
                _rejection(
                    paper,
                    GateStage.NEGATIVE_TERM,
                    f"matched negative profile terms: {', '.join(matched_negative)}",
                )
            )
            continue

        kept.append(paper)
        flags.extend(
            ContextFlag(arxiv_id=paper.arxiv_id, term=finding.term, detail=finding.describe())
            for finding in findings
            if finding.outcome is ContextOutcome.PENALIZED
        )

    return GateResult(kept=kept, rejections=rejections, context_flags=flags)
