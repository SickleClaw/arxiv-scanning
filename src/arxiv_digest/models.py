"""Core validated data models for retrieval and future digest stages."""

import re
from datetime import datetime
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ABSTRACT_SUMMARY_BASIS = (
    "Based only on the title, abstract, and arXiv metadata; the full paper was not read."
)
DIGEST_SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and validates assignments."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    return value


class DateWindow(StrictModel):
    """Half-open UTC retrieval interval: start is inclusive and end is exclusive."""

    start: datetime
    end: datetime

    _start_aware = field_validator("start")(_require_aware)
    _end_aware = field_validator("end")(_require_aware)

    @model_validator(mode="after")
    def validate_order(self) -> "DateWindow":
        """Ensure the window has positive duration."""
        if self.end <= self.start:
            raise ValueError("retrieval window end must be after start")
        return self


class Paper(StrictModel):
    """Normalized metadata for one canonical arXiv paper version."""

    arxiv_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    abstract: str = Field(min_length=1)
    primary_category: str = Field(min_length=1)
    categories: list[str] = Field(min_length=1)
    published_at: datetime
    updated_at: datetime
    abstract_url: str
    pdf_url: str
    doi: str | None = None
    journal_reference: str | None = None

    _published_aware = field_validator("published_at")(_require_aware)
    _updated_aware = field_validator("updated_at")(_require_aware)

    @field_validator("title", "abstract")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        """Collapse whitespace while preserving Unicode text."""
        return re.sub(r"\s+", " ", value).strip()

    @field_validator("authors", "categories")
    @classmethod
    def normalize_string_lists(cls, values: list[str]) -> list[str]:
        """Normalize and de-duplicate ordered metadata lists."""
        normalized = [re.sub(r"\s+", " ", value).strip() for value in values]
        return list(dict.fromkeys(value for value in normalized if value))

    @model_validator(mode="after")
    def validate_dates_and_category(self) -> "Paper":
        """Reject chronologically invalid records and repair category inclusion."""
        if self.updated_at < self.published_at:
            raise ValueError("updated_at cannot precede published_at")
        if self.primary_category not in self.categories:
            self.categories.insert(0, self.primary_category)
        return self


class ScoreBreakdown(StrictModel):
    """Ranking components, each normalized to the inclusive range 0..1."""

    semantic_relevance: float = Field(ge=0.0, le=1.0)
    keyword_relevance: float = Field(ge=0.0, le=1.0)
    category_relevance: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    feedback_affinity: float = Field(ge=0.0, le=1.0)
    final_preselection_score: float = Field(ge=0.0, le=1.0)
    explanation: str


class PaperSummary(StrictModel):
    """Validated abstract-grounded summary with concise, explicit scope limits."""

    one_sentence_takeaway: str = Field(min_length=1)
    brief_summary: str = Field(min_length=1)
    why_relevant: str = Field(min_length=1)
    methods_or_systems: list[str] = Field(min_length=1)
    limitations: str = Field(min_length=1)
    summary_basis: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator(
        "one_sentence_takeaway",
        "brief_summary",
        "why_relevant",
        "limitations",
        "summary_basis",
    )
    @classmethod
    def normalize_summary_text(cls, value: str) -> str:
        """Collapse whitespace in provider output before applying length checks."""
        return re.sub(r"\s+", " ", value).strip()

    @field_validator("methods_or_systems")
    @classmethod
    def normalize_methods(cls, values: list[str]) -> list[str]:
        """Normalize and de-duplicate the supplied metadata-grounded method labels."""
        normalized = [re.sub(r"\s+", " ", value).strip() for value in values]
        result = list(dict.fromkeys(value for value in normalized if value))
        if not result:
            raise ValueError("methods_or_systems must contain at least one nonblank value")
        return result

    @field_validator("one_sentence_takeaway")
    @classmethod
    def limit_takeaway(cls, value: str) -> str:
        """Enforce the report's 35-word takeaway ceiling."""
        if len(value.split()) > 35:
            raise ValueError("one_sentence_takeaway must contain no more than 35 words")
        return value

    @field_validator("brief_summary")
    @classmethod
    def limit_brief_summary(cls, value: str) -> str:
        """Reject verbose provider output well beyond the 80-130 word target."""
        if len(value.split()) > 150:
            raise ValueError("brief_summary must contain no more than 150 words")
        return value

    @field_validator("why_relevant")
    @classmethod
    def limit_relevance(cls, value: str) -> str:
        """Keep the relevance explanation close to its 30-70 word target."""
        if len(value.split()) > 80:
            raise ValueError("why_relevant must contain no more than 80 words")
        return value

    @field_validator("limitations")
    @classmethod
    def limit_limitations(cls, value: str) -> str:
        """Keep the limitations statement concise."""
        if len(value.split()) > 45:
            raise ValueError("limitations must contain no more than 45 words")
        return value

    @field_validator("summary_basis")
    @classmethod
    def require_abstract_basis(cls, value: str) -> str:
        """Prevent a provider from implying that the full paper was inspected."""
        if value != ABSTRACT_SUMMARY_BASIS:
            raise ValueError(f"summary_basis must be exactly: {ABSTRACT_SUMMARY_BASIS}")
        return value

    @model_validator(mode="after")
    def reject_inspection_claims(self) -> "PaperSummary":
        """Reject common phrases that imply evidence outside the supplied metadata."""
        narrative = " ".join(
            (
                self.one_sentence_takeaway,
                self.brief_summary,
                self.why_relevant,
                self.limitations,
            )
        ).casefold()
        forbidden = (
            r"\b(?:we|i) (?:read|reviewed|examined|inspected) the (?:full )?paper\b",
            r"\b(?:the )?(?:figures|equations|appendices) "
            r"(?:show|demonstrate|confirm|reveal)\b",
        )
        if any(re.search(pattern, narrative) for pattern in forbidden):
            raise ValueError("summary must not imply inspection beyond the abstract and metadata")
        return self


class GateStage(StrEnum):
    """Which deterministic gate removed a paper."""

    CATEGORY = "category"
    HARD_RULE = "hard_rule"
    CONTEXT = "context"
    NEGATIVE_TERM = "negative_term"
    HISTORY = "history"


class GateRejection(StrictModel):
    """One paper removed before scoring, with the evidence that removed it.

    No paper leaves the pipeline silently. Over-filtering is the mirror image of
    the bug this phase fixes and it is far harder to notice, so the record of
    what was discarded is the only thing that makes it visible.
    """

    arxiv_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    primary_category: str = Field(min_length=1)
    categories: list[str] = Field(min_length=1)
    stage: GateStage
    reason: str = Field(min_length=1)
    rule_id: str | None = None


class ContextFlag(StrictModel):
    """An ambiguous term a kept paper used without supporting context."""

    arxiv_id: str = Field(min_length=1)
    term: str = Field(min_length=1)
    detail: str = Field(min_length=1)


class RecommendationType(StrEnum):
    """Intended role of a paper in the final diversified list."""

    DIRECT = "direct"
    ADJACENT = "adjacent"
    WILDCARD = "wildcard"


class Recommendation(StrictModel):
    """Report record combining metadata, scores, and a grounded summary."""

    paper: Paper
    score: ScoreBreakdown
    summary: PaperSummary
    rank: int = Field(ge=1)
    recommendation_type: RecommendationType


class DigestArtifact(StrictModel):
    """Versioned canonical data used by reports and the local dashboard."""

    schema_version: str
    run_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    generated_at: datetime
    timezone: str = Field(min_length=1)
    retrieval_window: DateWindow
    records_retrieved: int = Field(ge=0)
    records_after_deduplication: int = Field(ge=0)
    records_ranked: int = Field(ge=0)
    records_selected: int = Field(ge=0)
    summary_mode: str = Field(min_length=1)
    summary_fallbacks: int = Field(ge=0)
    summary_basis: str = Field(min_length=1)
    profile_name: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    profile_hash: str = Field(min_length=1)
    recommendations: list[Recommendation]
    near_misses: list["RankedPaper"] = Field(max_length=5)
    ranked_candidates: list["RankedPaper"]

    _generated_aware = field_validator("generated_at")(_require_aware)

    @field_validator("timezone")
    @classmethod
    def require_known_timezone(cls, value: str) -> str:
        """Ensure history and report dates can be displayed consistently."""
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value

    @field_validator("schema_version")
    @classmethod
    def require_supported_schema(cls, value: str) -> str:
        """Reject artifacts created for an incompatible dashboard schema."""
        if value != DIGEST_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {DIGEST_SCHEMA_VERSION!r}, received {value!r}"
            )
        return value

    @field_validator("summary_basis")
    @classmethod
    def require_summary_basis(cls, value: str) -> str:
        """Keep the canonical artifact explicit about its abstract-only evidence."""
        if value != ABSTRACT_SUMMARY_BASIS:
            raise ValueError(f"summary_basis must be exactly: {ABSTRACT_SUMMARY_BASIS}")
        return value

    @model_validator(mode="after")
    def validate_contents(self) -> "DigestArtifact":
        """Validate counts and relationships between selected and ranked records."""
        if self.records_after_deduplication > self.records_retrieved:
            raise ValueError("deduplicated record count cannot exceed retrieved record count")
        if self.records_ranked != len(self.ranked_candidates):
            raise ValueError("records_ranked must equal the ranked candidate count")
        if self.records_ranked > self.records_after_deduplication:
            raise ValueError("ranked record count cannot exceed deduplicated record count")
        if self.records_selected != len(self.recommendations):
            raise ValueError("records_selected must equal the recommendation count")
        if self.summary_fallbacks > self.records_selected:
            raise ValueError("summary_fallbacks cannot exceed records_selected")
        ranked_by_id = {item.paper.arxiv_id: item for item in self.ranked_candidates}
        ranked_ids = set(ranked_by_id)
        if len(ranked_ids) != len(self.ranked_candidates):
            raise ValueError("ranked_candidates must contain unique canonical arXiv IDs")
        selected_ids = [item.paper.arxiv_id for item in self.recommendations]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("recommendations must contain unique canonical arXiv IDs")
        if not set(selected_ids).issubset(ranked_ids):
            raise ValueError("every recommendation must be present in ranked_candidates")
        if any(
            item.paper != ranked_by_id[item.paper.arxiv_id].paper
            or item.score != ranked_by_id[item.paper.arxiv_id].score
            for item in self.recommendations
        ):
            raise ValueError("recommendations must match their ranked candidate records")
        near_miss_ids = [item.paper.arxiv_id for item in self.near_misses]
        if len(near_miss_ids) != len(set(near_miss_ids)):
            raise ValueError("near_misses must contain unique canonical arXiv IDs")
        if set(near_miss_ids) & set(selected_ids):
            raise ValueError("near_misses cannot also be recommendations")
        if not set(near_miss_ids).issubset(ranked_ids):
            raise ValueError("every near miss must be present in ranked_candidates")
        if any(item != ranked_by_id[item.paper.arxiv_id] for item in self.near_misses):
            raise ValueError("near_misses must match their ranked candidate records")
        return self


class RankedPaper(StrictModel):
    """A candidate paired with its deterministic preselection score."""

    paper: Paper
    score: ScoreBreakdown


class SelectedPaper(StrictModel):
    """A ranked selection before summary generation."""

    paper: Paper
    score: ScoreBreakdown
    rank: int = Field(ge=1)
    recommendation_type: RecommendationType
    topic_key: str


class RankedSnapshot(StrictModel):
    """Machine-readable result of history filtering and offline ranking."""

    run_id: str
    source_run_id: str
    generated_at: datetime
    retrieval_window: DateWindow
    records_before_history: int = Field(ge=0)
    records_after_history: int = Field(ge=0)
    ranked_papers: list[RankedPaper]
    rejections: list[GateRejection] = Field(default_factory=list)

    _generated_aware = field_validator("generated_at")(_require_aware)


class SelectionSnapshot(StrictModel):
    """Machine-readable diversity-aware selection from a ranked snapshot."""

    run_id: str
    source_rank_run_id: str
    generated_at: datetime
    retrieval_window: DateWindow
    requested_limit: int = Field(ge=1)
    selected: list[SelectedPaper]

    _generated_aware = field_validator("generated_at")(_require_aware)


class HistoryRecord(StrictModel):
    """One append-only recommendation-history record."""

    run_id: str
    run_timestamp: datetime
    retrieval_window: DateWindow
    arxiv_id: str
    version: int = Field(ge=1)
    paper_updated_at: datetime
    rank: int = Field(ge=1)
    recommendation_type: RecommendationType
    final_score: float = Field(ge=0.0, le=1.0)
    report_path: str | None = None

    _run_aware = field_validator("run_timestamp")(_require_aware)
    _paper_updated_aware = field_validator("paper_updated_at")(_require_aware)


class QueryResult(StrictModel):
    """Retrieval counts for one configured broad query."""

    name: str
    records_received: int = Field(ge=0)


class CandidateSnapshot(StrictModel):
    """Machine-readable output of a fetch, after deduplication and gating.

    ``papers`` holds what survived the gates; ``records_after_deduplication``
    still reports the pre-gate count, so the two together say how much the gates
    removed. Everything they removed is in ``rejections``.
    """

    run_id: str
    generated_at: datetime
    retrieval_window: DateWindow
    queries: list[QueryResult]
    records_retrieved: int = Field(ge=0)
    records_after_deduplication: int = Field(ge=0)
    papers: list[Paper]
    rejections: list[GateRejection] = Field(default_factory=list)
    context_flags: list[ContextFlag] = Field(default_factory=list)

    _generated_aware = field_validator("generated_at")(_require_aware)
