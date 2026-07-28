"""Core validated data models for retrieval and future digest stages."""

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ABSTRACT_SUMMARY_BASIS = (
    "Based only on the title, abstract, and arXiv metadata; the full paper was not read."
)


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
    """Machine-readable output of a Milestone 2 fetch."""

    run_id: str
    generated_at: datetime
    retrieval_window: DateWindow
    queries: list[QueryResult]
    records_retrieved: int = Field(ge=0)
    records_after_deduplication: int = Field(ge=0)
    papers: list[Paper]

    _generated_aware = field_validator("generated_at")(_require_aware)
