"""Core validated data models for retrieval and future digest stages."""

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    """Future ranking components, each normalized to the inclusive range 0..1."""

    semantic_relevance: float = Field(ge=0.0, le=1.0)
    keyword_relevance: float = Field(ge=0.0, le=1.0)
    category_relevance: float = Field(ge=0.0, le=1.0)
    recency: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    feedback_affinity: float = Field(ge=0.0, le=1.0)
    final_preselection_score: float = Field(ge=0.0, le=1.0)
    explanation: str


class PaperSummary(StrictModel):
    """Validated abstract-grounded summary shape for a later milestone."""

    one_sentence_takeaway: str
    brief_summary: str
    why_relevant: str
    methods_or_systems: list[str]
    limitations: str
    summary_basis: str
    confidence: float = Field(ge=0.0, le=1.0)


class RecommendationType(StrEnum):
    """Intended role of a paper in the final diversified list."""

    DIRECT = "direct"
    ADJACENT = "adjacent"
    WILDCARD = "wildcard"


class Recommendation(StrictModel):
    """Future report record combining metadata, scores, and a grounded summary."""

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
    """A ranked Milestone 3 selection without a not-yet-generated summary."""

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
