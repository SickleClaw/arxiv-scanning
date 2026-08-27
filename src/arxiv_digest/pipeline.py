"""Pipeline orchestration through abstract-grounded Milestone 4 reports."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ValidationError

from arxiv_digest.arxiv_client import ArxivClient
from arxiv_digest.config import Settings
from arxiv_digest.domain_filter import (
    DisambiguationConfig,
    gate_papers,
    load_disambiguation,
)
from arxiv_digest.exceptions import ArxivDigestError
from arxiv_digest.history import (
    append_history,
    filter_recent_history,
    recommended_ids,
    records_for_selection,
)
from arxiv_digest.models import (
    CandidateSnapshot,
    DateWindow,
    DigestArtifact,
    GateRejection,
    GateStage,
    HistoryRecord,
    RankedSnapshot,
    Recommendation,
    SelectionSnapshot,
)
from arxiv_digest.normalization import deduplicate_papers
from arxiv_digest.ranking import rank_papers
from arxiv_digest.reporting import ReportPaths, build_digest_artifact, render_reports
from arxiv_digest.selection import select_diverse
from arxiv_digest.snapshot import create_snapshot
from arxiv_digest.summarization import (
    SummaryProvider,
    summarize_selection,
    summary_provider_name,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Milestone3Result:
    """Validated outputs and counts from one ranking-and-selection run."""

    ranked: RankedSnapshot
    selection: SelectionSnapshot
    history_excluded: int
    history_appended: int


@dataclass(frozen=True, slots=True)
class Milestone4Result:
    """Ranking, selection, summaries, reports, and post-report history counts."""

    ranked: RankedSnapshot
    selection: SelectionSnapshot
    recommendations: list[Recommendation]
    digest: DigestArtifact
    reports: ReportPaths
    history_excluded: int
    history_appended: int
    summary_fallbacks: int


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _rank_run_id(snapshot: CandidateSnapshot, settings: Settings) -> str:
    return _stable_id(
        "rank",
        snapshot.run_id,
        settings.profile.model_dump_json(),
    )


def _selection_run_id(rank_run_id: str, settings: Settings, limit: int) -> str:
    return _stable_id(
        "select",
        rank_run_id,
        str(limit),
        settings.profile.recommendation_mix.model_dump_json(),
        settings.profile.selection.model_dump_json(),
        str(settings.profile.max_papers_per_topic),
    )


def retrieve_candidates(
    settings: Settings,
    window: DateWindow,
    *,
    now: datetime,
    client: ArxivClient | None = None,
    disambiguation: DisambiguationConfig | None = None,
) -> CandidateSnapshot:
    """Retrieve, deduplicate, and gate a candidate snapshot with injectable I/O."""
    started = time.perf_counter()
    if client is None:
        with ArxivClient(settings.app.arxiv) as owned_client:
            result = owned_client.fetch(
                settings.profile.queries,
                window,
                settings.profile.max_candidate_count,
                settings.group.domain,
            )
    else:
        result = client.fetch(
            settings.profile.queries,
            window,
            settings.profile.max_candidate_count,
            settings.group.domain,
        )
    unique = deduplicate_papers(result.papers)
    rules = disambiguation if disambiguation is not None else load_disambiguation()
    gated = gate_papers(
        unique,
        domain=settings.group.domain,
        disambiguation=rules,
        negative_terms=settings.profile.negative_terms,
    )
    # Gate before truncating: off-domain papers must not consume the candidate
    # budget that relevant ones are competing for.
    kept = gated.kept[: settings.profile.max_candidate_count]
    snapshot = create_snapshot(
        window=window,
        profile_version=settings.profile.version,
        generated_at=now,
        query_results=result.query_results,
        raw_papers=result.papers,
        deduplicated_papers=unique,
        kept_papers=kept,
        rejections=gated.rejections,
        context_flags=[
            flag for flag in gated.context_flags if flag.arxiv_id in {p.arxiv_id for p in kept}
        ],
    )
    LOGGER.info(
        "run_id=%s stage=retrieve raw=%d deduplicated=%d gated_out=%d kept=%d elapsed_seconds=%.3f",
        snapshot.run_id,
        len(result.papers),
        len(unique),
        len(gated.rejections),
        len(kept),
        time.perf_counter() - started,
    )
    for rejection in gated.rejections:
        LOGGER.info(
            "run_id=%s stage=gate rejected=%s gate=%s rule=%s reason=%s",
            snapshot.run_id,
            rejection.arxiv_id,
            rejection.stage,
            rejection.rule_id or "-",
            rejection.reason,
        )
    return snapshot


def rank_snapshot(
    snapshot: CandidateSnapshot,
    settings: Settings,
    history: list[HistoryRecord],
    *,
    now: datetime,
) -> tuple[RankedSnapshot, int]:
    """Apply recent-history rules, hybrid ranking, and deterministic stable ordering."""
    started = time.perf_counter()
    eligible, excluded = filter_recent_history(
        snapshot.papers,
        history,
        now=now,
        exclusion_days=settings.profile.history_exclusion_days,
        allow_updated_resurfacing=settings.profile.allow_updated_resurfacing,
    )
    eligible_ids = {paper.arxiv_id for paper in eligible}
    rejections = [
        *snapshot.rejections,
        *(
            GateRejection(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                primary_category=paper.primary_category,
                categories=list(paper.categories),
                stage=GateStage.HISTORY,
                reason=(
                    f"recommended within the last {settings.profile.history_exclusion_days} days"
                ),
            )
            for paper in snapshot.papers
            if paper.arxiv_id not in eligible_ids
        ),
    ]
    ranked = rank_papers(
        eligible,
        settings.profile,
        snapshot.retrieval_window,
        recommended_ids(history),
    )
    run_id = _rank_run_id(snapshot, settings)
    result = RankedSnapshot(
        run_id=run_id,
        source_run_id=snapshot.run_id,
        generated_at=now,
        retrieval_window=snapshot.retrieval_window,
        records_before_history=len(snapshot.papers),
        records_after_history=len(eligible),
        ranked_papers=ranked,
        rejections=rejections,
    )
    LOGGER.info(
        "run_id=%s stage=rank before_history=%d excluded=%d ranked=%d elapsed_seconds=%.3f",
        run_id,
        len(snapshot.papers),
        excluded,
        len(ranked),
        time.perf_counter() - started,
    )
    return result, excluded


def select_snapshot(
    ranked: RankedSnapshot,
    settings: Settings,
    *,
    now: datetime,
    limit: int,
) -> SelectionSnapshot:
    """Create a deterministic MMR selection snapshot from ranked candidates."""
    started = time.perf_counter()
    selected = select_diverse(ranked.ranked_papers, settings.profile, limit=limit)
    run_id = _selection_run_id(ranked.run_id, settings, limit)
    result = SelectionSnapshot(
        run_id=run_id,
        source_rank_run_id=ranked.run_id,
        generated_at=now,
        retrieval_window=ranked.retrieval_window,
        requested_limit=limit,
        selected=selected,
    )
    LOGGER.info(
        "run_id=%s stage=select candidates=%d selected=%d elapsed_seconds=%.3f",
        run_id,
        len(ranked.ranked_papers),
        len(selected),
        time.perf_counter() - started,
    )
    return result


def run_milestone3(
    snapshot: CandidateSnapshot,
    settings: Settings,
    history: list[HistoryRecord],
    *,
    now: datetime,
    limit: int,
    persist_history: bool,
) -> Milestone3Result:
    """Rank, select, and optionally persist idempotent local history."""
    expected_rank_id = _rank_run_id(snapshot, settings)
    expected_selection_id = _selection_run_id(expected_rank_id, settings, limit)
    effective_history = [record for record in history if record.run_id != expected_selection_id]
    ranked, excluded = rank_snapshot(snapshot, settings, effective_history, now=now)
    selection = select_snapshot(ranked, settings, now=now, limit=limit)
    appended = 0
    if persist_history:
        records = records_for_selection(
            run_id=selection.run_id,
            run_timestamp=now,
            window=selection.retrieval_window,
            selected=selection.selected,
        )
        appended = append_history(settings.app.paths.history_file, records)
    return Milestone3Result(
        ranked=ranked,
        selection=selection,
        history_excluded=excluded,
        history_appended=appended,
    )


def run_milestone4(
    snapshot: CandidateSnapshot,
    settings: Settings,
    history: list[HistoryRecord],
    provider: SummaryProvider,
    *,
    now: datetime,
    limit: int,
    persist_history: bool,
) -> Milestone4Result:
    """Rank, summarize, render, then optionally persist history after report success."""
    started = time.perf_counter()
    milestone3 = run_milestone3(
        snapshot,
        settings,
        history,
        now=now,
        limit=limit,
        persist_history=False,
    )
    summaries = summarize_selection(
        milestone3.selection.selected,
        settings.profile,
        provider,
        max_provider_papers=settings.app.summarization.max_provider_papers,
    )
    digest = build_digest_artifact(
        snapshot=snapshot,
        ranked=milestone3.ranked,
        recommendations=summaries.recommendations,
        profile=settings.profile,
        run_id=milestone3.selection.run_id,
        generated_at=now,
        timezone=settings.app.reporting.timezone,
        summary_mode=summary_provider_name(provider),
        summary_fallbacks=summaries.fallback_count,
        near_miss_limit=settings.app.reporting.near_miss_limit,
    )
    reports = render_reports(
        digest,
        templates_dir=settings.app.paths.templates_dir,
        reports_dir=settings.app.paths.reports_dir,
    )
    appended = 0
    if persist_history:
        records = records_for_selection(
            run_id=milestone3.selection.run_id,
            run_timestamp=now,
            window=milestone3.selection.retrieval_window,
            selected=milestone3.selection.selected,
            report_path=str(reports.markdown),
        )
        appended = append_history(settings.app.paths.history_file, records)
    LOGGER.info(
        "run_id=%s stage=report summarized=%d fallbacks=%d markdown=%s html=%s "
        "elapsed_seconds=%.3f",
        milestone3.selection.run_id,
        len(summaries.recommendations),
        summaries.fallback_count,
        reports.markdown,
        reports.html,
        time.perf_counter() - started,
    )
    return Milestone4Result(
        ranked=milestone3.ranked,
        selection=milestone3.selection,
        recommendations=summaries.recommendations,
        digest=digest,
        reports=reports,
        history_excluded=milestone3.history_excluded,
        history_appended=appended,
        summary_fallbacks=summaries.fallback_count,
    )


def load_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    """Load one validated JSON artifact with an actionable domain error."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArxivDigestError(f"Cannot read JSON artifact {path}: {exc}") from exc
    try:
        return model_type.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise ArxivDigestError(f"Invalid JSON artifact {path}: {exc}") from exc


def write_model(model: BaseModel, path: Path) -> Path:
    """Atomically persist a validated JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        raise ArxivDigestError(f"Cannot write JSON artifact {path}: {exc}") from exc
    return path


def _dated_path(data_dir: Path, prefix: str, window: DateWindow) -> Path:
    inclusive_end = (window.end - timedelta(microseconds=1)).date().isoformat()
    return data_dir / f"{prefix}-{window.start.date().isoformat()}-{inclusive_end}.json"


def default_ranked_path(data_dir: Path, window: DateWindow) -> Path:
    """Return the stable same-window ranked snapshot path."""
    return _dated_path(data_dir, "ranked", window)


def default_selection_path(data_dir: Path, window: DateWindow) -> Path:
    """Return the stable same-window selection snapshot path."""
    return _dated_path(data_dir, "selection", window)
