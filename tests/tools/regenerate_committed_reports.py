"""Rebuild the committed digest reports under the current artifact schema.

DIGEST_SCHEMA_VERSION is validated strictly — a mismatch raises rather than
degrading — so a change to ScoreBreakdown makes every stored report unreadable.
With only two committed runs, regenerating them is cheaper and cleaner than
carrying a back-compatibility loader forever.

Usage::

    python tests/tools/regenerate_committed_reports.py

Each report is rebuilt from the papers of its own window, run through the
current gates, ranking, selection, and offline summaries. Recommendation
history is rewritten to match, since the run ids move with the profile hash.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from arxiv_digest.config import Settings, load_settings
from arxiv_digest.domain_filter import gate_papers, load_disambiguation
from arxiv_digest.history import records_for_selection
from arxiv_digest.models import (
    CandidateSnapshot,
    DateWindow,
    DigestArtifact,
    HistoryRecord,
    Paper,
    RankedSnapshot,
)
from arxiv_digest.pipeline import _rank_run_id, _selection_run_id
from arxiv_digest.ranking import rank_papers
from arxiv_digest.reporting import build_digest_artifact, render_reports
from arxiv_digest.selection import select_diverse
from arxiv_digest.snapshot import create_snapshot
from arxiv_digest.summarization import DeterministicSummaryProvider, summarize_selection

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
SELECTION_LIMIT = 10


def regenerate(path: Path, settings: Settings) -> tuple[DigestArtifact, list[HistoryRecord]]:
    """Rebuild one committed report and return it with its history records."""
    stored = json.loads(path.read_text(encoding="utf-8"))
    window = DateWindow.model_validate(stored["retrieval_window"])
    generated_at = datetime.fromisoformat(stored["generated_at"]).astimezone(UTC)
    timezone = stored["timezone"]

    papers = [Paper.model_validate(item["paper"]) for item in stored["ranked_candidates"]]
    retained = DATA / (
        f"candidates-{window.start.date().isoformat()}-"
        f"{datetime.fromordinal(window.end.date().toordinal() - 1).date().isoformat()}.json"
    )
    if retained.exists():
        source = CandidateSnapshot.model_validate_json(retained.read_text(encoding="utf-8"))
    else:
        # The fetch snapshot was not retained (data/*.json is gitignored), so the
        # artifact's own ranked candidates are what survives. Per-query counts
        # are left empty rather than invented.
        source = create_snapshot(
            window=window,
            profile_version=settings.profile.version,
            generated_at=generated_at,
            query_results=[],
            raw_papers=papers,
            deduplicated_papers=papers,
        )

    gated = gate_papers(
        source.papers,
        domain=settings.group.domain,
        disambiguation=load_disambiguation(),
        negative_terms=settings.profile.negative_terms,
    )
    snapshot = create_snapshot(
        window=window,
        profile_version=settings.profile.version,
        generated_at=generated_at,
        query_results=source.queries,
        raw_papers=[],
        deduplicated_papers=source.papers,
        kept_papers=gated.kept,
        rejections=gated.rejections,
        context_flags=gated.context_flags,
    )
    snapshot = snapshot.model_copy(
        update={"records_retrieved": max(source.records_retrieved, len(source.papers))}
    )

    ranked_papers = rank_papers(
        gated.kept,
        settings.profile,
        window,
        {flag.arxiv_id for flag in gated.context_flags},
    )
    rank_run_id = _rank_run_id(snapshot, settings)
    ranked = RankedSnapshot(
        run_id=rank_run_id,
        source_run_id=snapshot.run_id,
        generated_at=generated_at,
        retrieval_window=window,
        records_before_history=len(snapshot.papers),
        records_after_history=len(gated.kept),
        ranked_papers=ranked_papers,
        rejections=gated.rejections,
    )
    selection_run_id = _selection_run_id(rank_run_id, settings, SELECTION_LIMIT)
    selected = select_diverse(ranked_papers, settings.profile, limit=SELECTION_LIMIT)
    summaries = summarize_selection(
        selected,
        settings.profile,
        DeterministicSummaryProvider(),
        max_provider_papers=settings.app.summarization.max_provider_papers,
    )
    digest = build_digest_artifact(
        snapshot=snapshot,
        ranked=ranked,
        recommendations=summaries.recommendations,
        profile=settings.profile,
        run_id=selection_run_id,
        generated_at=generated_at,
        timezone=timezone,
        summary_mode="offline",
        summary_fallbacks=summaries.fallback_count,
        near_miss_limit=settings.app.reporting.near_miss_limit,
    )
    paths = render_reports(
        digest,
        templates_dir=ROOT / "templates",
        reports_dir=REPORTS,
    )
    records = records_for_selection(
        run_id=selection_run_id,
        run_timestamp=generated_at,
        window=window,
        selected=selected,
        report_path=str(paths.markdown.relative_to(ROOT)).replace("\\", "/"),
    )
    print(
        f"{path.name}: {len(source.papers)} candidates -> "
        f"{len(gated.kept)} kept, {len(gated.rejections)} gated out, "
        f"{len(summaries.recommendations)} recommended"
    )
    return digest, records


def main() -> None:
    """Rebuild every committed dated report and rewrite history to match."""
    settings = load_settings()
    dated = sorted(REPORTS.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-*.json"))
    if not dated:
        raise SystemExit("no committed dated reports found")
    history: list[HistoryRecord] = []
    for path in dated:
        _digest, records = regenerate(path, settings)
        history.extend(records)
    history_path = DATA / "history.jsonl"
    history_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in history), encoding="utf-8"
    )
    print(f"rewrote {history_path} with {len(history)} records")


if __name__ == "__main__":
    main()
