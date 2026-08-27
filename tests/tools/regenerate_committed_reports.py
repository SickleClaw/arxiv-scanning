"""Rebuild the committed digest reports under the current artifact schema.

DIGEST_SCHEMA_VERSION is validated strictly - a mismatch raises rather than
degrading - so a change to ScoreBreakdown makes every stored report unreadable.
With only two committed runs, regenerating them is cheaper and cleaner than
carrying a back-compatibility loader forever.

Usage::

    python tests/tools/regenerate_committed_reports.py

Each window is rebuilt from its committed candidate fixture, never from the
report it is about to overwrite, so running this twice produces the same
artifacts. ``data/*.json`` is gitignored, which is why the candidate sets live
under ``tests/fixtures/``. Recommendation history is rewritten to match, since
run ids move with the profile hash.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from arxiv_digest.config import Settings, load_settings
from arxiv_digest.domain_filter import gate_papers, load_disambiguation
from arxiv_digest.history import records_for_selection
from arxiv_digest.models import CandidateSnapshot, DigestArtifact, HistoryRecord, RankedSnapshot
from arxiv_digest.pipeline import _rank_run_id, _selection_run_id
from arxiv_digest.ranking import rank_papers
from arxiv_digest.reporting import build_digest_artifact, render_reports
from arxiv_digest.selection import select_diverse
from arxiv_digest.snapshot import create_snapshot
from arxiv_digest.summarization import DeterministicSummaryProvider, summarize_selection

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
SELECTION_LIMIT = 10

# Each committed run, pinned to the candidate set it was produced from and to
# the timestamp that names its report file.
WINDOWS = [
    ("candidates_2026-07-21.json", "2026-07-29T17:08:02.966397+00:00"),
    ("candidates_2026-07-26.json", "2026-08-03T13:07:41.884123+00:00"),
]


def regenerate(
    fixture: str, generated_at_iso: str, settings: Settings
) -> tuple[DigestArtifact, list[HistoryRecord]]:
    """Rebuild one committed report and return it with its history records."""
    source = CandidateSnapshot.model_validate_json((FIXTURES / fixture).read_text(encoding="utf-8"))
    generated_at = datetime.fromisoformat(generated_at_iso)
    window = source.retrieval_window

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
        raw_papers=source.papers,
        deduplicated_papers=source.papers,
        kept_papers=gated.kept,
        rejections=gated.rejections,
        context_flags=gated.context_flags,
    )
    # Carry the original retrieval total; the fixture holds only what survived
    # deduplication.
    snapshot = snapshot.model_copy(
        update={"records_retrieved": max(source.records_retrieved, len(source.papers))}
    )
    ranked_papers = rank_papers(
        gated.kept,
        settings.profile,
        window,
        {flag.arxiv_id for flag in gated.context_flags},
        settings.group.domain,
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
        timezone=settings.app.reporting.timezone,
        summary_mode="offline",
        summary_fallbacks=summaries.fallback_count,
        near_miss_limit=settings.app.reporting.near_miss_limit,
    )
    paths = render_reports(digest, templates_dir=ROOT / "templates", reports_dir=REPORTS)
    records = records_for_selection(
        run_id=selection_run_id,
        run_timestamp=generated_at,
        window=window,
        selected=selected,
        report_path=str(paths.markdown.relative_to(ROOT)).replace("\\", "/"),
    )
    print(
        f"{fixture}: {len(source.papers)} candidates -> {len(gated.kept)} kept, "
        f"{len(gated.rejections)} gated out, {len(summaries.recommendations)} recommended "
        f"-> {paths.json.name}"
    )
    return digest, records


def main() -> None:
    """Rebuild every committed run and rewrite history to match."""
    settings = load_settings()
    history: list[HistoryRecord] = []
    for fixture, generated_at in WINDOWS:
        _digest, records = regenerate(fixture, generated_at, settings)
        history.extend(records)
    history_path = DATA / "history.jsonl"
    history_path.write_text(
        "".join(record.model_dump_json() + "\n" for record in history), encoding="utf-8"
    )
    print(f"rewrote {history_path} with {len(history)} records")


if __name__ == "__main__":
    main()
