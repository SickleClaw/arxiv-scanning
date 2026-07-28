"""Tests for complete abstract-labeled Markdown and HTML reports."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from arxiv_digest.config import load_settings
from arxiv_digest.models import (
    ABSTRACT_SUMMARY_BASIS,
    CandidateSnapshot,
    DateWindow,
    RankedPaper,
    RankedSnapshot,
    Recommendation,
    RecommendationType,
    ScoreBreakdown,
)
from arxiv_digest.reporting import render_reports
from arxiv_digest.snapshot import create_snapshot
from arxiv_digest.summarization import DeterministicSummaryProvider

NOW = datetime(2026, 7, 28, 12, tzinfo=UTC)
WINDOW = DateWindow(
    start=datetime(2026, 7, 20, tzinfo=UTC),
    end=datetime(2026, 7, 28, tzinfo=UTC),
)


def _score(value: float) -> ScoreBreakdown:
    return ScoreBreakdown(
        semantic_relevance=value,
        keyword_relevance=value,
        category_relevance=value,
        recency=value,
        novelty=value,
        feedback_affinity=0.5,
        final_preselection_score=value,
        explanation="Strongest matched profile terms: spin ice.",
    )


def test_reports_include_required_metadata_escape_html_and_limit_near_misses(
    tmp_path: Path, paper_factory
) -> None:  # type: ignore[no-untyped-def]
    settings = load_settings()
    selected_paper = paper_factory(title="Spin ice <script>alert(1)</script>")
    other_papers = [
        paper_factory(
            arxiv_id=f"2607.50{index:03d}",
            title=f"Near miss {index}",
            published_at=WINDOW.start + timedelta(days=index),
            updated_at=WINDOW.start + timedelta(days=index),
        )
        for index in range(1, 7)
    ]
    papers = [selected_paper, *other_papers]
    snapshot: CandidateSnapshot = create_snapshot(
        window=WINDOW,
        profile_version=settings.profile.version,
        generated_at=NOW,
        query_results=[],
        raw_papers=papers,
        deduplicated_papers=papers,
    )
    ranked_items = [
        RankedPaper(paper=paper, score=_score(0.9 - index / 100))
        for index, paper in enumerate(papers)
    ]
    ranked = RankedSnapshot(
        run_id="rank-fixture",
        source_run_id=snapshot.run_id,
        generated_at=NOW,
        retrieval_window=WINDOW,
        records_before_history=len(papers),
        records_after_history=len(papers),
        ranked_papers=ranked_items,
    )
    summary = DeterministicSummaryProvider().summarize(
        selected_paper,
        ranked_items[0].score,
        settings.profile,
    )
    recommendation = Recommendation(
        paper=selected_paper,
        score=ranked_items[0].score,
        summary=summary,
        rank=1,
        recommendation_type=RecommendationType.DIRECT,
    )
    paths = render_reports(
        snapshot=snapshot,
        ranked=ranked,
        recommendations=[recommendation],
        profile=settings.profile,
        generated_at=NOW,
        timezone="America/New_York",
        templates_dir=Path("templates"),
        reports_dir=tmp_path,
        near_miss_limit=5,
    )
    markdown = paths.markdown.read_text(encoding="utf-8")
    html = paths.html.read_text(encoding="utf-8")
    assert ABSTRACT_SUMMARY_BASIS in markdown
    assert "Records after deduplication: 7" in markdown
    assert "Recommendation type:** direct" in markdown
    assert "Near miss 5" in markdown
    assert "Near miss 6" not in markdown
    assert "<script>alert(1)</script>" in markdown
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Abstract-based summaries" in html
    assert paths.latest_markdown.read_text(encoding="utf-8") == markdown
    assert paths.latest_html.read_text(encoding="utf-8") == html
