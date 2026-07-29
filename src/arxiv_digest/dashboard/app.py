"""Local, read-only Streamlit dashboard for completed digest artifacts."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import streamlit as st

from arxiv_digest.dashboard.components import (
    render_candidate,
    render_paper_detail,
    render_recommendation,
)
from arxiv_digest.dashboard.data import (
    DashboardDataError,
    load_digest,
    load_digest_history,
    repeated_papers,
)
from arxiv_digest.dashboard.filters import (
    DateField,
    ExplorerPaper,
    SortField,
    build_explorer_papers,
    filter_papers,
    sort_papers,
)
from arxiv_digest.models import DigestArtifact, RecommendationType

LOGGER = logging.getLogger(__name__)
_VIEWS = ("Current Digest", "Candidate Explorer", "History", "Paper Detail")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", type=Path, default=Path("reports/latest.json"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    arguments, _unknown = parser.parse_known_args()
    return arguments


def _header(digest: DigestArtifact) -> None:
    generated = digest.generated_at.astimezone(ZoneInfo(digest.timezone))
    st.title(digest.title)
    st.info(f"Abstract-based summaries only. {digest.summary_basis}")
    st.caption(
        f"Generated {generated:%Y-%m-%d %H:%M %Z} · profile "
        f"{digest.profile_name} {digest.profile_version} ({digest.profile_hash}) · "
        f"summary provider: {digest.summary_mode}"
    )


def _current_digest(digest: DigestArtifact) -> None:
    _header(digest)
    columns = st.columns(4)
    columns[0].metric("Retrieved", digest.records_retrieved)
    columns[1].metric("Deduplicated", digest.records_after_deduplication)
    columns[2].metric("Ranked", digest.records_ranked)
    columns[3].metric("Selected", digest.records_selected)
    st.caption(
        f"Retrieval window: {digest.retrieval_window.start.isoformat()} through "
        f"{digest.retrieval_window.end.isoformat()} (end exclusive)"
    )
    if len(digest.recommendations) < 10:
        st.warning(
            f"This digest contains {len(digest.recommendations)} recommendations; "
            "the eligible candidate pool did not produce ten selections."
        )
    for recommendation in sorted(digest.recommendations, key=lambda item: item.rank):
        render_recommendation(recommendation)
    st.subheader("Strong candidates that narrowly missed selection")
    if not digest.near_misses:
        st.info("No near-miss candidates are available in this digest.")
    for index, item in enumerate(digest.near_misses, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. {item.paper.title}**")
            st.caption(
                f"{item.paper.primary_category} · score "
                f"{item.score.final_preselection_score:.3f} · arXiv:{item.paper.arxiv_id}"
            )
            st.write(item.score.explanation)


def _date_range(
    papers: list[ExplorerPaper], date_field: DateField
) -> tuple[date | None, date | None]:
    if not papers:
        return None, None
    dates = [
        (
            item.ranked.paper.published_at.date()
            if date_field == "published"
            else item.ranked.paper.updated_at.date()
        )
        for item in papers
    ]
    selected = st.date_input(
        "Date range",
        value=(min(dates), max(dates)),
        min_value=min(dates),
        max_value=max(dates),
    )
    if isinstance(selected, tuple) and len(selected) == 2:
        return selected[0], selected[1]
    return None, None


def _candidate_explorer(digest: DigestArtifact) -> None:
    _header(digest)
    st.header("Candidate Explorer")
    papers = build_explorer_papers(digest)
    if not papers:
        st.info("This digest has no ranked candidate pool to explore.")
        return
    search = st.text_input("Search title, authors, and abstract")
    categories = sorted({item.ranked.paper.primary_category for item in papers})
    selected_categories = set(st.multiselect("Primary categories", categories))
    type_values = [item.value for item in RecommendationType]
    selected_type_values = st.multiselect("Recommendation types", type_values)
    selected_types = {RecommendationType(value) for value in selected_type_values}
    minimum_score = st.slider("Minimum final score", 0.0, 1.0, 0.0, 0.01)
    date_field = cast(
        DateField,
        st.radio("Date field", ("updated", "published"), horizontal=True),
    )
    start_date, end_date = _date_range(papers, date_field)
    selected_only = st.checkbox("Selected papers only")
    include_near_misses = st.checkbox(
        "Include near misses with selected-only filter",
        disabled=not selected_only,
    )
    sort_labels = {
        "Rank": "rank",
        "Score": "score",
        "Publication date": "publication_date",
        "Update date": "update_date",
    }
    sort_label = st.selectbox("Sort by", list(sort_labels))
    sort_by = cast(SortField, sort_labels[sort_label])
    descending = st.checkbox("Descending", value=sort_by != "rank")
    visible = filter_papers(
        papers,
        search=search,
        primary_categories=selected_categories,
        recommendation_types=selected_types,
        minimum_score=minimum_score,
        start_date=start_date,
        end_date=end_date,
        date_field=date_field,
        selected_only=selected_only,
        include_near_misses=include_near_misses,
    )
    visible = sort_papers(visible, sort_by=sort_by, descending=descending)
    st.caption(f"Showing {len(visible)} of {len(papers)} ranked candidates.")
    if not visible:
        st.info("No candidates match the current filters.")
    for item in visible:
        render_candidate(item)


def _history(reports_dir: Path) -> None:
    st.title("Digest History")
    st.info("History is read from dated local JSON reports; this view never changes files.")
    try:
        history, errors = load_digest_history(reports_dir)
    except DashboardDataError as exc:
        LOGGER.warning("Cannot load dashboard history: %s", exc)
        st.error(str(exc))
        return
    for error in errors:
        st.warning(error)
    if not history:
        st.info(f"No dated digest JSON reports were found in {reports_dir}.")
        return
    labels = [
        f"{entry.report_date.isoformat()} ({entry.recommendation_count} recommendations)"
        for entry, _digest in history
    ]
    selected_label = st.selectbox("Report date", labels)
    selected_index = labels.index(selected_label)
    entry, digest = history[selected_index]
    st.caption(f"Loaded {entry.path} · generated {entry.generated_at}")
    for recommendation in sorted(digest.recommendations, key=lambda item: item.rank):
        st.markdown(
            f"**{recommendation.rank}. [{recommendation.paper.title}]"
            f"({recommendation.paper.abstract_url})** — "
            f"{recommendation.recommendation_type.value}, "
            f"score {recommendation.score.final_preselection_score:.3f}, "
            f"v{recommendation.paper.version}"
        )
    st.subheader("Repeated recommendations")
    repeated = repeated_papers(history)
    if not repeated:
        st.info("No paper appears in more than one available digest.")
    else:
        st.dataframe(
            [
                {
                    "arXiv ID": item.arxiv_id,
                    "Title": item.title,
                    "Digest dates": ", ".join(value.isoformat() for value in item.report_dates),
                    "Versions": ", ".join(f"v{value}" for value in item.versions),
                    "Updated version resurfaced": item.updated_version_resurfaced,
                }
                for item in repeated
            ],
            hide_index=True,
            width="stretch",
        )


def _paper_detail(digest: DigestArtifact) -> None:
    st.title("Paper Detail")
    papers = build_explorer_papers(digest)
    if not papers:
        st.info("This digest has no ranked papers to inspect.")
        return
    options = {
        f"#{item.ranking_position} · {item.ranked.paper.title} "
        f"(arXiv:{item.ranked.paper.arxiv_id})": item
        for item in papers
    }
    selected = st.selectbox("Paper", list(options))
    render_paper_detail(options[selected])


def main() -> None:
    """Run the dashboard using only local validated digest artifacts."""
    st.set_page_config(page_title="Weekly arXiv Digest", layout="wide")
    arguments = _arguments()
    view = st.sidebar.radio("View", _VIEWS)
    st.sidebar.caption("Read-only local dashboard")
    if view == "History":
        _history(arguments.reports_dir)
        return
    try:
        digest = load_digest(arguments.report)
    except DashboardDataError as exc:
        LOGGER.warning("Cannot load dashboard report: %s", exc)
        st.title("Weekly arXiv Digest")
        st.error(str(exc))
        return
    if view == "Current Digest":
        _current_digest(digest)
    elif view == "Candidate Explorer":
        _candidate_explorer(digest)
    else:
        _paper_detail(digest)


if __name__ == "__main__":
    main()
