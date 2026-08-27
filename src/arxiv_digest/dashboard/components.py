"""Reusable Streamlit presentation components for digest records."""

from __future__ import annotations

import streamlit as st

from arxiv_digest.dashboard.filters import ExplorerPaper, strongest_profile_terms
from arxiv_digest.models import Paper, Recommendation, ScoreBreakdown


def _authors(paper: Paper, *, limit: int = 8) -> str:
    visible = paper.authors[:limit]
    suffix = f" and {len(paper.authors) - limit} more" if len(paper.authors) > limit else ""
    return ", ".join(visible) + suffix


def render_score_breakdown(score: ScoreBreakdown) -> None:
    """Render compact normalized score components and strongest matched terms."""
    st.markdown("**Score breakdown (0-1)**")
    st.dataframe(
        {
            "Signal": [
                "Semantic / TF-IDF",
                "Keywords",
                "Category",
                "Relevance (weighted)",
                "Recency (multiplier)",
                "Final score",
            ],
            "Value": [
                score.semantic_relevance,
                score.keyword_relevance,
                score.category_relevance,
                score.relevance,
                score.recency,
                score.final_preselection_score,
            ],
        },
        hide_index=True,
        width="stretch",
    )
    if score.context_penalty_applied:
        st.caption("An ambiguous term appeared without condensed-matter context.")
    terms = strongest_profile_terms(score.explanation)
    st.caption(f"Strongest profile terms: {', '.join(terms)}" if terms else score.explanation)


def render_recommendation(recommendation: Recommendation) -> None:
    """Render one complete abstract-grounded recommendation card."""
    paper = recommendation.paper
    with st.container(border=True):
        st.subheader(f"{recommendation.rank}. {paper.title}")
        st.caption(
            f"{_authors(paper)} · {paper.primary_category} · "
            f"{recommendation.recommendation_type.value} · "
            f"score {recommendation.score.final_preselection_score:.3f}"
        )
        st.caption(f"Categories: {', '.join(paper.categories)}")
        st.markdown(f"**Takeaway:** {recommendation.summary.one_sentence_takeaway}")
        st.write(recommendation.summary.brief_summary)
        st.markdown(f"**Why selected:** {recommendation.summary.why_relevant}")
        st.markdown(f"**Metadata limitation:** {recommendation.summary.limitations}")
        with st.expander("Score components and matched profile terms"):
            render_score_breakdown(recommendation.score)
        st.caption(
            f"Submitted {paper.published_at.date().isoformat()} · "
            f"updated {paper.updated_at.date().isoformat()} · "
            f"arXiv:{paper.arxiv_id}v{paper.version}"
        )
        abstract_column, pdf_column = st.columns(2)
        abstract_column.link_button("Open arXiv abstract", paper.abstract_url)
        pdf_column.link_button("Open PDF", paper.pdf_url)


def render_candidate(candidate: ExplorerPaper) -> None:
    """Render a ranked candidate with explicit selection status."""
    paper = candidate.ranked.paper
    if candidate.selected:
        status = f"Selected #{candidate.recommendation_rank} · {candidate.recommendation_type}"
    elif candidate.near_miss:
        status = "Near miss"
    else:
        status = "Not selected"
    with st.container(border=True):
        st.markdown(f"**#{candidate.ranking_position} · {paper.title}**")
        st.caption(
            f"{status} · {_authors(paper)} · {paper.primary_category} · "
            f"score {candidate.ranked.score.final_preselection_score:.3f}"
        )
        terms = strongest_profile_terms(candidate.ranked.score.explanation)
        st.caption(
            f"Strongest profile terms: {', '.join(terms)}"
            if terms
            else candidate.ranked.score.explanation
        )
        st.write(paper.abstract)


def render_paper_detail(candidate: ExplorerPaper) -> None:
    """Render all locally stored metadata, ranking signals, and summary when available."""
    paper = candidate.ranked.paper
    st.header(paper.title)
    if candidate.selected:
        status = "selected"
    elif candidate.near_miss:
        status = "near miss"
    else:
        status = "unselected"
    st.caption(
        f"arXiv:{paper.arxiv_id}v{paper.version} · {status} · ranked "
        f"#{candidate.ranking_position} · {paper.primary_category}"
    )
    st.markdown(f"**Authors:** {_authors(paper, limit=20)}")
    st.markdown(f"**Categories:** {', '.join(paper.categories)}")
    st.markdown(
        f"**Submitted:** {paper.published_at.isoformat()}  \n"
        f"**Updated:** {paper.updated_at.isoformat()}"
    )
    if paper.doi:
        st.markdown(f"**DOI:** {paper.doi}")
    if paper.journal_reference:
        st.markdown(f"**Journal reference:** {paper.journal_reference}")
    st.subheader("Abstract")
    st.write(paper.abstract)
    if candidate.summary is not None:
        st.subheader("Abstract-grounded summary")
        st.markdown(f"**Takeaway:** {candidate.summary.one_sentence_takeaway}")
        st.write(candidate.summary.brief_summary)
        st.markdown(f"**Why relevant:** {candidate.summary.why_relevant}")
        st.markdown(f"**Methods or systems:** {', '.join(candidate.summary.methods_or_systems)}")
        st.markdown(f"**Limitations:** {candidate.summary.limitations}")
        st.caption(
            f"{candidate.summary.summary_basis} Confidence: {candidate.summary.confidence:.2f}."
        )
    else:
        st.info("No summary was generated because this candidate was not selected.")
    render_score_breakdown(candidate.ranked.score)
    abstract_column, pdf_column = st.columns(2)
    abstract_column.link_button("Open arXiv abstract", paper.abstract_url)
    pdf_column.link_button("Open PDF", paper.pdf_url)
