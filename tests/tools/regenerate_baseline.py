"""Regenerate the frozen ranking baseline after an intentional scoring change.

Usage::

    python tests/tools/regenerate_baseline.py

Reads the committed candidate fixture and the live research profile, re-ranks
offline with no history exclusions, and rewrites
``tests/fixtures/baseline_2026-07-21.json``. Review the resulting diff: it is
the record of what a scoring change actually did.
"""

from __future__ import annotations

import json
from pathlib import Path

from arxiv_digest.config import load_settings
from arxiv_digest.models import CandidateSnapshot
from arxiv_digest.ranking import rank_papers

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CANDIDATES = FIXTURES / "candidates_2026-07-21.json"
BASELINE = FIXTURES / "baseline_2026-07-21.json"
NOTE = (
    "Frozen output of rank_papers() over tests/fixtures/candidates_2026-07-21.json "
    "using config/research_profile.yaml, with no history exclusions. Regenerate with "
    "'python tests/tools/regenerate_baseline.py' after an intentional scoring change."
)


def build_baseline() -> dict[str, object]:
    """Rank the committed fixture window and return the serializable baseline."""
    snapshot = CandidateSnapshot.model_validate_json(CANDIDATES.read_text(encoding="utf-8"))
    settings = load_settings()
    ranked = rank_papers(snapshot.papers, settings.profile, snapshot.retrieval_window, set())
    return {
        "_note": NOTE,
        "profile_name": settings.profile.name,
        "profile_version": settings.profile.version,
        "window": {
            "start": snapshot.retrieval_window.start.isoformat(),
            "end": snapshot.retrieval_window.end.isoformat(),
        },
        "ranking": [
            {
                "rank": index,
                "arxiv_id": item.paper.arxiv_id,
                "primary_category": item.paper.primary_category,
                "final_preselection_score": round(item.score.final_preselection_score, 6),
                "components": {
                    "semantic_relevance": round(item.score.semantic_relevance, 6),
                    "keyword_relevance": round(item.score.keyword_relevance, 6),
                    "category_relevance": round(item.score.category_relevance, 6),
                    "recency": round(item.score.recency, 6),
                },
            }
            for index, item in enumerate(ranked, start=1)
        ],
    }


def main() -> None:
    """Write the regenerated baseline fixture."""
    BASELINE.write_text(
        json.dumps(build_baseline(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {BASELINE}")


if __name__ == "__main__":
    main()
