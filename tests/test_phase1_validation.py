"""End-to-end validation against the failure that prompted Phase 1.

The design report named two papers from the 2026-07-21 -> 2026-07-28 run: a
hep-ph dark-monopole paper that took a wildcard slot, and a physics.plasm-ph
X-ray Thomson scattering paper that ranked fifth. Both had to go, and the
frustrated-magnetism and neutron-scattering work had to stay.
"""

from __future__ import annotations

import json
from pathlib import Path

from arxiv_digest.config import load_settings
from arxiv_digest.domain_filter import gate_papers, load_disambiguation
from arxiv_digest.models import CandidateSnapshot, GateStage
from arxiv_digest.ranking import rank_papers
from arxiv_digest.selection import select_diverse

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
DIGEST = ROOT / "reports" / "2026-07-29-weekly-arxiv-digest.json"

HEP_PH_DARK_MONOPOLES = "2607.20843"
PLASMA_THOMSON_SCATTERING = "2607.25481"
NEUTRON_SCATTERING_PAPER = "2607.23490"
KAGOME_SPIN_LIQUID_PAPER = "2509.09189"


def run_window(fixture: str):  # type: ignore[no-untyped-def]
    """Gate, rank, and select one window exactly as the pipeline does."""
    settings = load_settings()
    snapshot = CandidateSnapshot.model_validate_json(
        (FIXTURES / fixture).read_text(encoding="utf-8")
    )
    gated = gate_papers(
        snapshot.papers,
        domain=settings.group.domain,
        disambiguation=load_disambiguation(),
        negative_terms=settings.profile.negative_terms,
    )
    ranked = rank_papers(
        gated.kept,
        settings.profile,
        snapshot.retrieval_window,
        {flag.arxiv_id for flag in gated.context_flags},
        settings.group.domain,
    )
    return settings, snapshot, gated, ranked, select_diverse(ranked, settings.profile, limit=10)


def test_the_hep_ph_monopole_paper_is_rejected_with_a_named_reason() -> None:
    _settings, _snapshot, gated, _ranked, selected = run_window("candidates_2026-07-21.json")
    rejection = next(item for item in gated.rejections if item.arxiv_id == HEP_PH_DARK_MONOPOLES)
    assert rejection.stage is GateStage.CATEGORY
    assert rejection.reason
    assert HEP_PH_DARK_MONOPOLES not in {item.paper.arxiv_id for item in selected}


def test_the_plasma_paper_is_dropped_by_relevance_rather_than_by_gate() -> None:
    """It passes every gate and still does not make the list — layers dividing work."""
    settings, _snapshot, gated, ranked, selected = run_window("candidates_2026-07-21.json")
    assert PLASMA_THOMSON_SCATTERING in {item.arxiv_id for item in gated.kept}
    entry = next(item for item in ranked if item.paper.arxiv_id == PLASMA_THOMSON_SCATTERING)
    assert entry.score.final_preselection_score < settings.profile.selection.wildcard_min_score
    assert PLASMA_THOMSON_SCATTERING not in {item.paper.arxiv_id for item in selected}


def test_the_work_that_should_stay_stays() -> None:
    _settings, _snapshot, _gated, _ranked, selected = run_window("candidates_2026-07-21.json")
    chosen = {item.paper.arxiv_id for item in selected}
    assert NEUTRON_SCATTERING_PAPER in chosen
    assert KAGOME_SPIN_LIQUID_PAPER in chosen


def test_every_selected_paper_is_in_the_group_field() -> None:
    for fixture in ("candidates_2026-07-21.json", "candidates_2026-07-26.json"):
        settings, _snapshot, _gated, _ranked, selected = run_window(fixture)
        for item in selected:
            classification = settings.group.domain.classify(item.paper.categories)
            assert settings.group.domain.included(item.paper.categories), (
                f"{item.paper.arxiv_id} selected with {classification}: {item.paper.categories}"
            )


def test_no_in_field_paper_is_ever_rejected_by_a_gate() -> None:
    """The recall side of the check, and the one that is harder to notice."""
    for fixture in ("candidates_2026-07-21.json", "candidates_2026-07-26.json"):
        settings, _snapshot, gated, _ranked, _selected = run_window(fixture)
        for rejection in gated.rejections:
            assert not settings.group.domain.included(rejection.categories), (
                f"{rejection.arxiv_id} was in-field and still rejected: {rejection.reason}"
            )


def test_every_rejection_states_which_gate_and_why() -> None:
    for fixture in ("candidates_2026-07-21.json", "candidates_2026-07-26.json"):
        _settings, _snapshot, gated, _ranked, _selected = run_window(fixture)
        for rejection in gated.rejections:
            assert rejection.reason.strip()
            assert rejection.stage in set(GateStage)
            if rejection.stage is GateStage.HARD_RULE:
                assert rejection.rule_id


def test_the_committed_digest_contains_no_off_domain_papers() -> None:
    """The shipped artifact, not just the recomputation."""
    settings = load_settings()
    digest = json.loads(DIGEST.read_text(encoding="utf-8"))
    identifiers = {item["paper"]["arxiv_id"] for item in digest["recommendations"]}
    assert HEP_PH_DARK_MONOPOLES not in identifiers
    assert PLASMA_THOMSON_SCATTERING not in identifiers
    for item in digest["recommendations"]:
        assert settings.group.domain.included(item["paper"]["categories"])
