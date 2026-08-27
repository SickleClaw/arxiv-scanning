"""Tests for `arxiv-digest explain`, the audit path for one paper in one run."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from arxiv_digest.cli import app

FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = FIXTURES / "candidates_2026-07-21.json"
WINDOW = ["--start", "2026-07-21", "--end", "2026-07-28"]

runner = CliRunner()


def explain(arxiv_id: str) -> str:
    result = runner.invoke(app, ["explain", arxiv_id, "--snapshot", str(SNAPSHOT), *WINDOW])
    assert result.exit_code == 0, result.output
    return result.stdout


def test_explaining_a_gated_paper_names_the_rule_that_rejected_it() -> None:
    """The hep-ph paper the audit named, and why it is gone."""
    output = explain("2607.20843")
    assert "EXCLUDE" in output
    assert "cosmological-monopole" in output
    assert "Cosmological topological defects" in output
    assert "REJECTED at the category gate" in output


def test_a_gated_paper_is_not_given_a_score() -> None:
    """Scoring a paper the gates reject would invite weighing a dead number."""
    output = explain("2607.20843")
    assert "SCORE" not in output
    assert "no score" in output


def test_explaining_a_retained_paper_shows_the_whole_component_breakdown() -> None:
    output = explain("2607.23490")
    assert "verdict         passed" in output
    for component in ("semantic", "keyword", "category", "relevance", "recency", "final"):
        assert component in output
    assert "tier            direct" in output
    assert "Strongest matched profile terms" in output


def test_a_paper_rejected_by_score_rather_than_gate_says_so() -> None:
    """The plasma paper passes every gate and is removed by relevance alone."""
    output = explain("2607.25481")
    assert "SOFT" in output
    assert "verdict         passed" in output
    assert "rejected: below wildcard_min_score" in output


def test_the_in_field_shield_is_visible_in_the_output() -> None:
    output = explain("2607.23490")
    assert "in-field papers are never hard-rejected" in output


def test_an_unretrieved_identifier_is_distinguished_from_a_rejection() -> None:
    """'Never retrieved' and 'rejected' are different problems."""
    result = runner.invoke(app, ["explain", "9999.99999", "--snapshot", str(SNAPSHOT), *WINDOW])
    assert result.exit_code == 1
    assert "never retrieved" in result.output


def test_a_missing_snapshot_is_actionable() -> None:
    result = runner.invoke(
        app, ["explain", "2607.23490", "--snapshot", "does-not-exist.json", *WINDOW]
    )
    assert result.exit_code == 1
    assert "Candidate snapshot does not exist" in result.output
