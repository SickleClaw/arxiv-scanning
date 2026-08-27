"""Structural tests for safe weekly automation and repository-backed state."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from arxiv_digest.dashboard.data import load_digest, load_digest_history
from arxiv_digest.history import load_history
from arxiv_digest.models import HistoryRecord

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "weekly_digest.yml"


def _workflow() -> tuple[dict[str, object], str]:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded, text


def test_workflow_yaml_schedule_dispatch_inputs_and_concurrency() -> None:
    workflow, _text = _workflow()
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    assert triggers["schedule"] == [{"cron": "0 13 * * 1"}]
    dispatch = triggers["workflow_dispatch"]
    inputs = dispatch["inputs"]
    assert set(inputs) == {
        "days",
        "limit",
        "summary_mode",
        "send_email",
        "persist_reports",
    }
    assert inputs["summary_mode"]["options"] == ["auto", "offline", "openai"]
    assert inputs["send_email"]["default"] is False
    assert inputs["persist_reports"]["default"] is True
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_workflow_separates_permissions_quality_generation_artifacts_and_email() -> None:
    workflow, text = _workflow()
    jobs = workflow["jobs"]
    assert set(jobs) == {"validate", "generate", "persist", "email"}
    assert workflow["permissions"] == {"contents": "read"}
    for job_name in ("validate", "generate", "email"):
        assert jobs[job_name]["permissions"] == {"contents": "read"}
    assert jobs["persist"]["permissions"] == {"contents": "write"}
    validate_runs = "\n".join(str(step.get("run", "")) for step in jobs["validate"]["steps"])
    for command in (
        "uv sync --locked",
        "ruff format --check",
        "ruff check",
        "mypy src",
        "pytest --cov",
        "doctor --production",
    ):
        assert command in validate_runs
    assert "ARXIV_CONTACT_EMAIL" in text
    assert '*".invalid"' in text
    assert "DAYS <= 31" in text
    assert "LIMIT <= 50" in text
    assert "actions/upload-artifact@v4" in text
    assert "reports/run-summary.json" in text
    assert "retention-days: 30" in text
    assert "arxiv-digest email-report" in text


def test_persistence_is_allowlisted_idempotent_and_never_force_pushes() -> None:
    workflow, text = _workflow()
    persist = workflow["jobs"]["persist"]
    persist_text = "\n".join(str(step.get("run", "")) for step in persist["steps"])
    assert "git pull --rebase" in persist_text
    assert "git diff --cached --quiet" in persist_text
    assert "github-actions[bot]" in persist_text
    assert "git add -A" not in persist_text
    assert "git add ." not in persist_text
    assert "--force" not in persist_text
    assert "personal access token" not in text.lower()
    assert "data/history.jsonl" in persist_text
    assert "reports/latest.json" in persist_text
    assert "weekly-arxiv-digest\\.json" in persist_text
    for forbidden in ("candidates", "ranked-", "selection-", ".env", "smtp_password"):
        assert forbidden not in persist_text.lower()


def test_committed_latest_is_fully_recorded_in_history_and_dashboard_history() -> None:
    """The newest committed digest must be recorded in history, run for run.

    History is append-only and CI adds a run every week, so this asserts the
    latest run's records agree with the latest report rather than asserting the
    file holds exactly one run.
    """
    reports_dir = REPOSITORY_ROOT / "reports"
    latest = load_digest(reports_dir / "latest.json")
    history = load_history(REPOSITORY_ROOT / "data" / "history.jsonl")
    dashboard_history, errors = load_digest_history(reports_dir)
    assert errors == []
    latest_records = [record for record in history if record.run_id == latest.run_id]
    assert len(latest_records) == len(latest.recommendations)
    assert {record.arxiv_id for record in latest_records} == {
        recommendation.paper.arxiv_id for recommendation in latest.recommendations
    }
    assert dashboard_history[0][1].run_id == latest.run_id


def test_history_runs_are_internally_consistent() -> None:
    """Every history run must carry one record per rank, with no duplicate papers."""
    history = load_history(REPOSITORY_ROOT / "data" / "history.jsonl")
    assert history
    by_run: dict[str, list[HistoryRecord]] = defaultdict(list)
    for record in history:
        by_run[record.run_id].append(record)
    for run_id, records in by_run.items():
        ranks = sorted(record.rank for record in records)
        assert ranks == list(range(1, len(records) + 1)), run_id
        identifiers = {record.arxiv_id for record in records}
        assert len(identifiers) == len(records), run_id
