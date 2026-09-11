"""Structural tests for safe weekly automation and repository-backed state."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from arxiv_digest.dashboard.data import load_digest, load_digest_history
from arxiv_digest.history import load_history
from arxiv_digest.models import DigestArtifact, HistoryRecord

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


def _assert_latest_history(latest: DigestArtifact, history: list[HistoryRecord]) -> None:
    """Allow older runs while checking every persisted field of the latest run."""
    records = [record for record in history if record.run_id == latest.run_id]
    assert len(records) == len(latest.recommendations)
    assert {record.arxiv_id for record in records} == {
        item.paper.arxiv_id for item in latest.recommendations
    }
    by_id = {record.arxiv_id: record for record in records}
    for item in latest.recommendations:
        record = by_id[item.paper.arxiv_id]
        assert record.run_timestamp == latest.generated_at
        assert record.retrieval_window == latest.retrieval_window
        assert record.version == item.paper.version
        assert record.paper_updated_at == item.paper.updated_at
        assert record.rank == item.rank
        assert record.recommendation_type == item.recommendation_type
        assert record.final_score == item.score.final_preselection_score
        assert record.report_path
        report_date = latest.generated_at.astimezone(ZoneInfo(latest.timezone)).date()
        assert Path(record.report_path).name == f"{report_date}-weekly-arxiv-digest.md"


def test_committed_latest_matches_its_history_and_dashboard_history() -> None:
    reports_dir = REPOSITORY_ROOT / "reports"
    latest = load_digest(reports_dir / "latest.json")
    history = load_history(REPOSITORY_ROOT / "data" / "history.jsonl")
    dashboard_history, errors = load_digest_history(reports_dir)
    assert errors == []
    _assert_latest_history(latest, history)
    assert dashboard_history[0][1].run_id == latest.run_id


def test_latest_history_accepts_multiple_runs() -> None:
    latest = load_digest(REPOSITORY_ROOT / "reports" / "latest.json")
    records = [
        record
        for record in load_history(REPOSITORY_ROOT / "data" / "history.jsonl")
        if record.run_id == latest.run_id
    ]
    older = [record.model_copy(update={"run_id": "older-run"}) for record in records]
    _assert_latest_history(latest, older + records)


@pytest.mark.parametrize(
    "corruption",
    ["missing", "duplicate", "identity", "version", "rank", "score", "window", "timestamp", "path"],
)
def test_latest_history_rejects_inconsistent_records(corruption: str) -> None:
    latest = load_digest(REPOSITORY_ROOT / "reports" / "latest.json")
    records = [
        record
        for record in load_history(REPOSITORY_ROOT / "data" / "history.jsonl")
        if record.run_id == latest.run_id
    ]
    assert records
    first = records[0]
    changes: dict[str, dict[str, object]] = {
        "identity": {"arxiv_id": "0000.00000"},
        "version": {"version": first.version + 1},
        "rank": {"rank": first.rank + 1},
        "score": {"final_score": 1.0 if first.final_score != 1.0 else 0.0},
        "window": {
            "retrieval_window": latest.retrieval_window.model_copy(
                update={"start": latest.retrieval_window.end}
            )
        },
        "timestamp": {"run_timestamp": latest.retrieval_window.start},
        "path": {"report_path": "reports/wrong-report.md"},
    }
    if corruption == "missing":
        records.pop()
    elif corruption == "duplicate":
        records.append(first)
    else:
        records[0] = first.model_copy(update=changes[corruption])
    with pytest.raises(AssertionError):
        _assert_latest_history(latest, records)
