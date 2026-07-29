"""CLI, launcher, import, and Streamlit rendering smoke tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from arxiv_digest.cli import app
from arxiv_digest.dashboard.launcher import build_dashboard_command, launch_dashboard
from arxiv_digest.exceptions import ConfigurationError

runner = CliRunner()


def test_dashboard_command_construction_is_shell_independent(tmp_path: Path) -> None:
    report = tmp_path / "reports with spaces" / "latest.json"
    reports_dir = tmp_path / "reports with spaces"
    command = build_dashboard_command(report, reports_dir)
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert "--server.address=127.0.0.1" in command
    assert "--browser.gatherUsageStats=false" in command
    assert command[-4:] == ["--report", str(report), "--reports-dir", str(reports_dir)]
    assert all("shell" not in argument for argument in command)


def test_launch_dashboard_returns_runner_status(monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        returncode = 7

    calls: list[tuple[list[str], bool]] = []

    def fake_runner(command: list[str], *, check: bool) -> Result:
        calls.append((command, check))
        return Result()

    monkeypatch.setattr(
        "arxiv_digest.dashboard.launcher.importlib.util.find_spec", lambda _name: True
    )
    assert launch_dashboard(Path("report.json"), Path("reports"), runner=fake_runner) == 7
    assert calls[0][1] is False


def test_launch_dashboard_has_actionable_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "arxiv_digest.dashboard.launcher.importlib.util.find_spec", lambda _name: None
    )
    with pytest.raises(ConfigurationError, match="uv sync --extra dashboard"):
        launch_dashboard(Path("report.json"), Path("reports"))


def test_cli_dashboard_passes_explicit_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: list[tuple[Path, Path]] = []

    def fake_launch(report: Path, reports_dir: Path) -> int:
        received.append((report, reports_dir))
        return 0

    monkeypatch.setattr("arxiv_digest.cli.launch_dashboard", fake_launch)
    report = tmp_path / "chosen.json"
    reports_dir = tmp_path / "history"
    result = runner.invoke(
        app,
        ["dashboard", "--report", str(report), "--reports-dir", str(reports_dir)],
    )
    assert result.exit_code == 0, result.output
    assert received == [(report, reports_dir)]


def test_dashboard_modules_import_without_running_pipeline() -> None:
    assert importlib.import_module("arxiv_digest.dashboard.app") is not None
    assert importlib.import_module("arxiv_digest.dashboard.data") is not None


def test_streamlit_renders_current_digest_and_all_views(tmp_path: Path, digest_factory) -> None:  # type: ignore[no-untyped-def]
    report = tmp_path / "latest.json"
    dated = tmp_path / "2026-07-28-weekly-arxiv-digest.json"
    contents = digest_factory().model_dump_json(indent=2)
    report.write_text(contents, encoding="utf-8")
    dated.write_text(contents, encoding="utf-8")
    script = f"""
import sys
sys.argv = [
    "dashboard",
    "--report", {str(report)!r},
    "--reports-dir", {str(tmp_path)!r},
]
from arxiv_digest.dashboard.app import main
main()
"""
    test_app = AppTest.from_string(
        script,
        default_timeout=20,
    )
    test_app.run()
    assert not test_app.exception
    assert any("Personalized Weekly" in title.value for title in test_app.title)
    for view in ("Candidate Explorer", "History", "Paper Detail"):
        test_app.sidebar.radio[0].set_value(view).run()
        assert not test_app.exception
