"""Shell-independent launcher for the optional local Streamlit dashboard."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Protocol, cast

from arxiv_digest.dashboard.paths import resolve_dashboard_paths
from arxiv_digest.exceptions import ConfigurationError


class _CompletedProcess(Protocol):
    returncode: int


class _Runner(Protocol):
    def __call__(self, command: list[str], *, check: bool) -> _CompletedProcess: ...


def build_dashboard_command(report: Path, reports_dir: Path) -> list[str]:
    """Build a loopback-only Streamlit command without shell interpolation."""
    app_path = Path(__file__).with_name("app.py")
    paths = resolve_dashboard_paths(report=report, reports_dir=reports_dir)
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address=127.0.0.1",
        "--browser.serverAddress=127.0.0.1",
        "--browser.gatherUsageStats=false",
        "--",
        "--report",
        str(paths.report),
        "--reports-dir",
        str(paths.reports_dir),
    ]


def launch_dashboard(
    report: Path,
    reports_dir: Path,
    *,
    runner: _Runner | None = None,
) -> int:
    """Launch Streamlit in the foreground and return its exit status."""
    if importlib.util.find_spec("streamlit") is None:
        raise ConfigurationError(
            "The local dashboard dependency is not installed. Run "
            "'uv sync --extra dashboard' and retry."
        )
    command = build_dashboard_command(report, reports_dir)
    active_runner = runner or cast(_Runner, subprocess.run)
    try:
        completed = active_runner(command, check=False)
    except OSError as exc:
        raise ConfigurationError(f"Cannot start the local dashboard: {exc}") from exc
    return int(completed.returncode)
