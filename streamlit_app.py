"""Streamlit Community Cloud entry point for the read-only digest dashboard."""

from pathlib import Path

from arxiv_digest.dashboard.app import main

if __name__ == "__main__":
    main(repository_root=Path(__file__).resolve().parent)
