"""Tests for strict layered configuration loading."""

from pathlib import Path

import pytest

from arxiv_digest.config import load_settings, redact_mapping
from arxiv_digest.exceptions import ConfigurationError


def test_default_configuration_loads_and_weights_sum_to_one() -> None:
    settings = load_settings()
    weights = settings.profile.ranking_weights
    assert weights.semantic_relevance == pytest.approx(0.45)
    assert sum(
        (
            weights.semantic_relevance,
            weights.keyword_relevance,
            weights.category_relevance,
            weights.recency,
            weights.feedback_or_novelty,
        )
    ) == pytest.approx(1.0)
    assert len(settings.profile.queries) == 6


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARXIV_DIGEST_APP__ARXIV__PAGE_SIZE", "17")
    monkeypatch.setenv("ARXIV_DIGEST_PROFILE__MAX_CANDIDATE_COUNT", "123")
    monkeypatch.setenv("OPENAI_SUMMARY_MODEL", "configured-summary-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "configured-embedding-model")
    settings = load_settings()
    assert settings.app.arxiv.page_size == 17
    assert settings.profile.max_candidate_count == 123
    assert settings.app.summarization.openai_summary_model == "configured-summary-model"
    assert settings.app.summarization.openai_embedding_model == "configured-embedding-model"


def test_missing_or_invalid_configuration_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Cannot read configuration"):
        load_settings(tmp_path / "missing.yaml", Path("config/research_profile.yaml"))
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must contain a mapping"):
        load_settings(invalid, Path("config/research_profile.yaml"))


def test_secret_like_values_are_redacted_recursively() -> None:
    source = {"api_key": "secret", "nested": [{"smtp_password": "password"}], "safe": "ok"}
    assert redact_mapping(source) == {
        "api_key": "***REDACTED***",
        "nested": [{"smtp_password": "***REDACTED***"}],
        "safe": "ok",
    }
