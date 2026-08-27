"""Tests for strict layered configuration loading."""

from pathlib import Path

import pytest

from arxiv_digest.config import load_settings, redact_mapping, settings_as_json
from arxiv_digest.exceptions import ConfigurationError


def test_default_configuration_loads_and_weights_sum_to_one() -> None:
    settings = load_settings()
    weights = settings.profile.ranking_weights
    # Authored as 0.45/0.25/0.10 and renormalized over the three relevance
    # components, since recency and novelty are no longer weighted terms.
    assert weights.semantic_relevance == pytest.approx(0.45 / 0.80)
    assert weights.keyword_relevance == pytest.approx(0.25 / 0.80)
    assert weights.category_relevance == pytest.approx(0.10 / 0.80)
    assert sum(
        (
            weights.semantic_relevance,
            weights.keyword_relevance,
            weights.category_relevance,
        )
    ) == pytest.approx(1.0)
    assert len(settings.profile.queries) == 6


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARXIV_DIGEST_APP__ARXIV__PAGE_SIZE", "17")
    monkeypatch.setenv("ARXIV_DIGEST_PROFILE__MAX_CANDIDATE_COUNT", "123")
    monkeypatch.setenv("OPENAI_SUMMARY_MODEL", "configured-summary-model")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "configured-embedding-model")
    monkeypatch.setenv("SMTP_HOST", "smtp.test.invalid")
    monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_USE_TLS", "false")
    monkeypatch.setenv("SMTP_USE_SSL", "true")
    monkeypatch.setenv("SMTP_USERNAME", "digest-user")
    monkeypatch.setenv("SMTP_PASSWORD", "smtp-secret-value")
    monkeypatch.setenv("DIGEST_FROM_EMAIL", "digest@test.invalid")
    monkeypatch.setenv("DIGEST_TO_EMAIL", "one@test.invalid,two@test.invalid")
    settings = load_settings()
    assert settings.app.arxiv.page_size == 17
    assert settings.profile.max_candidate_count == 123
    assert settings.app.summarization.openai_summary_model == "configured-summary-model"
    assert settings.app.summarization.openai_embedding_model == "configured-embedding-model"
    assert settings.app.delivery.smtp_port == 465
    assert settings.app.delivery.smtp_use_ssl is True
    assert settings.app.delivery.to_emails == ["one@test.invalid", "two@test.invalid"]
    serialized = settings_as_json(settings)
    assert "smtp-secret-value" not in serialized
    assert '"smtp_password": "***REDACTED***"' in serialized


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
