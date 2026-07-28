"""Strict YAML configuration loading with environment-variable overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from arxiv_digest.exceptions import ConfigurationError
from arxiv_digest.models import StrictModel


class PathsConfig(StrictModel):
    """Filesystem locations used by current and future pipeline stages."""

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    history_file: Path = Path("data/history.jsonl")
    templates_dir: Path = Path("templates")


class ArxivConfig(StrictModel):
    """Operational limits and connection settings for the official arXiv API."""

    api_url: str = "https://export.arxiv.org/api/query"
    contact_email: str = "researcher@example.org"
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    read_timeout_seconds: float = Field(default=30.0, gt=0)
    page_size: int = Field(default=50, ge=1, le=100)
    retry_count: int = Field(default=3, ge=0, le=8)
    retry_backoff_seconds: float = Field(default=2.0, gt=0)
    rate_limit_seconds: float = Field(default=3.0, ge=3.0)
    overlap_days: int = Field(default=1, ge=0, le=7)

    @field_validator("contact_email")
    @classmethod
    def validate_contact_email(cls, value: str) -> str:
        """Require an email-shaped contact without imposing deliverability checks."""
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError("contact_email must be an email-shaped address")
        return value


class LoggingConfig(StrictModel):
    """Standard-library logging settings."""

    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        """Accept the conventional Python logging levels."""
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("unsupported logging level")
        return normalized


class SummarizationConfig(StrictModel):
    """Offline-first summary provider settings with optional OpenAI model names."""

    provider: Literal["offline", "openai"] = "offline"
    max_provider_papers: int = Field(default=12, ge=1, le=50)
    validation_retries: int = Field(default=2, ge=0, le=4)
    openai_summary_model: str | None = None
    openai_embedding_model: str | None = None

    @field_validator("openai_summary_model", "openai_embedding_model")
    @classmethod
    def reject_blank_models(cls, value: str | None) -> str | None:
        """Treat omitted model names distinctly from invalid blank names."""
        if value is not None and not value.strip():
            raise ValueError("OpenAI model names cannot be blank")
        return value


class ReportingConfig(StrictModel):
    """Reader-facing report output settings."""

    timezone: str = "UTC"
    near_miss_limit: int = Field(default=5, ge=0, le=5)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        """Require a timezone available to the standard-library zoneinfo database."""
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown IANA timezone: {value}") from exc
        return value


class AppConfig(StrictModel):
    """Operational application configuration."""

    paths: PathsConfig = PathsConfig()
    arxiv: ArxivConfig = ArxivConfig()
    logging: LoggingConfig = LoggingConfig()
    summarization: SummarizationConfig = SummarizationConfig()
    reporting: ReportingConfig = ReportingConfig()


class RankingWeights(StrictModel):
    """Editable future ranking weights; values are normalized when loaded."""

    semantic_relevance: float = Field(ge=0)
    keyword_relevance: float = Field(ge=0)
    category_relevance: float = Field(ge=0)
    recency: float = Field(ge=0)
    feedback_or_novelty: float = Field(ge=0)

    @model_validator(mode="after")
    def normalize(self) -> RankingWeights:
        """Normalize non-negative weights to sum to one."""
        fields = (
            "semantic_relevance",
            "keyword_relevance",
            "category_relevance",
            "recency",
            "feedback_or_novelty",
        )
        total = sum(getattr(self, field) for field in fields)
        if total <= 0:
            raise ValueError("at least one ranking weight must be positive")
        for field in fields:
            object.__setattr__(self, field, getattr(self, field) / total)
        return self


class RecommendationMix(StrictModel):
    """Target counts for a later diversity-aware selection stage."""

    direct: int = Field(ge=0)
    adjacent: int = Field(ge=0)
    wildcard: int = Field(ge=0)


class SelectionConfig(StrictModel):
    """Deterministic MMR and recommendation-strength thresholds."""

    mmr_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    direct_min_score: float = Field(default=0.28, ge=0.0, le=1.0)
    adjacent_min_score: float = Field(default=0.20, ge=0.0, le=1.0)
    wildcard_min_score: float = Field(default=0.12, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> SelectionConfig:
        """Require strength buckets to descend from direct to wildcard."""
        if not (self.direct_min_score >= self.adjacent_min_score >= self.wildcard_min_score):
            raise ValueError(
                "selection score thresholds must satisfy direct >= adjacent >= wildcard"
            )
        return self


class QueryConfig(StrictModel):
    """One broad, editable arXiv candidate query."""

    name: str = Field(min_length=1)
    terms: list[str] = Field(min_length=1)

    @field_validator("terms")
    @classmethod
    def reject_blank_terms(cls, values: list[str]) -> list[str]:
        """Prevent a blank term from broadening a query unexpectedly."""
        if any(not value.strip() for value in values):
            raise ValueError("query terms cannot be blank")
        return values


class ResearchProfile(StrictModel):
    """User-editable research interests and future ranking preferences."""

    name: str
    version: str
    description: str
    priority_topics: list[str]
    exact_phrases: dict[str, float]
    materials: dict[str, float]
    methods: dict[str, float]
    categories: dict[str, float]
    negative_terms: list[str]
    author_boosts: dict[str, float]
    ranking_weights: RankingWeights
    recommendation_mix: RecommendationMix
    history_exclusion_days: int = Field(ge=0)
    allow_updated_resurfacing: bool
    max_candidate_count: int = Field(ge=1, le=1000)
    max_papers_per_topic: int = Field(ge=1)
    selection: SelectionConfig = SelectionConfig()
    queries: list[QueryConfig] = Field(min_length=1)


class Settings(BaseSettings):
    """Combined settings, with nested environment variables taking precedence."""

    app: AppConfig
    profile: ResearchProfile

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="ARXIV_DIGEST_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Place environment values ahead of YAML-derived init values."""
        del settings_cls, env_settings, dotenv_settings, file_secret_settings
        return (init_settings,)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration file {path}: {exc}") from exc
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration file {path} must contain a mapping")
    return value


def _environment_overrides() -> dict[str, Any]:
    """Parse nested ``ARXIV_DIGEST_`` variables into a validation-ready mapping."""
    prefix = "ARXIV_DIGEST_"
    overrides: dict[str, Any] = {}
    for name, raw_value in os.environ.items():
        if not name.startswith(prefix):
            continue
        path = [part.lower() for part in name[len(prefix) :].split("__") if part]
        if not path:
            continue
        target = overrides
        for part in path[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigurationError(f"Conflicting environment override path: {name}")
            target = child
        target[path[-1]] = yaml.safe_load(raw_value)
    model_aliases = {
        "OPENAI_SUMMARY_MODEL": "openai_summary_model",
        "OPENAI_EMBEDDING_MODEL": "openai_embedding_model",
    }
    for environment_name, field_name in model_aliases.items():
        if value := os.environ.get(environment_name):
            app = overrides.setdefault("app", {})
            if not isinstance(app, dict):
                raise ConfigurationError("Conflicting environment override path: app")
            summarization = app.setdefault("summarization", {})
            if not isinstance(summarization, dict):
                raise ConfigurationError("Conflicting environment override path: app.summarization")
            summarization[field_name] = value
    return overrides


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested configuration mappings without mutating inputs."""
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_settings(
    app_path: Path = Path("config/app.yaml"),
    profile_path: Path = Path("config/research_profile.yaml"),
) -> Settings:
    """Load YAML files, then apply nested ``ARXIV_DIGEST_`` environment overrides."""
    try:
        combined = {
            "app": _read_yaml(app_path),
            "profile": _read_yaml(profile_path),
        }
        merged = _deep_merge(combined, _environment_overrides())
        app_config = AppConfig.model_validate(merged["app"])
        profile = ResearchProfile.model_validate(merged["profile"])
        return Settings(app=app_config, profile=profile)
    except ValidationError as exc:
        raise ConfigurationError(f"Configuration validation failed: {exc}") from exc


_SENSITIVE_FRAGMENTS = ("password", "secret", "token", "api_key", "credential")


def redact_mapping(value: Any) -> Any:
    """Recursively redact values whose keys look sensitive."""
    if isinstance(value, dict):
        return {
            key: "***REDACTED***"
            if any(fragment in str(key).lower() for fragment in _SENSITIVE_FRAGMENTS)
            else redact_mapping(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    return value


def settings_as_json(settings: Settings) -> str:
    """Serialize validated settings without exposing any future secret fields."""
    safe = redact_mapping(settings.model_dump(mode="json"))
    return json.dumps(safe, indent=2, sort_keys=True, ensure_ascii=False)
