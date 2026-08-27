"""Strict YAML configuration loading with environment-variable overrides."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from enum import StrEnum
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


class DeliveryConfig(StrictModel):
    """Optional SMTP settings; sending still requires an explicit CLI request."""

    enabled: bool = False
    smtp_host: str | None = None
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    from_email: str | None = None
    to_emails: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    @field_validator("to_emails", mode="before")
    @classmethod
    def parse_recipients(cls, value: Any) -> Any:
        """Accept YAML lists or comma-separated environment values."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("smtp_host", "smtp_username", "smtp_password", "from_email")
    @classmethod
    def reject_blank_optional_values(cls, value: str | None) -> str | None:
        """Reject whitespace-only SMTP values while allowing omitted settings."""
        if value is not None and not value.strip():
            raise ValueError("SMTP string settings cannot be blank")
        return value

    @field_validator("from_email")
    @classmethod
    def validate_sender(cls, value: str | None) -> str | None:
        """Require an email-shaped sender when configured."""
        if value is not None and not _email_shaped(value):
            raise ValueError("from_email must be an email-shaped address")
        return value

    @field_validator("to_emails")
    @classmethod
    def validate_recipients(cls, values: list[str]) -> list[str]:
        """Require every configured recipient to be email-shaped and unique."""
        if any(not _email_shaped(value) for value in values):
            raise ValueError("to_emails must contain only email-shaped addresses")
        if len(values) != len(set(values)):
            raise ValueError("to_emails cannot contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_transport_and_auth(self) -> DeliveryConfig:
        """Prevent ambiguous transport security and half-configured authentication."""
        if self.smtp_use_tls and self.smtp_use_ssl:
            raise ValueError("smtp_use_tls and smtp_use_ssl cannot both be enabled")
        if (self.smtp_username is None) != (self.smtp_password is None):
            raise ValueError("smtp_username and smtp_password must be configured together")
        return self


def _email_shaped(value: str) -> bool:
    local, separator, domain = value.strip().partition("@")
    return bool(local and separator and domain and " " not in value)


class AppConfig(StrictModel):
    """Operational application configuration."""

    paths: PathsConfig = PathsConfig()
    arxiv: ArxivConfig = ArxivConfig()
    logging: LoggingConfig = LoggingConfig()
    summarization: SummarizationConfig = SummarizationConfig()
    reporting: ReportingConfig = ReportingConfig()
    delivery: DeliveryConfig = DeliveryConfig()


class RankingWeights(StrictModel):
    """Editable relevance weights; values are normalized when loaded.

    Only relevance components appear here. Recency is a multiplier, not a
    weighted term, and novelty is gone: as a weighted component contributing a
    flat 0.10 to every unseen paper it guaranteed any recent paper a score floor
    that no amount of irrelevance could fall below.
    """

    semantic_relevance: float = Field(ge=0)
    keyword_relevance: float = Field(ge=0)
    category_relevance: float = Field(ge=0)

    @model_validator(mode="after")
    def normalize(self) -> RankingWeights:
        """Normalize non-negative weights to sum to one."""
        fields = (
            "semantic_relevance",
            "keyword_relevance",
            "category_relevance",
        )
        total = sum(getattr(self, field) for field in fields)
        if total <= 0:
            raise ValueError("at least one ranking weight must be positive")
        for field in fields:
            object.__setattr__(self, field, getattr(self, field) / total)
        return self


class FacetWeights(StrictModel):
    """Relative evidential strength of each kind of configured term.

    A paper matching two techniques and no topic is usually less relevant than
    one matching a topic plus a material, so the facets a profile already
    declares should not be flattened together. These are proto-facets: the full
    facet model arrives with researcher profiles in Phase 4.

    PROVISIONAL: chosen by argument, not calibrated. Revisit against a labelled
    evaluation set.
    """

    exact_phrases: float = Field(default=1.0, ge=0.0, le=1.0)
    materials: float = Field(default=1.0, ge=0.0, le=1.0)
    methods: float = Field(default=0.8, ge=0.0, le=1.0)


class ScoringConfig(StrictModel):
    """Tunables for the deterministic relevance score."""

    keyword_saturation: float = Field(default=2.5, gt=0.0)
    """Evidence needed for a mid-range keyword score.

    The score is ``1 - exp(-evidence / keyword_saturation)`` where one match on
    a top-weighted term in the title contributes 1.0 of evidence. At 2.5, three
    such matches score about 0.70.

    PROVISIONAL: revisit against a labelled evaluation set (design report 19.5).
    """

    facet_weights: FacetWeights = FacetWeights()

    recency_weight: float = Field(default=0.05, ge=0.0, le=0.5)
    """How much a paper's position in the window may modulate its score.

    Applied as ``1 - recency_weight + recency_weight * recency``, so recency can
    break a tie between comparable papers but can never manufacture a score for
    an irrelevant one. As an additive component worth 0.10 it did exactly that.
    """

    in_field_category_floor: float = Field(default=0.4, ge=0.0, le=1.0)
    """Category score for an in-field paper whose subcategory the profile omits.

    The group profile says which categories are the field; the research profile
    says which parts of it are most interesting. A cond-mat paper in an unlisted
    subcategory is not off-topic, it is merely unlisted, and scoring it 0.0
    overstates the evidence against it now that category is a third of relevance.
    """

    context_penalty: float = Field(default=0.85, gt=0.0, le=1.0)
    """Multiplier for a paper using an ambiguous term without supporting context.

    PROVISIONAL. The gate flags rather than rejects these (design report 6.3);
    this is where the flag is paid for.
    """


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


_CATEGORY_PATTERN = re.compile(r"^[A-Za-z0-9-]+(?:\.(?:[A-Za-z0-9-]+|\*))?$")


def normalize_category(value: str) -> str:
    """Case-fold an arXiv category or category pattern for stable comparison."""
    return value.strip().casefold()


def category_matches(pattern: str, category: str) -> bool:
    """Match one arXiv category against a configured pattern.

    Three shapes occur in real arXiv metadata and all three must work: exact
    archives and subjects (``hep-ph``, ``cond-mat.str-el``), wildcards over an
    archive (``cond-mat.*``), and the bare archive name that older cross-lists
    still carry (``cond-mat``). A wildcard therefore also matches the bare
    archive, otherwise legacy cross-lists would escape the domain gate.
    """
    normalized_pattern = normalize_category(pattern)
    normalized_category = normalize_category(category)
    if normalized_pattern.endswith(".*"):
        archive = normalized_pattern[:-2]
        return normalized_category == archive or normalized_category.startswith(f"{archive}.")
    return normalized_category == normalized_pattern


def matches_any(patterns: Sequence[str], categories: Sequence[str]) -> list[str]:
    """Return the categories matched by any configured pattern, in input order."""
    return [
        category
        for category in categories
        if any(category_matches(pattern, category) for pattern in patterns)
    ]


class DomainClass(StrEnum):
    """How the domain gate classifies one paper's category list."""

    INCLUDE = "include"
    SOFT = "soft"
    EXCLUDE = "exclude"
    UNKNOWN = "unknown"


class DomainConfig(StrictModel):
    """The group's stable statement of which arXiv categories are its field.

    This is deliberately not derived from individual interests: one member's
    adjacent interest must not silently widen the gate for everyone.
    """

    include_categories: list[str] = Field(min_length=1)
    soft_categories: list[str] = Field(default_factory=list)
    exclude_categories: list[str] = Field(default_factory=list)

    @field_validator("include_categories", "soft_categories", "exclude_categories")
    @classmethod
    def validate_patterns(cls, values: list[str]) -> list[str]:
        """Require well-formed, unique, non-blank category patterns.

        Patterns keep their authored case. arXiv writes subject classes in mixed
        case (``cs.LG``, ``astro-ph.CO``) and these strings are emitted verbatim
        into ``cat:`` query clauses; comparison folds case separately.
        """
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("category patterns cannot be blank")
        invalid = [value for value in stripped if not _CATEGORY_PATTERN.fullmatch(value)]
        if invalid:
            raise ValueError(f"malformed arXiv category patterns: {', '.join(sorted(invalid))}")
        folded = [normalize_category(value) for value in stripped]
        if len(folded) != len(set(folded)):
            raise ValueError("category patterns cannot repeat within one list")
        return stripped

    @model_validator(mode="after")
    def validate_disjoint(self) -> DomainConfig:
        """Reject a pattern configured in more than one list as ambiguous intent."""
        seen: dict[str, str] = {}
        for label, patterns in (
            ("include_categories", self.include_categories),
            ("soft_categories", self.soft_categories),
            ("exclude_categories", self.exclude_categories),
        ):
            for pattern in patterns:
                folded = normalize_category(pattern)
                if folded in seen:
                    raise ValueError(
                        f"category pattern {pattern!r} appears in both {seen[folded]} and {label}"
                    )
                seen[folded] = label
        return self

    def included(self, categories: Sequence[str]) -> list[str]:
        """Return the paper categories that fall inside the group's field."""
        return matches_any(self.include_categories, categories)

    def soft(self, categories: Sequence[str]) -> list[str]:
        """Return the paper categories that are adjacent rather than in-field."""
        return matches_any(self.soft_categories, categories)

    def excluded(self, categories: Sequence[str]) -> list[str]:
        """Return the paper categories the group considers out of field."""
        return matches_any(self.exclude_categories, categories)

    def classify(self, categories: Sequence[str]) -> DomainClass:
        """Classify a paper by its full category list.

        Inclusion wins over exclusion by design. A paper cross-listed into the
        group's own archive is in scope no matter what else it is filed under,
        which is what keeps genuine cross-disciplinary work reachable.
        """
        if self.included(categories):
            return DomainClass.INCLUDE
        if self.soft(categories):
            return DomainClass.SOFT
        if self.excluded(categories):
            return DomainClass.EXCLUDE
        return DomainClass.UNKNOWN


class GroupProfile(StrictModel):
    """Lab-level profile. Phase 1 uses only the domain gate."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    domain: DomainConfig


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
    scoring: ScoringConfig = ScoringConfig()
    queries: list[QueryConfig] = Field(min_length=1)


class Settings(BaseSettings):
    """Combined settings, with nested environment variables taking precedence."""

    app: AppConfig
    profile: ResearchProfile
    group: GroupProfile

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


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read one YAML file that must contain a mapping, with actionable errors."""
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
    delivery_aliases = {
        "SMTP_ENABLED": "enabled",
        "SMTP_HOST": "smtp_host",
        "SMTP_PORT": "smtp_port",
        "SMTP_USERNAME": "smtp_username",
        "SMTP_PASSWORD": "smtp_password",
        "SMTP_USE_TLS": "smtp_use_tls",
        "SMTP_USE_SSL": "smtp_use_ssl",
        "SMTP_TIMEOUT_SECONDS": "timeout_seconds",
        "DIGEST_FROM_EMAIL": "from_email",
        "DIGEST_TO_EMAIL": "to_emails",
    }
    for environment_name, field_name in delivery_aliases.items():
        if delivery_raw := os.environ.get(environment_name):
            app = overrides.setdefault("app", {})
            if not isinstance(app, dict):
                raise ConfigurationError("Conflicting environment override path: app")
            delivery = app.setdefault("delivery", {})
            if not isinstance(delivery, dict):
                raise ConfigurationError("Conflicting environment override path: app.delivery")
            delivery[field_name] = yaml.safe_load(delivery_raw)
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
    group_path: Path = Path("profiles/group.yaml"),
) -> Settings:
    """Load YAML files, then apply nested ``ARXIV_DIGEST_`` environment overrides."""
    try:
        combined = {
            "app": read_yaml_mapping(app_path),
            "profile": read_yaml_mapping(profile_path),
            "group": read_yaml_mapping(group_path),
        }
        merged = _deep_merge(combined, _environment_overrides())
        app_config = AppConfig.model_validate(merged["app"])
        profile = ResearchProfile.model_validate(merged["profile"])
        group = GroupProfile.model_validate(merged["group"])
        return Settings(app=app_config, profile=profile, group=group)
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
