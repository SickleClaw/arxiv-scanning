"""Measured per-term ambiguity, cached indefinitely.

A term's ambiguity is a property of the literature, not a matter of opinion, and
arXiv will report it directly: ask how many papers use the term, then how many
of those are in the group's field. `spin ice` comes back at 0.01 and `magnetic
monopole` at 0.76, which is the whole disambiguation problem in two numbers.

This turns a hand-maintained list of dangerous terms into a measurement. When a
new member adds `skyrmion`, `Majorana`, or `holography`, the risk is flagged
without anyone having to have anticipated that particular word.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field, ValidationError, field_validator

from arxiv_digest.arxiv_client import ArxivClient, quote_term
from arxiv_digest.config import DomainConfig
from arxiv_digest.exceptions import ConfigurationError
from arxiv_digest.models import StrictModel

TRUSTED_BELOW = 0.15
"""At or below this, a term is unambiguous enough to trust for lexical matching."""

GUARD_ABOVE = 0.40
"""Above this, a term needs a context guard and a reduced lexical weight."""

DEFAULT_TABLE_PATH = Path("data/term_ambiguity.json")


class AmbiguityRecord(StrictModel):
    """One measured term."""

    term: str = Field(min_length=1)
    total_results: int = Field(ge=0)
    in_field_results: int = Field(ge=0)
    measured_at: datetime

    @field_validator("measured_at")
    @classmethod
    def require_aware(cls, value: datetime) -> datetime:
        """Keep timestamps comparable across machines."""
        if value.tzinfo is None:
            raise ValueError("measured_at must include a timezone")
        return value

    @property
    def ambiguity(self) -> float:
        """Fraction of this term's arXiv hits that fall outside the group's field."""
        if self.total_results <= 0:
            return 0.0
        return 1.0 - (self.in_field_results / self.total_results)

    @property
    def needs_context_guard(self) -> bool:
        """Whether this term is ambiguous enough to require a context requirement."""
        return self.ambiguity > GUARD_ABOVE

    def describe(self) -> str:
        """Render the measurement the way the warning should read to a physicist."""
        percent = (
            0.0 if self.total_results == 0 else 100.0 * self.in_field_results / self.total_results
        )
        return (
            f"{self.term!r} matches {self.total_results} arXiv papers, "
            f"only {percent:.0f}% in the group's field (ambiguity {self.ambiguity:.2f})"
        )


class AmbiguityTable(StrictModel):
    """A cached set of measurements, keyed by the domain they were measured against."""

    domain_signature: str = Field(min_length=1)
    records: dict[str, AmbiguityRecord] = Field(default_factory=dict)

    def stale_for(self, domain: DomainConfig) -> bool:
        """Whether the domain has changed since these numbers were measured."""
        return self.domain_signature != domain_signature(domain)


def domain_signature(domain: DomainConfig) -> str:
    """Identify the in-field category set a measurement was taken against."""
    return "|".join(sorted(pattern.casefold() for pattern in domain.include_categories))


def in_field_query(term: str, domain: DomainConfig) -> str:
    """Build the counting query for the in-field share of one term's hits.

    In-field categories only, deliberately. Retrieval also admits the adjacent
    ones, but this measurement answers "is this term's usage predominantly our
    field?", and counting broad neighbours like ``cs.LG`` toward that would make
    every term look safer than it is.
    """
    guard = " OR ".join(f"cat:{pattern}" for pattern in domain.include_categories)
    return f"{quote_term(term)} AND ({guard})"


def measure_term(client: ArxivClient, term: str, domain: DomainConfig) -> AmbiguityRecord:
    """Measure one term with two counting queries."""
    return AmbiguityRecord(
        term=term,
        total_results=client.count(quote_term(term)),
        in_field_results=client.count(in_field_query(term, domain)),
        measured_at=datetime.now(UTC),
    )


def measure_terms(
    client: ArxivClient,
    terms: Sequence[str],
    domain: DomainConfig,
    *,
    cached: AmbiguityTable | None = None,
    refresh: bool = False,
) -> AmbiguityTable:
    """Measure every term, reusing cached values unless asked to refresh.

    Two requests per term, cached indefinitely: a term's ambiguity moves at the
    speed of the literature, not of a weekly run.
    """
    signature = domain_signature(domain)
    reusable = (
        {}
        if refresh or cached is None or cached.domain_signature != signature
        else dict(cached.records)
    )
    records = dict(reusable)
    for term in terms:
        if term not in records:
            records[term] = measure_term(client, term, domain)
    return AmbiguityTable(domain_signature=signature, records=records)


def load_table(path: Path = DEFAULT_TABLE_PATH) -> AmbiguityTable | None:
    """Load a cached table; a missing file simply means nothing is measured yet."""
    if not path.exists():
        return None
    try:
        return AmbiguityTable.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise ConfigurationError(f"Invalid term ambiguity table {path}: {exc}") from exc


def save_table(table: AmbiguityTable, path: Path = DEFAULT_TABLE_PATH) -> Path:
    """Atomically persist a measured table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(table.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def unguarded_risky_terms(
    table: AmbiguityTable, configured_guards: Sequence[str]
) -> list[AmbiguityRecord]:
    """Return measured-risky terms that have no context requirement configured.

    Matching is by containment in either direction, so the guard on ``monopole``
    covers the profile's ``magnetic monopole``.
    """
    guards = [guard.casefold() for guard in configured_guards]
    risky = [record for record in table.records.values() if record.needs_context_guard]
    return sorted(
        (
            record
            for record in risky
            if not any(
                guard in record.term.casefold() or record.term.casefold() in guard
                for guard in guards
            )
        ),
        key=lambda record: (-record.ambiguity, record.term),
    )
