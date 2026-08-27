"""Tests for identity, title normalization, and cross-query de-duplication."""

from datetime import UTC, datetime

import pytest

from arxiv_digest.normalization import (
    count_term,
    deduplicate_papers,
    normalized_title_key,
    parse_arxiv_identity,
    tokenize,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://arxiv.org/abs/2607.12345v2", ("2607.12345", 2)),
        ("cond-mat/0501234v3", ("cond-mat/0501234", 3)),
    ],
)
def test_parse_arxiv_identity(value: str, expected: tuple[str, int]) -> None:
    assert parse_arxiv_identity(value) == expected


def test_invalid_arxiv_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid versioned"):
        parse_arxiv_identity("https://example.com/not-an-arxiv-id")


def test_title_key_ignores_trivial_punctuation_and_spacing() -> None:
    assert normalized_title_key("Spin-Ice: Dynamics") == normalized_title_key(
        " spin ice — dynamics! "
    )


def test_deduplicate_prefers_new_version_and_merges_categories(paper_factory) -> None:  # type: ignore[no-untyped-def]
    old = paper_factory(categories=["cond-mat.str-el"])
    new = paper_factory(
        version=2,
        updated_at=datetime(2026, 7, 28, tzinfo=UTC),
        categories=["cond-mat.str-el", "physics.comp-ph"],
    )
    title_variant = paper_factory(
        arxiv_id="2607.99999",
        version=5,
        title="A spin-ice paper!",
        published_at=datetime(2026, 7, 24, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, tzinfo=UTC),
    )
    result = deduplicate_papers([old, title_variant, new])
    assert len(result) == 1
    assert result[0].version == 2
    assert result[0].categories == ["cond-mat.str-el", "physics.comp-ph"]


LATEX_HO = "Ho$_2$Ti$_2$O$_7$"
UNICODE_HO = "Ho\u2082Ti\u2082O\u2087"
MATHRM_HO = chr(92) + "mathrm{Ho}_2" + chr(92) + "mathrm{Ti}_2" + chr(92) + "mathrm{O}_7"


@pytest.mark.parametrize("written", [LATEX_HO, UNICODE_HO, "Ho2Ti2O7", MATHRM_HO])
def test_every_way_arxiv_writes_a_formula_folds_to_one_token(written: str) -> None:
    """The configured material must match however the author typed it."""
    assert tokenize(written) == ("ho2ti2o7",)
    assert count_term(tokenize(written), "Ho2Ti2O7") == 1


def test_a_formula_written_in_a_profile_matches_a_paper_written_in_latex() -> None:
    """Matching is symmetric: the configured term is normalized the same way."""
    tokens = tokenize("Magnetism of ZnFe$_2$O$_4$ spinels")
    assert count_term(tokens, "ZnFe2O4") == 1
    assert count_term(tokens, "ZnFe\u2082O\u2084") == 1


def test_real_arxiv_titles_yield_their_formulae_as_single_tokens() -> None:
    """Both formula-bearing titles from the committed window, verbatim."""
    kagome = tokenize(
        "Signatures of field-induced multi-color kagome spin liquids "
        "in the dipole-octupole pyrochlore Ce$_2$Hf$_2$O$_7$"
    )
    assert "ce2hf2o7" in kagome
    hematite = tokenize(
        "GHz non-reciprocal optical conductivity in hematite "
        "($\u03b1$-$" + chr(92) + "text{Fe}_2$O$_3$)"
    )
    assert "fe2o3" in hematite


def test_ordinary_prose_is_unaffected_by_formula_folding() -> None:
    assert tokenize("Dynamics in spin ice") == ("dynamics", "in", "spin", "ice")
    assert count_term(tokenize("A study of spin ice"), "spin ice") == 1
    assert count_term(tokenize("Spineless optimization"), "spinel") == 0


def test_fractional_stoichiometry_still_splits_on_the_solidus() -> None:
    """A known limit, recorded rather than hidden.

    ``V$_{1/3}$NbS$_2$`` normalizes to ``V1/3NbS2`` and the solidus splits it.
    Removing solidi outright would merge ordinary prose like "and/or", so this
    waits for the richer surface-form handling in profile compilation.
    """
    assert tokenize("altermagnetism in V$_{1/3}$NbS$_2$")[-2:] == ("v1", "3nbs2")
