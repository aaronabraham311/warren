"""Unit tests for the FX normalization helper (data_sources/fx.py)."""

import pytest

from data_sources.fx import (
    FALLBACK_FX_RATES,
    normalize_currency,
    to_usd,
)


def test_to_usd_identity_for_usd() -> None:
    assert to_usd(1_000, "USD") == 1_000.0
    # A supplied rate is ignored for USD — always identity.
    assert to_usd(1_000, "USD", rate=1.5) == 1_000.0


def test_to_usd_identity_for_unknown_currency() -> None:
    # Unknown / unsupported currency is never converted (no crash, no guess).
    assert to_usd(1_000, "GBP") == 1_000.0
    assert to_usd(1_000, "JPY", rate=0.007) == 1_000.0


def test_to_usd_identity_for_none_currency() -> None:
    assert to_usd(1_000, None) == 1_000.0
    assert to_usd(1_000, "") == 1_000.0


def test_to_usd_none_amount_returns_none() -> None:
    assert to_usd(None, "EUR") is None
    assert to_usd(None, "USD") is None
    assert to_usd(None, None) is None


def test_to_usd_converts_eur_with_explicit_rate() -> None:
    assert to_usd(100, "EUR", rate=1.10) == pytest.approx(110.0)


def test_to_usd_converts_pln_with_explicit_rate() -> None:
    assert to_usd(100, "PLN", rate=0.25) == pytest.approx(25.0)


def test_to_usd_falls_back_to_committed_table_without_rate() -> None:
    assert to_usd(100, "EUR") == pytest.approx(100 * FALLBACK_FX_RATES["EUR"])
    assert to_usd(100, "PLN") == pytest.approx(100 * FALLBACK_FX_RATES["PLN"])


def test_to_usd_lowercase_currency_normalized() -> None:
    assert to_usd(100, "eur", rate=1.10) == pytest.approx(110.0)


def test_normalize_currency() -> None:
    assert normalize_currency("  eur ") == "EUR"
    assert normalize_currency("USD") == "USD"
    assert normalize_currency(None) is None
    assert normalize_currency("   ") is None
