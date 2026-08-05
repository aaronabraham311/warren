"""Canonical symbol convention (data_sources/symbols.py).

Pure string transformation — no network. Each test maps 1:1 to a G2 acceptance
criterion: exchange suffixes are preserved, US share classes are dashed for
Yahoo, and existing US behaviour is unchanged.
"""

import pytest

from data_sources.symbols import (
    KNOWN_EXCHANGE_SUFFIXES,
    canonical_symbol,
    to_finnhub_symbol,
    to_yahoo_symbol,
)
from data_sources.yfinance_client import _yf_symbol


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        # Exchange suffixes are preserved verbatim (dot kept).
        ("DIR.MI", "DIR.MI"),
        ("CIRSA.MC", "CIRSA.MC"),
        ("KPL.WA", "KPL.WA"),
        # US share classes are dashed for Yahoo.
        ("BRK.B", "BRK-B"),
        ("BF.B", "BF-B"),
        # No dot → unchanged (plain US ticker).
        ("AAPL", "AAPL"),
    ],
)
def test_to_yahoo_symbol_maps_expected(ticker: str, expected: str) -> None:
    assert to_yahoo_symbol(ticker) == expected


def test_yf_symbol_delegates_to_shared_helper() -> None:
    # _yf_symbol is the yfinance-side entry point; it must agree with the shared mapper.
    for ticker in ("DIR.MI", "CIRSA.MC", "KPL.WA", "BRK.B", "BF.B", "AAPL"):
        assert _yf_symbol(ticker) == to_yahoo_symbol(ticker)


def test_yf_symbol_regression_us_behaviour() -> None:
    # Existing US behaviour must be unchanged.
    assert _yf_symbol("BRK.B") == "BRK-B"
    assert _yf_symbol("AAPL") == "AAPL"


def test_canonical_symbol_uppercases_and_trims() -> None:
    assert canonical_symbol("  brk.b ") == "BRK.B"
    assert canonical_symbol("dir.mi") == "DIR.MI"


def test_to_finnhub_symbol_is_canonical_form() -> None:
    # Finnhub keeps dots (both share class and exchange suffix), just uppercased.
    assert to_finnhub_symbol("brk.b") == "BRK.B"
    assert to_finnhub_symbol("DIR.MI") == "DIR.MI"
    assert to_finnhub_symbol("aapl") == "AAPL"


def test_known_exchange_suffixes_cover_slice() -> None:
    assert {"MI", "MC", "WA"} <= KNOWN_EXCHANGE_SUFFIXES


def test_yfinance_and_finnhub_share_canonical_base() -> None:
    # Both data-source mappers derive from the same canonical form: they agree on
    # plain US tickers, and each applies its own source-specific spelling for dots.
    for ticker in ("AAPL", "MSFT"):
        assert _yf_symbol(ticker) == to_finnhub_symbol(ticker) == canonical_symbol(ticker)
    # Share class: yfinance dashes, finnhub keeps the canonical dot.
    assert _yf_symbol("BRK.B") == "BRK-B"
    assert to_finnhub_symbol("BRK.B") == canonical_symbol("BRK.B") == "BRK.B"
