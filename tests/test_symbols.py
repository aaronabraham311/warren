"""Canonical symbol convention (data_sources/symbols.py).

Pure string transformation — no network. Each test maps 1:1 to a G2 acceptance
criterion: exchange suffixes are preserved, US share classes are dashed for
Yahoo, and existing US behaviour is unchanged.
"""

import re

import pytest

from data_sources.symbols import (
    KNOWN_EXCHANGE_SUFFIXES,
    TICKER_PATTERN,
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


# ── G3: shared TICKER_PATTERN (the single source of truth for every model) ────────

# Non-US exchange suffixes the gem-hunt slice must accept, plus US baselines.
_VALID_TICKERS = (
    "DIR.MI",
    "CIRSA.MC",
    "KPL.WA",
    "480S.MC",
    "4MB.WA",
    "WAMI28.MI",
    "AAPL",
    "BRK.B",
    "BRK-B",
)
# Things that must still be rejected as obvious garbage.
_INVALID_TICKERS = (
    "aapl",
    "TOOLONG",
    "123",
    "123456.MC",
    "ABC1234.MI",
    "BRK.b",
    "",
    "BRK.",
    ".MI",
    "DIR.MIL",
    "A_BC.WA",
    "ABC!.MI",
)


@pytest.mark.parametrize("ticker", _VALID_TICKERS)
def test_ticker_pattern_accepts_us_and_slice_symbols(ticker: str) -> None:
    assert re.match(TICKER_PATTERN, ticker) is not None


@pytest.mark.parametrize("ticker", _INVALID_TICKERS)
def test_ticker_pattern_rejects_garbage(ticker: str) -> None:
    assert re.match(TICKER_PATTERN, ticker) is None


@pytest.mark.parametrize(
    "ticker",
    ("DIR.MI", "CIRSA.MC", "KPL.WA", "480S.MC", "4MB.WA", "WAMI28.MI", "AAPL", "BRK.B"),
)
def test_suffix_ticker_passes_all_three_validators(ticker: str) -> None:
    # The single shared pattern means AnalysisOutput, Holding/WatchlistEntry, and
    # EvalExample all accept the same non-US symbols — they can never drift again.
    from datetime import date

    from agent.models import AnalysisOutput, LynchBuffettSignals
    from agent.portfolio import Holding, WatchlistEntry
    from eval.golden_set import EvalExample, EvalExpectations, RecommendationExpectation

    analysis = AnalysisOutput(
        ticker=ticker,
        analysis_type="discovery",
        recommendation="hold",
        confidence=0.8,
        thesis="t" * 10,
        lynch_signals=LynchBuffettSignals(pros=[], cons=[]),
        buffett_signals=LynchBuffettSignals(pros=[], cons=[]),
        key_risks=["risk"],
    )
    assert analysis.ticker == ticker
    assert Holding(ticker=ticker, shares=1, cost_basis=1, purchase_date=date(2023, 1, 1)).ticker
    assert WatchlistEntry(ticker=ticker).ticker == ticker
    example = EvalExample(
        ticker=ticker,
        notes="curated",
        last_curated=date(2026, 1, 1),
        expectations=EvalExpectations(
            recommendation=RecommendationExpectation(allowed=["hold"]),
        ),
    )
    assert example.ticker == ticker
