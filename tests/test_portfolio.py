"""Tests for agent/portfolio.py — CSV loading, validation, and DB sync."""

import textwrap
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from agent.portfolio import (
    Holding,
    WatchlistEntry,
    load_portfolio,
    load_watchlist,
    sync_holdings_to_db,
)
from data_sources.errors import DataSourceError
from data_sources.yfinance_client import PriceData
from storage.models import Holding as HoldingRow

# ── Helpers ───────────────────────────────────────────────────────────────────


def _price(ticker: str = "AAPL") -> PriceData:
    return PriceData(
        ticker=ticker,
        current_price=182.5,
        previous_close=180.0,
        day_change_pct=1.39,
        volume=55_000_000,
        as_of=datetime.now(timezone.utc),
        data_age_hours=0,
    )


def _not_found(ticker: str) -> DataSourceError:
    return DataSourceError(error_code="not_found", message=f"No price data for {ticker}")


class _FakeYF:
    """Minimal yfinance client fake for ticker-validation tests."""

    def __init__(self, valid_tickers: set[str]) -> None:
        self._valid = valid_tickers

    def get_price(self, ticker: str) -> PriceData | DataSourceError:
        if ticker in self._valid:
            return _price(ticker)
        return _not_found(ticker)


def _csv(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


# ── load_portfolio ─────────────────────────────────────────────────────────────


def test_load_portfolio_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv = _csv(
        tmp_path,
        "portfolio.csv",
        """\
        ticker,shares,cost_basis,purchase_date
        AAPL,10,150.00,2023-01-15
        MSFT,5,280.50,2022-11-30
        """,
    )
    monkeypatch.setattr("agent.portfolio.yfinance_client", lambda: _FakeYF({"AAPL", "MSFT"}))
    holdings = load_portfolio(csv)
    assert len(holdings) == 2
    expected = Holding(
        ticker="AAPL", shares=10.0, cost_basis=150.0, purchase_date=date(2023, 1, 15)
    )
    assert holdings[0] == expected
    assert holdings[1].ticker == "MSFT"


def test_load_portfolio_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Portfolio CSV not found"):
        load_portfolio(tmp_path / "nonexistent.csv")


def test_load_portfolio_negative_shares(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv = _csv(
        tmp_path,
        "portfolio.csv",
        """\
        ticker,shares,cost_basis,purchase_date
        AAPL,-5,150.00,2023-01-15
        """,
    )
    monkeypatch.setattr("agent.portfolio.yfinance_client", lambda: _FakeYF({"AAPL"}))
    with pytest.raises(ValueError, match="Portfolio CSV validation failed"):
        load_portfolio(csv, validate_tickers=False)


def test_load_portfolio_invalid_ticker_pattern(tmp_path: Path) -> None:
    csv = _csv(
        tmp_path,
        "portfolio.csv",
        """\
        ticker,shares,cost_basis,purchase_date
        aapl,10,150.00,2023-01-15
        """,
    )
    with pytest.raises(ValueError, match="Portfolio CSV validation failed"):
        load_portfolio(csv, validate_tickers=False)


def test_load_portfolio_duplicate_tickers(tmp_path: Path) -> None:
    csv = _csv(
        tmp_path,
        "portfolio.csv",
        """\
        ticker,shares,cost_basis,purchase_date
        AAPL,10,150.00,2023-01-15
        AAPL,5,160.00,2023-06-01
        """,
    )
    with pytest.raises(ValueError, match="Duplicate tickers.*AAPL"):
        load_portfolio(csv, validate_tickers=False)


def test_load_portfolio_collects_all_bad_rows(tmp_path: Path) -> None:
    csv = _csv(
        tmp_path,
        "portfolio.csv",
        """\
        ticker,shares,cost_basis,purchase_date
        AAPL,-1,150.00,2023-01-15
        MSFT,-2,280.00,2022-11-30
        """,
    )
    with pytest.raises(ValueError) as exc_info:
        load_portfolio(csv, validate_tickers=False)
    msg = str(exc_info.value)
    assert "row 2" in msg
    assert "row 3" in msg


def test_load_portfolio_invalid_ticker_via_yfinance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv = _csv(
        tmp_path,
        "portfolio.csv",
        """\
        ticker,shares,cost_basis,purchase_date
        AAPL,10,150.00,2023-01-15
        FAKE,5,10.00,2023-01-01
        BOGUS,3,5.00,2023-01-01
        """,
    )
    monkeypatch.setattr("agent.portfolio.yfinance_client", lambda: _FakeYF({"AAPL"}))
    with pytest.raises(ValueError) as exc_info:
        load_portfolio(csv, validate_tickers=True)
    msg = str(exc_info.value)
    assert "BOGUS" in msg
    assert "FAKE" in msg


def test_load_portfolio_skip_ticker_validation(tmp_path: Path) -> None:
    csv = _csv(
        tmp_path,
        "portfolio.csv",
        """\
        ticker,shares,cost_basis,purchase_date
        AAPL,10,150.00,2023-01-15
        """,
    )
    holdings = load_portfolio(csv, validate_tickers=False)
    assert len(holdings) == 1


# ── load_watchlist ─────────────────────────────────────────────────────────────


def test_load_watchlist_happy_path(tmp_path: Path) -> None:
    csv = _csv(
        tmp_path,
        "watchlist.csv",
        """\
        ticker,notes
        COST,Membership model moat
        V,Network effect
        """,
    )
    entries = load_watchlist(csv)
    assert len(entries) == 2
    assert entries[0] == WatchlistEntry(ticker="COST", notes="Membership model moat")


def test_load_watchlist_empty_notes(tmp_path: Path) -> None:
    csv = _csv(
        tmp_path,
        "watchlist.csv",
        """\
        ticker,notes
        COST,
        V,
        """,
    )
    entries = load_watchlist(csv)
    assert all(e.notes == "" for e in entries)


def test_load_watchlist_missing_notes_column(tmp_path: Path) -> None:
    csv = _csv(
        tmp_path,
        "watchlist.csv",
        """\
        ticker
        COST
        V
        """,
    )
    entries = load_watchlist(csv)
    assert all(e.notes == "" for e in entries)


def test_load_watchlist_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Watchlist CSV not found"):
        load_watchlist(tmp_path / "nonexistent.csv")


def test_load_watchlist_invalid_ticker_pattern(tmp_path: Path) -> None:
    csv = _csv(
        tmp_path,
        "watchlist.csv",
        """\
        ticker,notes
        cost,lowercase ticker
        """,
    )
    with pytest.raises(ValueError, match="Watchlist CSV validation failed"):
        load_watchlist(csv)


# ── sync_holdings_to_db ────────────────────────────────────────────────────────


def test_sync_holdings_to_db_writes_exact_rows(db_engine: Engine) -> None:
    holdings = [
        Holding(ticker="AAPL", shares=10.0, cost_basis=150.0, purchase_date=date(2023, 1, 15)),
        Holding(ticker="MSFT", shares=5.0, cost_basis=280.5, purchase_date=date(2022, 11, 30)),
    ]
    prices = {"AAPL": 182.5, "MSFT": 310.0}

    with Session(db_engine) as session:
        sync_holdings_to_db(holdings, session, prices)

    with Session(db_engine) as session:
        rows = session.execute(select(HoldingRow)).scalars().all()

    assert len(rows) == 2
    by_ticker = {r.ticker: r for r in rows}
    assert by_ticker["AAPL"].shares == 10.0
    assert by_ticker["AAPL"].cost_basis == 150.0
    assert by_ticker["AAPL"].purchase_date == date(2023, 1, 15)
    assert by_ticker["AAPL"].current_price == 182.5
    assert by_ticker["MSFT"].current_price == 310.0


def test_sync_holdings_to_db_overwrites_previous(db_engine: Engine) -> None:
    old = [Holding(ticker="AAPL", shares=10.0, cost_basis=150.0, purchase_date=date(2023, 1, 15))]
    new = [Holding(ticker="MSFT", shares=5.0, cost_basis=280.5, purchase_date=date(2022, 11, 30))]

    with Session(db_engine) as session:
        sync_holdings_to_db(old, session, {})
    with Session(db_engine) as session:
        sync_holdings_to_db(new, session, {})

    with Session(db_engine) as session:
        rows = session.execute(select(HoldingRow)).scalars().all()

    assert len(rows) == 1
    assert rows[0].ticker == "MSFT"


def test_sync_holdings_to_db_missing_price_stored_as_none(db_engine: Engine) -> None:
    holdings = [
        Holding(ticker="AAPL", shares=10.0, cost_basis=150.0, purchase_date=date(2023, 1, 15))
    ]

    with Session(db_engine) as session:
        sync_holdings_to_db(holdings, session, {})

    with Session(db_engine) as session:
        row = session.execute(select(HoldingRow)).scalar_one()

    assert row.current_price is None
