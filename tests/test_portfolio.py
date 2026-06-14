import time
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from agent.portfolio import (
    PortfolioError,
    WatchlistEntry,
    load_portfolio,
    load_watchlist,
    sync_holdings_to_db,
    sync_watchlist_to_db,
)
from storage.models import Holding as HoldingRow
from storage.models import Watchlist as WatchlistRow


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


# ── load_portfolio validation ─────────────────────────────────────────────────


def test_negative_shares_raises(tmp_path: Path) -> None:
    csv = _write(
        tmp_path / "p.csv",
        "ticker,shares,cost_basis,purchase_date\nAAPL,-10,150.00,2023-01-15\n",
    )
    with pytest.raises(PortfolioError):
        load_portfolio(csv, validate_tickers=False)


def test_invalid_ticker_pattern_lists_all_bad_tickers(tmp_path: Path) -> None:
    csv = _write(
        tmp_path / "p.csv",
        "ticker,shares,cost_basis,purchase_date\n"
        "aapl,10,150.00,2023-01-15\n"  # lowercase — fails pattern
        "TOOLONG,5,10.00,2023-01-15\n"  # >5 chars — fails pattern
        "MSFT,5,280.50,2022-11-30\n",  # valid
    )
    with pytest.raises(PortfolioError) as exc:
        load_portfolio(csv, validate_tickers=False)
    msg = str(exc.value)
    # Both bad rows are reported at once.
    assert "aapl" in msg
    assert "TOOLONG" in msg


def test_duplicate_tickers_raises_with_list(tmp_path: Path) -> None:
    csv = _write(
        tmp_path / "p.csv",
        "ticker,shares,cost_basis,purchase_date\n"
        "AAPL,10,150.00,2023-01-15\n"
        "AAPL,5,160.00,2023-02-15\n",
    )
    with pytest.raises(PortfolioError) as exc:
        load_portfolio(csv, validate_tickers=False)
    assert "duplicate" in str(exc.value).lower()
    assert "AAPL" in str(exc.value)


def test_missing_file_raises_filenotfound_with_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError) as exc:
        load_portfolio(missing, validate_tickers=False)
    assert "nope.csv" in str(exc.value)


def test_valid_portfolio_loads(tmp_path: Path) -> None:
    csv = _write(
        tmp_path / "p.csv",
        "ticker,shares,cost_basis,purchase_date\n"
        "AAPL,10,150.00,2023-01-15\n"
        "MSFT,5,280.50,2022-11-30\n",
    )
    holdings = load_portfolio(csv, validate_tickers=False)
    assert [h.ticker for h in holdings] == ["AAPL", "MSFT"]
    assert holdings[0].shares == 10.0
    assert holdings[0].purchase_date.isoformat() == "2023-01-15"


def test_ticker_validation_lists_all_invalid(tmp_path: Path) -> None:
    csv = _write(
        tmp_path / "p.csv",
        "ticker,shares,cost_basis,purchase_date\n"
        "AAPL,10,150.00,2023-01-15\n"
        "FAKE,5,10.00,2023-01-15\n"
        "NOPE,5,10.00,2023-01-15\n",
    )
    valid = {"AAPL"}
    with pytest.raises(PortfolioError) as exc:
        load_portfolio(csv, validate_tickers=True, validator=lambda t: t in valid)
    assert "FAKE" in str(exc.value)
    assert "NOPE" in str(exc.value)


def test_thirty_tickers_load_under_five_seconds(tmp_path: Path) -> None:
    # Tickers must match ^[A-Z]{1,5}$ — generate two-letter all-caps symbols.
    rows = "\n".join(
        f"{chr(65 + i % 26)}{chr(65 + i // 26)},{i + 1},{10.0 + i},2023-01-15" for i in range(30)
    )
    csv = _write(tmp_path / "p.csv", "ticker,shares,cost_basis,purchase_date\n" + rows + "\n")
    start = time.monotonic()
    holdings = load_portfolio(csv, validate_tickers=True, validator=lambda _t: True)
    assert len(holdings) == 30
    assert time.monotonic() - start < 5.0


# ── load_watchlist ─────────────────────────────────────────────────────────────


def test_watchlist_accepts_empty_notes(tmp_path: Path) -> None:
    csv = _write(
        tmp_path / "w.csv",
        "ticker,notes\nCOST,membership moat\nTSM,\n",
    )
    entries = load_watchlist(csv)
    assert entries == [
        WatchlistEntry(ticker="COST", notes="membership moat"),
        WatchlistEntry(ticker="TSM", notes=""),
    ]


# ── persistence ────────────────────────────────────────────────────────────────


def test_sync_holdings_to_db_matches_csv(tmp_path: Path, db_session: Session) -> None:
    csv = _write(
        tmp_path / "p.csv",
        "ticker,shares,cost_basis,purchase_date\n"
        "AAPL,10,150.00,2023-01-15\n"
        "MSFT,5,280.50,2022-11-30\n",
    )
    holdings = load_portfolio(csv, validate_tickers=False)
    sync_holdings_to_db(holdings, db_session, {"AAPL": 200.0})

    rows = db_session.execute(select(HoldingRow).order_by(HoldingRow.ticker)).scalars().all()
    assert [r.ticker for r in rows] == ["AAPL", "MSFT"]
    aapl = next(r for r in rows if r.ticker == "AAPL")
    assert aapl.shares == 10.0
    assert aapl.cost_basis == 150.0
    assert aapl.current_price == 200.0
    assert aapl.purchase_date is not None
    assert aapl.purchase_date.isoformat() == "2023-01-15"
    # MSFT had no current price in the dict → None.
    assert next(r for r in rows if r.ticker == "MSFT").current_price is None


def test_sync_holdings_is_a_snapshot(tmp_path: Path, db_session: Session) -> None:
    first = load_portfolio(
        _write(
            tmp_path / "p1.csv",
            "ticker,shares,cost_basis,purchase_date\nAAPL,10,150.00,2023-01-15\n",
        ),
        validate_tickers=False,
    )
    sync_holdings_to_db(first, db_session, {})

    second = load_portfolio(
        _write(
            tmp_path / "p2.csv",
            "ticker,shares,cost_basis,purchase_date\nMSFT,5,280.50,2022-11-30\n",
        ),
        validate_tickers=False,
    )
    sync_holdings_to_db(second, db_session, {})

    rows = db_session.execute(select(HoldingRow)).scalars().all()
    assert [r.ticker for r in rows] == ["MSFT"]  # AAPL overwritten


def test_sync_watchlist_to_db(tmp_path: Path, db_session: Session) -> None:
    entries = load_watchlist(
        _write(tmp_path / "w.csv", "ticker,notes\nCOST,moat\nTSM,\n"),
    )
    sync_watchlist_to_db(entries, db_session)
    rows = db_session.execute(select(WatchlistRow).order_by(WatchlistRow.ticker)).scalars().all()
    assert {r.ticker: r.notes for r in rows} == {"COST": "moat", "TSM": ""}
