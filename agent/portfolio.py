"""Portfolio & watchlist CSV ingestion: parse → validate → persist.

This is the agent's input data. It must fail *loudly* on malformed input (bad
ticker patterns, non-positive shares/cost basis, duplicate tickers, unknown
tickers) at startup rather than silently producing wrong analyses mid-run.

All validation errors for a file are aggregated and raised once as a
``PortfolioError`` so the user sees every problem at once, not one per re-run.
Ticker existence is checked through the cached ``YFinanceClient`` singleton (all
network lives in ``data_sources/``); an injectable ``validator`` keeps tests
offline.
"""

import csv
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import delete
from sqlalchemy.orm import Session

from agent.tools._clients import yfinance_client
from data_sources.symbols import TICKER_PATTERN
from data_sources.yfinance_client import PriceData
from storage.models import Holding as HoldingRow
from storage.models import Watchlist as WatchlistRow


class PortfolioError(ValueError):
    """Raised when a CSV fails validation. Aggregates every problem found."""


# ── Pydantic models ───────────────────────────────────────────────────────────


class Holding(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN)
    shares: float = Field(gt=0)
    cost_basis: float = Field(gt=0)
    purchase_date: date


class WatchlistEntry(BaseModel):
    ticker: str = Field(pattern=TICKER_PATTERN)
    notes: str = ""


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV file not found: {path.resolve()}")
    with path.open(newline="") as fh:
        return [row for row in csv.DictReader(fh)]


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'row'}: {e['msg']}" for e in exc.errors()
    )


def _find_duplicates(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for t in tickers:
        if t in seen and t not in dupes:
            dupes.append(t)
        seen.add(t)
    return dupes


# ── Loaders ───────────────────────────────────────────────────────────────────


def _default_ticker_validator(ticker: str) -> bool:
    """True if yfinance has price data for the ticker (cached, throttled)."""
    return isinstance(yfinance_client().get_price(ticker), PriceData)


def load_portfolio(
    path: Path,
    validate_tickers: bool = True,
    validator: Callable[[str], bool] | None = None,
) -> list[Holding]:
    """Parse and validate ``data/portfolio.csv`` into ``Holding`` models.

    Raises ``FileNotFoundError`` if the file is missing, and ``PortfolioError``
    listing *every* problem at once (row-level validation failures, duplicate
    tickers, and — when ``validate_tickers`` — tickers yfinance cannot resolve).
    """
    rows = _read_rows(path)

    holdings: list[Holding] = []
    errors: list[str] = []
    for i, row in enumerate(rows, start=2):  # row 1 is the header
        try:
            holdings.append(Holding(**row))
        except ValidationError as exc:
            label = (row.get("ticker") or "").strip() or f"row {i}"
            errors.append(f"{label} — {_format_validation_error(exc)}")

    dupes = _find_duplicates([h.ticker for h in holdings])
    if dupes:
        errors.append(f"duplicate tickers: {', '.join(dupes)}")

    if errors:
        raise PortfolioError("Invalid portfolio.csv:\n  " + "\n  ".join(errors))

    if validate_tickers:
        check = validator or _default_ticker_validator
        invalid = [h.ticker for h in holdings if not check(h.ticker)]
        if invalid:
            raise PortfolioError(f"Unknown tickers (not found via yfinance): {', '.join(invalid)}")

    return holdings


def load_watchlist(path: Path) -> list[WatchlistEntry]:
    """Parse and validate ``data/watchlist.csv``. An empty notes column is allowed."""
    rows = _read_rows(path)

    entries: list[WatchlistEntry] = []
    errors: list[str] = []
    for i, row in enumerate(rows, start=2):
        try:
            entries.append(WatchlistEntry(**{k: v for k, v in row.items() if v is not None}))
        except ValidationError as exc:
            label = (row.get("ticker") or "").strip() or f"row {i}"
            errors.append(f"{label} — {_format_validation_error(exc)}")

    dupes = _find_duplicates([e.ticker for e in entries])
    if dupes:
        errors.append(f"duplicate tickers: {', '.join(dupes)}")

    if errors:
        raise PortfolioError("Invalid watchlist.csv:\n  " + "\n  ".join(errors))

    return entries


# ── Persistence (snapshot: overwrite each run) ────────────────────────────────


def sync_holdings_to_db(
    holdings: list[Holding],
    session: Session,
    current_prices: dict[str, float],
) -> None:
    """Overwrite the holdings table with the current CSV snapshot."""
    now = datetime.now(timezone.utc)
    session.execute(delete(HoldingRow))
    for h in holdings:
        session.add(
            HoldingRow(
                ticker=h.ticker,
                shares=h.shares,
                cost_basis=h.cost_basis,
                purchase_date=h.purchase_date,
                current_price=current_prices.get(h.ticker),
                updated_at=now,
            )
        )
    session.commit()


def sync_watchlist_to_db(entries: list[WatchlistEntry], session: Session) -> None:
    """Overwrite the watchlist table with the current CSV snapshot."""
    now = datetime.now(timezone.utc)
    session.execute(delete(WatchlistRow))
    for e in entries:
        session.add(WatchlistRow(ticker=e.ticker, notes=e.notes, added_at=now))
    session.commit()
