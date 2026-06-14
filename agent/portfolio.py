import csv
from datetime import date, datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import delete
from sqlalchemy.orm import Session

from agent.tools._clients import yfinance_client
from data_sources.errors import DataSourceError
from storage.models import Holding as HoldingRow

_PORTFOLIO_FILE = Path("data/portfolio.csv")
_WATCHLIST_FILE = Path("data/watchlist.csv")


class Holding(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$")
    shares: float = Field(gt=0)
    cost_basis: float = Field(gt=0)
    purchase_date: date


class WatchlistEntry(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z]{1,5}$")
    notes: str = ""


def load_portfolio(path: Path = _PORTFOLIO_FILE, validate_tickers: bool = True) -> list[Holding]:
    if not path.exists():
        raise FileNotFoundError(f"Portfolio CSV not found at expected path: {path}")

    holdings: list[Holding] = []
    row_errors: list[str] = []
    with path.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            try:
                holdings.append(Holding(**{k: (v or "").strip() for k, v in row.items() if k}))
            except ValidationError as exc:
                row_errors.append(f"row {i}: {exc}")
    if row_errors:
        raise ValueError("Portfolio CSV validation failed:\n" + "\n".join(row_errors))

    tickers = [h.ticker for h in holdings]
    dupes = sorted({t for t in tickers if tickers.count(t) > 1})
    if dupes:
        raise ValueError(f"Duplicate tickers in portfolio CSV: {', '.join(dupes)}")

    if validate_tickers:
        client = yfinance_client()
        invalid: list[str] = []
        for h in holdings:
            result = client.get_price(h.ticker)
            if isinstance(result, DataSourceError) and result.error_code == "not_found":
                invalid.append(h.ticker)
        if invalid:
            bad = ", ".join(sorted(invalid))
            raise ValueError(f"Tickers not found via yfinance (invalid or delisted): {bad}")

    return holdings


def load_watchlist(path: Path = _WATCHLIST_FILE) -> list[WatchlistEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Watchlist CSV not found at expected path: {path}")

    entries: list[WatchlistEntry] = []
    row_errors: list[str] = []
    with path.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            try:
                entries.append(
                    WatchlistEntry(
                        ticker=(row.get("ticker") or "").strip(),
                        notes=(row.get("notes") or "").strip(),
                    )
                )
            except ValidationError as exc:
                row_errors.append(f"row {i}: {exc}")
    if row_errors:
        raise ValueError("Watchlist CSV validation failed:\n" + "\n".join(row_errors))

    return entries


def sync_holdings_to_db(
    holdings: list[Holding],
    session: Session,
    current_prices: dict[str, float],
) -> None:
    session.execute(delete(HoldingRow))
    now = datetime.now(timezone.utc)
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
