"""Screening universe: sorted S&P 500 ∪ watchlist, weekly-refreshed.

The list of tickers handed to the Haiku screening pass is embedded in the prompt's
system prefix, so it must be **byte-for-byte identical** across nightly runs within a
week to keep the prompt cache warm (a miss costs ~3× per token). A single added or
removed ticker breaks the whole prefix, so we refresh the S&P 500 list on a 7-day
cadence — frequent enough to track real index changes, infrequent enough not to thrash
the cache on every mid-week rebalance.

The list lives in SQLite (``UniverseSnapshot``, a single ``id = 1`` row with a
``refreshed_at`` date). When the snapshot is older than a week we re-fetch from
Wikipedia via ``SP500Client``; if that live fetch fails we fall back to the committed
``data/sp500.csv`` so a run never crashes on a transient network error. The returned
universe is always ``sorted(set(...))`` — deterministic and deduped, hence identical
across machines for the same inputs.

The ``fetcher`` argument is an injection seam mirroring ``agent.portfolio``'s
``validator``: tests pass a callable so the suite stays offline.
"""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from data_sources.errors import DataSourceError
from data_sources.sp500_client import SP500Client
from storage.models import UniverseSnapshot

SP500_PATH = Path("data/sp500.csv")
REFRESH_INTERVAL_DAYS = 7

# A no-arg callable returning the S&P 500 list or an error — the SP500Client seam.
SP500Fetcher = Callable[[], "list[str] | DataSourceError"]


@dataclass
class Snapshot:
    tickers: list[str]
    refreshed_at: date

    @property
    def age_days(self) -> int:
        return (date.today() - self.refreshed_at).days


def load_snapshot(session: Session) -> Snapshot | None:
    row = session.get(UniverseSnapshot, 1)
    if row is None:
        return None
    return Snapshot(tickers=list(json.loads(row.tickers_json)), refreshed_at=row.refreshed_at)


def save_snapshot(session: Session, tickers: list[str]) -> None:
    """Upsert the single ``id = 1`` snapshot row with today's date, sorted tickers."""
    payload = json.dumps(sorted(set(tickers)))
    today = date.today()
    row = session.get(UniverseSnapshot, 1)
    if row is None:
        session.add(UniverseSnapshot(id=1, tickers_json=payload, refreshed_at=today))
    else:
        row.tickers_json = payload
        row.refreshed_at = today
    session.commit()


def _load_fallback_csv(path: Path = SP500_PATH) -> list[str]:
    with path.open(newline="") as fh:
        return [
            ticker for row in csv.DictReader(fh) if (ticker := (row.get("ticker") or "").strip())
        ]


def _default_fetcher() -> list[str] | DataSourceError:
    return SP500Client().get_sp500_constituents()


def fetch_sp500_list(fetcher: SP500Fetcher | None = None) -> list[str]:
    """Fetch the live S&P 500 list, falling back to ``data/sp500.csv`` on any error."""
    result = (fetcher or _default_fetcher)()
    if isinstance(result, DataSourceError):
        return _load_fallback_csv()
    return result


def get_current_universe(
    session: Session,
    watchlist: list[str],
    fetcher: SP500Fetcher | None = None,
) -> list[str]:
    """Return tonight's sorted, deduped S&P 500 ∪ watchlist universe.

    Re-fetches the S&P 500 list (and persists a new snapshot) only when there is no
    snapshot yet or the existing one is older than ``REFRESH_INTERVAL_DAYS``.
    """
    snapshot = load_snapshot(session)
    if snapshot is None or snapshot.age_days > REFRESH_INTERVAL_DAYS:
        sp500 = fetch_sp500_list(fetcher)
        save_snapshot(session, sp500)
    else:
        sp500 = snapshot.tickers

    return sorted(set(sp500) | set(watchlist))
