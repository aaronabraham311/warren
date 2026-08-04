"""Screening universe: sorted constituent set ∪ watchlist, weekly-refreshed.

Two universes share this machinery, discriminated by a ``kind`` key on the
``UniverseSnapshot`` cache:

  - ``"sp500"`` — the default nightly US universe (S&P 500 ∪ watchlist), fetched from
    Wikipedia via ``SP500Client`` (``get_current_universe``).
  - ``"gem_hunt"`` — the global 3-exchange universe for gem-hunt mode (Milan ∪ Madrid
    ∪ Warsaw ∪ watchlist), fetched via ``ExchangeClient`` (``get_gem_hunt_universe``).

The list lives in SQLite (``UniverseSnapshot``, one row per ``kind`` with a
``refreshed_at`` date). When the snapshot is older than ``REFRESH_INTERVAL_DAYS`` we
re-fetch from the live source; if that fetch fails we fall back to the committed
``data/*.csv`` so a run never crashes on a transient network error. The 7-day cadence
exists purely to avoid re-scraping the constituent source on every nightly run — it is
**not** a prompt-cache concern: the universe never enters an LLM prompt prefix, and
screening is deterministic per-ticker Python in ``agent.screening``. The returned
universe is always ``sorted(set(...))`` — deterministic and deduped.

The ``fetcher`` / ``fetchers`` arguments are injection seams mirroring
``agent.portfolio``'s ``validator``: tests pass callables so the suite stays offline.
"""

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from data_sources.errors import DataSourceError
from data_sources.exchange_client import EXCHANGE_SPECS, ExchangeClient
from data_sources.sp500_client import SP500Client
from storage.models import UniverseSnapshot

SP500_PATH = Path("data/sp500.csv")
REFRESH_INTERVAL_DAYS = 7

# CSV fallbacks for the three gem-hunt exchanges, keyed by exchange key.
EXCHANGE_CSV_PATHS: dict[str, Path] = {
    "milan": Path("data/milan.csv"),
    "madrid": Path("data/madrid.csv"),
    "warsaw": Path("data/warsaw.csv"),
}
GEM_HUNT_EXCHANGES = ("milan", "madrid", "warsaw")

# A no-arg callable returning a constituent list or an error — the client seam.
ConstituentFetcher = Callable[[], "list[str] | DataSourceError"]
SP500Fetcher = ConstituentFetcher


@dataclass
class Snapshot:
    tickers: list[str]
    refreshed_at: date

    @property
    def age_days(self) -> int:
        return (date.today() - self.refreshed_at).days


def load_snapshot(session: Session, kind: str = "sp500") -> Snapshot | None:
    row = session.get(UniverseSnapshot, kind)
    if row is None:
        return None
    return Snapshot(tickers=list(json.loads(row.tickers_json)), refreshed_at=row.refreshed_at)


def save_snapshot(session: Session, tickers: list[str], kind: str = "sp500") -> None:
    """Upsert the ``kind`` snapshot row with today's date and sorted, deduped tickers."""
    payload = json.dumps(sorted(set(tickers)))
    today = date.today()
    row = session.get(UniverseSnapshot, kind)
    if row is None:
        session.add(UniverseSnapshot(kind=kind, tickers_json=payload, refreshed_at=today))
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
    snapshot = load_snapshot(session, kind="sp500")
    if snapshot is None or snapshot.age_days > REFRESH_INTERVAL_DAYS:
        sp500 = fetch_sp500_list(fetcher)
        save_snapshot(session, sp500, kind="sp500")
    else:
        sp500 = snapshot.tickers

    return sorted(set(sp500) | set(watchlist))


def _default_exchange_fetcher(key: str) -> ConstituentFetcher:
    spec = EXCHANGE_SPECS[key]
    return lambda: ExchangeClient(spec).get_constituents()


def fetch_exchange_list(key: str, fetcher: ConstituentFetcher | None = None) -> list[str]:
    """Fetch one exchange's constituents, falling back to its committed CSV on any error."""
    result = (fetcher or _default_exchange_fetcher(key))()
    if isinstance(result, DataSourceError):
        return _load_fallback_csv(EXCHANGE_CSV_PATHS[key])
    return result


def get_gem_hunt_universe(
    session: Session,
    watchlist: list[str],
    fetchers: dict[str, ConstituentFetcher] | None = None,
) -> list[str]:
    """Return the sorted, deduped global gem-hunt universe ∪ watchlist.

    The universe is Euronext Growth Milan ∪ Bolsa de Madrid ∪ GPW Warsaw. Each
    exchange is fetched live with a per-exchange CSV fallback, then cached under the
    ``"gem_hunt"`` snapshot kind. Re-fetches (and persists) only when there is no
    snapshot yet or the existing one is older than ``REFRESH_INTERVAL_DAYS``.

    Args:
        fetchers: Optional per-exchange fetcher overrides keyed by exchange key
            (``"milan"``/``"madrid"``/``"warsaw"``) — the offline-test injection seam.
    """
    snapshot = load_snapshot(session, kind="gem_hunt")
    if snapshot is None or snapshot.age_days > REFRESH_INTERVAL_DAYS:
        overrides = fetchers or {}
        constituents: list[str] = []
        for key in GEM_HUNT_EXCHANGES:
            constituents += fetch_exchange_list(key, overrides.get(key))
        save_snapshot(session, constituents, kind="gem_hunt")
    else:
        constituents = snapshot.tickers

    return sorted(set(constituents) | set(watchlist))
