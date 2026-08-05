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
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from data_sources.errors import DataSourceError
from data_sources.exchange_client import EXCHANGE_SPECS, ExchangeClient
from data_sources.security_identity import SecurityIdentity
from data_sources.sp500_client import SP500Client
from storage.models import SecurityIdentityRecord, UniverseSnapshot

SP500_PATH = Path("data/sp500.csv")
REFRESH_INTERVAL_DAYS = 7

# CSV fallbacks for the three gem-hunt exchanges, keyed by exchange key.
EXCHANGE_CSV_PATHS: dict[str, Path] = {
    "milan": Path("data/milan.csv"),
    "madrid": Path("data/madrid.csv"),
    "warsaw": Path("data/warsaw.csv"),
}
GEM_HUNT_EXCHANGES = ("milan", "madrid", "warsaw")
GEM_HUNT_IDENTITY_VENUES = frozenset(("euronext_growth_milan", "bme_growth", "newconnect"))

# A no-arg callable returning a constituent list or an error — the client seam.
ConstituentPayload = list[str] | list[SecurityIdentity] | DataSourceError
ConstituentFetcher = Callable[[], ConstituentPayload]
SP500Fetcher = Callable[[], list[str] | DataSourceError]


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


def _load_exchange_fallback(key: str) -> tuple[list[str], list[SecurityIdentity]]:
    path = EXCHANGE_CSV_PATHS[key]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    tickers = [(row.get("ticker") or "").strip() for row in rows]
    identities: list[SecurityIdentity] = []
    for row in rows:
        ticker = (row.get("ticker") or "").strip()
        isin = (row.get("isin") or "").strip()
        legal_name = (row.get("legal_name") or "").strip()
        resolved_at = (row.get("resolved_at") or "").strip()
        if not all((ticker, isin, legal_name, resolved_at)):
            continue
        identities.append(
            SecurityIdentity(
                canonical_ticker=ticker,
                venue=(row.get("venue") or "").strip(),
                mic=(row.get("mic") or "").strip() or None,
                exchange_symbol=(row.get("exchange_symbol") or "").strip(),
                isin=isin,
                legal_name=legal_name,
                identity_source_url=(row.get("identity_source_url") or "").strip(),
                resolved_at=datetime.fromisoformat(resolved_at),
            )
        )
    return [ticker for ticker in tickers if ticker], identities


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


def _partition_exchange_payload(
    key: str, payload: ConstituentPayload
) -> tuple[list[str], list[SecurityIdentity]]:
    if isinstance(payload, DataSourceError):
        return _load_exchange_fallback(key)
    if not payload:
        return [], []
    first = payload[0]
    if isinstance(first, SecurityIdentity):
        identities = [item for item in payload if isinstance(item, SecurityIdentity)]
        return [item.canonical_ticker for item in identities], identities
    return [item for item in payload if isinstance(item, str)], []


def fetch_exchange_list(key: str, fetcher: ConstituentFetcher | None = None) -> list[str]:
    """Fetch one exchange's constituents, falling back to its committed CSV on any error."""
    tickers, _ = _partition_exchange_payload(key, (fetcher or _default_exchange_fetcher(key))())
    return tickers


def save_security_identities(session: Session, identities: list[SecurityIdentity]) -> None:
    """Upsert verified identities without committing the surrounding refresh."""
    for identity in identities:
        if identity.isin is None or identity.legal_name is None:
            continue
        key = (identity.venue, identity.isin)
        row = session.get(SecurityIdentityRecord, key)
        conflicts = session.scalars(
            select(SecurityIdentityRecord).where(
                and_(
                    SecurityIdentityRecord.is_active.is_(True),
                    or_(
                        SecurityIdentityRecord.canonical_ticker == identity.canonical_ticker,
                        and_(
                            SecurityIdentityRecord.venue == identity.venue,
                            SecurityIdentityRecord.exchange_symbol == identity.exchange_symbol,
                        ),
                    ),
                )
            )
        ).all()
        stale = [conflict for conflict in conflicts if conflict is not row]
        for conflict in stale:
            conflict.is_active = False
            conflict.superseded_by_isin = identity.isin
        if row is None:
            row = SecurityIdentityRecord(
                venue=identity.venue,
                isin=identity.isin,
                canonical_ticker=identity.canonical_ticker,
                mic=identity.mic,
                exchange_symbol=identity.exchange_symbol,
                legal_name=identity.legal_name,
                identity_source_url=identity.identity_source_url,
                resolved_at=identity.resolved_at,
                is_active=True,
                superseded_by_isin=None,
            )
            session.add(row)
        else:
            row.canonical_ticker = identity.canonical_ticker
            row.mic = identity.mic
            row.exchange_symbol = identity.exchange_symbol
            row.legal_name = identity.legal_name
            row.identity_source_url = identity.identity_source_url
            row.resolved_at = identity.resolved_at
            row.is_active = True
            row.superseded_by_isin = None


def _has_security_identity_coverage(session: Session) -> bool:
    venues = set(
        session.scalars(
            select(SecurityIdentityRecord.venue)
            .where(SecurityIdentityRecord.is_active.is_(True))
            .distinct()
        )
    )
    return GEM_HUNT_IDENTITY_VENUES.issubset(venues)


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
    needs_identity_bootstrap = fetchers is None and not _has_security_identity_coverage(session)
    if snapshot is None or snapshot.age_days > REFRESH_INTERVAL_DAYS or needs_identity_bootstrap:
        overrides = fetchers or {}
        constituents: list[str] = []
        identities: list[SecurityIdentity] = []
        for key in GEM_HUNT_EXCHANGES:
            payload = (overrides.get(key) or _default_exchange_fetcher(key))()
            venue_tickers, venue_identities = _partition_exchange_payload(key, payload)
            constituents += venue_tickers
            identities += venue_identities
        save_security_identities(session, identities)
        save_snapshot(session, constituents, kind="gem_hunt")
    else:
        constituents = snapshot.tickers

    return sorted(set(constituents) | set(watchlist))
