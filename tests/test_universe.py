from datetime import date, datetime, timedelta, timezone

import pytest
import requests
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from agent import universe
from data_sources.errors import DataSourceError
from data_sources.security_identity import SecurityIdentity
from data_sources.sp500_client import SP500Client
from storage.models import SecurityIdentityRecord, UniverseSnapshot

# ── SP500Client.parse / fetch ─────────────────────────────────────────────────

_HTML = """
<html><body>
<table id="constituents">
  <tr><th>Symbol</th><th>Security</th></tr>
  <tr><td>MSFT</td><td>Microsoft</td></tr>
  <tr><td>AAPL</td><td>Apple</td></tr>
  <tr><td>BRK.B</td><td>Berkshire Hathaway</td></tr>
</table>
</body></html>
"""


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    """Stands in for requests.Session — returns canned HTML or raises."""

    def __init__(self, *, text: str | None = None, exc: Exception | None = None) -> None:
        self._text = text
        self._exc = exc
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: int) -> _FakeResp:
        if self._exc is not None:
            raise self._exc
        assert self._text is not None
        return _FakeResp(self._text)


def _client_with_session(monkeypatch: pytest.MonkeyPatch, session: _FakeSession) -> SP500Client:
    client = SP500Client()
    monkeypatch.setattr(client, "_session", session)
    return client


def test_parse_returns_tickers_with_dots_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_session(monkeypatch, _FakeSession(text=_HTML))
    result = client.get_sp500_constituents()
    assert result == ["MSFT", "AAPL", "BRK-B"]  # order preserved; BRK.B → BRK-B


def test_missing_table_returns_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_session(
        monkeypatch, _FakeSession(text="<html><body>no table</body></html>")
    )
    result = client.get_sp500_constituents()
    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"


def test_network_failure_returns_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_session(monkeypatch, _FakeSession(exc=requests.ConnectionError("boom")))
    result = client.get_sp500_constituents()
    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


# ── universe orchestration ────────────────────────────────────────────────────


def _counting_fetcher(tickers: list[str]) -> tuple[universe.SP500Fetcher, list[int]]:
    """A fetcher returning ``tickers`` and a one-element list recording call count."""
    calls = [0]

    def fetch() -> list[str]:
        calls[0] += 1
        return tickers

    return fetch, calls


def test_universe_is_sorted_and_deduped(db_session: Session) -> None:
    fetch, _ = _counting_fetcher(["MSFT", "AAPL", "AAPL", "NVDA"])
    result = universe.get_current_universe(db_session, watchlist=["NVDA", "TSLA"], fetcher=fetch)
    assert result == ["AAPL", "MSFT", "NVDA", "TSLA"]  # sorted, NVDA deduped across both sets


def test_no_redundant_fetch_within_the_week(db_session: Session) -> None:
    fetch, calls = _counting_fetcher(["MSFT", "AAPL"])

    first = universe.get_current_universe(db_session, watchlist=[], fetcher=fetch)
    snap_first = universe.load_snapshot(db_session)

    second = universe.get_current_universe(db_session, watchlist=[], fetcher=fetch)
    snap_second = universe.load_snapshot(db_session)

    assert snap_first is not None and snap_second is not None
    assert first == second
    assert calls[0] == 1  # second call served from the snapshot, no re-fetch
    assert snap_first.refreshed_at == snap_second.refreshed_at == date.today()


def test_stale_snapshot_triggers_refresh(db_session: Session) -> None:
    # Seed an 8-day-old snapshot directly (older than REFRESH_INTERVAL_DAYS=7).
    stale = date.today() - timedelta(days=8)
    db_session.add(UniverseSnapshot(kind="sp500", tickers_json='["OLD"]', refreshed_at=stale))
    db_session.commit()

    fetch, calls = _counting_fetcher(["MSFT", "AAPL"])
    result = universe.get_current_universe(db_session, watchlist=[], fetcher=fetch)

    refreshed = universe.load_snapshot(db_session)
    assert refreshed is not None
    assert calls[0] == 1  # stale → re-fetched
    assert result == ["AAPL", "MSFT"]
    assert refreshed.refreshed_at == date.today()


def test_fetch_failure_falls_back_to_csv(db_session: Session) -> None:
    def failing() -> DataSourceError:
        return DataSourceError(error_code="network", message="down")

    result = universe.get_current_universe(db_session, watchlist=["ZZZZ"], fetcher=failing)

    assert "ZZZZ" in result  # watchlist still merged in
    assert "AAPL" in result  # real constituents loaded from data/sp500.csv fallback
    assert len(result) > 100  # the committed fallback list is the full S&P 500


def test_same_inputs_are_byte_identical(db_session: Session) -> None:
    # Same constituent set in a different order must yield byte-identical output,
    # so two machines fetching the (unordered) list still build the same prefix.
    fetch1, _ = _counting_fetcher(["NVDA", "MSFT", "AAPL"])
    fetch2, _ = _counting_fetcher(["AAPL", "NVDA", "MSFT"])

    u1 = universe.get_current_universe(db_session, watchlist=["TSLA"], fetcher=fetch1)

    # Clear the snapshot so the second build re-fetches rather than reusing it.
    db_session.execute(delete(UniverseSnapshot))
    db_session.commit()
    u2 = universe.get_current_universe(db_session, watchlist=["TSLA"], fetcher=fetch2)

    assert u1 == u2


def test_save_snapshot_stores_sorted_json(db_session: Session) -> None:
    universe.save_snapshot(db_session, ["MSFT", "AAPL", "AAPL"])
    row = db_session.get(UniverseSnapshot, "sp500")
    assert row is not None
    assert row.tickers_json == '["AAPL", "MSFT"]'  # sorted + deduped


# ── gem-hunt universe orchestration ───────────────────────────────────────────


def _exchange_fetchers(
    payloads: dict[str, list[str]],
) -> tuple[dict[str, universe.ConstituentFetcher], dict[str, int]]:
    """Per-exchange counting fetchers returning the given payloads."""
    calls: dict[str, int] = {k: 0 for k in payloads}

    def make(key: str, tickers: list[str]) -> universe.ConstituentFetcher:
        def fetch() -> list[str]:
            calls[key] += 1
            return tickers

        return fetch

    return {k: make(k, v) for k, v in payloads.items()}, calls


def test_gem_hunt_universe_unions_exchanges_and_watchlist(db_session: Session) -> None:
    fetchers, _ = _exchange_fetchers(
        {"milan": ["DIR.MI", "RACE.MI"], "madrid": ["CIRSA.MC"], "warsaw": ["KPL.WA"]}
    )
    result = universe.get_gem_hunt_universe(
        db_session, watchlist=["ZZZZ", "RACE.MI"], fetchers=fetchers
    )
    assert result == sorted({"DIR.MI", "RACE.MI", "CIRSA.MC", "KPL.WA", "ZZZZ"})
    # The three known gems the G10 canary depends on are present.
    assert {"DIR.MI", "CIRSA.MC", "KPL.WA"}.issubset(result)


def test_gem_hunt_fetch_failure_falls_back_to_csv(db_session: Session) -> None:
    def failing() -> DataSourceError:
        return DataSourceError(error_code="network", message="down")

    fetchers: dict[str, universe.ConstituentFetcher] = {
        "milan": failing,
        "madrid": failing,
        "warsaw": failing,
    }
    result = universe.get_gem_hunt_universe(db_session, watchlist=[], fetchers=fetchers)
    # All three regenerated junior-market fallbacks are loaded. Historical synthetic
    # canary symbols are not injected into the authoritative venue lists.
    assert len(result) >= 500
    assert any(ticker.endswith(".MI") for ticker in result)
    assert any(ticker.endswith(".MC") for ticker in result)
    assert any(ticker.endswith(".WA") for ticker in result)


def test_gem_hunt_snapshot_is_reused_within_the_week(db_session: Session) -> None:
    fetchers, calls = _exchange_fetchers(
        {"milan": ["DIR.MI"], "madrid": ["CIRSA.MC"], "warsaw": ["KPL.WA"]}
    )

    first = universe.get_gem_hunt_universe(db_session, watchlist=[], fetchers=fetchers)
    second = universe.get_gem_hunt_universe(db_session, watchlist=[], fetchers=fetchers)

    assert first == second
    assert calls == {"milan": 1, "madrid": 1, "warsaw": 1}  # second served from snapshot
    # The gem_hunt snapshot is a distinct row; the sp500 snapshot is untouched.
    assert universe.load_snapshot(db_session, kind="gem_hunt") is not None
    assert universe.load_snapshot(db_session, kind="sp500") is None


def test_sp500_and_gem_hunt_snapshots_coexist(db_session: Session) -> None:
    """Regression: the two universe kinds are independent rows, not one shared slot."""
    sp_fetch, _ = _counting_fetcher(["MSFT", "AAPL"])
    universe.get_current_universe(db_session, watchlist=[], fetcher=sp_fetch)

    gem_fetchers, _ = _exchange_fetchers(
        {"milan": ["DIR.MI"], "madrid": ["CIRSA.MC"], "warsaw": ["KPL.WA"]}
    )
    universe.get_gem_hunt_universe(db_session, watchlist=[], fetchers=gem_fetchers)

    sp = universe.load_snapshot(db_session, kind="sp500")
    gem = universe.load_snapshot(db_session, kind="gem_hunt")
    assert sp is not None and gem is not None
    assert set(sp.tickers) == {"MSFT", "AAPL"}
    assert set(gem.tickers) == {"DIR.MI", "CIRSA.MC", "KPL.WA"}


def test_verified_security_identities_are_upserted(db_session: Session) -> None:
    first = SecurityIdentity(
        canonical_ticker="OLD.MI",
        venue="euronext_growth_milan",
        mic="EXGM",
        exchange_symbol="OLD",
        isin="IT0000000001",
        legal_name="Old Name",
        identity_source_url="https://example.test/old",
        resolved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    renamed = SecurityIdentity(
        canonical_ticker="NEW.MI",
        venue="euronext_growth_milan",
        mic="EXGM",
        exchange_symbol="NEW",
        isin="IT0000000001",
        legal_name="New Name",
        identity_source_url="https://example.test/new",
        resolved_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
    )

    universe.save_security_identities(db_session, [first])
    universe.save_security_identities(db_session, [renamed])
    db_session.commit()

    rows = db_session.scalars(select(SecurityIdentityRecord)).all()
    assert len(rows) == 1
    assert rows[0].canonical_ticker == "NEW.MI"
    assert rows[0].legal_name == "New Name"


def test_identity_reassignment_preserves_superseded_isin(db_session: Session) -> None:
    original = SecurityIdentity(
        canonical_ticker="SAME.MI",
        venue="euronext_growth_milan",
        mic="EXGM",
        exchange_symbol="SAME",
        isin="IT0000000001",
        legal_name="Original Issuer",
        identity_source_url="https://example.test/old",
        resolved_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    reassigned = SecurityIdentity(
        canonical_ticker="SAME.MI",
        venue="euronext_growth_milan",
        mic="EXGM",
        exchange_symbol="SAME",
        isin="IT0000000002",
        legal_name="Replacement Issuer",
        identity_source_url="https://example.test/new",
        resolved_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
    )

    universe.save_security_identities(db_session, [original])
    universe.save_security_identities(db_session, [reassigned])
    db_session.commit()

    rows = db_session.scalars(
        select(SecurityIdentityRecord).order_by(SecurityIdentityRecord.isin)
    ).all()
    assert [(row.isin, row.is_active, row.superseded_by_isin) for row in rows] == [
        ("IT0000000001", False, "IT0000000002"),
        ("IT0000000002", True, None),
    ]


def test_identity_coverage_requires_all_junior_venues(db_session: Session) -> None:
    identities = [
        SecurityIdentity(
            canonical_ticker=f"TEST{index}{suffix}",
            venue=venue,
            mic=mic,
            exchange_symbol=f"TEST{index}",
            isin=f"XX000000000{index}",
            legal_name=f"Issuer {index}",
            identity_source_url="https://example.test/source",
            resolved_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        )
        for index, (venue, suffix, mic) in enumerate(
            [
                ("euronext_growth_milan", ".MI", "EXGM"),
                ("bme_growth", ".MC", None),
                ("newconnect", ".WA", None),
            ],
            start=1,
        )
    ]

    universe.save_security_identities(db_session, identities[:2])
    assert not universe._has_security_identity_coverage(db_session)
    universe.save_security_identities(db_session, identities[2:])
    assert universe._has_security_identity_coverage(db_session)


def test_missing_isin_is_not_persisted(db_session: Session) -> None:
    incomplete = SecurityIdentity(
        canonical_ticker="KPL.WA",
        venue="newconnect",
        mic=None,
        exchange_symbol="KPL",
        isin=None,
        legal_name=None,
        identity_source_url="https://scanner.tradingview.com/poland/scan",
        resolved_at=datetime.now(timezone.utc),
    )

    universe.save_security_identities(db_session, [incomplete])
    db_session.commit()

    assert db_session.scalar(select(SecurityIdentityRecord.venue)) is None
