from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import Session

from agent.cooldown import (
    COOLDOWN_DAYS,
    CooldownResult,
    clear_cooldown,
    filter_universe_for_cooldown,
    get_cooldown_entry,
    has_material_event,
    set_cooldown,
)
from data_sources.finnhub_client import NewsItem
from storage.models import DiscoveryCooldown


def _news(headline: str = "", summary: str = "") -> NewsItem:
    return NewsItem(
        headline=headline,
        summary=summary,
        source="",
        datetime=datetime(2024, 1, 1, tzinfo=timezone.utc),
        url="",
    )


def _live_entry(session: Session, ticker: str) -> None:
    """Insert a cooldown entry that won't expire for 7 days."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    session.add(
        DiscoveryCooldown(
            ticker=ticker,
            flagged_at=now,
            expires_at=now + timedelta(days=COOLDOWN_DAYS),
            suppression_reason="test",
        )
    )
    session.commit()


def _expired_entry(session: Session, ticker: str) -> None:
    """Insert a cooldown entry that has already expired."""
    past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=COOLDOWN_DAYS + 1)
    session.add(
        DiscoveryCooldown(
            ticker=ticker,
            flagged_at=past,
            expires_at=past + timedelta(days=COOLDOWN_DAYS),
            suppression_reason="test",
        )
    )
    session.commit()


# ---------------------------------------------------------------------------
# filter_universe_for_cooldown
# ---------------------------------------------------------------------------


def test_no_entry_is_active(db_session: Session) -> None:
    result = filter_universe_for_cooldown(["AAPL"], db_session, {})
    assert result == CooldownResult(active=["AAPL"], suppressed=[])


def test_expired_entry_is_active(db_session: Session) -> None:
    _expired_entry(db_session, "AAPL")
    result = filter_universe_for_cooldown(["AAPL"], db_session, {})
    assert result == CooldownResult(active=["AAPL"], suppressed=[])


def test_live_entry_no_news_is_suppressed(db_session: Session) -> None:
    _live_entry(db_session, "AAPL")
    result = filter_universe_for_cooldown(["AAPL"], db_session, {})
    assert result == CooldownResult(active=[], suppressed=["AAPL"])


def test_live_entry_benign_news_is_suppressed(db_session: Session) -> None:
    _live_entry(db_session, "AAPL")
    result = filter_universe_for_cooldown(
        ["AAPL"], db_session, {"AAPL": [_news("Apple releases new color options")]}
    )
    assert result == CooldownResult(active=[], suppressed=["AAPL"])


def test_live_entry_material_news_overrides_cooldown(db_session: Session) -> None:
    _live_entry(db_session, "AAPL")
    result = filter_universe_for_cooldown(
        ["AAPL"], db_session, {"AAPL": [_news("Apple earnings beat estimates")]}
    )
    assert result == CooldownResult(active=["AAPL"], suppressed=[])
    # entry cleared
    assert get_cooldown_entry(db_session, "AAPL") is None


def test_mixed_candidates(db_session: Session) -> None:
    _live_entry(db_session, "MSFT")
    result = filter_universe_for_cooldown(["AAPL", "MSFT", "GOOG"], db_session, {})
    assert set(result.active) == {"AAPL", "GOOG"}
    assert result.suppressed == ["MSFT"]


# ---------------------------------------------------------------------------
# has_material_event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headline",
    [
        "Q3 earnings beat",
        "Company misses guidance",
        "Major acquisition announced",
        "Merger talks confirmed",
        "FDA approval granted",
        "Analyst upgrade issued",
        "Analyst downgrade issued",
        "Guidance raised",
        "Share buyback program",
        "Dividend increase",
        "New CEO appointed",
        "Restructuring plan unveiled",
        "Mass layoff announced",
        "Class action lawsuit filed",
        "SEC investigation opened",
    ],
)
def test_has_material_event_true(headline: str) -> None:
    assert has_material_event([_news(headline=headline)]) is True


def test_has_material_event_false_for_benign_news() -> None:
    assert has_material_event([_news("Apple opens new store in Chicago")]) is False


def test_has_material_event_false_empty() -> None:
    assert has_material_event([]) is False


def test_has_material_event_checks_summary_too() -> None:
    assert has_material_event([_news(headline="", summary="merger discussions underway")]) is True


# ---------------------------------------------------------------------------
# set_cooldown / get_cooldown_entry / clear_cooldown
# ---------------------------------------------------------------------------


def test_set_cooldown_creates_entry(db_session: Session) -> None:
    set_cooldown(db_session, "AAPL", reason="appeared_in_discovery")
    entry = get_cooldown_entry(db_session, "AAPL")
    assert entry is not None
    assert entry.ticker == "AAPL"
    assert entry.suppression_reason == "appeared_in_discovery"
    assert entry.expires_at is not None
    assert entry.flagged_at is not None
    delta = entry.expires_at - entry.flagged_at
    assert delta.days == COOLDOWN_DAYS


def test_set_cooldown_upserts(db_session: Session) -> None:
    set_cooldown(db_session, "AAPL", reason="first")
    first_flagged = get_cooldown_entry(db_session, "AAPL")
    assert first_flagged is not None

    set_cooldown(db_session, "AAPL", reason="second")
    second = get_cooldown_entry(db_session, "AAPL")
    assert second is not None
    assert second.suppression_reason == "second"
    # expiry reset to a new 7-day window
    assert second.expires_at is not None and second.flagged_at is not None
    assert (second.expires_at - second.flagged_at).days == COOLDOWN_DAYS


def test_get_cooldown_entry_returns_none_when_absent(db_session: Session) -> None:
    assert get_cooldown_entry(db_session, "AAPL") is None


def test_clear_cooldown_removes_entry(db_session: Session) -> None:
    _live_entry(db_session, "AAPL")
    assert get_cooldown_entry(db_session, "AAPL") is not None
    clear_cooldown(db_session, "AAPL")
    assert get_cooldown_entry(db_session, "AAPL") is None


def test_clear_cooldown_noop_when_absent(db_session: Session) -> None:
    clear_cooldown(db_session, "AAPL")  # should not raise


def test_clear_cooldown_ticker_becomes_active(db_session: Session) -> None:
    _live_entry(db_session, "AAPL")
    clear_cooldown(db_session, "AAPL")
    result = filter_universe_for_cooldown(["AAPL"], db_session, {})
    assert result == CooldownResult(active=["AAPL"], suppressed=[])
