from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.orm import Session

from data_sources.finnhub_client import NewsItem
from storage.models import DiscoveryCooldown

COOLDOWN_DAYS = 7

_MATERIAL_KEYWORDS = [
    "earnings",
    "beat",
    "miss",
    "acquisition",
    "merger",
    "fda approval",
    "upgrade",
    "downgrade",
    "guidance",
    "buyback",
    "dividend",
    "ceo",
    "restructuring",
    "layoff",
    "lawsuit",
    "investigation",
]


@dataclass
class CooldownResult:
    active: list[str]
    suppressed: list[str]


def filter_universe_for_cooldown(
    candidates: list[str],
    session: Session,
    recent_news: dict[str, list[NewsItem]],
) -> CooldownResult:
    """Split candidates into active/suppressed based on 7-day cooldown state.

    A ticker under cooldown moves back to active if a material news event is detected,
    which also clears its cooldown entry so it can be re-flagged after analysis.
    """
    now = _naive_utcnow()
    active: list[str] = []
    suppressed: list[str] = []

    for ticker in candidates:
        entry = get_cooldown_entry(session, ticker)
        if entry is None or (entry.expires_at is not None and entry.expires_at < now):
            active.append(ticker)
            continue

        if has_material_event(recent_news.get(ticker, [])):
            clear_cooldown(session, ticker)
            active.append(ticker)
        else:
            suppressed.append(ticker)

    return CooldownResult(active=active, suppressed=suppressed)


def has_material_event(news_items: list[NewsItem]) -> bool:
    """Return True if any news item headline or summary contains a material keyword."""
    for item in news_items:
        text = (item.headline + " " + item.summary).lower()
        if any(kw in text for kw in _MATERIAL_KEYWORDS):
            return True
    return False


def set_cooldown(session: Session, ticker: str, reason: str) -> None:
    """Flag a ticker; upserts so re-flagging resets the 7-day expiry."""
    now = _naive_utcnow()
    expires = now + timedelta(days=COOLDOWN_DAYS)
    existing = session.get(DiscoveryCooldown, ticker)
    if existing is not None:
        existing.flagged_at = now
        existing.expires_at = expires
        existing.suppression_reason = reason
    else:
        session.add(
            DiscoveryCooldown(
                ticker=ticker,
                flagged_at=now,
                expires_at=expires,
                suppression_reason=reason,
            )
        )
    session.commit()


def get_cooldown_entry(session: Session, ticker: str) -> DiscoveryCooldown | None:
    return session.get(DiscoveryCooldown, ticker)


def clear_cooldown(session: Session, ticker: str) -> None:
    session.execute(delete(DiscoveryCooldown).where(DiscoveryCooldown.ticker == ticker))
    session.commit()


def _naive_utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
