"""GDELT DOC 2.0 client — keyless cross-language adverse news search.

Uses the ArtList mode of the GDELT DOC 2.0 API to retrieve article metadata
(title, URL, tone, GKG themes) for a given query. No API key required.
Responses are cached to SQLite (7 days) since GDELT results are stable over
that window and the API has no published rate limit.

Public methods return DataSourceError on failure rather than raising.
"""

import sqlite3
import time
from collections.abc import Callable

import requests
from pydantic import BaseModel, TypeAdapter

from data_sources.cache import CacheStore, make_key
from data_sources.errors import DataSourceError

_DEFAULT_TIMEOUT = 30
_MAX_RECORDS = 250


def _as_str(v: object) -> str:
    return v if isinstance(v, str) else ""


def _as_float(v: object) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
    return 0.0


class RawArticle(BaseModel):
    url: str
    title: str
    seendate: str  # "YYYYMMDDTHHMMSSZ"
    domain: str
    language: str
    sourcecountry: str
    # tone and themes are NOT returned by the DOC 2.0 ArtList endpoint —
    # they are GKG enrichment fields. Kept as optional for forward-compatibility.
    tone: float | None = None
    themes: str = ""


_ARTICLE_ADAPTER = TypeAdapter(list[RawArticle])


class GDELTClient:
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        db_conn: sqlite3.Connection,
        *,
        cache_ttl_h: float = 168.0,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._cache = CacheStore(db_conn)
        self._ttl = cache_ttl_h
        self._sleep = _sleep
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "Warren/1.0 (research; github.com/aaronabraham311/warren)"
        )

    def get_adverse_articles(
        self,
        query: str,
        lookback_days: int,
        max_records: int = _MAX_RECORDS,
    ) -> list[RawArticle] | DataSourceError:
        key = make_key("gdelt_adverse", query, str(lookback_days))
        cached = self._cache.get(key)
        if cached is not None:
            return list(_ARTICLE_ADAPTER.validate_json(cached))

        params: dict[str, str] = {
            "query": f"{query} tone<-2",
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(max_records),
            "sort": "ToneAsc",
            "timespan": f"{lookback_days}d",
        }
        try:
            resp = self._session.get(self.BASE_URL, params=params, timeout=_DEFAULT_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
        except requests.RequestException as exc:
            return DataSourceError(error_code="network", message=str(exc))
        except (ValueError, KeyError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))

        try:
            articles = self._parse(raw)
        except (ValueError, KeyError, TypeError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))

        self._cache.set(key, _ARTICLE_ADAPTER.dump_json(articles).decode(), self._ttl)
        return articles

    @staticmethod
    def _parse(raw: object) -> list[RawArticle]:
        if not isinstance(raw, dict):
            raise ValueError("unexpected GDELT response shape (expected a dict)")
        articles_raw = raw.get("articles")
        if articles_raw is None:
            return []
        if not isinstance(articles_raw, list):
            raise ValueError("unexpected GDELT articles shape (expected a list)")
        articles: list[RawArticle] = []
        for entry in articles_raw:
            if not isinstance(entry, dict):
                continue
            raw_tone = entry.get("tone")
            articles.append(
                RawArticle(
                    url=_as_str(entry.get("url")),
                    title=_as_str(entry.get("title")),
                    seendate=_as_str(entry.get("seendate")),
                    domain=_as_str(entry.get("domain")),
                    language=_as_str(entry.get("language")),
                    sourcecountry=_as_str(entry.get("sourcecountry")),
                    tone=_as_float(raw_tone) if raw_tone is not None else None,
                    themes=_as_str(entry.get("themes")),
                )
            )
        return articles
