"""OFAC Sanctions List Search client (US Treasury).

Checks an entity against the OFAC Specially Designated Nationals (SDN) and
other consolidated sanctions lists via the free, keyless Treasury API:

    GET https://sanctionsapi.treasury.gov/api/ofac/search?name=<name>&type=<type>

Coverage is US sanctions only (OFAC programs such as SDGT, UKRAINE-EO13685,
RUSSIA-EO14024, IRAN, etc.). This is narrower than OpenSanctions but costs
nothing and requires no account. For broader PEP / EU / UN list coverage,
self-host OpenSanctions (see ticket backlog).

Public methods return ``DataSourceError`` on failure rather than raising.
Responses are cached for 7 days (168 h).
"""

import sqlite3
from typing import Literal

import requests
from pydantic import BaseModel, TypeAdapter

from data_sources.cache import CacheStore, make_key
from data_sources.errors import DataSourceError

# ── Pydantic output schemas ────────────────────────────────────────────────────

RiskCategory = Literal["sanction", "pep", "criminal", "debarment", "other"]


class WatchlistMatch(BaseModel):
    matched_name: str
    match_score: float
    risk_categories: list[RiskCategory]
    datasets: list[str]
    linked_entities: list[str]


class WatchlistResult(BaseModel):
    entity_name: str
    entity_type: str
    matches: list[WatchlistMatch]
    asymmetry_note: str


_RESULT_ADAPTER = TypeAdapter(WatchlistResult)

# ── Helpers ────────────────────────────────────────────────────────────────────

_ASYMMETRY_NOTE = (
    "No matches found is not proof of clean status — "
    "OFAC coverage is US sanctions only and is not exhaustive."
)

_ENTITY_TYPE_TO_PARAM: dict[str, str] = {
    "person": "individual",
    "company": "entity",
    "vessel": "vessel",
    "aircraft": "aircraft",
}


def _as_str(v: object) -> str:
    return v if isinstance(v, str) else ""


def _as_float(v: object) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return 0.0


def _as_str_list(v: object) -> list[str]:
    if not isinstance(v, list):
        return []
    return [_as_str(item) for item in v if isinstance(item, str)]


# ── OFACClient ─────────────────────────────────────────────────────────────────


class OFACClient:
    SEARCH_URL = "https://sanctionsapi.treasury.gov/api/ofac/search"

    def __init__(
        self,
        db_conn: sqlite3.Connection,
        *,
        cache_ttl_h: float = 168.0,
        _session: requests.Session | None = None,
    ) -> None:
        self._cache = CacheStore(db_conn)
        self._ttl_h = cache_ttl_h
        self._session = _session if _session is not None else requests.Session()

    # ── search_entity ──────────────────────────────────────────────────────────

    def search_entity(
        self,
        entity_name: str,
        entity_type: str,
        country_hint: str | None,
    ) -> WatchlistResult | DataSourceError:
        key = make_key("ofac_search", entity_name.lower(), entity_type)
        cached = self._cache.get(key)
        if cached is not None:
            return _RESULT_ADAPTER.validate_json(cached)

        params: dict[str, str] = {"name": entity_name}
        ofac_type = _ENTITY_TYPE_TO_PARAM.get(entity_type)
        if ofac_type:
            params["type"] = ofac_type

        try:
            resp = self._session.get(
                self.SEARCH_URL,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()
            result = self._parse_response(entity_name, entity_type, raw)
        except requests.RequestException as exc:
            return DataSourceError(error_code="network", message=str(exc))
        except (KeyError, ValueError, TypeError) as exc:
            return DataSourceError(error_code="parse", message=str(exc))

        self._cache.set(key, result.model_dump_json(), self._ttl_h)
        return result

    @staticmethod
    def _parse_response(entity_name: str, entity_type: str, raw: object) -> WatchlistResult:
        if not isinstance(raw, dict):
            raise ValueError(f"unexpected response shape: {type(raw)}")
        if raw.get("error") is not None:
            raise ValueError(f"OFAC API error: {raw['error']}")
        raw_results = raw.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("'results' is not a list")

        matches: list[WatchlistMatch] = []
        for entry in raw_results:
            if not isinstance(entry, dict):
                continue
            programs = _as_str_list(entry.get("programs", []))
            # OFAC AKAs carry alternate names — use them as linked entity refs.
            akas_raw = entry.get("akas", [])
            linked: list[str] = []
            if isinstance(akas_raw, list):
                for aka in akas_raw:
                    if not isinstance(aka, dict):
                        continue
                    parts = [
                        _as_str(aka.get("firstName")),
                        _as_str(aka.get("lastName")),
                    ]
                    aka_name = " ".join(p for p in parts if p).strip()
                    if aka_name:
                        linked.append(aka_name)
            # OFAC scores come as integers 0-100; normalise to 0.0-1.0.
            raw_score = entry.get("score")
            score = _as_float(raw_score) / 100.0 if isinstance(raw_score, (int, float)) else 0.0
            matches.append(
                WatchlistMatch(
                    matched_name=_as_str(entry.get("name")) or entity_name,
                    match_score=score,
                    risk_categories=["sanction"],
                    datasets=programs if programs else ["OFAC-SDN"],
                    linked_entities=linked,
                )
            )

        return WatchlistResult(
            entity_name=entity_name,
            entity_type=entity_type,
            matches=matches,
            asymmetry_note=_ASYMMETRY_NOTE,
        )
