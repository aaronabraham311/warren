"""OpenSanctions hosted match-API client.

Checks an entity (person, company, vessel, aircraft) against the OpenSanctions
dataset — which aggregates hundreds of sanctions lists (OFAC, EU FSF, UN, …),
PEP databases, and criminal-interest indexes — and returns structured matches
with scores, risk categories, datasets, and linked entity references.

Non-commercial license accepted (Decision #32); logged once at construction.
V1 uses the hosted API; self-hosting is a later ticket.

Public methods return ``DataSourceError`` on failure rather than raising.
Responses are cached for 7 days (168 h).
"""

import logging
import sqlite3
from typing import Any, Literal

import requests
from pydantic import BaseModel, TypeAdapter

from data_sources.cache import CacheStore, make_key
from data_sources.errors import DataSourceError

_log = logging.getLogger(__name__)

_LICENSE_NOTE = (
    "OpenSanctions data is used under a non-commercial license "
    "(Decision #32). See https://www.opensanctions.org/licensing/ for terms."
)

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

# ── Field helpers (keep mypy strict; no Any in returns) ───────────────────────


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


# ── Topic → risk category mapping ─────────────────────────────────────────────


def _map_topic(topic: str) -> RiskCategory:
    if topic == "sanction" or topic.startswith("sanction."):
        return "sanction"
    if topic == "pep" or topic.startswith("pep.") or topic.startswith("role.pep"):
        return "pep"
    if topic.startswith("crime"):
        return "criminal"
    if topic == "debarment" or topic.startswith("debarment."):
        return "debarment"
    return "other"


def _map_topics(topics: list[str]) -> list[RiskCategory]:
    seen: set[RiskCategory] = set()
    out: list[RiskCategory] = []
    for t in topics:
        cat = _map_topic(t)
        if cat not in seen:
            seen.add(cat)
            out.append(cat)
    return out


# ── OpenSanctionsClient ────────────────────────────────────────────────────────

_ASYMMETRY_NOTE = (
    "No matches found is not proof of clean status — "
    "OpenSanctions coverage is broad but not exhaustive."
)

_ENTITY_TYPE_TO_SCHEMA: dict[str, str] = {
    "person": "Person",
    "company": "Organization",
    "vessel": "Vessel",
    "aircraft": "Aircraft",
}


class OpenSanctionsClient:
    MATCH_URL = "https://api.opensanctions.org/match/default"

    def __init__(
        self,
        db_conn: sqlite3.Connection,
        *,
        api_key: str = "",
        cache_ttl_h: float = 168.0,
        _session: requests.Session | None = None,
    ) -> None:
        _log.info(_LICENSE_NOTE)
        self._api_key = api_key
        self._cache = CacheStore(db_conn)
        self._ttl_h = cache_ttl_h
        self._session = _session if _session is not None else requests.Session()

    # ── match_entity ───────────────────────────────────────────────────────────

    def match_entity(
        self,
        entity_name: str,
        entity_type: str,
        country_hint: str | None,
    ) -> WatchlistResult | DataSourceError:
        key = make_key("opensanctions_match", entity_name.lower(), entity_type, country_hint or "")
        cached = self._cache.get(key)
        if cached is not None:
            return _RESULT_ADAPTER.validate_json(cached)

        schema = _ENTITY_TYPE_TO_SCHEMA.get(entity_type, "Thing")
        properties: dict[str, list[str]] = {"name": [entity_name]}
        if country_hint:
            properties["country"] = [country_hint]

        payload: dict[str, Any] = {"queries": {"q0": {"schema": schema, "properties": properties}}}
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"ApiKey {self._api_key}"

        try:
            resp = self._session.post(
                self.MATCH_URL,
                json=payload,
                headers=headers,
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
        responses = raw.get("responses")
        if not isinstance(responses, dict):
            raise ValueError("'responses' key missing or not a dict")
        # We always send exactly one query keyed "q0".
        q0 = responses.get("q0", {})
        if not isinstance(q0, dict):
            raise ValueError("'responses.q0' is not a dict")
        raw_results = q0.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("'results' is not a list")

        matches: list[WatchlistMatch] = []
        for entry in raw_results:
            if not isinstance(entry, dict):
                continue
            props = entry.get("properties", {})
            topics = _as_str_list(props.get("topics", []) if isinstance(props, dict) else [])
            names = _as_str_list(props.get("name", []) if isinstance(props, dict) else [])
            caption = _as_str(entry.get("caption"))
            matched_name = caption or (names[0] if names else entity_name)
            matches.append(
                WatchlistMatch(
                    matched_name=matched_name,
                    match_score=_as_float(entry.get("score")),
                    risk_categories=_map_topics(topics),
                    datasets=_as_str_list(entry.get("datasets", [])),
                    linked_entities=_as_str_list(entry.get("referents", [])),
                )
            )

        return WatchlistResult(
            entity_name=entity_name,
            entity_type=entity_type,
            matches=matches,
            asymmetry_note=_ASYMMETRY_NOTE,
        )
