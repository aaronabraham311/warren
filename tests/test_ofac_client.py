"""Unit tests for OFACClient.

All HTTP calls are mocked via an injected requests.Session — no live network,
no API key needed (the autouse socket guard enforces this). Recorded payloads
come from eval/fixtures/TEST/ofac/search_entity/ via the ofac_fixture conftest
fixture. The SQLite cache uses the in-memory ofac_conn fixture.
"""

import sqlite3
from unittest.mock import MagicMock

import pytest
import requests

from data_sources.cache import make_key
from data_sources.errors import DataSourceError
from data_sources.ofac_client import OFACClient, WatchlistMatch, WatchlistResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_client(conn: sqlite3.Connection, mock_session: MagicMock) -> OFACClient:
    return OFACClient(conn, _session=mock_session)


def _mock_session(payload: object, status_code: int = 200) -> MagicMock:
    session = MagicMock(spec=requests.Session)
    resp = MagicMock()
    resp.json.return_value = payload
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=MagicMock(status_code=status_code)
        )
    else:
        resp.raise_for_status.return_value = None
    session.get.return_value = resp
    return session


def _expire_cache(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("UPDATE cache SET expires_at = '2000-01-01T00:00:00+00:00' WHERE key = ?", (key,))
    conn.commit()


# ── Parse ─────────────────────────────────────────────────────────────────────


def test_search_entity_parses_fixture(
    ofac_conn: sqlite3.Connection, ofac_fixture: dict[str, object]
) -> None:
    session = _mock_session(ofac_fixture["match"])
    client = _make_client(ofac_conn, session)

    result = client.search_entity("Test Sanctioned Person", "person", None)

    assert isinstance(result, WatchlistResult)
    assert result.entity_name == "Test Sanctioned Person"
    assert result.entity_type == "person"
    assert len(result.matches) == 2
    assert result.asymmetry_note != ""

    first = result.matches[0]
    assert isinstance(first, WatchlistMatch)
    assert first.matched_name == "TEST SANCTIONED PERSON"
    assert first.match_score == pytest.approx(0.99)
    assert first.risk_categories == ["sanction"]
    assert "UKRAINE-EO13685" in first.datasets
    assert "RUSSIA-EO14024" in first.datasets
    assert "Test SANCTIONED" in first.linked_entities

    second = result.matches[1]
    assert second.matched_name == "TEST RELATED ENTITY"
    assert second.match_score == pytest.approx(0.72)
    assert "SDGT" in second.datasets


def test_search_entity_empty_results(
    ofac_conn: sqlite3.Connection, ofac_fixture: dict[str, object]
) -> None:
    session = _mock_session(ofac_fixture["empty"])
    client = _make_client(ofac_conn, session)

    result = client.search_entity("Clean Entity", "company", None)

    assert isinstance(result, WatchlistResult)
    assert result.matches == []
    assert result.asymmetry_note != ""


def test_search_entity_type_param_sent(
    ofac_conn: sqlite3.Connection, ofac_fixture: dict[str, object]
) -> None:
    session = _mock_session(ofac_fixture["match"])
    client = _make_client(ofac_conn, session)

    client.search_entity("Test Entity", "person", None)

    _, kwargs = session.get.call_args
    assert kwargs["params"]["type"] == "individual"


def test_search_entity_company_maps_to_entity_param(
    ofac_conn: sqlite3.Connection, ofac_fixture: dict[str, object]
) -> None:
    session = _mock_session(ofac_fixture["match"])
    client = _make_client(ofac_conn, session)

    client.search_entity("Test Corp", "company", None)

    _, kwargs = session.get.call_args
    assert kwargs["params"]["type"] == "entity"


def test_score_normalised_to_0_to_1(
    ofac_conn: sqlite3.Connection, ofac_fixture: dict[str, object]
) -> None:
    session = _mock_session(ofac_fixture["match"])
    client = _make_client(ofac_conn, session)

    result = client.search_entity("Test", "person", None)

    assert isinstance(result, WatchlistResult)
    for match in result.matches:
        assert 0.0 <= match.match_score <= 1.0


def test_missing_programs_fallback_to_ofac_sdn(ofac_conn: sqlite3.Connection) -> None:
    payload = {
        "error": None,
        "results": [{"uid": "1", "name": "No Programs Entity", "score": 80, "programs": []}],
    }
    session = _mock_session(payload)
    client = _make_client(ofac_conn, session)

    result = client.search_entity("No Programs Entity", "person", None)

    assert isinstance(result, WatchlistResult)
    assert result.matches[0].datasets == ["OFAC-SDN"]


# ── Caching ───────────────────────────────────────────────────────────────────


def test_search_entity_caches_result(
    ofac_conn: sqlite3.Connection, ofac_fixture: dict[str, object]
) -> None:
    session = _mock_session(ofac_fixture["match"])
    client = _make_client(ofac_conn, session)

    client.search_entity("Test Sanctioned Person", "person", None)
    client.search_entity("Test Sanctioned Person", "person", None)

    assert session.get.call_count == 1, "second call must be a cache hit"


def test_search_entity_expired_cache_refetches(
    ofac_conn: sqlite3.Connection, ofac_fixture: dict[str, object]
) -> None:
    session = _mock_session(ofac_fixture["match"])
    client = _make_client(ofac_conn, session)

    client.search_entity("Test Sanctioned Person", "person", None)
    _expire_cache(ofac_conn, make_key("ofac_search", "test sanctioned person", "person"))
    client.search_entity("Test Sanctioned Person", "person", None)

    assert session.get.call_count == 2, "expired cache should trigger a refetch"


def test_search_entity_different_types_separate_cache(
    ofac_conn: sqlite3.Connection, ofac_fixture: dict[str, object]
) -> None:
    session = _mock_session(ofac_fixture["match"])
    client = _make_client(ofac_conn, session)

    client.search_entity("Test Entity", "person", None)
    client.search_entity("Test Entity", "company", None)

    assert session.get.call_count == 2


# ── Error paths ───────────────────────────────────────────────────────────────


def test_search_entity_network_error(ofac_conn: sqlite3.Connection) -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.ConnectionError("connection refused")
    client = _make_client(ofac_conn, session)

    result = client.search_entity("Test Entity", "person", None)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


def test_search_entity_http_error(ofac_conn: sqlite3.Connection) -> None:
    session = _mock_session({}, status_code=500)
    client = _make_client(ofac_conn, session)

    result = client.search_entity("Test Entity", "person", None)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


def test_search_entity_api_error_field(ofac_conn: sqlite3.Connection) -> None:
    session = _mock_session({"error": "Internal server error", "results": []})
    client = _make_client(ofac_conn, session)

    result = client.search_entity("Test Entity", "person", None)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"


def test_search_entity_non_dict_response(ofac_conn: sqlite3.Connection) -> None:
    session = _mock_session(["unexpected", "list"])
    client = _make_client(ofac_conn, session)

    result = client.search_entity("Test Entity", "person", None)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"
