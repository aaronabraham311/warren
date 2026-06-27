"""Unit tests for OpenSanctionsClient.

All HTTP calls are mocked via an injected requests.Session mock — no live network,
no API key needed (the autouse socket guard enforces this). Recorded payloads come
from eval/fixtures/TEST/opensanctions/match_entity/ via the opensanctions_fixture
conftest fixture. The SQLite cache uses the in-memory opensanctions_conn fixture.
"""

import sqlite3
from unittest.mock import MagicMock

import pytest
import requests

from data_sources.cache import make_key
from data_sources.errors import DataSourceError
from data_sources.opensanctions_client import (
    OpenSanctionsClient,
    WatchlistMatch,
    WatchlistResult,
    _map_topic,
    _map_topics,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_client(
    conn: sqlite3.Connection,
    mock_session: MagicMock,
    api_key: str = "test-key",
) -> OpenSanctionsClient:
    return OpenSanctionsClient(conn, api_key=api_key, _session=mock_session)


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
    session.post.return_value = resp
    return session


def _expire_cache(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("UPDATE cache SET expires_at = '2000-01-01T00:00:00+00:00' WHERE key = ?", (key,))
    conn.commit()


# ── Topic mapping ─────────────────────────────────────────────────────────────


def test_map_topic_sanction() -> None:
    assert _map_topic("sanction") == "sanction"
    assert _map_topic("sanction.linked") == "sanction"


def test_map_topic_pep() -> None:
    assert _map_topic("pep") == "pep"
    assert _map_topic("pep.national") == "pep"
    assert _map_topic("role.pep") == "pep"
    assert _map_topic("role.pep.regional") == "pep"


def test_map_topic_criminal() -> None:
    assert _map_topic("crime") == "criminal"
    assert _map_topic("crime.fraud") == "criminal"
    assert _map_topic("crime.theft") == "criminal"


def test_map_topic_debarment() -> None:
    assert _map_topic("debarment") == "debarment"
    assert _map_topic("debarment.wto") == "debarment"


def test_map_topic_other() -> None:
    assert _map_topic("poi") == "other"
    assert _map_topic("role.rca") == "other"
    assert _map_topic("unknown") == "other"


def test_map_topics_deduplicates() -> None:
    topics = ["sanction", "sanction.linked", "role.pep"]
    result = _map_topics(topics)
    assert result.count("sanction") == 1
    assert "pep" in result


# ── match_entity — parse ──────────────────────────────────────────────────────


def test_match_entity_parses_fixture(
    opensanctions_conn: sqlite3.Connection, opensanctions_fixture: dict[str, object]
) -> None:
    session = _mock_session(opensanctions_fixture["match"])
    client = _make_client(opensanctions_conn, session)

    result = client.match_entity("Test Sanctioned Person", "person", None)

    assert isinstance(result, WatchlistResult)
    assert result.entity_name == "Test Sanctioned Person"
    assert result.entity_type == "person"
    assert len(result.matches) == 2
    assert result.asymmetry_note != ""

    first = result.matches[0]
    assert isinstance(first, WatchlistMatch)
    assert first.matched_name == "Test Sanctioned Person"
    assert first.match_score == pytest.approx(0.92)
    assert "sanction" in first.risk_categories
    assert "pep" in first.risk_categories
    assert "us_ofac_sdn" in first.datasets
    assert "eu_fsf" in first.datasets
    assert "linked-entity-1" in first.linked_entities

    second = result.matches[1]
    assert "criminal" in second.risk_categories
    assert "debarment" in second.risk_categories


def test_match_entity_empty_results(
    opensanctions_conn: sqlite3.Connection, opensanctions_fixture: dict[str, object]
) -> None:
    session = _mock_session(opensanctions_fixture["empty"])
    client = _make_client(opensanctions_conn, session)

    result = client.match_entity("Clean Entity", "company", None)

    assert isinstance(result, WatchlistResult)
    assert result.matches == []
    assert result.asymmetry_note != ""


def test_match_entity_country_hint_sent_in_properties(
    opensanctions_conn: sqlite3.Connection, opensanctions_fixture: dict[str, object]
) -> None:
    session = _mock_session(opensanctions_fixture["match"])
    client = _make_client(opensanctions_conn, session)

    client.match_entity("Test Sanctioned Person", "person", "ru")

    _, kwargs = session.post.call_args
    payload = kwargs["json"] if "json" in kwargs else session.post.call_args[0][1]
    q0_props = payload["queries"]["q0"]["properties"]
    assert q0_props["country"] == ["ru"]


def test_match_entity_api_key_sent_in_header(
    opensanctions_conn: sqlite3.Connection, opensanctions_fixture: dict[str, object]
) -> None:
    session = _mock_session(opensanctions_fixture["match"])
    client = _make_client(opensanctions_conn, session, api_key="my-secret-key")

    client.match_entity("Test Entity", "person", None)

    _, kwargs = session.post.call_args
    headers = kwargs.get("headers", {})
    assert headers.get("Authorization") == "ApiKey my-secret-key"


def test_match_entity_no_api_key_omits_auth_header(
    opensanctions_conn: sqlite3.Connection, opensanctions_fixture: dict[str, object]
) -> None:
    session = _mock_session(opensanctions_fixture["match"])
    client = _make_client(opensanctions_conn, session, api_key="")

    client.match_entity("Test Entity", "person", None)

    _, kwargs = session.post.call_args
    headers = kwargs.get("headers", {})
    assert "Authorization" not in headers


# ── Caching ───────────────────────────────────────────────────────────────────


def test_match_entity_caches_result(
    opensanctions_conn: sqlite3.Connection, opensanctions_fixture: dict[str, object]
) -> None:
    session = _mock_session(opensanctions_fixture["match"])
    client = _make_client(opensanctions_conn, session)

    client.match_entity("Test Sanctioned Person", "person", None)
    client.match_entity("Test Sanctioned Person", "person", None)

    assert session.post.call_count == 1, "second call must be a cache hit"


def test_match_entity_expired_cache_refetches(
    opensanctions_conn: sqlite3.Connection, opensanctions_fixture: dict[str, object]
) -> None:
    session = _mock_session(opensanctions_fixture["match"])
    client = _make_client(opensanctions_conn, session)

    client.match_entity("Test Sanctioned Person", "person", None)
    _expire_cache(
        opensanctions_conn,
        make_key("opensanctions_match", "test sanctioned person", "person", ""),
    )
    client.match_entity("Test Sanctioned Person", "person", None)

    assert session.post.call_count == 2, "expired cache should trigger a refetch"


def test_match_entity_different_country_hints_separate_cache_entries(
    opensanctions_conn: sqlite3.Connection, opensanctions_fixture: dict[str, object]
) -> None:
    session = _mock_session(opensanctions_fixture["match"])
    client = _make_client(opensanctions_conn, session)

    client.match_entity("Test Entity", "person", "ru")
    client.match_entity("Test Entity", "person", "cn")

    assert session.post.call_count == 2


# ── Error paths ───────────────────────────────────────────────────────────────


def test_match_entity_network_error_returns_datasource_error(
    opensanctions_conn: sqlite3.Connection,
) -> None:
    session = MagicMock(spec=requests.Session)
    session.post.side_effect = requests.ConnectionError("connection refused")
    client = _make_client(opensanctions_conn, session)

    result = client.match_entity("Test Entity", "person", None)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


def test_match_entity_http_error_returns_datasource_error(
    opensanctions_conn: sqlite3.Connection,
) -> None:
    session = _mock_session({}, status_code=429)
    client = _make_client(opensanctions_conn, session)

    result = client.match_entity("Test Entity", "person", None)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


def test_match_entity_malformed_response_returns_parse_error(
    opensanctions_conn: sqlite3.Connection,
) -> None:
    session = _mock_session({"not_responses": True})
    client = _make_client(opensanctions_conn, session)

    result = client.match_entity("Test Entity", "person", None)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"


def test_match_entity_non_dict_response_returns_parse_error(
    opensanctions_conn: sqlite3.Connection,
) -> None:
    session = _mock_session(["unexpected", "list"])
    client = _make_client(opensanctions_conn, session)

    result = client.match_entity("Test Entity", "person", None)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"


# ── License note ──────────────────────────────────────────────────────────────


def test_license_note_logged_at_construction(
    opensanctions_conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="data_sources.opensanctions_client"):
        OpenSanctionsClient(opensanctions_conn, _session=MagicMock(spec=requests.Session))

    assert any("non-commercial" in record.message.lower() for record in caplog.records)
