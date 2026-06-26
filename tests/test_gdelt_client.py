"""Unit tests for GDELTClient.

All HTTP calls are mocked via monkeypatching requests.Session.get — no live
network, guaranteed by the autouse socket guard in conftest.py. The cache uses
the in-memory gdelt_conn fixture. Fixture payloads come from
eval/fixtures/AAPL/gdelt/get_adverse_articles/apple_30d.json, which matches
the actual GDELT DOC 2.0 ArtList response shape.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
import requests

from data_sources.cache import make_key
from data_sources.errors import DataSourceError
from data_sources.gdelt_client import GDELTClient, RawArticle


def _no_sleep(seconds: float) -> None:
    pass


def _make_client(
    conn: sqlite3.Connection,
    mock_session: MagicMock,
    _sleep: object = _no_sleep,
) -> GDELTClient:
    with patch("data_sources.gdelt_client.requests.Session", return_value=mock_session):
        return GDELTClient(conn, _sleep=_sleep)  # type: ignore[arg-type]


def _mock_response(data: object, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status.side_effect = (
        None if status_code == 200 else requests.HTTPError(response=resp)
    )
    return resp


def _expire_cache(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("UPDATE cache SET expires_at = '2000-01-01T00:00:00+00:00' WHERE key = ?", (key,))
    conn.commit()


# ── Parsing ───────────────────────────────────────────────────────────────────


def test_parse_returns_articles_from_fixture(
    gdelt_conn: sqlite3.Connection, gdelt_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.get.return_value = _mock_response(gdelt_fixture)
    client = _make_client(gdelt_conn, mock)

    result = client.get_adverse_articles("Apple Inc", 30)

    assert isinstance(result, list)
    assert len(result) == 5
    assert all(isinstance(a, RawArticle) for a in result)
    # Fields present in actual GDELT ArtList response
    assert result[0].domain == "mondaq.com"
    assert result[0].language == "English"
    assert result[0].seendate == "20260527T174500Z"
    # tone and themes are optional; GDELT DOC API does not return them
    assert result[0].tone is None
    assert result[0].themes == ""


def test_parse_includes_non_english_articles(
    gdelt_conn: sqlite3.Connection, gdelt_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.get.return_value = _mock_response(gdelt_fixture)
    client = _make_client(gdelt_conn, mock)

    result = client.get_adverse_articles("Apple Inc", 30)

    assert isinstance(result, list)
    languages = {a.language for a in result}
    assert "French" in languages


def test_parse_with_tone_and_themes_when_present(gdelt_conn: sqlite3.Connection) -> None:
    """Theme and tone fields are parsed when the API happens to return them."""
    payload = {
        "articles": [
            {
                "url": "https://example.com/article",
                "url_mobile": "",
                "title": "Company Under Investigation",
                "seendate": "20260601T120000Z",
                "socialimage": "",
                "domain": "example.com",
                "language": "English",
                "sourcecountry": "US",
                "tone": -6.5,
                "themes": "CRIME_FRAUD CORRUPTION",
            }
        ]
    }
    mock = MagicMock()
    mock.get.return_value = _mock_response(payload)
    client = _make_client(gdelt_conn, mock)

    result = client.get_adverse_articles("Company", 7)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].tone == pytest.approx(-6.5)
    assert result[0].themes == "CRIME_FRAUD CORRUPTION"


def test_empty_articles_key_returns_empty_list(gdelt_conn: sqlite3.Connection) -> None:
    mock = MagicMock()
    mock.get.return_value = _mock_response({"articles": []})
    client = _make_client(gdelt_conn, mock)

    result = client.get_adverse_articles("NoResults Corp", 30)

    assert result == []


def test_missing_articles_key_returns_empty_list(gdelt_conn: sqlite3.Connection) -> None:
    mock = MagicMock()
    mock.get.return_value = _mock_response({})
    client = _make_client(gdelt_conn, mock)

    result = client.get_adverse_articles("NoResults Corp", 30)

    assert result == []


def test_malformed_entry_is_skipped(gdelt_conn: sqlite3.Connection) -> None:
    payload = {
        "articles": [
            "not_a_dict",
            {
                "url": "https://example.com/ok",
                "url_mobile": "",
                "title": "Valid Article",
                "seendate": "20260601T000000Z",
                "socialimage": "",
                "domain": "example.com",
                "language": "English",
                "sourcecountry": "US",
            },
        ]
    }
    mock = MagicMock()
    mock.get.return_value = _mock_response(payload)
    client = _make_client(gdelt_conn, mock)

    result = client.get_adverse_articles("Corp", 7)

    assert isinstance(result, list)
    assert len(result) == 1


# ── Network errors ────────────────────────────────────────────────────────────


def test_network_error_returns_datasource_error(gdelt_conn: sqlite3.Connection) -> None:
    mock = MagicMock()
    mock.get.side_effect = requests.ConnectionError("connection refused")
    client = _make_client(gdelt_conn, mock)

    result = client.get_adverse_articles("Apple Inc", 30)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


def test_http_error_returns_datasource_error(gdelt_conn: sqlite3.Connection) -> None:
    mock = MagicMock()
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")
    mock.get.return_value = resp
    client = _make_client(gdelt_conn, mock)

    result = client.get_adverse_articles("Apple Inc", 30)

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


# ── Caching ───────────────────────────────────────────────────────────────────


def test_cache_hit_skips_http(
    gdelt_conn: sqlite3.Connection, gdelt_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.get.return_value = _mock_response(gdelt_fixture)
    client = _make_client(gdelt_conn, mock)

    # First call — populates cache.
    result1 = client.get_adverse_articles("Apple Inc", 30)
    assert isinstance(result1, list)
    assert mock.get.call_count == 1

    # Second call — served from cache, no new HTTP request.
    result2 = client.get_adverse_articles("Apple Inc", 30)
    assert isinstance(result2, list)
    assert mock.get.call_count == 1
    assert len(result1) == len(result2)


def test_expired_cache_refetches(
    gdelt_conn: sqlite3.Connection, gdelt_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.get.return_value = _mock_response(gdelt_fixture)
    client = _make_client(gdelt_conn, mock)

    client.get_adverse_articles("Apple Inc", 30)
    key = make_key("gdelt_adverse", "Apple Inc", "30")
    _expire_cache(gdelt_conn, key)

    client.get_adverse_articles("Apple Inc", 30)

    assert mock.get.call_count == 2


def test_different_queries_cached_independently(
    gdelt_conn: sqlite3.Connection, gdelt_fixture: dict[str, object]
) -> None:
    mock = MagicMock()
    mock.get.return_value = _mock_response(gdelt_fixture)
    client = _make_client(gdelt_conn, mock)

    client.get_adverse_articles("Apple Inc", 30)
    client.get_adverse_articles("Samsung Electronics", 30)

    assert mock.get.call_count == 2
