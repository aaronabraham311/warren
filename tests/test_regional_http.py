"""Offline tests for the common regional filing HTTP boundary."""

import sqlite3
from io import BytesIO
from unittest.mock import MagicMock

import requests

from data_sources.errors import DataSourceError
from data_sources.filing_models import SourceSystem
from data_sources.regional_http import HttpDocument, RegionalHttpClient


def _response(
    status: int = 200,
    *,
    url: str = "https://newconnect.pl/announcements",
    text: str = "payload",
    headers: dict[str, str] | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = url
    response._content = text.encode()
    response.raw = BytesIO(response._content)
    response.headers.update(headers or {})
    return response


def _client(session: MagicMock) -> RegionalHttpClient:
    return RegionalHttpClient(
        source=SourceSystem.EBI,
        allowed_hosts=frozenset({"newconnect.pl"}),
        session=session,
        min_interval_seconds=0,
        _sleep=lambda _: None,
    )


def test_host_allowlist_rejects_http_and_unofficial_hosts_without_network() -> None:
    session = MagicMock(spec=requests.Session)
    client = _client(session)

    for url in (
        "http://newconnect.pl/announcements",
        "https://newconnect.pl.evil.example/announcements",
        "https://mirror.example/announcements",
    ):
        result = client.get_text(url)
        assert isinstance(result, DataSourceError)
        assert result.error_code == "parse"
    session.get.assert_not_called()


def test_timeout_retries_then_returns_typed_network_error() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.Timeout("slow")

    result = _client(session).get_text("https://newconnect.pl/announcements")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"
    assert result.stage == "discovery"
    assert session.get.call_count == 3


def test_declared_oversized_response_is_rejected() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _response(headers={"Content-Length": "6"})
    client = RegionalHttpClient(
        source=SourceSystem.EBI,
        allowed_hosts=frozenset({"newconnect.pl"}),
        session=session,
        max_response_bytes=5,
        min_interval_seconds=0,
    )

    result = client.get_text("https://newconnect.pl/announcements")

    assert isinstance(result, DataSourceError)
    assert "size limit" in result.message


def test_chunked_oversized_response_is_rejected() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _response(text="123456")
    client = RegionalHttpClient(
        source=SourceSystem.EBI,
        allowed_hosts=frozenset({"newconnect.pl"}),
        session=session,
        max_response_bytes=5,
        min_interval_seconds=0,
    )

    result = client.get_text("https://newconnect.pl/announcements")

    assert isinstance(result, DataSourceError)
    assert "size limit" in result.message


def test_rate_limit_retries_and_honors_bounded_retry_after() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _response(429, headers={"Retry-After": "90"})
    sleeps: list[float] = []
    client = RegionalHttpClient(
        source=SourceSystem.EBI,
        allowed_hosts=frozenset({"newconnect.pl"}),
        session=session,
        min_interval_seconds=0,
        _sleep=sleeps.append,
    )

    result = client.get_text("https://newconnect.pl/announcements")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "rate_limit"
    assert session.get.call_count == 3
    assert sleeps == [30.0, 30.0]


def test_cache_uses_canonical_parameter_order_and_skips_second_request() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _response(headers={"ETag": '"v1"'})
    client = RegionalHttpClient(
        source=SourceSystem.EBI,
        allowed_hosts=frozenset({"newconnect.pl"}),
        db_conn=sqlite3.connect(":memory:"),
        session=session,
        min_interval_seconds=0,
    )

    first = client.get_text(
        "https://newconnect.pl/announcements", params={"page": "1", "issuer": "CFG"}
    )
    cached = client.get_text(
        "https://newconnect.pl/announcements", params={"issuer": "CFG", "page": "1"}
    )

    assert isinstance(first, HttpDocument)
    assert isinstance(cached, HttpDocument)
    assert cached.from_cache is True
    assert cached.etag == '"v1"'
    session.get.assert_called_once()


def test_cached_final_url_is_revalidated_for_each_clients_allowlist() -> None:
    connection = sqlite3.connect(":memory:")
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _response(url="https://mirror.example/archive")
    broad = RegionalHttpClient(
        source=SourceSystem.EBI,
        allowed_hosts=frozenset({"newconnect.pl", "mirror.example"}),
        db_conn=connection,
        session=session,
        min_interval_seconds=0,
    )
    narrow_session = MagicMock(spec=requests.Session)
    narrow = RegionalHttpClient(
        source=SourceSystem.EBI,
        allowed_hosts=frozenset({"newconnect.pl"}),
        db_conn=connection,
        session=narrow_session,
        min_interval_seconds=0,
    )

    assert isinstance(broad.get_text("https://newconnect.pl/archive"), HttpDocument)
    result = narrow.get_text("https://newconnect.pl/archive")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"
    narrow_session.get.assert_not_called()


def test_redirect_to_unofficial_host_is_rejected() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _response(
        302,
        url="https://newconnect.pl/announcements",
        headers={"Location": "https://mirror.example/archive"},
    )

    result = _client(session).get_text("https://newconnect.pl/announcements")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"
    session.get.assert_called_once()
    assert session.get.call_args.kwargs["allow_redirects"] is False


def test_allowed_redirect_is_followed_without_reapplying_original_params() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [
        _response(
            302,
            url="https://newconnect.pl/archive",
            headers={"Location": "/archive?page=2"},
        ),
        _response(200, url="https://newconnect.pl/archive?page=2"),
    ]

    result = _client(session).get_text(
        "https://newconnect.pl/archive", params={"issuer": "CFG", "page": "1"}
    )

    assert isinstance(result, HttpDocument)
    assert result.url.endswith("page=2")
    assert session.get.call_args_list[0].kwargs["params"] == {"issuer": "CFG", "page": "1"}
    assert session.get.call_args_list[1].kwargs["params"] is None


def test_redirect_from_effective_query_url_to_bare_url_is_not_a_false_loop() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [
        _response(
            302,
            url="https://newconnect.pl/archive?issuer=CFG",
            headers={"Location": "/archive"},
        ),
        _response(200, url="https://newconnect.pl/archive"),
    ]

    result = _client(session).get_text("https://newconnect.pl/archive", params={"issuer": "CFG"})

    assert isinstance(result, HttpDocument)
    assert result.url == "https://newconnect.pl/archive"
    assert session.get.call_count == 2


def test_redirect_loop_is_bounded() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [
        _response(302, url="https://newconnect.pl/a", headers={"Location": "/b"}),
        _response(302, url="https://newconnect.pl/b", headers={"Location": "/a"}),
    ]

    result = _client(session).get_text("https://newconnect.pl/a")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"
    assert "loop" in result.message
    assert session.get.call_count == 2


def test_redirect_chain_stops_at_hard_limit() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [
        _response(302, url="https://newconnect.pl/a", headers={"Location": "/b"}),
        _response(302, url="https://newconnect.pl/b", headers={"Location": "/c"}),
        _response(302, url="https://newconnect.pl/c", headers={"Location": "/d"}),
    ]
    client = RegionalHttpClient(
        source=SourceSystem.EBI,
        allowed_hosts=frozenset({"newconnect.pl"}),
        session=session,
        max_redirects=2,
        min_interval_seconds=0,
        _sleep=lambda _: None,
    )

    result = client.get_text("https://newconnect.pl/a")

    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"
    assert "limit" in result.message
    assert session.get.call_count == 3
