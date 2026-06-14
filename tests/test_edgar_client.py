import json
import sqlite3
from datetime import date

import pytest

from data_sources.edgar_client import MAX_CHARS, EDGARClient, FilingSection
from data_sources.errors import DataSourceError

# ── Fixture payloads ──────────────────────────────────────────────────────────

COMPANY_TICKERS = json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})

SUBMISSIONS = json.dumps(
    {
        "filings": {
            "recent": {
                "form": ["8-K", "10-K", "10-K"],
                "filingDate": ["2023-12-01", "2023-11-03", "2022-10-28"],
                "reportDate": ["2023-12-01", "2023-09-30", "2022-09-24"],
                "accessionNumber": [
                    "0000320193-23-000110",
                    "0000320193-23-000106",
                    "0000320193-22-000108",
                ],
                "primaryDocument": ["aapl-8k.htm", "aapl-20230930.htm", "aapl-20220924.htm"],
            }
        }
    }
)


def _filing_html(mdna_body: str) -> str:
    return f"""<html><body>
    <p>Table of Contents</p>
    <p>Item 1. Business .... Item 1A. Risk Factors .... Item 7. MD&amp;A .... Item 8.</p>
    <p>Item 1. Business</p><p>Apple designs phones.</p>
    <p>Item 1A. Risk Factors</p><p>Supply chain risk and competition.</p>
    <p>Item 7. Management's Discussion and Analysis</p>
    <p>{mdna_body}</p>
    <p>Item 7A. Quantitative and Qualitative Disclosures</p><p>rates</p>
    <p>Item 8. Financial Statements and Supplementary Data</p><p>tables</p>
    </body></html>"""


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


def _route(url: str, mdna_body: str) -> _FakeResponse:
    if "company_tickers.json" in url:
        return _FakeResponse(200, COMPANY_TICKERS)
    if "/submissions/CIK" in url:
        return _FakeResponse(200, SUBMISSIONS)
    if "/Archives/" in url:
        return _FakeResponse(200, _filing_html(mdna_body))
    return _FakeResponse(404, "")


def _build_client(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    mdna_body: str = "Revenue grew strongly this year.",
) -> tuple[EDGARClient, list[str]]:
    """Build an EDGARClient whose session.get is faked. Returns (client, urls_seen)."""
    client = EDGARClient(conn, _sleep=lambda _s: None)
    urls: list[str] = []

    def fake_get(url: str, timeout: int | None = None) -> _FakeResponse:
        urls.append(url)
        return _route(url, mdna_body)

    monkeypatch.setattr(client._session, "get", fake_get)
    return client, urls


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_mdna_returns_content(
    edgar_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _urls = _build_client(edgar_conn, monkeypatch)
    result = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(result, FilingSection)
    assert "Revenue grew strongly" in result.text
    assert result.text.strip() != ""
    assert result.filing_date == date(2023, 11, 3)
    assert result.fiscal_year == 2023
    assert result.truncated is False
    assert result.word_count > 0
    assert result.edgar_url.endswith("aapl-20230930.htm")


def test_truncation_caps_at_max_chars(
    edgar_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _urls = _build_client(edgar_conn, monkeypatch, mdna_body="x " * (MAX_CHARS // 2 + 5000))
    result = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(result, FilingSection)
    assert len(result.text) == MAX_CHARS
    assert result.truncated is True


def test_second_call_uses_cache_zero_network(
    edgar_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, urls = _build_client(edgar_conn, monkeypatch)
    first = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(first, FilingSection)
    calls_after_first = len(urls)
    assert calls_after_first == 3  # company_tickers + submissions + document

    second = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(second, FilingSection)
    assert len(urls) == calls_after_first  # zero additional EDGAR requests
    assert second.text == first.text


def test_invalid_ticker_returns_not_found(
    edgar_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _urls = _build_client(edgar_conn, monkeypatch)
    result = client.get_filing_section("ZZZZ", "10-K", "mdna")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"


def test_user_agent_header_always_present(edgar_conn: sqlite3.Connection) -> None:
    client = EDGARClient(edgar_conn)
    assert client._session.headers["User-Agent"] == EDGARClient.HEADERS["User-Agent"]


def test_risk_factors_section_extracted(
    edgar_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _urls = _build_client(edgar_conn, monkeypatch)
    result = client.get_filing_section("AAPL", "10-K", "risk_factors")
    assert isinstance(result, FilingSection)
    assert "Supply chain risk" in result.text


def test_fiscal_year_selects_older_filing(
    edgar_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _urls = _build_client(edgar_conn, monkeypatch)
    result = client.get_filing_section("AAPL", "10-K", "mdna", fiscal_year=2022)
    assert isinstance(result, FilingSection)
    assert result.fiscal_year == 2022
    assert result.filing_date == date(2022, 10, 28)


def test_http_error_returns_network(
    edgar_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = EDGARClient(edgar_conn, _sleep=lambda _s: None)

    def fake_get(url: str, timeout: int | None = None) -> _FakeResponse:
        return _FakeResponse(503, "")

    monkeypatch.setattr(client._session, "get", fake_get)
    result = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "network"


def test_non_json_200_returns_parse(
    edgar_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200 response with a non-JSON body must be returned as parse, never raised."""
    client = EDGARClient(edgar_conn, _sleep=lambda _s: None)

    def fake_get(url: str, timeout: int | None = None) -> _FakeResponse:
        return _FakeResponse(200, "<html>EDGAR is temporarily unavailable</html>")

    monkeypatch.setattr(client._session, "get", fake_get)
    result = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"


def test_malformed_submissions_returns_parse(
    edgar_conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid JSON with a null 'filings' key must be returned as parse, never raised."""
    client = EDGARClient(edgar_conn, _sleep=lambda _s: None)

    def fake_get(url: str, timeout: int | None = None) -> _FakeResponse:
        if "company_tickers.json" in url:
            return _FakeResponse(200, COMPANY_TICKERS)
        return _FakeResponse(200, json.dumps({"filings": None}))

    monkeypatch.setattr(client._session, "get", fake_get)
    result = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"
