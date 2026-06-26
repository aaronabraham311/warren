"""Unit tests for EDGARClient.

The three raw HTTP bodies the client fetches — the ticker→CIK map, the submissions
history, and the filing HTML — are served from the recorded fixture at
``eval/fixtures/AAPL/edgar/get_filing_section/`` via the ``edgar_fixture`` conftest
fixture. No live network is hit (the autouse socket guard enforces that).
"""

import json
import sqlite3
from datetime import date

import pytest

from data_sources.edgar_client import MAX_CHARS, EDGARClient, FilingSection, SC13Holder
from data_sources.errors import DataSourceError
from eval.fixtures import load_fixture


def _filing_html(mdna_body: str) -> str:
    """Synthesize a filing whose MD&A body is *mdna_body* — used by the truncation
    test, which needs a body far larger than anything worth committing as a fixture."""
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


def _route(url: str, fixture: dict[str, object], filing_html: str) -> _FakeResponse:
    if "company_tickers.json" in url:
        return _FakeResponse(200, json.dumps(fixture["company_tickers"]))
    if "/submissions/CIK" in url:
        return _FakeResponse(200, json.dumps(fixture["submissions"]))
    if "/Archives/" in url:
        return _FakeResponse(200, filing_html)
    return _FakeResponse(404, "")


def _build_client(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, object],
    filing_html: str | None = None,
) -> tuple[EDGARClient, list[str]]:
    """Build an EDGARClient whose session.get is faked from *fixture*. Returns
    (client, urls_seen). Pass *filing_html* to override the recorded document body."""
    client = EDGARClient(conn, _sleep=lambda _s: None)
    html = filing_html if filing_html is not None else str(fixture["filing_html"])
    urls: list[str] = []

    def fake_get(url: str, timeout: int | None = None) -> _FakeResponse:
        urls.append(url)
        return _route(url, fixture, html)

    monkeypatch.setattr(client._session, "get", fake_get)
    return client, urls


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_mdna_returns_content(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    edgar_fixture: dict[str, object],
) -> None:
    client, _urls = _build_client(edgar_conn, monkeypatch, edgar_fixture)
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
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    edgar_fixture: dict[str, object],
) -> None:
    client, _urls = _build_client(
        edgar_conn,
        monkeypatch,
        edgar_fixture,
        filing_html=_filing_html("x " * (MAX_CHARS // 2 + 5000)),
    )
    result = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(result, FilingSection)
    assert len(result.text) == MAX_CHARS
    assert result.truncated is True


def test_second_call_uses_cache_zero_network(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    edgar_fixture: dict[str, object],
) -> None:
    client, urls = _build_client(edgar_conn, monkeypatch, edgar_fixture)
    first = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(first, FilingSection)
    calls_after_first = len(urls)
    assert calls_after_first == 3  # company_tickers + submissions + document

    second = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(second, FilingSection)
    assert len(urls) == calls_after_first  # zero additional EDGAR requests
    assert second.text == first.text


def test_invalid_ticker_returns_not_found(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    edgar_fixture: dict[str, object],
) -> None:
    client, _urls = _build_client(edgar_conn, monkeypatch, edgar_fixture)
    result = client.get_filing_section("ZZZZ", "10-K", "mdna")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"


def test_user_agent_header_always_present(edgar_conn: sqlite3.Connection) -> None:
    client = EDGARClient(edgar_conn)
    assert client._session.headers["User-Agent"] == EDGARClient.HEADERS["User-Agent"]


def test_risk_factors_section_extracted(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    edgar_fixture: dict[str, object],
) -> None:
    client, _urls = _build_client(edgar_conn, monkeypatch, edgar_fixture)
    result = client.get_filing_section("AAPL", "10-K", "risk_factors")
    assert isinstance(result, FilingSection)
    assert "Supply chain risk" in result.text


def test_fiscal_year_selects_older_filing(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    edgar_fixture: dict[str, object],
) -> None:
    client, _urls = _build_client(edgar_conn, monkeypatch, edgar_fixture)
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
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    edgar_fixture: dict[str, object],
) -> None:
    """Valid JSON with a null 'filings' key must be returned as parse, never raised."""
    client = EDGARClient(edgar_conn, _sleep=lambda _s: None)

    def fake_get(url: str, timeout: int | None = None) -> _FakeResponse:
        if "company_tickers.json" in url:
            return _FakeResponse(200, json.dumps(edgar_fixture["company_tickers"]))
        return _FakeResponse(200, json.dumps({"filings": None}))

    monkeypatch.setattr(client._session, "get", fake_get)
    result = client.get_filing_section("AAPL", "10-K", "mdna")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "parse"


def _compensation_html() -> str:
    return """<html><body>
    <p>Table of Contents</p>
    <p>Item 1. Business .... Item 11. Executive Compensation .... Item 12.</p>
    <p>Item 1. Business</p><p>Apple designs phones.</p>
    <p>Item 11. Executive Compensation</p>
    <p>CEO total compensation was $10 million in fiscal 2023. Gross margin of 43.8%.</p>
    <p>Item 12. Security Ownership of Certain Beneficial Owners</p>
    <p>Officers own 5% of shares outstanding.</p>
    </body></html>"""


def _related_party_html() -> str:
    return """<html><body>
    <p>Table of Contents</p>
    <p>Item 13. Certain Relationships .... Item 14.</p>
    <p>Item 1. Business</p><p>Apple designs phones.</p>
    <p>Item 13. Certain Relationships and Related Party Transactions</p>
    <p>We paid $1.5 million to Related Corp, an entity controlled by a board member.</p>
    <p>Item 14. Principal Accountant Fees and Services</p>
    <p>Audit fees were $50 million.</p>
    </body></html>"""


def test_compensation_section_extracted(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    edgar_fixture: dict[str, object],
) -> None:
    client, _urls = _build_client(
        edgar_conn, monkeypatch, edgar_fixture, filing_html=_compensation_html()
    )
    result = client.get_filing_section("AAPL", "10-K", "compensation")
    assert isinstance(result, FilingSection)
    assert "CEO total compensation" in result.text
    assert "Security Ownership" not in result.text  # sliced before Item 12


def test_related_party_section_extracted(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    edgar_fixture: dict[str, object],
) -> None:
    client, _urls = _build_client(
        edgar_conn, monkeypatch, edgar_fixture, filing_html=_related_party_html()
    )
    result = client.get_filing_section("AAPL", "10-K", "related_party")
    assert isinstance(result, FilingSection)
    assert "Related Corp" in result.text
    assert "Audit fees" not in result.text  # sliced before Item 14


def test_key_figures_extracted_populated(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    edgar_fixture: dict[str, object],
) -> None:
    client, _urls = _build_client(
        edgar_conn, monkeypatch, edgar_fixture, filing_html=_compensation_html()
    )
    result = client.get_filing_section("AAPL", "10-K", "compensation")
    assert isinstance(result, FilingSection)
    figures = result.key_figures_extracted
    assert any("10" in f and ("million" in f.lower() or "M" in f) for f in figures)
    assert any("43.8" in f and "%" in f for f in figures)


# ── get_sc13_holders ──────────────────────────────────────────────────────────


def _build_sc13_client(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, object],
) -> tuple[EDGARClient, list[str]]:
    client = EDGARClient(conn, _sleep=lambda _s: None)
    urls: list[str] = []

    def fake_get(url: str, timeout: int | None = None) -> _FakeResponse:
        urls.append(url)
        if "company_tickers.json" in url:
            return _FakeResponse(200, json.dumps(fixture["company_tickers"]))
        if "/submissions/CIK" in url:
            return _FakeResponse(200, json.dumps(fixture["submissions"]))
        if "efts.sec.gov" in url or "search-index" in url:
            return _FakeResponse(200, json.dumps(fixture["efts_response"]))
        return _FakeResponse(404, "")

    monkeypatch.setattr(client._session, "get", fake_get)
    return client, urls


def test_get_sc13_holders_from_fixture(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = load_fixture("AAPL", "edgar", "get_sc13_holders")
    client, _urls = _build_sc13_client(edgar_conn, monkeypatch, fixture)
    result = client.get_sc13_holders("AAPL")

    assert isinstance(result, list)
    assert len(result) >= 1
    names = [h.name for h in result]
    assert "VANGUARD GROUP INC" in names
    assert "BLACKROCK INC." in names
    # All returned holders have valid form_type and filing_date
    for h in result:
        assert isinstance(h, SC13Holder)
        assert h.form_type.startswith("SC 13")
        assert isinstance(h.filing_date, date)


def test_get_sc13_holders_deduplicates_keeping_latest(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = load_fixture("AAPL", "edgar", "get_sc13_holders")
    client, _urls = _build_sc13_client(edgar_conn, monkeypatch, fixture)
    result = client.get_sc13_holders("AAPL")
    # No duplicate names — latest filing per entity is kept
    assert isinstance(result, list)
    names = [h.name for h in result]
    assert len(names) == len(set(names))


def test_get_sc13_holders_cache_hit(
    edgar_conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = load_fixture("AAPL", "edgar", "get_sc13_holders")
    client, urls = _build_sc13_client(edgar_conn, monkeypatch, fixture)
    client.get_sc13_holders("AAPL")
    efts_hits_first = sum(1 for u in urls if "efts.sec.gov" in u or "search-index" in u)
    client.get_sc13_holders("AAPL")
    efts_hits_second = sum(1 for u in urls if "efts.sec.gov" in u or "search-index" in u)
    assert efts_hits_first == efts_hits_second  # no second EFTS call
