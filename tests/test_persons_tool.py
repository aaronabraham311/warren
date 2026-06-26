"""Tests for GetKeyPersonsTool.

All network calls are mocked. yfinance is patched at yf.Ticker; EDGAR HTTP
calls are patched via EDGARClient._get. Fixtures from eval/fixtures/AAPL/.
"""

import json
import sqlite3
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.persons import GetKeyPersonsInput, GetKeyPersonsTool, KeyPersonsData
from data_sources.edgar_client import EDGARClient, SC13Holder
from data_sources.yfinance_client import (
    InstitutionalHolderRecord,
    KeyPersonsRaw,
    OfficerRecord,
    YFinanceClient,
)
from eval.fixtures import load_fixture

# ── Helpers ───────────────────────────────────────────────────────────────────


def _no_sleep(seconds: float) -> None:
    pass


def _make_yf_client(conn: sqlite3.Connection) -> YFinanceClient:
    return YFinanceClient(conn, _sleep=_no_sleep)


def _make_edgar_client(conn: sqlite3.Connection) -> EDGARClient:
    return EDGARClient(conn, _sleep=_no_sleep)


def _fixture_yf() -> dict[str, object]:
    return load_fixture("AAPL", "yfinance", "get_key_persons")


def _fixture_edgar() -> dict[str, object]:
    return load_fixture("AAPL", "edgar", "get_sc13_holders")


# ── Fake run context ──────────────────────────────────────────────────────────


class _FakeCtx:
    pass


# ── YFinanceClient.get_key_persons unit tests ─────────────────────────────────


def _as_ih_rows(
    fixture: dict[str, object],
) -> list[tuple[int, dict[str, object]]]:
    rows: list[tuple[int, dict[str, object]]] = []
    ih_raw = fixture.get("institutional_holders")
    if not isinstance(ih_raw, list):
        return rows
    for i, row in enumerate(ih_raw):
        if not isinstance(row, dict):
            continue
        rows.append(
            (
                i,
                {
                    "Holder": row.get("Holder"),
                    "Shares": row.get("Shares"),
                    "% Out": row.get("pct_held"),
                    "Value": row.get("Value"),
                },
            )
        )
    return rows


def _mock_ticker_for_key_persons(fixture: dict[str, object]) -> MagicMock:
    """Build a ticker mock for get_key_persons using the recorded fixture."""
    t = MagicMock()
    t.info = {
        "regularMarketPrice": fixture.get("regularMarketPrice"),
        "currentPrice": fixture.get("currentPrice"),
        "lastFiscalYearEnd": fixture.get("lastFiscalYearEnd"),
        "companyOfficers": fixture.get("companyOfficers", []),
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }
    ih_df = MagicMock()
    ih_df.iterrows.return_value = iter(_as_ih_rows(fixture))
    t.institutional_holders = ih_df
    return t


def test_yf_get_key_persons_officers_and_holders(yf_conn: sqlite3.Connection) -> None:
    fixture = _fixture_yf()
    client = _make_yf_client(yf_conn)
    mock_ticker = _mock_ticker_for_key_persons(fixture)
    with patch("data_sources.yfinance_client.yf.Ticker", return_value=mock_ticker):
        result = client.get_key_persons("AAPL")

    assert isinstance(result, KeyPersonsRaw)
    assert result.ticker == "AAPL"
    assert len(result.officers) == 4
    assert result.officers[0].name == "Timothy D. Cook"
    assert result.officers[0].title == "CEO & Director"
    assert result.officers[0].total_pay_usd == 63151817
    assert result.officers[0].year_born == 1961
    # institutional holders
    assert len(result.institutional_holders) == 4
    assert result.institutional_holders[0].name == "Vanguard Group Inc"
    assert result.institutional_holders[0].pct_held == pytest.approx(0.0796)


def test_yf_get_key_persons_not_found(yf_conn: sqlite3.Connection) -> None:
    client = _make_yf_client(yf_conn)
    empty_ticker = MagicMock()
    empty_ticker.info = {}
    with patch("data_sources.yfinance_client.yf.Ticker", return_value=empty_ticker):
        from data_sources.errors import DataSourceError

        result = client.get_key_persons("ZZZZ")
    assert isinstance(result, DataSourceError)
    assert result.error_code == "not_found"


# ── EDGARClient.get_sc13_holders unit tests ───────────────────────────────────


def _make_edgar_http_mock(fixture: dict[str, object]) -> MagicMock:
    """Return a mock for EDGARClient._get that serves fixture data."""
    company_tickers = fixture["company_tickers"]
    submissions = fixture["submissions"]
    efts_response = fixture["efts_response"]

    def _get_side_effect(url: str) -> MagicMock:
        resp = MagicMock()
        if "company_tickers" in url:
            resp.text = json.dumps(company_tickers)
        elif "submissions" in url:
            resp.text = json.dumps(submissions)
        elif "efts.sec.gov" in url or "search-index" in url:
            resp.text = json.dumps(efts_response)
        else:
            resp.text = "{}"
        resp.status_code = 200
        return resp

    return MagicMock(side_effect=_get_side_effect)


def test_edgar_get_sc13_holders(edgar_conn: sqlite3.Connection) -> None:
    fixture = _fixture_edgar()
    client = _make_edgar_client(edgar_conn)
    with patch.object(client, "_get", _make_edgar_http_mock(fixture)):
        result = client.get_sc13_holders("AAPL")

    assert isinstance(result, list)
    names = {h.name for h in result}
    assert "VANGUARD GROUP INC" in names
    assert "BLACKROCK INC." in names
    assert "GEODE CAPITAL MANAGEMENT, LLC" in names
    # All are SC 13G variants
    for h in result:
        assert "13G" in h.form_type or "13D" in h.form_type


def test_edgar_get_sc13_holders_cached(edgar_conn: sqlite3.Connection) -> None:
    """Second call returns cached result without hitting the network."""
    fixture = _fixture_edgar()
    client = _make_edgar_client(edgar_conn)
    mock_get = _make_edgar_http_mock(fixture)
    with patch.object(client, "_get", mock_get):
        client.get_sc13_holders("AAPL")
        call_count_after_first = mock_get.call_count
        client.get_sc13_holders("AAPL")
    assert mock_get.call_count == call_count_after_first  # no new calls on second run


# ── GetKeyPersonsTool integration tests ───────────────────────────────────────


def _officers_from_fixture(yf_fixture: dict[str, object]) -> list[OfficerRecord]:
    officers: list[OfficerRecord] = []
    officers_raw = yf_fixture.get("companyOfficers")
    if not isinstance(officers_raw, list):
        return officers
    for o in officers_raw:
        if not isinstance(o, dict):
            continue
        name = o.get("name")
        title = o.get("title")
        if not isinstance(name, str) or not isinstance(title, str):
            continue
        year_born_raw = o.get("yearBorn")
        year_born = int(year_born_raw) if isinstance(year_born_raw, (int, float)) else None
        total_pay_raw = o.get("totalPay")
        total_pay: int | None = None
        if isinstance(total_pay_raw, dict):
            raw_val = total_pay_raw.get("raw")
            total_pay = int(raw_val) if isinstance(raw_val, (int, float)) else None
        elif isinstance(total_pay_raw, (int, float)):
            total_pay = int(total_pay_raw)
        officers.append(
            OfficerRecord(name=name, title=title, year_born=year_born, total_pay_usd=total_pay)
        )
    return officers


def _ih_from_fixture(yf_fixture: dict[str, object]) -> list[InstitutionalHolderRecord]:
    holders: list[InstitutionalHolderRecord] = []
    ih_raw = yf_fixture.get("institutional_holders")
    if not isinstance(ih_raw, list):
        return holders
    for ih in ih_raw:
        if not isinstance(ih, dict):
            continue
        name = ih.get("Holder")
        if not isinstance(name, str):
            continue
        shares_raw = ih.get("Shares")
        pct_raw = ih.get("pct_held")
        value_raw = ih.get("Value")
        holders.append(
            InstitutionalHolderRecord(
                name=name,
                shares=int(shares_raw) if isinstance(shares_raw, (int, float)) else None,
                pct_held=float(pct_raw) if isinstance(pct_raw, (int, float)) else None,
                value=int(value_raw) if isinstance(value_raw, (int, float)) else None,
            )
        )
    return holders


def _sc13_holders_from_fixture(edgar_fixture: dict[str, object]) -> list[SC13Holder]:
    holders: list[SC13Holder] = []
    efts = edgar_fixture.get("efts_response")
    if not isinstance(efts, dict):
        return holders
    hits_obj = efts.get("hits")
    if not isinstance(hits_obj, dict):
        return holders
    hits = hits_obj.get("hits")
    if not isinstance(hits, list):
        return holders
    for h in hits:
        if not isinstance(h, dict):
            continue
        source = h.get("_source")
        if not isinstance(source, dict):
            continue
        name = source.get("entity_name")
        form_type = source.get("form_type")
        file_date_str = source.get("file_date")
        if (
            not isinstance(name, str)
            or not isinstance(form_type, str)
            or not isinstance(file_date_str, str)
        ):
            continue
        holders.append(
            SC13Holder(
                name=name,
                form_type=form_type,
                filing_date=date.fromisoformat(file_date_str),
            )
        )
    return holders


def _inject_clients(
    monkeypatch: pytest.MonkeyPatch,
    yf_fixture: dict[str, object],
    edgar_fixture: dict[str, object] | None = None,
) -> None:
    """Monkeypatch the tool-layer singletons with fixture-backed mocks."""
    yf_mock = MagicMock(spec=YFinanceClient)
    raw = KeyPersonsRaw(
        ticker="AAPL",
        as_of=date.today(),
        officers=_officers_from_fixture(yf_fixture),
        institutional_holders=_ih_from_fixture(yf_fixture),
        data_age_hours=0,
    )
    yf_mock.get_key_persons.return_value = raw

    edgar_mock = MagicMock(spec=EDGARClient)
    if edgar_fixture is not None:
        edgar_mock.get_sc13_holders.return_value = _sc13_holders_from_fixture(edgar_fixture)
    else:
        from data_sources.errors import DataSourceError

        edgar_mock.get_sc13_holders.return_value = DataSourceError(
            error_code="network", message="simulated network error"
        )

    monkeypatch.setattr("agent.tools.persons.yfinance_client", lambda: yf_mock)
    monkeypatch.setattr("agent.tools.persons.edgar_client", lambda: edgar_mock)


def test_tool_ok_full(monkeypatch: pytest.MonkeyPatch) -> None:
    yf_fix = _fixture_yf()
    ed_fix = _fixture_edgar()
    _inject_clients(monkeypatch, yf_fix, ed_fix)

    tool = GetKeyPersonsTool()
    result = tool.run(GetKeyPersonsInput(ticker="AAPL"), _FakeCtx())  # type: ignore[arg-type]

    from agent.tools.base import ToolResultOk

    assert isinstance(result, ToolResultOk)
    data = result.data
    assert isinstance(data, KeyPersonsData)
    assert data.ticker == "AAPL"
    # Officers (4) + institutional holders (4) + EDGAR deduped additions
    person_names = {p.name for p in data.persons}
    assert "Timothy D. Cook" in person_names
    assert "Vanguard Group Inc" in person_names
    # AAPL is widely held — no single entity at ≥20%
    assert data.controlling_holder_identified is False
    assert data.source_notes == []


def test_tool_controlling_holder_identified(monkeypatch: pytest.MonkeyPatch) -> None:
    """A holder with pct_held ≥ 0.20 triggers controlling_holder_identified=True."""
    yf_fix: dict[str, object] = {
        "regularMarketPrice": 50.0,
        "currentPrice": 50.0,
        "lastFiscalYearEnd": 1727654400,
        "companyOfficers": [
            {"name": "Jane Doe", "title": "CEO", "yearBorn": 1975, "totalPay": {"raw": 500000}}
        ],
        "institutional_holders": [
            {
                "Holder": "Founding Family Trust",
                "Shares": 30000000,
                "pct_held": 0.35,
                "Value": 1500000000,
            }
        ],
    }
    _inject_clients(monkeypatch, yf_fix, None)

    tool = GetKeyPersonsTool()
    result = tool.run(GetKeyPersonsInput(ticker="AAPL"), _FakeCtx())  # type: ignore[arg-type]

    from agent.tools.base import ToolResultOk

    assert isinstance(result, ToolResultOk)
    data = result.data
    assert isinstance(data, KeyPersonsData)
    assert data.controlling_holder_identified is True


def test_tool_edgar_failure_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    """EDGAR error → ToolResultOk with source_note; yfinance data still returned."""
    yf_fix = _fixture_yf()
    _inject_clients(monkeypatch, yf_fix, None)  # edgar_fixture=None → network error

    tool = GetKeyPersonsTool()
    result = tool.run(GetKeyPersonsInput(ticker="AAPL"), _FakeCtx())  # type: ignore[arg-type]

    from agent.tools.base import ToolResultOk

    assert isinstance(result, ToolResultOk)
    data = result.data
    assert isinstance(data, KeyPersonsData)
    assert len(data.source_notes) == 1
    assert "EDGAR" in data.source_notes[0]
    # Officers and institutional holders still present
    assert any(p.source == "yfinance_officers" for p in data.persons)
    assert any(p.source == "yfinance_holders" for p in data.persons)


def test_tool_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """yfinance returns DataSourceError → ToolResultError(not_found)."""
    from data_sources.errors import DataSourceError

    yf_mock = MagicMock(spec=YFinanceClient)
    yf_mock.get_key_persons.return_value = DataSourceError(
        error_code="not_found", message="No data for ZZZZ"
    )
    monkeypatch.setattr("agent.tools.persons.yfinance_client", lambda: yf_mock)

    tool = GetKeyPersonsTool()
    result = tool.run(GetKeyPersonsInput(ticker="ZZZZ"), _FakeCtx())  # type: ignore[arg-type]

    from agent.tools.base import ToolResultError

    assert isinstance(result, ToolResultError)
    assert result.error_code == "not_found"


def test_tool_registered() -> None:
    """get_key_persons appears in the tool registry and definitions."""
    from agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY

    assert "get_key_persons" in TOOL_REGISTRY
    names = [d["name"] for d in TOOL_DEFINITIONS]
    assert "get_key_persons" in names
