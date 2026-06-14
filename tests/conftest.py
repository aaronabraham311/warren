import socket
import sqlite3
from collections.abc import Callable, Generator
from typing import NoReturn
from unittest.mock import MagicMock

import anthropic
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import storage.engine as eng
from eval.fixtures import load_fixture
from storage.models import Base


@pytest.fixture(autouse=True)
def _no_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that attempts a real network connection.

    Blocks at the socket layer so the guarantee holds regardless of which HTTP
    library a client uses. Mocked calls never reach the socket, so they are
    unaffected — only an unmocked live call (requests.get, yfinance.Ticker, the
    finnhub SDK, …) trips this and turns into a loud failure.
    """

    def _blocked(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("Live network access is disabled in tests; mock the call.")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)


@pytest.fixture()
def edgar_fixture() -> dict[str, object]:
    """Recorded EDGAR payloads: {company_tickers, submissions, filing_html}."""
    return load_fixture("AAPL", "edgar", "get_filing_section")


@pytest.fixture()
def finnhub_fixture() -> dict[str, object]:
    """Recorded Finnhub payloads: {"news": [...], "financials": {...}}."""
    return {
        "news": load_fixture("AAPL", "finnhub", "get_news")["items"],
        "financials": load_fixture("AAPL", "finnhub", "get_basic_financials"),
    }


@pytest.fixture()
def db_engine(monkeypatch: pytest.MonkeyPatch) -> Generator[Engine, None, None]:
    """In-memory SQLite engine; patches storage.engine.engine for the test."""
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(test_engine)
    monkeypatch.setattr(eng, "engine", test_engine)
    yield test_engine
    test_engine.dispose()


@pytest.fixture()
def yf_conn() -> Generator[sqlite3.Connection, None, None]:
    """In-memory SQLite connection for YFinanceClient cache tests."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture()
def edgar_conn() -> Generator[sqlite3.Connection, None, None]:
    """In-memory SQLite connection for EDGARClient cache tests."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture()
def finnhub_conn() -> Generator[sqlite3.Connection, None, None]:
    """In-memory SQLite connection for FinnhubClient cache tests."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture()
def db_session(db_engine: Engine) -> Generator[Session, None, None]:
    with Session(db_engine) as session:
        yield session


def make_usage(input_tokens: int = 100, output_tokens: int = 50) -> anthropic.types.Usage:
    return anthropic.types.Usage(input_tokens=input_tokens, output_tokens=output_tokens)


def make_end_turn(
    text: str, input_tokens: int = 100, output_tokens: int = 50
) -> anthropic.types.Message:
    return anthropic.types.Message(
        id="msg_01",
        type="message",
        role="assistant",
        content=[anthropic.types.TextBlock(type="text", text=text)],
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=make_usage(input_tokens, output_tokens),
    )


def make_tool_use(
    tool_name: str,
    tool_input: dict[str, object],
    tool_id: str = "toolu_01",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> anthropic.types.Message:
    return anthropic.types.Message(
        id="msg_02",
        type="message",
        role="assistant",
        content=[
            anthropic.types.ToolUseBlock(
                id=tool_id, type="tool_use", name=tool_name, input=tool_input
            )
        ],
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
        stop_sequence=None,
        usage=make_usage(input_tokens, output_tokens),
    )


VALID_ANALYSIS_JSON = """{
  "ticker": "AAPL",
  "analysis_type": "holding",
  "recommendation": "hold",
  "confidence": 0.72,
  "thesis": "Apple has a durable moat and strong free cash flow but trades near fair value.",
  "lynch_signals": ["dominant brand", "consistent earnings"],
  "buffett_signals": ["high ROE", "strong FCF", "consumer moat"],
  "key_risks": ["valuation stretched", "China exposure"],
  "data_quality_notes": []
}"""


@pytest.fixture()
def mock_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[list[anthropic.types.Message]], MagicMock]:
    """Returns a factory: call with a list of Message responses to queue them up.

    The mock client is injected directly into analyze_ticker via monkeypatching
    the module-level default so tests don't need to pass it explicitly.
    """

    def _setup(responses: list[anthropic.types.Message]) -> MagicMock:
        queue = list(responses)
        mock_client = MagicMock(spec=anthropic.Anthropic)
        mock_client.messages.create.side_effect = lambda **_kw: queue.pop(0)
        # Patch the Anthropic constructor so analyze_ticker's fallback path
        # (client=None) also receives the mock client.
        monkeypatch.setattr("agent.loop.anthropic.Anthropic", MagicMock(return_value=mock_client))
        return mock_client

    return _setup
