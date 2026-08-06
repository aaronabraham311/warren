from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agent.requests import (
    Clarification,
    FollowUpKind,
    RecentContext,
    RequestIntent,
    RunnableRequest,
    StoredResultFollowUp,
    UnsupportedRequest,
    parse_request,
)
from agent.service import RunMode


@pytest.mark.parametrize(
    ("text", "ticker", "persona"),
    [
        ("Analyze AAPL", "AAPL", None),
        ("analyse brk.b", "BRK.B", None),
        ("Take a look at COST using DIRT.", "COST", "dirt"),
        ("Evaluate kpl.wa with default persona", "KPL.WA", "default"),
    ],
)
def test_analyze_grammar_normalizes_canonical_ticker(
    text: str, ticker: str, persona: str | None
) -> None:
    result = parse_request(text)
    assert result == RunnableRequest(RequestIntent.ANALYZE, (ticker,), persona)  # type: ignore[arg-type]


def test_compare_takes_precedence_over_analyze_phrase() -> None:
    result = parse_request("Analyze COST and compare it with WMT")
    assert result == RunnableRequest(RequestIntent.COMPARE, ("COST", "WMT"))


@pytest.mark.parametrize(
    "text",
    [
        "Compare COST with WMT",
        "COST vs WMT",
        "Compare AAPL, MSFT, COST and WMT using DIRT",
    ],
)
def test_compare_accepts_two_to_four_explicit_tickers(text: str) -> None:
    result = parse_request(text)
    assert isinstance(result, RunnableRequest)
    assert result.intent is RequestIntent.COMPARE
    assert 2 <= len(result.tickers) <= 4


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("Review my portfolio", RequestIntent.PORTFOLIO),
        ("Run discovery", RequestIntent.DISCOVERY),
        ("Find candidates", RequestIntent.DISCOVERY),
        ("Run gem hunt", RequestIntent.GEM_HUNT),
        ("Find dirt-cheap European stocks", RequestIntent.GEM_HUNT),
    ],
)
def test_non_ticker_workflows(text: str, intent: RequestIntent) -> None:
    result = parse_request(text)
    assert isinstance(result, RunnableRequest)
    assert result.intent is intent


@pytest.mark.parametrize(
    "text",
    [
        "Compare AAPL",
        "Compare AAPL MSFT COST WMT TSLA",
        "Analyze AAPL MSFT",
        "Analyze",
        "Review my portfolio and run discovery",
        "",
    ],
)
def test_ambiguous_or_incomplete_input_requests_clarification(text: str) -> None:
    assert isinstance(parse_request(text), Clarification)


def test_unsupported_input_is_bounded_and_does_not_import_or_call_service() -> None:
    with patch("agent.service.execute_run") as execute:
        result = parse_request("What will interest rates do next year?")
    assert isinstance(result, UnsupportedRequest)
    assert len(result.explanation) < 240
    execute.assert_not_called()


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("show risks", FollowUpKind.RISKS),
        ("show data-quality notes", FollowUpKind.DATA_QUALITY),
        ("show Lynch signals", FollowUpKind.LYNCH),
        ("show Buffett signals", FollowUpKind.BUFFETT),
        ("show evidence", FollowUpKind.EVIDENCE),
        ("Why hold?", FollowUpKind.WHY),
        ("show another ticker from this run", FollowUpKind.SELECT_TICKER),
    ],
)
def test_recent_followups_are_typed_structured_views(text: str, kind: FollowUpKind) -> None:
    result = parse_request(text, recent=RecentContext(("COST", "WMT"), "COST"))
    expected_ticker = "WMT" if kind is FollowUpKind.SELECT_TICKER else None
    assert result == StoredResultFollowUp(kind, expected_ticker)


def test_followup_requires_recent_result_and_selected_ticker_must_belong_to_run() -> None:
    assert isinstance(parse_request("show risks"), Clarification)
    recent = RecentContext(("COST", "WMT"), "COST")
    assert parse_request("show WMT from this run", recent=recent) == StoredResultFollowUp(
        FollowUpKind.SELECT_TICKER, "WMT"
    )
    assert isinstance(parse_request("show AAPL from this run", recent=recent), Clarification)


@pytest.mark.parametrize(
    ("parsed", "mode", "persona"),
    [
        (RunnableRequest(RequestIntent.ANALYZE, ("AAPL",)), RunMode.TICKERS, "default"),
        (RunnableRequest(RequestIntent.COMPARE, ("AAPL", "MSFT")), RunMode.TICKERS, "default"),
        (RunnableRequest(RequestIntent.PORTFOLIO), RunMode.PORTFOLIO, "default"),
        (RunnableRequest(RequestIntent.DISCOVERY, persona="dirt"), RunMode.DISCOVERY, "dirt"),
        (RunnableRequest(RequestIntent.GEM_HUNT), RunMode.GEM_HUNT, "dirt"),
    ],
)
def test_runnable_request_converts_to_validated_service_request(
    parsed: RunnableRequest, mode: RunMode, persona: str
) -> None:
    request = parsed.to_run_request(max_cost_usd=2.5)
    assert request.mode is mode
    assert request.persona == persona
    assert request.max_cost_usd == 2.5


def test_service_validation_remains_authoritative_during_conversion() -> None:
    with pytest.raises(ValidationError):
        RunnableRequest(RequestIntent.ANALYZE, ("NOT$",)).to_run_request()
