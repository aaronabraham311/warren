"""Tests for agent.screening — Haiku PASS/FAIL screening pass."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import anthropic
import pytest

from agent.screening import (
    DEFAULT_SCREEN_CRITERIA,
    ScreeningResult,
    _run_sequential_screening,
    run_screening_pass,
    screening_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYSTEM = "You are a test persona."
UNIVERSE = ["AAPL", "MSFT", "GOOG"]


def _text_message(text: str) -> anthropic.types.Message:
    return anthropic.types.Message(
        id="msg_01",
        type="message",
        role="assistant",
        content=[anthropic.types.TextBlock(type="text", text=text)],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=anthropic.types.Usage(input_tokens=5, output_tokens=1),
    )


def _make_batch_result_item(ticker: str, verdict: str) -> MagicMock:
    item = MagicMock()
    item.custom_id = f"screen-{ticker}"
    item.result.type = "succeeded"
    text_block = anthropic.types.TextBlock(type="text", text=verdict)
    item.result.message.content = [text_block]
    return item


# ---------------------------------------------------------------------------
# screening_prompt
# ---------------------------------------------------------------------------


def test_screening_prompt_contains_ticker() -> None:
    prompt = screening_prompt("MSFT", DEFAULT_SCREEN_CRITERIA)
    assert "MSFT" in prompt


def test_screening_prompt_contains_criteria_values() -> None:
    criteria = {
        "pe_max": 25.0,
        "peg_max": 1.2,
        "roe_min": 0.15,
        "de_max": 0.8,
        "rev_growth_min": 0.08,
    }
    prompt = screening_prompt("NVDA", criteria)
    assert "25" in prompt
    assert "1.2" in prompt
    assert "0.15" in prompt


def test_screening_prompt_ends_with_pass_fail() -> None:
    prompt = screening_prompt("AAPL", DEFAULT_SCREEN_CRITERIA)
    assert "PASS or FAIL" in prompt


# ---------------------------------------------------------------------------
# Sequential path
# ---------------------------------------------------------------------------


def test_sequential_pass_and_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """MSFT returns PASS; AAPL and GOOG return FAIL."""
    responses = {
        "AAPL": _text_message("FAIL"),
        "MSFT": _text_message("PASS"),
        "GOOG": _text_message("FAIL"),
    }

    def _create(**kwargs: object) -> anthropic.types.Message:
        prompt_text = str(kwargs.get("messages", ""))
        for ticker, msg in responses.items():
            if ticker in prompt_text:
                return msg
        return _text_message("FAIL")

    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.side_effect = _create

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        result = _run_sequential_screening(UNIVERSE, SYSTEM, DEFAULT_SCREEN_CRITERIA)

    assert result.candidates == ["MSFT"]
    assert abs(result.pass_rate - 1 / 3) < 1e-9
    assert result.batch_id is None


def test_run_screening_pass_sequential_returns_nonempty(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _text_message("PASS")

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        result = run_screening_pass(UNIVERSE, SYSTEM, use_batch_api=False)

    assert len(result.candidates) > 0
    assert result.batch_id is None


def test_sequential_empty_universe() -> None:
    with patch("agent.screening.anthropic.Anthropic"):
        result = _run_sequential_screening([], SYSTEM, DEFAULT_SCREEN_CRITERIA)

    assert result.candidates == []
    assert result.pass_rate == 0.0


def test_sequential_case_insensitive_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Haiku might respond 'pass' in lowercase — should still be accepted."""
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _text_message("pass")

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        result = _run_sequential_screening(["AAPL"], SYSTEM, DEFAULT_SCREEN_CRITERIA)

    assert result.candidates == ["AAPL"]


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------


def _make_mock_batch_client(
    batch_id: str,
    statuses: list[str],
    items: list[MagicMock],
) -> MagicMock:
    """Build a mock Anthropic client whose batch API returns canned data."""
    mock_client = MagicMock(spec=anthropic.Anthropic)

    mock_batch = MagicMock()
    mock_batch.id = batch_id
    mock_client.messages.batches.create.return_value = mock_batch

    retrieve_responses = []
    for s in statuses:
        r = MagicMock()
        r.processing_status = s
        retrieve_responses.append(r)
    mock_client.messages.batches.retrieve.side_effect = retrieve_responses

    mock_client.messages.batches.results.return_value = iter(items)
    return mock_client


def test_batch_path_returns_candidates() -> None:
    items = [
        _make_batch_result_item("AAPL", "FAIL"),
        _make_batch_result_item("MSFT", "PASS"),
        _make_batch_result_item("GOOG", "FAIL"),
    ]
    mock_client = _make_mock_batch_client("batch_abc", ["ended"], items)
    sleep_calls: list[float] = []

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        result = run_screening_pass(UNIVERSE, SYSTEM, use_batch_api=True, _sleep=sleep_calls.append)

    assert result.candidates == ["MSFT"]
    assert result.batch_id == "batch_abc"
    assert len(sleep_calls) == 0  # status was "ended" immediately


def test_batch_polls_until_ended() -> None:
    items = [_make_batch_result_item("AAPL", "PASS")]
    mock_client = _make_mock_batch_client(
        "batch_xyz", ["in_progress", "in_progress", "ended"], items
    )
    sleep_calls: list[float] = []

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        run_screening_pass(["AAPL"], SYSTEM, use_batch_api=True, _sleep=sleep_calls.append)

    assert len(sleep_calls) == 2


def test_batch_skips_errored_items() -> None:
    errored = MagicMock()
    errored.custom_id = "screen-ERR"
    errored.result.type = "errored"

    items = [
        _make_batch_result_item("MSFT", "PASS"),
        errored,
    ]
    mock_client = _make_mock_batch_client("batch_1", ["ended"], items)

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        result = run_screening_pass(
            ["MSFT", "ERR"], SYSTEM, use_batch_api=True, _sleep=lambda _: None
        )

    assert result.candidates == ["MSFT"]
    assert "ERR" not in result.candidates


def test_batch_pass_rate() -> None:
    items = [
        _make_batch_result_item("A", "PASS"),
        _make_batch_result_item("B", "PASS"),
        _make_batch_result_item("C", "FAIL"),
        _make_batch_result_item("D", "FAIL"),
    ]
    mock_client = _make_mock_batch_client("batch_2", ["ended"], items)

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        result = run_screening_pass(
            ["A", "B", "C", "D"], SYSTEM, use_batch_api=True, _sleep=lambda _: None
        )

    assert abs(result.pass_rate - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Phase event logging
# ---------------------------------------------------------------------------


def test_phase_events_emitted_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _text_message("PASS")

    logged: list[tuple[str, dict[str, object]]] = []

    mock_logger = MagicMock()
    mock_logger.log.side_effect = lambda event, **kw: logged.append((event, kw))

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        run_screening_pass(["AAPL"], SYSTEM, use_batch_api=False, logger=mock_logger)

    events = [e for e, _ in logged]
    assert "phase_started" in events
    assert "phase_completed" in events

    started_kw = next(kw for e, kw in logged if e == "phase_started")
    assert started_kw["phase"] == "screening"
    assert started_kw["universe_size"] == 1
    assert started_kw["model"] == "claude-haiku-4-5-20251001"
    assert started_kw["use_batch_api"] is False

    completed_kw = next(kw for e, kw in logged if e == "phase_completed")
    assert completed_kw["phase"] == "screening"
    assert "candidates_surfaced" in completed_kw
    assert "pass_rate" in completed_kw
    assert "batch_id" in completed_kw


def test_phase_events_emitted_batch() -> None:
    items = [_make_batch_result_item("AAPL", "PASS")]
    mock_client = _make_mock_batch_client("batch_log", ["ended"], items)

    logged: list[tuple[str, dict[str, object]]] = []
    mock_logger = MagicMock()
    mock_logger.log.side_effect = lambda event, **kw: logged.append((event, kw))

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        run_screening_pass(
            ["AAPL"], SYSTEM, use_batch_api=True, logger=mock_logger, _sleep=lambda _: None
        )

    started_kw = next(kw for e, kw in logged if e == "phase_started")
    assert started_kw["use_batch_api"] is True

    completed_kw = next(kw for e, kw in logged if e == "phase_completed")
    assert completed_kw["batch_id"] == "batch_log"


def test_no_logger_does_not_crash() -> None:
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _text_message("FAIL")

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        result = run_screening_pass(["AAPL"], SYSTEM, use_batch_api=False, logger=None)

    assert isinstance(result, ScreeningResult)


# ---------------------------------------------------------------------------
# Cooldown — verified at the caller level (screening is DB-free)
# ---------------------------------------------------------------------------


def test_screening_accepts_pre_filtered_universe() -> None:
    """Verify that passing a cooldown-filtered list is all that's needed.

    The screening module must not touch any DB or cooldown state itself.
    Passing a filtered list (e.g. ["MSFT"]) is sufficient for isolation.
    """
    mock_client = MagicMock(spec=anthropic.Anthropic)
    mock_client.messages.create.return_value = _text_message("PASS")

    with patch("agent.screening.anthropic.Anthropic", return_value=mock_client):
        # Only MSFT is passed — AAPL was suppressed by cooldown upstream
        result = run_screening_pass(["MSFT"], SYSTEM, use_batch_api=False)

    assert result.candidates == ["MSFT"]
    assert mock_client.messages.create.call_count == 1
