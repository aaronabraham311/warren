"""Tests for the Lynch/Buffett blended persona system prompt.

All assertions operate on the SYSTEM_PROMPT string — no live API calls.
validate_prompt_length() is tested with a monkeypatched Anthropic client.
"""

from unittest.mock import MagicMock

import pytest

from agent.persona import SYSTEM_PROMPT, DefaultPersona, validate_prompt_length  # noqa: E402

# ── Structural section presence ───────────────────────────────────────────────


def test_lynch_heuristics_section_present() -> None:
    assert "### Lynch Heuristics" in SYSTEM_PROMPT


def test_buffett_heuristics_section_present() -> None:
    assert "### Buffett Heuristics" in SYSTEM_PROMPT


# ── Output requirements — all 8 required fields ───────────────────────────────


@pytest.mark.parametrize(
    "field",
    [
        "recommendation",
        "confidence",
        "thesis",
        "lynch_signals",
        "buffett_signals",
        "key_risks",
        "data_quality_notes",
        "tool_calls_made",
    ],
)
def test_output_field_mentioned(field: str) -> None:
    assert field in SYSTEM_PROMPT, f"Required output field '{field}' not found in SYSTEM_PROMPT"


# ── Guardrails — all five required topics ─────────────────────────────────────


def test_guardrail_data_citation() -> None:
    """Never recommend without citing specific numbers from tools."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "citation" in prompt_lower or "cite" in prompt_lower or "citing" in prompt_lower


def test_guardrail_missing_data() -> None:
    """Missing or null data must be handled explicitly."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "missing" in prompt_lower or "null" in prompt_lower


def test_guardrail_hold_validity() -> None:
    """Hold is a valid recommendation."""
    assert "hold" in SYSTEM_PROMPT.lower()
    # The guardrail should explicitly validate hold as correct/valid/common
    assert any(
        phrase in SYSTEM_PROMPT
        for phrase in ["Hold is valid", "hold is valid", '"hold" is correct', "hold is correct"]
    )


def test_guardrail_sell_catalyst() -> None:
    """Sell recommendation requires a specific catalyst."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "sell" in prompt_lower and "catalyst" in prompt_lower


def test_guardrail_source_conflict() -> None:
    """Source conflict between yfinance and Finnhub must be addressed."""
    assert "yfinance" in SYSTEM_PROMPT and "Finnhub" in SYSTEM_PROMPT
    assert "conflict" in SYSTEM_PROMPT.lower() or "prefer" in SYSTEM_PROMPT.lower()


# ── Tool usage strategy — all current tools listed ────────────────────────────


@pytest.mark.parametrize(
    "tool_name",
    [
        "get_quote",
        "get_fundamentals",
        "get_growth_metrics",
        "read_filing",
        "get_news",
        "get_holding_context",
        "screen_universe",
        "get_valuation_multiples",
        "get_quality_metrics",
        "get_insider_activity",
        "get_peer_comparison",
        "get_financial_strength",
        "estimate_intrinsic_value",
    ],
)
def test_tool_mentioned_in_prompt(tool_name: str) -> None:
    assert tool_name in SYSTEM_PROMPT, f"Tool '{tool_name}' not found in SYSTEM_PROMPT"


# ── validate_prompt_length ────────────────────────────────────────────────────


def _mock_client(token_count: int) -> MagicMock:
    mock = MagicMock()
    count_result = MagicMock()
    count_result.input_tokens = token_count
    mock.messages.count_tokens.return_value = count_result
    return mock


def test_validate_prompt_length_passes_at_threshold() -> None:
    validate_prompt_length(client=_mock_client(4096))


def test_validate_prompt_length_passes_above_threshold() -> None:
    validate_prompt_length(client=_mock_client(5000))


def test_validate_prompt_length_fails_below_threshold() -> None:
    with pytest.raises(AssertionError, match="below Haiku"):
        validate_prompt_length(client=_mock_client(4095))


def test_validate_prompt_length_passes_at_zero_would_fail() -> None:
    with pytest.raises(AssertionError):
        validate_prompt_length(client=_mock_client(0))


# ── DefaultPersona ────────────────────────────────────────────────────────────


def test_default_persona_returns_system_prompt() -> None:
    assert DefaultPersona().system_prompt is SYSTEM_PROMPT


# ── Prompt quality checks ─────────────────────────────────────────────────────


def test_prompt_has_peg_guidance() -> None:
    assert "PEG" in SYSTEM_PROMPT


def test_prompt_has_margin_of_safety() -> None:
    assert "margin of safety" in SYSTEM_PROMPT.lower()


def test_prompt_has_json_output_schema() -> None:
    assert '"recommendation"' in SYSTEM_PROMPT
    assert '"confidence"' in SYSTEM_PROMPT
    assert '"thesis"' in SYSTEM_PROMPT
