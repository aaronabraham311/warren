"""Tests for the Lynch/Buffett blended persona and the DIRT deep-value persona.

All assertions operate on prompt strings — no live API calls.
validate_prompt_length() is tested with a monkeypatched Anthropic client.
"""

from unittest.mock import MagicMock

import pytest

from agent.persona import (  # noqa: E402
    DIRT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    DefaultPersona,
    DirtPersona,
    validate_prompt_length,
)

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


# ── DirtPersona ───────────────────────────────────────────────────────────────


def test_dirt_persona_returns_dirt_system_prompt() -> None:
    assert DirtPersona().system_prompt is DIRT_SYSTEM_PROMPT


def test_dirt_prompt_has_five_steps() -> None:
    for step in [
        "Step 1",
        "Step 2",
        "Step 3",
        "Step 4",
        "Step 5",
    ]:
        assert step in DIRT_SYSTEM_PROMPT, f"'{step}' not found in DIRT_SYSTEM_PROMPT"


@pytest.mark.parametrize(
    "keyword",
    [
        "Cheapness",
        "Operational Quality",
        "Capital Allocation",
        "Coverage-Gap",
        "Source Verification",
    ],
)
def test_dirt_prompt_step_names(keyword: str) -> None:
    assert keyword in DIRT_SYSTEM_PROMPT, (
        f"Step keyword '{keyword}' not found in DIRT_SYSTEM_PROMPT"
    )  # noqa: E501


def test_dirt_prompt_cash_burn_guardrail() -> None:
    prompt_lower = DIRT_SYSTEM_PROMPT.lower()
    assert "burns cash" in prompt_lower or "negative free cash flow" in prompt_lower


def test_dirt_prompt_dirt_signals_required() -> None:
    assert "dirt_signals" in DIRT_SYSTEM_PROMPT
    assert "non-null" in DIRT_SYSTEM_PROMPT or "MUST be non-null" in DIRT_SYSTEM_PROMPT


def test_dirt_prompt_universe_limitation_note() -> None:
    assert "universe-limitation note" in DIRT_SYSTEM_PROMPT or "DIRT universe" in DIRT_SYSTEM_PROMPT


def test_dirt_prompt_universe_note_is_global_not_us_only() -> None:
    # The globalized note must no longer claim a US-only universe...
    assert "US-only" not in DIRT_SYSTEM_PROMPT
    # ...and must name the three non-US exchanges in the slice.
    assert "Euronext Growth Milan" in DIRT_SYSTEM_PROMPT
    assert "Bolsa de Madrid" in DIRT_SYSTEM_PROMPT
    assert "GPW Warsaw" in DIRT_SYSTEM_PROMPT
    for suffix in [".MI", ".MC", ".WA"]:
        assert suffix in DIRT_SYSTEM_PROMPT


def test_dirt_prompt_market_cap_gates_are_usd_normalized() -> None:
    assert "USD-normalized" in DIRT_SYSTEM_PROMPT
    # The $2B and $5B gates are annotated as USD.
    assert "$2B (USD-normalized)" in DIRT_SYSTEM_PROMPT
    assert "$5B (USD-normalized)" in DIRT_SYSTEM_PROMPT


def test_dirt_prompt_sec_filing_graceful_degradation() -> None:
    prompt_lower = DIRT_SYSTEM_PROMPT.lower()
    # Non-US names lack SEC/EDGAR filings — degrade to news + fundamentals.
    assert "degrade gracefully" in prompt_lower
    assert "get_news" in DIRT_SYSTEM_PROMPT
    assert "fundamentals" in prompt_lower
    assert "sec_degradation" in DIRT_SYSTEM_PROMPT


def test_dirt_prompt_has_json_output_schema() -> None:
    assert '"recommendation"' in DIRT_SYSTEM_PROMPT
    assert '"dirt_signals"' in DIRT_SYSTEM_PROMPT


# ── Step 4.5 — integrity check + asymmetry guardrail ─────────────────────────


def test_dirt_prompt_has_step_45() -> None:
    assert "Step 4.5" in DIRT_SYSTEM_PROMPT


def test_dirt_prompt_asymmetry_rule() -> None:
    prompt_lower = DIRT_SYSTEM_PROMPT.lower()
    assert "clean scan" in prompt_lower
    assert "never raise confidence" in prompt_lower or "must never raise confidence" in prompt_lower


def test_dirt_prompt_integrity_scan_observability() -> None:
    assert "integrity_scan" in DIRT_SYSTEM_PROMPT


def test_dirt_prompt_step45_calls_get_key_persons() -> None:
    assert "get_key_persons" in DIRT_SYSTEM_PROMPT


def test_dirt_prompt_step45_adverse_categories() -> None:
    prompt_lower = DIRT_SYSTEM_PROMPT.lower()
    assert "fraud" in prompt_lower
    assert "corruption" in prompt_lower
    assert "governance" in prompt_lower


# ── Lack-of-control / minority-discount guardrail ─────────────────────────────


def test_prompt_mentions_control_tools() -> None:
    assert "get_key_persons" in SYSTEM_PROMPT
    assert "get_capital_allocation" in SYSTEM_PROMPT


def test_prompt_has_lack_of_control_guardrail() -> None:
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "lack-of-control guardrail" in prompt_lower
    assert "controlling_holder_identified" in SYSTEM_PROMPT
    assert "shareholder_yield_pct" in SYSTEM_PROMPT


def test_dirt_prompt_has_control_discount_check() -> None:
    prompt_lower = DIRT_SYSTEM_PROMPT.lower()
    assert "control-discount check" in prompt_lower
    assert "controlling_holder_identified" in DIRT_SYSTEM_PROMPT
    assert "shareholder_yield_pct" in DIRT_SYSTEM_PROMPT
