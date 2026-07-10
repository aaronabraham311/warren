"""Tests for the golden eval set — one test per ticket acceptance criterion.

These assert over the YAML files as they exist on disk, so a hand-edit that breaks the
curation contract (a missing date, a bullish call allowed on an impaired business, an
ambiguous case that quietly acquired a preferred answer) fails the suite.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from eval.golden_set import (
    EXAMPLES_DIR,
    EvalExample,
    RecommendationExpectation,
    load_all_examples,
    load_eval_example,
)

AMBIGUOUS = {"INTC", "PYPL", "NKE", "SBUX"}
CLEAR_BUY = {"COST", "V"}
CLEAR_HOLD = {"AAPL", "BRK.B", "MSFT", "XOM", "META", "NVDA"}
CLEAR_SELL = {"WBA"}

EXAMPLE_PATHS = sorted(EXAMPLES_DIR.glob("*.yaml"))


@pytest.fixture(scope="module")
def examples() -> list[EvalExample]:
    return load_all_examples()


@pytest.fixture(scope="module")
def by_ticker(examples: list[EvalExample]) -> dict[str, EvalExample]:
    return {e.ticker: e for e in examples}


# ── AC: every YAML validates against EvalExample without a ValidationError ────────────


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda p: p.stem)
def test_example_validates(path: Path) -> None:
    example = load_eval_example(path)
    assert example.ticker


# ── AC: at least 13 YAML files exist under eval/examples/ ─────────────────────────────


def test_at_least_thirteen_examples(examples: list[EvalExample]) -> None:
    assert len(examples) >= 13


def test_load_all_examples_is_sorted_by_filename(examples: list[EvalExample]) -> None:
    assert [e.ticker for e in examples] == [load_eval_example(p).ticker for p in EXAMPLE_PATHS]


# ── AC: all four categories are represented ──────────────────────────────────────────


def test_all_categories_present(by_ticker: dict[str, EvalExample]) -> None:
    tickers = set(by_ticker)
    for category in (CLEAR_BUY, CLEAR_HOLD, CLEAR_SELL, AMBIGUOUS):
        assert category <= tickers, f"missing {category - tickers}"


def test_clear_buy_prefers_buy(by_ticker: dict[str, EvalExample]) -> None:
    for ticker in CLEAR_BUY:
        assert by_ticker[ticker].expectations.recommendation.preferred == "buy"


def test_clear_hold_prefers_hold(by_ticker: dict[str, EvalExample]) -> None:
    for ticker in CLEAR_HOLD:
        assert by_ticker[ticker].expectations.recommendation.preferred == "hold"


# ── AC: WBA excludes buy; at least one other ticker shares that constraint ────────────


def test_wba_allows_only_sell_and_hold(by_ticker: dict[str, EvalExample]) -> None:
    recommendation = by_ticker["WBA"].expectations.recommendation
    assert set(recommendation.allowed) == {"sell", "hold"}
    assert "buy" not in recommendation.allowed
    assert recommendation.preferred == "sell"


def test_buy_excluded_for_at_least_two_tickers(examples: list[EvalExample]) -> None:
    excluded = {e.ticker for e in examples if "buy" not in e.expectations.recommendation.allowed}
    assert "WBA" in excluded
    assert len(excluded) >= 2, f"expected a second buy-excluding ticker, got {excluded}"


# ── AC: the ambiguous cases allow all three and have no preferred answer ──────────────


@pytest.mark.parametrize("ticker", sorted(AMBIGUOUS))
def test_ambiguous_cases_are_open(ticker: str, by_ticker: dict[str, EvalExample]) -> None:
    recommendation = by_ticker[ticker].expectations.recommendation
    assert set(recommendation.allowed) == {"buy", "sell", "hold"}
    assert recommendation.preferred is None


def test_at_least_three_ambiguous_cases(examples: list[EvalExample]) -> None:
    open_cases = [
        e
        for e in examples
        if set(e.expectations.recommendation.allowed) == {"buy", "sell", "hold"}
        and e.expectations.recommendation.preferred is None
    ]
    assert len(open_cases) >= 3


def test_ambiguous_cases_require_both_sides(by_ticker: dict[str, EvalExample]) -> None:
    """An open case must force the agent to argue both directions, not just one."""
    for ticker in AMBIGUOUS:
        buffett = by_ticker[ticker].expectations.buffett_signals
        assert buffett.pros is not None and buffett.pros.min_count >= 1
        assert buffett.cons is not None and buffett.cons.min_count >= 1


# ── AC: last_curated populated and min_specific_numbers >= 3 everywhere ───────────────


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda p: p.stem)
def test_curation_metadata(path: Path) -> None:
    example = load_eval_example(path)
    assert isinstance(example.last_curated, date)
    assert example.expectations.numerical_grounding.min_specific_numbers >= 3
    assert example.notes.strip()


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda p: p.stem)
def test_hype_language_is_forbidden(path: Path) -> None:
    forbidden = load_eval_example(path).expectations.thesis_must_not_mention
    assert "guaranteed" in forbidden


# ── Schema behaviour ─────────────────────────────────────────────────────────────────


def test_unknown_key_is_rejected() -> None:
    """A misspelled expectation key must fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        RecommendationExpectation.model_validate({"allowed": ["hold"], "prefered": "hold"})


def test_preferred_must_be_in_allowed() -> None:
    with pytest.raises(ValidationError):
        RecommendationExpectation.model_validate({"allowed": ["sell", "hold"], "preferred": "buy"})


def test_preferred_may_be_null() -> None:
    recommendation = RecommendationExpectation.model_validate(
        {"allowed": ["buy", "sell", "hold"], "preferred": None},
    )
    assert recommendation.preferred is None


def test_empty_allowed_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RecommendationExpectation.model_validate({"allowed": []})


def test_filename_must_match_ticker(tmp_path: Path) -> None:
    payload = {
        "ticker": "AAPL",
        "notes": "n",
        "last_curated": "2026-07-09",
        "expectations": {"recommendation": {"allowed": ["hold"]}},
    }
    path = tmp_path / "msft.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="does not match ticker"):
        load_eval_example(path)


def test_dotted_ticker_maps_to_underscore_stem(tmp_path: Path) -> None:
    payload = {
        "ticker": "BRK.B",
        "notes": "n",
        "last_curated": "2026-07-09",
        "expectations": {"recommendation": {"allowed": ["hold"]}},
    }
    path = tmp_path / "brk_b.yaml"
    path.write_text(yaml.safe_dump(payload))
    assert load_eval_example(path).ticker == "BRK.B"


def test_invalid_file_error_names_the_path(tmp_path: Path) -> None:
    path = tmp_path / "aapl.yaml"
    path.write_text(yaml.safe_dump({"ticker": "AAPL"}))
    with pytest.raises(ValueError, match="aapl.yaml"):
        load_eval_example(path)
